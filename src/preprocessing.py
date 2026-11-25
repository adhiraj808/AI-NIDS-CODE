"""
Data Preprocessing Pipeline for AI-NIDS.
Prevents data leakage by ensuring all imputations, scalers, and encodings
are fitted exclusively on the training subset and applied to the test subset.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config.settings import (
    BENIGN_LABEL,
    DEFAULT_DROP_COLUMNS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    LABEL_COLUMN,
    MALICIOUS_LABEL,
)

logger = logging.getLogger("AI-NIDS.Preprocessing")


class PreprocessingPipeline:
    """
    Leakage-free, state-aware preprocessing pipeline for network traffic flows.
    """

    def __init__(
        self,
        test_size: float = DEFAULT_TEST_SIZE,
        random_state: int = DEFAULT_RANDOM_STATE,
        drop_columns: Optional[List[str]] = None,
        apply_scaling: bool = True,
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.drop_columns = drop_columns if drop_columns is not None else DEFAULT_DROP_COLUMNS
        self.apply_scaling = apply_scaling

        # Transformers (fit ONLY on training split)
        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None

        # State tracking
        self.feature_names: List[str] = []
        self.is_fitted: bool = False
        self.preprocessing_summary: Dict[str, Any] = {}

    def clean_raw_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Step 1: Clean raw inputs before splitting.
        - Removes duplicates
        - Drops non-predictive metadata/identifier columns
        - Replaces infinity values with NaN
        """
        initial_rows = len(df)
        initial_cols = len(df.columns)

        # 1. Deduplication
        df_cleaned = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df_cleaned)

        # 2. Exclude non-predictive or high-leakage metadata columns
        cols_to_drop = [c for c in self.drop_columns if c in df_cleaned.columns]
        if "Original_Attack_Type" in df_cleaned.columns:
            # Preserve original labels in a separate series or drop from feature matrix
            cols_to_drop.append("Original_Attack_Type")

        df_cleaned = df_cleaned.drop(columns=cols_to_drop, errors="ignore")

        # 3. Handle infinities across numeric columns
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        num_inf = np.isneginf(df_cleaned[numeric_cols].values).sum() + np.isposinf(df_cleaned[numeric_cols].values).sum()
        df_cleaned[numeric_cols] = df_cleaned[numeric_cols].replace([np.inf, -np.inf], np.nan)

        # 4. Total missing values detection
        missing_count = int(df_cleaned.isna().sum().sum())

        audit = {
            "initial_rows": initial_rows,
            "initial_cols": initial_cols,
            "cleaned_rows": len(df_cleaned),
            "duplicates_removed": duplicates_removed,
            "columns_dropped": cols_to_drop,
            "infinite_values_replaced": int(num_inf),
            "total_missing_values_detected": missing_count,
        }

        logger.info(
            "Cleaned raw data: %d rows remaining (-%d dupes). Columns dropped: %s. Missing values: %d",
            len(df_cleaned),
            duplicates_removed,
            cols_to_drop,
            missing_count,
        )

        return df_cleaned, audit

    def fit_transform(
        self,
        df: pd.DataFrame,
        label_col: str = LABEL_COLUMN,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Executes end-to-end preprocessing pipeline:
        1. Clean raw data (deduplication, infinity replacement)
        2. Separate X and y
        3. Stratified split (to strictly avoid leakage)
        4. Fit Imputer and Scaler on X_train ONLY
        5. Transform X_train and X_test

        Returns:
            X_train (scaled/imputed np.ndarray)
            X_test (scaled/imputed np.ndarray)
            y_train (binary encoded np.ndarray: 0=BENIGN, 1=MALICIOUS)
            y_test (binary encoded np.ndarray: 0=BENIGN, 1=MALICIOUS)
            feature_names (List[str])
        """
        cleaned_df, clean_audit = self.clean_raw_data(df)

        if label_col not in cleaned_df.columns:
            raise KeyError(f"Label column '{label_col}' not found in DataFrame.")

        # Separate features and target
        y_raw = cleaned_df[label_col].astype(str).str.strip()
        X_df = cleaned_df.drop(columns=[label_col])

        # Binary encoding: BENIGN -> 0, MALICIOUS -> 1
        y = np.where(y_raw.str.upper() == BENIGN_LABEL, 0, 1)

        # Keep strictly numeric features
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_dropped = [c for c in X_df.columns if c not in numeric_cols]
        if non_numeric_dropped:
            logger.warning("Dropping non-numeric features: %s", non_numeric_dropped)

        X_df = X_df[numeric_cols]
        self.feature_names = numeric_cols

        # Imbalance analysis
        benign_count = int((y == 0).sum())
        malicious_count = int((y == 1).sum())
        total_samples = len(y)
        imbalance_ratio = round(benign_count / max(malicious_count, 1), 2)

        # Stratified Train-Test Split (Guarantees identical class distribution without leakage)
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X_df,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        logger.info(
            "Stratified split executed: Train=%d (%.1f%%), Test=%d (%.1f%%). Seed=%d",
            len(y_train),
            (1 - self.test_size) * 100,
            len(y_test),
            self.test_size * 100,
            self.random_state,
        )

        # Imputation: fit strictly on training set
        self.imputer = SimpleImputer(strategy="median")
        X_train_imputed = self.imputer.fit_transform(X_train_df)
        X_test_imputed = self.imputer.transform(X_test_df)

        # Feature Scaling: fit strictly on training set
        if self.apply_scaling:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train_imputed)
            X_test = self.scaler.transform(X_test_imputed)
        else:
            self.scaler = None
            X_train = X_train_imputed
            X_test = X_test_imputed

        self.is_fitted = True

        # Preprocessing summary audit report
        self.preprocessing_summary = {
            **clean_audit,
            "non_numeric_dropped": non_numeric_dropped,
            "final_feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "total_samples": total_samples,
            "class_distribution": {
                "BENIGN": benign_count,
                "MALICIOUS": malicious_count,
                "benign_percentage": round((benign_count / total_samples) * 100, 2),
                "malicious_percentage": round((malicious_count / total_samples) * 100, 2),
                "imbalance_ratio_benign_to_malicious": imbalance_ratio,
            },
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "test_size": self.test_size,
            "random_seed": self.random_state,
            "scaling_applied": self.apply_scaling,
            "imputation_strategy": "median (fitted on train only)",
            "leakage_prevention": "Strictly stratified train-test split before fitting transforms",
        }

        return X_train, X_test, y_train, y_test, self.feature_names

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms incoming novel/unseen network traffic flows using the fitted pipeline.
        Used during threat analysis and inference.
        """
        if not self.is_fitted or self.imputer is None:
            raise RuntimeError("Pipeline must be fitted via fit_transform() before calling transform().")

        df_in = df.copy()

        # Match expected features; if any missing, impute zero; if extra, drop
        for feat in self.feature_names:
            if feat not in df_in.columns:
                df_in[feat] = 0.0

        X_df = df_in[self.feature_names]

        # Replace infinities
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
        X_df[numeric_cols] = X_df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        # Impute and scale using learned training parameters
        X_imputed = self.imputer.transform(X_df)
        if self.scaler is not None:
            return self.scaler.transform(X_imputed)
        return X_imputed

    def get_summary(self) -> Dict[str, Any]:
        """Returns the complete audit log of preprocessing decisions."""
        return self.preprocessing_summary
