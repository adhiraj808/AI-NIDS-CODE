"""
Feature Engineering and Selection Module for AI-NIDS.
Provides correlation analysis, variance thresholding, and feature importance helpers.
"""

from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

logger = logging.getLogger("AI-NIDS.FeatureEngineering")


def calculate_feature_correlations(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    top_n: int = 15,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Computes Pearson correlation matrix and correlations against a target column.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    target_corrs = None
    if target_col and target_col in numeric_df.columns:
        target_corrs = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)

    return corr_matrix, target_corrs


def filter_low_variance_features(
    X: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.0,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Removes constant and quasi-constant features below a variance threshold.

    Returns:
        X_filtered (np.ndarray)
        retained_features (List[str])
        dropped_features (List[str])
    """
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)

    mask = selector.get_support()
    retained = [feature_names[i] for i, m in enumerate(mask) if m]
    dropped = [feature_names[i] for i, m in enumerate(mask) if not m]

    logger.info("Variance filter: %d features retained, %d dropped.", len(retained), len(dropped))
    return X_filtered, retained, dropped


def rank_features_by_importance(
    importances: np.ndarray,
    feature_names: List[str],
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Creates a ranked pandas DataFrame of feature importances.
    """
    # Guard against length mismatch
    min_len = min(len(importances), len(feature_names))
    df = pd.DataFrame({
        "Feature": feature_names[:min_len],
        "Importance": importances[:min_len],
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    df["Rank"] = df.index + 1
    return df.head(top_n)
