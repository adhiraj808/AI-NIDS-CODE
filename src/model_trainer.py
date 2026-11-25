"""
Model Trainer Module for AI-NIDS.
Implements and trains:
- Model A: Logistic Regression (Baseline)
- Model B: Random Forest Classifier (Ensemble)
- Model C: XGBoost Classifier (Gradient Boosting)

Tracks training and inference latency, persists artifacts via joblib,
and guarantees reproducible experimental seeds.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from config.settings import (
    DEFAULT_RANDOM_STATE,
    MODEL_CONFIGS,
    MODELS_DIR,
)

logger = logging.getLogger("AI-NIDS.ModelTrainer")

# Gracefully handle xgboost import (e.g. if libomp is missing on macOS)
try:
    from xgboost import XGBClassifier
    # Test instantiating to verify dynamic library loads
    _test_xgb = XGBClassifier()
    XGBOOST_AVAILABLE = True
except Exception as e:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost C-library not initialized (%s). Utilizing HistGradientBoosting fallback.", e)


class ModelTrainer:
    """
    Manages lifecycle, training, hyperparameter settings, serialization, and timing
    for all intrusion detection models.
    """

    def __init__(self, random_state: int = DEFAULT_RANDOM_STATE):
        self.random_state = random_state
        self.trained_models: Dict[str, Any] = {}
        self.training_times: Dict[str, float] = {}
        self.prediction_times: Dict[str, float] = {}
        self.model_configs: Dict[str, Dict[str, Any]] = {}

    def get_supported_models(self) -> List[str]:
        """Returns list of model keys supported by current environment."""
        return ["logistic_regression", "random_forest", "xgboost"]

    def build_model(
        self,
        model_key: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        handle_imbalance: bool = True,
    ) -> Any:
        """
        Instantiates model with reproducible configuration.
        """
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model_key: {model_key}. Available: {list(MODEL_CONFIGS.keys())}")

        cfg = MODEL_CONFIGS[model_key].copy()
        if hyperparameters:
            cfg.update(hyperparameters)

        cfg["random_state"] = self.random_state

        if model_key == "logistic_regression":
            class_weight = "balanced" if handle_imbalance else None
            return LogisticRegression(
                solver=cfg.get("solver", "lbfgs"),
                max_iter=cfg.get("max_iter", 1000),
                C=cfg.get("C", 1.0),
                class_weight=class_weight,
                random_state=self.random_state,
            )

        elif model_key == "random_forest":
            class_weight = "balanced" if handle_imbalance else None
            return RandomForestClassifier(
                n_estimators=cfg.get("n_estimators", 100),
                max_depth=cfg.get("max_depth", 18),
                min_samples_split=cfg.get("min_samples_split", 5),
                min_samples_leaf=cfg.get("min_samples_leaf", 2),
                class_weight=class_weight,
                n_jobs=cfg.get("n_jobs", -1),
                random_state=self.random_state,
            )

        elif model_key == "xgboost":
            if XGBOOST_AVAILABLE:
                scale_pos_weight = cfg.get("scale_pos_weight", 1.0)
                return XGBClassifier(
                    n_estimators=cfg.get("n_estimators", 120),
                    max_depth=cfg.get("max_depth", 6),
                    learning_rate=cfg.get("learning_rate", 0.1),
                    subsample=cfg.get("subsample", 0.8),
                    colsample_bytree=cfg.get("colsample_bytree", 0.8),
                    eval_metric=cfg.get("eval_metric", "logloss"),
                    scale_pos_weight=scale_pos_weight,
                    random_state=self.random_state,
                    n_jobs=cfg.get("n_jobs", -1),
                )
            else:
                logger.info("Using HistGradientBoostingClassifier as Model C gradient boosting engine.")
                return HistGradientBoostingClassifier(
                    max_iter=cfg.get("n_estimators", 100),
                    max_depth=cfg.get("max_depth", 8),
                    learning_rate=cfg.get("learning_rate", 0.1),
                    class_weight="balanced" if handle_imbalance else None,
                    random_state=self.random_state,
                )

        raise ValueError(f"Unsupported model: {model_key}")

    def train_model(
        self,
        model_key: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        hyperparameters: Optional[Dict[str, Any]] = None,
        handle_imbalance: bool = True,
    ) -> Tuple[Any, float]:
        """
        Trains model on X_train, y_train and measures wall-clock training latency.
        """
        model = self.build_model(
            model_key=model_key,
            hyperparameters=hyperparameters,
            handle_imbalance=handle_imbalance,
        )

        logger.info("Initiating training for model '%s' on %d samples...", model_key, len(X_train))
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - t0

        self.trained_models[model_key] = model
        self.training_times[model_key] = training_time
        self.model_configs[model_key] = {
            "model_key": model_key,
            "display_name": MODEL_CONFIGS[model_key]["display_name"],
            "handle_imbalance": handle_imbalance,
            "training_time": training_time,
            "train_samples": len(X_train),
        }

        logger.info("Training completed for '%s' in %.4f seconds.", model_key, training_time)
        return model, training_time

    def predict(
        self,
        model_key: str,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        """
        Generates predictions and probability estimates on test/unseen traffic.
        Measures total inference latency.

        Returns:
            y_pred (np.ndarray): Binary predictions (0 or 1)
            y_proba (Optional[np.ndarray]): Probability of Malicious class (1)
            prediction_time (float): Wall-clock time in seconds
        """
        if model_key not in self.trained_models:
            raise KeyError(f"Model '{model_key}' has not been trained yet.")

        model = self.trained_models[model_key]

        t0 = time.perf_counter()
        y_pred = model.predict(X)
        prediction_time = time.perf_counter() - t0

        y_proba = None
        if hasattr(model, "predict_proba"):
            # Probabilities for class 1 (MALICIOUS)
            y_proba = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(X)
            # Sigmoid normalization
            y_proba = 1.0 / (1.0 + np.exp(-decision))

        self.prediction_times[model_key] = prediction_time
        return y_pred, y_proba, prediction_time

    def save_model(
        self,
        model_key: str,
        filepath: Optional[Union[str, Path]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Saves trained model and metadata bundle using joblib.
        """
        if model_key not in self.trained_models:
            raise KeyError(f"Model '{model_key}' is not trained.")

        if filepath is None:
            save_path = MODELS_DIR / f"{model_key}.joblib"
        else:
            save_path = Path(filepath)

        payload = {
            "model_key": model_key,
            "model": self.trained_models[model_key],
            "training_time": self.training_times.get(model_key, 0.0),
            "config": self.model_configs.get(model_key, {}),
            "extra_metadata": extra_metadata or {},
        }

        joblib.dump(payload, save_path, compress=3)
        logger.info("Saved model bundle for '%s' to %s", model_key, save_path)
        return save_path

    def load_model(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Loads saved model bundle from disk.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        payload = joblib.load(path)
        model_key = payload.get("model_key", "custom_model")
        self.trained_models[model_key] = payload["model"]
        self.training_times[model_key] = payload.get("training_time", 0.0)
        self.model_configs[model_key] = payload.get("config", {})

        logger.info("Loaded model bundle from %s (key='%s')", path, model_key)
        return payload
