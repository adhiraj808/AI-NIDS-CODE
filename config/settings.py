"""
Configuration settings for AI-NIDS: AI-Powered Network Intrusion Detection System.
Provides paths, hyperparameter presets, CICIDS2017 feature definitions,
and reproducible seeds across the entire pipeline.
"""

from pathlib import Path
import os

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Core Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

MODELS_DIR = PROJECT_ROOT / "models" / "trained"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CHARTS_DIR = RESULTS_DIR / "charts"
REPORTS_DIR = RESULTS_DIR / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure runtime directories exist
for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAMPLE_DATA_DIR,
    MODELS_DIR,
    METRICS_DIR,
    CHARTS_DIR,
    REPORTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Reproducibility & Splitting
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.20
DEFAULT_CHUNK_SIZE = 50000

# Classification Targets
LABEL_COLUMN = "Label"
BENIGN_LABEL = "BENIGN"
MALICIOUS_LABEL = "MALICIOUS"

# Non-predictive or high-leakage metadata columns in CICIDS2017 to exclude from training
DEFAULT_DROP_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Protocol",
    "Timestamp",
    "SimillarHTTP",
    "Inbound",
]

# Canonical 78 feature columns in CICIDS2017 (standardized names without trailing spaces)
CICIDS_COMMON_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]

# Model Configuration presets
MODEL_CONFIGS = {
    "logistic_regression": {
        "display_name": "Logistic Regression (Baseline)",
        "solver": "lbfgs",
        "max_iter": 1000,
        "C": 1.0,
        "class_weight": "balanced",
        "random_state": DEFAULT_RANDOM_STATE,
    },
    "random_forest": {
        "display_name": "Random Forest Classifier",
        "n_estimators": 100,
        "max_depth": 18,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": DEFAULT_RANDOM_STATE,
    },
    "xgboost": {
        "display_name": "XGBoost Classifier",
        "n_estimators": 120,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": DEFAULT_RANDOM_STATE,
        "n_jobs": -1,
    },
}

# Explainability limits
SHAP_MAX_DISPLAY = 15
SHAP_SAMPLE_SIZE = 300
