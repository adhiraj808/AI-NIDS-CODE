"""
Model Evaluation and Metric Calculation Module for AI-NIDS.
Computes comprehensive security-focused metrics, including False Negative Rate (FNR),
False Positive Rate (FPR), Confusion Matrices, ROC-AUC, and latency.
Also provides visual comparison generators (ROC curves, confusion matrices, bar charts).
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config.settings import BENIGN_LABEL, MALICIOUS_LABEL

logger = logging.getLogger("AI-NIDS.Evaluation")


def calculate_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    training_time: float = 0.0,
    prediction_time: float = 0.0,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Calculates full suite of classification and operational security metrics.
    Handles zero-division gracefully.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # Extract TN, FP, FN, TP
    # Class 0: BENIGN, Class 1: MALICIOUS
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Core rates
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Security-critical rates:
    # FPR: Proportion of benign flows falsely flagged as malicious (causes alert fatigue)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # FNR: Proportion of actual attacks missed by the model (critical security vulnerability)
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # Specificity: True Negative Rate
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # ROC-AUC
    roc_auc: Optional[float] = None
    if y_proba is not None:
        try:
            # Check if there are at least two distinct classes in y_true
            if len(np.unique(y_true)) > 1:
                roc_auc = float(roc_auc_score(y_true, y_proba))
        except Exception as err:
            logger.warning("ROC-AUC computation skipped for %s: %s", model_name, err)

    # Samples per second
    throughput = len(y_true) / prediction_time if prediction_time > 0 else 0.0

    metrics = {
        "model_name": model_name,
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1, 6),
        "roc_auc": round(roc_auc, 6) if roc_auc is not None else "N/A",
        "false_positive_rate": round(fpr, 6),
        "false_negative_rate": round(fnr, 6),
        "specificity": round(specificity, 6),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "training_time_seconds": round(training_time, 4),
        "prediction_time_seconds": round(prediction_time, 4),
        "throughput_samples_per_sec": round(throughput, 1),
        "confusion_matrix": cm.tolist(),
        "total_test_samples": len(y_true),
    }

    return metrics


def compile_comparison_table(metrics_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Transforms multiple model metrics dictionaries into a clean summary DataFrame.
    """
    rows = []
    for m in metrics_list:
        rows.append({
            "Model": m["model_name"],
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall (Sensitivity)": m["recall"],
            "F1-Score": m["f1_score"],
            "ROC-AUC": m["roc_auc"],
            "False Negative Rate (FNR)": m["false_negative_rate"],
            "False Positive Rate (FPR)": m["false_positive_rate"],
            "Train Time (s)": m["training_time_seconds"],
            "Inference Time (s)": m["prediction_time_seconds"],
            "Throughput (samples/s)": m["throughput_samples_per_sec"],
        })
    return pd.DataFrame(rows)


def explain_metric_significance() -> Dict[str, str]:
    """
    Provides educational cybersecurity context on why standard metrics like accuracy
    are dangerously inadequate for intrusion detection.
    """
    return {
        "Accuracy Paradox": (
            "In intrusion detection datasets such as CICIDS2017, benign network traffic typically "
            "represents 80% to 99% of total flow records. A degenerate or trivial model that predicts every single packet as "
            "BENIGN will mathematically achieve 99% accuracy while having a 0% recall and a catastrophic 100% False Negative Rate. "
            "Thus, accuracy alone conceals catastrophic security vulnerabilities."
        ),
        "False Negative Rate (FNR)": (
            "FNR = FN / (FN + TP). Represents actual malicious network attacks (e.g. DoS, PortScan, Exfiltration) that bypassed "
            "the detector undetected. In cybersecurity defense, False Negatives carry the highest operational risk, leading to "
            "unmitigated network compromise, lateral movement, and persistent threats."
        ),
        "False Positive Rate (FPR)": (
            "FPR = FP / (FP + TN). Represents legitimate benign network sessions erroneously classified as intrusions. "
            "High FPR induces Security Operations Center (SOC) analyst alert fatigue, delays responses to real threats, and may "
            "result in blocking critical legitimate enterprise business traffic."
        ),
        "F1-Score": (
            "Harmonic mean of Precision and Recall. F1-score provides a much more honest and robust benchmark of model quality "
            "under severe class imbalance than overall accuracy."
        ),
        "ROC-AUC": (
            "Area Under the Receiver Operating Characteristic curve. Measures discriminatory capability across all potential "
            "classification thresholds, independent of specific operational decision cutoff values."
        ),
    }


def compute_roc_curves_data(
    models_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Dict[str, Any]]:
    """
    Computes false positive rates and true positive rates across thresholds for plotting ROC curves.

    Args:
        models_predictions: Dict mapping model_name -> (y_true, y_proba)

    Returns:
        Dict mapping model_name -> {"fpr": list, "tpr": list, "auc": float}
    """
    roc_data = {}
    for name, (y_true, y_proba) in models_predictions.items():
        if y_proba is None or len(np.unique(y_true)) < 2:
            continue
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc_val = float(roc_auc_score(y_true, y_proba))
            roc_data[name] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": round(auc_val, 4),
            }
        except Exception as err:
            logger.warning("ROC curve calculation failed for %s: %s", name, err)

    return roc_data
