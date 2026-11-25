"""
Unit tests for model evaluation and metrics.
Verifies correct calculation of security metrics: FNR, FPR, Confusion Matrix, and ROC-AUC.
"""

import numpy as np
import pytest

from src.evaluation import calculate_comprehensive_metrics, compile_comparison_table


def test_metrics_calculation_standard():
    # Ground truth: 4 benign (0), 4 malicious (1)
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Predictions: 1 false positive, 1 false negative
    y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.8, 0.4, 0.85, 0.9, 0.95])

    metrics = calculate_comprehensive_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        training_time=1.23,
        prediction_time=0.04,
        model_name="TestModel",
    )

    # TN=3, FP=1, FN=1, TP=3
    assert metrics["true_negatives"] == 3
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_positives"] == 3

    # Accuracy = (3+3)/8 = 0.75
    assert metrics["accuracy"] == 0.75
    # Recall = 3 / (3+1) = 0.75
    assert metrics["recall"] == 0.75
    # Precision = 3 / (3+1) = 0.75
    assert metrics["precision"] == 0.75
    # FPR = FP / (FP + TN) = 1 / 4 = 0.25
    assert metrics["false_positive_rate"] == 0.25
    # FNR = FN / (FN + TP) = 1 / 4 = 0.25
    assert metrics["false_negative_rate"] == 0.25
    # ROC-AUC should be a float
    assert isinstance(metrics["roc_auc"], float)


def test_zero_division_graceful_handling():
    # Extreme edge case: all benign, all predicted benign
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])

    metrics = calculate_comprehensive_metrics(
        y_true=y_true,
        y_pred=y_pred,
        model_name="AllBenignModel",
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 0.0  # zero_division handles cleanly
    assert metrics["recall"] == 0.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["false_negative_rate"] == 0.0


def test_compile_comparison_table():
    m1 = calculate_comprehensive_metrics(np.array([0, 1]), np.array([0, 1]), model_name="M1")
    m2 = calculate_comprehensive_metrics(np.array([0, 1]), np.array([1, 1]), model_name="M2")

    df = compile_comparison_table([m1, m2])
    assert len(df) == 2
    assert "Model" in df.columns
    assert "False Negative Rate (FNR)" in df.columns
    assert "False Positive Rate (FPR)" in df.columns
