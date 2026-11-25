"""
Unit tests for SHAP Explainability module.
Verifies:
(a) SHAP values have shape [n_samples, n_features]
(b) For XGBoost/HistGB, sum of SHAP values plus base value matches the model's raw margin output within small tolerance
(c) Output files are created in results/shap/ and results/shap/local/
(d) Results are reproducible with the fixed random seed
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from src.explain_shap import ShapExplainerSuite, SHAP_DIR, LOCAL_SHAP_DIR


@pytest.fixture
def synthetic_models_and_data():
    rng = np.random.RandomState(42)
    n_samples = 150
    n_features = 10
    feature_names = [f"Feature_{i}" for i in range(n_features)]

    X = rng.randn(n_samples, n_features)
    y = rng.randint(0, 2, n_samples)

    rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42).fit(X, y)
    xgb = HistGradientBoostingClassifier(max_iter=15, max_depth=4, random_state=42).fit(X, y)

    return rf, xgb, X, y, feature_names


def test_shap_values_shape(synthetic_models_and_data):
    """(a) Test that SHAP values have shape [n_samples, n_features] for both models."""
    rf, xgb, X, y, feature_names = synthetic_models_and_data
    suite = ShapExplainerSuite(rf_model=rf, xgb_model=xgb, feature_names=feature_names, random_state=42)

    X_eval, y_eval, _, _ = suite.select_evaluation_subset(X, y, sample_size=40)

    # Test Random Forest
    _, shap_rf, _ = suite.compute_tree_shap_values(rf, X_eval)
    assert shap_rf.shape == (len(X_eval), len(feature_names)), f"RF SHAP shape mismatch: {shap_rf.shape}"

    # Test XGBoost / HistGB
    _, shap_xgb, _ = suite.compute_tree_shap_values(xgb, X_eval)
    assert shap_xgb.shape == (len(X_eval), len(feature_names)), f"XGB SHAP shape mismatch: {shap_xgb.shape}"


def test_xgboost_margin_additivity(synthetic_models_and_data):
    """
    (b) Test that for XGBoost / HistGB, sum of SHAP values plus base value matches
    the model's raw margin output within small tolerance (efficiency property of Shapley values).
    """
    rf, xgb, X, y, feature_names = synthetic_models_and_data
    suite = ShapExplainerSuite(rf_model=rf, xgb_model=xgb, feature_names=feature_names, random_state=42)

    X_test_slice = X[:30]
    _, shap_xgb, base_val = suite.compute_tree_shap_values(xgb, X_test_slice)

    # Raw margin output from model
    raw_decision = xgb.decision_function(X_test_slice)

    # Sum of SHAP values + base value
    reconstructed_margin = shap_xgb.sum(axis=1) + base_val

    max_diff = np.max(np.abs(reconstructed_margin - raw_decision))
    assert max_diff < 1e-4, f"Additivity failure: max difference {max_diff} exceeds tolerance 1e-4"


def test_shap_output_files_exist():
    """(c) Test that global and local SHAP output files exist and are non-empty."""
    required_files = [
        SHAP_DIR / "shap_top15_random_forest.csv",
        SHAP_DIR / "shap_top15_xgboost.csv",
        SHAP_DIR / "shap_summary_random_forest.png",
        SHAP_DIR / "shap_bar_random_forest.png",
        SHAP_DIR / "shap_summary_xgboost.png",
        SHAP_DIR / "shap_bar_xgboost.png",
        SHAP_DIR / "shap_vs_gini.csv",
        LOCAL_SHAP_DIR / "local_explanations_random_forest.csv",
        LOCAL_SHAP_DIR / "local_explanations_xgboost.csv",
    ]

    for fpath in required_files:
        assert fpath.exists(), f"Expected SHAP output file missing: {fpath}"
        assert fpath.stat().st_size > 0, f"Expected non-empty file: {fpath}"


def test_shap_reproducibility(synthetic_models_and_data):
    """(d) Test that SHAP calculations are perfectly reproducible with fixed seed."""
    rf, xgb, X, y, feature_names = synthetic_models_and_data

    suite1 = ShapExplainerSuite(rf_model=rf, xgb_model=xgb, feature_names=feature_names, random_state=42)
    _, shap1, _ = suite1.compute_tree_shap_values(xgb, X[:25])

    suite2 = ShapExplainerSuite(rf_model=rf, xgb_model=xgb, feature_names=feature_names, random_state=42)
    _, shap2, _ = suite2.compute_tree_shap_values(xgb, X[:25])

    assert np.allclose(shap1, shap2, atol=1e-8), "SHAP results are not reproducible across identical runs!"
