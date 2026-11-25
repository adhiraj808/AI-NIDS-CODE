"""
Unit tests for model training and threat prediction inference engine.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from config.settings import BENIGN_LABEL, MALICIOUS_LABEL, MODELS_DIR
from src.model_trainer import ModelTrainer
from src.prediction import ThreatClassifier
from src.preprocessing import PreprocessingPipeline


@pytest.fixture
def synthetic_train_data():
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame({
        "Flow Duration": rng.uniform(10, 1000, size=n),
        "Total Fwd Packets": rng.poisson(lam=10, size=n),
        "Total Backward Packets": rng.poisson(lam=8, size=n),
        "Flow Packets/s": rng.uniform(1, 50, size=n),
        "Label": [BENIGN_LABEL if i < 150 else MALICIOUS_LABEL for i in range(n)],
    })
    return df


def test_model_trainer_train_and_save(synthetic_train_data, tmp_path):
    pipeline = PreprocessingPipeline(test_size=0.25, random_state=42)
    X_train, X_test, y_train, y_test, features = pipeline.fit_transform(synthetic_train_data)

    trainer = ModelTrainer(random_state=42)

    # Train Logistic Regression
    lr_model, lr_time = trainer.train_model("logistic_regression", X_train, y_train)
    assert lr_model is not None
    assert lr_time >= 0.0

    # Train Random Forest
    rf_model, rf_time = trainer.train_model("random_forest", X_train, y_train)
    assert rf_model is not None
    assert rf_time >= 0.0

    # Test predictions
    preds, probs, pred_time = trainer.predict("random_forest", X_test)
    assert len(preds) == len(X_test)
    assert probs is not None
    assert len(probs) == len(X_test)

    # Test serialization
    saved_path = tmp_path / "test_rf.joblib"
    trainer.save_model("random_forest", filepath=saved_path)
    assert saved_path.exists()

    # Test reload
    new_trainer = ModelTrainer()
    loaded_bundle = new_trainer.load_model(saved_path)
    assert loaded_bundle["model_key"] == "random_forest"
    preds_reloaded, _, _ = new_trainer.predict("random_forest", X_test)
    assert np.array_equal(preds, preds_reloaded)


def test_threat_classifier_inference(synthetic_train_data):
    pipeline = PreprocessingPipeline(test_size=0.25, random_state=42)
    X_train, X_test, y_train, y_test, features = pipeline.fit_transform(synthetic_train_data)

    trainer = ModelTrainer(random_state=42)
    trainer.train_model("random_forest", X_train, y_train)

    rf_model = trainer.trained_models["random_forest"]
    classifier = ThreatClassifier(model=rf_model, preprocessor=pipeline)

    test_stream = pd.DataFrame({
        "Flow Duration": [10.0, 990.0],
        "Total Fwd Packets": [2, 100],
        "Total Backward Packets": [1, 2],
        "Flow Packets/s": [5.0, 250.0],
    })

    results, summary = classifier.classify_traffic(test_stream, confidence_threshold=0.50)

    assert "Predicted_Label" in results.columns
    assert "Malicious_Probability" in results.columns
    assert "Threat_Level" in results.columns
    assert summary["total_inspected_flows"] == 2
    assert set(results["Predicted_Label"].unique()).issubset({BENIGN_LABEL, MALICIOUS_LABEL})
