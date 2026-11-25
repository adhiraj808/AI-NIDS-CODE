"""
Unit tests for data loader and preprocessing pipeline.
Verifies data leakage prevention, stratified splitting, and infinity/missing value handling.
"""

import numpy as np
import pandas as pd
import pytest

from config.settings import BENIGN_LABEL, MALICIOUS_LABEL, SAMPLE_DATA_DIR
from src.data_loader import DataLoader
from src.preprocessing import PreprocessingPipeline


@pytest.fixture
def sample_dataset_path():
    path = SAMPLE_DATA_DIR / "CICIDS2017_Benchmark_Sample.csv"
    if not path.exists():
        from src.sample_data import generate_cicids_benchmark_sample
        generate_cicids_benchmark_sample(n_samples=500, output_filepath=path)
    return path


def test_data_loader_load_and_detect_label(sample_dataset_path):
    loader = DataLoader()
    df = loader.load_csv(sample_dataset_path, max_rows=200)

    assert not df.empty
    assert len(df) == 200
    assert loader.detected_label_col is not None
    assert loader.detected_label_col.lower() == "label"


def test_label_binarization(sample_dataset_path):
    loader = DataLoader()
    df = loader.load_csv(sample_dataset_path, max_rows=300)
    df_bin, original_labels = loader.binarize_labels(df)

    assert "Original_Attack_Type" in df_bin.columns
    unique_labels = set(df_bin["Label"].unique())
    assert unique_labels.issubset({BENIGN_LABEL, MALICIOUS_LABEL})
    assert len(original_labels) == 300


def test_preprocessing_pipeline_prevents_leakage():
    # Construct synthetic toy dataset with missing values and infinities
    toy_df = pd.DataFrame({
        "Flow Duration": [100.0, np.inf, 200.0, -np.inf, 150.0, 300.0, 250.0, 400.0, 350.0, 500.0],
        "Total Fwd Packets": [1, 2, np.nan, 4, 2, 5, 3, 6, 4, 7],
        "Source IP": [f"192.168.1.{i}" for i in range(10)],  # Non-predictive to drop
        "Label": [BENIGN_LABEL, BENIGN_LABEL, MALICIOUS_LABEL, BENIGN_LABEL, MALICIOUS_LABEL,
                  BENIGN_LABEL, MALICIOUS_LABEL, BENIGN_LABEL, BENIGN_LABEL, MALICIOUS_LABEL],
    })

    pipeline = PreprocessingPipeline(test_size=0.30, random_state=42)
    X_train, X_test, y_train, y_test, features = pipeline.fit_transform(toy_df)

    # Asserts
    assert "Source IP" not in features
    assert pipeline.is_fitted
    assert not np.isnan(X_train).any(), "Training feature matrix contains NaN values"
    assert not np.isinf(X_train).any(), "Training feature matrix contains Inf values"
    assert not np.isnan(X_test).any(), "Test feature matrix contains NaN values"
    assert not np.isinf(X_test).any(), "Test feature matrix contains Inf values"
    assert len(y_train) + len(y_test) == 10

    # Scaler mean must match training data mean close to 0
    assert np.allclose(X_train.mean(axis=0), 0, atol=1e-6)


def test_transform_unseen_data():
    toy_train = pd.DataFrame({
        "Flow Duration": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
        "Total Fwd Packets": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "Label": [BENIGN_LABEL, BENIGN_LABEL, MALICIOUS_LABEL, BENIGN_LABEL, MALICIOUS_LABEL, BENIGN_LABEL],
    })
    pipeline = PreprocessingPipeline(test_size=0.33, random_state=42)
    pipeline.fit_transform(toy_train)

    toy_novel = pd.DataFrame({
        "Flow Duration": [150.0, np.inf],
        "Total Fwd Packets": [np.nan, 5.0],
    })

    X_novel = pipeline.transform(toy_novel)
    assert X_novel.shape == (2, 2)
    assert not np.isnan(X_novel).any()
    assert not np.isinf(X_novel).any()
