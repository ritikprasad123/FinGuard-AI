"""Unit tests for Autoencoder Neural Anomaly Detection Engine."""

import pytest
import numpy as np
import pandas as pd
from src.data.demo_generator import generate_synthetic_demo_data
from src.models.autoencoder import AutoencoderAnomalyModel
from src.models.base import InferenceResult


@pytest.fixture
def sample_data():
    return generate_synthetic_demo_data(n_samples=100, fraud_ratio=0.03, save_to_file=False)


def test_autoencoder_fit_predict(sample_data):
    ae = AutoencoderAnomalyModel(hidden_layer_sizes=(16, 8, 16), max_iter=25, random_state=42)
    assert not ae.is_trained

    ae.fit(sample_data)
    assert ae.is_trained
    assert len(ae.feature_names_in_) == 30

    scores = ae.predict_score(sample_data)
    assert len(scores) == 100
    assert (scores >= 0.0).all() and (scores <= 1.0).all()


def test_autoencoder_persistence(sample_data, tmp_path):
    ae = AutoencoderAnomalyModel(hidden_layer_sizes=(16, 8, 16), max_iter=20, random_state=42)
    ae.fit(sample_data)

    save_path = tmp_path / "test_autoencoder.joblib"
    ae.save(save_path)
    assert save_path.is_file()

    loaded = AutoencoderAnomalyModel.load(save_path)
    assert loaded.is_trained

    orig_scores = ae.predict_score(sample_data)
    loaded_scores = loaded.predict_score(sample_data)
    np.testing.assert_allclose(orig_scores, loaded_scores, rtol=1e-5)


def test_autoencoder_single_inference(sample_data):
    ae = AutoencoderAnomalyModel(max_iter=20, random_state=42)
    ae.fit(sample_data)

    res: InferenceResult = ae.predict_single(sample_data.iloc[0], transaction_id="TXN-AE-001")
    assert res.transaction_id == "TXN-AE-001"
    assert 0.0 <= res.anomaly_score <= 1.0
    assert res.prediction_label in ["ANOMALY", "NORMAL"]


def test_autoencoder_batch_inference(sample_data):
    ae = AutoencoderAnomalyModel(max_iter=20, random_state=42)
    ae.fit(sample_data)

    results = ae.predict_batch(sample_data.head(20), threshold=0.70)
    assert len(results) == 20
    assert all(isinstance(r, InferenceResult) for r in results)
    assert all(r.applied_threshold == 0.70 for r in results)


def test_autoencoder_missing_columns(sample_data):
    ae = AutoencoderAnomalyModel(max_iter=20, random_state=42)
    ae.fit(sample_data)

    invalid_df = sample_data.drop(columns=["V10"])
    with pytest.raises(ValueError, match="missing required model columns"):
        ae.predict_score(invalid_df)


def test_autoencoder_empty_dataset():
    ae = AutoencoderAnomalyModel(max_iter=20, random_state=42)
    with pytest.raises(ValueError, match="empty dataset"):
        ae.fit(pd.DataFrame())


def test_autoencoder_unfitted_scoring(sample_data):
    ae = AutoencoderAnomalyModel(max_iter=20, random_state=42)
    with pytest.raises(RuntimeError, match="must be fitted"):
        ae.predict_score(sample_data)


def test_autoencoder_missing_artifact():
    with pytest.raises(FileNotFoundError):
        AutoencoderAnomalyModel.load("non_existent_autoencoder_artifact_xyz.joblib")
