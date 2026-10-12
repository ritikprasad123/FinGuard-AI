"""Unit tests for Isolation Forest Anomaly Detection Engine."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.data.demo_generator import generate_synthetic_demo_data
from src.models.isolation_forest import IsolationForestAnomalyModel
from src.models.base import InferenceResult


@pytest.fixture
def sample_data():
    return generate_synthetic_demo_data(n_samples=150, fraud_ratio=0.04, save_to_file=False)


def test_isolation_forest_fit_predict(sample_data):
    model = IsolationForestAnomalyModel(n_estimators=50, random_state=42)
    assert not model.is_trained

    model.fit(sample_data)
    assert model.is_trained
    assert len(model.feature_names_in_) == 30

    scores = model.predict_score(sample_data)
    assert len(scores) == 150
    assert (scores >= 0.0).all() and (scores <= 1.0).all()

    preds = model.predict(sample_data, threshold=0.60)
    assert len(preds) == 150
    assert set(np.unique(preds)).issubset({0, 1})


def test_isolation_forest_persistence(sample_data, tmp_path):
    model = IsolationForestAnomalyModel(n_estimators=50, random_state=42)
    model.fit(sample_data)

    save_path = tmp_path / "test_iforest.joblib"
    model.save(save_path)
    assert save_path.is_file()

    loaded_model = IsolationForestAnomalyModel.load(save_path)
    assert loaded_model.is_trained
    assert loaded_model.model_id == model.model_id

    # Verify identical outputs
    orig_scores = model.predict_score(sample_data)
    loaded_scores = loaded_model.predict_score(sample_data)
    np.testing.assert_allclose(orig_scores, loaded_scores, rtol=1e-5)


def test_isolation_forest_single_inference(sample_data):
    model = IsolationForestAnomalyModel(n_estimators=50, random_state=42)
    model.fit(sample_data)

    single_row = sample_data.iloc[0]
    res: InferenceResult = model.predict_single(single_row, transaction_id="TXN-UNIT-001", threshold=0.55)

    assert res.transaction_id == "TXN-UNIT-001"
    assert 0.0 <= res.anomaly_score <= 1.0
    assert res.applied_threshold == 0.55
    assert res.prediction_label in ["ANOMALY", "NORMAL"]
    assert res.risk_category in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    assert isinstance(res.top_contributing_features, dict)


def test_isolation_forest_batch_inference(sample_data):
    model = IsolationForestAnomalyModel(n_estimators=50, random_state=42)
    model.fit(sample_data)

    batch_res = model.predict_batch(sample_data, threshold=0.65)
    assert len(batch_res) == 150
    assert all(isinstance(r, InferenceResult) for r in batch_res)


def test_isolation_forest_missing_columns(sample_data):
    model = IsolationForestAnomalyModel(n_estimators=50, random_state=42)
    model.fit(sample_data)

    invalid_df = sample_data.drop(columns=["V14"])
    with pytest.raises(ValueError, match="missing required model columns"):
        model.predict_score(invalid_df)


def test_isolation_forest_empty_dataset():
    model = IsolationForestAnomalyModel()
    with pytest.raises(ValueError, match="empty dataset"):
        model.fit(pd.DataFrame())


def test_isolation_forest_unfitted_scoring(sample_data):
    model = IsolationForestAnomalyModel()
    with pytest.raises(RuntimeError, match="must be fitted"):
        model.predict_score(sample_data)


def test_isolation_forest_missing_artifact():
    with pytest.raises(FileNotFoundError):
        IsolationForestAnomalyModel.load("non_existent_model_artifact_xyz.joblib")