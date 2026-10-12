"""Unit tests for Local Outlier Factor (LOF) Novelty Detector."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.data.demo_generator import generate_synthetic_demo_data
from src.models.lof_detector import LocalOutlierFactorAnomalyModel


@pytest.fixture
def synthetic_txns():
    return generate_synthetic_demo_data(n_samples=150, fraud_ratio=0.06, save_to_file=False)


def test_lof_initialization():
    model = LocalOutlierFactorAnomalyModel(n_neighbors=15, default_threshold=0.60)
    assert model.model_name == "Local Outlier Factor (LOF)"
    assert model.threshold == 0.60
    assert not model.is_trained


def test_lof_fit_and_predict(synthetic_txns):
    model = LocalOutlierFactorAnomalyModel(n_neighbors=10, max_train_samples=100, random_state=42)
    model.fit(synthetic_txns)

    assert model.is_trained
    assert len(model.feature_names_in_) > 0

    scores = model.predict_score(synthetic_txns)
    assert len(scores) == len(synthetic_txns)
    assert (scores >= 0.0).all()
    assert (scores <= 1.0).all()

    # Binary prediction
    preds = model.predict(synthetic_txns, threshold=0.60)
    assert len(preds) == len(synthetic_txns)
    assert set(np.unique(preds)).issubset({0, 1})


def test_lof_single_prediction(synthetic_txns):
    model = LocalOutlierFactorAnomalyModel(n_neighbors=10, random_state=42)
    model.fit(synthetic_txns)

    row = synthetic_txns.iloc[0]
    res = model.predict_single(row, transaction_id="TXN-LOF-TEST")

    assert res.transaction_id == "TXN-LOF-TEST"
    assert 0.0 <= res.anomaly_score <= 1.0
    assert res.risk_category in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    assert res.prediction_label in ["ANOMALY", "NORMAL"]


def test_lof_persistence_and_loading(tmp_path, synthetic_txns):
    model = LocalOutlierFactorAnomalyModel(n_neighbors=10, random_state=42)
    model.fit(synthetic_txns)

    save_path = tmp_path / "test_lof.joblib"
    model.save(save_path)
    assert save_path.is_file()

    loaded = LocalOutlierFactorAnomalyModel.load(save_path)
    assert loaded.is_trained
    assert loaded.model_name == model.model_name
    assert loaded.threshold == model.threshold

    scores_orig = model.predict_score(synthetic_txns[:20])
    scores_loaded = loaded.predict_score(synthetic_txns[:20])
    np.testing.assert_allclose(scores_orig, scores_loaded, atol=1e-5)


def test_lof_empty_data_handling():
    model = LocalOutlierFactorAnomalyModel()
    with pytest.raises(ValueError, match="empty dataset"):
        model.fit(pd.DataFrame())

    # Untrained scoring
    with pytest.raises(RuntimeError, match="must be fitted"):
        model.predict_score(pd.DataFrame({"V1": [1.0]}))
