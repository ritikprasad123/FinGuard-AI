"""Unit tests for Model Registry."""

import pytest
from src.data.demo_generator import generate_synthetic_demo_data
from src.models.isolation_forest import IsolationForestAnomalyModel
from src.models.registry import (
    get_active_model,
    set_active_model,
    get_active_model_metadata,
    list_saved_models
)


def test_registry_lifecycle():
    df = generate_synthetic_demo_data(n_samples=50, save_to_file=False)
    model = IsolationForestAnomalyModel(n_estimators=25, random_state=42)
    model.fit(df)

    set_active_model(model)
    active = get_active_model()
    assert active is not None
    assert active.model_id == model.model_id

    meta = get_active_model_metadata()
    assert meta["is_active"]
    assert meta["model_name"] == "Isolation Forest"

    saved = list_saved_models()
    assert isinstance(saved, list)