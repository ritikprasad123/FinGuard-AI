"""Anomaly detection algorithms, models, and registry."""

from src.models.base import BaseAnomalyModel, InferenceResult, classify_risk_tier, get_model_engine_status
from src.models.isolation_forest import IsolationForestAnomalyModel
from src.models.autoencoder import AutoencoderAnomalyModel
from src.models.evaluation import (
    evaluate_anomaly_model,
    evaluate_threshold_grid,
    find_optimal_threshold,
    ModelEvaluationReport
)
from src.models.registry import (
    get_active_model,
    set_active_model,
    get_active_model_metadata,
    list_saved_models
)

__all__ = [
    "BaseAnomalyModel",
    "InferenceResult",
    "classify_risk_tier",
    "get_model_engine_status",
    "IsolationForestAnomalyModel",
    "AutoencoderAnomalyModel",
    "evaluate_anomaly_model",
    "evaluate_threshold_grid",
    "find_optimal_threshold",
    "ModelEvaluationReport",
    "get_active_model",
    "set_active_model",
    "get_active_model_metadata",
    "list_saved_models"
]
