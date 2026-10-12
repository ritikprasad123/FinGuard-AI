"""Base abstractions, contracts, and telemetry status for anomaly detection models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional, Union
import time
import numpy as np
import pandas as pd
from config.settings import settings


@dataclass
class InferenceResult:
    """Structured telemetry output for single-transaction and batch scoring."""
    transaction_id: str
    anomaly_score: float
    is_anomaly: bool
    applied_threshold: float
    risk_category: str
    prediction_label: str
    top_contributing_features: Dict[str, float] = field(default_factory=dict)
    inference_latency_ms: float = 0.0
    model_id: str = "UnknownModel"

    def to_dict(self) -> Dict[str, Any]:
        """Converts result to serializable dictionary."""
        return asdict(self)


def classify_risk_tier(score: float) -> str:
    """Classifies continuous anomaly score into categorical risk tier."""
    if score >= settings.RISK_CRITICAL_THRESHOLD:
        return "CRITICAL"
    if score >= settings.RISK_HIGH_THRESHOLD:
        return "HIGH"
    if score >= settings.RISK_MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def get_model_engine_status() -> Dict[str, Any]:
    """Returns engine telemetry for overview cards."""
    from src.models.registry import get_active_model
    active_m = get_active_model()
    if active_m and active_m.is_trained:
        return {
            "status_code": "ACTIVE INFERENCE",
            "status_label": f"Active Engine: {active_m.model_name}",
            "is_model_trained": True,
            "active_model_name": active_m.model_name,
            "model_id": active_m.model_id,
            "threshold": active_m.threshold,
            "framework": "Scikit-Learn & NumPy"
        }
    return {
        "status_code": "PHASE 2 ENGINE READY",
        "status_label": "Awaiting Model Training / Activation",
        "is_model_trained": False,
        "active_model_name": "Statistical Rule Baseline",
        "model_id": "RuleBaseline",
        "threshold": settings.RISK_HIGH_THRESHOLD,
        "framework": "Scikit-Learn & NumPy"
    }


class BaseAnomalyModel(ABC):
    """Abstract base contract for financial anomaly detection engines."""

    def __init__(self, model_name: str, model_id: str, default_threshold: float = 0.65):
        self.model_name = model_name
        self.model_id = model_id
        self.threshold = default_threshold
        self.is_trained: bool = False
        self.feature_names_in_: List[str] = []

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BaseAnomalyModel":
        """Fits unsupervised detector on feature matrix X."""
        pass

    @abstractmethod
    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """Returns continuous anomaly scores bounded in [0.0, 1.0]."""
        pass

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Returns binary predictions (1=Anomaly, 0=Normal)."""
        thresh = threshold if threshold is not None else self.threshold
        scores = self.predict_score(X)
        return (scores >= thresh).astype(int)

    def predict_single(
        self,
        row: pd.Series,
        transaction_id: str = "TXN-INFER",
        threshold: Optional[float] = None
    ) -> InferenceResult:
        """Performs real-time scoring on a single transaction."""
        start_t = time.perf_counter()
        thresh = threshold if threshold is not None else self.threshold
        df_row = pd.DataFrame([row])
        score = float(self.predict_score(df_row)[0])
        is_anom = score >= thresh
        category = classify_risk_tier(score)
        label = "ANOMALY" if is_anom else "NORMAL"
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        top_feats = {}
        for c in ["V14", "V17", "V12", "V10", "Amount"]:
            if c in row:
                try:
                    top_feats[c] = round(float(row[c]), 4)
                except Exception:
                    pass

        return InferenceResult(
            transaction_id=transaction_id,
            anomaly_score=round(score, 4),
            is_anomaly=is_anom,
            applied_threshold=round(thresh, 4),
            risk_category=category,
            prediction_label=label,
            top_contributing_features=top_feats,
            inference_latency_ms=round(elapsed_ms, 2),
            model_id=self.model_id
        )

    def predict_batch(
        self,
        X: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> List[InferenceResult]:
        """Performs batch scoring across multiple transactions."""
        if not self.is_trained:
            raise RuntimeError(f"Model {self.model_id} must be fitted prior to scoring.")

        thresh = threshold if threshold is not None else self.threshold
        scores = self.predict_score(X)

        results = []
        for idx, (original_idx, row) in enumerate(X.iterrows()):
            tid = str(row.get("Transaction_ID", f"TXN-BATCH-{original_idx}"))
            score = float(scores[idx])
            is_anom = score >= thresh
            category = classify_risk_tier(score)
            label = "ANOMALY" if is_anom else "NORMAL"

            top_feats = {}
            for c in ["V14", "V17", "V12", "V10", "Amount"]:
                if c in row:
                    try:
                        top_feats[c] = round(float(row[c]), 4)
                    except Exception:
                        pass

            results.append(InferenceResult(
                transaction_id=tid,
                anomaly_score=round(score, 4),
                is_anomaly=is_anom,
                applied_threshold=round(thresh, 4),
                risk_category=category,
                prediction_label=label,
                top_contributing_features=top_feats,
                inference_latency_ms=0.0,
                model_id=self.model_id
            ))
        return results
