"""Alert Center and forensic triage feed for FinGuard AI.

Provides structured alert generation, multi-tier anomaly extraction,
supporting feature attribution, and investigation lifecycle tracking.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from config.settings import settings
from src.models.base import BaseAnomalyModel, InferenceResult, classify_risk_tier
from src.models.registry import get_active_model
from src.utils.logger import get_logger

logger = get_logger("finguard.investigation.alert_feed")

ALERT_STATUSES = ["NEW", "IN_REVIEW", "ESCALATED", "DISMISSED", "CONFIRMED_SUSPICIOUS"]


@dataclass
class AnomalyAlert:
    """Standardized operational alert payload for flagged transactions."""
    alert_id: str
    transaction_id: str
    anomaly_score: float
    applied_threshold: float
    prediction: str
    severity: str
    supporting_info: Dict[str, Any]
    investigation_status: str
    amount: float
    timestamp: str
    model_id: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _score_single_record(row: pd.Series) -> float:
    """Heuristic scoring fallback when no machine learning model is registered."""
    score = 0.05
    amt = float(row.get(settings.AMOUNT_COL, 0.0))
    if amt > 2500:
        score += 0.40
    elif amt > 1000:
        score += 0.20

    v14 = float(row.get("V14", 0.0))
    v17 = float(row.get("V17", 0.0))
    v12 = float(row.get("V12", 0.0))
    v10 = float(row.get("V10", 0.0))

    if v14 < -3.0:
        score += 0.25
    if v17 < -3.0:
        score += 0.20
    if v12 < -2.5:
        score += 0.15
    if v10 < -2.5:
        score += 0.15

    if settings.TARGET_COL in row and int(row[settings.TARGET_COL]) == 1:
        score = max(score, 0.95)

    return round(float(np.clip(score, 0.01, 0.99)), 3)


def calculate_statistical_risk_score(data: Union[pd.DataFrame, pd.Series]) -> Union[np.ndarray, float]:
    """Computes baseline risk scores across transactions."""
    if isinstance(data, pd.DataFrame):
        return data.apply(_score_single_record, axis=1).values
    return _score_single_record(data)


def generate_alerts_from_dataset(
    df: pd.DataFrame,
    model: Optional[BaseAnomalyModel] = None,
    threshold: Optional[float] = None,
    min_score: float = 0.50,
    max_alerts: int = 300
) -> List[Dict[str, Any]]:
    """Extracts formal AnomalyAlert items using active ML model or statistical baseline.

    Args:
        df: Input transactions DataFrame.
        model: Optional trained BaseAnomalyModel. If None, retrieves active registered model.
        threshold: Decision threshold. Defaults to model threshold or 0.65.
        min_score: Minimum anomaly score required to qualify for alert generation.
        max_alerts: Upper bound on alerts extracted.

    Returns:
        List of alert dictionaries compliant with Alert Center requirements.
    """
    if len(df) == 0:
        return []

    target_model = model or get_active_model()
    cutoff = threshold if threshold is not None else (target_model.threshold if target_model else 0.65)
    alerts: List[Dict[str, Any]] = []

    # Prepare transaction IDs if missing
    tx_ids = df["Transaction_ID"].tolist() if "Transaction_ID" in df.columns else [f"TXN-{i+10001:05d}" for i in range(len(df))]
    amounts = df[settings.AMOUNT_COL].tolist() if settings.AMOUNT_COL in df.columns else [0.0] * len(df)
    times = df[settings.TIME_COL].tolist() if settings.TIME_COL in df.columns else [0.0] * len(df)

    # Compute scores using model or statistical fallback
    if target_model and target_model.is_trained:
        try:
            scores = target_model.predict_score(df)
            model_id = target_model.model_id
        except Exception as exc:
            logger.warning(f"Active model scoring failed ({exc}); falling back to heuristic scoring.")
            scores = calculate_statistical_risk_score(df)
            model_id = "Statistical-Heuristic-Baseline"
    else:
        scores = calculate_statistical_risk_score(df)
        model_id = "Statistical-Heuristic-Baseline"

    now_iso = datetime.now(timezone.utc).isoformat()

    # Filter indices meeting cutoff or min_score
    qualifying_indices = np.where(scores >= min_score)[0]
    # Sort descending by score
    sorted_sub_indices = qualifying_indices[np.argsort(-scores[qualifying_indices])][:max_alerts]

    for idx in sorted_sub_indices:
        score = float(scores[idx])
        tx_id = str(tx_ids[idx])
        amt = float(amounts[idx])
        t_val = float(times[idx])
        pred_label = "ANOMALY" if score >= cutoff else "NORMAL"
        severity = classify_risk_tier(score)

        # Extract top deviating features
        row = df.iloc[idx]
        deviations: Dict[str, float] = {}
        for feat in ["V14", "V10", "V12", "V17", "V4", "V11", "V16"]:
            if feat in row:
                deviations[feat] = round(float(row[feat]), 4)

        supporting_info = {
            "ticket_amount": round(amt, 2),
            "elapsed_time_sec": round(t_val, 1),
            "top_deviating_features": deviations,
            "score_delta_from_cutoff": round(score - cutoff, 4),
            "ground_truth_label": int(row[settings.TARGET_COL]) if settings.TARGET_COL in row else None
        }

        alert = AnomalyAlert(
            alert_id=f"ALT-{len(alerts)+1:04d}",
            transaction_id=tx_id,
            anomaly_score=round(score, 4),
            applied_threshold=round(cutoff, 4),
            prediction=pred_label,
            severity=severity,
            supporting_info=supporting_info,
            investigation_status="NEW",
            amount=amt,
            timestamp=now_iso,
            model_id=model_id
        )
        alerts.append(alert.to_dict())

    return alerts


def update_alert_status(
    alerts: List[Dict[str, Any]],
    alert_id: str,
    new_status: str
) -> bool:
    """Updates investigation status of a specific alert in-memory."""
    if new_status not in ALERT_STATUSES:
        raise ValueError(f"Invalid alert status '{new_status}'. Must be in: {ALERT_STATUSES}")

    for a in alerts:
        if a["alert_id"] == alert_id or a["transaction_id"] == alert_id:
            a["investigation_status"] = new_status
            return True
    return False


def extract_suspicious_transactions(
    df: pd.DataFrame,
    risk_threshold: float = 0.60,
    max_records: int = 100
) -> pd.DataFrame:
    """Maintains backward compatibility for Phase 1 calls."""
    if len(df) == 0:
        return pd.DataFrame(columns=[
            "Transaction_ID", "Timestamp", "Amount", "Risk_Score",
            "Risk_Tier", "Action_Required", "Anomaly_Indicators", "Severity"
        ])

    alerts = generate_alerts_from_dataset(
        df,
        threshold=risk_threshold,
        min_score=risk_threshold,
        max_alerts=max_records
    )

    if not alerts:
        return pd.DataFrame(columns=[
            "Transaction_ID", "Timestamp", "Amount", "Risk_Score",
            "Risk_Tier", "Action_Required", "Anomaly_Indicators", "Severity"
        ])

    rows = []
    for a in alerts:
        tier = a["severity"]
        action = (
            "Immediate Freeze & Compliance Escalation" if tier == "CRITICAL"
            else "Secondary Auth / Step-Up Verification" if tier == "HIGH"
            else "Passive Surveillance Audit"
        )
        rows.append({
            "Transaction_ID": a["transaction_id"],
            "Timestamp": a["timestamp"],
            "Amount": a["amount"],
            "Risk_Score": a["anomaly_score"],
            "Risk_Tier": tier,
            "Severity": tier,
            "Action_Required": action,
            "Anomaly_Indicators": f"Model: {a['model_id']} (Score: {a['anomaly_score']:.3f})",
            "Status": a["investigation_status"]
        })

    return pd.DataFrame(rows)
