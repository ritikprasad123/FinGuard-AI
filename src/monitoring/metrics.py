"""Operational surveillance metrics, transaction velocity, and risk volume calculations."""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from config.settings import settings


def calculate_transaction_metrics(
    df: pd.DataFrame,
    prediction_col: Optional[str] = None
) -> Dict[str, Any]:
    """Calculates operational and statistical KPIs across the active transaction stream.

    Args:
        df: Input transaction DataFrame.
        prediction_col: Optional column containing binary model predictions (1=Anomaly, 0=Normal).

    Returns:
        Dictionary of computed metric values.
    """
    total_txns = len(df)
    if total_txns == 0:
        return {
            "total_transactions": 0,
            "anomalies_detected": 0,
            "normal_transactions": 0,
            "anomaly_rate_pct": 0.0,
            "normal_rate_pct": 100.0,
            "total_volume_usd": 0.0,
            "avg_amount_usd": 0.0,
            "max_amount_usd": 0.0,
            "min_amount_usd": 0.0,
            "ground_truth_available": False,
            "ground_truth_frauds": 0,
            "ground_truth_fraud_rate_pct": 0.0,
            "critical_risk_count": 0,
            "high_risk_count": 0,
            "medium_risk_count": 0,
            "low_risk_count": 0,
        }

    # Volume Calculations
    has_amount = settings.AMOUNT_COL in df.columns
    total_volume = float(df[settings.AMOUNT_COL].sum()) if has_amount else 0.0
    avg_amount = float(df[settings.AMOUNT_COL].mean()) if has_amount else 0.0
    max_amount = float(df[settings.AMOUNT_COL].max()) if has_amount else 0.0
    min_amount = float(df[settings.AMOUNT_COL].min()) if has_amount else 0.0

    # Prediction / Anomaly Flagging
    if prediction_col and prediction_col in df.columns:
        anomalies_count = int((df[prediction_col] == 1).sum())
        normal_count = total_txns - anomalies_count
    elif "Risk_Score" in df.columns:
        anomalies_count = int((df["Risk_Score"] >= settings.RISK_HIGH_THRESHOLD).sum())
        normal_count = total_txns - anomalies_count
    elif settings.TARGET_COL in df.columns:
        anomalies_count = int((df[settings.TARGET_COL] == 1).sum())
        normal_count = total_txns - anomalies_count
    else:
        anomalies_count = 0
        normal_count = total_txns

    anomaly_rate = round((anomalies_count / total_txns) * 100, 3) if total_txns > 0 else 0.0
    normal_rate = round(100.0 - anomaly_rate, 3)

    # Ground Truth Statistics (If Class column is present)
    has_gt = settings.TARGET_COL in df.columns
    gt_frauds = int((df[settings.TARGET_COL] == 1).sum()) if has_gt else 0
    gt_fraud_rate = round((gt_frauds / total_txns) * 100, 4) if (has_gt and total_txns > 0) else 0.0

    # Risk Tier Breakdown
    crit_count, high_count, med_count, low_count = 0, 0, 0, 0
    if "Risk_Score" in df.columns:
        crit_count = int((df["Risk_Score"] >= settings.RISK_CRITICAL_THRESHOLD).sum())
        high_count = int(((df["Risk_Score"] >= settings.RISK_HIGH_THRESHOLD) & (df["Risk_Score"] < settings.RISK_CRITICAL_THRESHOLD)).sum())
        med_count = int(((df["Risk_Score"] >= settings.RISK_MEDIUM_THRESHOLD) & (df["Risk_Score"] < settings.RISK_HIGH_THRESHOLD)).sum())
        low_count = int((df["Risk_Score"] < settings.RISK_MEDIUM_THRESHOLD).sum())
    else:
        low_count = normal_count
        high_count = anomalies_count

    return {
        "total_transactions": total_txns,
        "anomalies_detected": anomalies_count,
        "normal_predictions": normal_count,
        "normal_transactions": normal_count,
        "anomaly_rate_pct": anomaly_rate,
        "normal_rate_pct": normal_rate,
        "total_volume_usd": total_volume,
        "avg_amount_usd": avg_amount,
        "max_amount_usd": max_amount,
        "min_amount_usd": min_amount,
        "ground_truth_available": has_gt,
        "ground_truth_frauds": gt_frauds,
        "ground_truth_fraud_rate_pct": gt_fraud_rate,
        "critical_risk_count": crit_count,
        "high_risk_count": high_count,
        "medium_risk_count": med_count,
        "low_risk_count": low_count,
    }


def calculate_hourly_velocity(df: pd.DataFrame, time_col: str = "Time") -> pd.DataFrame:
    """Computes hourly transaction counts and volume velocity over the stream."""
    if len(df) == 0 or time_col not in df.columns:
        return pd.DataFrame(columns=["Hour", "Transaction_Count", "Total_Amount", "Anomaly_Count"])

    temp = df.copy()
    temp["Hour"] = (temp[time_col] // 3600).astype(int)

    has_target = settings.TARGET_COL in temp.columns
    has_risk = "Risk_Score" in temp.columns

    if has_risk:
        temp["_is_anomaly"] = (temp["Risk_Score"] >= settings.RISK_HIGH_THRESHOLD).astype(int)
    elif has_target:
        temp["_is_anomaly"] = (temp[settings.TARGET_COL] == 1).astype(int)
    else:
        temp["_is_anomaly"] = 0

    agg_dict = {
        time_col: "count",
        "_is_anomaly": "sum"
    }
    if settings.AMOUNT_COL in temp.columns:
        agg_dict[settings.AMOUNT_COL] = "sum"

    grouped = temp.groupby("Hour").agg(agg_dict).reset_index()
    rename_map = {
        time_col: "Transaction_Count",
        "_is_anomaly": "Anomaly_Count"
    }
    if settings.AMOUNT_COL in temp.columns:
        rename_map[settings.AMOUNT_COL] = "Total_Amount"

    grouped = grouped.rename(columns=rename_map)
    return grouped
