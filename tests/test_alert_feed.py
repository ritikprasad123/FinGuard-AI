"""Tests for Alert Feed and Investigation."""

import pytest
import pandas as pd
from src.data.demo_generator import generate_synthetic_demo_data
from src.investigation.alert_feed import extract_suspicious_transactions, calculate_statistical_risk_score


def test_calculate_statistical_risk_score():
    df = generate_synthetic_demo_data(n_samples=50, save_to_file=False)
    scores = calculate_statistical_risk_score(df)

    assert len(scores) == 50
    assert (scores >= 0.0).all()
    assert (scores <= 1.0).all()


def test_extract_suspicious_transactions():
    df = generate_synthetic_demo_data(n_samples=250, fraud_ratio=0.04, save_to_file=False)
    alerts = extract_suspicious_transactions(df, risk_threshold=0.60)

    assert len(alerts) > 0
    assert "Transaction_ID" in alerts.columns
    assert "Risk_Tier" in alerts.columns
    assert "Action_Required" in alerts.columns
    assert "Anomaly_Indicators" in alerts.columns


def test_generate_alerts_from_dataset():
    from src.investigation.alert_feed import generate_alerts_from_dataset
    df = generate_synthetic_demo_data(n_samples=100, fraud_ratio=0.10, save_to_file=False)
    alerts = generate_alerts_from_dataset(df, threshold=0.60, min_score=0.50, max_alerts=25)

    assert isinstance(alerts, list)
    assert len(alerts) > 0
    first_alert = alerts[0]
    assert "alert_id" in first_alert
    assert "transaction_id" in first_alert
    assert "anomaly_score" in first_alert
    assert "severity" in first_alert
    assert "supporting_info" in first_alert
    assert first_alert["investigation_status"] == "NEW"


def test_update_alert_status():
    from src.investigation.alert_feed import generate_alerts_from_dataset, update_alert_status
    df = generate_synthetic_demo_data(n_samples=50, fraud_ratio=0.08, save_to_file=False)
    alerts = generate_alerts_from_dataset(df, threshold=0.55, min_score=0.50, max_alerts=10)

    if alerts:
        a_id = alerts[0]["alert_id"]
        res = update_alert_status(alerts, a_id, "ESCALATED")
        assert res is True
        assert alerts[0]["investigation_status"] == "ESCALATED"

        with pytest.raises(ValueError, match="Invalid alert status"):
            update_alert_status(alerts, a_id, "INVALID_STATUS_STRING")