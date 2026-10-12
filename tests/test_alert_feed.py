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