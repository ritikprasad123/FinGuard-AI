"""Tests for Metrics Calculation Engine."""

import pytest
import pandas as pd
from config.settings import settings
from src.monitoring.metrics import calculate_transaction_metrics, calculate_hourly_velocity
from src.data.demo_generator import generate_synthetic_demo_data


def test_calculate_metrics_empty():
    m = calculate_transaction_metrics(pd.DataFrame())
    assert m["total_transactions"] == 0
    assert m["total_volume_usd"] == 0.0
    assert m["anomalies_detected"] == 0


def test_calculate_metrics_with_data():
    df = generate_synthetic_demo_data(n_samples=200, fraud_ratio=0.02, save_to_file=False)
    m = calculate_transaction_metrics(df)

    assert m["total_transactions"] == 200
    assert m["total_volume_usd"] > 0
    assert m["avg_amount_usd"] > 0
    assert m["anomalies_detected"] >= 1
    assert m["ground_truth_available"]


def test_calculate_hourly_velocity():
    df = generate_synthetic_demo_data(n_samples=300, save_to_file=False)
    vel = calculate_hourly_velocity(df)

    assert len(vel) > 0
    assert "Hour" in vel.columns
    assert "Transaction_Count" in vel.columns
    assert vel["Transaction_Count"].sum() == 300