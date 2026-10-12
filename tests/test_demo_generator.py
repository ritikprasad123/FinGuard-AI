"""Tests for Synthetic Demo Generator."""

import pytest
from config.settings import settings
from src.data.demo_generator import generate_synthetic_demo_data


def test_generate_synthetic_demo_data_shape():
    n_samples = 300
    df = generate_synthetic_demo_data(n_samples=n_samples, fraud_ratio=0.01, save_to_file=False)

    assert len(df) == n_samples
    assert settings.TIME_COL in df.columns
    assert settings.AMOUNT_COL in df.columns
    assert settings.TARGET_COL in df.columns
    assert "_DATA_SOURCE" in df.columns
    assert df["_IS_DEMO"].all()
    assert (df[settings.AMOUNT_COL] > 0).all()  # All amounts positive


def test_generate_synthetic_demo_data_anomalies():
    n_samples = 500
    df = generate_synthetic_demo_data(n_samples=n_samples, fraud_ratio=0.02, save_to_file=False)
    anom_count = (df[settings.TARGET_COL] == 1).sum()

    assert anom_count >= 1
    assert "Risk_Score" in df.columns
    assert "Risk_Reason" in df.columns