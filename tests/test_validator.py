"""Tests for Data Quality Validator."""

import pytest
import pandas as pd
import numpy as np
from config.settings import settings
from src.data.validator import validate_transaction_data


def test_validate_empty_dataframe():
    df = pd.DataFrame()
    report = validate_transaction_data(df)
    assert not report.is_valid
    assert report.total_rows == 0
    assert "Dataset is empty" in report.errors[0]


def test_validate_missing_columns():
    df = pd.DataFrame({"Time": [1.0, 2.0], "Amount": [10.0, 20.0]})
    report = validate_transaction_data(df)
    assert not report.is_valid
    assert len(report.missing_required_columns) > 0
    assert any("Missing" in err for err in report.errors)


def test_validate_negative_amount():
    cols = settings.REQUIRED_BENCHMARK_FEATURES
    data = {c: [1.0, 2.0] for c in cols}
    data["Amount"] = [10.0, -50.0]  # Negative amount
    df = pd.DataFrame(data)

    report = validate_transaction_data(df)
    assert not report.is_valid
    assert report.negative_amount_count == 1
    assert any("negative Amount" in err for err in report.errors)


def test_validate_perfect_dataset():
    cols = settings.REQUIRED_BENCHMARK_FEATURES + [settings.TARGET_COL]
    data = {c: np.random.randn(50) for c in cols}
    data["Amount"] = np.abs(np.random.randn(50) * 100) + 1.0
    data["Time"] = np.arange(50)
    data["Class"] = [0] * 48 + [1] * 2
    df = pd.DataFrame(data)

    report = validate_transaction_data(df)
    assert report.is_valid
    assert report.total_rows == 50
    assert report.total_columns == 31
    assert report.has_target_column
    assert report.class_distribution["0"] == 48
    assert report.class_distribution["1"] == 2
    assert report.fraud_rate_pct == 4.0
    assert report.imbalance_ratio == 24.0