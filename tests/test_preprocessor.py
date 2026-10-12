"""Tests for Feature Preprocessor and Train/Test Split."""

import pytest
import pandas as pd
import numpy as np
from src.data.demo_generator import generate_synthetic_demo_data
from src.features.preprocessor import TransactionPreprocessor, split_transaction_data


def test_transaction_preprocessor_fit_transform():
    df = generate_synthetic_demo_data(n_samples=100, save_to_file=False)
    prep = TransactionPreprocessor(scaler_type="RobustScaler")

    assert not prep.is_fitted
    df_scaled = prep.fit_transform(df)

    assert prep.is_fitted
    assert "Scaled_Amount" in df_scaled.columns
    assert "Scaled_Time" in df_scaled.columns
    assert "V14" in df_scaled.columns
    assert len(df_scaled) == 100


def test_split_transaction_data_prevent_leak():
    df = generate_synthetic_demo_data(n_samples=200, fraud_ratio=0.05, save_to_file=False)
    X_train, X_test, y_train, y_test = split_transaction_data(df, test_size=0.25, random_state=42)

    assert len(X_train) == 150
    assert len(X_test) == 50
    assert "Class" not in X_train.columns  # Target not in features
    assert "Transaction_ID" not in X_train.columns  # Metadata excluded
    assert y_train is not None
    assert len(y_train) == 150


def test_preprocessor_serialization(tmp_path):
    df = generate_synthetic_demo_data(n_samples=50, save_to_file=False)
    prep = TransactionPreprocessor()
    prep.fit(df)

    save_file = tmp_path / "test_prep.joblib"
    prep.save(save_file)
    assert save_file.is_file()

    loaded_prep = TransactionPreprocessor.load(save_file)
    assert loaded_prep.is_fitted
    assert loaded_prep.scaler_type == prep.scaler_type