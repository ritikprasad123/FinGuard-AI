"""Tests for Executive Quality Summary and Markdown Report Generation."""

import pytest
import pandas as pd
import numpy as np
from src.data.demo_generator import generate_synthetic_demo_data
from src.data.validator import validate_transaction_data, DataQualityReport
from src.reports.quality_report import (
    generate_executive_quality_summary,
    format_quality_markdown_report
)


@pytest.fixture
def sample_report():
    df = generate_synthetic_demo_data(n_samples=150, fraud_ratio=0.04, save_to_file=False)
    return validate_transaction_data(df)


def test_generate_executive_quality_summary(sample_report):
    summary = generate_executive_quality_summary(sample_report)
    assert isinstance(summary, dict)
    assert summary["status"] == "PASSED"
    assert "150" in summary["total_transactions"]
    assert summary["features_count"] >= 30
    assert "completeness_pct" in summary
    assert "ground_truth_status" in summary
    assert summary["error_count"] == 0


def test_format_quality_markdown_report(sample_report):
    md = format_quality_markdown_report(sample_report)
    assert isinstance(md, str)
    assert "Data Quality Audit Report" in md
    assert "Total Transactions Ingested" in md
    assert "Ground-Truth Class Distribution" in md


def test_format_quality_markdown_report_with_errors():
    empty_df = pd.DataFrame()
    report = validate_transaction_data(empty_df)
    md = format_quality_markdown_report(report)
    assert "FAILED" in md
    assert "Validation Errors" in md
