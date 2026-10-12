"""Unit tests for Investigation Report Generation."""

import pytest
import pandas as pd
from src.reports.investigation_report import (
    generate_case_investigation_report,
    generate_alert_triage_batch_report
)


def test_case_investigation_report_generation():
    case_data = {
        "case_id": "CASE-2026-0042",
        "title": "International Card Velocity Spike",
        "status": "INVESTIGATING",
        "priority": "CRITICAL",
        "assigned_analyst": "Officer-Jenkins",
        "created_at": "2026-10-12T10:00:00Z",
        "updated_at": "2026-10-12T11:30:00Z",
        "summary": "Series of high-value latent anomalies detected on card ending 4091.",
        "linked_transactions": ["TXN-SURV-10088", "TXN-SURV-10089"],
        "notes": [
            {
                "timestamp": "2026-10-12T10:15:00Z",
                "author": "Officer-Jenkins",
                "content": "Cardholder reported travel notice was not registered."
            }
        ],
        "history": [
            {
                "timestamp": "2026-10-12T10:00:00Z",
                "action": "CASE_CREATED",
                "details": "Case created",
                "author": "System"
            }
        ]
    }

    report = generate_case_investigation_report(case_data)

    assert "FINGUARD AI — COMPLIANCE CASE INVESTIGATION REPORT" in report
    assert "CASE-2026-0042" in report
    assert "CRITICAL" in report
    assert "Officer-Jenkins" in report
    assert "TXN-SURV-10088" in report
    assert "Cardholder reported travel notice" in report
    assert "Regulatory & Model Limitations Notice" in report


def test_alert_triage_batch_report_generation():
    alerts = [
        {
            "alert_id": "ALT-0001",
            "transaction_id": "TXN-8801",
            "amount": 2500.0,
            "anomaly_score": 0.88,
            "severity": "CRITICAL",
            "investigation_status": "NEW"
        },
        {
            "alert_id": "ALT-0002",
            "transaction_id": "TXN-8802",
            "amount": 350.0,
            "anomaly_score": 0.68,
            "severity": "HIGH",
            "investigation_status": "IN_REVIEW"
        }
    ]

    report = generate_alert_triage_batch_report(alerts, model_name="Isolation Forest", threshold=0.65)
    assert "ANOMALY ALERT TRIAGE SUMMARY REPORT" in report
    assert "Total Alerts Flagged:** 2" in report
    assert "ALT-0001" in report
    assert "TXN-8801" in report
    assert "$2,500.00" in report
