"""Unit tests for AI Investigation Assistant."""

import pytest
import pandas as pd
from src.investigation.assistant import InvestigationAssistant


@pytest.fixture
def anomalous_transaction():
    return {
        "Transaction_ID": "TXN-TEST-AI-01",
        "Amount": 3450.00,
        "Time": 86400.0,
        "V14": -9.25,
        "V10": -6.40,
        "V12": -5.10,
        "V17": -4.80,
        "V4": 4.10
    }


def test_ai_assistant_analysis_structure(anomalous_transaction):
    briefing = InvestigationAssistant.analyze_transaction(
        transaction_data=anomalous_transaction,
        anomaly_score=0.912,
        threshold=0.65,
        model_id="IsolationForest-Production"
    )

    assert briefing["transaction_id"] == "TXN-TEST-AI-01"
    assert briefing["anomaly_score"] == 0.912
    assert briefing["threshold"] == 0.65
    assert briefing["score_delta"] == round(0.912 - 0.65, 4)
    assert briefing["risk_tier"] == "CRITICAL"
    assert briefing["prediction"] == "ANOMALY"
    assert briefing["amount"] == 3450.00
    assert "CRITICAL" in briefing["amount_severity"]

    # Verify mathematical rationale
    assert "0.912" in briefing["flag_reason"]
    assert "CRITICAL" in briefing["flag_reason"]

    # Verify top deviating features
    top_feats = briefing["top_deviating_features"]
    assert len(top_feats) > 0
    assert top_feats[0]["feature"] == "V14"
    assert top_feats[0]["z_score"] < -5.0

    # Verify suggested steps
    assert len(briefing["suggested_steps"]) >= 3

    # Verify limitations statement
    assert "STATISTICAL SURVEILLANCE LIMITATION NOTICE" in briefing["limitations"]
    assert "do NOT constitute legal proof of fraudulent intent" in briefing["limitations"]


def test_ai_assistant_markdown_generation(anomalous_transaction):
    briefing = InvestigationAssistant.analyze_transaction(
        transaction_data=anomalous_transaction,
        anomaly_score=0.885,
        threshold=0.65
    )

    md = InvestigationAssistant.generate_briefing_markdown(briefing)
    assert "### AI Investigation Briefing: `TXN-TEST-AI-01`" in md
    assert "Mathematical Anomaly Rationale" in md
    assert "Recommended Forensic Investigation Steps" in md
    assert "Compliance & Evidentiary Limitations" in md
    assert "| **V14** |" in md
