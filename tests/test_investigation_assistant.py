"""Unit tests for AI Investigation Assistant and Deterministic Query Interface."""

import pytest
import pandas as pd
import numpy as np
from src.investigation.assistant import InvestigationAssistant
from src.investigation.case_manager import CaseManager
from src.data.demo_generator import generate_synthetic_demo_data


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


@pytest.fixture
def scored_dataset():
    df = generate_synthetic_demo_data(n_samples=100, fraud_ratio=0.10, save_to_file=False)
    df["Transaction_ID"] = [f"TXN-SURV-{i+10001:05d}" for i in range(len(df))]
    # Assign simulated risk scores
    scores = np.linspace(0.1, 0.95, len(df))
    df["Risk_Score"] = scores
    df["anomaly_score"] = scores
    return df


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


def test_query_highest_anomaly_scores(scored_dataset):
    res = InvestigationAssistant.query(
        prompt="Which transactions have the highest anomaly scores?",
        df=scored_dataset,
        threshold=0.65
    )
    assert res["intent"] == "TOP_ANOMALIES"
    assert len(res["data"]) == 5
    assert "Top 5 Extreme Anomaly Detections" in res["response_markdown"]
    assert res["data"][0]["score"] >= res["data"][-1]["score"]


def test_query_flagged_patterns(scored_dataset):
    res = InvestigationAssistant.query(
        prompt="What patterns appear among flagged transactions?",
        df=scored_dataset,
        threshold=0.65
    )
    assert res["intent"] == "FLAGGED_PATTERNS"
    assert "patterns" in res["data"]
    assert len(res["data"]["patterns"]) > 0


def test_query_threshold_count(scored_dataset):
    res = InvestigationAssistant.query(
        prompt="How many transactions were flagged above the selected threshold?",
        df=scored_dataset,
        threshold=0.70
    )
    assert res["intent"] == "THRESHOLD_COUNT"
    assert res["data"]["threshold"] == 0.70
    assert res["data"]["flagged"] > 0
    assert res["data"]["total"] == 100


def test_query_explain_transaction(scored_dataset):
    first_tx_id = scored_dataset.iloc[-1]["Transaction_ID"]
    res = InvestigationAssistant.query(
        prompt=f"Why was {first_tx_id} flagged?",
        df=scored_dataset,
        threshold=0.65
    )
    assert res["intent"] == "EXPLAIN_TRANSACTION"
    assert first_tx_id in res["title"]
    assert "Mathematical Anomaly Rationale" in res["response_markdown"]


def test_query_case_summary(tmp_path):
    db_file = tmp_path / "test_assistant_cases.db"
    cm = CaseManager(db_path=db_file)
    cid = cm.create_case(
        title="Test Case for Assistant",
        priority="HIGH",
        assigned_analyst="Analyst-1",
        summary="Automated test case summary.",
        initial_note="First forensic observation."
    )
    res = InvestigationAssistant.query(
        prompt=f"Summarize this investigation case {cid}",
        case_manager=cm
    )
    assert res["intent"] == "CASE_SUMMARY"
    assert cid in res["title"]
    assert "Test Case for Assistant" in res["response_markdown"]
    assert "First forensic observation." in res["response_markdown"]
