"""Unit tests for SQLite Case Management System."""

import pytest
from pathlib import Path
from src.investigation.case_manager import CaseManager


@pytest.fixture
def temp_case_manager(tmp_path):
    db_file = tmp_path / "test_cases.db"
    return CaseManager(db_path=db_file)


def test_case_creation_and_retrieval(temp_case_manager):
    cid = temp_case_manager.create_case(
        title="Unauthorized overseas card testing",
        priority="CRITICAL",
        assigned_analyst="Analyst-42",
        linked_transactions=["TXN-1001", "TXN-1002"],
        summary="High-velocity micro charges observed across foreign gateways.",
        initial_note="Opening case upon automated surveillance alert."
    )

    assert cid.startswith("CASE-")
    case = temp_case_manager.get_case(cid)
    assert case is not None
    assert case["title"] == "Unauthorized overseas card testing"
    assert case["priority"] == "CRITICAL"
    assert case["status"] == "OPEN"
    assert case["assigned_analyst"] == "Analyst-42"
    assert "TXN-1001" in case["linked_transactions"]
    assert "TXN-1002" in case["linked_transactions"]
    assert len(case["notes"]) == 1
    assert case["notes"][0]["content"] == "Opening case upon automated surveillance alert."
    assert len(case["history"]) >= 1


def test_case_status_transition_and_audit_history(temp_case_manager):
    cid = temp_case_manager.create_case(title="Suspicious volume spike", priority="HIGH")

    # Update status to INVESTIGATING
    res = temp_case_manager.update_case_status(
        cid, "INVESTIGATING", author="Officer-Smith", reason="Contacting issuing bank"
    )
    assert res is True

    case = temp_case_manager.get_case(cid)
    assert case["status"] == "INVESTIGATING"

    # Verify audit history
    actions = [h["action"] for h in case["history"]]
    assert "STATUS_CHANGE" in actions

    # Reject invalid status
    with pytest.raises(ValueError, match="Invalid status"):
        temp_case_manager.update_case_status(cid, "INVALID_STATUS_CODE")


def test_case_notes_logging(temp_case_manager):
    cid = temp_case_manager.create_case(title="Test notes logging")

    nid1 = temp_case_manager.add_case_note(cid, author="Analyst-A", content="First observation: V14 is -8.5.")
    nid2 = temp_case_manager.add_case_note(cid, author="Analyst-B", content="Second observation: IP mismatch.")

    assert nid1 > 0 and nid2 > nid1

    case = temp_case_manager.get_case(cid)
    assert len(case["notes"]) == 2
    assert case["notes"][0]["author"] == "Analyst-A"
    assert case["notes"][1]["author"] == "Analyst-B"

    # Reject empty note
    with pytest.raises(ValueError, match="cannot be empty"):
        temp_case_manager.add_case_note(cid, author="Analyst-A", content="   ")


def test_link_transaction_and_lookup(temp_case_manager):
    cid = temp_case_manager.create_case(title="Multi-transaction cluster")
    temp_case_manager.link_transaction_to_case(cid, "TXN-ALPHA-99")
    temp_case_manager.link_transaction_to_case(cid, "TXN-BETA-88")

    matches_alpha = temp_case_manager.find_cases_by_transaction("TXN-ALPHA-99")
    assert len(matches_alpha) == 1
    assert matches_alpha[0]["case_id"] == cid

    matches_none = temp_case_manager.find_cases_by_transaction("NON-EXISTENT-TXN")
    assert len(matches_none) == 0


def test_list_cases_filtering_and_metrics(temp_case_manager):
    temp_case_manager.create_case(title="Critical Case", priority="CRITICAL")
    c2 = temp_case_manager.create_case(title="High Case", priority="HIGH")
    temp_case_manager.update_case_status(c2, "ESCALATED_AML")

    all_cases = temp_case_manager.list_cases()
    assert len(all_cases) == 2

    crit_cases = temp_case_manager.list_cases(priority="CRITICAL")
    assert len(crit_cases) == 1
    assert crit_cases[0]["priority"] == "CRITICAL"

    aml_cases = temp_case_manager.list_cases(status="ESCALATED_AML")
    assert len(aml_cases) == 1
    assert aml_cases[0]["status"] == "ESCALATED_AML"

    metrics = temp_case_manager.get_case_metrics()
    assert metrics["total_cases"] == 2
    assert metrics["critical_priority"] == 1
    assert metrics["escalated_aml"] == 1


def test_alert_persistence_acknowledgement_and_linking(temp_case_manager):
    alert_payload = {
        "alert_id": "ALT-TEST-001",
        "transaction_id": "TXN-ALERT-999",
        "anomaly_score": 0.88,
        "applied_threshold": 0.65,
        "prediction": "ANOMALY",
        "severity": "CRITICAL",
        "amount": 4500.0,
        "timestamp": "2026-10-12T12:00:00Z",
        "model_id": "IsolationForest-v1",
        "investigation_status": "NEW"
    }

    # 1. Save alert
    saved_id = temp_case_manager.save_or_update_alert(alert_payload)
    assert saved_id == "ALT-TEST-001"

    # 2. Retrieve alert
    fetched = temp_case_manager.get_alert("ALT-TEST-001")
    assert fetched is not None
    assert fetched["transaction_id"] == "TXN-ALERT-999"
    assert fetched["anomaly_score"] == 0.88
    assert not fetched["acknowledged"]

    # 3. Acknowledge alert
    ack_res = temp_case_manager.acknowledge_alert("ALT-TEST-001", analyst_name="Senior-Analyst")
    assert ack_res is True

    fetched_after_ack = temp_case_manager.get_alert("ALT-TEST-001")
    assert fetched_after_ack["acknowledged"] is True
    assert fetched_after_ack["acknowledged_by"] == "Senior-Analyst"
    assert fetched_after_ack["status"] == "ACKNOWLEDGED"

    # 4. Link alert to an investigation case
    case_id = temp_case_manager.create_case(title="Linked from Alert Test", priority="CRITICAL")
    link_res = temp_case_manager.link_alert_to_case("ALT-TEST-001", case_id)
    assert link_res is True

    fetched_linked = temp_case_manager.get_alert("ALT-TEST-001")
    assert fetched_linked["linked_case_id"] == case_id
    assert fetched_linked["status"] == "UNDER_INVESTIGATION"

    # Verify transaction was linked to case
    case_obj = temp_case_manager.get_case(case_id)
    assert "TXN-ALERT-999" in case_obj["linked_transactions"]

    # 5. Alert metrics
    alert_metrics = temp_case_manager.get_alert_metrics()
    assert alert_metrics["total_alerts"] == 1
    assert alert_metrics["critical_alerts"] == 1
    assert alert_metrics["linked_to_case"] == 1


def test_app_settings_persistence(temp_case_manager):
    temp_case_manager.set_setting("decision_threshold", 0.72)
    temp_case_manager.set_setting("alert_email_notifications", {"enabled": True, "recipient": "compliance@bank.com"})

    thresh = temp_case_manager.get_setting("decision_threshold")
    assert thresh == 0.72

    email_cfg = temp_case_manager.get_setting("alert_email_notifications")
    assert isinstance(email_cfg, dict)
    assert email_cfg["enabled"] is True

    missing = temp_case_manager.get_setting("non_existent_key", default="DEFAULT_VAL")
    assert missing == "DEFAULT_VAL"

