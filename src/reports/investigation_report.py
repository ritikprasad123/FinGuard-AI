"""Investigation report generation engine for FinGuard AI compliance audits."""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import pandas as pd
from config.settings import settings


def generate_case_investigation_report(
    case_data: Dict[str, Any],
    transactions_df: Optional[pd.DataFrame] = None
) -> str:
    """Generates a formal compliance investigation Markdown report for a case.

    Args:
        case_data: Dictionary representing case details, notes, and history from CaseManager.
        transactions_df: Optional DataFrame containing full transaction feature records.

    Returns:
        Structured Markdown investigation report ready for audit review or PDF export.
    """
    case_id = case_data.get("case_id", "CASE-UNKNOWN")
    title = case_data.get("title", "Untitled Investigation")
    status = case_data.get("status", "OPEN")
    priority = case_data.get("priority", "HIGH")
    analyst = case_data.get("assigned_analyst", "Compliance Desk")
    created_at = case_data.get("created_at", "N/A")
    updated_at = case_data.get("updated_at", "N/A")
    summary = case_data.get("summary", "No case summary provided.")
    linked_txns = case_data.get("linked_transactions", [])
    notes = case_data.get("notes", [])
    history = case_data.get("history", [])

    lines = [
        f"# FINGUARD AI — COMPLIANCE CASE INVESTIGATION REPORT",
        f"**Case Reference:** `{case_id}` | **Status:** `{status}` | **Priority:** `{priority}`",
        f"**Assigned Officer:** {analyst} | **Opened:** {created_at} | **Last Modified:** {updated_at}",
        "---",
        "",
        "## 1. Case Synopsis & Executive Summary",
        f"> {summary}",
        "",
        f"**Subject:** {title}",
        f"**Linked Transaction Count:** {len(linked_txns):,}",
        "",
        "## 2. Associated Transaction Details",
    ]

    if linked_txns:
        lines.extend([
            "| Transaction Reference | Amount | Anomaly Score | Prediction | Risk Tier |",
            "| :--- | :---: | :---: | :---: | :---: |"
        ])

        for tx_id in linked_txns:
            # Check if details are present in transactions_df
            amt_str = "$—"
            score_str = "—"
            pred_str = "INVESTIGATED"
            tier_str = priority

            if transactions_df is not None and len(transactions_df) > 0:
                match = transactions_df[transactions_df["Transaction_ID"] == tx_id]
                if len(match) > 0:
                    r = match.iloc[0]
                    amt = float(r.get(settings.AMOUNT_COL, 0.0))
                    score = float(r.get("Risk_Score", r.get("anomaly_score", 0.0)))
                    amt_str = f"${amt:,.2f}"
                    score_str = f"{score:.3f}"
                    pred_str = "ANOMALY" if score >= 0.65 else "NORMAL"
                    tier_str = str(r.get("Severity", r.get("risk_tier", priority)))

            lines.append(f"| `{tx_id}` | {amt_str} | `{score_str}` | `{pred_str}` | `{tier_str}` |")
    else:
        lines.append("*No individual transaction references currently linked to this case file.*")

    lines.extend([
        "",
        "## 3. Chronological Investigator Notes & Evidence Log",
    ])

    if notes:
        for idx, n in enumerate(notes, 1):
            ts = n.get("timestamp", "N/A")
            author = n.get("author", "Analyst")
            content = n.get("content", "")
            lines.extend([
                f"### Note #{idx} — [{ts}] by `{author}`",
                f"{content}",
                ""
            ])
    else:
        lines.append("*No investigation notes appended to date.*")

    lines.extend([
        "",
        "## 4. Tamper-Evident Audit History",
        "| Timestamp (UTC) | Action | Details | Officer |",
        "| :--- | :--- | :--- | :--- |"
    ])

    if history:
        for h in history:
            ts = h.get("timestamp", "N/A")
            act = h.get("action", "ACTION")
            det = h.get("details", "")
            auth = h.get("author", "System")
            lines.append(f"| {ts} | `{act}` | {det} | {auth} |")
    else:
        lines.append("| N/A | INITIAL_STATE | Case created | System |")

    lines.extend([
        "",
        "## 5. Disposition & Next Actions",
        f"- **Current Disposition:** `{status}`",
        "- **Standard Escalation Protocol:** If confirmed suspicious, export this memorandum and forward to the Designated Anti-Money Laundering (AML) Compliance Officer.",
        "",
        "## 6. Regulatory & Model Limitations Notice",
        "*FinGuard AI utilizes statistical anomaly detection and unsupervised outlier ranking. "
        "Flagged anomalies denote variance from historical baseline parameters and do NOT constitute confirmed legal "
        "proof of fraud. No cardholder identity or bank account history has been invented. All decisions must be "
        "reviewed and confirmed by an authorized compliance officer in accordance with applicable consumer financial protection laws.*",
        "",
        f"---",
        f"*Report generated by FinGuard AI Regulatory Engine on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*"
    ])

    return "\n".join(lines)


def generate_alert_triage_batch_report(
    alerts: List[Dict[str, Any]],
    model_name: str = "Isolation Forest",
    threshold: float = 0.65
) -> str:
    """Generates a summary batch report for compliance audits of flagged alerts."""
    crit_count = sum(1 for a in alerts if a.get("severity") == "CRITICAL")
    high_count = sum(1 for a in alerts if a.get("severity") == "HIGH")
    total_val = sum(float(a.get("amount", 0.0)) for a in alerts)

    lines = [
        "# FINGUARD AI — ANOMALY ALERT TRIAGE SUMMARY REPORT",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"**Active Model:** `{model_name}` | **Cutoff Threshold:** `{threshold:.2f}`",
        f"**Total Alerts Flagged:** {len(alerts):,} | **Total Exposed Value:** `${total_val:,.2f}`",
        f"**Critical Alerts:** {crit_count:,} | **High Alerts:** {high_count:,}",
        "---",
        "",
        "## Priority Alert Queue",
        "| Alert ID | Transaction ID | Amount | Score | Severity | Status |",
        "| :--- | :--- | :---: | :---: | :---: | :--- |"
    ]

    for a in alerts[:50]:  # Top 50 in summary table
        aid = a.get("alert_id", "—")
        tid = a.get("transaction_id", "—")
        amt = float(a.get("amount", 0.0))
        sc = float(a.get("anomaly_score", 0.0))
        sev = a.get("severity", "MEDIUM")
        st = a.get("investigation_status", "NEW")
        lines.append(f"| `{aid}` | `{tid}` | `${amt:,.2f}` | `{sc:.3f}` | `{sev}` | `{st}` |")

    lines.extend([
        "",
        "## Compliance Attestation",
        "*This report contains unconfirmed statistical anomalies flagged by automated surveillance. "
        "No final adverse action may be taken against any consumer account based solely on this automated output.*"
    ])

    return "\n".join(lines)
