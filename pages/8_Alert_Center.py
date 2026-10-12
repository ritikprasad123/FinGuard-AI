"""FinGuard AI - Alert Center & Forensic Triage Console.

Real-time surveillance alert triage, analyst acknowledgement, severity prioritization,
and persistent SQLite case linkage for suspicious financial transaction clusters.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import settings
from components.theme import configure_page
from components.cards import render_header, render_metric_card, render_banner
from src.investigation.case_manager import CaseManager
from src.investigation.alert_feed import generate_alerts_from_dataset
from src.models.registry import get_active_model
from src.data.loader import check_benchmark_dataset_exists, load_csv_dataset
from src.data.demo_generator import generate_synthetic_demo_data

configure_page(page_title="Alert Center")
case_manager = CaseManager()

# Session State Setup
if "dataset" not in st.session_state or st.session_state.dataset is None:
    bench_exists, bench_path = check_benchmark_dataset_exists()
    if bench_exists:
        try:
            df_bench, rep = load_csv_dataset(bench_path)
            st.session_state.dataset = df_bench
            st.session_state.data_source_type = "BENCHMARK"
        except Exception:
            pass

render_header(
    title="Alert Center & Forensic Triage",
    subtitle="Continuous operational alert triage, analyst acknowledgement, severity prioritization, and case linkage.",
    badge_text="ALERT DESK ACTIVE",
    badge_type="high"
)

if "dataset" not in st.session_state or st.session_state.dataset is None:
    render_banner(
        title="Dataset Required for Alert Generation",
        message="Please generate synthetic demo transactions or import the benchmark dataset to populate the Alert Center.",
        level="info"
    )
    if st.button("Generate Synthetic Demo Batch Now", type="primary"):
        demo_df = generate_synthetic_demo_data(n_samples=settings.DEMO_SAMPLES_COUNT, save_to_file=True)
        st.session_state.dataset = demo_df
        st.session_state.data_source_type = "DEMO"
        st.rerun()
    st.stop()

df = st.session_state.dataset
active_model = get_active_model()
active_threshold = active_model.threshold if (active_model and active_model.is_trained) else 0.65

# Ensure alerts are populated in SQLite from active dataset if table is empty
alert_stats = case_manager.get_alert_metrics()
if alert_stats["total_alerts"] == 0:
    fresh_alerts = generate_alerts_from_dataset(
        df, model=active_model, threshold=active_threshold, min_score=0.55, max_alerts=50
    )
    if fresh_alerts:
        case_manager.save_alerts_batch(fresh_alerts)
        alert_stats = case_manager.get_alert_metrics()

# Operational KPI Row
ac1, ac2, ac3, ac4 = st.columns(4)
with ac1:
    st.markdown(
        render_metric_card(
            title="Total Generated Alerts",
            value=f"{alert_stats['total_alerts']:,}",
            subtitle="Surveillance detections",
            status_color=settings.tokens.ACCENT_BLUE,
            badge_text="DATABASE"
        ),
        unsafe_allow_html=True
    )
with ac2:
    st.markdown(
        render_metric_card(
            title="Unacknowledged (New)",
            value=f"{alert_stats['unacknowledged_alerts']:,}",
            subtitle="Pending analyst triage",
            status_color=settings.tokens.PRIMARY_RED if alert_stats['unacknowledged_alerts'] > 0 else settings.tokens.SUCCESS_GREEN,
            badge_text="ACTION REQUIRED" if alert_stats['unacknowledged_alerts'] > 0 else "CLEARED"
        ),
        unsafe_allow_html=True
    )
with ac3:
    st.markdown(
        render_metric_card(
            title="Critical Severity",
            value=f"{alert_stats['critical_alerts']:,}",
            subtitle="Score >= 0.85",
            status_color=settings.tokens.CRITICAL_RED,
            badge_text="HIGH RISK"
        ),
        unsafe_allow_html=True
    )
with ac4:
    st.markdown(
        render_metric_card(
            title="Escalated to Cases",
            value=f"{alert_stats['linked_to_case']:,}",
            subtitle="Active investigation dossiers",
            status_color=settings.tokens.SUCCESS_GREEN,
            badge_text="LINKED"
        ),
        unsafe_allow_html=True
    )

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

# Triage Filter Controls
st.markdown("### Operational Alert Queue Filters")
flt_c1, flt_c2, flt_c3 = st.columns([1.5, 1.2, 1.5])
with flt_c1:
    search_q = st.text_input("Search Alert ID or Transaction ID", placeholder="e.g. ALT-0001 or TXN-SURV-10045")
with flt_c2:
    status_filter = st.selectbox(
        "Triage Status",
        options=["ALL", "NEW", "ACKNOWLEDGED", "UNDER_INVESTIGATION", "DISMISSED"],
        index=0
    )
with flt_c3:
    sev_filter = st.selectbox(
        "Severity Rating",
        options=["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
        index=0
    )

# Retrieve filtered alerts from SQLite
alerts_list = case_manager.list_alerts(
    status=status_filter if status_filter != "ALL" else None,
    severity=sev_filter if sev_filter != "ALL" else None,
    search_query=search_q if search_q.strip() else None,
    limit=150
)

st.markdown(f"**Found {len(alerts_list)} matching alert(s)**")

if not alerts_list:
    st.info("No alerts match the selected criteria. Use the filters above to adjust the view.")
    if st.button("Generate Fresh Alerts from Current Transactions", type="primary"):
        fresh = generate_alerts_from_dataset(
            df, model=active_model, threshold=active_threshold, min_score=0.50, max_alerts=50
        )
        case_manager.save_alerts_batch(fresh)
        st.success(f"Generated {len(fresh)} fresh alerts.")
        st.rerun()
else:
    # Action Toolbar: Sync / Export
    exp_c1, exp_c2 = st.columns([3, 1])
    with exp_c2:
        df_export = pd.DataFrame(alerts_list)
        csv_data = df_export.to_csv(index=False)
        st.download_button(
            label="Download Alerts (CSV)",
            data=csv_data,
            file_name="finguard_alerts_queue.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Available cases for linking
    active_cases = case_manager.list_cases()
    case_choices = {f"{c['case_id']}: {c['title']}": c['case_id'] for c in active_cases}

    for alert in alerts_list:
        a_id = alert["alert_id"]
        tx_id = alert["transaction_id"]
        score = alert["anomaly_score"]
        cutoff = alert["applied_threshold"]
        amount = alert["amount"]
        sev = alert["severity"]
        status = alert["status"]
        ack = alert["acknowledged"]
        linked_cid = alert.get("linked_case_id")
        created_time = alert.get("timestamp", "")[:19].replace("T", " ")

        sev_color = (
            settings.tokens.CRITICAL_RED if sev == "CRITICAL"
            else settings.tokens.PRIMARY_RED if sev == "HIGH"
            else settings.tokens.WARNING_AMBER if sev == "MEDIUM"
            else settings.tokens.SECONDARY_TEXT
        )

        with st.container():
            st.markdown(
                f"""
                <div style="background: var(--card-bg); border: 1px solid var(--border-color); border-left: 4px solid {sev_color}; border-radius: 6px; padding: 1rem 1.25rem; margin-bottom: 0.75rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <div>
                            <span style="font-family: 'JetBrains Mono', monospace; font-size: 1rem; font-weight: 700; color: #F4F6F8;">{a_id}</span>
                            <span style="margin-left: 0.75rem; color: #9AA4B2; font-size: 0.85rem;">Transaction: <code>{tx_id}</code></span>
                            <span style="margin-left: 0.75rem; color: #9AA4B2; font-size: 0.85rem;">Amount: <strong style="color: #F4F6F8;">${amount:,.2f}</strong></span>
                        </div>
                        <div style="display: flex; gap: 0.5rem; align-items: center;">
                            <span style="background: rgba(229, 72, 77, 0.15); color: {sev_color}; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">{sev}</span>
                            <span style="background: rgba(154, 164, 178, 0.15); color: #F4F6F8; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 500;">{status}</span>
                            {"<span style='background: rgba(53, 201, 138, 0.15); color: #35C98A; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem;'>ACKNOWLEDGED</span>" if ack else "<span style='background: rgba(255, 77, 90, 0.15); color: #FF4D5A; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem;'>UNACKNOWLEDGED</span>"}
                        </div>
                    </div>
                    <div style="margin-top: 0.5rem; font-size: 0.82rem; color: #9AA4B2; display: flex; gap: 1.5rem; flex-wrap: wrap;">
                        <span>Anomaly Score: <code style="color: #FF4D5A; font-weight: 600;">{score:.4f}</code> (Cutoff: {cutoff:.2f})</span>
                        <span>Model: {alert.get('model_id', 'SurveillanceEngine')}</span>
                        <span>Logged: {created_time} UTC</span>
                        {f"<span style='color: #35C98A;'>Linked Case: <strong>{linked_cid}</strong></span>" if linked_cid else ""}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Interactive action buttons
            b_col1, b_col2, b_col3, b_col4 = st.columns([1.2, 1.5, 1.8, 1.2])

            with b_col1:
                if not ack:
                    if st.button("Acknowledge", key=f"ack_{a_id}"):
                        case_manager.acknowledge_alert(a_id, analyst_name="Compliance-Analyst-01")
                        st.success(f"Alert {a_id} acknowledged.")
                        st.rerun()
                else:
                    st.caption("Acknowledged")

            with b_col2:
                if status != "DISMISSED":
                    if st.button("Dismiss Alert", key=f"dsm_{a_id}"):
                        case_manager.dismiss_alert(a_id, analyst_name="Compliance-Analyst-01", reason="Benign false alarm")
                        st.success(f"Alert {a_id} dismissed.")
                        st.rerun()

            with b_col3:
                if not linked_cid and case_choices:
                    chosen_case_lbl = st.selectbox(
                        "Link to Case",
                        options=list(case_choices.keys()),
                        key=f"sel_case_{a_id}",
                        label_visibility="collapsed"
                    )
                    if st.button("Link Case", key=f"link_{a_id}"):
                        chosen_cid = case_choices[chosen_case_lbl]
                        case_manager.link_alert_to_case(a_id, chosen_cid)
                        st.success(f"Linked {a_id} to case {chosen_cid}.")
                        st.rerun()

            with b_col4:
                if not linked_cid:
                    if st.button("Open New Case", key=f"newcase_{a_id}"):
                        new_cid = case_manager.create_case(
                            title=f"Alert {a_id} Investigation ({sev} - ${amount:,.2f})",
                            priority=sev if sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"] else "HIGH",
                            assigned_analyst="Compliance-Analyst-01",
                            linked_transactions=[tx_id],
                            summary=f"Automated case opened from Alert {a_id}. Anomaly score {score:.4f} exceeded threshold {cutoff:.2f}.",
                            initial_note=f"Triage triage initiated for transaction {tx_id} with score {score:.4f}."
                        )
                        case_manager.link_alert_to_case(a_id, new_cid)
                        st.success(f"Created and linked case {new_cid}!")
                        st.rerun()
