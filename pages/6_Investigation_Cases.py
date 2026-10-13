"""FinGuard AI - Case Management Desk.

Persistent SQLite case repository, investigation notes, transaction linkage,
status transitions, and tamper-evident audit history.
"""

import sys
from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import settings
from components.theme import configure_page
from components.cards import render_header, render_metric_card
from src.investigation.case_manager import CaseManager
from src.reports.investigation_report import generate_case_investigation_report

configure_page(page_title="Investigation Cases")
case_manager = CaseManager()

render_header(
    title="Investigation Cases & Incident Desk",
    subtitle="Persistent SQLite investigation dossiers, chronological notes, status tracking, and audit trails.",
    badge_text="SQLITE PERSISTENT",
    badge_type="high"
)

# Metrics Row
case_metrics = case_manager.get_case_metrics()
cm1, cm2, cm3, cm4 = st.columns(4)
with cm1:
    st.markdown(
        render_metric_card(
            title="Total Case Files",
            value=f"{case_metrics['total_cases']:,}",
            subtitle="Database entries",
            status_color=settings.tokens.ACCENT_BLUE,
            badge_text="DATABASE"
        ),
        unsafe_allow_html=True
    )
with cm2:
    st.markdown(
        render_metric_card(
            title="Open / Active",
            value=f"{case_metrics['open_cases']:,}",
            subtitle="Awaiting analyst review",
            status_color=settings.tokens.PRIMARY_RED,
            badge_text="ACTIVE"
        ),
        unsafe_allow_html=True
    )
with cm3:
    st.markdown(
        render_metric_card(
            title="Under Investigation",
            value=f"{case_metrics['in_investigation']:,}",
            subtitle="Inquiry in progress",
            status_color=settings.tokens.WARNING_AMBER,
            badge_text="INQUIRY"
        ),
        unsafe_allow_html=True
    )
with cm4:
    st.markdown(
        render_metric_card(
            title="Escalated AML",
            value=f"{case_metrics['escalated_aml']:,}",
            subtitle="Forwarded to AML Unit",
            status_color=settings.tokens.CRITICAL_RED,
            badge_text="AML"
        ),
        unsafe_allow_html=True
    )

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

tab_view, tab_create = st.tabs(["Active Case Dossiers", "Open New Case File"])

with tab_create:
    st.markdown("#### Open New Forensic Investigation Case")
    with st.form("create_case_form"):
        title_in = st.text_input("Case Headline / Subject", placeholder="e.g. Card-not-present velocity outlier across high-risk MCC")
        c1, c2 = st.columns(2)
        with c1:
            prio_in = st.selectbox("Priority Level", options=["CRITICAL", "HIGH", "MEDIUM", "LOW"], index=1)
        with c2:
            analyst_in = st.text_input("Lead Investigating Officer", value="Compliance-Analyst-01")

        txns_in = st.text_input("Linked Transaction IDs (Comma separated)", placeholder="e.g. TXN-SURV-10023, TXN-SURV-10024")
        summary_in = st.text_area("Executive Summary", placeholder="Contextual synopsis of triggering events and anomalies...")
        note_in = st.text_area("Initial Case Note", placeholder="Preliminary observations, initial actions...")

        if st.form_submit_button("Create Case in Persistent Database", type="primary"):
            if not title_in.strip():
                st.error("Case headline is required.")
            else:
                txn_list = [t.strip() for t in txns_in.split(",") if t.strip()]
                new_id = case_manager.create_case(
                    title=title_in.strip(),
                    priority=prio_in,
                    assigned_analyst=analyst_in.strip(),
                    linked_transactions=txn_list,
                    summary=summary_in.strip(),
                    initial_note=note_in.strip() if note_in.strip() else None
                )
                st.success(f"Successfully opened case `{new_id}`! Switching to dossier view...")
                st.rerun()

with tab_view:
    f_c1, f_c2 = st.columns(2)
    with f_c1:
        st_filter = st.selectbox("Filter Status", options=["ALL"] + CaseManager.VALID_STATUSES)
    with f_c2:
        pr_filter = st.selectbox("Filter Priority", options=["ALL"] + CaseManager.VALID_PRIORITIES)

    cases = case_manager.list_cases(status=st_filter, priority=pr_filter)

    if not cases:
        st.info("No cases matching active filters in SQLite repository.")
    else:
        case_map = {f"{c['case_id']} — {c['title']} [{c['status']}]": c['case_id'] for c in cases}
        selected_label = st.selectbox("Select Case File", options=list(case_map.keys()))
        selected_id = case_map[selected_label]

        case_data = case_manager.get_case(selected_id)
        if case_data:
            # Case Header Card
            st.markdown(
                f"""
                <div class="fg-card" style="margin-top: 1rem; border-left: 3px solid var(--accent-red);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span class="fg-badge fg-badge-danger">{case_data['priority']} PRIORITY</span>
                            <span class="fg-badge fg-badge-neutral" style="margin-left: 0.5rem;">{case_data['status']}</span>
                            <span style="font-size: 0.8rem; color: var(--secondary-text); margin-left: 0.75rem;">Opened: {case_data['created_at'][:19]} UTC</span>
                        </div>
                        <div style="font-size: 0.8rem; color: var(--secondary-text);">
                            <strong>Officer:</strong> {case_data['assigned_analyst']}
                        </div>
                    </div>
                    <h4 style="margin-top: 0.75rem; margin-bottom: 0.5rem; color: var(--primary-text);">{case_data['title']}</h4>
                    <p style="color: var(--secondary-text); font-size: 0.88rem; line-height: 1.5;">{case_data['summary'] or 'No summary provided.'}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Linked Transactions
            st.markdown("##### Linked Transactions")
            txs = case_data.get("linked_transactions", [])
            st.write(", ".join([f"`{t}`" for t in txs]) if txs else "*No transactions linked yet.*")

            # Actions Tab
            act_n, act_s, act_l = st.tabs(["Post Note", "Transition Status", "Link Transaction"])

            with act_n:
                with st.form(f"note_form_{selected_id}"):
                    author_txt = st.text_input("Analyst Author", value="Compliance-Analyst-01")
                    note_txt = st.text_area("Case Observation / Memo")
                    if st.form_submit_button("Post Timestamped Note", type="primary"):
                        if note_txt.strip():
                            case_manager.add_case_note(selected_id, author_txt, note_txt)
                            st.success("Note logged in audit history!")
                            st.rerun()

            with act_s:
                with st.form(f"status_form_{selected_id}"):
                    new_stat = st.selectbox(
                        "Target Status",
                        options=CaseManager.VALID_STATUSES,
                        index=CaseManager.VALID_STATUSES.index(case_data['status'])
                    )
                    stat_reason = st.text_input("Reason / Compliance Justification", placeholder="e.g. Cardholder confirmed legitimate travel purchase")
                    if st.form_submit_button("Update Case Status", type="primary"):
                        case_manager.update_case_status(selected_id, new_stat, reason=stat_reason)
                        st.success(f"Case status updated to {new_stat}!")
                        st.rerun()

            with act_l:
                with st.form(f"link_form_{selected_id}"):
                    new_tx = st.text_input("Transaction ID to Associate", placeholder="e.g. TXN-SURV-10099")
                    if st.form_submit_button("Link to Case", type="primary"):
                        if new_tx.strip():
                            case_manager.link_transaction_to_case(selected_id, new_tx.strip())
                            st.success(f"Linked {new_tx}!")
                            st.rerun()

            # Notes Timeline
            st.markdown("##### Chronological Notes Log")
            if case_data["notes"]:
                for n in case_data["notes"]:
                    st.markdown(
                        f"""
                        <div style="background: var(--bg-card); padding: 0.75rem 1rem; border-radius: 6px; margin-bottom: 0.5rem; border: 1px solid var(--border-color);">
                            <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--secondary-text);">
                                <span><strong>{n['author']}</strong></span>
                                <span>{n['timestamp'][:19]} UTC</span>
                            </div>
                            <div style="font-size: 0.85rem; color: var(--primary-text); margin-top: 0.35rem;">{n['content']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No notes recorded to date.")

            # Audit History
            with st.expander("Tamper-Evident Audit Trail", expanded=False):
                st.dataframe(pd.DataFrame(case_data["history"])[["timestamp", "action", "details", "author"]], use_container_width=True)

            # Export Report
            st.markdown("##### Case Report Export")
            rep_md = generate_case_investigation_report(case_data)
            st.download_button(
                label=f"Download Investigation Report ({selected_id}.md)",
                data=rep_md,
                file_name=f"{selected_id}_case_report.md",
                mime="text/markdown"
            )


# Legal & Regulatory Decision-Support Notice
st.markdown(
    f"<div class='fg-legal-disclaimer'><strong>Regulatory Compliance Notice:</strong> {settings.DISCLAIMER_NOTICE}</div>",
    unsafe_allow_html=True
)
