"""FinGuard AI - Investigation Reports Hub.

Generates audit-compliant compliance investigation reports, batch triage summaries,
model evaluation reports, and data quality certificates in exportable Markdown and CSV formats.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import settings
from components.theme import configure_page
from components.cards import render_header
from src.investigation.case_manager import CaseManager
from src.investigation.alert_feed import generate_alerts_from_dataset
from src.models.registry import get_active_model
from src.models.evaluation import evaluate_anomaly_model, split_transaction_data
from src.reports.investigation_report import (
    generate_case_investigation_report,
    generate_alert_triage_batch_report
)
try:
    from src.reports.quality_report import generate_markdown_quality_report
except ImportError:
    from src.reports.quality_report import format_quality_markdown_report as generate_markdown_quality_report
from src.data.validator import validate_transaction_data

configure_page(page_title="Investigation Reports")
case_manager = CaseManager()

render_header(
    title="Compliance & Investigation Reports",
    subtitle="Generate exportable audit-ready memoranda, batch triage reports, and pipeline quality certificates.",
    badge_text="REPORTS DESK",
    badge_type="high"
)

# Tabbed Interface for 4 Report Types
tab_case_rep, tab_triage_rep, tab_model_rep, tab_quality_rep = st.tabs([
    "Case Investigation Dossier",
    "Batch Alert Triage Report",
    "Model Evaluation Report",
    "Data Quality Audit Certificate"
])

# -------------------------------------------------------------
# TAB 1: CASE INVESTIGATION DOSSIER
# -------------------------------------------------------------
with tab_case_rep:
    st.markdown("### Formal Case Investigation Dossier")
    cases = case_manager.list_cases()

    if not cases:
        st.info("No active cases in SQLite database. Create a case from 'Case Management' or 'Alert Center'.")
    else:
        case_map = {f"{c['case_id']} — {c['title']} ({c['status']})": c['case_id'] for c in cases}
        chosen_label = st.selectbox("Select Case File for Report Generation", options=list(case_map.keys()))
        chosen_id = case_map[chosen_label]

        case_obj = case_manager.get_case(chosen_id)
        if case_obj:
            dataset_ctx = st.session_state.get("dataset")
            case_md = generate_case_investigation_report(case_obj, transactions_df=dataset_ctx)

            c_btn1, c_btn2 = st.columns(2)
            with c_btn1:
                st.download_button(
                    label=f"Download Case Dossier ({chosen_id}.md)",
                    data=case_md,
                    file_name=f"{chosen_id}_investigation_dossier.md",
                    mime="text/markdown",
                    type="primary",
                    use_container_width=True
                )
            with c_btn2:
                df_case_meta = pd.DataFrame([{
                    "Case_ID": case_obj["case_id"],
                    "Title": case_obj["title"],
                    "Status": case_obj["status"],
                    "Priority": case_obj["priority"],
                    "Assigned_Analyst": case_obj["assigned_analyst"],
                    "Created_At": case_obj["created_at"],
                    "Linked_Transactions": ", ".join(case_obj.get("linked_transactions", []))
                }])
                st.download_button(
                    label=f"Download Case Metadata (CSV)",
                    data=df_case_meta.to_csv(index=False),
                    file_name=f"{chosen_id}_metadata.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            st.markdown("#### Document Preview")
            st.markdown(
                f"<div style='background: var(--card-bg); padding: 1.5rem; border-radius: 6px; border: 1px solid var(--border-color); font-size: 0.88rem; line-height: 1.6;'>",
                unsafe_allow_html=True
            )
            st.markdown(case_md)
            st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 2: BATCH ALERT TRIAGE REPORT
# -------------------------------------------------------------
with tab_triage_rep:
    st.markdown("### Batch Alert Triage Summary Report")
    df_ctx = st.session_state.get("dataset")

    if df_ctx is None:
        st.info("No dataset loaded in surveillance session.")
    else:
        active_model = get_active_model()
        cutoff_val = active_model.threshold if (active_model and active_model.is_trained) else 0.65
        model_name_str = active_model.model_name if active_model else "Heuristic Engine"

        alerts = generate_alerts_from_dataset(df_ctx, model=active_model, threshold=cutoff_val, max_alerts=100)
        triage_md = generate_alert_triage_batch_report(alerts, model_name=model_name_str, threshold=cutoff_val)

        t_btn1, t_btn2 = st.columns(2)
        with t_btn1:
            st.download_button(
                label="Download Batch Triage Report (.md)",
                data=triage_md,
                file_name=f"finguard_alert_triage_report_{datetime.now(timezone.utc).strftime('%Y%m%d')}.md",
                mime="text/markdown",
                type="primary",
                use_container_width=True
            )
        with t_btn2:
            df_alerts_export = pd.DataFrame(alerts)
            if not df_alerts_export.empty:
                st.download_button(
                    label="Download Flagged Alerts (CSV)",
                    data=df_alerts_export.to_csv(index=False),
                    file_name=f"finguard_alerts_{datetime.now(timezone.utc).strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        st.markdown("#### Document Preview")
        st.markdown(
            f"<div style='background: var(--card-bg); padding: 1.5rem; border-radius: 6px; border: 1px solid var(--border-color); font-size: 0.88rem; line-height: 1.6;'>",
            unsafe_allow_html=True
        )
        st.markdown(triage_md)
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 3: MODEL EVALUATION PERFORMANCE REPORT
# -------------------------------------------------------------
with tab_model_rep:
    st.markdown("### Model Evaluation & Validation Report")
    df_ctx = st.session_state.get("dataset")

    if df_ctx is None:
        st.info("No dataset loaded in surveillance session.")
    elif settings.TARGET_COL not in df_ctx.columns:
        st.warning("Ground-truth 'Class' labels not present in the current dataset. Supervised evaluation requires benchmark labels.")
    else:
        active_model = get_active_model()
        cutoff_val = active_model.threshold if (active_model and active_model.is_trained) else 0.65
        model_name_str = active_model.model_name if active_model else "Active Engine"

        # Split and evaluate
        X_tr, X_te, y_tr, y_te = split_transaction_data(df_ctx, target_col="Class", test_size=0.20, random_state=42)
        if active_model and not active_model.is_trained:
            active_model.fit(X_tr)

        rep = evaluate_anomaly_model(active_model, X_te, y_te, threshold=cutoff_val)
        gen_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        eval_report_md = f"""# FINGUARD AI - MODEL PERFORMANCE VALIDATION AUDIT
**Generated:** {gen_time}
**Target Model:** {model_name_str} ({active_model.model_id if active_model else 'N/A'})
**Decision Cutoff Threshold:** {cutoff_val:.4f}
**Evaluation Dataset Partition:** Holdout Test Set (N={rep.total_samples:,})

---

## 1. Executive Performance Metrics
- **Precision (Positive Predictive Value):** {rep.precision:.2%}
- **Recall (Fraud Coverage / Sensitivity):** {rep.recall:.2%}
- **F1-Score (Harmonic Mean):** {rep.f1_score:.4f}
- **PR-AUC (Average Precision Area):** {rep.pr_auc:.4f}
- **ROC-AUC (Receiver Operating Curve Area):** {rep.roc_auc:.4f}

## 2. Confusion Matrix Audit
- **True Positives (Correctly Flagged Frauds):** {rep.true_positives:,}
- **False Positives (False Alarms Triaged):** {rep.false_positives:,}
- **True Negatives (Correctly Cleared Normal):** {rep.true_negatives:,}
- **False Negatives (Missed Frauds):** {rep.false_negatives:,}

## 3. Financial Compliance & Imbalance Caveat
In payment processing, fraud prevalence is typically below 0.2%. Overall raw Accuracy is misleading
because naive normal-class predicting achieves 99.8% while missing all actual fraud events.
Precision-Recall optimization is the required compliance standard.
Model anomaly scores indicate empirical statistical deviation and must be validated via analyst inquiry.
"""

        m_btn1, m_btn2 = st.columns(2)
        with m_btn1:
            st.download_button(
                label="Download Model Evaluation Report (.md)",
                data=eval_report_md,
                file_name=f"finguard_model_eval_report_{datetime.now(timezone.utc).strftime('%Y%m%d')}.md",
                mime="text/markdown",
                type="primary",
                use_container_width=True
            )
        with m_btn2:
            metrics_df = pd.DataFrame([rep.to_dict()])
            st.download_button(
                label="Download Evaluation Metrics (CSV)",
                data=metrics_df.to_csv(index=False),
                file_name=f"finguard_eval_metrics_{datetime.now(timezone.utc).strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.markdown("#### Document Preview")
        st.markdown(
            f"<div style='background: var(--card-bg); padding: 1.5rem; border-radius: 6px; border: 1px solid var(--border-color); font-size: 0.88rem; line-height: 1.6;'>",
            unsafe_allow_html=True
        )
        st.markdown(eval_report_md)
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 4: DATA QUALITY AUDIT CERTIFICATE
# -------------------------------------------------------------
with tab_quality_rep:
    st.markdown("### Data Quality & Pipeline Audit Certificate")
    df_ctx = st.session_state.get("dataset")

    if df_ctx is None:
        st.info("No dataset loaded in surveillance session.")
    else:
        quality_rep = validate_transaction_data(df_ctx)
        cert_md = generate_markdown_quality_report(quality_rep)

        st.download_button(
            label="Download Data Quality Certificate (.md)",
            data=cert_md,
            file_name=f"finguard_quality_audit_certificate_{datetime.now(timezone.utc).strftime('%Y%m%d')}.md",
            mime="text/markdown",
            type="primary",
            use_container_width=True
        )

        st.markdown("#### Document Preview")
        st.markdown(
            f"<div style='background: var(--card-bg); padding: 1.5rem; border-radius: 6px; border: 1px solid var(--border-color); font-size: 0.88rem; line-height: 1.6;'>",
            unsafe_allow_html=True
        )
        st.markdown(cert_md)
        st.markdown("</div>", unsafe_allow_html=True)
