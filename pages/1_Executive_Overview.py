"""FinGuard AI - Executive Overview Dashboard.

Operational KPIs, active model telemetry, benchmark ground-truth reconciliation,
and surveillance visualizations for compliance risk officers.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import settings
from components.theme import configure_page
from components.cards import render_header, render_metric_card, render_banner
from components.charts import (
    create_transaction_activity_chart,
    create_amount_distribution_chart,
    create_hourly_volume_chart,
    create_anomaly_score_distribution_chart
)
from components.setup_card import render_setup_card
from src.data.loader import check_benchmark_dataset_exists, load_csv_dataset
from src.data.demo_generator import generate_synthetic_demo_data
from src.models.registry import get_active_model
from src.models.base import classify_risk_tier
from src.monitoring.metrics import calculate_transaction_metrics, calculate_hourly_velocity
from src.investigation.alert_feed import generate_alerts_from_dataset, calculate_statistical_risk_score
from src.investigation.case_manager import CaseManager
from src.utils.logger import get_logger

logger = get_logger("finguard.pages.executive_overview")

# Configure page
configure_page(page_title="Executive Overview")

# Initialize case manager for database metrics
case_manager = CaseManager()

# Session State Setup
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "data_source_type" not in st.session_state:
    st.session_state.data_source_type = None

# Auto-detect existing benchmark dataset on disk if not yet loaded
bench_exists, bench_path = check_benchmark_dataset_exists()
if st.session_state.dataset is None and bench_exists:
    try:
        df_bench, rep = load_csv_dataset(bench_path)
        st.session_state.dataset = df_bench
        st.session_state.data_source_type = "BENCHMARK"
    except Exception as exc:
        logger.warning(f"Could not automatically load benchmark dataset: {exc}")

# Header
render_header(
    title="Executive Surveillance Overview",
    subtitle="Continuous financial anomaly intelligence, active model performance, and operational triage KPIs.",
    badge_text="SURVEILLANCE ACTIVE" if st.session_state.dataset is not None else "SETUP REQUIRED",
    badge_type="success" if st.session_state.dataset is not None else "neutral"
)

# Zero-Data Setup Screen
if st.session_state.dataset is None:
    render_banner(
        title="Surveillance Dataset Required",
        message=(
            f"The public credit card fraud benchmark dataset was not found at <code>{settings.DEFAULT_BENCHMARK_PATH}</code>. "
            "To maintain strict financial compliance, FinGuard AI never fabricates missing historical records. "
            "You can generate synthetic demo transactions or import the verified benchmark dataset."
        ),
        level="info"
    )
    render_setup_card()

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    if st.button("Generate Synthetic Demo Batch (5,000 Transactions)", type="primary"):
        with st.spinner("Generating demo transaction stream..."):
            demo_df = generate_synthetic_demo_data(
                n_samples=settings.DEMO_SAMPLES_COUNT,
                fraud_ratio=settings.DEMO_ANOMALY_RATIO,
                save_to_file=True
            )
            st.session_state.dataset = demo_df
            st.session_state.data_source_type = "DEMO"
            st.rerun()
    st.stop()

# Data is available
df = st.session_state.dataset
is_demo = st.session_state.data_source_type == "DEMO"

if is_demo:
    render_banner(
        title="OPERATIONAL STATUS: DEMO SURVEILLANCE ACTIVE",
        message=(
            "Current surveillance metrics are computed on synthetic demonstration transactions. "
            "All telemetry, alerts, and model scoring represent local demonstration execution."
        ),
        level="demo",
        badge="DEMO DATA (SYNTHETIC)"
    )

# Retrieve active ML model and score dataset
active_model = get_active_model()
model_cutoff = active_model.threshold if (active_model and active_model.is_trained) else 0.65
model_name = active_model.model_name if (active_model and active_model.is_trained) else "Heuristic Rule Baseline"

# Ensure dataset has Risk_Score and Prediction computed
if "Transaction_ID" not in df.columns:
    df["Transaction_ID"] = [f"TXN-SURV-{i+10001:05d}" for i in range(len(df))]

if "Risk_Score" not in df.columns or "anomaly_score" not in df.columns:
    with st.spinner("Scoring dataset with active anomaly detection engine..."):
        if active_model and active_model.is_trained:
            try:
                scores = active_model.predict_score(df)
            except Exception:
                scores = calculate_statistical_risk_score(df)
        else:
            scores = calculate_statistical_risk_score(df)
        df["Risk_Score"] = np.round(scores, 4)
        df["anomaly_score"] = df["Risk_Score"]
        df["Prediction"] = np.where(df["Risk_Score"] >= model_cutoff, "ANOMALY", "NORMAL")
        df["Severity"] = [classify_risk_tier(s) for s in df["Risk_Score"]]
        st.session_state.dataset = df

# Metrics Calculation
total_txns = len(df)
anomalies_flagged = int((df["Prediction"] == "ANOMALY").sum()) if "Prediction" in df.columns else int((df["Risk_Score"] >= model_cutoff).sum())
normal_predictions = total_txns - anomalies_flagged
anomaly_rate = (anomalies_flagged / total_txns) if total_txns > 0 else 0.0

has_ground_truth = settings.TARGET_COL in df.columns
ground_truth_frauds = int(df[settings.TARGET_COL].sum()) if has_ground_truth else None
ground_truth_rate = (ground_truth_frauds / total_txns) if (has_ground_truth and total_txns > 0) else None

# Operational Case Metrics from SQLite
case_metrics = case_manager.get_case_metrics()

# -------------------------------------------------------------
# PRIMARY KPI METRIC CARDS ROW
# -------------------------------------------------------------
st.markdown("### Surveillance Telemetry & Risk KPIs")
kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)

with kpi_c1:
    st.markdown(
        render_metric_card(
            title="Transactions Analysed",
            value=f"{total_txns:,}",
            subtitle=f"Active Stream ({st.session_state.data_source_type})",
            status_color=settings.tokens.ACCENT_BLUE,
            badge_text="INGESTED"
        ),
        unsafe_allow_html=True
    )

with kpi_c2:
    st.markdown(
        render_metric_card(
            title="Anomalies Flagged",
            value=f"{anomalies_flagged:,}",
            subtitle=f"{anomaly_rate:.2%} of total volume",
            status_color=settings.tokens.PRIMARY_RED,
            badge_text="MODEL ANOMALIES"
        ),
        unsafe_allow_html=True
    )

with kpi_c3:
    st.markdown(
        render_metric_card(
            title="Normal Predictions",
            value=f"{normal_predictions:,}",
            subtitle=f"{(1.0 - anomaly_rate):.2%} cleared volume",
            status_color=settings.tokens.SUCCESS_GREEN,
            badge_text="NORMAL"
        ),
        unsafe_allow_html=True
    )

with kpi_c4:
    if has_ground_truth:
        st.markdown(
            render_metric_card(
                title="Benchmark Ground Truth",
                value=f"{ground_truth_frauds:,}",
                subtitle=f"Class=1 rate: {ground_truth_rate:.3%}",
                status_color=settings.tokens.WARNING_AMBER,
                badge_text="GROUND TRUTH"
            ),
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            render_metric_card(
                title="Active Engine",
                value=model_name,
                subtitle=f"Cutoff: {model_cutoff:.2f}",
                status_color=settings.tokens.PRIMARY_RED,
                badge_text="ACTIVE ML"
            ),
            unsafe_allow_html=True
        )

# Secondary KPI Row: Model Performance & Investigation Desk
st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
kpi2_c1, kpi2_c2, kpi2_c3, kpi2_c4 = st.columns(4)

with kpi2_c1:
    st.markdown(
        render_metric_card(
            title="Active Model ID",
            value=active_model.model_id if active_model else "Heuristic Baseline",
            subtitle=f"Threshold Cutoff: {model_cutoff:.2f}",
            status_color=settings.tokens.PRIMARY_RED,
            badge_text="DEPLOYED"
        ),
        unsafe_allow_html=True
    )

with kpi2_c2:
    crit_count = int((df["Severity"] == "CRITICAL").sum()) if "Severity" in df.columns else 0
    st.markdown(
        render_metric_card(
            title="Critical Risk Tier",
            value=f"{crit_count:,}",
            subtitle="Score >= 0.85 (Immediate hold)",
            status_color=settings.tokens.CRITICAL_RED,
            badge_text="CRITICAL"
        ),
        unsafe_allow_html=True
    )

with kpi2_c3:
    st.markdown(
        render_metric_card(
            title="Open Cases (SQLite)",
            value=f"{case_metrics['open_cases']:,}",
            subtitle=f"{case_metrics['in_investigation']} in active investigation",
            status_color=settings.tokens.WARNING_AMBER,
            badge_text="CASE DESK"
        ),
        unsafe_allow_html=True
    )

with kpi2_c4:
    st.markdown(
        render_metric_card(
            title="Escalated AML Cases",
            value=f"{case_metrics['escalated_aml']:,}",
            subtitle=f"Total cases: {case_metrics['total_cases']}",
            status_color=settings.tokens.PRIMARY_RED,
            badge_text="AML UNIT"
        ),
        unsafe_allow_html=True
    )

st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# INTERACTIVE CHARTS
# -------------------------------------------------------------
st.markdown("### Surveillance Visualizations")
tab_v1, tab_v2 = st.tabs(["Anomaly Velocity & Distribution", "Monetary Exposure & Hourly Velocity"])

with tab_v1:
    ch_col1, ch_col2 = st.columns([1.5, 1])
    with ch_col1:
        # Scatter timeline of transactions
        sample_size = min(1500, len(df))
        chart_sample = df.sample(sample_size, random_state=42) if len(df) > sample_size else df

        fig_scatter = px.scatter(
            chart_sample,
            x=settings.TIME_COL,
            y=settings.AMOUNT_COL,
            color="Prediction",
            color_discrete_map={"ANOMALY": "#E5484D", "NORMAL": "#35C98A"},
            hover_data=["Transaction_ID", "Risk_Score", "Severity"],
            title=f"Transaction Flow & Flagged Anomalies (Sample N={sample_size:,})",
            labels={settings.TIME_COL: "Elapsed Time (Seconds)", settings.AMOUNT_COL: "Ticket Amount ($)"}
        )
        fig_scatter.update_layout(
            paper_bgcolor="#0B0D10",
            plot_bgcolor="#15191F",
            font=dict(color="#F4F6F8", family="Inter"),
            margin=dict(l=40, r=20, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_scatter.update_xaxes(gridcolor="#292F39")
        fig_scatter.update_yaxes(gridcolor="#292F39")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with ch_col2:
        # Anomaly score distribution chart
        st.plotly_chart(
            create_anomaly_score_distribution_chart(df["Risk_Score"].values, threshold=model_cutoff),
            use_container_width=True
        )

with tab_v2:
    ch2_col1, ch2_col2 = st.columns([1.2, 1])
    with ch2_col1:
        st.plotly_chart(
            create_amount_distribution_chart(df),
            use_container_width=True
        )
    with ch2_col2:
        st.plotly_chart(
            create_hourly_volume_chart(calculate_hourly_velocity(df)),
            use_container_width=True
        )

# -------------------------------------------------------------
# RECENT ALERTS QUEUE & INVESTIGATION STATUS
# -------------------------------------------------------------
st.markdown("---")
st.markdown("### Operational Alert Queue (Priority Stream)")
alerts = generate_alerts_from_dataset(df, model=active_model, threshold=model_cutoff, max_alerts=15)

if alerts:
    al_df = pd.DataFrame([{
        "Alert ID": a["alert_id"],
        "Transaction ID": a["transaction_id"],
        "Ticket Amount": a["amount"],
        "Anomaly Score": a["anomaly_score"],
        "Severity": a["severity"],
        "Investigation Status": a["investigation_status"],
        "Model ID": a["model_id"]
    } for a in alerts])

    st.dataframe(
        al_df.style.format({
            "Ticket Amount": "${:,.2f}",
            "Anomaly Score": "{:.4f}"
        }),
        use_container_width=True,
        height=260
    )
else:
    st.info("No active alerts currently exceed the operating threshold.")

# Navigation Quick Links
st.markdown(
    """
    <div style="background: var(--bg-card); padding: 1rem; border-radius: 6px; border: 1px solid var(--border-color); margin-top: 1rem;">
        <div style="font-weight: 600; color: var(--primary-text); margin-bottom: 0.5rem;">Forensic Action Next Steps:</div>
        <div style="color: var(--secondary-text); font-size: 0.85rem; line-height: 1.6;">
            Navigate to <strong>Transaction Monitoring</strong> to filter individual transactions &bull;
            Inspect <strong>Model Performance</strong> to evaluate Precision/Recall metrics &bull;
            Consult <strong>AI Investigation Assistant</strong> for mathematical feature attribution &bull;
            Manage active cases in <strong>Case Management</strong>.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
