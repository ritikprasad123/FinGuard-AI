"""FinGuard AI - Forensic Analytics & Deep Intelligence Studio.

Comprehensive statistical surveillance analytics, population distributions,
time-series velocity tracking, latent PCA multidimensional comparisons,
and clear mathematical explanations for compliance officers.
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
from components.cards import render_header, render_metric_card
from components.charts import (
    create_amount_distribution_chart,
    create_anomaly_score_distribution_chart,
    create_hourly_volume_chart,
    create_pca_scatter_chart
)
from src.data.loader import check_benchmark_dataset_exists, load_csv_dataset
from src.data.demo_generator import generate_synthetic_demo_data
from src.models.registry import get_active_model
from src.models.base import classify_risk_tier
from src.investigation.alert_feed import calculate_statistical_risk_score
from src.monitoring.metrics import calculate_hourly_velocity

configure_page(page_title="Forensic Analytics")

# Auto-initialize fallback if dataset is None
if "dataset" not in st.session_state or st.session_state.dataset is None:
    bench_exists, bench_path = check_benchmark_dataset_exists()
    if bench_exists:
        try:
            df_bench, rep = load_csv_dataset(bench_path)
            st.session_state.dataset = df_bench
            st.session_state.data_source_type = "BENCHMARK"
        except Exception:
            pass
    if st.session_state.dataset is None and settings.DEFAULT_DEMO_PATH.is_file():
        try:
            st.session_state.dataset = pd.read_csv(settings.DEFAULT_DEMO_PATH)
            st.session_state.data_source_type = "DEMO (SYNTHETIC)"
        except Exception:
            pass
    if st.session_state.dataset is None:
        demo_df = generate_synthetic_demo_data(n_samples=settings.DEMO_SAMPLES_COUNT, save_to_file=True)
        st.session_state.dataset = demo_df
        st.session_state.data_source_type = "DEMO (SYNTHETIC)"

df = st.session_state.dataset
active_model = get_active_model()
active_threshold = active_model.threshold if (active_model and active_model.is_trained) else 0.65

# Ensure Risk_Score and Prediction exist
if "Risk_Score" not in df.columns or "anomaly_score" not in df.columns:
    if active_model and active_model.is_trained:
        try:
            df["Risk_Score"] = np.round(active_model.predict_score(df), 4)
        except Exception:
            df["Risk_Score"] = np.round(calculate_statistical_risk_score(df), 4)
    else:
        df["Risk_Score"] = np.round(calculate_statistical_risk_score(df), 4)
    df["anomaly_score"] = df["Risk_Score"]
    df["Prediction"] = np.where(df["Risk_Score"] >= active_threshold, "ANOMALY", "NORMAL")
    df["Severity"] = [classify_risk_tier(s) for s in df["Risk_Score"]]
    st.session_state.dataset = df

render_header(
    title="Forensic Analytics & Deep Intelligence",
    subtitle="Statistical population dynamics, multi-dimensional PCA comparisons, velocity timelines, and visual attribution.",
    badge_text="ADVANCED ANALYTICS",
    badge_type="info"
)

# Top KPI Summary Row
flagged_mask = df["Risk_Score"] >= active_threshold
flagged_df = df[flagged_mask]
cleared_df = df[~flagged_mask]

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        render_metric_card(
            title="Total Volume Audited",
            value=f"{len(df):,}",
            subtitle=f"{st.session_state.get('data_source_type', 'ACTIVE')} stream",
            status_color=settings.tokens.ACCENT_BLUE
        ),
        unsafe_allow_html=True
    )
with k2:
    st.markdown(
        render_metric_card(
            title="Flagged Anomalies",
            value=f"{len(flagged_df):,}",
            subtitle=f"{(len(flagged_df)/len(df)):.2%} of population",
            status_color=settings.tokens.CRITICAL_RED
        ),
        unsafe_allow_html=True
    )
with k3:
    med_flagged_amt = flagged_df[settings.AMOUNT_COL].median() if len(flagged_df) > 0 else 0.0
    med_norm_amt = cleared_df[settings.AMOUNT_COL].median() if len(cleared_df) > 0 else 0.0
    st.markdown(
        render_metric_card(
            title="Flagged Median Amount",
            value=f"${med_flagged_amt:,.2f}",
            subtitle=f"Cleared median: ${med_norm_amt:,.2f}",
            status_color=settings.tokens.WARNING_AMBER
        ),
        unsafe_allow_html=True
    )
with k4:
    st.markdown(
        render_metric_card(
            title="Decision Threshold",
            value=f"{active_threshold:.3f}",
            subtitle=f"Active: {active_model.model_name if active_model else 'Baseline'}",
            status_color=settings.tokens.SUCCESS_GREEN
        ),
        unsafe_allow_html=True
    )

st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# TABBED ANALYTICS SECTIONS
# -------------------------------------------------------------
tab_amt, tab_scores, tab_compare, tab_time, tab_features, tab_models = st.tabs([
    "1. Amount Distributions",
    "2. Anomaly Score Dynamics",
    "3. Flagged vs Cleared",
    "4. Time-Series Velocity",
    "5. Latent Feature Drivers",
    "6. Multi-Model Benchmark"
])

# -------------------------------------------------------------
# TAB 1: AMOUNT DISTRIBUTIONS
# -------------------------------------------------------------
with tab_amt:
    st.markdown("### Transaction Amount Exposure Distribution")
    st.caption("Visualizes the monetary spread of cleared vs. flagged transactions across standard and log scales.")

    ch_amt1, ch_amt2 = st.columns(2)
    with ch_amt1:
        st.plotly_chart(create_amount_distribution_chart(df), use_container_width=True)
    with ch_amt2:
        # Boxplot Comparison
        fig_box = px.box(
            df,
            x="Prediction",
            y=settings.AMOUNT_COL,
            color="Prediction",
            color_discrete_map={"ANOMALY": "#F43F5E", "NORMAL": "#10B981"},
            title="Ticket Amount Quartile Comparison",
            log_y=True,
            labels={settings.AMOUNT_COL: "Ticket Amount ($ - Log Scale)", "Prediction": "Surveillance Verdict"}
        )
        fig_box.update_layout(
            paper_bgcolor="#131B2E",
            plot_bgcolor="#131B2E",
            font=dict(color="#F3F4F6", family="Inter"),
            margin=dict(l=40, r=20, t=40, b=40),
            showlegend=False
        )
        fig_box.update_xaxes(gridcolor="#1E2C4A")
        fig_box.update_yaxes(gridcolor="#1E2C4A")
        st.plotly_chart(fig_box, use_container_width=True)

    st.info(
        "💡 **Forensic Chart Explanation:** Financial fraud datasets exhibit extreme bimodal ticket profiles. "
        "Normal retail transactions follow a right-skewed log-normal distribution centered around $50-$150, "
        "while fraudulent card drains cluster at high-value limits ($1,000+) or low micro-testing values ($1-$5)."
    )

# -------------------------------------------------------------
# TAB 2: ANOMALY SCORE DYNAMICS
# -------------------------------------------------------------
with tab_scores:
    st.markdown("### Anomaly Score Population Density")
    st.caption("Continuous mathematical scoring distribution bounded in [0.0, 1.0].")

    st.plotly_chart(create_anomaly_score_distribution_chart(df["Risk_Score"].values, threshold=active_threshold), use_container_width=True)

    st.info(
        f"💡 **Forensic Chart Explanation:** The vertical marker shows the current decision threshold (`{active_threshold:.2f}`). "
        "Transactions to the right of the line are classified as anomalies and routed to the Alert Center for human triage. "
        "Scores are continuous distance metrics rather than calibrated probabilities."
    )

# -------------------------------------------------------------
# TAB 3: FLAGGED VS CLEARED COMPARISONS
# -------------------------------------------------------------
with tab_compare:
    st.markdown("### Comparative Cohort Audit: Flagged vs Cleared")
    st.caption("Side-by-side behavioral metrics separating flagged anomalies from cleared transactions.")

    comp_summary = pd.DataFrame([
        {
            "Metric": "Transaction Count",
            "Cleared Volume": f"{len(cleared_df):,}",
            "Flagged Volume": f"{len(flagged_df):,}",
            "Variance Factor": f"{len(cleared_df)/(len(flagged_df) or 1):.1f}x"
        },
        {
            "Metric": "Mean Ticket Amount",
            "Cleared Volume": f"${cleared_df[settings.AMOUNT_COL].mean():,.2f}",
            "Flagged Volume": f"${flagged_df[settings.AMOUNT_COL].mean():,.2f}",
            "Variance Factor": f"{(flagged_df[settings.AMOUNT_COL].mean()/(cleared_df[settings.AMOUNT_COL].mean() or 1)):.2f}x"
        },
        {
            "Metric": "Max Ticket Amount",
            "Cleared Volume": f"${cleared_df[settings.AMOUNT_COL].max():,.2f}",
            "Flagged Volume": f"${flagged_df[settings.AMOUNT_COL].max():,.2f}",
            "Variance Factor": "N/A"
        },
        {
            "Metric": "Mean Anomaly Score",
            "Cleared Volume": f"{cleared_df['Risk_Score'].mean():.4f}",
            "Flagged Volume": f"{flagged_df['Risk_Score'].mean():.4f}",
            "Variance Factor": f"{(flagged_df['Risk_Score'].mean()/(cleared_df['Risk_Score'].mean() or 1)):.2f}x"
        }
    ])
    st.dataframe(comp_summary, use_container_width=True)

# -------------------------------------------------------------
# TAB 4: TIME-SERIES VELOCITY
# -------------------------------------------------------------
with tab_time:
    st.markdown("### Temporal Velocity & Volume Analysis")
    st.caption("Examines transaction arrival velocity and potential nighttime fraud bursts.")

    if settings.TIME_COL in df.columns:
        hourly_df = calculate_hourly_velocity(df)
        st.plotly_chart(create_hourly_volume_chart(hourly_df), use_container_width=True)

        st.info(
            "💡 **Forensic Chart Explanation:** Time is recorded as elapsed seconds from the start of the observation cycle. "
            "Financial fraud spikes frequently coincide with low-volume off-peak hours (e.g., 2 AM - 5 AM local time), "
            "when automated card-testing bots exploit reduced manual merchant scrutiny."
        )
    else:
        st.warning("Timestamp column not detected in current dataset.")

# -------------------------------------------------------------
# TAB 5: LATENT FEATURE DRIVERS
# -------------------------------------------------------------
with tab_features:
    st.markdown("### Latent PCA Dimension Variance")
    st.caption("Identifies which principal component dimensions deviate most sharply between cohorts.")

    pca_cols = [c for c in settings.PCA_FEATURE_COLS if c in df.columns]
    if pca_cols:
        diffs = []
        for c in pca_cols:
            mean_c = cleared_df[c].mean() if len(cleared_df) > 0 else 0.0
            mean_f = flagged_df[c].mean() if len(flagged_df) > 0 else 0.0
            diffs.append({"Feature": c, "Cleared_Mean": round(mean_c, 3), "Flagged_Mean": round(mean_f, 3), "Absolute_Delta": round(abs(mean_f - mean_c), 3)})

        diff_df = pd.DataFrame(diffs).sort_values(by="Absolute_Delta", ascending=False).head(10)

        fig_bar = px.bar(
            diff_df,
            x="Feature",
            y="Absolute_Delta",
            title="Top 10 Latent Features Driving Anomaly Separation (|Flagged - Cleared|)",
            color="Absolute_Delta",
            color_continuous_scale="Tealgrn"
        )
        fig_bar.update_layout(
            paper_bgcolor="#131B2E",
            plot_bgcolor="#131B2E",
            font=dict(color="#F3F4F6", family="Inter"),
            margin=dict(l=40, r=20, t=40, b=40)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.info(
            "💡 **Forensic Chart Explanation:** Principal components V14, V10, V12, and V17 typically exhibit the greatest mathematical "
            "displacement in credit card fraud detection. These dimensions represent orthogonal combinations of cardholder behavior, "
            "geographic dispersion, and merchant risk attributes."
        )

# -------------------------------------------------------------
# TAB 6: MULTI-MODEL BENCHMARK
# -------------------------------------------------------------
with tab_models:
    st.markdown("### Multi-Model Architecture Radar & Benchmark")
    st.caption("Comparison of Isolation Forest, Local Outlier Factor, and Neural Autoencoder paradigms.")

    model_comp_data = pd.DataFrame([
        {"Architecture": "Isolation Forest", "Algorithmic Basis": "Recursive Random Partition Trees", "Inference Speed": "Sub-millisecond (Fast)", "Memory Footprint": "Low (~1 MB)", "Streaming Compatibility": "High", "Status": "Active Primary"},
        {"Architecture": "Local Outlier Factor", "Algorithmic Basis": "k-Nearest Neighbor Local Density", "Inference Speed": "Bounded Sampling (~10ms)", "Memory Footprint": "Moderate (O(k·N))", "Streaming Compatibility": "Moderate (novelty=True)", "Status": "Supported Alternative"},
        {"Architecture": "Neural Autoencoder", "Algorithmic Basis": "Bottleneck Reconstruction Error (MSE)", "Inference Speed": "Fast (~2ms)", "Memory Footprint": "Low (~40 KB weights)", "Streaming Compatibility": "High", "Status": "Supported Alternative"}
    ])
    st.dataframe(model_comp_data, use_container_width=True)

# Legal & Regulatory Decision-Support Notice
st.markdown(
    f"<div class='fg-legal-disclaimer'><strong>Regulatory Compliance Notice:</strong> {settings.DISCLAIMER_NOTICE}</div>",
    unsafe_allow_html=True
)
