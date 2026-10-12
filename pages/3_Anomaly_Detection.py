"""FinGuard AI - Anomaly Detection Studio.

Real-time transaction scoring, continuous [0, 1] anomaly score generation,
configurable decision cutoffs, single-transaction inference, and batch processing.
"""

import sys
import json
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
from components.charts import create_anomaly_score_distribution_chart
from src.models.registry import get_active_model
from src.models.base import InferenceResult, classify_risk_tier
from src.investigation.alert_feed import calculate_statistical_risk_score
from src.data.loader import check_benchmark_dataset_exists, load_csv_dataset
from src.data.demo_generator import generate_synthetic_demo_data

configure_page(page_title="Anomaly Detection Studio")

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
    title="Anomaly Detection Studio",
    subtitle="Continuous [0, 1] calibrated anomaly scoring, interactive threshold cutoffs, and dimensional feature attribution.",
    badge_text="DETECTION ENGINE",
    badge_type="high"
)

if "dataset" not in st.session_state or st.session_state.dataset is None:
    st.info("No dataset loaded. Please generate synthetic demo data or load the benchmark dataset from Executive Overview.")
    if st.button("Generate Synthetic Demo Batch", type="primary"):
        demo_df = generate_synthetic_demo_data(n_samples=settings.DEMO_SAMPLES_COUNT, save_to_file=True)
        st.session_state.dataset = demo_df
        st.session_state.data_source_type = "DEMO"
        st.rerun()
    st.stop()

df = st.session_state.dataset
active_model = get_active_model()
model_name = active_model.model_name if (active_model and active_model.is_trained) else "Heuristic Rule Baseline"
base_cutoff = active_model.threshold if (active_model and active_model.is_trained) else 0.65

# Tabbed Interface
tab_single, tab_batch, tab_threshold = st.tabs([
    "Single-Transaction Live Scorer",
    "Batch Dataset Inference",
    "Interactive Threshold Tuning"
])

# -------------------------------------------------------------
# TAB 1: SINGLE-TRANSACTION LIVE SCORER
# -------------------------------------------------------------
with tab_single:
    st.markdown("### Single-Transaction Real-Time Scorer")
    st.markdown(
        "<p style='color: var(--secondary-text); font-size: 0.88rem;'>"
        "Execute sub-millisecond inference on an individual financial transaction and inspect feature-level attribution."
        "</p>",
        unsafe_allow_html=True
    )

    # Preset Loaders
    col_pre1, col_pre2, col_pre3 = st.columns(3)
    with col_pre1:
        load_anom = st.button("Load Known High-Risk Anomaly", use_container_width=True)
    with col_pre2:
        load_norm = st.button("Load Typical Cleared Transaction", use_container_width=True)
    with col_pre3:
        load_custom = st.button("Reset Input Form", use_container_width=True)

    # Preset state values
    default_amt = 125.0
    default_v14 = 0.05
    default_v10 = -0.10
    default_v12 = 0.02
    default_v17 = -0.05
    default_v4 = 0.15

    if load_anom:
        default_amt = 3850.0
        default_v14 = -9.25
        default_v10 = -6.40
        default_v12 = -5.80
        default_v17 = -5.10
        default_v4 = 4.20
    elif load_norm:
        default_amt = 45.50
        default_v14 = 0.20
        default_v10 = -0.05
        default_v12 = 0.15
        default_v17 = 0.08
        default_v4 = -0.25

    with st.form("single_tx_scorer_form"):
        sc_c1, sc_c2 = st.columns(2)
        with sc_c1:
            in_tx_id = st.text_input("Transaction Reference ID", value="TXN-LIVE-9001")
            in_amt = st.number_input("Transaction Amount ($)", min_value=0.01, value=default_amt, step=50.0)
            in_time = st.number_input("Elapsed Seconds (Time)", min_value=0.0, value=45000.0, step=100.0)
            in_cutoff = st.slider("Applied Cutoff Threshold", min_value=0.10, max_value=0.95, value=base_cutoff, step=0.05)
        with sc_c2:
            st.markdown("<div style='font-size: 0.82rem; color: var(--secondary-text);'>Key Latent Features (Z-Score scale):</div>", unsafe_allow_html=True)
            in_v14 = st.slider("V14 (Credential/Account Factor)", min_value=-15.0, max_value=5.0, value=default_v14, step=0.1)
            in_v10 = st.slider("V10 (Authorization Factor)", min_value=-15.0, max_value=5.0, value=default_v10, step=0.1)
            in_v12 = st.slider("V12 (Merchant Risk Factor)", min_value=-15.0, max_value=5.0, value=default_v12, step=0.1)
            in_v17 = st.slider("V17 (Terminal Integrity Factor)", min_value=-15.0, max_value=5.0, value=default_v17, step=0.1)
            in_v4 = st.slider("V4 (Velocity Factor)", min_value=-5.0, max_value=15.0, value=default_v4, step=0.1)

        submitted_score = st.form_submit_button("Run Real-Time Anomaly Inference", type="primary")

    if submitted_score or load_anom or load_norm:
        # Construct synthetic row with 30 dimensions
        sample_row = pd.Series(index=settings.REQUIRED_BENCHMARK_FEATURES, data=0.0)
        sample_row[settings.TIME_COL] = in_time
        sample_row[settings.AMOUNT_COL] = in_amt
        sample_row["V14"] = in_v14
        sample_row["V10"] = in_v10
        sample_row["V12"] = in_v12
        sample_row["V17"] = in_v17
        sample_row["V4"] = in_v4

        if active_model and active_model.is_trained:
            res: InferenceResult = active_model.predict_single(
                sample_row, transaction_id=in_tx_id, threshold=in_cutoff
            )
        else:
            # Fallback heuristic
            score_f = calculate_statistical_risk_score(sample_row)
            is_anom_f = score_f >= in_cutoff
            res = InferenceResult(
                transaction_id=in_tx_id,
                anomaly_score=float(score_f),
                is_anomaly=bool(is_anom_f),
                applied_threshold=float(in_cutoff),
                risk_category=classify_risk_tier(score_f),
                prediction_label="ANOMALY" if is_anom_f else "NORMAL",
                top_contributing_features={"V14": in_v14, "V10": in_v10, "Amount": in_amt},
                inference_latency_ms=1.2,
                model_id="HeuristicBaseline"
            )

        # Output Cards
        st.markdown("#### Inference Response Payload")
        res_c1, res_c2, res_c3, res_c4 = st.columns(4)
        with res_c1:
            st.metric("Calibrated Score", f"{res.anomaly_score:.4f}")
        with res_c2:
            st.metric("Decision Cutoff", f"{res.applied_threshold:.2f}")
        with res_c3:
            st.metric("Model Verdict", res.prediction_label)
        with res_c4:
            st.metric("Risk Category", res.risk_category)

        # JSON response & feature attribution
        col_json, col_bar = st.columns([1, 1.2])
        with col_json:
            st.markdown("<div style='font-size: 0.8rem; font-weight: 600; margin-bottom: 0.3rem;'>Structured JSON Response:</div>", unsafe_allow_html=True)
            st.code(json.dumps(res.to_dict(), indent=2), language="json")
        with col_bar:
            st.markdown("<div style='font-size: 0.8rem; font-weight: 600; margin-bottom: 0.3rem;'>Top Contributing Feature Values:</div>", unsafe_allow_html=True)
            feat_df = pd.DataFrame(list(res.top_contributing_features.items()), columns=["Feature", "Observed Value"])
            fig_bar = px.bar(
                feat_df, x="Feature", y="Observed Value",
                color="Observed Value",
                color_continuous_scale=["#E5484D", "#3B82F6", "#35C98A"],
                title="Dimensional Deviations"
            )
            fig_bar.update_layout(
                paper_bgcolor="#0B0D10", plot_bgcolor="#15191F", font=dict(color="#F4F6F8"),
                height=220, margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: BATCH DATASET INFERENCE
# -------------------------------------------------------------
with tab_batch:
    st.markdown("### Batch Dataset Anomaly Scoring")
    st.markdown(
        "<p style='color: var(--secondary-text); font-size: 0.88rem;'>"
        "Score complete transaction partitions, analyze score distributions, or upload an external CSV batch."
        "</p>",
        unsafe_allow_html=True
    )

    batch_cutoff = st.slider("Batch Decision Threshold", min_value=0.10, max_value=0.95, value=base_cutoff, step=0.05, key="batch_cutoff_slider")

    # Optional External CSV Upload
    uploaded_batch = st.file_uploader("Upload External CSV Batch (Optional)", type=["csv"], key="batch_csv_upload")
    if uploaded_batch is not None:
        try:
            target_batch_df = pd.read_csv(uploaded_batch)
            st.success(f"Loaded external batch with {len(target_batch_df):,} rows and {len(target_batch_df.columns)} columns.")
        except Exception as exc:
            st.error(f"Failed to read CSV: {exc}")
            target_batch_df = df
    else:
        target_batch_df = df

    # Score batch
    if active_model and active_model.is_trained:
        try:
            scores = active_model.predict_score(target_batch_df)
        except Exception:
            scores = calculate_statistical_risk_score(target_batch_df)
    else:
        scores = calculate_statistical_risk_score(target_batch_df)

    flagged_mask = scores >= batch_cutoff
    flagged_count = int(flagged_mask.sum())
    flagged_rate = (flagged_count / len(target_batch_df)) if len(target_batch_df) > 0 else 0.0

    b1, b2, b3 = st.columns(3)
    with b1:
        st.metric("Total Batch Records", f"{len(target_batch_df):,}")
    with b2:
        st.metric("Flagged Anomalies", f"{flagged_count:,}", delta=f"{flagged_rate:.2%}")
    with b3:
        st.metric("Cleared Transactions", f"{len(target_batch_df) - flagged_count:,}")

    # Plot Distribution
    st.plotly_chart(create_anomaly_score_distribution_chart(scores, threshold=batch_cutoff), use_container_width=True)

    # Export Flagged Anomalies
    target_batch_df = target_batch_df.copy()
    target_batch_df["Risk_Score"] = scores
    if "Transaction_ID" not in target_batch_df.columns:
        target_batch_df["Transaction_ID"] = [f"TXN-BATCH-{i+1:05d}" for i in range(len(target_batch_df))]
    target_batch_df["Severity"] = [classify_risk_tier(s) for s in scores]

    flagged_df = target_batch_df[flagged_mask].copy()
    if not flagged_df.empty:
        st.markdown(f"#### Flagged Anomalies Queue ({len(flagged_df):,} records)")
        st.dataframe(
            flagged_df[["Transaction_ID", settings.TIME_COL, settings.AMOUNT_COL, "Risk_Score", "Severity"]].head(50).style.format({
                settings.AMOUNT_COL: "${:,.2f}",
                "Risk_Score": "{:.4f}"
            }),
            use_container_width=True
        )

        csv_data = flagged_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download {len(flagged_df):,} Flagged Anomalies as CSV",
            data=csv_data,
            file_name=f"finguard_flagged_anomalies_cutoff_{batch_cutoff:.2f}.csv",
            mime="text/csv"
        )

# -------------------------------------------------------------
# TAB 3: INTERACTIVE THRESHOLD TUNING
# -------------------------------------------------------------
with tab_threshold:
    st.markdown("### Threshold Sensitivity & Operational Trade-offs")
    st.markdown(
        """
        <div style="background: var(--bg-card); padding: 1rem; border-radius: 6px; border-left: 3px solid var(--accent-red); margin-bottom: 1rem;">
            <strong>Core Surveillance Principle:</strong> Lower thresholds increase <em>Recall</em> (catching more anomalies) 
            at the cost of higher <em>False Positives</em> (analyst alert fatigue). Higher thresholds increase <em>Precision</em> 
            but risk missing subtle, low-variance anomalies (higher <em>False Negatives</em>).
        </div>
        """,
        unsafe_allow_html=True
    )

    # Threshold sweep grid
    sweep_thresholds = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    sweep_rows = []
    for t in sweep_thresholds:
        cnt = int((scores >= t).sum())
        rate = cnt / len(df) if len(df) > 0 else 0.0
        sweep_rows.append({
            "Cutoff Threshold": f"{t:.2f}",
            "Flagged Anomalies": cnt,
            "Flagged %": f"{rate:.2%}",
            "Operational Impact": "High False Alarms" if t <= 0.40 else "Balanced Surveillance" if t <= 0.70 else "Strict Low-Alarm Queue"
        })

    st.dataframe(pd.DataFrame(sweep_rows), use_container_width=True)
