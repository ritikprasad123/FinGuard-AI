"""FinGuard AI - AI Investigation Assistant.

Evidence-grounded forensic reasoning engine for transaction anomalies,
latent feature attribution, and compliance investigation protocols.
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
from components.cards import render_header, render_metric_card
from src.investigation.assistant import InvestigationAssistant
from src.models.registry import get_active_model
from src.data.loader import check_benchmark_dataset_exists, load_csv_dataset
from src.data.demo_generator import generate_synthetic_demo_data

configure_page(page_title="AI Investigation Assistant")

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
    title="AI Forensic Investigation Assistant",
    subtitle="Evidence-grounded mathematical rationale, latent PCA attribution, and actionable investigative protocols.",
    badge_text="AI ASSISTANT",
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
model_cutoff = active_model.threshold if (active_model and active_model.is_trained) else 0.65
model_id_str = active_model.model_id if active_model else "SurveillanceEngine"

# Transaction Selection
avail_tx_ids = df["Transaction_ID"].head(300).tolist() if "Transaction_ID" in df.columns else [f"TXN-{i+1}" for i in range(100)]
default_tx = st.session_state.get("selected_tx_for_ai", {}).get("Transaction_ID", avail_tx_ids[0])

st.markdown("### Transaction Target Selection")
ai_col1, ai_col2 = st.columns([2, 1])
with ai_col1:
    chosen_tx_id = st.selectbox(
        "Select Transaction for Forensic Synthesis",
        options=avail_tx_ids,
        index=avail_tx_ids.index(default_tx) if default_tx in avail_tx_ids else 0
    )
with ai_col2:
    st.markdown("<div style='height: 1.8rem;'></div>", unsafe_allow_html=True)
    reanalyze = st.button("Re-Synthesize Evidence Briefing", type="primary")

target_row = df[df["Transaction_ID"] == chosen_tx_id].iloc[0]
score_val = float(target_row.get("Risk_Score", target_row.get("anomaly_score", 0.68)))

# Execute AI Analysis
briefing = InvestigationAssistant.analyze_transaction(
    transaction_data=target_row,
    anomaly_score=score_val,
    threshold=model_cutoff,
    model_id=model_id_str
)

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

# Executive KPI Row
k_c1, k_c2, k_c3, k_c4 = st.columns(4)
with k_c1:
    st.metric("Anomaly Score", f"{briefing['anomaly_score']:.4f}")
with k_c2:
    st.metric("Decision Cutoff", f"{briefing['threshold']:.4f}")
with k_c3:
    st.metric("Risk Category", briefing["risk_tier"])
with k_c4:
    st.metric("Ticket Amount", f"${briefing['amount']:,.2f}")

# 1. Mathematical Rationale
st.markdown("### 1. Mathematical Anomaly Rationale")
st.info(briefing["flag_reason"])

# 2. Key Latent Feature Deviations Table
st.markdown("### 2. Key Latent Feature Deviations (vs Population Baseline)")
dev_df = pd.DataFrame(briefing["top_deviating_features"])
st.dataframe(
    dev_df[["feature", "value", "z_score", "description"]].rename(columns={
        "feature": "Principal Component",
        "value": "Observed Value",
        "z_score": "Standardized Deviation (Z-Score)",
        "description": "Diagnostic Factor"
    }),
    use_container_width=True
)

# 3. Recommended Forensic Protocol
st.markdown("### 3. Recommended Forensic Investigation Steps")
for idx, step in enumerate(briefing["suggested_steps"], 1):
    st.markdown(f"**Step {idx}:** {step}")

# 4. Mandatory Limitations Statement
st.markdown("---")
st.markdown("### 4. Regulatory & Evidentiary Limitations")
st.warning(f"⚠️ {briefing['limitations']}")

# Download Markdown Briefing
st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
markdown_briefing = InvestigationAssistant.generate_briefing_markdown(briefing)
st.download_button(
    label=f"Download AI Forensic Briefing ({chosen_tx_id}.md)",
    data=markdown_briefing,
    file_name=f"{chosen_tx_id}_forensic_briefing.md",
    mime="text/markdown"
)
