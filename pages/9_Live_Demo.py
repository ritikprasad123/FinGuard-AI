"""FinGuard AI - Live-Style Simulated Stream Monitor.

Transaction-by-transaction model inference stream emulation,
rolling anomaly score velocity, alert generation, and non-blocking execution.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import settings
from components.theme import configure_page
from components.cards import render_header, render_banner, render_metric_card
from src.monitoring.stream_monitor import SimulatedStreamMonitor
from src.models.registry import get_active_model
from src.data.loader import check_benchmark_dataset_exists, load_csv_dataset
from src.data.demo_generator import generate_synthetic_demo_data

configure_page(page_title="Live Monitoring Demo")

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
    title="Live Monitoring Demo & Stream Surveillance",
    subtitle="Transaction-by-transaction model inference, live anomaly velocity tracking, and alert triggers.",
    badge_text="STREAM SURVEILLANCE",
    badge_type="high"
)

render_banner(
    title="DEMO DATA (SIMULATED STREAM)",
    message=(
        "This feed is a synthetic simulation engineered for local compliance and surveillance verification. "
        "It is not connected to live banking clearing houses, SWIFT, Fedwire, or card network gateways."
    ),
    level="demo",
    badge="SIMULATED STREAM"
)

if "dataset" not in st.session_state or st.session_state.dataset is None:
    if settings.DEFAULT_DEMO_PATH.is_file():
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
model_cutoff = active_model.threshold if (active_model and active_model.is_trained) else 0.65

# Initialize or retrieve Stream Monitor in session state
if "stream_monitor" not in st.session_state or st.session_state.stream_monitor is None:
    st.session_state.stream_monitor = SimulatedStreamMonitor(
        data_source=df, model=active_model, threshold=model_cutoff
    )

stream_mon: SimulatedStreamMonitor = st.session_state.stream_monitor
stream_mon.set_threshold(model_cutoff)

# Stream Controls Bar
st.markdown("### Stream Controls")
c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("Step 1 Transaction", use_container_width=True):
        stream_mon.step(1)
        st.rerun()
with c2:
    if st.button("Step 10 Transactions", use_container_width=True):
        stream_mon.step(10)
        st.rerun()
with c3:
    if st.button("Step 50 Transactions", type="primary", use_container_width=True):
        stream_mon.step(50)
        st.rerun()
with c4:
    if st.button("Reset Stream to Beginning", use_container_width=True):
        stream_mon.reset()
        st.success("Stream reset to index 0.")
        st.rerun()

# Stream Metrics
metrics = stream_mon.get_metrics()
st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Transactions Ingested", f"{metrics['total_streamed']:,}")
with m2:
    st.metric("Anomalies Flagged", f"{metrics['total_alerts']:,}")
with m3:
    st.metric("Stream Alert Rate", f"{metrics['alert_rate']:.2%}")
with m4:
    st.metric("Buffer Progress", f"{metrics['progress_pct']:.1f}%")

# Stream Graph & Table
recent_df = stream_mon.get_stream_dataframe(max_recent=100)

if not recent_df.empty:
    st.markdown("### Live Anomaly Score Velocity")
    fig_strm = go.Figure()

    fig_strm.add_trace(go.Scatter(
        x=recent_df["stream_index"],
        y=recent_df["anomaly_score"],
        mode="lines+markers",
        name="Score",
        line=dict(color="#3B82F6", width=1.5),
        marker=dict(
            size=6,
            color=np.where(recent_df["anomaly_score"] >= model_cutoff, "#E5484D", "#35C98A")
        ),
        hovertemplate="Index: %{x}<br>Score: %{y:.4f}<extra></extra>"
    ))

    # Cutoff Line
    fig_strm.add_shape(
        type="line",
        x0=recent_df["stream_index"].min(),
        x1=recent_df["stream_index"].max(),
        y0=model_cutoff,
        y1=model_cutoff,
        line=dict(color="#E5484D", width=2, dash="dash")
    )

    fig_strm.update_layout(
        paper_bgcolor="#0B0D10",
        plot_bgcolor="#15191F",
        font=dict(color="#F4F6F8", family="Inter"),
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(title="Stream Sequence Index", gridcolor="#292F39"),
        yaxis=dict(title="Calibrated Score", range=[0.0, 1.05], gridcolor="#292F39"),
        height=280,
        showlegend=False
    )
    st.plotly_chart(fig_strm, use_container_width=True)

    st.markdown("### Recent Streamed Transactions (Tail 15)")
    st.dataframe(
        recent_df[["stream_index", "transaction_id", "amount", "anomaly_score", "prediction", "risk_tier"]].tail(15).style.format({
            "amount": "${:,.2f}",
            "anomaly_score": "{:.4f}"
        }),
        use_container_width=True
    )
else:
    st.info("Click 'Step 10' or 'Step 50' above to stream transactions and observe per-record model inference.")


# Legal & Regulatory Decision-Support Notice
st.markdown(
    f"<div class='fg-legal-disclaimer'><strong>Regulatory Compliance Notice:</strong> {settings.DISCLAIMER_NOTICE}</div>",
    unsafe_allow_html=True
)
