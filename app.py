"""FinGuard AI - Intelligent Transaction Risk Monitoring.

Main application entry point with enterprise multi-page navigation,
persistent sidebar telemetry, and zero-friction initialization.
"""

import sys
from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import settings
from components.theme import configure_page
from src.data.loader import check_benchmark_dataset_exists, load_csv_dataset
from src.data.demo_generator import generate_synthetic_demo_data
from src.models.registry import get_active_model

# Global session state initialization
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "data_source_type" not in st.session_state:
    st.session_state.data_source_type = None

# Auto-detect benchmark dataset on disk, or fall back to verified demo data
if st.session_state.dataset is None:
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
        try:
            demo_df = generate_synthetic_demo_data(
                n_samples=settings.DEMO_SAMPLES_COUNT,
                fraud_ratio=settings.DEMO_ANOMALY_RATIO,
                save_to_file=True
            )
            st.session_state.dataset = demo_df
            st.session_state.data_source_type = "DEMO (SYNTHETIC)"
        except Exception:
            pass

# Configure 10-Page Production Navigation (Sections 6 & 10)
pages = [
    st.Page("pages/1_Executive_Overview.py", title="Executive Overview", default=True),
    st.Page("pages/2_Transaction_Explorer.py", title="Transaction Explorer"),
    st.Page("pages/3_Anomaly_Detection.py", title="Anomaly Detection"),
    st.Page("pages/4_Model_Lab.py", title="Model Lab"),
    st.Page("pages/5_Alert_Center.py", title="Alert Center"),
    st.Page("pages/6_Investigation_Cases.py", title="Investigation Cases"),
    st.Page("pages/7_Analytics.py", title="Forensic Analytics"),
    st.Page("pages/8_Data_Management.py", title="Data Management"),
    st.Page("pages/9_Live_Demo.py", title="Live Monitoring Demo"),
    st.Page("pages/10_Settings.py", title="Settings & Health"),
]

# Persistent Sidebar Telemetry
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.5rem 0; margin-bottom: 0.75rem;">
            <div style="font-size: 1.25rem; font-weight: 700; color: #F3F4F6; letter-spacing: -0.02em;">
                🛡️ FinGuard AI
            </div>
            <div style="font-size: 0.78rem; color: #00D4FF; font-weight: 500; margin-top: 0.15rem;">
                Intelligent Transaction Risk Monitoring
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    active_m = get_active_model()
    m_name = active_m.model_name if (active_m and active_m.is_trained) else "Isolation Forest (Default)"
    m_cutoff = active_m.threshold if (active_m and active_m.is_trained) else 0.65
    data_src = st.session_state.get("data_source_type", "DEMO (SYNTHETIC)")

    st.markdown(
        f"""
        <div style="background: #131B2E; padding: 0.85rem; border-radius: 8px; border: 1px solid #1E2C4A; font-size: 0.78rem; line-height: 1.6; margin-bottom: 1rem;">
            <div><strong>Active Engine:</strong> <span style="color: #00D4FF; font-weight: 600;">{m_name}</span></div>
            <div style="margin-top: 0.25rem;"><strong>Decision Cutoff:</strong> <code>{m_cutoff:.2f}</code></div>
            <div style="margin-top: 0.25rem;"><strong>Data Source:</strong> <span style="color: #FBBF24;">{data_src}</span></div>
            <div style="margin-top: 0.25rem;"><strong>Inference Mode:</strong> Continuous Scoring</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Mandatory Legal Decision-Support Notice
    st.markdown(
        f"""
        <div style="font-size: 0.72rem; color: #64748B; background: rgba(11, 15, 25, 0.7); border: 1px solid #1E2C4A; border-radius: 6px; padding: 0.65rem 0.8rem; line-height: 1.5;">
            <strong>Decision-Support Notice:</strong><br>
            {settings.DISCLAIMER_NOTICE}
        </div>
        """,
        unsafe_allow_html=True
    )

pg = st.navigation(pages)
pg.run()
