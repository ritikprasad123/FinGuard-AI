"""FinGuard AI - Enterprise Real-Time Financial Anomaly Detection.

Main application entry point with enterprise multi-page navigation.
"""

import sys
from pathlib import Path
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import settings
from components.theme import configure_page
from src.data.loader import check_benchmark_dataset_exists, load_csv_dataset
from src.models.registry import get_active_model

# Global session state initialization
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "data_source_type" not in st.session_state:
    st.session_state.data_source_type = None

# Auto-detect benchmark dataset on disk
bench_exists, bench_path = check_benchmark_dataset_exists()
if st.session_state.dataset is None and bench_exists:
    try:
        df_bench, rep = load_csv_dataset(bench_path)
        st.session_state.dataset = df_bench
        st.session_state.data_source_type = "BENCHMARK"
    except Exception:
        pass

# Configure 10-page Enterprise Navigation
pages = [
    st.Page("pages/1_Executive_Overview.py", title="Executive Overview", default=True),
    st.Page("pages/2_Transaction_Monitoring.py", title="Transaction Monitoring"),
    st.Page("pages/3_Anomaly_Detection.py", title="Anomaly Detection Lab"),
    st.Page("pages/4_Model_Performance.py", title="Model Performance"),
    st.Page("pages/5_AI_Investigation_Assistant.py", title="AI Investigation Assistant"),
    st.Page("pages/6_Case_Management.py", title="Case Management"),
    st.Page("pages/7_Live_Style_Monitor.py", title="Live-Style Monitor"),
    st.Page("pages/8_Alert_Center.py", title="Alert Center"),
    st.Page("pages/9_Investigation_Reports.py", title="Investigation Reports"),
    st.Page("pages/10_Dataset_and_Model_Status.py", title="Dataset & Model Health"),
]

# Persistent Sidebar Telemetry
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.5rem 0; margin-bottom: 0.5rem;">
            <div style="font-size: 1.15rem; font-weight: 700; color: #F4F6F8; letter-spacing: -0.02em;">
                🛡️ FinGuard AI
            </div>
            <div style="font-size: 0.75rem; color: #9AA4B2; margin-top: 0.2rem;">
                Enterprise Surveillance & Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    active_m = get_active_model()
    m_name = active_m.model_name if (active_m and active_m.is_trained) else "Heuristic Rule Baseline"
    m_cutoff = active_m.threshold if (active_m and active_m.is_trained) else 0.65

    st.markdown(
        f"""
        <div style="background: #15191F; padding: 0.75rem; border-radius: 6px; border: 1px solid #292F39; font-size: 0.78rem; line-height: 1.6; margin-bottom: 0.75rem;">
            <div><strong>Active Model:</strong> <span style="color: #35C98A; font-weight: 600;">{m_name}</span></div>
            <div style="margin-top: 0.25rem;"><strong>Decision Cutoff:</strong> <code>{m_cutoff:.2f}</code></div>
            <div style="margin-top: 0.25rem;"><strong>Detection Mode:</strong> Continuous Scoring</div>
        </div>
        """,
        unsafe_allow_html=True
    )

pg = st.navigation(pages)
pg.run()
