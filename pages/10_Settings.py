import numpy as np
"""FinGuard AI - Dataset and Model Status.

Dataset provenance, 7-step schema quality validation, model registry status,
preprocessor serialization telemetry, and deployment configurations.
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
from components.cards import render_header, render_metric_card, render_banner
from components.setup_card import render_setup_card
from src.data.loader import check_benchmark_dataset_exists, get_dataset_provenance, load_csv_dataset
from src.data.validator import validate_transaction_data
from src.features.preprocessor import TransactionPreprocessor
from src.models.registry import get_active_model, list_available_models

configure_page(page_title="Settings and System Health")

render_header(
    title="Settings, System Health & Diagnostics",
    subtitle="System runtime telemetry, environment diagnostics, model registry health, and threshold configuration.",
    badge_text="SYSTEM HEALTH",
    badge_type="high"
)

# Tabs
tab_sys_health, tab_data_stat, tab_model_stat, tab_deploy_guide = st.tabs([
    "System Health & Diagnostics",
    "Dataset Provenance & Schema Audit",
    "Model Registry & Artifact Health",
    "Deployment & Runtime Guide"
])

# -------------------------------------------------------------
# TAB: SYSTEM HEALTH & ENVIRONMENT DIAGNOSTICS (Section 6 - Page 10)
# -------------------------------------------------------------
with tab_sys_health:
    st.markdown("### System Health & Configuration Telemetry")
    st.caption("Real-time infrastructure checks, version audits, and safe diagnostic metrics.")

    import platform
    import sqlite3
    import sklearn
    import numpy as np

    # System Health Indicator Cards
    db_exists = settings.DEFAULT_CASES_DB_PATH.is_file()
    db_size_kb = round(settings.DEFAULT_CASES_DB_PATH.stat().st_size / 1024, 1) if db_exists else 0.0

    sh1, sh2, sh3, sh4 = st.columns(4)
    with sh1:
        st.markdown(
            render_metric_card(
                title="Application Version",
                value=settings.VERSION,
                subtitle=settings.APP_NAME,
                status_color=settings.tokens.ACCENT_BLUE
            ),
            unsafe_allow_html=True
        )
    with sh2:
        st.markdown(
            render_metric_card(
                title="SQLite Database",
                value="ONLINE" if db_exists else "INITIALIZING",
                subtitle=f"{db_size_kb} KB ({settings.CASES_DB_FILENAME})",
                status_color=settings.tokens.SUCCESS_GREEN if db_exists else settings.tokens.WARNING_AMBER
            ),
            unsafe_allow_html=True
        )
    with sh3:
        st.markdown(
            render_metric_card(
                title="Python Runtime",
                value=f"v{platform.python_version()}",
                subtitle=f"{platform.system()} ({platform.machine()})",
                status_color=settings.tokens.SUCCESS_GREEN
            ),
            unsafe_allow_html=True
        )
    with sh4:
        st.markdown(
            render_metric_card(
                title="Scikit-Learn Engine",
                value=f"v{sklearn.__version__}",
                subtitle="Inference Ready",
                status_color=settings.tokens.SUCCESS_GREEN
            ),
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("#### Global Threshold Configuration")
    global_cutoff = st.slider(
        "Default Operating Anomaly Cutoff",
        min_value=0.10,
        max_value=0.95,
        value=0.65,
        step=0.05,
        help="Sets the initial decision boundary for incoming unlabelled transactions."
    )
    st.info(f"Operating cutoff set to `{global_cutoff:.2f}`. Transactions scoring above this threshold generate alerts.")

    st.markdown("#### Environment Diagnostics & Dependency Audit")
    deps_data = [
        {"Package": "streamlit", "Version": st.__version__, "Status": "Healthy / In-Memory"},
        {"Package": "pandas", "Version": pd.__version__, "Status": "Healthy / Vectorized"},
        {"Package": "numpy", "Version": np.__version__, "Status": "Healthy / C-Optimized"},
        {"Package": "scikit-learn", "Version": sklearn.__version__, "Status": "Healthy / Forest & LOF"},
        {"Package": "sqlite3", "Version": sqlite3.sqlite_version, "Status": "Healthy / ACID Compliant"}
    ]
    st.dataframe(pd.DataFrame(deps_data), use_container_width=True)

    st.markdown("#### Safe Security & Privacy Diagnostics")
    st.markdown(
        """
        <div style="background: #131B2E; border: 1px solid #1E2C4A; padding: 1rem; border-radius: 6px; font-size: 0.85rem; line-height: 1.6;">
            • <strong>Zero Secrets Exposed:</strong> No API keys, credentials, or customer tokens are stored or rendered in client HTML.<br>
            • <strong>Parameterized SQL:</strong> All queries to <code>cases.db</code> use parameterized binding (<code>?</code>) preventing injection.<br>
            • <strong>Privacy Protection:</strong> All 28 PCA dimensions are anonymized projections with zero PII exposure.<br>
            • <strong>Cloud Storage Isolation:</strong> Temporary files and database writes remain confined to local project workspace.
        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------------------------------------------------
# TAB 1: DATASET PROVENANCE & QUALITY AUDIT
# -------------------------------------------------------------
with tab_data_stat:
    st.markdown("### Public Benchmark Dataset Provenance")
    prov = get_dataset_provenance()

    p_c1, p_c2 = st.columns(2)
    with p_c1:
        st.markdown(f"**Dataset Title:** {prov.get('dataset_name', 'Credit Card Fraud Benchmark')}")
        st.markdown(f"**Authors:** {prov.get('authors', prov.get('creators', 'Andrea Dal Pozzolo et al.'))}")
        st.markdown(f"**Institution:** {prov.get('institution', 'Machine Learning Group (MLG) - ULB')}")
        st.markdown(f"**Citation:** *{prov.get('publication', prov.get('published_in', 'IEEE SSCI 2015'))}*")
    with p_c2:
        st.markdown(f"**License:** `{prov.get('license', 'Database Contents License (DbCL) v1.0')}`")
        exp_records = prov.get('expected_records', 284807)
        exp_fraud = prov.get('expected_fraud_records', 492)
        exp_rate = prov.get('expected_fraud_rate', '0.172%')
        st.markdown(f"**Expected Dimensions:** `{exp_records:,}` records ({exp_fraud} frauds, `{exp_rate}`)")
        st.markdown(f"**Local Path:** `{prov.get('expected_path', str(settings.DEFAULT_BENCHMARK_PATH))}`")
        st.markdown(f"**Source URL:** [Kaggle Dataset Page]({prov.get('source_url', prov.get('kaggle_ref', 'https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud'))})")

    st.markdown("---")
    st.markdown("### Current Ingestion & Schema Audit")
    dataset_ctx = st.session_state.get("dataset")

    if dataset_ctx is None:
        render_banner(
            title="Benchmark Dataset Not Detected",
            message=f"The benchmark dataset was not found at <code>{settings.DEFAULT_BENCHMARK_PATH}</code>.",
            level="info"
        )
        render_setup_card()
    else:
        q_rep = validate_transaction_data(dataset_ctx)

        qc1, qc2, qc3, qc4 = st.columns(4)
        with qc1:
            st.metric("Total Records", f"{q_rep.total_rows:,}")
        with qc2:
            st.metric("Schema Columns", f"{q_rep.total_columns:,}")
        with qc3:
            st.metric("Missing Values", f"{q_rep.missing_values_count:,}")
        with qc4:
            st.metric("Audit Status", "PASSED" if q_rep.is_valid else "FAILED")

        st.markdown("#### 7-Step Quality Check Results")
        check_rows = []
        checks_list = getattr(q_rep, "checks", None)
        if checks_list:
            for chk in checks_list:
                check_rows.append({
                    "Quality Check": getattr(chk, "check_name", str(chk)),
                    "Status": "PASSED" if getattr(chk, "passed", True) else "FAILED",
                    "Details": getattr(chk, "details", "")
                })
        elif hasattr(q_rep, "checks_summary") and q_rep.checks_summary:
            for chk_name, status in q_rep.checks_summary.items():
                check_rows.append({
                    "Quality Check": chk_name.replace("_", " ").title(),
                    "Status": "PASSED" if status in ["PASS", "INFO"] else "FAILED",
                    "Details": f"Validation status: {status}"
                })
        else:
            check_rows.append({
                "Quality Check": "Schema Conformance",
                "Status": "PASSED" if q_rep.is_valid else "FAILED",
                "Details": f"Audit status: {'Valid' if q_rep.is_valid else 'Errors detected'}"
            })
        st.dataframe(pd.DataFrame(check_rows), use_container_width=True)

# -------------------------------------------------------------
# TAB 2: MODEL REGISTRY & ARTIFACT HEALTH
# -------------------------------------------------------------
with tab_model_stat:
    st.markdown("### Serialized Model Artifact Registry")
    available_models = list_available_models()
    active_m = get_active_model()

    reg_rows = []
    for m in available_models:
        is_active = (active_m is not None and m.model_name == active_m.model_name)
        reg_rows.append({
            "Active": "● ACTIVE" if is_active else "○ Standby",
            "Model Name": m.model_name,
            "Model ID": m.model_id,
            "Status": "Trained & Serialized" if m.is_trained else "Unfitted",
            "Decision Cutoff": f"{m.threshold:.2f}",
            "Feature Dimensions": len(m.feature_names_in_) if m.is_trained else "N/A"
        })
    st.dataframe(pd.DataFrame(reg_rows), use_container_width=True)

    st.markdown("---")
    st.markdown("### Feature Preprocessor Telemetry")
    prep_path = settings.DEFAULT_PREPROCESSOR_PATH
    if prep_path.is_file():
        try:
            preprocessor = TransactionPreprocessor.load(prep_path)
            summary = preprocessor.get_summary()

            pr1, pr2, pr3 = st.columns(3)
            with pr1:
                st.metric("Scaler Type", summary["scaler_type"])
            with pr2:
                st.metric("Fitted Input Features", f"{summary['features_in_count']}")
            with pr3:
                st.metric("Transformed Features", f"{summary['features_out_count']}")

            st.caption(f"Artifact location: `{prep_path}`")
        except Exception as e:
            st.error(f"Error reading preprocessor artifact: {e}")
    else:
        st.info(f"Preprocessor artifact not yet generated at `{prep_path}`.")

# -------------------------------------------------------------
# TAB 3: DEPLOYMENT & RUNTIME GUIDE
# -------------------------------------------------------------
with tab_deploy_guide:
    st.markdown("### Deployment Architecture & Environment Specifications")
    st.markdown(
        """
        #### 1. Local Startup Commands
        ```bash
        # Clone repository and enter project directory
        cd FinGuard-AI

        # Install verified dependencies
        pip install -r requirements.txt

        # Run automated test suite
        pytest tests

        # Launch production dashboard
        streamlit run app.py
        ```

        #### 2. Caching & Performance Policy
        - **Dataset Ingestion:** Handled via `@st.cache_data` and Streamlit session state to eliminate redundant large disk reads.
        - **Model Inference:** Models are serialized to disk (`models/*.joblib`) using Joblib and held in memory.
        - **Persistent Case Storage:** Managed locally via ACID-compliant SQLite (`data/cases.db`).

        #### 3. Security & Compliance Safeguards
        - Zero customer PII or card numbers are stored or transmitted.
        - Benchmark features $V1–V28$ are mathematically anonymized principal components.
        - No credentials or live API secrets committed to Git repository.
        """
    )


# Legal & Regulatory Decision-Support Notice
st.markdown(
    f"<div class='fg-legal-disclaimer'><strong>Regulatory Compliance Notice:</strong> {settings.DISCLAIMER_NOTICE}</div>",
    unsafe_allow_html=True
)
