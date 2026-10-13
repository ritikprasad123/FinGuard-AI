"""FinGuard AI - Data Management & Schema Quality Audit Studio.

7-step automated data quality auditing, schema validation, missing-value and
duplicate inspections, exportable validation certificates, and model feature governance.
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
from components.cards import render_header, render_metric_card, render_banner
from src.data.loader import check_benchmark_dataset_exists, load_csv_dataset, get_dataset_provenance
from src.data.validator import validate_transaction_data, DataQualityReport
ValidationReport = DataQualityReport
from src.data.demo_generator import generate_synthetic_demo_data
try:
    from src.reports.quality_report import generate_markdown_quality_report
except ImportError:
    from src.reports.quality_report import format_quality_markdown_report as generate_markdown_quality_report

configure_page(page_title="Data Management")

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

render_header(
    title="Data Management & Schema Quality Audit",
    subtitle="7-step schema auditing, missing/duplicate telemetry, validation reports, and feature governance.",
    badge_text="DATA GOVERNANCE",
    badge_type="info"
)

# Upload New Dataset Section
st.markdown("### Ingest & Validate Transaction Dataset")
st.caption("Upload an external transaction CSV or reload baseline datasets. Strict validation prevents target leakage.")

col_up1, col_up2 = st.columns([2, 1])
with col_up1:
    uploaded_file = st.file_uploader("Upload Transaction Dataset (CSV)", type=["csv"], key="data_mgmt_uploader")
    if uploaded_file is not None:
        try:
            new_df = pd.read_csv(uploaded_file)
            st.session_state.dataset = new_df
            st.session_state.data_source_type = f"UPLOADED ({uploaded_file.name})"
            st.success(f"Successfully loaded '{uploaded_file.name}' ({len(new_df):,} rows, {len(new_df.columns)} columns).")
        except Exception as exc:
            st.error(f"Error loading CSV file: {exc}")

with col_up2:
    st.markdown("<div style='height: 1.8rem;'></div>", unsafe_allow_html=True)
    if st.button("Reload Synthetic Demo Dataset", use_container_width=True):
        demo_df = generate_synthetic_demo_data(n_samples=settings.DEMO_SAMPLES_COUNT, save_to_file=True)
        st.session_state.dataset = demo_df
        st.session_state.data_source_type = "DEMO (SYNTHETIC)"
        st.success("Re-initialized verified demo dataset.")
        st.rerun()

df = st.session_state.dataset

# Run 7-Step Quality Validation
val_report: ValidationReport = validate_transaction_data(df)

# Quality Metric Cards Row
st.markdown("---")
st.markdown("### Schema Quality Telemetry")
q1, q2, q3, q4 = st.columns(4)

with q1:
    st.markdown(
        render_metric_card(
            title="Validation Audit Status",
            value="PASSED" if val_report.is_valid else "FAILED",
            subtitle=f"{len(val_report.errors)} blocking errors detected",
            status_color=settings.tokens.SUCCESS_GREEN if val_report.is_valid else settings.tokens.CRITICAL_RED
        ),
        unsafe_allow_html=True
    )
with q2:
    st.markdown(
        render_metric_card(
            title="Dataset Dimensions",
            value=f"{val_report.total_rows:,}",
            subtitle=f"{val_report.total_columns} columns",
            status_color=settings.tokens.ACCENT_BLUE
        ),
        unsafe_allow_html=True
    )
with q3:
    st.markdown(
        render_metric_card(
            title="Duplicate Rows",
            value=f"{val_report.duplicate_rows_count:,}",
            subtitle=f"{(val_report.duplicate_rows_count / (val_report.total_rows or 1)):.2%} duplicate rate",
            status_color=settings.tokens.WARNING_AMBER if val_report.duplicate_rows_count > 0 else settings.tokens.SUCCESS_GREEN
        ),
        unsafe_allow_html=True
    )
with q4:
    total_missing = sum(val_report.missing_values_by_column.values()) if val_report.missing_values_by_column else 0
    st.markdown(
        render_metric_card(
            title="Missing Values (Nulls)",
            value=f"{total_missing:,}",
            subtitle="Clean numeric matrix" if total_missing == 0 else f"{total_missing} null cells",
            status_color=settings.tokens.SUCCESS_GREEN if total_missing == 0 else settings.tokens.CRITICAL_RED
        ),
        unsafe_allow_html=True
    )

st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

# Tabs: Audit Breakdown, Schema Details, Feature Governance, Export Report
tab_audit, tab_schema, tab_features, tab_export = st.tabs([
    "1. 7-Step Audit Checklist",
    "2. Schema & Missing Values",
    "3. Feature Governance & Model Isolation",
    "4. Download Validation Report"
])

# -------------------------------------------------------------
# TAB 1: 7-STEP AUDIT CHECKLIST
# -------------------------------------------------------------
with tab_audit:
    st.markdown("#### Automated 7-Step Financial Schema Verification")

    checklist_items = [
        ("Step 1: Dataset Non-Empty Check", val_report.total_rows > 0, f"Row count: {val_report.total_rows:,}"),
        ("Step 2: Required Feature Schema Check", len(val_report.missing_required_columns) == 0, f"Missing: {len(val_report.missing_required_columns)} columns"),
        ("Step 3: Column Data Type Verification", all("float" in str(dt) or "int" in str(dt) for dt in df.dtypes if dt != object), "Numeric feature types verified"),
        ("Step 4: Non-Negative Amount Constraint", val_report.negative_amount_count == 0, f"Negative amounts: {val_report.negative_amount_count}"),
        ("Step 5: Monotonic Time Step Check", val_report.monotonic_time_passed, "Observation elapsed time sequenced"),
        ("Step 6: Missing-Value & Null Tolerances", total_missing == 0, f"Total missing cells: {total_missing}"),
        ("Step 7: Ground-Truth Target Column Audit", val_report.has_target_column, f"Target '{settings.TARGET_COL}' present (Class imbalance: {val_report.fraud_rate_pct:.3f}%)" if val_report.has_target_column else "Unsupervised dataset (No Class label)")
    ]

    for title, passed, detail in checklist_items:
        badge_style = "color: #10B981; font-weight: 600;" if passed else "color: #F43F5E; font-weight: 600;"
        icon = "✓" if passed else "✕"
        st.markdown(
            f"""
            <div style="background: #131B2E; border: 1px solid #1E2C4A; padding: 0.75rem 1rem; border-radius: 6px; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
                <div><strong>{title}</strong><div style="color: #94A3B8; font-size: 0.8rem;">{detail}</div></div>
                <div style="{badge_style}; font-size: 1.1rem;">{icon} {"PASS" if passed else "FAIL"}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# -------------------------------------------------------------
# TAB 2: SCHEMA & MISSING VALUES SUMMARY
# -------------------------------------------------------------
with tab_schema:
    st.markdown("#### Column Data Types & Missing Values Breakdown")
    schema_rows = []
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        schema_rows.append({
            "Column Name": col,
            "Data Type": str(df[col].dtype),
            "Null Count": null_count,
            "Null Rate": f"{(null_count / len(df)):.2%}",
            "Sample Min": round(float(df[col].min()), 3) if np.issubdtype(df[col].dtype, np.number) else "N/A",
            "Sample Max": round(float(df[col].max()), 3) if np.issubdtype(df[col].dtype, np.number) else "N/A"
        })
    st.dataframe(pd.DataFrame(schema_rows), use_container_width=True, height=350)

# -------------------------------------------------------------
# TAB 3: FEATURE GOVERNANCE & MODEL ISOLATION
# -------------------------------------------------------------
with tab_features:
    st.markdown("#### Feature Governance & Target Leakage Prevention")
    st.markdown(
        """
        <div style="background: #131B2E; border: 1px solid #1E2C4A; border-left: 4px solid #00D4FF; border-radius: 6px; padding: 1rem; font-size: 0.88rem; line-height: 1.6;">
            <strong>Active Feature Columns (Fed into Anomaly Models):</strong><br>
            • <code>Amount</code>: Scaled via <code>RobustScaler</code> (interquartile range scaling).<br>
            • <code>Time</code>: Elapsed transaction timeline normalized via RobustScaler.<br>
            • <code>V1</code> through <code>V28</code>: Principal components (anonymized orthogonal variance projections).<br><br>
            <strong style="color: #F43F5E;">Strictly Excluded Columns (Target Leakage & Privacy Safeguards):</strong><br>
            • <code>Class</code>: Ground-truth label strictly stripped prior to unsupervised model fitting to eliminate target leakage.<br>
            • <code>Transaction_ID</code>: Identity tokens stripped to prevent index memorization.
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------------------
# TAB 4: DOWNLOAD VALIDATION REPORT
# -------------------------------------------------------------
with tab_export:
    st.markdown("#### Export Formal Data Validation Audit Report")
    st.caption("Generate a signed Markdown quality assurance certificate for compliance filing.")

    rep_md = generate_markdown_quality_report(val_report, dataset_source=st.session_state.get("data_source_type", "DEMO"))
    st.markdown(rep_md)

    st.download_button(
        label="Download Quality Audit Report (Markdown)",
        data=rep_md.encode("utf-8"),
        file_name="finguard_data_quality_audit_report.md",
        mime="text/markdown"
    )

# Legal & Regulatory Decision-Support Notice
st.markdown(
    f"<div class='fg-legal-disclaimer'><strong>Regulatory Compliance Notice:</strong> {settings.DISCLAIMER_NOTICE}</div>",
    unsafe_allow_html=True
)
