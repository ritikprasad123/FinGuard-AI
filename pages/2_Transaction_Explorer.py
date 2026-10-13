"""FinGuard AI - Transaction Monitoring & Advanced Search.

Real-time transaction surveillance table with multi-parameter filtering,
sorting, pagination, and dimensional transaction inspection.
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
from src.models.registry import get_active_model
from src.models.base import classify_risk_tier
from src.investigation.alert_feed import calculate_statistical_risk_score
from src.investigation.case_manager import CaseManager
from src.data.loader import check_benchmark_dataset_exists, load_csv_dataset
from src.data.demo_generator import generate_synthetic_demo_data

configure_page(page_title="Transaction Explorer")
case_manager = CaseManager()

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
    title="Transaction Explorer & Surveillance",
    subtitle="Continuous high-throughput transaction auditing, multi-parameter filtering, and latent dimension inspection.",
    badge_text="MONITORING ACTIVE",
    badge_type="high"
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
active_threshold = active_model.threshold if (active_model and active_model.is_trained) else 0.65

# Ensure core columns
if "Transaction_ID" not in df.columns:
    df["Transaction_ID"] = [f"TXN-SURV-{i+10001:05d}" for i in range(len(df))]

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

# Filter Bar
st.markdown("### Surveillance Filter Console")
with st.container():
    f_c1, f_c2, f_c3, f_c4 = st.columns([1.5, 1.2, 1.2, 1.5])
    with f_c1:
        search_tx = st.text_input("Search Transaction ID", placeholder="e.g. TXN-SURV-10045")
    with f_c2:
        pred_filter = st.selectbox(
            "Prediction / Tier",
            options=["ALL", "ANOMALY ONLY", "NORMAL ONLY", "CRITICAL TIER", "HIGH TIER", "MEDIUM TIER", "LOW TIER"]
        )
    with f_c3:
        min_score = st.slider("Min Anomaly Score", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    with f_c4:
        amt_max_val = float(max(100.0, df[settings.AMOUNT_COL].max()))
        amt_range = st.slider("Amount Range ($)", min_value=0.0, max_value=amt_max_val, value=(0.0, amt_max_val), step=25.0)

# Filter execution
filtered_df = df.copy()
if search_tx:
    filtered_df = filtered_df[filtered_df["Transaction_ID"].str.contains(search_tx, case=False, na=False)]

if pred_filter == "ANOMALY ONLY":
    filtered_df = filtered_df[filtered_df["Prediction"] == "ANOMALY"]
elif pred_filter == "NORMAL ONLY":
    filtered_df = filtered_df[filtered_df["Prediction"] == "NORMAL"]
elif pred_filter == "CRITICAL TIER":
    filtered_df = filtered_df[filtered_df["Severity"] == "CRITICAL"]
elif pred_filter == "HIGH TIER":
    filtered_df = filtered_df[filtered_df["Severity"] == "HIGH"]
elif pred_filter == "MEDIUM TIER":
    filtered_df = filtered_df[filtered_df["Severity"] == "MEDIUM"]
elif pred_filter == "LOW TIER":
    filtered_df = filtered_df[filtered_df["Severity"] == "LOW"]

if min_score > 0.0:
    filtered_df = filtered_df[filtered_df["Risk_Score"] >= min_score]

filtered_df = filtered_df[
    (filtered_df[settings.AMOUNT_COL] >= amt_range[0]) &
    (filtered_df[settings.AMOUNT_COL] <= amt_range[1])
]

# Sorting & Pagination Controls
st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
s_c1, s_c2, s_c3 = st.columns([1.5, 1.5, 2])
with s_c1:
    sort_by = st.selectbox("Sort Field", options=["Risk_Score", "Amount", "Time", "Transaction_ID"])
with s_c2:
    sort_dir = st.selectbox("Sort Order", options=["Descending", "Ascending"])
with s_c3:
    page_size = st.selectbox("Items Per Page", options=[25, 50, 100], index=0)

ascending = (sort_dir == "Ascending")
filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)

total_records = len(filtered_df)
total_pages = max(1, (total_records + page_size - 1) // page_size)

p_col1, p_col2 = st.columns([1, 3])
with p_col1:
    curr_page = st.number_input(f"Page (of {total_pages})", min_value=1, max_value=total_pages, value=1, step=1)
with p_col2:
    st.markdown(
        f"<div style='margin-top: 1.8rem; font-size: 0.85rem; color: var(--secondary-text);'>"
        f"Showing {min(total_records, (curr_page-1)*page_size + 1):,} - "
        f"{min(total_records, curr_page*page_size):,} of {total_records:,} transactions"
        f"</div>",
        unsafe_allow_html=True
    )

start_idx = (curr_page - 1) * page_size
page_df = filtered_df.iloc[start_idx:start_idx + page_size]

# Render Table
display_cols = ["Transaction_ID", settings.TIME_COL, settings.AMOUNT_COL, "Risk_Score", "Prediction", "Severity"]
if settings.TARGET_COL in page_df.columns:
    display_cols.append(settings.TARGET_COL)

st.dataframe(
    page_df[display_cols].style.format({
        settings.AMOUNT_COL: "${:,.2f}",
        "Risk_Score": "{:.4f}",
        settings.TIME_COL: "{:,.0f}"
    }),
    use_container_width=True,
    height=400
)

# Export Filtered Results
col_dl1, col_dl2 = st.columns([2, 1])
with col_dl1:
    st.caption(f"Showing page {curr_page} of {total_pages} ({len(filtered_df):,} matching transactions).")
with col_dl2:
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Export Filtered Results (CSV: {len(filtered_df):,} rows)",
        data=csv_bytes,
        file_name="finguard_filtered_transactions.csv",
        mime="text/csv",
        use_container_width=True
    )

# Deep Transaction Inspection
st.markdown("---")
st.markdown("### Transaction Detail Inspector")
inspect_id = st.selectbox("Select Transaction to Inspect", options=page_df["Transaction_ID"].tolist()) if len(page_df) > 0 else None

if inspect_id:
    target_row = page_df[page_df["Transaction_ID"] == inspect_id].iloc[0]
    i1, i2, i3, i4 = st.columns(4)
    with i1:
        st.metric("Transaction ID", inspect_id)
    with i2:
        st.metric("Ticket Amount", f"${target_row[settings.AMOUNT_COL]:,.2f}")
    with i3:
        st.metric("Anomaly Score", f"{target_row['Risk_Score']:.4f}")
    with i4:
        st.metric("Model Classification", target_row["Prediction"])

    with st.expander("Full 30 Latent Feature Dimensions (V1 - V28, Time, Amount)", expanded=False):
        pca_data = {col: round(float(target_row[col]), 4) for col in settings.PCA_FEATURE_COLS if col in target_row}
        pca_df = pd.DataFrame(list(pca_data.items()), columns=["Component", "Value"])
        st.dataframe(pca_df, use_container_width=True, height=260)

    col_act1, col_act2 = st.columns(2)
    with col_act1:
        if st.button("Send to AI Investigation Assistant", key=f"btn_send_ai_{inspect_id}", type="primary"):
            st.session_state.selected_tx_for_ai = target_row.to_dict()
            st.success(f"Loaded {inspect_id} into AI Assistant! Navigate to 'AI Investigation Assistant'.")
    with col_act2:
        if st.button("Open Case File for Transaction", key=f"btn_case_open_{inspect_id}"):
            new_cid = case_manager.create_case(
                title=f"Flagged Transaction {inspect_id} (${target_row[settings.AMOUNT_COL]:,.2f})",
                priority=str(target_row["Severity"]),
                assigned_analyst="Monitoring-Desk",
                linked_transactions=[inspect_id],
                summary=f"Automated case opened from Monitoring desk for {inspect_id} (Score: {target_row['Risk_Score']:.4f}).",
                initial_note=f"Triage risk: {target_row['Severity']} rating exceeding threshold {active_threshold}."
            )
            st.success(f"Created Case: `{new_cid}` in persistent SQLite database!")


# Legal & Regulatory Decision-Support Notice
st.markdown(
    f"<div class='fg-legal-disclaimer'><strong>Regulatory Compliance Notice:</strong> {settings.DISCLAIMER_NOTICE}</div>",
    unsafe_allow_html=True
)
