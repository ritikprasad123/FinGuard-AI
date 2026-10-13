"""FinGuard AI - Model Performance & Evaluation Studio.

Out-of-sample model evaluation, confusion matrix heatmaps, Precision-Recall curves,
threshold trade-off sweeps, baseline comparisons, and model registry management.
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
from components.charts import (
    create_confusion_matrix_chart,
    create_precision_recall_curve_chart,
    create_threshold_tradeoff_chart
)
from src.features.preprocessor import split_transaction_data, TransactionPreprocessor
from src.models.isolation_forest import IsolationForestAnomalyModel
from src.models.autoencoder import AutoencoderAnomalyModel
from src.models.evaluation import (
    evaluate_anomaly_model,
    evaluate_threshold_grid,
    find_optimal_threshold,
    evaluate_baseline_comparisons,
    ModelEvaluationReport
)
from src.models.registry import get_active_model, set_active_model, list_available_models
from src.data.loader import check_benchmark_dataset_exists, load_csv_dataset
from src.data.demo_generator import generate_synthetic_demo_data

configure_page(page_title="Model Lab")

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
    title="Model Lab & Comparative Evaluation",
    subtitle="Multi-model training studio, empirical holdout comparisons, decision trade-off curves, and configuration exports.",
    badge_text="METRICS VERIFIED",
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

# Model Selector & Switcher
st.markdown("### Active Model Deployment")
col_reg1, col_reg2 = st.columns([2, 1])
with col_reg1:
    available_models = list_available_models()
    model_opts = {m.model_name: m.model_name for m in available_models} if available_models else {"Isolation Forest (Default)": "Isolation Forest"}
    chosen_m_name = st.selectbox("Select Active Production Engine", options=list(model_opts.keys()))
with col_reg2:
    st.markdown("<div style='height: 1.8rem;'></div>", unsafe_allow_html=True)
    if st.button("Set as Active Engine", type="primary"):
        matching = [m for m in available_models if m.model_name == chosen_m_name]
        if matching:
            set_active_model(matching[0])
            st.success(f"Switched active model to: {chosen_m_name}")
            st.rerun()

# Out-of-Sample Evaluation Execution
st.markdown("---")
st.markdown("### Out-of-Sample Holdout Evaluation")

if settings.TARGET_COL not in df.columns:
    st.warning("The current dataset lacks ground-truth 'Class' labels for supervised metric calculation. Metrics require benchmark labels.")
    st.stop()

# Cache train/test split to avoid recomputing
@st.cache_data
def get_cached_split(data_len: int, target_sum: int):
    # Sanitize input df so only numeric feature columns and Class are split
    clean_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["Risk_Score", "anomaly_score"]]
    if settings.TARGET_COL in df.columns and settings.TARGET_COL not in clean_cols:
        clean_cols.append(settings.TARGET_COL)
    return split_transaction_data(df[clean_cols], target_col="Class", test_size=0.20, random_state=42)

X_train, X_test, y_train, y_test = get_cached_split(len(df), int(df[settings.TARGET_COL].sum()))

# Ensure model is trained
if active_model and not active_model.is_trained:
    with st.spinner(f"Fitting {active_model.model_name} on training partition (N={len(X_train):,})..."):
        active_model.fit(X_train)

# Slider for decision threshold
eval_cutoff = st.slider(
    "Evaluation Decision Threshold",
    min_value=0.10,
    max_value=0.95,
    value=float(active_model.threshold if active_model else 0.65),
    step=0.05
)

# Run Evaluation
report: ModelEvaluationReport = evaluate_anomaly_model(active_model, X_test, y_test, threshold=eval_cutoff)

# Metric KPI Row
m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
with m_c1:
    st.metric("Precision", f"{report.precision:.2%}")
with m_c2:
    st.metric("Recall", f"{report.recall:.2%}")
with m_c3:
    st.metric("F1-Score", f"{report.f1_score:.4f}")
with m_c4:
    st.metric("PR-AUC", f"{report.pr_auc:.4f}")
with m_c5:
    st.metric("ROC-AUC", f"{report.roc_auc:.4f}")

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

# Confusion Matrix and Tradeoff Charts
ch_col1, ch_col2 = st.columns([1, 1.4])
with ch_col1:
    st.markdown("#### Confusion Matrix")
    cm_payload = {
        "true_positives": report.true_positives,
        "false_positives": report.false_positives,
        "true_negatives": report.true_negatives,
        "false_negatives": report.false_negatives
    }
    st.plotly_chart(
        create_confusion_matrix_chart(cm_payload, threshold=eval_cutoff),
        use_container_width=True
    )

with ch_col2:
    st.markdown("#### Threshold Sensitivity Sweep")
    grid_df = evaluate_threshold_grid(active_model, X_test, y_test)
    st.plotly_chart(create_threshold_tradeoff_chart(grid_df), use_container_width=True)

# Curves Row: PR Curve and ROC Curve
crv_col1, crv_col2 = st.columns(2)
with crv_col1:
    st.markdown("#### Precision-Recall Operating Curve")
    test_scores = active_model.predict_score(X_test)
    st.plotly_chart(
        create_precision_recall_curve_chart(y_test.values, test_scores, pr_auc=report.pr_auc, current_threshold=eval_cutoff),
        use_container_width=True
    )

with crv_col2:
    st.markdown("#### ROC Operating Curve")
    from components.charts import create_roc_curve_chart
    st.plotly_chart(
        create_roc_curve_chart(y_test.values, test_scores, roc_auc=report.roc_auc),
        use_container_width=True
    )

# Operational Trade-Off Analysis
st.markdown("---")
st.markdown("### Operational Risk Trade-Off & Imbalance Caveats")
st.markdown(
    """
    <div style="background: var(--card-bg); border: 1px solid var(--border-color); border-left: 4px solid var(--warning-amber); border-radius: 6px; padding: 1.2rem; font-size: 0.88rem; line-height: 1.6; margin-bottom: 1.5rem;">
        <strong style="color: #F4F6F8; font-size: 0.95rem;">Fraud Class Imbalance & Evaluation Caveat:</strong><br>
        In financial payment streams, fraudulent transactions constitute a minute fraction (typically &lt; 0.2%).
        Therefore, <strong style="color: #FF4D5A;">raw Accuracy is highly misleading</strong>: a naive system predicting every transaction as legitimate achieves ~99.8% accuracy while missing 100% of actual fraud.
        <ul style="margin-top: 0.5rem; margin-bottom: 0.5rem; padding-left: 1.25rem;">
            <li><strong>High Decision Cutoff (e.g. 0.80):</strong> Maximizes Precision and reduces false alarms for compliance desks, but increases Missed Frauds (False Negatives).</li>
            <li><strong>Low Decision Cutoff (e.g. 0.50):</strong> Maximizes Recall by capturing more suspicious transactions, but increases False Alarms (False Positives) requiring analyst triage.</li>
            <li><strong>Model Predictions:</strong> Flagged anomalies are mathematical deviation indicators, not confirmed financial crime. Confirmed outcomes require investigative audit.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

# Multi-Model Benchmark Comparison
st.markdown("### Multi-Model Benchmark Comparison")
st.caption("Empirical performance comparison of unsupervised anomaly detection engines on holdout partition.")

comp_rows = []
# 1. Active Model
comp_rows.append({
    "Engine": f"Active: {active_model.model_name}",
    "Type": "Unsupervised Active",
    "Precision": f"{report.precision:.2%}",
    "Recall": f"{report.recall:.2%}",
    "F1-Score": f"{report.f1_score:.4f}",
    "PR-AUC": f"{report.pr_auc:.4f}",
    "ROC-AUC": f"{report.roc_auc:.4f}",
    "False Alarms": report.false_positives,
    "Captured Frauds": report.true_positives
})

# 2. Evaluate other available models if requested
if st.button("Run Comprehensive Multi-Model Benchmark (Isolation Forest vs LOF vs Autoencoder)"):
    with st.spinner("Benchmarking all available models on holdout partition..."):
        from src.models.isolation_forest import IsolationForestAnomalyModel
        from src.models.lof_detector import LocalOutlierFactorAnomalyModel
        from src.models.autoencoder import AutoencoderAnomalyModel

        candidates = [
            ("Isolation Forest", IsolationForestAnomalyModel(n_estimators=100, random_state=42)),
            ("Local Outlier Factor (LOF)", LocalOutlierFactorAnomalyModel(n_neighbors=20, random_state=42)),
            ("Neural Autoencoder", AutoencoderAnomalyModel(max_iter=100, random_state=42))
        ]

        for cand_name, cand_model in candidates:
            if cand_name == active_model.model_name:
                continue
            try:
                cand_model.fit(X_train)
                rep = evaluate_anomaly_model(cand_model, X_test, y_test, threshold=eval_cutoff)
                comp_rows.append({
                    "Engine": cand_name,
                    "Type": "Unsupervised Candidate",
                    "Precision": f"{rep.precision:.2%}",
                    "Recall": f"{rep.recall:.2%}",
                    "F1-Score": f"{rep.f1_score:.4f}",
                    "PR-AUC": f"{rep.pr_auc:.4f}",
                    "ROC-AUC": f"{rep.roc_auc:.4f}",
                    "False Alarms": rep.false_positives,
                    "Captured Frauds": rep.true_positives
                })
            except Exception as exc:
                st.warning(f"Could not benchmark {cand_name}: {exc}")

# Baselines
baselines = evaluate_baseline_comparisons(X_test, y_test)
for b_name, b_val in baselines.items():
    comp_rows.append({
        "Engine": b_name,
        "Type": "Reference Baseline",
        "Precision": f"{b_val['precision']:.2%}",
        "Recall": f"{b_val['recall']:.2%}",
        "F1-Score": f"{b_val['f1']:.4f}",
        "PR-AUC": "N/A",
        "ROC-AUC": "N/A",
        "False Alarms": b_val["false_positives"],
        "Captured Frauds": b_val["true_positives"]
    })

st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)


# -------------------------------------------------------------
# REPRODUCIBLE MODEL CONFIGURATION EXPORT (Section 6 - Page 4)
# -------------------------------------------------------------
st.markdown("---")
st.markdown("### Reproducible Model Architecture Configuration")
st.caption("Export audited serialization metadata and operational hyperparameters for governance compliance.")

import json
config_export = {
    "app_name": settings.APP_NAME,
    "version": settings.VERSION,
    "active_model_name": active_model.model_name if active_model else "None",
    "active_model_id": active_model.model_id if active_model else "None",
    "evaluation_threshold": round(eval_cutoff, 4),
    "feature_dimensions": settings.REQUIRED_BENCHMARK_FEATURES,
    "scaler_type": settings.SCALER_TYPE,
    "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
    "provenance": {
        "dataset_rows": len(df),
        "data_source_type": st.session_state.get("data_source_type", "DEMO"),
        "has_ground_truth": settings.TARGET_COL in df.columns
    },
    "metrics_snapshot": {
        "precision": report.precision,
        "recall": report.recall,
        "f1_score": report.f1_score,
        "pr_auc": report.pr_auc,
        "roc_auc": report.roc_auc,
        "true_positives": report.true_positives,
        "false_positives": report.false_positives,
        "false_negatives": report.false_negatives
    }
}

config_json_str = json.dumps(config_export, indent=2)

col_exp1, col_exp2 = st.columns([2, 1])
with col_exp1:
    st.json(config_export)
with col_exp2:
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    st.download_button(
        label="Download Reproducible Config (JSON)",
        data=config_json_str.encode("utf-8"),
        file_name=f"finguard_model_config_{active_model.model_id if active_model else 'default'}.json",
        mime="application/json",
        use_container_width=True
    )

# Legal & Regulatory Decision-Support Notice
st.markdown(
    f"<div class='fg-legal-disclaimer'><strong>Regulatory Compliance Notice:</strong> {settings.DISCLAIMER_NOTICE}</div>",
    unsafe_allow_html=True
)
