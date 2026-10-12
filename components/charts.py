"""Plotly dark financial intelligence visualization charts for FinGuard AI.

Built using bespoke design tokens: #0B0D10 background, #15191F surfaces,
#FF4D5A critical crimson accents, and #35C98A success green.
"""

from typing import Optional, Dict, Any, List, Union
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve
from config.settings import settings


def _apply_dark_theme(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    """Applies unified FinGuard dark theme styling to Plotly figure."""
    fig.update_layout(
        title={
            "text": title,
            "font": {"family": "Inter, sans-serif", "size": 14, "color": settings.tokens.PRIMARY_TEXT},
            "x": 0.02,
            "y": 0.95
        },
        height=height,
        paper_bgcolor=settings.tokens.CARD_BG,
        plot_bgcolor=settings.tokens.CARD_BG,
        margin=dict(l=45, r=35, t=55, b=45),
        font={"family": "Inter, sans-serif", "color": settings.tokens.SECONDARY_TEXT, "size": 11},
        legend={
            "font": {"family": "Inter, sans-serif", "size": 11, "color": settings.tokens.PRIMARY_TEXT},
            "bgcolor": "rgba(0,0,0,0)",
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0
        },
        xaxis=dict(
            showgrid=True,
            gridcolor=settings.tokens.BORDER_COLOR,
            zeroline=False,
            color=settings.tokens.SECONDARY_TEXT
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=settings.tokens.BORDER_COLOR,
            zeroline=False,
            color=settings.tokens.SECONDARY_TEXT
        ),
        hoverlabel=dict(
            bgcolor=settings.tokens.ELEVATED_CARD_BG,
            font_size=12,
            font_family="JetBrains Mono, monospace",
            bordercolor=settings.tokens.BORDER_COLOR
        )
    )
    return fig


def create_transaction_activity_chart(df: pd.DataFrame, max_points: int = 2500) -> go.Figure:
    """Generates financial transaction volume timeline highlighting anomalies."""
    if len(df) == 0:
        fig = go.Figure()
        return _apply_dark_theme(fig, "No Transaction Activity Data Available")

    plot_df = df.sample(n=min(len(df), max_points), random_state=42) if len(df) > max_points else df.copy()

    if "ML_Prediction" in plot_df.columns:
        is_anomaly = plot_df["ML_Prediction"] == 1
    elif "Risk_Score" in plot_df.columns:
        is_anomaly = plot_df["Risk_Score"] >= settings.RISK_HIGH_THRESHOLD
    elif settings.TARGET_COL in plot_df.columns:
        is_anomaly = plot_df[settings.TARGET_COL] == 1
    else:
        is_anomaly = pd.Series([False] * len(plot_df))

    fig = go.Figure()

    normal_subset = plot_df[~is_anomaly]
    fig.add_trace(go.Scatter(
        x=normal_subset[settings.TIME_COL],
        y=normal_subset[settings.AMOUNT_COL],
        mode="markers",
        name="Normal Transaction",
        marker=dict(
            color=settings.tokens.SUCCESS_GREEN,
            size=4,
            opacity=0.45
        ),
        hovertemplate="Time: %{x}s<br>Amount: $%{y:,.2f}<extra>Normal</extra>"
    ))

    anomaly_subset = plot_df[is_anomaly]
    if len(anomaly_subset) > 0:
        fig.add_trace(go.Scatter(
            x=anomaly_subset[settings.TIME_COL],
            y=anomaly_subset[settings.AMOUNT_COL],
            mode="markers",
            name="Suspicious Anomaly",
            marker=dict(
                color=settings.tokens.CRITICAL_RED,
                size=8,
                symbol="circle",
                line=dict(width=1.5, color="#FFFFFF")
            ),
            hovertemplate="Time: %{x}s<br>Amount: $%{y:,.2f}<extra>SUSPICIOUS</extra>"
        ))

    fig.update_xaxes(title_text="Time (Elapsed Seconds)")
    fig.update_yaxes(title_text="Amount (USD)")
    return _apply_dark_theme(fig, "Transaction Stream Timeline & Anomaly Detection")


def create_amount_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Generates financial ticket size distribution comparing normal vs suspicious transactions."""
    if len(df) == 0:
        fig = go.Figure()
        return _apply_dark_theme(fig, "No Amount Distribution Data Available")

    if "ML_Prediction" in df.columns:
        is_anomaly = df["ML_Prediction"] == 1
    elif "Risk_Score" in df.columns:
        is_anomaly = df["Risk_Score"] >= settings.RISK_HIGH_THRESHOLD
    elif settings.TARGET_COL in df.columns:
        is_anomaly = df[settings.TARGET_COL] == 1
    else:
        is_anomaly = pd.Series([False] * len(df))

    fig = go.Figure()

    normal_amts = df[~is_anomaly][settings.AMOUNT_COL].dropna()
    anomaly_amts = df[is_anomaly][settings.AMOUNT_COL].dropna()

    fig.add_trace(go.Box(
        y=normal_amts,
        name="Normal",
        marker_color=settings.tokens.SUCCESS_GREEN,
        boxpoints="outliers",
        line_width=1.5
    ))

    if len(anomaly_amts) > 0:
        fig.add_trace(go.Box(
            y=anomaly_amts,
            name="Suspicious",
            marker_color=settings.tokens.CRITICAL_RED,
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
            line_width=1.5
        ))

    fig.update_yaxes(title_text="Amount (USD)", type="log")
    return _apply_dark_theme(fig, "Transaction Amount Distribution (Logarithmic Scale)")


def create_hourly_volume_chart(velocity_df: pd.DataFrame) -> go.Figure:
    """Generates grouped bar chart of transaction counts and anomaly velocity over time."""
    if len(velocity_df) == 0:
        fig = go.Figure()
        return _apply_dark_theme(fig, "No Velocity Data Available")

    # If raw transaction DataFrame passed without 'Hour', calculate hourly velocity automatically
    if "Hour" not in velocity_df.columns:
        from src.monitoring.metrics import calculate_hourly_velocity
        velocity_df = calculate_hourly_velocity(velocity_df)

    if len(velocity_df) == 0 or "Hour" not in velocity_df.columns:
        fig = go.Figure()
        return _apply_dark_theme(fig, "No Velocity Data Available")

    fig = go.Figure()

    count_series = velocity_df["Transaction_Count"] if "Transaction_Count" in velocity_df.columns else velocity_df.iloc[:, 1]
    fig.add_trace(go.Bar(
        x=velocity_df["Hour"],
        y=count_series,
        name="Total Volume",
        marker_color="#2A3546"
    ))

    if "Anomaly_Count" in velocity_df.columns:
        fig.add_trace(go.Bar(
            x=velocity_df["Hour"],
            y=velocity_df["Anomaly_Count"],
            name="Flagged Anomalies",
            marker_color=settings.tokens.PRIMARY_RED
        ))

    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Hour Window (Elapsed)")
    fig.update_yaxes(title_text="Transaction Count")
    return _apply_dark_theme(fig, "Hourly Transaction Volume & Anomaly Velocity")


def create_pca_scatter_chart(df: pd.DataFrame, feat_x: str = "V14", feat_y: str = "V17") -> go.Figure:
    """Generates 2D scatter of key latent PCA discriminant features."""
    if len(df) == 0 or feat_x not in df.columns or feat_y not in df.columns:
        fig = go.Figure()
        return _apply_dark_theme(fig, "Latent Feature Space Unavailable")

    plot_df = df.sample(n=min(len(df), 2000), random_state=42) if len(df) > 2000 else df.copy()

    if "ML_Prediction" in plot_df.columns:
        is_anomaly = plot_df["ML_Prediction"] == 1
    elif "Risk_Score" in plot_df.columns:
        is_anomaly = plot_df["Risk_Score"] >= settings.RISK_HIGH_THRESHOLD
    elif settings.TARGET_COL in plot_df.columns:
        is_anomaly = plot_df[settings.TARGET_COL] == 1
    else:
        is_anomaly = pd.Series([False] * len(plot_df))

    fig = go.Figure()

    normal = plot_df[~is_anomaly]
    fig.add_trace(go.Scatter(
        x=normal[feat_x],
        y=normal[feat_y],
        mode="markers",
        name="Normal",
        marker=dict(color=settings.tokens.SUCCESS_GREEN, size=5, opacity=0.4)
    ))

    flagged = plot_df[is_anomaly]
    if len(flagged) > 0:
        fig.add_trace(go.Scatter(
            x=flagged[feat_x],
            y=flagged[feat_y],
            mode="markers",
            name="Suspicious",
            marker=dict(color=settings.tokens.CRITICAL_RED, size=9, symbol="diamond")
        ))

    fig.update_xaxes(title_text=f"Latent Feature {feat_x}")
    fig.update_yaxes(title_text=f"Latent Feature {feat_y}")
    return _apply_dark_theme(fig, f"Latent Feature Clustering ({feat_x} vs {feat_y})")


# -------------------------------------------------------------
# PHASE 2 SPECIFIC VISUALIZATIONS
# -------------------------------------------------------------

def create_anomaly_score_distribution_chart(scores: np.ndarray, threshold: float = 0.65) -> go.Figure:
    """Plots histogram of model anomaly scores with decision threshold cutoff."""
    if len(scores) == 0:
        fig = go.Figure()
        return _apply_dark_theme(fig, "No Score Distribution Available")

    fig = go.Figure()

    # Scores histogram
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=40,
        name="Score Density",
        marker=dict(
            color="#2A3546",
            line=dict(color="#1F2633", width=1)
        ),
        opacity=0.85
    ))

    # Vertical threshold line
    fig.add_vline(
        x=threshold,
        line_width=2.5,
        line_dash="dash",
        line_color=settings.tokens.CRITICAL_RED,
        annotation_text=f"Cutoff: {threshold:.2f}",
        annotation_position="top right",
        annotation_font_color=settings.tokens.CRITICAL_RED,
        annotation_font_family="JetBrains Mono, monospace"
    )

    fig.update_xaxes(title_text="Anomaly Score [0.0 = Normal, 1.0 = Highly Anomalous]", range=[0.0, 1.0])
    fig.update_yaxes(title_text="Transaction Count")
    return _apply_dark_theme(fig, "Anomaly Score Frequency Distribution & Decision Boundary")


def create_confusion_matrix_chart(
    cm_dict: Optional[Dict[str, int]] = None,
    threshold: float = 0.65,
    tp: Optional[int] = None,
    fp: Optional[int] = None,
    tn: Optional[int] = None,
    fn: Optional[int] = None,
    **kwargs
) -> go.Figure:
    """Renders 2x2 confusion matrix heatmap with counts and audit percentages."""
    if cm_dict is not None and isinstance(cm_dict, dict):
        _tp = int(cm_dict.get("true_positives", cm_dict.get("tp", 0)))
        _fp = int(cm_dict.get("false_positives", cm_dict.get("fp", 0)))
        _tn = int(cm_dict.get("true_negatives", cm_dict.get("tn", 0)))
        _fn = int(cm_dict.get("false_negatives", cm_dict.get("fn", 0)))
    else:
        _tp = int(tp) if tp is not None else int(kwargs.get("true_positives", 0))
        _fp = int(fp) if fp is not None else int(kwargs.get("false_positives", 0))
        _tn = int(tn) if tn is not None else int(kwargs.get("true_negatives", 0))
        _fn = int(fn) if fn is not None else int(kwargs.get("false_negatives", 0))

    thresh_val = float(kwargs.get("threshold", threshold))
    z = [[_tn, _fp], [_fn, _tp]]
    total = _tp + _fp + _tn + _fn
    total_safe = max(1, total)

    text_annotations = [
        [
            f"True Negatives<br><b>{_tn:,}</b><br>({_tn/total_safe:.2%})",
            f"False Alarms (FP)<br><b>{_fp:,}</b><br>({_fp/total_safe:.2%})"
        ],
        [
            f"Missed Frauds (FN)<br><b>{_fn:,}</b><br>({_fn/total_safe:.2%})",
            f"True Positives (TP)<br><b>{_tp:,}</b><br>({_tp/total_safe:.2%})"
        ]
    ]

    colorscale = [
        [0.0, "#10141A"],
        [0.5, "#1B222D"],
        [1.0, "#2B384B"]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=["Predicted Normal", "Predicted Anomaly"],
        y=["Actual Legitimate", "Actual Fraud"],
        text=text_annotations,
        texttemplate="%{text}",
        textfont={"family": "Inter, sans-serif", "size": 12, "color": "#FFFFFF"},
        colorscale=colorscale,
        showscale=False
    ))

    fig.update_xaxes(side="bottom")
    return _apply_dark_theme(fig, f"Evaluation Confusion Matrix (Threshold = {thresh_val:.2f})", height=320)


def create_threshold_tradeoff_chart(
    grid_df: pd.DataFrame,
    selected_threshold: Optional[float] = None,
    **kwargs
) -> go.Figure:
    """Visualizes Precision, Recall, and False Alarm tradeoff across decision boundaries."""
    if len(grid_df) == 0:
        fig = go.Figure()
        return _apply_dark_theme(fig, "No Threshold Grid Data Available")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=grid_df["Threshold"],
        y=grid_df["Precision"],
        mode="lines+markers",
        name="Precision",
        line=dict(color="#3B82F6", width=2.5),
        marker=dict(size=6),
        hovertemplate="Threshold: %{x:.2f}<br>Precision: %{y:.3f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=grid_df["Threshold"],
        y=grid_df["Recall"],
        mode="lines+markers",
        name="Recall",
        line=dict(color=settings.tokens.SUCCESS_GREEN, width=2.5),
        marker=dict(size=6),
        hovertemplate="Threshold: %{x:.2f}<br>Recall: %{y:.3f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=grid_df["Threshold"],
        y=grid_df["F1_Score"],
        mode="lines+markers",
        name="F1-Score",
        line=dict(color=settings.tokens.PRIMARY_RED, width=3),
        marker=dict(size=7),
        hovertemplate="Threshold: %{x:.2f}<br>F1-Score: %{y:.3f}<extra></extra>"
    ))

    cutoff = selected_threshold if selected_threshold is not None else kwargs.get("threshold")
    if cutoff is not None:
        fig.add_vline(
            x=cutoff,
            line_width=2,
            line_dash="dot",
            line_color="#F4F6F8",
            annotation_text=f"Active: {cutoff:.2f}",
            annotation_position="bottom right",
            annotation_font_color="#F4F6F8"
        )

    fig.update_xaxes(title_text="Decision Cutoff Threshold", range=[0.05, 1.0])
    fig.update_yaxes(title_text="Performance Metric [0.0 - 1.0]", range=[0.0, 1.05])
    return _apply_dark_theme(fig, "Precision vs Recall vs F1-Score Tradeoff Across Thresholds")


def create_precision_recall_curve_chart(
    y_true: Any,
    scores: np.ndarray,
    pr_auc: float = 0.0,
    current_threshold: Optional[float] = None,
    **kwargs
) -> go.Figure:
    """Plots Precision-Recall curve with ground-truth baseline comparison."""
    from sklearn.metrics import auc
    y_arr = np.asarray(y_true)
    prec, rec, thresholds = precision_recall_curve(y_arr, scores)
    if pr_auc <= 0.0:
        pr_auc = float(auc(rec, prec))
    prior = float(y_arr.mean()) if len(y_arr) > 0 else 0.0

    fig = go.Figure()

    # PR curve
    fig.add_trace(go.Scatter(
        x=rec,
        y=prec,
        mode="lines",
        name=f"Model PR Curve (AUC={pr_auc:.3f})",
        line=dict(color=settings.tokens.PRIMARY_RED, width=3),
        fill="tozeroy",
        fillcolor="rgba(229, 72, 77, 0.08)"
    ))

    # Baseline reference line
    fig.add_trace(go.Scatter(
        x=[0.0, 1.0],
        y=[prior, prior],
        mode="lines",
        name=f"No-Skill Baseline ({prior:.3%})",
        line=dict(color=settings.tokens.SECONDARY_TEXT, width=1.5, dash="dash")
    ))

    # Optional cutoff marker if current_threshold is provided
    cutoff = current_threshold if current_threshold is not None else kwargs.get("threshold")
    if cutoff is not None and len(thresholds) > 0:
        idx = np.argmin(np.abs(thresholds - cutoff))
        if idx < len(rec) and idx < len(prec):
            fig.add_trace(go.Scatter(
                x=[rec[idx]],
                y=[prec[idx]],
                mode="markers",
                name=f"Operating Point ({cutoff:.2f})",
                marker=dict(color="#FFFFFF", size=10, symbol="diamond", line=dict(color=settings.tokens.PRIMARY_RED, width=2))
            ))

    fig.update_xaxes(title_text="Recall (Fraud Coverage)", range=[0.0, 1.0])
    fig.update_yaxes(title_text="Precision (Positive Predictive Value)", range=[0.0, 1.05])
    return _apply_dark_theme(fig, f"Precision-Recall Curve (PR-AUC = {pr_auc:.4f})")


def create_roc_curve_chart(
    y_true: Any,
    scores: np.ndarray,
    roc_auc: float = 0.0
) -> go.Figure:
    """Plots Receiver Operating Characteristic (ROC) curve with diagonal chance line."""
    from sklearn.metrics import roc_curve, auc
    y_arr = np.asarray(y_true)
    fpr, tpr, _ = roc_curve(y_arr, scores)
    if roc_auc <= 0.0:
        roc_auc = float(auc(fpr, tpr))

    fig = go.Figure()

    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name=f"ROC Curve (AUC={roc_auc:.4f})",
        line=dict(color="#3B82F6", width=3),
        fill="tozeroy",
        fillcolor="rgba(59, 130, 246, 0.08)"
    ))

    # Chance diagonal
    fig.add_trace(go.Scatter(
        x=[0.0, 1.0],
        y=[0.0, 1.0],
        mode="lines",
        name="Random Classifier (AUC=0.50)",
        line=dict(color=settings.tokens.SECONDARY_TEXT, width=1.5, dash="dash")
    ))

    fig.update_xaxes(title_text="False Positive Rate (Fall-out)", range=[0.0, 1.0])
    fig.update_yaxes(title_text="True Positive Rate (Sensitivity / Recall)", range=[0.0, 1.05])
    return _apply_dark_theme(fig, f"ROC Operating Characteristic Curve (ROC-AUC = {roc_auc:.4f})")



def create_transaction_feature_analysis_chart(
    feature_deviations: Dict[str, float],
    transaction_id: str = "Transaction"
) -> go.Figure:
    """Waterfall/bar chart displaying feature deviations for a single flagged transaction."""
    if not feature_deviations:
        fig = go.Figure()
        return _apply_dark_theme(fig, "No Feature Deviations Available")

    feats = list(feature_deviations.keys())
    vals = [feature_deviations[k] for k in feats]

    colors = [settings.tokens.CRITICAL_RED if abs(v) > 2.0 else settings.tokens.PRIMARY_RED if abs(v) > 1.0 else "#4B5565" for v in vals]

    fig = go.Figure(go.Bar(
        x=feats,
        y=vals,
        marker_color=colors,
        text=[f"{v:+.2f}" for v in vals],
        textposition="outside"
    ))

    fig.update_xaxes(title_text="Latent Feature Component")
    fig.update_yaxes(title_text="Normalized Feature Value / Deviation")
    return _apply_dark_theme(fig, f"Forensic Feature Profile: {transaction_id}", height=320)
