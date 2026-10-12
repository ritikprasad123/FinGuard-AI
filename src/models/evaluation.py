"""Model evaluation, precision-recall optimization, and threshold calibration for FinGuard AI."""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from src.models.base import BaseAnomalyModel


@dataclass
class ModelEvaluationReport:
    """Rigorous evaluation metrics for imbalanced financial anomaly detection."""
    model_name: str
    threshold: float
    total_samples: int
    total_frauds: int
    predicted_anomalies: int
    precision: float
    recall: float
    f1_score: float
    pr_auc: float
    roc_auc: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    false_positive_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_anomaly_model(
    model: BaseAnomalyModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: Optional[float] = None
) -> ModelEvaluationReport:
    """Evaluates an anomaly detection model against ground-truth validation labels.

    Args:
        model: Trained BaseAnomalyModel instance.
        X_test: Validation feature matrix (out-of-sample).
        y_test: Ground-truth binary labels (1=Fraud, 0=Normal).
        threshold: Decision threshold. Defaults to model.threshold.

    Returns:
        ModelEvaluationReport with mathematically verified metrics.
    """
    thresh = threshold if threshold is not None else model.threshold
    scores = model.predict_score(X_test)
    preds = (scores >= thresh).astype(int)
    y_true = y_test.values

    p = float(precision_score(y_true, preds, zero_division=0))
    r = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))

    try:
        pr_auc = float(average_precision_score(y_true, scores))
    except Exception:
        pr_auc = 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, scores))
    except Exception:
        roc_auc = 0.5

    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return ModelEvaluationReport(
        model_name=model.model_name,
        threshold=round(thresh, 4),
        total_samples=len(X_test),
        total_frauds=int(y_test.sum()),
        predicted_anomalies=int(preds.sum()),
        precision=round(p, 4),
        recall=round(r, 4),
        f1_score=round(f1, 4),
        pr_auc=round(pr_auc, 4),
        roc_auc=round(roc_auc, 4),
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        false_positive_rate=round(fpr, 4)
    )


def evaluate_threshold_grid(
    model: BaseAnomalyModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: Optional[List[float]] = None
) -> pd.DataFrame:
    """Sweeps threshold grid to evaluate precision-recall tradeoffs."""
    if thresholds is None:
        thresholds = list(np.linspace(0.10, 0.95, 18))

    rows = []
    for t in thresholds:
        rep = evaluate_anomaly_model(model, X_test, y_test, threshold=float(t))
        rows.append({
            "Threshold": rep.threshold,
            "Precision": rep.precision,
            "Recall": rep.recall,
            "F1_Score": rep.f1_score,
            "PR_AUC": rep.pr_auc,
            "ROC_AUC": rep.roc_auc,
            "False_Positives (False Alarms)": rep.false_positives,
            "True_Positives (Detected Frauds)": rep.true_positives,
            "False_Negatives (Missed)": rep.false_negatives,
            "False_Positive_Rate": rep.false_positive_rate
        })

    return pd.DataFrame(rows)


def find_optimal_threshold(
    grid_df: pd.DataFrame,
    criterion: str = "F1_Score"
) -> Tuple[float, float]:
    """Finds optimal decision threshold maximizing target evaluation metric."""
    if len(grid_df) == 0:
        raise ValueError("Grid dataframe cannot be empty.")

    best_row = grid_df.loc[grid_df[criterion].idxmax()]
    return float(best_row["Threshold"]), float(best_row[criterion])


def evaluate_baseline_comparisons(
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Dict[str, Any]]:
    """Evaluates standard baseline models to establish comparative benchmarks without leakage."""
    total = len(X_test)
    n_fraud = int(y_test.sum())
    prior_rate = n_fraud / total if total > 0 else 0.0

    # 1. Random Class-Proportion Guessing Baseline
    np.random.seed(42)
    random_preds = (np.random.rand(total) < prior_rate).astype(int)
    cm_rand = confusion_matrix(y_test.values, random_preds, labels=[0, 1])

    # 2. Heuristic Rule-Based Baseline (Amounts > 99th percentile + V14 variance)
    heuristic_preds = np.zeros(total, dtype=int)
    if "Amount" in X_test.columns:
        amt_p99 = np.percentile(X_test["Amount"].dropna(), 99)
        heuristic_preds = (X_test["Amount"] > amt_p99).astype(int).values

    cm_rule = confusion_matrix(y_test.values, heuristic_preds, labels=[0, 1])

    return {
        "Random Proportional Baseline": {
            "precision": round(float(precision_score(y_test, random_preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, random_preds, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, random_preds, zero_division=0)), 4),
            "false_positives": int(cm_rand[0, 1]),
            "true_positives": int(cm_rand[1, 1]),
            "description": f"Random bernoulli trial sampling at ground-truth prevalence ({prior_rate:.3%})"
        },
        "Heuristic Value-Threshold Rule": {
            "precision": round(float(precision_score(y_test, heuristic_preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, heuristic_preds, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, heuristic_preds, zero_division=0)), 4),
            "false_positives": int(cm_rule[0, 1]),
            "true_positives": int(cm_rule[1, 1]),
            "description": "Deterministic rule flagging transactions exceeding the 99th percentile ticket amount"
        }
    }
