"""Unit tests for Model Evaluation and Threshold Grid Analysis."""

import pytest
import pandas as pd
import numpy as np
from src.data.demo_generator import generate_synthetic_demo_data
from src.models.isolation_forest import IsolationForestAnomalyModel
from src.models.evaluation import (
    evaluate_anomaly_model,
    evaluate_threshold_grid,
    find_optimal_threshold,
    evaluate_baseline_comparisons,
    ModelEvaluationReport
)


@pytest.fixture
def eval_data():
    return generate_synthetic_demo_data(n_samples=250, fraud_ratio=0.06, save_to_file=False)


def test_evaluation_metrics_actual_calculation(eval_data):
    model = IsolationForestAnomalyModel(n_estimators=50, random_state=42)
    model.fit(eval_data)

    report: ModelEvaluationReport = evaluate_anomaly_model(model, eval_data, eval_data["Class"], threshold=0.60)

    assert 0.0 <= report.precision <= 1.0
    assert 0.0 <= report.recall <= 1.0
    assert 0.0 <= report.f1_score <= 1.0
    assert 0.0 <= report.pr_auc <= 1.0
    assert report.total_samples == 250
    assert report.total_frauds == 15
    assert report.true_positives + report.false_negatives == 15
    assert report.true_negatives + report.false_positives == 235
    assert isinstance(report.to_dict(), dict)


def test_threshold_grid_tradeoff_behaviour(eval_data):
    model = IsolationForestAnomalyModel(n_estimators=50, random_state=42)
    model.fit(eval_data)

    thresholds = [0.30, 0.50, 0.70, 0.85]
    grid_df = evaluate_threshold_grid(model, eval_data, eval_data["Class"], thresholds=thresholds)

    assert len(grid_df) == 4
    assert "Precision" in grid_df.columns
    assert "Recall" in grid_df.columns
    assert "False_Positives (False Alarms)" in grid_df.columns

    # As threshold increases, false alarms should monotonically decrease or remain equal
    fp_values = grid_df["False_Positives (False Alarms)"].values
    assert fp_values[0] >= fp_values[-1]


def test_find_optimal_threshold(eval_data):
    model = IsolationForestAnomalyModel(n_estimators=50, random_state=42)
    model.fit(eval_data)

    grid_df = evaluate_threshold_grid(model, eval_data, eval_data["Class"])
    best_thresh, best_val = find_optimal_threshold(grid_df, criterion="F1_Score")

    assert 0.0 <= best_thresh <= 1.0
    assert 0.0 <= best_val <= 1.0

    with pytest.raises(ValueError, match="cannot be empty"):
        find_optimal_threshold(pd.DataFrame())


def test_baseline_comparisons_metrics(eval_data):
    X = eval_data.drop(columns=["Class"])
    y = eval_data["Class"]

    baselines = evaluate_baseline_comparisons(X, y)
    assert "Random Proportional Baseline" in baselines
    assert "Heuristic Value-Threshold Rule" in baselines

    for name, b_res in baselines.items():
        assert "precision" in b_res
        assert "recall" in b_res
        assert "f1" in b_res
        assert "false_positives" in b_res
        assert "true_positives" in b_res
        assert 0.0 <= b_res["precision"] <= 1.0
        assert 0.0 <= b_res["recall"] <= 1.0
