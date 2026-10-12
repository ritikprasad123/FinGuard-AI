"""Unit tests for Simulated Stream Monitor."""

import pytest
import pandas as pd
from src.data.demo_generator import generate_synthetic_demo_data
from src.monitoring.stream_monitor import SimulatedStreamMonitor


@pytest.fixture
def stream_data():
    return generate_synthetic_demo_data(n_samples=50, fraud_ratio=0.10, save_to_file=False)


def test_stream_monitor_step_and_bounds(stream_data):
    monitor = SimulatedStreamMonitor(data_source=stream_data, threshold=0.60)
    assert monitor.current_index == 0
    assert len(monitor.streamed_records) == 0

    # Step 5 transactions
    res = monitor.step(batch_size=5)
    assert len(res) == 5
    assert monitor.current_index == 5
    assert len(monitor.streamed_records) == 5

    # Step beyond remaining
    res_more = monitor.step(batch_size=100)
    assert len(res_more) == 45
    assert monitor.current_index == 50
    assert len(monitor.streamed_records) == 50

    # Stepping when buffer is exhausted returns empty without errors
    res_empty = monitor.step(batch_size=5)
    assert len(res_empty) == 0


def test_stream_monitor_alert_generation(stream_data):
    # Set low threshold to guarantee alerts
    monitor = SimulatedStreamMonitor(data_source=stream_data, threshold=0.10)
    monitor.step(batch_size=20)

    metrics = monitor.get_metrics()
    assert metrics["total_streamed"] == 20
    assert metrics["total_alerts"] > 0
    assert metrics["is_demo_stream"] is True
    assert "DEMO DATA (SIMULATED STREAM)" in metrics["disclaimer"]

    df_stream = monitor.get_stream_dataframe()
    assert len(df_stream) == 20
    assert "anomaly_score" in df_stream.columns


def test_stream_monitor_reset(stream_data):
    monitor = SimulatedStreamMonitor(data_source=stream_data)
    monitor.step(batch_size=15)
    assert monitor.current_index == 15
    assert len(monitor.streamed_records) == 15

    monitor.reset()
    assert monitor.current_index == 0
    assert len(monitor.streamed_records) == 0
    assert len(monitor.alerts_generated) == 0
