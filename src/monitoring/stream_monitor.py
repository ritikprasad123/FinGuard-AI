"""Simulated Real-Time Transaction Stream Monitor.

Provides transaction-by-transaction model inference simulation for testing
surveillance pipelines without blocking infinite loops or external live APIs.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from config.settings import settings
from src.models.base import BaseAnomalyModel, InferenceResult, classify_risk_tier
from src.models.registry import get_active_model
from src.investigation.alert_feed import calculate_statistical_risk_score
from src.utils.logger import get_logger

logger = get_logger("finguard.monitoring.stream_monitor")


class SimulatedStreamMonitor:
    """Stateful, non-blocking stream emulator executing per-transaction inference."""

    STREAM_DISCLAIMER = (
        "DEMO DATA (SIMULATED STREAM) NOTICE: This transaction feed is a synthetic simulation "
        "engineered for local surveillance verification. It is not connected to live banking clearing houses, "
        "SWIFT, Fedwire, or card network gateways. All transaction records are simulated."
    )

    def __init__(
        self,
        data_source: pd.DataFrame,
        model: Optional[BaseAnomalyModel] = None,
        threshold: float = 0.65
    ):
        """Initializes stream monitor buffer and model reference."""
        self.data_source = data_source.copy() if data_source is not None else pd.DataFrame()
        self.model = model or get_active_model()
        self.threshold = threshold
        self.current_index: int = 0
        self.is_paused: bool = True
        self.streamed_records: List[Dict[str, Any]] = []
        self.alerts_generated: List[Dict[str, Any]] = []

    def set_threshold(self, threshold: float) -> None:
        """Updates decision threshold for ongoing stream inference."""
        self.threshold = float(threshold)

    def step(self, batch_size: int = 1) -> List[Dict[str, Any]]:
        """Processes the next `batch_size` transactions sequentially without blocking loops.

        Args:
            batch_size: Number of transactions to ingest in this tick (default: 1).

        Returns:
            List of processed transaction results.
        """
        if len(self.data_source) == 0:
            return []

        results: List[Dict[str, Any]] = []
        end_idx = min(self.current_index + batch_size, len(self.data_source))

        for idx in range(self.current_index, end_idx):
            row = self.data_source.iloc[idx]
            tx_id = str(row.get("Transaction_ID", f"TXN-STRM-{idx+10001:05d}"))
            amt = float(row.get(settings.AMOUNT_COL, 0.0))
            time_val = float(row.get(settings.TIME_COL, 0.0))

            # Run single transaction inference
            if self.model and self.model.is_trained:
                try:
                    inf_res: InferenceResult = self.model.predict_single(
                        row, transaction_id=tx_id, threshold=self.threshold
                    )
                    score = inf_res.anomaly_score
                    is_anom = inf_res.is_anomaly
                    pred = inf_res.prediction_label
                    tier = inf_res.risk_category
                    top_feats = inf_res.top_contributing_features
                    latency = inf_res.inference_latency_ms
                    model_id = inf_res.model_id
                except Exception as exc:
                    logger.warning(f"Stream model inference error ({exc}); using fallback.")
                    score = float(calculate_statistical_risk_score(row))
                    is_anom = score >= self.threshold
                    pred = "ANOMALY" if is_anom else "NORMAL"
                    tier = classify_risk_tier(score)
                    top_feats = {"Amount": amt}
                    latency = 1.0
                    model_id = "BaselineFallback"
            else:
                score = float(calculate_statistical_risk_score(row))
                is_anom = score >= self.threshold
                pred = "ANOMALY" if is_anom else "NORMAL"
                tier = classify_risk_tier(score)
                top_feats = {"Amount": amt}
                latency = 1.0
                model_id = "Heuristic-Rule-Engine"

            record = {
                "stream_index": idx + 1,
                "transaction_id": tx_id,
                "time": time_val,
                "amount": amt,
                "anomaly_score": score,
                "is_anomaly": is_anom,
                "prediction": pred,
                "risk_tier": tier,
                "threshold": self.threshold,
                "model_id": model_id,
                "latency_ms": latency,
                "top_features": top_feats,
                "_is_demo": True
            }

            self.streamed_records.append(record)
            results.append(record)

            if is_anom:
                alert_item = {
                    "alert_id": f"ALT-STRM-{len(self.alerts_generated)+1:04d}",
                    "transaction_id": tx_id,
                    "anomaly_score": score,
                    "applied_threshold": self.threshold,
                    "prediction": pred,
                    "severity": tier,
                    "amount": amt,
                    "time": time_val,
                    "investigation_status": "NEW",
                    "top_features": top_feats,
                    "stream_index": idx + 1
                }
                self.alerts_generated.append(alert_item)

        self.current_index = end_idx
        return results

    def reset(self) -> None:
        """Resets the stream to initial position and clears telemetry buffers."""
        self.current_index = 0
        self.is_paused = True
        self.streamed_records.clear()
        self.alerts_generated.clear()
        logger.info("Reset simulated transaction stream.")

    def get_metrics(self) -> Dict[str, Any]:
        """Returns real-time stream velocity, risk rate, and buffer metrics."""
        total = len(self.streamed_records)
        alerts = len(self.alerts_generated)
        avg_score = (
            float(np.mean([r["anomaly_score"] for r in self.streamed_records]))
            if total > 0 else 0.0
        )
        alert_rate = (alerts / total) if total > 0 else 0.0
        buffer_total = len(self.data_source)

        return {
            "total_streamed": total,
            "total_alerts": alerts,
            "alert_rate": round(alert_rate, 4),
            "average_score": round(avg_score, 4),
            "current_index": self.current_index,
            "buffer_total": buffer_total,
            "progress_pct": round((self.current_index / buffer_total) * 100, 1) if buffer_total > 0 else 0.0,
            "is_exhausted": self.current_index >= buffer_total,
            "is_demo_stream": True,
            "disclaimer": self.STREAM_DISCLAIMER
        }

    def get_stream_dataframe(self, max_recent: int = 200) -> pd.DataFrame:
        """Returns recent streamed transactions as a pandas DataFrame."""
        if not self.streamed_records:
            return pd.DataFrame(columns=[
                "stream_index", "transaction_id", "time", "amount",
                "anomaly_score", "prediction", "risk_tier"
            ])
        recent = self.streamed_records[-max_recent:]
        return pd.DataFrame(recent)
