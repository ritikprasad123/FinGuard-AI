"""Local Outlier Factor (LOF) Novelty Detection Model for FinGuard AI.

Supports novelty detection inference on unseen streaming transactions
using scikit-learn's LocalOutlierFactor with novelty=True and memory-safe
bounded sampling for enterprise-scale transaction volumes.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import joblib

from config.settings import settings
from src.models.base import BaseAnomalyModel, InferenceResult
from src.utils.logger import get_logger

logger = get_logger("finguard.models.lof_detector")


class LocalOutlierFactorAnomalyModel(BaseAnomalyModel):
    """Local Outlier Factor detector configured for novelty scoring on previously unseen data."""

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: Union[float, str] = 0.02,
        metric: str = "minkowski",
        default_threshold: float = 0.65,
        max_train_samples: int = 5000,
        random_state: int = 42
    ):
        """Initializes LOF Novelty Model.

        Args:
            n_neighbors: Number of neighbors for local density estimation.
            contamination: Expected proportion of outliers in the data.
            metric: Distance metric for nearest neighbors calculation.
            default_threshold: Normalized anomaly score cutoff [0.0, 1.0].
            max_train_samples: Upper limit on training set size to avoid O(N^2) memory blowout.
            random_state: Seed for reproducible downsampling.
        """
        super().__init__(
            model_name="Local Outlier Factor (LOF)",
            model_id=f"LOF-k{n_neighbors}-Novelty",
            default_threshold=default_threshold
        )
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self.max_train_samples = max_train_samples
        self.random_state = random_state

        self.model: Optional[LocalOutlierFactor] = None
        self.score_offset_: float = 0.0
        self.score_scale_: float = 1.0

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "LocalOutlierFactorAnomalyModel":
        """Fits LOF novelty detector on training features with safe sampling if needed.

        Args:
            X: Training DataFrame containing transaction features.
            y: Optional ground-truth labels (strictly unused to prevent data leakage).

        Returns:
            self with fitted model and calibrated score parameters.
        """
        if len(X) == 0:
            raise ValueError("Cannot fit Local Outlier Factor on an empty dataset.")

        exclude_cols = [
            "Class", "Transaction_ID", "Timestamp", "_DATA_SOURCE", "_IS_DEMO",
            "Risk_Score", "Risk_Reason", "Risk_Tier", "Action_Required", "Anomaly_Indicators",
            "Prediction", "Severity", "anomaly_score", "investigation_status", "alert_id"
        ]
        num_df = X.select_dtypes(include=[np.number])
        feature_cols = [c for c in num_df.columns if c not in exclude_cols]
        if not feature_cols:
            raise ValueError("No valid numeric feature columns available for LOF training.")

        self.feature_names_in_ = feature_cols

        # Memory-safe downsampling for nearest neighbor matrix to protect system RAM
        if len(X) > self.max_train_samples:
            logger.info(
                f"Dataset size ({len(X)}) exceeds LOF memory-safe threshold ({self.max_train_samples}). "
                f"Sampling {self.max_train_samples} transactions for nearest-neighbor reference index."
            )
            sampled_df = num_df.sample(n=self.max_train_samples, random_state=self.random_state)
            X_train = sampled_df[feature_cols].copy()
        else:
            X_train = num_df[feature_cols].copy()

        # Fill any missing values with median for robust distance computation
        X_train = X_train.fillna(X_train.median())

        effective_k = min(self.n_neighbors, len(X_train) - 1)
        if effective_k < 2:
            effective_k = 2

        logger.info(f"Fitting LocalOutlierFactor(novelty=True, n_neighbors={effective_k}) on {len(X_train)} samples.")

        self.model = LocalOutlierFactor(
            n_neighbors=effective_k,
            novelty=True,  # Crucial: enables decision_function and predict on unseen data
            contamination=self.contamination,
            metric=self.metric,
            n_jobs=-1
        )
        self.model.fit(X_train.values)

        # Calibrate score distribution: decision_function produces positive scores for inliers, negative for outliers
        train_raw_scores = self.model.decision_function(X_train.values)
        p5 = float(np.percentile(train_raw_scores, 5))
        p95 = float(np.percentile(train_raw_scores, 95))
        
        # Raw scores: smaller (more negative) = more anomalous.
        # Invert so higher = more anomalous
        self.score_offset_ = p95
        self.score_scale_ = max(abs(p95 - p5), 1e-4)

        self.is_trained = True
        logger.info(f"Fitted LOF novelty model successfully. Reference scale: {self.score_scale_:.4f}")
        return self

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """Computes continuous anomaly score scaled to [0.0, 1.0].

        Higher score = higher anomaly likelihood.
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError(f"Model {self.model_id} must be fitted prior to scoring.")

        if len(X) == 0:
            return np.array([], dtype=float)

        missing_cols = [c for c in self.feature_names_in_ if c not in X.columns]
        if missing_cols:
            raise ValueError(f"Input data is missing required feature columns: {missing_cols}")

        X_input = X[self.feature_names_in_].copy()
        X_input = X_input.fillna(X_input.median())

        # In scikit-learn LOF with novelty=True:
        # decision_function returns negative values for outliers, positive for normal instances.
        raw_decisions = self.model.decision_function(X_input.values)

        # Invert: higher raw_decisions means normal -> invert so higher means anomalous
        normalized_scores = (self.score_offset_ - raw_decisions) / self.score_scale_
        # Bound score in [0.0, 1.0]
        return np.clip(normalized_scores, 0.0, 1.0)

    def save(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Persists trained LOF model artifact to disk."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Cannot persist an untrained Local Outlier Factor model.")

        if filepath is None:
            filepath = settings.MODELS_DIR / "lof_detector_default.joblib"
        else:
            filepath = Path(filepath)

        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "threshold": self.threshold,
            "feature_names_in_": self.feature_names_in_,
            "n_neighbors": self.n_neighbors,
            "contamination": self.contamination,
            "metric": self.metric,
            "max_train_samples": self.max_train_samples,
            "score_offset_": self.score_offset_,
            "score_scale_": self.score_scale_,
            "is_trained": self.is_trained,
            "model_obj": self.model
        }
        joblib.dump(payload, filepath)
        logger.info(f"Persisted LOF model artifact to: {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "LocalOutlierFactorAnomalyModel":
        """Loads serialized LOF model artifact from disk."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"LOF model artifact not found at: {path}")

        payload = joblib.load(path)
        instance = cls(
            n_neighbors=payload.get("n_neighbors", 20),
            contamination=payload.get("contamination", 0.02),
            metric=payload.get("metric", "minkowski"),
            default_threshold=payload.get("threshold", 0.65),
            max_train_samples=payload.get("max_train_samples", 5000)
        )
        instance.model_name = payload.get("model_name", "Local Outlier Factor (LOF)")
        instance.model_id = payload.get("model_id", "LOF-Novelty")
        instance.threshold = payload.get("threshold", 0.65)
        instance.feature_names_in_ = payload.get("feature_names_in_", [])
        instance.score_offset_ = payload.get("score_offset_", 0.0)
        instance.score_scale_ = payload.get("score_scale_", 1.0)
        instance.is_trained = payload.get("is_trained", False)
        instance.model = payload.get("model_obj", None)
        logger.info(f"Loaded LOF model artifact from: {path} (Features: {len(instance.feature_names_in_)})")
        return instance

