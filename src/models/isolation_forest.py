"""Isolation Forest Anomaly Detection Model for FinGuard AI."""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
from config.settings import settings
from src.models.base import BaseAnomalyModel, InferenceResult
from src.utils.logger import get_logger

logger = get_logger("finguard.models.isolation_forest")


class IsolationForestAnomalyModel(BaseAnomalyModel):
    """Isolation Forest anomaly detection wrapper with calibrated [0, 1] risk scoring."""

    def __init__(
        self,
        n_estimators: int = 150,
        contamination: float = 0.01,
        random_state: int = 42,
        default_threshold: float = 0.65
    ):
        super().__init__(
            model_name="Isolation Forest",
            model_id=f"IsoForest-{n_estimators}-trees",
            default_threshold=default_threshold
        )
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state

        self.model: Optional[IsolationForest] = None
        self.score_min_: float = -1.0
        self.score_max_: float = 0.5

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "IsolationForestAnomalyModel":
        """Fits Isolation Forest on features, strictly excluding target labels."""
        if len(X) == 0:
            raise ValueError("Cannot fit model on an empty dataset.")

        exclude_cols = ["Class", "Transaction_ID", "Timestamp", "_DATA_SOURCE", "_IS_DEMO", "Risk_Score", "Risk_Reason"]
        feature_cols = [c for c in X.columns if c not in exclude_cols]
        self.feature_names_in_ = feature_cols

        X_train = X[feature_cols].copy()
        logger.info(f"Training Isolation Forest with {self.n_estimators} estimators on {len(X_train)} samples.")

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X_train)

        raw_scores = -self.model.score_samples(X_train)
        self.score_min_ = float(np.min(raw_scores))
        self.score_max_ = float(np.max(raw_scores))
        if self.score_max_ <= self.score_min_:
            self.score_max_ = self.score_min_ + 1.0

        self.is_trained = True
        return self

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """Returns continuous anomaly scores calibrated between 0.0 and 1.0."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be fitted prior to scoring.")

        missing_cols = [c for c in self.feature_names_in_ if c not in X.columns]
        if missing_cols:
            raise ValueError(f"Input is missing required model columns: {missing_cols}")

        X_test = X[self.feature_names_in_].copy()
        raw_scores = -self.model.score_samples(X_test)
        norm_scores = (raw_scores - self.score_min_) / (self.score_max_ - self.score_min_)
        return np.clip(norm_scores, 0.0, 1.0)

    def save(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Persists trained model artifact to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained model.")

        save_path = Path(filepath) if filepath else settings.MODELS_DIR / f"{self.model_id}.joblib"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, save_path)
        logger.info(f"Saved Isolation Forest to: {save_path}")
        return save_path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "IsolationForestAnomalyModel":
        """Loads persisted model artifact from disk."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        return joblib.load(path)
