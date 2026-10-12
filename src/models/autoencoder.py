"""Neural Autoencoder Reconstruction Anomaly Detection Model for FinGuard AI."""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
import joblib
from config.settings import settings
from src.models.base import BaseAnomalyModel, InferenceResult
from src.utils.logger import get_logger

logger = get_logger("finguard.models.autoencoder")


class AutoencoderAnomalyModel(BaseAnomalyModel):
    """Neural bottleneck reconstruction autoencoder for financial transaction anomaly detection."""

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (16, 8, 16),
        max_iter: int = 150,
        random_state: int = 42,
        default_threshold: float = 0.65,
        early_stopping: bool = True
    ):
        super().__init__(
            model_name="Neural Autoencoder",
            model_id=f"Autoencoder-MLP-{'-'.join(map(str, hidden_layer_sizes))}",
            default_threshold=default_threshold
        )
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self.early_stopping = early_stopping

        self.model: Optional[MLPRegressor] = None
        self.error_p5_: float = 0.0
        self.error_p99_: float = 1.0

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "AutoencoderAnomalyModel":
        """Fits neural autoencoder to reconstruct its own input features (X -> X)."""
        if len(X) == 0:
            raise ValueError("Cannot fit Autoencoder on an empty dataset.")

        exclude_cols = [
            "Class", "Transaction_ID", "Timestamp", "_DATA_SOURCE", "_IS_DEMO",
            "Risk_Score", "Risk_Reason", "Risk_Tier", "Action_Required", "Anomaly_Indicators"
        ]
        feature_cols = [c for c in X.columns if c not in exclude_cols]
        self.feature_names_in_ = feature_cols

        X_train = X[feature_cols].copy()
        logger.info(f"Training Autoencoder on {len(X_train)} records (architecture={self.hidden_layer_sizes}).")

        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=self.early_stopping,
            n_iter_no_change=10,
            batch_size="auto"
        )

        self.model.fit(X_train, X_train)

        X_pred = self.model.predict(X_train)
        reconstruction_errors = np.mean((X_train.values - X_pred) ** 2, axis=1)

        self.error_p5_ = float(np.percentile(reconstruction_errors, 5))
        self.error_p99_ = float(np.percentile(reconstruction_errors, 99))
        if self.error_p99_ <= self.error_p5_:
            self.error_p99_ = self.error_p5_ + 1.0

        self.is_trained = True
        return self

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """Computes reconstruction error anomaly score scaled to [0.0, 1.0]."""
        if not self.is_trained or self.model is None:
            raise RuntimeError(f"Model {self.model_id} must be fitted prior to scoring.")

        if len(X) == 0:
            return np.array([])

        missing_cols = [c for c in self.feature_names_in_ if c not in X.columns]
        if missing_cols:
            raise ValueError(f"Input is missing required model columns: {missing_cols}")

        X_test = X[self.feature_names_in_].copy()
        X_pred = self.model.predict(X_test)
        errors = np.mean((X_test.values - X_pred) ** 2, axis=1)

        norm_scores = (errors - self.error_p5_) / (self.error_p99_ - self.error_p5_)
        return np.clip(norm_scores, 0.0, 1.0)

    def save(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Persists trained model artifact to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained model.")

        save_path = Path(filepath) if filepath else settings.MODELS_DIR / f"{self.model_id}.joblib"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, save_path)
        logger.info(f"Saved Autoencoder to: {save_path}")
        return save_path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "AutoencoderAnomalyModel":
        """Loads persisted model artifact from disk."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        return joblib.load(path)
