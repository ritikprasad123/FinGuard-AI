"""Feature preprocessing, leak-free scaling, and train/test dataset partitioning."""

from pathlib import Path
from typing import Tuple, Optional, Union, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger("finguard.features.preprocessor")


class TransactionPreprocessor:
    """Production-grade feature transformer with leak-free scaling and serialization."""

    def __init__(
        self,
        scaler_type: str = "RobustScaler",
        scale_amount: bool = True,
        scale_time: bool = True,
        passthrough_pca: bool = True
    ):
        """Initializes preprocessor.

        Args:
            scaler_type: 'RobustScaler' or 'StandardScaler'. RobustScaler handles financial outliers.
            scale_amount: Whether to scale the transaction Amount column.
            scale_time: Whether to scale the elapsed Time column.
            passthrough_pca: Whether to include V1-V28 PCA features without altering variance.
        """
        self.scaler_type = scaler_type
        self.scale_amount = scale_amount
        self.scale_time = scale_time
        self.passthrough_pca = passthrough_pca

        self.amount_scaler: Optional[Union[RobustScaler, StandardScaler]] = None
        self.time_scaler: Optional[Union[RobustScaler, StandardScaler]] = None
        self.is_fitted: bool = False
        self.feature_names_in_: List[str] = []
        self.feature_names_out_: List[str] = []

    def _create_scaler(self) -> Union[RobustScaler, StandardScaler]:
        """Factory for selected scaler."""
        if self.scaler_type.lower() == "standardscaler":
            return StandardScaler()
        return RobustScaler()

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TransactionPreprocessor":
        """Fits scalers exclusively on the provided training features.

        Args:
            X: Input training DataFrame containing features.
            y: Ignored (to maintain scikit-learn API compatibility and prevent target leakage).

        Returns:
            self: The fitted preprocessor instance.
        """
        logger.info(f"Fitting TransactionPreprocessor on {len(X)} records (scaler_type={self.scaler_type}).")
        self.feature_names_in_ = list(X.columns)

        if self.scale_amount and settings.AMOUNT_COL in X.columns:
            self.amount_scaler = self._create_scaler()
            self.amount_scaler.fit(X[[settings.AMOUNT_COL]])

        if self.scale_time and settings.TIME_COL in X.columns:
            self.time_scaler = self._create_scaler()
            self.time_scaler.fit(X[[settings.TIME_COL]])

        out_names = []
        if self.scale_time and settings.TIME_COL in X.columns:
            out_names.append("Scaled_Time")
        elif settings.TIME_COL in X.columns:
            out_names.append(settings.TIME_COL)

        if self.passthrough_pca:
            pca_present = [col for col in settings.PCA_FEATURE_COLS if col in X.columns]
            out_names.extend(pca_present)

        if self.scale_amount and settings.AMOUNT_COL in X.columns:
            out_names.append("Scaled_Amount")
        elif settings.AMOUNT_COL in X.columns:
            out_names.append(settings.AMOUNT_COL)

        self.feature_names_out_ = out_names
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms transaction features using parameters fitted on the training split.

        Args:
            X: Input DataFrame to transform.

        Returns:
            Transformed pandas DataFrame with scaled features.

        Raises:
            RuntimeError: If transform is called prior to fitting.
        """
        if not self.is_fitted:
            raise RuntimeError("TransactionPreprocessor must be fitted before transforming data.")

        df_out = pd.DataFrame(index=X.index)

        # 1. Transform Time
        if self.scale_time and settings.TIME_COL in X.columns and self.time_scaler is not None:
            scaled_time = self.time_scaler.transform(X[[settings.TIME_COL]])
            df_out["Scaled_Time"] = scaled_time.flatten()
        elif settings.TIME_COL in X.columns:
            df_out[settings.TIME_COL] = X[settings.TIME_COL].values

        # 2. Passthrough PCA Features
        if self.passthrough_pca:
            pca_present = [col for col in settings.PCA_FEATURE_COLS if col in X.columns]
            for col in pca_present:
                df_out[col] = X[col].values

        # 3. Transform Amount
        if self.scale_amount and settings.AMOUNT_COL in X.columns and self.amount_scaler is not None:
            scaled_amt = self.amount_scaler.transform(X[[settings.AMOUNT_COL]])
            df_out["Scaled_Amount"] = scaled_amt.flatten()
        elif settings.AMOUNT_COL in X.columns:
            df_out[settings.AMOUNT_COL] = X[settings.AMOUNT_COL].values

        return df_out

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fits on training data and transforms it in a single call."""
        return self.fit(X, y).transform(X)

    def save(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Persists fitted preprocessor to disk using joblib.

        Args:
            filepath: Destination path. Defaults to settings.DEFAULT_PREPROCESSOR_PATH.

        Returns:
            Path where the preprocessor was saved.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted preprocessor.")

        save_path = Path(filepath) if filepath else settings.DEFAULT_PREPROCESSOR_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, save_path)
        logger.info(f"Persisted preprocessor artifact to: {save_path}")
        return save_path

    @classmethod
    def load(cls, filepath: Optional[Union[str, Path]] = None) -> "TransactionPreprocessor":
        """Loads a persisted preprocessor from disk.

        Args:
            filepath: Path to artifact. Defaults to settings.DEFAULT_PREPROCESSOR_PATH.

        Returns:
            Fitted TransactionPreprocessor instance.
        """
        load_path = Path(filepath) if filepath else settings.DEFAULT_PREPROCESSOR_PATH
        if not load_path.is_file():
            raise FileNotFoundError(f"Preprocessor artifact not found at: {load_path}")

        instance = joblib.load(load_path)
        logger.info(f"Loaded preprocessor artifact from: {load_path}")
        return instance

    def get_summary(self) -> Dict[str, Any]:
        """Returns metadata summary of the preprocessor configuration."""
        amount_stats = None
        if self.amount_scaler and hasattr(self.amount_scaler, "center_"):
            amount_stats = {
                "center": float(self.amount_scaler.center_[0]),
                "scale": float(self.amount_scaler.scale_[0])
            }

        return {
            "is_fitted": self.is_fitted,
            "scaler_type": self.scaler_type,
            "features_in_count": len(self.feature_names_in_),
            "features_out_count": len(self.feature_names_out_),
            "features_out": self.feature_names_out_,
            "amount_scaling_stats": amount_stats
        }


def split_transaction_data(
    df: pd.DataFrame,
    target_col: str = "Class",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
    """Performs stratified train/test split to maintain class ratio and prevent target leakage.

    Args:
        df: Input transaction DataFrame.
        target_col: Target column name. Defaults to settings.TARGET_COL.
        test_size: Fraction of samples reserved for test split.
        random_state: Deterministic random state seed.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test). If target_col is absent, y_train and y_test are None.
    """
    logger.info(f"Executing train/test split (test_size={test_size:.1%}, random_state={random_state}).")

    exclude_cols = [
        target_col, "Transaction_ID", "Timestamp", "_DATA_SOURCE", "_IS_DEMO",
        "Risk_Score", "Risk_Reason", "Risk_Tier", "Action_Required", "Anomaly_Indicators",
        "Prediction", "Severity", "anomaly_score", "investigation_status", "alert_id"
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].select_dtypes(include=[np.number]).copy()

    if target_col in df.columns:
        y = df[target_col].copy()
        class_counts = y.value_counts()
        stratify = y if (len(class_counts) > 1 and (class_counts >= 2).all()) else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
        logger.info(f"Split completed (stratified={stratify is not None}): Train={len(X_train)} (fraud={int(y_train.sum())}), "
                    f"Test={len(X_test)} (fraud={int(y_test.sum())})")
        return X_train, X_test, y_train, y_test
    else:
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
        logger.info(f"Unsupervised split completed: Train={len(X_train)}, Test={len(X_test)}")
        return X_train, X_test, None, None
