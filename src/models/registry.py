"""Model registry and active inference engine manager for FinGuard AI."""

from typing import Optional, Dict, Any, List
from pathlib import Path
from config.settings import settings
from src.models.base import BaseAnomalyModel
from src.utils.logger import get_logger

logger = get_logger("finguard.models.registry")

_ACTIVE_MODEL: Optional[BaseAnomalyModel] = None


def get_active_model() -> Optional[BaseAnomalyModel]:
    """Returns the globally active inference model, auto-loading default artifact if present."""
    global _ACTIVE_MODEL
    if _ACTIVE_MODEL is None:
        if_path = settings.MODELS_DIR / "isolation_forest_default.joblib"
        if if_path.is_file():
            try:
                from src.models.isolation_forest import IsolationForestAnomalyModel
                _ACTIVE_MODEL = IsolationForestAnomalyModel.load(if_path)
                logger.info(f"Auto-loaded default Isolation Forest model from: {if_path}")
            except Exception as exc:
                logger.warning(f"Could not auto-load default Isolation Forest artifact: {exc}")
    return _ACTIVE_MODEL


def set_active_model(model: BaseAnomalyModel) -> None:
    """Sets the globally active model for surveillance stream inference."""
    global _ACTIVE_MODEL
    _ACTIVE_MODEL = model
    logger.info(f"Registered active surveillance model: {model.model_name} (ID: {model.model_id})")


def list_available_models() -> List[BaseAnomalyModel]:
    """Returns list of instantiated or loaded available models."""
    models: List[BaseAnomalyModel] = []

    # 1. Load Isolation Forest if artifact exists
    if_path = settings.MODELS_DIR / "isolation_forest_default.joblib"
    if if_path.is_file():
        try:
            from src.models.isolation_forest import IsolationForestAnomalyModel
            models.append(IsolationForestAnomalyModel.load(if_path))
        except Exception as e:
            logger.warning(f"Error loading Isolation Forest: {e}")

    # 2. Load Local Outlier Factor if artifact exists
    lof_path = settings.MODELS_DIR / "lof_detector_default.joblib"
    if lof_path.is_file():
        try:
            from src.models.lof_detector import LocalOutlierFactorAnomalyModel
            models.append(LocalOutlierFactorAnomalyModel.load(lof_path))
        except Exception as e:
            logger.warning(f"Error loading LOF artifact: {e}")

    # 3. Load Autoencoder if artifact exists
    ae_path = settings.MODELS_DIR / "autoencoder_default.joblib"
    if ae_path.is_file():
        try:
            from src.models.autoencoder import AutoencoderAnomalyModel
            models.append(AutoencoderAnomalyModel.load(ae_path))
        except Exception as e:
            logger.warning(f"Error loading Autoencoder: {e}")

    # 4. If any model is not loaded from disk, provide default fresh instantiations
    existing_names = {m.model_name for m in models}
    
    from src.models.isolation_forest import IsolationForestAnomalyModel
    from src.models.lof_detector import LocalOutlierFactorAnomalyModel
    from src.models.autoencoder import AutoencoderAnomalyModel

    if "Isolation Forest" not in existing_names:
        models.append(IsolationForestAnomalyModel())
    if "Local Outlier Factor (LOF)" not in existing_names:
        models.append(LocalOutlierFactorAnomalyModel())
    if "Neural Autoencoder" not in existing_names:
        models.append(AutoencoderAnomalyModel())

    return models


def get_active_model_metadata() -> Dict[str, Any]:
    """Returns metadata dictionary for the currently active inference model."""
    m = get_active_model()
    if m is None:
        return {
            "is_active": False,
            "model_name": "None (Awaiting Model)",
            "model_id": "None",
            "threshold": 0.65,
            "is_trained": False,
            "feature_count": 0,
            "framework": "Scikit-Learn / NumPy"
        }

    return {
        "is_active": True,
        "model_name": m.model_name,
        "model_id": m.model_id,
        "threshold": m.threshold,
        "is_trained": m.is_trained,
        "feature_count": len(m.feature_names_in_) if m.is_trained else 0,
        "framework": "Scikit-Learn / NumPy"
    }


def list_saved_models() -> List[Dict[str, Any]]:
    """Lists persisted model artifacts available in the models directory."""
    if not settings.MODELS_DIR.exists():
        return []

    results = []
    for f in settings.MODELS_DIR.glob("*.joblib"):
        if f.name == settings.PREPROCESSOR_FILENAME:
            continue
        results.append({
            "filename": f.name,
            "path": str(f),
            "size_kb": round(f.stat().st_size / 1024, 1),
            "modified": f.stat().st_mtime
        })
    return results
