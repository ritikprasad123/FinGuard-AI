"""Dataset loader and benchmark provenance engine for FinGuard AI.

Safely loads CSV files, verifies existence, executes quality audits, and provides
verified benchmark dataset provenance without fabricating missing records.
"""

from pathlib import Path
from typing import Tuple, Optional, Union, Dict, Any
import io
import pandas as pd
from config.settings import settings
from src.data.validator import validate_transaction_data, DataQualityReport
from src.utils.logger import get_logger

logger = get_logger("finguard.data.loader")


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the requested financial dataset does not exist."""
    pass


class DatasetValidationError(ValueError):
    """Raised when the dataset fails critical financial schema requirements."""
    pass


def check_benchmark_dataset_exists(custom_path: Optional[Union[str, Path]] = None) -> Tuple[bool, Path]:
    """Checks whether the official benchmark credit card dataset is present on disk.

    Args:
        custom_path: Optional custom path. If None, checks settings.DEFAULT_BENCHMARK_PATH.

    Returns:
        Tuple of (exists: bool, path: Path)
    """
    path = Path(custom_path) if custom_path else settings.DEFAULT_BENCHMARK_PATH
    exists = path.is_file() and path.stat().st_size > 0
    return exists, path


def load_csv_dataset(
    source: Union[str, Path, io.BytesIO, io.StringIO],
    nrows: Optional[int] = None,
    validate: bool = True
) -> Tuple[pd.DataFrame, Optional[DataQualityReport]]:
    """Loads a transaction dataset from a file path or file-like buffer.

    Args:
        source: File path or in-memory buffer.
        nrows: Optional row limit for preview or quick testing.
        validate: Whether to immediately run data quality validation.

    Returns:
        Tuple of (loaded DataFrame, DataQualityReport or None)

    Raises:
        DatasetNotFoundError: If the specified file path does not exist.
        ValueError: If loading fails due to malformed data.
    """
    if isinstance(source, (str, Path)):
        file_path = Path(source)
        if not file_path.is_file():
            logger.error(f"Dataset file not found at: {file_path}")
            raise DatasetNotFoundError(
                f"Dataset file not found at '{file_path}'. "
                f"Please ensure '{settings.BENCHMARK_FILENAME}' is placed in '{settings.DATA_RAW_DIR}' "
                "or use the FinGuard Setup Guide to import verified data."
            )
        logger.info(f"Loading transaction dataset from path: {file_path}")
        df = pd.read_csv(file_path, nrows=nrows)
    else:
        logger.info("Loading transaction dataset from in-memory buffer / file upload.")
        df = pd.read_csv(source, nrows=nrows)

    report = None
    if validate:
        report = validate_transaction_data(df)
        logger.info(f"Loaded dataset: {len(df)} rows. Validation status: is_valid={report.is_valid}")

    return df, report


def get_dataset_provenance() -> Dict[str, Any]:
    """Returns verified metadata and provenance for the public credit card benchmark dataset."""
    return {
        "dataset_name": "Credit Card Fraud Detection Benchmark",
        "creators": "Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, Gianluca Bontempi",
        "authors": "Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, Gianluca Bontempi",
        "institution": "Machine Learning Group (MLG) - Universite Libre de Bruxelles (ULB)",
        "published_in": "IEEE Symposium Series on Computational Intelligence (SSCI), 2015",
        "publication": "IEEE Symposium Series on Computational Intelligence (SSCI), 2015",
        "license": "Database Contents License (DbCL) v1.0 / Open Database License (ODbL)",
        "kaggle_ref": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
        "source_url": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
        "kaggle_slug": "mlg-ulb/creditcardfraud",
        "kaggle_cli_command": "kaggle datasets download -d mlg-ulb/creditcardfraud",
        "expected_columns": f"{len(settings.ALL_BENCHMARK_COLUMNS)} columns (Time, Amount, V1-V28, Class)",
        "expected_target": "Class (0 = Normal transaction, 1 = Fraudulent transaction)",
        "expected_shape": "284,807 transactions across 2 days (492 frauds, ~0.172% fraud rate)",
        "expected_records": 284807,
        "expected_fraud_records": 492,
        "expected_fraud_rate": "0.172%",
        "expected_path": str(settings.DEFAULT_BENCHMARK_PATH),
        "data_anonymization": "Features V1-V28 are PCA transformed due to customer confidentiality; Time and Amount are non-transformed."
    }

