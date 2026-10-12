"""Data ingestion, validation, and demo generator package."""
from src.data.validator import validate_transaction_data, DataQualityReport
from src.data.loader import load_csv_dataset, check_benchmark_dataset_exists, get_dataset_provenance
from src.data.demo_generator import generate_synthetic_demo_data

__all__ = [
 "validate_transaction_data",
 "DataQualityReport",
 "load_csv_dataset",
 "check_benchmark_dataset_exists",
 "get_dataset_provenance",
 "generate_synthetic_demo_data"
]