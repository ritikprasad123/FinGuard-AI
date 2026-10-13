"""Data quality validation, schema auditing, and statistical profiling engine."""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger("finguard.data.validator")


@dataclass
class DataQualityReport:
    """Comprehensive data quality audit report for transaction datasets."""
    is_valid: bool
    total_rows: int
    total_columns: int
    columns_present: List[str]
    missing_required_columns: List[str]
    missing_values_count: int
    missing_values_by_column: Dict[str, int]
    missing_values_pct: float
    duplicate_rows_count: int
    duplicate_rows_pct: float
    has_target_column: bool
    target_column_name: Optional[str]
    class_distribution: Dict[str, int]
    fraud_rate_pct: Optional[float]
    imbalance_ratio: Optional[float]
    negative_amount_count: int
    min_amount: Optional[float]
    max_amount: Optional[float]
    mean_amount: Optional[float]
    median_amount: Optional[float]
    non_numeric_columns: List[str]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checks_summary: Dict[str, str] = field(default_factory=dict)

    @property
    def checks(self) -> List[Any]:
        """Helper returning list of structured check results for dashboard rendering."""
        res = []
        descriptions = {
            "row_count": ("Record Ingestion Check", f"Validated {self.total_rows:,} records."),
            "schema_columns": ("Schema Conformance Check", f"Verified required benchmark features. Missing: {len(self.missing_required_columns)}"),
            "missing_values": ("Completeness & Null Check", f"{self.missing_values_count:,} missing values ({self.missing_values_pct:.2f}%)."),
            "duplicates": ("Deduplication Check", f"{self.duplicate_rows_count:,} duplicate records ({self.duplicate_rows_pct:.2f}%)."),
            "numeric_types": ("Feature Data Types Check", f"Non-numeric columns: {len(self.non_numeric_columns)}"),
            "amount_integrity": ("Financial Amount Integrity", f"Negative amounts: {self.negative_amount_count}"),
            "ground_truth": ("Ground Truth Reconciliation", f"Target verified: {self.has_target_column} ({'Fraud rate: ' + str(self.fraud_rate_pct) + '%' if self.fraud_rate_pct is not None else 'Unsupervised'})")
        }
        for k, status in self.checks_summary.items():
            name, details = descriptions.get(k, (k.replace('_', ' ').title(), f"Status: {status}"))
            res.append(type("CheckItem", (), {
                "check_name": name,
                "passed": status in ["PASS", "INFO"],
                "status": status,
                "details": details
            })())
        return res

    def to_dict(self) -> Dict[str, Any]:
        """Converts report to dictionary representation."""
        return asdict(self)


def validate_transaction_data(
    df: pd.DataFrame,
    expected_features: Optional[List[str]] = None,
    target_col: Optional[str] = None
) -> DataQualityReport:
    """Performs rigorous quality audit and schema validation on transaction dataset.

    Args:
        df: Input pandas DataFrame containing transaction records.
        expected_features: List of required features. Defaults to settings.REQUIRED_BENCHMARK_FEATURES.
        target_col: Optional ground truth column. Defaults to settings.TARGET_COL.

    Returns:
        DataQualityReport instance with complete statistical findings and validation status.
    """
    if expected_features is None:
        expected_features = settings.REQUIRED_BENCHMARK_FEATURES

    if target_col is None:
        target_col = settings.TARGET_COL

    errors: List[str] = []
    warnings: List[str] = []
    checks: Dict[str, str] = {}

    total_rows = len(df)
    total_cols = len(df.columns)
    columns_present = list(df.columns)

    logger.info(f"Initiating validation for dataframe with {total_rows} rows and {total_cols} columns.")

    # 1. Empty Check
    if total_rows == 0:
        errors.append("Dataset is empty. Zero records were ingested.")
        checks["row_count"] = "FAIL"
        return DataQualityReport(
            is_valid=False,
            total_rows=0,
            total_columns=total_cols,
            columns_present=columns_present,
            missing_required_columns=expected_features,
            missing_values_count=0,
            missing_values_by_column={},
            missing_values_pct=0.0,
            duplicate_rows_count=0,
            duplicate_rows_pct=0.0,
            has_target_column=False,
            target_column_name=target_col,
            class_distribution={},
            fraud_rate_pct=None,
            imbalance_ratio=None,
            negative_amount_count=0,
            min_amount=None,
            max_amount=None,
            mean_amount=None,
            median_amount=None,
            non_numeric_columns=[],
            errors=errors,
            warnings=warnings,
            checks_summary=checks
        )

    checks["row_count"] = "PASS"

    # 2. Schema Verification (Required Columns)
    missing_required = [col for col in expected_features if col not in df.columns]
    if missing_required:
        errors.append(f"Missing {len(missing_required)} required benchmark feature(s): {', '.join(missing_required[:5])}" +
                      (f" and {len(missing_required)-5} more" if len(missing_required) > 5 else ""))
        checks["schema_columns"] = "FAIL"
    else:
        checks["schema_columns"] = "PASS"

    # 3. Missing Value Profiling
    missing_by_col = df.isnull().sum().to_dict()
    missing_values_count = int(df.isnull().sum().sum())
    total_cells = total_rows * total_cols
    missing_pct = round((missing_values_count / total_cells) * 100, 4) if total_cells > 0 else 0.0

    if missing_values_count > 0:
        warnings.append(f"Found {missing_values_count} missing value(s) ({missing_pct:.2f}% of cells) across {sum(1 for v in missing_by_col.values() if v > 0)} columns.")
        checks["missing_values"] = "WARN"
    else:
        checks["missing_values"] = "PASS"

    # 4. Duplicate Record Analysis
    duplicate_count = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_count / total_rows) * 100, 3) if total_rows > 0 else 0.0

    if duplicate_count > 0:
        warnings.append(f"Detected {duplicate_count} duplicate row(s) ({duplicate_pct:.2f}% of dataset).")
        checks["duplicates"] = "WARN"
    else:
        checks["duplicates"] = "PASS"

    # 5. Numerical Type Validation
    non_numeric_cols = []
    for col in expected_features:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric_cols.append(col)

    if non_numeric_cols:
        errors.append(f"Expected numerical types but found non-numeric data in: {', '.join(non_numeric_cols)}")
        checks["numeric_types"] = "FAIL"
    else:
        checks["numeric_types"] = "PASS"

    # 6. Amount Integrity Check
    negative_amount_count = 0
    min_amount, max_amount, mean_amount, median_amount = None, None, None, None

    if settings.AMOUNT_COL in df.columns and pd.api.types.is_numeric_dtype(df[settings.AMOUNT_COL]):
        negative_amount_count = int((df[settings.AMOUNT_COL] < 0).sum())
        if negative_amount_count > 0:
            errors.append(f"Financial integrity failure: Found {negative_amount_count} record(s) with negative Amount.")
            checks["amount_integrity"] = "FAIL"
        else:
            checks["amount_integrity"] = "PASS"

        min_amount = float(df[settings.AMOUNT_COL].min())
        max_amount = float(df[settings.AMOUNT_COL].max())
        mean_amount = float(df[settings.AMOUNT_COL].mean())
        median_amount = float(df[settings.AMOUNT_COL].median())
    else:
        checks["amount_integrity"] = "WARN"

    # 7. Class Distribution & Target Profiling
    has_target = target_col in df.columns
    class_dist: Dict[str, int] = {}
    fraud_rate_pct = None
    imbalance_ratio = None

    if has_target:
        value_counts = df[target_col].value_counts().to_dict()
        class_dist = {str(k): int(v) for k, v in value_counts.items()}
        checks["ground_truth"] = "PASS"

        # Check binary 0 / 1 distribution
        if 1 in value_counts or "1" in class_dist:
            fraud_count = int(value_counts.get(1, value_counts.get("1", 0)))
            normal_count = int(value_counts.get(0, value_counts.get("0", 0)))
            fraud_rate_pct = round((fraud_count / total_rows) * 100, 4) if total_rows > 0 else 0.0

            if fraud_count > 0:
                imbalance_ratio = round(normal_count / fraud_count, 2)
            else:
                imbalance_ratio = float("inf")
    else:
        checks["ground_truth"] = "INFO"
        warnings.append(f"Ground truth target column '{target_col}' not present. Unsupervised mode active.")

    is_valid = len(errors) == 0

    return DataQualityReport(
        is_valid=is_valid,
        total_rows=total_rows,
        total_columns=total_cols,
        columns_present=columns_present,
        missing_required_columns=missing_required,
        missing_values_count=missing_values_count,
        missing_values_by_column=missing_by_col,
        missing_values_pct=missing_pct,
        duplicate_rows_count=duplicate_count,
        duplicate_rows_pct=duplicate_pct,
        has_target_column=has_target,
        target_column_name=target_col if has_target else None,
        class_distribution=class_dist,
        fraud_rate_pct=fraud_rate_pct,
        imbalance_ratio=imbalance_ratio,
        negative_amount_count=negative_amount_count,
        min_amount=min_amount,
        max_amount=max_amount,
        mean_amount=mean_amount,
        median_amount=median_amount,
        non_numeric_columns=non_numeric_cols,
        errors=errors,
        warnings=warnings,
        checks_summary=checks
    )


# Backward-compatibility alias
ValidationReport = DataQualityReport
