"""Tests for Dataset Loader."""

import pytest
import io
import pandas as pd
from pathlib import Path
from src.data.loader import (
    load_csv_dataset,
    check_benchmark_dataset_exists,
    get_dataset_provenance,
    DatasetNotFoundError
)


def test_check_benchmark_dataset_exists():
    exists, path = check_benchmark_dataset_exists()
    assert isinstance(exists, bool)
    assert isinstance(path, Path)


def test_load_csv_dataset_not_found():
    with pytest.raises(DatasetNotFoundError):
        load_csv_dataset("non_existent_file_path_12345.csv")


def test_load_csv_dataset_buffer():
    csv_data = "Time,Amount,V1,Class\n0,100.5,0.5,0\n1,250.0,-1.2,1\n"
    buffer = io.StringIO(csv_data)
    df, report = load_csv_dataset(buffer, validate=True)

    assert len(df) == 2
    assert "Amount" in df.columns
    assert report is not None
    assert report.total_rows == 2


def test_dataset_provenance():
    prov = get_dataset_provenance()
    assert "Credit Card" in prov["dataset_name"]
    assert "ULB" in prov["institution"]
    assert "DbCL" in prov["license"]