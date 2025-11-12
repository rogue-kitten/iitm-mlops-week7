import pytest
import pandas as pd
import os

IRIS_CSV_PATH = "data/iris.csv"
EXPECTED_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species",
]
EXPECTED_ROW_COUNT = 45


@pytest.fixture
def iris_dataframe():
    """Fixture to load iris dataset"""
    return pd.read_csv(IRIS_CSV_PATH)


def test_data_file_exists():
    """Verify IRIS data file exists"""
    assert os.path.exists(IRIS_CSV_PATH), f"Data file not found at {IRIS_CSV_PATH}"


def test_data_loading(iris_dataframe):
    """Verify IRIS data loads correctly"""
    assert iris_dataframe is not None
    assert len(iris_dataframe) > 0


def test_data_shape(iris_dataframe):
    """Verify data has expected number of rows"""
    assert len(iris_dataframe) == EXPECTED_ROW_COUNT, (
        f"Expected {EXPECTED_ROW_COUNT} rows, got {len(iris_dataframe)}"
    )


def test_required_columns(iris_dataframe):
    """Verify all required columns are present"""
    for col in EXPECTED_COLUMNS:
        assert col in iris_dataframe.columns, f"Required column '{col}' not found"


def test_no_missing_values(iris_dataframe):
    """Verify no missing values in dataset"""
    assert iris_dataframe.isnull().sum().sum() == 0, "Dataset contains missing values"


def test_data_types(iris_dataframe):
    """Verify feature columns have correct data types"""
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    for col in feature_cols:
        assert pd.api.types.is_numeric_dtype(iris_dataframe[col]), (
            f"Column '{col}' should be numeric"
        )
