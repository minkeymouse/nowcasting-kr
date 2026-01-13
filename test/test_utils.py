"""Tests for utility functions in src/utils.py."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
import joblib
import pickle
import tempfile
import shutil

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import (
    load_model_checkpoint,
    preprocess_data_for_model,
    extract_forecast_values
)


class MockScaler:
    """Mock scaler for testing."""
    def transform(self, X):
        # Simple standardization: subtract mean, divide by std
        X = np.asarray(X)
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        return (X - mean) / std


class MockDataLoader:
    """Mock data loader with scaler."""
    def __init__(self, scaler=None):
        self.scaler = scaler


# ============================================================================
# Tests for load_model_checkpoint
# ============================================================================

def test_load_model_checkpoint_joblib(tmp_path):
    """Test loading model with joblib."""
    # Create a test model
    test_model = {"model_type": "test", "params": [1, 2, 3]}
    checkpoint_path = tmp_path / "model.pkl"
    
    # Save with joblib
    joblib.dump(test_model, checkpoint_path)
    
    # Load using helper
    loaded_model = load_model_checkpoint(checkpoint_path)
    
    assert loaded_model == test_model
    assert loaded_model["model_type"] == "test"
    assert loaded_model["params"] == [1, 2, 3]


def test_load_model_checkpoint_pickle_fallback(tmp_path, monkeypatch):
    """Test pickle fallback when joblib fails."""
    test_model = {"model_type": "test", "params": [1, 2, 3]}
    checkpoint_path = tmp_path / "model.pkl"
    
    # Save with pickle
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(test_model, f)
    
    # Mock joblib.load to fail
    original_load = joblib.load
    def failing_joblib_load(path):
        raise Exception("Joblib load failed")
    
    monkeypatch.setattr(joblib, 'load', failing_joblib_load)
    
    # Load should fallback to pickle
    loaded_model = load_model_checkpoint(checkpoint_path)
    
    assert loaded_model == test_model


def test_load_model_checkpoint_not_found(tmp_path):
    """Test error when checkpoint doesn't exist."""
    checkpoint_path = tmp_path / "nonexistent.pkl"
    
    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        load_model_checkpoint(checkpoint_path)


# ============================================================================
# Tests for preprocess_data_for_model
# ============================================================================

def test_preprocess_data_for_model_with_scaler():
    """Test preprocessing with data loader scaler."""
    # Create test data
    data = pd.DataFrame({
        'col1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'col2': [10.0, 20.0, 30.0, 40.0, 50.0]
    })
    
    # Create mock scaler
    scaler = MockScaler()
    data_loader = MockDataLoader(scaler=scaler)
    
    # Preprocess
    result = preprocess_data_for_model(data, data_loader)
    
    # Should be standardized
    assert isinstance(result, np.ndarray)
    assert result.shape == (5, 2)
    # Mean should be approximately 0 for standardized data
    assert np.allclose(result.mean(axis=0), 0, atol=1e-10)


def test_preprocess_data_for_model_without_scaler():
    """Test preprocessing without scaler (returns raw values)."""
    data = pd.DataFrame({
        'col1': [1.0, 2.0, 3.0],
        'col2': [10.0, 20.0, 30.0]
    })
    
    # Preprocess without scaler
    result = preprocess_data_for_model(data, data_loader=None)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 2)
    np.testing.assert_array_equal(result, data.values)


def test_preprocess_data_for_model_with_columns():
    """Test preprocessing with column selection."""
    data = pd.DataFrame({
        'col1': [1.0, 2.0, 3.0],
        'col2': [10.0, 20.0, 30.0],
        'col3': [100.0, 200.0, 300.0]
    })
    
    # Preprocess with column selection
    result = preprocess_data_for_model(data, columns=['col1', 'col3'])
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 2)
    np.testing.assert_array_equal(result, data[['col1', 'col3']].values)


def test_preprocess_data_for_model_no_scaler_attribute():
    """Test preprocessing with data_loader but no scaler attribute."""
    data = pd.DataFrame({
        'col1': [1.0, 2.0, 3.0],
        'col2': [10.0, 20.0, 30.0]
    })
    
    # Data loader without scaler
    data_loader = MockDataLoader(scaler=None)
    
    result = preprocess_data_for_model(data, data_loader)
    
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, data.values)


def test_preprocess_data_for_model_with_nans_impute_true():
    """Test that missing values are imputed when impute_missing=True."""
    data = pd.DataFrame({
        'series1': [1.0, 2.0, np.nan, 4.0, 5.0],
        'series2': [np.nan, 2.0, 3.0, 4.0, np.nan],
        'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
    })
    data.index = pd.date_range('2024-01-01', periods=5)
    
    result = preprocess_data_for_model(data, impute_missing=True)
    
    # Should have no NaNs
    assert not np.isnan(result).any(), "Result should have no NaNs when impute_missing=True"
    assert result.shape == (5, 2), "Should have 5 rows and 2 columns (excluding 'date')"


def test_preprocess_data_for_model_with_nans_impute_false():
    """Test that missing values are preserved when impute_missing=False."""
    data = pd.DataFrame({
        'series1': [1.0, 2.0, np.nan, 4.0, 5.0],
        'series2': [np.nan, 2.0, 3.0, 4.0, np.nan],
    })
    data.index = pd.date_range('2024-01-01', periods=5)
    
    result = preprocess_data_for_model(data, impute_missing=False)
    
    # Should preserve NaNs
    assert np.isnan(result).any(), "Result should preserve NaNs when impute_missing=False"


# ============================================================================
# Tests for extract_forecast_values
# ============================================================================

def test_extract_forecast_values_tuple():
    """Test extraction from tuple (X_forecast, Z_forecast)."""
    X_forecast = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    Z_forecast = np.array([[0.1], [0.2]])
    forecast_result = (X_forecast, Z_forecast)
    
    # Extract first time step
    values = extract_forecast_values(forecast_result, n_series=3, horizon_idx=0)
    
    assert isinstance(values, np.ndarray)
    assert values.shape == (3,)
    np.testing.assert_array_equal(values, [1.0, 2.0, 3.0])
    
    # Extract last time step
    values_last = extract_forecast_values(forecast_result, n_series=3, horizon_idx=-1)
    np.testing.assert_array_equal(values_last, [4.0, 5.0, 6.0])


def test_extract_forecast_values_array():
    """Test extraction from numpy array."""
    forecast_result = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    
    # Extract first time step
    values = extract_forecast_values(forecast_result, n_series=3, horizon_idx=0)
    
    assert isinstance(values, np.ndarray)
    assert values.shape == (3,)
    np.testing.assert_array_equal(values, [1.0, 2.0, 3.0])
    
    # Extract last time step
    values_last = extract_forecast_values(forecast_result, n_series=3, horizon_idx=-1)
    np.testing.assert_array_equal(values_last, [4.0, 5.0, 6.0])


def test_extract_forecast_values_dataframe():
    """Test extraction from pandas DataFrame."""
    forecast_result = pd.DataFrame({
        'series1': [1.0, 4.0],
        'series2': [2.0, 5.0],
        'series3': [3.0, 6.0]
    })
    
    # Extract first row
    values = extract_forecast_values(forecast_result, n_series=3, horizon_idx=0)
    
    assert isinstance(values, np.ndarray)
    assert values.shape == (3,)
    np.testing.assert_array_equal(values, [1.0, 2.0, 3.0])
    
    # Extract last row
    values_last = extract_forecast_values(forecast_result, n_series=3, horizon_idx=-1)
    np.testing.assert_array_equal(values_last, [4.0, 5.0, 6.0])


def test_extract_forecast_values_series():
    """Test extraction from pandas Series."""
    forecast_result = pd.Series([1.0, 2.0, 3.0], index=['s1', 's2', 's3'])
    
    values = extract_forecast_values(forecast_result, n_series=3, horizon_idx=0)
    
    assert isinstance(values, np.ndarray)
    assert values.shape == (3,)
    np.testing.assert_array_equal(values, [1.0, 2.0, 3.0])


def test_extract_forecast_values_1d_array():
    """Test extraction from 1D array."""
    forecast_result = np.array([1.0, 2.0, 3.0])
    
    values = extract_forecast_values(forecast_result, n_series=3, horizon_idx=0)
    
    assert isinstance(values, np.ndarray)
    assert values.shape == (3,)
    np.testing.assert_array_equal(values, [1.0, 2.0, 3.0])


def test_extract_forecast_values_truncate():
    """Test truncation when forecast has more series than expected."""
    forecast_result = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    
    values = extract_forecast_values(forecast_result, n_series=3, horizon_idx=0)
    
    assert isinstance(values, np.ndarray)
    assert values.shape == (3,)
    np.testing.assert_array_equal(values, [1.0, 2.0, 3.0])


def test_extract_forecast_values_pad():
    """Test padding when forecast has fewer series than expected."""
    forecast_result = np.array([[1.0, 2.0]])
    
    values = extract_forecast_values(forecast_result, n_series=4, horizon_idx=0)
    
    assert isinstance(values, np.ndarray)
    assert values.shape == (4,)
    assert values[0] == 1.0
    assert values[1] == 2.0
    assert np.isnan(values[2])
    assert np.isnan(values[3])


def test_extract_forecast_values_middle_index():
    """Test extraction with middle index."""
    forecast_result = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    
    values = extract_forecast_values(forecast_result, n_series=3, horizon_idx=1)
    
    assert isinstance(values, np.ndarray)
    assert values.shape == (3,)
    np.testing.assert_array_equal(values, [4.0, 5.0, 6.0])


def test_extract_forecast_values_exact_match():
    """Test when forecast series count exactly matches expected."""
    forecast_result = np.array([[1.0, 2.0, 3.0]])
    
    values = extract_forecast_values(forecast_result, n_series=3, horizon_idx=0)
    
    assert isinstance(values, np.ndarray)
    assert values.shape == (3,)
    np.testing.assert_array_equal(values, [1.0, 2.0, 3.0])
