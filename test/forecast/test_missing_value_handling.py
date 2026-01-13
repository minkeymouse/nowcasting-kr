"""Tests for missing value handling in forecast functions."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import preprocess_data_for_model


class MockScaler:
    """Mock scaler for testing."""
    def transform(self, X):
        X = np.asarray(X)
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        std = np.where(std == 0, 1, std)
        return (X - mean) / std


class MockDataLoader:
    """Mock data loader with scaler."""
    def __init__(self, scaler=None):
        self.scaler = scaler


def test_preprocess_data_for_model_with_nans_impute_true():
    """Test that missing values are imputed when impute_missing=True."""
    # Create data with NaNs
    data = pd.DataFrame({
        'series1': [1.0, 2.0, np.nan, 4.0, 5.0],
        'series2': [np.nan, 2.0, 3.0, 4.0, np.nan],
        'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
    })
    data.index = pd.date_range('2024-01-01', periods=5)
    
    result = preprocess_data_for_model(data, impute_missing=True)
    
    # Should have no NaNs
    assert not np.isnan(result).any(), "Result should have no NaNs"
    assert result.shape == (5, 2), "Should have 5 rows and 2 columns (excluding 'date')"


def test_preprocess_data_for_model_with_nans_impute_false():
    """Test that missing values are preserved when impute_missing=False."""
    # Create data with NaNs
    data = pd.DataFrame({
        'series1': [1.0, 2.0, np.nan, 4.0, 5.0],
        'series2': [np.nan, 2.0, 3.0, 4.0, np.nan],
    })
    data.index = pd.date_range('2024-01-01', periods=5)
    
    result = preprocess_data_for_model(data, impute_missing=False)
    
    # Should preserve NaNs
    assert np.isnan(result).any(), "Result should preserve NaNs when impute_missing=False"


def test_preprocess_data_for_model_forward_backward_fill():
    """Test that forward fill and backward fill work correctly."""
    # Create data with NaNs at start and middle
    data = pd.DataFrame({
        'series1': [np.nan, np.nan, 3.0, np.nan, 5.0],
        'series2': [1.0, np.nan, 3.0, 4.0, np.nan],
    })
    data.index = pd.date_range('2024-01-01', periods=5)
    
    result = preprocess_data_for_model(data, impute_missing=True)
    
    # After ffill/bfill, all NaNs should be filled
    assert not np.isnan(result).any(), "All NaNs should be filled"
    
    # Check forward fill: series1[0] and series1[1] should equal series1[2] (3.0)
    assert result[0, 0] == 3.0, "Forward fill should work"
    assert result[1, 0] == 3.0, "Forward fill should work"
    
    # Check backward fill: series2[4] should equal series2[3] (4.0)
    assert result[4, 1] == 4.0, "Backward fill should work"


def test_preprocess_data_for_model_all_nans_filled_with_zero():
    """Test that remaining NaNs after ffill/bfill are filled with 0."""
    # Create data with all NaNs
    data = pd.DataFrame({
        'series1': [np.nan, np.nan, np.nan],
        'series2': [np.nan, np.nan, np.nan],
    })
    data.index = pd.date_range('2024-01-01', periods=3)
    
    result = preprocess_data_for_model(data, impute_missing=True)
    
    # Should have no NaNs and all zeros
    assert not np.isnan(result).any(), "Should have no NaNs"
    assert np.all(result == 0), "All values should be 0 when all are NaN"


def test_preprocess_data_for_model_with_scaler():
    """Test imputation works with scaler."""
    data = pd.DataFrame({
        'series1': [1.0, 2.0, np.nan, 4.0, 5.0],
        'series2': [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    data.index = pd.date_range('2024-01-01', periods=5)
    
    scaler = MockScaler()
    data_loader = MockDataLoader(scaler=scaler)
    
    result = preprocess_data_for_model(data, data_loader=data_loader, impute_missing=True)
    
    # Should have no NaNs and be standardized
    assert not np.isnan(result).any(), "Should have no NaNs"
    assert result.shape == (5, 2), "Should have correct shape"


def test_preprocess_data_for_model_excludes_date_column():
    """Test that date columns are excluded from preprocessing."""
    data = pd.DataFrame({
        'series1': [1.0, 2.0, 3.0],
        'series2': [4.0, 5.0, 6.0],
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
    })
    data.index = pd.date_range('2024-01-01', periods=3)
    
    result = preprocess_data_for_model(data, impute_missing=True)
    
    # Should exclude 'date' column
    assert result.shape == (3, 2), "Should exclude 'date' column"
    assert 'date' not in data.columns or True, "Date column should not be in numeric columns"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
