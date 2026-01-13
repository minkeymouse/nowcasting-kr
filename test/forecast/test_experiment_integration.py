"""Quick integration tests for experiment flow with minimal setup."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
import tempfile
import shutil

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import (
    preprocess_data_for_model,
    filter_and_prepare_test_data,
    get_target_series_from_dataset,
    get_target_series_from_model
)


def test_data_imputation_for_sktime_models():
    """Test that data with NaNs gets properly imputed for sktime models."""
    # Create test data with NaNs (simulating mixed-frequency)
    dates = pd.date_range('2024-01-01', periods=10, freq='W-FRI')
    data = pd.DataFrame({
        'series1': [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, np.nan, 9.0, 10.0],
        'series2': [10.0, np.nan, 30.0, 40.0, 50.0, np.nan, 70.0, 80.0, 90.0, 100.0],
        'target': [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
    }, index=dates)
    
    # Test imputation
    result = preprocess_data_for_model(data, impute_missing=True)
    
    assert not np.isnan(result).any(), "All NaNs should be imputed"
    assert result.shape == (10, 3), "Should preserve all rows and columns"


def test_filter_and_prepare_with_nans():
    """Test filtering test data when it contains NaNs."""
    dates = pd.date_range('2024-01-01', periods=20, freq='W-FRI')
    data = pd.DataFrame({
        'target': range(20),
        'series1': [i if i % 3 != 0 else np.nan for i in range(20)],
        'series2': range(20, 40)
    }, index=dates)
    
    start_date = '2024-02-01'
    end_date = '2024-05-01'
    available_targets = ['target', 'series1', 'series2']
    
    filtered, start_ts, end_ts = filter_and_prepare_test_data(
        data, start_date, end_date, available_targets
    )
    
    assert len(filtered) > 0, "Should have filtered data"
    assert start_ts <= end_ts, "Start should be before end"
    assert all(col in filtered.columns for col in available_targets), "Should have all targets"


def test_imputation_preserves_data_structure():
    """Test that imputation doesn't break data structure."""
    dates = pd.date_range('2024-01-01', periods=5, freq='W-FRI')
    data = pd.DataFrame({
        'series1': [1.0, np.nan, 3.0, np.nan, 5.0],
        'series2': [10.0, 20.0, 30.0, 40.0, 50.0],
    }, index=dates)
    
    # Impute
    result = preprocess_data_for_model(data, impute_missing=True)
    
    # Check shape preserved
    assert result.shape == (5, 2), "Shape should be preserved"
    
    # Check non-NaN values preserved
    assert result[0, 0] == 1.0, "First value should be preserved"
    assert result[2, 0] == 3.0, "Third value should be preserved"
    assert result[4, 0] == 5.0, "Last value should be preserved"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
