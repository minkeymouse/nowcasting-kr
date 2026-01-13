"""Quick integration tests for experiment flow - fast verification."""

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
    get_weekly_dates
)
from src.forecast.sktime import _run_recursive_forecast_sktime


@pytest.fixture
def sample_test_data():
    """Create sample test data with NaNs."""
    dates = pd.date_range('2024-01-01', periods=20, freq='W-FRI')
    data = pd.DataFrame({
        'target': range(20),
        'series1': [i if i % 3 != 0 else np.nan for i in range(20)],
        'series2': range(20, 40),
        'date': dates
    }, index=dates)
    return data


def test_imputation_removes_all_nans(sample_test_data):
    """Quick test: imputation should remove all NaNs."""
    result = preprocess_data_for_model(sample_test_data, impute_missing=True)
    assert not np.isnan(result).any(), "All NaNs should be removed"
    assert result.shape[0] == 20, "Should preserve number of rows"


def test_frequency_inference_for_forecasting_horizon(sample_test_data):
    """Quick test: frequency should be inferred correctly."""
    freq = pd.infer_freq(sample_test_data.index)
    assert freq == 'W-FRI', f"Should infer W-FRI, got {freq}"


def test_filter_and_prepare_handles_date_range(sample_test_data):
    """Quick test: filtering should work with date ranges."""
    start = '2024-02-01'
    end = '2024-03-01'
    available = ['target', 'series1', 'series2']
    
    filtered, start_ts, end_ts = filter_and_prepare_test_data(
        sample_test_data, start, end, available
    )
    
    assert len(filtered) > 0, "Should have filtered data"
    assert start_ts <= end_ts, "Start should be before end"


def test_imputation_preserves_data_integrity():
    """Quick test: imputation shouldn't break data structure."""
    data = pd.DataFrame({
        'a': [1.0, 2.0, np.nan, 4.0],
        'b': [10.0, np.nan, 30.0, 40.0]
    }, index=pd.date_range('2024-01-01', periods=4, freq='W'))
    
    result = preprocess_data_for_model(data, impute_missing=True)
    
    assert result.shape == (4, 2), "Shape should be preserved"
    assert not np.isnan(result).any(), "No NaNs"
    # Non-NaN values should be preserved
    assert result[0, 0] == 1.0
    assert result[3, 0] == 4.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])  # -x stops at first failure
