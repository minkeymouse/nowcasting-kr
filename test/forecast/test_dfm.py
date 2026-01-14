"""Tests for DFM forecasting functions."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Check model availability
try:
    from dfm_python import DFM
    DFM_AVAILABLE = True
except ImportError as e:
    DFM_AVAILABLE = False
    IMPORT_ERROR = str(e)

from src.forecast.dfm import (
    forecast,
    run_recursive_forecast,
    run_multi_horizon_forecast
)


@pytest.fixture
def sample_data():
    """Create sample time series data."""
    dates = pd.date_range('2020-01-01', periods=200, freq='W')
    return pd.DataFrame(
        np.random.randn(200, 3),
        columns=['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
        index=dates
    )


@pytest.mark.skipif(not DFM_AVAILABLE, reason=f"DFM not available: {IMPORT_ERROR if not DFM_AVAILABLE else ''}")
def test_forecast_dfm_interface():
    """Test DFM forecast function interface."""
    # Test that function exists and has correct signature
    assert callable(forecast)
    assert callable(run_recursive_forecast)
    assert callable(run_multi_horizon_forecast)


@pytest.mark.skipif(not DFM_AVAILABLE, reason=f"DFM not available: {IMPORT_ERROR if not DFM_AVAILABLE else ''}")
def test_run_recursive_forecast_dfm_interface(sample_data):
    """Test DFM recursive forecast function interface."""
    # Create a dummy checkpoint path (won't actually work without trained model)
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "model.pkl"
        dataset_path = Path(tmpdir) / "dataset.pkl"
        
        # Test that function signature is correct
        # This will fail without actual model, but tests the interface
        try:
            run_recursive_forecast(
                checkpoint_path=checkpoint_path,
                dataset_path=dataset_path,
                test_data=sample_data.iloc[150:160],
                horizon=1,
                start_date='2020-01-01',
                end_date='2020-01-10',
                target_series=['KOEQUIPTE'],
                data_loader=None
            )
        except (FileNotFoundError, ValueError):
            # Expected - tests that function exists and is callable
            pass


@pytest.mark.skipif(not DFM_AVAILABLE, reason=f"DFM not available: {IMPORT_ERROR if not DFM_AVAILABLE else ''}")
def test_run_multi_horizon_forecast_dfm_interface(sample_data):
    """Test DFM multi-horizon forecast function interface."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "model.pkl"
        dataset_path = Path(tmpdir) / "dataset.pkl"
        horizons = [4, 8, 12]
        
        # Test that function signature is correct
        try:
            run_multi_horizon_forecast(
                checkpoint_path=checkpoint_path,
                dataset_path=dataset_path,
                horizons=horizons,
                start_date='2020-01-01',
                test_data=sample_data,
                target_series=['KOEQUIPTE'],
                data_loader=None
            )
        except (FileNotFoundError, ValueError):
            # Expected - tests that function exists and is callable
            pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
