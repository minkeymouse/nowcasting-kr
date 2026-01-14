"""Tests for Mamba model forecasting."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
import tempfile
import torch
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Check model availability
try:
    from mamba_ssm import Mamba2
    MAMBA_AVAILABLE = True
except ImportError as e:
    MAMBA_AVAILABLE = False
    IMPORT_ERROR = str(e)

from src.forecast.mamba import (
    load_mamba_model,
    run_recursive_forecast,
    run_multi_horizon_forecast,
    forecast
)
from src.train.mamba import MambaForecaster, TimeSeriesDataset


@pytest.fixture
def sample_data():
    """Create sample time series data."""
    dates = pd.date_range('2020-01-01', periods=200, freq='W')
    return pd.DataFrame(
        np.random.randn(200, 3),
        columns=['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
        index=dates
    )


@pytest.fixture
def mock_data_loader():
    """Create a mock data loader."""
    class MockDataLoader:
        def __init__(self):
            dates = pd.date_range('2020-01-01', periods=200, freq='W')
            self.original = pd.DataFrame(
                np.random.randn(200, 3) * 10 + 100,
                columns=['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
                index=dates
            )
            self.processed = self.original.copy()
    return MockDataLoader()


@pytest.fixture
def trained_mamba_model(tmp_path, sample_data, mock_data_loader):
    """Create and save a trained Mamba model for testing."""
    if not MAMBA_AVAILABLE:
        pytest.skip("Mamba not available")
    
    from src.train.mamba import train_mamba_model
    
    model_params = {
        'target_series': ['KOEQUIPTE', 'KOWRCCNSE'],
        'prediction_length': 4,
        'context_length': 32,
        'n_layers': 2,
        'd_model': 64,
        'd_state': 16,
        'max_epochs': 1,
        'batch_size': 8,
        'learning_rate': 0.001,
    }
    
    train_mamba_model(
        model_type='mamba',
        cfg=None,
        data=sample_data,
        model_name='mamba_test',
        outputs_dir=tmp_path,
        model_params=model_params,
        data_loader=mock_data_loader
    )
    
    return tmp_path / "model.pkl"


@pytest.mark.skipif(not MAMBA_AVAILABLE, reason=f"Mamba not available: {IMPORT_ERROR if not MAMBA_AVAILABLE else ''}")
def test_load_mamba_model(trained_mamba_model):
    """Test loading Mamba model from checkpoint."""
    model, metadata, input_proj, output_proj = load_mamba_model(trained_mamba_model)
    
    assert model is not None, "Model should be loaded"
    assert metadata is not None, "Metadata should be loaded"
    assert 'd_model' in metadata
    assert 'context_length' in metadata
    assert 'prediction_length' in metadata


@pytest.mark.skipif(not MAMBA_AVAILABLE, reason=f"Mamba not available: {IMPORT_ERROR if not MAMBA_AVAILABLE else ''}")
def test_run_recursive_forecast(trained_mamba_model, sample_data, mock_data_loader):
    """Test recursive forecasting."""
    # Use a subset of data for testing
    test_data = sample_data.iloc[100:150].copy()
    
    # Use dates that exist in the test data
    start_date = test_data.index[5].strftime('%Y-%m-%d')
    end_date = test_data.index[10].strftime('%Y-%m-%d')
    
    predictions, actuals, dates, target_series = run_recursive_forecast(
        checkpoint_path=trained_mamba_model,
        test_data=test_data,
        start_date=start_date,
        end_date=end_date,
        model_type='mamba',
        target_series=['KOEQUIPTE', 'KOWRCCNSE'],
        data_loader=mock_data_loader,
        update_params=False
    )
    
    assert len(predictions) > 0, "Should have predictions"
    assert len(actuals) > 0, "Should have actuals"
    assert len(dates) == len(predictions), "Dates should match predictions"
    assert predictions.shape[1] == len(target_series), "Prediction shape should match target series"


@pytest.mark.skipif(not MAMBA_AVAILABLE, reason=f"Mamba not available: {IMPORT_ERROR if not MAMBA_AVAILABLE else ''}")
def test_run_multi_horizon_forecast(trained_mamba_model, sample_data, mock_data_loader):
    """Test multi-horizon forecasting."""
    test_data = sample_data.iloc[100:150].copy()
    
    # Use a date that exists in the test data (after enough context)
    start_date = test_data.index[35].strftime('%Y-%m-%d')
    
    horizons = [4, 8, 12]
    forecasts, target_series = run_multi_horizon_forecast(
        checkpoint_path=trained_mamba_model,
        horizons=horizons,
        start_date=start_date,
        test_data=test_data,
        model_type='mamba',
        target_series=['KOEQUIPTE', 'KOWRCCNSE'],
        data_loader=mock_data_loader,
        return_weekly_forecasts=False
    )
    
    assert len(forecasts) == len(horizons), "Should have forecasts for all horizons"
    for horizon in horizons:
        assert horizon in forecasts, f"Forecast for horizon {horizon}w should exist"
        forecast_val = forecasts[horizon]
        if isinstance(forecast_val, np.ndarray):
            assert len(forecast_val) == len(target_series), f"Forecast shape should match target series for horizon {horizon}w"


@pytest.mark.skipif(not MAMBA_AVAILABLE, reason=f"Mamba not available: {IMPORT_ERROR if not MAMBA_AVAILABLE else ''}")
def test_forecast_function(trained_mamba_model, sample_data):
    """Test simple forecast function."""
    test_data = sample_data.iloc[100:150].copy()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy model to temp directory
        import shutil
        temp_checkpoint = Path(tmpdir) / "model.pkl"
        shutil.copy(trained_mamba_model, temp_checkpoint)
        shutil.copy(trained_mamba_model.parent / "metadata.pkl", Path(tmpdir) / "metadata.pkl")
        
        # Check if input/output projections exist
        input_proj_path = trained_mamba_model.parent / "input_proj.pkl"
        output_proj_path = trained_mamba_model.parent / "output_proj.pkl"
        if input_proj_path.exists():
            shutil.copy(input_proj_path, Path(tmpdir) / "input_proj.pkl")
        if output_proj_path.exists():
            shutil.copy(output_proj_path, Path(tmpdir) / "output_proj.pkl")
        
        forecast(
            checkpoint_path=temp_checkpoint,
            horizon=4,
            model_type='mamba',
            recursive=False,
            test_data=test_data,
            update_params=False
        )
        
        # Verify forecasts were saved
        forecasts_dir = Path(tmpdir) / "forecasts"
        assert forecasts_dir.exists(), "Forecasts directory should be created"
        assert (forecasts_dir / "predictions.npy").exists(), "Predictions should be saved"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
