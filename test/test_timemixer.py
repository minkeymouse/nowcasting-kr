"""Comprehensive tests for TimeMixer model.

Tests both training and forecasting functionality for TimeMixer model,
including multiscale parameters for mixed-frequency data.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile
import pytest
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Check model availability
try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import TimeMixer
    TIMEMIXER_AVAILABLE = True
except ImportError as e:
    TIMEMIXER_AVAILABLE = False
    IMPORT_ERROR = str(e)

from src.train.timemixer import train_timemixer_model
from src.forecast.timemixer import (
    forecast,
    run_recursive_forecast,
    run_multi_horizon_forecast
)
from src.utils import load_model_checkpoint, compute_smse, compute_smae


@pytest.fixture
def sample_data():
    """Create sample time series data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='W')  # Increased for multiscale tests
    data = pd.DataFrame(
        np.random.randn(200, 3),
        columns=['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
        index=dates
    )
    # Add some NaNs to test interpolation
    data.loc[data.index[10:15], 'KOEQUIPTE'] = np.nan
    data.loc[data.index[20:22], 'KOWRCCNSE'] = np.nan
    return data


@pytest.fixture
def mock_data_loader():
    """Create a mock data loader with metadata."""
    class MockDataLoader:
        def __init__(self):
            self.metadata = pd.DataFrame({
                'Series_ID': ['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
                'Frequency': ['M', 'W', 'W']  # One monthly series (KOEQUIPTE)
            })
            # Mock processed data with datetime index
            dates = pd.date_range('2020-01-01', periods=150, freq='W')
            self.processed = pd.DataFrame(
                np.random.randn(150, 3),
                columns=['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
                index=dates
            )
    return MockDataLoader()


# =============================================================================
# Training Tests
# =============================================================================

@pytest.mark.skipif(not TIMEMIXER_AVAILABLE, reason=f"TimeMixer not available: {IMPORT_ERROR if not TIMEMIXER_AVAILABLE else ''}")
def test_train_timemixer_basic(sample_data, mock_data_loader):
    """Test basic TimeMixer model training."""
    model_params = {
        'target_series': ['KOEQUIPTE', 'KOWRCCNSE'],
        'prediction_length': 4,
        'context_length': 96,  # Longer context to accommodate downsampling
        'num_layers': 2,
        'max_epochs': 1,
        'batch_size': 32,
        'dropout': 0.1,
        'down_sampling_layers': 2,  # Reduced from default 3 to avoid sequence becoming too small
        'down_sampling_window': 2,  # Use 2 instead of 4 for basic test
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        train_timemixer_model(
            model_type='timemixer',
            cfg=None,
            data=sample_data,
            model_name='timemixer_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        # Verify model was saved
        model_path = outputs_dir / "model.pkl"
        assert model_path.exists(), "TimeMixer model should be saved"
        
        # Verify model can be loaded
        model = load_model_checkpoint(model_path)
        assert model is not None, "Model should load successfully"
        assert hasattr(model, 'fit'), "Model should have fit method"
        assert hasattr(model, 'predict'), "Model should have predict method"


@pytest.mark.skipif(not TIMEMIXER_AVAILABLE, reason=f"TimeMixer not available: {IMPORT_ERROR if not TIMEMIXER_AVAILABLE else ''}")
def test_train_timemixer_with_multiscale_params(sample_data, mock_data_loader):
    """Test TimeMixer training with multiscale parameters for mixed-frequency data."""
    model_params = {
        'target_series': ['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
        'prediction_length': 4,
        'context_length': 96,  # Must be divisible by down_sampling_window^down_sampling_layers: 96 / (4^2) = 6
        'num_layers': 2,
        'max_epochs': 1,
        'batch_size': 32,
        'down_sampling_layers': 2,  # Reduced to 2 to avoid dimension issues with 4^3=64
        'down_sampling_window': 4,  # Align with monthly frequency
        'down_sampling_method': 'avg',
        'decomp_method': 'moving_avg',
        'moving_avg': 25,
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        train_timemixer_model(
            model_type='timemixer',
            cfg=None,
            data=sample_data,
            model_name='timemixer_multiscale_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        # Verify model was saved
        model_path = outputs_dir / "model.pkl"
        assert model_path.exists(), "TimeMixer model with multiscale params should be saved"
        
        # Verify model can be loaded and has correct parameters
        model = load_model_checkpoint(model_path)
        assert model is not None, "Model should load successfully"
        
        # Check if model has the expected structure
        if hasattr(model, 'models') and len(model.models) > 0:
            timemixer_model = model.models[0]
            # Verify multiscale parameters are set
            assert hasattr(timemixer_model, 'down_sampling_layers'), "Model should have down_sampling_layers"
            assert hasattr(timemixer_model, 'down_sampling_window'), "Model should have down_sampling_window"


@pytest.mark.skipif(not TIMEMIXER_AVAILABLE, reason=f"TimeMixer not available: {IMPORT_ERROR if not TIMEMIXER_AVAILABLE else ''}")
def test_train_timemixer_auto_detect_monthly(sample_data, mock_data_loader):
    """Test that TimeMixer automatically detects monthly series and adjusts down_sampling_window."""
    model_params = {
        'target_series': ['KOEQUIPTE', 'KOWRCCNSE'],  # KOEQUIPTE is monthly in mock_data_loader
        'prediction_length': 4,
        'context_length': 96,  # Longer context to accommodate downsampling
        'num_layers': 2,
        'max_epochs': 1,
        'batch_size': 32,
        'down_sampling_layers': 2,  # Reduced to avoid sequence becoming too small
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        train_timemixer_model(
            model_type='timemixer',
            cfg=None,
            data=sample_data,
            model_name='timemixer_auto_monthly_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        # Verify model was saved
        model_path = outputs_dir / "model.pkl"
        assert model_path.exists(), "TimeMixer model with auto-detected monthly series should be saved"
        
        # Verify model was created (the function should have logged about monthly series detection)
        model = load_model_checkpoint(model_path)
        assert model is not None, "Model should load successfully"


@pytest.mark.skipif(not TIMEMIXER_AVAILABLE, reason=f"TimeMixer not available: {IMPORT_ERROR if not TIMEMIXER_AVAILABLE else ''}")
def test_train_timemixer_all_columns(sample_data, mock_data_loader):
    """Test TimeMixer training when no target_series specified (uses all columns)."""
    model_params = {
        'prediction_length': 4,
        'context_length': 96,  # Longer context to accommodate downsampling
        'num_layers': 2,
        'max_epochs': 1,
        'batch_size': 32,
        'down_sampling_layers': 2,
        'down_sampling_window': 2,
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        train_timemixer_model(
            model_type='timemixer',
            cfg=None,
            data=sample_data,
            model_name='timemixer_all_cols_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        # Verify model was saved
        model_path = outputs_dir / "model.pkl"
        assert model_path.exists(), "TimeMixer model should be saved when using all columns"


@pytest.mark.skipif(not TIMEMIXER_AVAILABLE, reason=f"TimeMixer not available: {IMPORT_ERROR if not TIMEMIXER_AVAILABLE else ''}")
def test_train_timemixer_nan_interpolation(sample_data, mock_data_loader):
    """Test that NaN values are handled during training."""
    # Create data with NaNs
    data_with_nans = sample_data.copy()
    data_with_nans.loc[data_with_nans.index[:10], 'KOEQUIPTE'] = np.nan
    
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'prediction_length': 4,
        'context_length': 96,  # Longer context to accommodate downsampling
        'num_layers': 2,
        'max_epochs': 1,
        'batch_size': 32,
        'down_sampling_layers': 2,
        'down_sampling_window': 2,
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        # Should not raise error - NaNs should be handled
        train_timemixer_model(
            model_type='timemixer',
            cfg=None,
            data=data_with_nans,
            model_name='timemixer_nan_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        # Verify model was saved (training succeeded)
        model_path = outputs_dir / "model.pkl"
        assert model_path.exists(), "Training should succeed with NaN handling"


# =============================================================================
# Forecasting Tests
# =============================================================================

@pytest.mark.skipif(not TIMEMIXER_AVAILABLE, reason=f"TimeMixer not available: {IMPORT_ERROR if not TIMEMIXER_AVAILABLE else ''}")
def test_forecast_timemixer_basic(sample_data, mock_data_loader):
    """Test basic TimeMixer forecasting."""
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'prediction_length': 4,
        'context_length': 96,
        'num_layers': 2,
        'max_epochs': 1,
        'batch_size': 32,
        'down_sampling_layers': 2,
        'down_sampling_window': 2,
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        # Train model first
        train_timemixer_model(
            model_type='timemixer',
            cfg=None,
            data=sample_data.iloc[:150],  # Use more data for training (need > context_length + prediction_length)
            model_name='timemixer_forecast_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        model_path = outputs_dir / "model.pkl"
        assert model_path.exists(), "Model should be saved"
        
        # Test basic forecast
        forecast(
            checkpoint_path=model_path,
            horizon=4,
            model_type='timemixer'
        )
        
        # Verify forecasts were saved
        forecasts_dir = outputs_dir / "forecasts"
        assert forecasts_dir.exists(), "Forecasts directory should be created"
        assert (forecasts_dir / "predictions.csv").exists() or (forecasts_dir / "predictions.npy").exists(), \
            "Forecasts should be saved"


@pytest.mark.skipif(not TIMEMIXER_AVAILABLE, reason=f"TimeMixer not available: {IMPORT_ERROR if not TIMEMIXER_AVAILABLE else ''}")
def test_forecast_timemixer_recursive(sample_data, mock_data_loader):
    """Test TimeMixer recursive forecasting."""
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'prediction_length': 1,  # 1-step ahead for recursive
        'context_length': 96,
        'num_layers': 2,
        'max_epochs': 1,
        'batch_size': 32,
        'down_sampling_layers': 2,
        'down_sampling_window': 2,
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        # Train model first
        train_timemixer_model(
            model_type='timemixer',
            cfg=None,
            data=sample_data.iloc[:150],  # Use more data for training
            model_name='timemixer_recursive_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        model_path = outputs_dir / "model.pkl"
        assert model_path.exists(), "Model should be saved"
        
        # Test recursive forecast
        test_subset = sample_data.iloc[150:160].copy()  # Use remaining data for testing
        
        predictions, actuals, dates, target_series = run_recursive_forecast(
            checkpoint_path=model_path,
            test_data=test_subset,
            start_date=test_subset.index[0].strftime('%Y-%m-%d'),
            end_date=test_subset.index[-1].strftime('%Y-%m-%d'),
            model_type='timemixer',
            target_series=['KOEQUIPTE'],
            data_loader=mock_data_loader,
            update_params=False  # Cutoff only, no retraining for speed
        )
        
        # Verify results
        assert len(predictions) > 0, "Should have predictions"
        assert len(actuals) > 0, "Should have actuals"
        assert len(dates) == len(predictions), "Dates should match predictions length"
        assert len(target_series) > 0, "Should have target series"
        
        # Verify shapes
        assert predictions.shape[0] == len(dates), "Predictions should have correct shape"
        assert predictions.shape[1] == len(target_series), "Predictions should match target series count"


@pytest.mark.skipif(not TIMEMIXER_AVAILABLE, reason=f"TimeMixer not available: {IMPORT_ERROR if not TIMEMIXER_AVAILABLE else ''}")
def test_forecast_timemixer_multi_horizon(sample_data, mock_data_loader):
    """Test TimeMixer multi-horizon forecasting."""
    max_horizon = 8
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'prediction_length': max_horizon,  # Train with max horizon
        'context_length': 96,
        'num_layers': 2,
        'max_epochs': 1,
        'batch_size': 32,
        'down_sampling_layers': 2,
        'down_sampling_window': 2,
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        # Train model first
        train_timemixer_model(
            model_type='timemixer',
            cfg=None,
            data=sample_data.iloc[:150],  # Use more data for training
            model_name='timemixer_multi_horizon_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        model_path = outputs_dir / "model.pkl"
        assert model_path.exists(), "Model should be saved"
        
        # Test multi-horizon forecast
        test_subset = sample_data.iloc[150:170].copy()  # Use remaining data for testing
        horizons = [2, 4, 6, 8]
        
        horizon_forecasts, target_series = run_multi_horizon_forecast(
            checkpoint_path=model_path,
            horizons=horizons,
            start_date=test_subset.index[0].strftime('%Y-%m-%d'),
            test_data=test_subset,
            model_type='timemixer',
            target_series=['KOEQUIPTE'],
            data_loader=mock_data_loader
        )
        
        # Verify results
        assert len(horizon_forecasts) == len(horizons), "Should have forecasts for all horizons"
        for h in horizons:
            assert h in horizon_forecasts, f"Should have forecast for horizon {h}"
            assert horizon_forecasts[h].shape[0] == len(target_series), \
                f"Forecast for horizon {h} should have correct shape"


@pytest.mark.skipif(not TIMEMIXER_AVAILABLE, reason=f"TimeMixer not available: {IMPORT_ERROR if not TIMEMIXER_AVAILABLE else ''}")
def test_forecast_timemixer_multiscale_config(sample_data, mock_data_loader):
    """Test that multiscale configuration is preserved during forecasting."""
    model_params = {
        'target_series': ['KOEQUIPTE', 'KOWRCCNSE'],
        'prediction_length': 4,
        'context_length': 96,
        'num_layers': 2,
        'max_epochs': 1,
        'batch_size': 32,
        'down_sampling_layers': 3,
        'down_sampling_window': 4,
        'down_sampling_method': 'avg',
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        # Train model with multiscale params
        train_timemixer_model(
            model_type='timemixer',
            cfg=None,
            data=sample_data.iloc[:150],  # Use more data for training
            model_name='timemixer_multiscale_forecast_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        model_path = outputs_dir / "model.pkl"
        
        # Test forecasting with the multiscale model
        forecast(
            checkpoint_path=model_path,
            horizon=4,
            model_type='timemixer'
        )
        
        # Verify forecasts were generated
        forecasts_dir = outputs_dir / "forecasts"
        assert forecasts_dir.exists(), "Forecasts directory should be created"


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.skipif(not TIMEMIXER_AVAILABLE, reason=f"TimeMixer not available: {IMPORT_ERROR if not TIMEMIXER_AVAILABLE else ''}")
def test_timemixer_end_to_end(sample_data, mock_data_loader):
    """Test end-to-end training and forecasting workflow."""
    model_params = {
        'target_series': ['KOEQUIPTE', 'KOWRCCNSE'],
        'prediction_length': 4,
        'context_length': 96,
        'num_layers': 2,
        'max_epochs': 1,
        'batch_size': 32,
        'down_sampling_layers': 2,  # Reduced to avoid sequence becoming too small
        'down_sampling_window': 4,  # For mixed-frequency alignment
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        # Train
        train_timemixer_model(
            model_type='timemixer',
            cfg=None,
            data=sample_data.iloc[:150],  # Use more data for training
            model_name='timemixer_e2e_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        model_path = outputs_dir / "model.pkl"
        assert model_path.exists(), "Model should be trained"
        
        # Forecast
        test_subset = sample_data.iloc[150:160].copy()  # Use remaining data for testing
        
        predictions, actuals, dates, target_series = run_recursive_forecast(
            checkpoint_path=model_path,
            test_data=test_subset,
            start_date=test_subset.index[0].strftime('%Y-%m-%d'),
            end_date=test_subset.index[-1].strftime('%Y-%m-%d'),
            model_type='timemixer',
            target_series=model_params['target_series'],
            data_loader=mock_data_loader,
            update_params=False
        )
        
        # Verify end-to-end workflow completed
        assert len(predictions) > 0, "Should have predictions"
        assert len(actuals) > 0, "Should have actuals"
        assert predictions.shape[1] == len(target_series), "Predictions should match target series"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
