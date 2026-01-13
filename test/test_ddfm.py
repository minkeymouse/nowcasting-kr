"""Comprehensive tests for DDFM (Deep Dynamic Factor Model).

Tests both training and forecasting functionality for DDFM model.
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
    from dfm_python import DDFM, DDFMDataset
    from dfm_python.config import DDFMConfig
    import torch
    DDFM_AVAILABLE = True
except ImportError as e:
    DDFM_AVAILABLE = False
    IMPORT_ERROR = str(e)

from src.train.ddfm import train_ddfm_model
from src.forecast.ddfm import (
    forecast,
    run_recursive_forecast,
    run_multi_horizon_forecast
)
from src.utils import load_model_checkpoint, compute_smse, compute_smae


@pytest.fixture
def sample_data():
    """Create sample time series data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=150, freq='W')
    data = pd.DataFrame(
        np.random.randn(150, 3),
        columns=['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
        index=dates
    )
    # Add some NaNs to test handling
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
                'Frequency': ['W', 'W', 'M']  # One monthly series
            })
            # Mock processed data (unscaled, for scaler fitting)
            dates = pd.date_range('2020-01-01', periods=150, freq='W')
            self.processed = pd.DataFrame(
                np.random.randn(150, 3) * 10 + 100,  # Non-standardized data
                columns=['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
                index=dates
            )
    return MockDataLoader()


# =============================================================================
# Training Tests
# =============================================================================

@pytest.mark.skipif(not DDFM_AVAILABLE, reason=f"DDFM not available: {IMPORT_ERROR if not DDFM_AVAILABLE else ''}")
def test_train_ddfm_basic(sample_data, mock_data_loader):
    """Test basic DDFM model training."""
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'num_factors': 1,
        'encoder_layers': [32, 16, 1],  # Small network for faster tests
        'activation': 'relu',
        'learning_rate': 0.001,
        'n_mc_samples': 3,  # Reduced for faster tests
        'window_size': 50,
        'max_epoch': 3,  # Reduced for faster tests
        'tolerance': 0.001,
        'disp': 5
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        train_ddfm_model(
            model_type='ddfm',
            cfg=None,
            data=sample_data,
            model_name='ddfm_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        # Verify model was saved
        model_path = outputs_dir / "model.pkl"
        assert model_path.exists(), "DDFM model should be saved"
        
        # Verify dataset was saved
        dataset_path = outputs_dir / "dataset.pkl"
        assert dataset_path.exists(), "DDFM dataset should be saved"
        
        # Verify model can be loaded
        dataset = joblib.load(dataset_path)
        model = DDFM.load(model_path, dataset=dataset)
        assert model is not None, "Model should load successfully"
        assert hasattr(model, 'predict'), "Model should have predict method"
        assert hasattr(model, 'build_state_space'), "Model should have build_state_space method"


@pytest.mark.skipif(not DDFM_AVAILABLE, reason=f"DDFM not available: {IMPORT_ERROR if not DDFM_AVAILABLE else ''}")
def test_train_ddfm_multiple_targets(sample_data, mock_data_loader):
    """Test DDFM training with multiple target series."""
    model_params = {
        'target_series': ['KOEQUIPTE', 'KOWRCCNSE'],
        'num_factors': 2,
        'encoder_layers': [32, 16, 2],  # 2 factors
        'activation': 'relu',
        'learning_rate': 0.001,
        'n_mc_samples': 3,
        'window_size': 50,
        'max_epoch': 3,
        'tolerance': 0.001,
        'disp': 5
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        train_ddfm_model(
            model_type='ddfm',
            cfg=None,
            data=sample_data,
            model_name='ddfm_multiple_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        model_path = outputs_dir / "model.pkl"
        assert model_path.exists(), "DDFM model should be saved"


@pytest.mark.skipif(not DDFM_AVAILABLE, reason=f"DDFM not available: {IMPORT_ERROR if not DDFM_AVAILABLE else ''}")
def test_train_ddfm_with_nans(sample_data, mock_data_loader):
    """Test that DDFM can handle NaN values in training data."""
    # Create data with more NaNs
    data_with_nans = sample_data.copy()
    data_with_nans.loc[data_with_nans.index[:20], 'KOEQUIPTE'] = np.nan
    
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'num_factors': 1,
        'encoder_layers': [32, 16, 1],
        'activation': 'relu',
        'learning_rate': 0.001,
        'n_mc_samples': 3,
        'window_size': 50,
        'max_epoch': 3,
        'tolerance': 0.001,
        'disp': 5
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        # Should not raise error - DDFM handles NaNs internally
        train_ddfm_model(
            model_type='ddfm',
            cfg=None,
            data=data_with_nans,
            model_name='ddfm_nan_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        model_path = outputs_dir / "model.pkl"
        assert model_path.exists(), "Training should succeed with NaNs"


@pytest.mark.skipif(not DDFM_AVAILABLE, reason=f"DDFM not available: {IMPORT_ERROR if not DDFM_AVAILABLE else ''}")
def test_train_ddfm_state_space_built(sample_data, mock_data_loader):
    """Test that state-space model is built after training."""
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'num_factors': 1,
        'encoder_layers': [32, 16, 1],
        'activation': 'relu',
        'learning_rate': 0.001,
        'n_mc_samples': 3,
        'window_size': 50,
        'max_epoch': 3,
        'tolerance': 0.001,
        'disp': 5
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        train_ddfm_model(
            model_type='ddfm',
            cfg=None,
            data=sample_data,
            model_name='ddfm_statespace_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        model_path = outputs_dir / "model.pkl"
        dataset_path = outputs_dir / "dataset.pkl"
        
        # Load model and verify state-space is built
        dataset = joblib.load(dataset_path)
        model = DDFM.load(model_path, dataset=dataset)
        
        assert hasattr(model, 'state_space_params'), "Model should have state_space_params"
        assert model.state_space_params is not None, "State-space should be built"


# =============================================================================
# Forecasting Tests
# =============================================================================

@pytest.mark.skipif(not DDFM_AVAILABLE, reason=f"DDFM not available: {IMPORT_ERROR if not DDFM_AVAILABLE else ''}")
def test_forecast_ddfm_basic(sample_data, mock_data_loader):
    """Test basic DDFM forecasting."""
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'num_factors': 1,
        'encoder_layers': [32, 16, 1],
        'activation': 'relu',
        'learning_rate': 0.001,
        'n_mc_samples': 3,
        'window_size': 50,
        'max_epoch': 3,
        'tolerance': 0.001,
        'disp': 5
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        # Train model first
        train_ddfm_model(
            model_type='ddfm',
            cfg=None,
            data=sample_data,
            model_name='ddfm_forecast_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        checkpoint_path = outputs_dir / "model.pkl"
        
        # Test basic forecast
        forecast(
            checkpoint_path=checkpoint_path,
            horizon=4,
            model_type='ddfm',
            recursive=False
        )
        
        # Verify forecasts were saved
        forecasts_dir = outputs_dir / "forecasts"
        assert forecasts_dir.exists(), "Forecasts directory should be created"


@pytest.mark.skipif(not DDFM_AVAILABLE, reason=f"DDFM not available: {IMPORT_ERROR if not DDFM_AVAILABLE else ''}")
def test_forecast_ddfm_recursive(sample_data, mock_data_loader):
    """Test DDFM recursive forecasting."""
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'num_factors': 1,
        'encoder_layers': [32, 16, 1],
        'activation': 'relu',
        'learning_rate': 0.001,
        'n_mc_samples': 3,
        'window_size': 50,
        'max_epoch': 3,
        'tolerance': 0.001,
        'disp': 5
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        # Train model first
        train_ddfm_model(
            model_type='ddfm',
            cfg=None,
            data=sample_data,
            model_name='ddfm_recursive_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        checkpoint_path = outputs_dir / "model.pkl"
        dataset_path = outputs_dir / "dataset.pkl"
        
        # Prepare test data
        test_data = sample_data.copy()
        
        # Run recursive forecast (note: DDFM recursive is limited)
        predictions, actuals, dates = run_recursive_forecast(
            checkpoint_path=checkpoint_path,
            dataset_path=dataset_path,
            test_data=test_data,
            start_date='2020-10-01',
            end_date='2020-11-01',
            model_type='ddfm',
            target_series=['KOEQUIPTE'],
            data_loader=mock_data_loader
        )
        
        # Verify outputs (DDFM recursive may return single forecast)
        assert len(predictions) > 0, "Should produce predictions"
        assert len(actuals) == len(predictions), "Actuals should match predictions"
        assert len(dates) == len(predictions), "Dates should match predictions"


@pytest.mark.skipif(not DDFM_AVAILABLE, reason=f"DDFM not available: {IMPORT_ERROR if not DDFM_AVAILABLE else ''}")
def test_forecast_ddfm_multi_horizon(sample_data, mock_data_loader):
    """Test DDFM multi-horizon forecasting."""
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'num_factors': 1,
        'encoder_layers': [32, 16, 1],
        'activation': 'relu',
        'learning_rate': 0.001,
        'n_mc_samples': 3,
        'window_size': 50,
        'max_epoch': 3,
        'tolerance': 0.001,
        'disp': 5
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        # Train model first
        train_ddfm_model(
            model_type='ddfm',
            cfg=None,
            data=sample_data,
            model_name='ddfm_multi_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        checkpoint_path = outputs_dir / "model.pkl"
        dataset_path = outputs_dir / "dataset.pkl"
        
        # Prepare test data
        test_data = sample_data.copy()
        
        # Run multi-horizon forecast
        horizons = [4, 8]
        forecasts = run_multi_horizon_forecast(
            checkpoint_path=checkpoint_path,
            dataset_path=dataset_path,
            horizons=horizons,
            start_date='2020-10-01',
            test_data=test_data,
            model_type='ddfm',
            target_series=['KOEQUIPTE'],
            data_loader=mock_data_loader
        )
        
        # Verify outputs
        assert len(forecasts) == len(horizons), "Should have forecasts for all horizons"
        for horizon in horizons:
            assert horizon in forecasts, f"Should have forecast for horizon {horizon}"
            assert len(forecasts[horizon]) > 0, f"Forecast for horizon {horizon} should not be empty"


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.skipif(not DDFM_AVAILABLE, reason=f"DDFM not available: {IMPORT_ERROR if not DDFM_AVAILABLE else ''}")
def test_full_pipeline_ddfm(sample_data, mock_data_loader):
    """Test full pipeline: train -> save -> load -> forecast."""
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'num_factors': 1,
        'encoder_layers': [32, 16, 1],
        'activation': 'relu',
        'learning_rate': 0.001,
        'n_mc_samples': 3,
        'window_size': 50,
        'max_epoch': 3,
        'tolerance': 0.001,
        'disp': 5
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        # Step 1: Train
        train_ddfm_model(
            model_type='ddfm',
            cfg=None,
            data=sample_data,
            model_name='ddfm_pipeline_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        checkpoint_path = outputs_dir / "model.pkl"
        dataset_path = outputs_dir / "dataset.pkl"
        assert checkpoint_path.exists(), "Model should be saved"
        assert dataset_path.exists(), "Dataset should be saved"
        
        # Step 2: Load
        dataset = joblib.load(dataset_path)
        model = DDFM.load(checkpoint_path, dataset=dataset)
        assert model is not None, "Model should load"
        
        # Step 3: Forecast
        test_data = sample_data.copy()
        predictions, actuals, dates = run_recursive_forecast(
            checkpoint_path=checkpoint_path,
            dataset_path=dataset_path,
            test_data=test_data,
            start_date='2020-10-01',
            end_date='2020-10-15',
            model_type='ddfm',
            target_series=['KOEQUIPTE'],
            data_loader=mock_data_loader
        )
        
        # Step 4: Compute metrics
        if len(actuals) > 0 and len(predictions) > 0:
            smse = compute_smse(actuals, predictions)
            smae = compute_smae(actuals, predictions)
            
            assert not np.isnan(smse), "sMSE should be valid"
            assert not np.isnan(smae), "sMAE should be valid"
            assert smse >= 0, "sMSE should be non-negative"
            assert smae >= 0, "sMAE should be non-negative"


@pytest.mark.skipif(not DDFM_AVAILABLE, reason=f"DDFM not available: {IMPORT_ERROR if not DDFM_AVAILABLE else ''}")
def test_ddfm_predict_returns_factors(sample_data, mock_data_loader):
    """Test that DDFM predict can return both series and factors."""
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'num_factors': 1,
        'encoder_layers': [32, 16, 1],
        'activation': 'relu',
        'learning_rate': 0.001,
        'n_mc_samples': 3,
        'window_size': 50,
        'max_epoch': 3,
        'tolerance': 0.001,
        'disp': 5
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        # Train model
        train_ddfm_model(
            model_type='ddfm',
            cfg=None,
            data=sample_data,
            model_name='ddfm_factors_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        model_path = outputs_dir / "model.pkl"
        dataset_path = outputs_dir / "dataset.pkl"
        
        # Load and predict with return_factors=True
        dataset = joblib.load(dataset_path)
        model = DDFM.load(model_path, dataset=dataset)
        
        predictions = model.predict(horizon=4, return_series=True, return_factors=True)
        
        # Should return tuple (series, factors)
        assert isinstance(predictions, tuple), "Should return tuple when both flags are True"
        assert len(predictions) == 2, "Should return (series, factors)"
        X_forecast, Z_forecast = predictions
        assert X_forecast.shape[0] == 4, "Series forecast should have correct horizon"
        assert Z_forecast.shape[0] == 4, "Factor forecast should have correct horizon"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
