"""Comprehensive tests for TFT, PatchTST, and iTransformer models.

Tests both training and forecasting functionality for all three attention-based models.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile
import pytest
import joblib
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Check model availability
try:
    from sktime.forecasting.patch_tst import PatchTSTForecaster
    from sktime.forecasting.pytorchforecasting import PytorchForecastingTFT
    from neuralforecast import NeuralForecast
    from neuralforecast.models import iTransformer
    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
    IMPORT_ERROR = str(e)

from src.train.sktime import train_sktime_model
from src.forecast.sktime import run_recursive_forecast, run_multi_horizon_forecast
from src.utils import load_model_checkpoint, compute_smse, compute_smae


@pytest.fixture
def sample_data():
    """Create sample time series data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=150, freq='W')  # Increased for PatchTST
    data = pd.DataFrame(
        np.random.randn(150, 3),
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
                'Frequency': ['W', 'W', 'M']  # One monthly series
            })
    return MockDataLoader()


# =============================================================================
# Training Tests
# =============================================================================

@pytest.mark.skipif(not MODELS_AVAILABLE, reason=f"Models not available: {IMPORT_ERROR if not MODELS_AVAILABLE else ''}")
def test_train_patchtst(sample_data, mock_data_loader):
    """Test PatchTST model training."""
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'prediction_length': 4,
        'context_length': 32,  # Must be > patch_length (default 16)
        'patch_len': 8,  # Smaller patch length
        'd_model': 64,
        'max_epochs': 1,
        'batch_size': 32,
        'validation_split': 0.0  # Disable validation to avoid dataset length issues
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        train_sktime_model(
            model_type='patchtst',
            cfg=None,
            data=sample_data,
            model_name='patchtst_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        # Verify model was saved
        model_path = outputs_dir / "model.pkl"
        assert model_path.exists(), "PatchTST model should be saved"
        
        # Verify model can be loaded
        model = load_model_checkpoint(model_path)
        assert model is not None, "Model should load successfully"
        assert hasattr(model, 'fit'), "Model should have fit method"
        assert hasattr(model, 'predict'), "Model should have predict method"


@pytest.mark.skipif(not MODELS_AVAILABLE, reason=f"Models not available: {IMPORT_ERROR if not MODELS_AVAILABLE else ''}")
def test_train_tft(sample_data, mock_data_loader):
    """Test TFT model training."""
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'prediction_length': 4,
        'context_length': 16,
        'hidden_size': 32,
        'max_epochs': 1,
        'batch_size': 32
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        train_sktime_model(
            model_type='tft',
            cfg=None,
            data=sample_data,
            model_name='tft_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        # Verify model was saved (may be .pkl or need dill)
        model_path = outputs_dir / "model.pkl"
        assert model_path.exists(), "TFT model should be saved"
        
        # Verify model can be loaded
        model = load_model_checkpoint(model_path)
        assert model is not None, "Model should load successfully"


@pytest.mark.skipif(not MODELS_AVAILABLE, reason=f"Models not available: {IMPORT_ERROR if not MODELS_AVAILABLE else ''}")
def test_train_itransformer(sample_data, mock_data_loader):
    """Test iTransformer model training."""
    model_params = {
        'target_series': ['KOEQUIPTE', 'KOWRCCNSE'],
        'prediction_length': 4,
        'context_length': 16,
        'n_heads': 4,
        'd_model': 64,
        'max_epochs': 1,
        'batch_size': 32
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        train_sktime_model(
            model_type='itransformer',
            cfg=None,
            data=sample_data,
            model_name='itransformer_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        # Verify model was saved
        model_path = outputs_dir / "model.pkl"
        assert model_path.exists(), "iTransformer model should be saved"
        
        # Verify model can be loaded
        model = load_model_checkpoint(model_path)
        assert model is not None, "Model should load successfully"


@pytest.mark.skipif(not MODELS_AVAILABLE, reason=f"Models not available: {IMPORT_ERROR if not MODELS_AVAILABLE else ''}")
def test_training_nan_interpolation(sample_data, mock_data_loader):
    """Test that NaN values are interpolated during training."""
    # Create data with NaNs
    data_with_nans = sample_data.copy()
    data_with_nans.loc[data_with_nans.index[:10], 'KOEQUIPTE'] = np.nan
    
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'prediction_length': 4,
        'context_length': 32,  # Must be > patch_length
        'patch_len': 8,  # Smaller patch length
        'max_epochs': 1,
        'batch_size': 32,
        'validation_split': 0.0  # Disable validation
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        # Should not raise error - NaNs should be interpolated
        train_sktime_model(
            model_type='patchtst',
            cfg=None,
            data=data_with_nans,
            model_name='patchtst_nan_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        # Verify model was saved (training succeeded)
        model_path = outputs_dir / "model.pkl"
        assert model_path.exists(), "Training should succeed with NaN interpolation"


# =============================================================================
# Forecasting Tests
# =============================================================================

@pytest.mark.skipif(not MODELS_AVAILABLE, reason=f"Models not available: {IMPORT_ERROR if not MODELS_AVAILABLE else ''}")
def test_forecast_patchtst_recursive(sample_data, mock_data_loader):
    """Test PatchTST recursive forecasting."""
    # Train model first
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'prediction_length': 1,
        'context_length': 32,  # Must be > patch_length
        'patch_len': 8,  # Smaller patch length
        'd_model': 64,
        'max_epochs': 1,
        'batch_size': 32,
        'validation_split': 0.0  # Disable validation
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        # Train
        train_sktime_model(
            model_type='patchtst',
            cfg=None,
            data=sample_data,
            model_name='patchtst_forecast_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        checkpoint_path = outputs_dir / "model.pkl"
        
        # Prepare test data
        test_data = sample_data.copy()
        
        # Run recursive forecast
        predictions, actuals, dates, actual_targets = run_recursive_forecast(
            checkpoint_path=checkpoint_path,
            test_data=test_data,
            start_date='2020-10-01',
            end_date='2020-11-01',
            model_type='patchtst',
            target_series=['KOEQUIPTE'],
            data_loader=mock_data_loader,
            update_params=False
        )
        
        # Verify outputs
        assert len(predictions) > 0, "Should produce predictions"
        assert len(actuals) == len(predictions), "Actuals should match predictions"
        assert len(dates) == len(predictions), "Dates should match predictions"
        assert 'KOEQUIPTE' in actual_targets, "Target series should be preserved"
        assert predictions.shape[1] == len(actual_targets), "Predictions should match target count"


@pytest.mark.skipif(not MODELS_AVAILABLE, reason=f"Models not available: {IMPORT_ERROR if not MODELS_AVAILABLE else ''}")
def test_forecast_tft_recursive(sample_data, mock_data_loader):
    """Test TFT recursive forecasting."""
    # Train model first
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'prediction_length': 1,
        'context_length': 16,
        'hidden_size': 32,
        'max_epochs': 1,
        'batch_size': 32
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        # Train
        train_sktime_model(
            model_type='tft',
            cfg=None,
            data=sample_data,
            model_name='tft_forecast_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        checkpoint_path = outputs_dir / "model.pkl"
        
        # Prepare test data
        test_data = sample_data.copy()
        
        # Run recursive forecast
        predictions, actuals, dates, actual_targets = run_recursive_forecast(
            checkpoint_path=checkpoint_path,
            test_data=test_data,
            start_date='2020-10-01',
            end_date='2020-11-01',
            model_type='tft',
            target_series=['KOEQUIPTE'],
            data_loader=mock_data_loader,
            update_params=False
        )
        
        # Verify outputs
        assert len(predictions) > 0, "Should produce predictions"
        assert len(actuals) == len(predictions), "Actuals should match predictions"
        assert len(dates) == len(predictions), "Dates should match predictions"


@pytest.mark.skipif(not MODELS_AVAILABLE, reason=f"Models not available: {IMPORT_ERROR if not MODELS_AVAILABLE else ''}")
def test_forecast_patchtst_multi_horizon(sample_data, mock_data_loader):
    """Test PatchTST multi-horizon forecasting."""
    # Train model with max horizon
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'prediction_length': 8,  # Max horizon
        'context_length': 32,  # Must be > patch_length
        'patch_len': 8,  # Smaller patch length
        'd_model': 64,
        'max_epochs': 1,
        'batch_size': 32,
        'validation_split': 0.0  # Disable validation
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        # Train
        train_sktime_model(
            model_type='patchtst',
            cfg=None,
            data=sample_data,
            model_name='patchtst_multi_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        checkpoint_path = outputs_dir / "model.pkl"
        
        # Prepare test data
        test_data = sample_data.copy()
        
        # Run multi-horizon forecast
        horizons = [4, 8]
        forecasts, actual_targets = run_multi_horizon_forecast(
            checkpoint_path=checkpoint_path,
            horizons=horizons,
            start_date='2020-10-01',
            test_data=test_data,
            model_type='patchtst',
            target_series=['KOEQUIPTE'],
            data_loader=mock_data_loader
        )
        
        # Verify outputs
        assert len(forecasts) == len(horizons), "Should have forecasts for all horizons"
        for horizon in horizons:
            assert horizon in forecasts, f"Should have forecast for horizon {horizon}"
            assert len(forecasts[horizon]) > 0, f"Forecast for horizon {horizon} should not be empty"


# =============================================================================
# Metrics Tests
# =============================================================================

def test_metrics_with_nan_values():
    """Test that metrics handle NaN values correctly."""
    # Create predictions and actuals with some NaNs
    actuals = np.array([
        [1.0, 2.0],
        [np.nan, 3.0],  # One NaN
        [3.0, 4.0],
        [np.nan, np.nan],  # Both NaN
        [5.0, 6.0]
    ])
    
    predictions = np.array([
        [1.1, 2.1],
        [2.9, 3.1],
        [3.1, 4.1],
        [4.9, 5.9],
        [5.1, 6.1]
    ])
    
    # Should compute metrics only on valid pairs
    smse = compute_smse(actuals, predictions)
    smae = compute_smae(actuals, predictions)
    
    # Should not be NaN or inf
    assert not np.isnan(smse), "sMSE should not be NaN"
    assert not np.isinf(smse), "sMSE should not be inf"
    assert not np.isnan(smae), "sMAE should not be NaN"
    assert not np.isinf(smae), "sMAE should not be inf"
    
    # Should be positive (unless perfect prediction)
    assert smse >= 0, "sMSE should be non-negative"
    assert smae >= 0, "sMAE should be non-negative"


def test_metrics_perfect_prediction():
    """Test metrics with perfect predictions."""
    actuals = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    predictions = actuals.copy()
    
    smse = compute_smse(actuals, predictions)
    smae = compute_smae(actuals, predictions)
    
    # Perfect predictions should give small values (may not be exactly 0 due to numerical precision)
    assert smse < 1e-10, "Perfect predictions should give near-zero sMSE"
    assert smae < 1e-10, "Perfect predictions should give near-zero sMAE"


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.skipif(not MODELS_AVAILABLE, reason=f"Models not available: {IMPORT_ERROR if not MODELS_AVAILABLE else ''}")
def test_full_pipeline_patchtst(sample_data, mock_data_loader):
    """Test full pipeline: train -> save -> load -> forecast."""
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'prediction_length': 4,
        'context_length': 32,  # Must be > patch_length
        'patch_len': 8,  # Smaller patch length
        'd_model': 64,
        'max_epochs': 1,
        'batch_size': 32,
        'validation_split': 0.0  # Disable validation
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        # Step 1: Train
        train_sktime_model(
            model_type='patchtst',
            cfg=None,
            data=sample_data,
            model_name='patchtst_pipeline_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        checkpoint_path = outputs_dir / "model.pkl"
        assert checkpoint_path.exists(), "Model should be saved"
        
        # Step 2: Load
        model = load_model_checkpoint(checkpoint_path)
        assert model is not None, "Model should load"
        
        # Step 3: Forecast
        test_data = sample_data.copy()
        predictions, actuals, dates, actual_targets = run_recursive_forecast(
            checkpoint_path=checkpoint_path,
            test_data=test_data,
            start_date='2020-10-01',
            end_date='2020-10-15',
            model_type='patchtst',
            target_series=['KOEQUIPTE'],
            data_loader=mock_data_loader
        )
        
        # Step 4: Compute metrics
        smse = compute_smse(actuals, predictions)
        smae = compute_smae(actuals, predictions)
        
        assert not np.isnan(smse), "sMSE should be valid"
        assert not np.isnan(smae), "sMAE should be valid"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
