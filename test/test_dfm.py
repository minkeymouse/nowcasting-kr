"""Comprehensive tests for DFM (Dynamic Factor Model).

Tests both training and forecasting functionality for DFM model.
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
    from dfm_python import DFM, DFMDataset
    from dfm_python.config import DFMConfig
    DFM_AVAILABLE = True
except ImportError as e:
    DFM_AVAILABLE = False
    IMPORT_ERROR = str(e)

from src.train.dfm import train_dfm_model
from src.forecast.dfm import (
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
            # Mock processed data
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

@pytest.mark.skipif(not DFM_AVAILABLE, reason=f"DFM not available: {IMPORT_ERROR if not DFM_AVAILABLE else ''}")
def test_train_dfm_basic(sample_data, mock_data_loader):
    """Test basic DFM model training."""
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'num_factors': 1,
        'p': 1,  # VAR order
        'max_iter': 5,  # Reduced for faster tests
        'threshold': 1e-3,
        'blocks': {
            'block1': {
                'series': ['KOEQUIPTE'],
                'num_factors': 1
            }
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        train_dfm_model(
            model_type='dfm',
            cfg=None,
            data=sample_data,
            model_name='dfm_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        # Verify model was saved
        model_path = outputs_dir / "model.pkl"
        assert model_path.exists(), "DFM model should be saved"
        
        # Verify dataset was saved
        dataset_path = outputs_dir / "dataset.pkl"
        assert dataset_path.exists(), "DFM dataset should be saved"
        
        # Verify model can be loaded
        model = DFM.load(model_path)
        assert model is not None, "Model should load successfully"
        assert hasattr(model, 'predict'), "Model should have predict method"


@pytest.mark.skipif(not DFM_AVAILABLE, reason=f"DFM not available: {IMPORT_ERROR if not DFM_AVAILABLE else ''}")
def test_train_dfm_multiple_targets(sample_data, mock_data_loader):
    """Test DFM training with multiple target series."""
    model_params = {
        'target_series': ['KOEQUIPTE', 'KOWRCCNSE'],
        'num_factors': 2,
        'p': 1,
        'max_iter': 5,
        'threshold': 1e-3,
        'blocks': {
            'block1': {
                'series': ['KOEQUIPTE', 'KOWRCCNSE'],
                'num_factors': 2
            }
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        train_dfm_model(
            model_type='dfm',
            cfg=None,
            data=sample_data,
            model_name='dfm_multiple_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        model_path = outputs_dir / "model.pkl"
        assert model_path.exists(), "DFM model should be saved"


@pytest.mark.skipif(not DFM_AVAILABLE, reason=f"DFM not available: {IMPORT_ERROR if not DFM_AVAILABLE else ''}")
def test_train_dfm_with_nans(sample_data, mock_data_loader):
    """Test that DFM can handle NaN values in training data."""
    # Create data with more NaNs
    data_with_nans = sample_data.copy()
    data_with_nans.loc[data_with_nans.index[:20], 'KOEQUIPTE'] = np.nan
    
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'num_factors': 1,
        'p': 1,
        'max_iter': 5,
        'threshold': 1e-3,
        'blocks': {
            'block1': {
                'series': ['KOEQUIPTE'],
                'num_factors': 1
            }
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        # Should not raise error - DFM handles NaNs internally
        train_dfm_model(
            model_type='dfm',
            cfg=None,
            data=data_with_nans,
            model_name='dfm_nan_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        model_path = outputs_dir / "model.pkl"
        assert model_path.exists(), "Training should succeed with NaNs"


# =============================================================================
# Forecasting Tests
# =============================================================================

@pytest.mark.skipif(not DFM_AVAILABLE, reason=f"DFM not available: {IMPORT_ERROR if not DFM_AVAILABLE else ''}")
def test_forecast_dfm_basic(sample_data, mock_data_loader):
    """Test basic DFM forecasting."""
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'num_factors': 1,
        'p': 1,
        'max_iter': 5,
        'threshold': 1e-3,
        'blocks': {
            'block1': {
                'series': ['KOEQUIPTE'],
                'num_factors': 1
            }
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        # Train model first
        train_dfm_model(
            model_type='dfm',
            cfg=None,
            data=sample_data,
            model_name='dfm_forecast_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        checkpoint_path = outputs_dir / "model.pkl"
        dataset_path = outputs_dir / "dataset.pkl"
        
        # Test basic forecast
        forecast(
            checkpoint_path=checkpoint_path,
            horizon=4,
            model_type='dfm',
            recursive=False
        )
        
        # Verify forecasts were saved
        forecasts_dir = outputs_dir / "forecasts"
        assert forecasts_dir.exists(), "Forecasts directory should be created"


@pytest.mark.skipif(not DFM_AVAILABLE, reason=f"DFM not available: {IMPORT_ERROR if not DFM_AVAILABLE else ''}")
def test_forecast_dfm_recursive(sample_data, mock_data_loader):
    """Test DFM recursive forecasting."""
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'num_factors': 1,
        'p': 1,
        'max_iter': 5,
        'threshold': 1e-3,
        'blocks': {
            'block1': {
                'series': ['KOEQUIPTE'],
                'num_factors': 1
            }
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        # Train model first
        train_dfm_model(
            model_type='dfm',
            cfg=None,
            data=sample_data,
            model_name='dfm_recursive_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        checkpoint_path = outputs_dir / "model.pkl"
        dataset_path = outputs_dir / "dataset.pkl"
        
        # Prepare test data
        test_data = sample_data.copy()
        
        # Run recursive forecast
        predictions, actuals, dates = run_recursive_forecast(
            checkpoint_path=checkpoint_path,
            dataset_path=dataset_path,
            test_data=test_data,
            start_date='2020-10-01',
            end_date='2020-11-01',
            model_type='dfm',
            target_series=['KOEQUIPTE'],
            data_loader=mock_data_loader
        )
        
        # Verify outputs
        assert len(predictions) > 0, "Should produce predictions"
        assert len(actuals) == len(predictions), "Actuals should match predictions"
        assert len(dates) == len(predictions), "Dates should match predictions"
        assert predictions.shape[1] == 1, "Should predict for one target series"


@pytest.mark.skipif(not DFM_AVAILABLE, reason=f"DFM not available: {IMPORT_ERROR if not DFM_AVAILABLE else ''}")
def test_forecast_dfm_multi_horizon(sample_data, mock_data_loader):
    """Test DFM multi-horizon forecasting."""
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'num_factors': 1,
        'p': 1,
        'max_iter': 5,
        'threshold': 1e-3,
        'blocks': {
            'block1': {
                'series': ['KOEQUIPTE'],
                'num_factors': 1
            }
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        # Train model first
        train_dfm_model(
            model_type='dfm',
            cfg=None,
            data=sample_data,
            model_name='dfm_multi_test',
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
            model_type='dfm',
            target_series=['KOEQUIPTE'],
            data_loader=mock_data_loader
        )
        
        # Verify outputs
        assert len(forecasts) == len(horizons), "Should have forecasts for all horizons"
        for horizon in horizons:
            assert horizon in forecasts, f"Should have forecast for horizon {horizon}"
            assert len(forecasts[horizon]) > 0, f"Forecast for horizon {horizon} should not be empty"
            assert forecasts[horizon].shape[0] == 1, f"Forecast should have correct shape for horizon {horizon}"


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.skipif(not DFM_AVAILABLE, reason=f"DFM not available: {IMPORT_ERROR if not DFM_AVAILABLE else ''}")
def test_full_pipeline_dfm(sample_data, mock_data_loader):
    """Test full pipeline: train -> save -> load -> forecast."""
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'num_factors': 1,
        'p': 1,
        'max_iter': 5,
        'threshold': 1e-3,
        'blocks': {
            'block1': {
                'series': ['KOEQUIPTE'],
                'num_factors': 1
            }
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        # Step 1: Train
        train_dfm_model(
            model_type='dfm',
            cfg=None,
            data=sample_data,
            model_name='dfm_pipeline_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        checkpoint_path = outputs_dir / "model.pkl"
        dataset_path = outputs_dir / "dataset.pkl"
        assert checkpoint_path.exists(), "Model should be saved"
        assert dataset_path.exists(), "Dataset should be saved"
        
        # Step 2: Load
        model = DFM.load(checkpoint_path)
        assert model is not None, "Model should load"
        
        # Step 3: Forecast
        test_data = sample_data.copy()
        predictions, actuals, dates = run_recursive_forecast(
            checkpoint_path=checkpoint_path,
            dataset_path=dataset_path,
            test_data=test_data,
            start_date='2020-10-01',
            end_date='2020-10-15',
            model_type='dfm',
            target_series=['KOEQUIPTE'],
            data_loader=mock_data_loader
        )
        
        # Step 4: Compute metrics
        smse = compute_smse(actuals, predictions)
        smae = compute_smae(actuals, predictions)
        
        assert not np.isnan(smse), "sMSE should be valid"
        assert not np.isnan(smae), "sMAE should be valid"
        assert smse >= 0, "sMSE should be non-negative"
        assert smae >= 0, "sMAE should be non-negative"


@pytest.mark.skipif(not DFM_AVAILABLE, reason=f"DFM not available: {IMPORT_ERROR if not DFM_AVAILABLE else ''}")
def test_dfm_model_update(sample_data, mock_data_loader):
    """Test that DFM model can be updated with new data."""
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'num_factors': 1,
        'p': 1,
        'max_iter': 5,
        'threshold': 1e-3,
        'blocks': {
            'block1': {
                'series': ['KOEQUIPTE'],
                'num_factors': 1
            }
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        # Train model
        train_dfm_model(
            model_type='dfm',
            cfg=None,
            data=sample_data,
            model_name='dfm_update_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        checkpoint_path = outputs_dir / "model.pkl"
        model = DFM.load(checkpoint_path)
        
        # Test update with new data
        new_data = sample_data.iloc[-10:].copy()
        from src.utils import preprocess_data_for_model
        new_data_processed = preprocess_data_for_model(
            new_data, mock_data_loader, columns=None, impute_missing=False
        )
        
        # Should be able to update model
        model.update(new_data_processed)
        
        # Should be able to forecast after update
        predictions = model.predict(horizon=4)
        assert predictions is not None, "Should be able to forecast after update"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
