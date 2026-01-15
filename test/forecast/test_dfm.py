"""Tests for DFM forecasting pipeline with scaler support."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
import tempfile
import joblib
from sklearn.preprocessing import StandardScaler

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Check model availability
try:
    from dfm_python import DFM, DFMDataset
    from dfm_python.config import DFMConfig
    DFM_AVAILABLE = True
except ImportError as e:
    DFM_AVAILABLE = False
    IMPORT_ERROR = str(e)

from src.forecast.dfm import run_recursive_forecast, run_multi_horizon_forecast


@pytest.fixture
def sample_training_data():
    """Create sample training data (standardized)."""
    dates = pd.date_range('2020-01-01', periods=100, freq='W')
    data = pd.DataFrame(
        np.random.randn(100, 3),
        columns=['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
        index=dates
    )
    return data


@pytest.fixture
def sample_test_data():
    """Create sample test data (standardized)."""
    dates = pd.date_range('2023-01-01', periods=20, freq='W')
    data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
        index=dates
    )
    return data


@pytest.fixture
def trained_dfm_model(sample_training_data, tmp_path):
    """Create and train a DFM model for testing."""
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'ar_lag': 2,
        'ar_order': 2,
        'threshold': 1e-3,
        'max_iter': 5,
        'blocks': {
            'Block_Global': {
                'num_factors': 1,
                'series': ['KOEQUIPTE', 'KOWRCCNSE', 'A001']
            }
        },
        'frequency': {
            'w': ['KOEQUIPTE', 'KOWRCCNSE', 'A001']
        },
        'clock': 'w',
        'mixed_freq': False,
        'scaler': 'standard'
    }
    
    config = DFMConfig.from_dict(model_params)
    dataset = DFMDataset(config=config, data=sample_training_data, target_series=['KOEQUIPTE'])
    
    # Create and fit scaler on original data (simulating training pipeline)
    original_data = sample_training_data * 100 + 50  # Simulate original scale
    original_targets = original_data[['KOEQUIPTE']]
    target_scaler = StandardScaler()
    target_scaler.fit(original_targets.values)
    dataset.target_scaler = target_scaler
    
    # Train model
    model = DFM(config)
    X = dataset.get_processed_data()
    model.fit(X=X, dataset=dataset)
    
    # Save model and dataset
    checkpoint_path = tmp_path / "model.pkl"
    dataset_path = tmp_path / "dataset.pkl"
    model.save(checkpoint_path)
    joblib.dump(dataset, dataset_path)
    
    return checkpoint_path, dataset_path


@pytest.mark.skipif(not DFM_AVAILABLE, reason=f"DFM not available: {IMPORT_ERROR if not DFM_AVAILABLE else ''}")
def test_dfm_load_preserves_scaler(trained_dfm_model):
    """Test that loading DFM model preserves target_scaler."""
    checkpoint_path, dataset_path = trained_dfm_model
    
    # Load model
    model = DFM.load(checkpoint_path)
    
    # Verify target_scaler is preserved
    assert hasattr(model, 'target_scaler'), "Loaded model should have target_scaler"
    assert model.target_scaler is not None, "target_scaler should not be None"
    assert isinstance(model.target_scaler, StandardScaler), "target_scaler should be StandardScaler"


@pytest.mark.skipif(not DFM_AVAILABLE, reason=f"DFM not available: {IMPORT_ERROR if not DFM_AVAILABLE else ''}")
def test_dfm_recursive_forecast_with_scaler(trained_dfm_model, sample_test_data):
    """Test DFM recursive forecast pipeline with scaler."""
    checkpoint_path, dataset_path = trained_dfm_model
    
    # Run recursive forecast (horizon is fixed to 1 internally)
    predictions, actuals, dates = run_recursive_forecast(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        test_data=sample_test_data,
        start_date='2023-01-01',
        end_date='2023-01-15',
        target_series=['KOEQUIPTE']
    )
    
    # Verify outputs
    assert predictions is not None, "Predictions should not be None"
    assert actuals is not None, "Actuals should not be None"
    assert dates is not None, "Dates should not be None"
    assert len(predictions) > 0, "Should have predictions"
    assert len(actuals) > 0, "Should have actuals"
    assert len(dates) > 0, "Should have dates"


@pytest.mark.skipif(not DFM_AVAILABLE, reason=f"DFM not available: {IMPORT_ERROR if not DFM_AVAILABLE else ''}")
def test_dfm_multi_horizon_forecast_with_scaler(trained_dfm_model, sample_test_data):
    """Test DFM multi-horizon forecast pipeline with scaler."""
    checkpoint_path, dataset_path = trained_dfm_model
    
    # Run multi-horizon forecast
    forecasts_dict = run_multi_horizon_forecast(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        horizons=[4, 8],
        start_date='2023-01-01',
        test_data=sample_test_data,
        target_series=['KOEQUIPTE']
    )
    
    # Verify outputs
    assert forecasts_dict is not None, "Forecasts dict should not be None"
    assert len(forecasts_dict) > 0, "Should have forecasts for at least one horizon"
    assert 4 in forecasts_dict, "Should have forecast for horizon 4"
    assert 8 in forecasts_dict, "Should have forecast for horizon 8"


@pytest.mark.skipif(not DFM_AVAILABLE, reason=f"DFM not available: {IMPORT_ERROR if not DFM_AVAILABLE else ''}")
def test_dfm_scaler_inverse_transform(trained_dfm_model):
    """Test that target_scaler can be used for inverse transformation."""
    checkpoint_path, _ = trained_dfm_model
    
    # Load model
    model = DFM.load(checkpoint_path)
    
    # Create some scaled predictions (simulating model output)
    scaled_predictions = np.array([[0.5], [-0.3], [1.2]])  # Standardized values
    
    # Inverse transform using model's target_scaler
    if model.target_scaler is not None:
        original_scale_predictions = model.target_scaler.inverse_transform(scaled_predictions)
        
        # Verify inverse transformation worked
        assert original_scale_predictions.shape == scaled_predictions.shape
        # Values should be in original scale (mean ~50, std ~100)
        assert np.abs(original_scale_predictions.mean()) > 10, "Inverse transformed values should be in original scale"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
