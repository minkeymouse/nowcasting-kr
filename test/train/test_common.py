"""Tests for common training utilities."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.train._common import (
    prepare_training_data,
    get_common_training_params,
    get_processed_data_from_loader,
    save_model_checkpoint,
    train_neuralforecast_model
)


@pytest.fixture
def sample_data():
    """Create sample time series data."""
    dates = pd.date_range('2020-01-01', periods=100, freq='W')
    return pd.DataFrame(
        np.random.randn(100, 3),
        columns=['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
        index=dates
    )


@pytest.fixture
def mock_data_loader():
    """Create a mock data loader."""
    class MockDataLoader:
        def __init__(self):
            dates = pd.date_range('2020-01-01', periods=100, freq='W')
            self.original = pd.DataFrame(
                np.random.randn(100, 3) * 10 + 100,
                columns=['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
                index=dates
            )
            self.processed = self.original.copy()
            self.standardized = pd.DataFrame(
                np.random.randn(100, 3),
                columns=['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
                index=dates
            )
    return MockDataLoader()


def test_prepare_training_data_basic(sample_data):
    """Test basic training data preparation."""
    model_params = {'target_series': ['KOEQUIPTE', 'KOWRCCNSE']}
    
    target_data, available_targets = prepare_training_data(
        sample_data, model_params, None
    )
    
    assert len(available_targets) == 2
    assert 'KOEQUIPTE' in available_targets
    assert 'KOWRCCNSE' in available_targets
    assert target_data.shape[1] == 2
    assert isinstance(target_data.index, pd.DatetimeIndex)


def test_prepare_training_data_all_columns(sample_data):
    """Test training data preparation when no target_series specified."""
    target_data, available_targets = prepare_training_data(
        sample_data, None, None
    )
    
    assert len(available_targets) == 3
    assert target_data.shape[1] == 3


def test_prepare_training_data_missing_series(sample_data):
    """Test training data preparation with missing target series."""
    model_params = {'target_series': ['KOEQUIPTE', 'MISSING_SERIES']}
    
    target_data, available_targets = prepare_training_data(
        sample_data, model_params, None
    )
    
    # Should use available series only
    assert len(available_targets) == 1
    assert 'KOEQUIPTE' in available_targets


def test_get_common_training_params():
    """Test common training parameters extraction."""
    model_params = {
        'prediction_length': 4,
        'context_length': 96,
        'batch_size': 128,
        'learning_rate': 0.001,
        'max_epochs': 50
    }
    
    common_params = get_common_training_params(model_params)
    
    assert common_params['h'] == 4
    assert common_params['input_size'] == 96
    assert common_params['batch_size'] == 128
    assert common_params['learning_rate'] == 0.001
    assert common_params['max_steps'] == 5000  # 50 * 100


def test_get_common_training_params_fallback():
    """Test common training parameters with fallbacks."""
    model_params = {
        'horizon': 1,  # Fallback for prediction_length
        'n_lags': 48,  # Fallback for context_length
        'max_epochs': 10
    }
    
    common_params = get_common_training_params(model_params)
    
    assert common_params['h'] == 1
    assert common_params['input_size'] == 48
    assert common_params['max_steps'] == 1000  # 10 * 100


def test_get_processed_data_from_loader(mock_data_loader):
    """Test getting processed data from loader."""
    data = pd.DataFrame(np.random.randn(100, 3), columns=['KOEQUIPTE', 'KOWRCCNSE', 'A001'])
    
    result = get_processed_data_from_loader(data, mock_data_loader, "TestModel")
    
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 100
    # Should use original data (raw, unstandardized)
    assert not np.allclose(result.values, data.values)  # Different from input


def test_get_processed_data_from_loader_no_loader():
    """Test getting processed data when no loader available."""
    data = pd.DataFrame(np.random.randn(100, 3), columns=['KOEQUIPTE', 'KOWRCCNSE', 'A001'])
    
    result = get_processed_data_from_loader(data, None, "TestModel")
    
    # Should return data as-is
    assert isinstance(result, pd.DataFrame)
    assert result.shape == data.shape


def test_save_model_checkpoint():
    """Test saving model checkpoint."""
    class MockModel:
        def __init__(self):
            self.value = 42
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.pkl"
        model = MockModel()
        
        save_model_checkpoint(model, model_path, "TestModel")
        
        assert model_path.exists()
        # Verify it can be loaded
        import joblib
        loaded = joblib.load(model_path)
        assert loaded.value == 42


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
