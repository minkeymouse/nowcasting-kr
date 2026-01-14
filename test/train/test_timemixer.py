"""Tests for TimeMixer model training."""

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
    from neuralforecast import NeuralForecast
    from neuralforecast.models import TimeMixer
    TIMEMIXER_AVAILABLE = True
except ImportError as e:
    TIMEMIXER_AVAILABLE = False
    IMPORT_ERROR = str(e)

from src.train.timemixer import train_timemixer_model, _create_timemixer_model


@pytest.fixture
def sample_data():
    """Create sample time series data."""
    dates = pd.date_range('2020-01-01', periods=150, freq='W')
    return pd.DataFrame(
        np.random.randn(150, 3),
        columns=['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
        index=dates
    )


@pytest.fixture
def mock_data_loader():
    """Create a mock data loader with metadata."""
    class MockDataLoader:
        def __init__(self):
            dates = pd.date_range('2020-01-01', periods=150, freq='W')
            self.original = pd.DataFrame(
                np.random.randn(150, 3) * 10 + 100,
                columns=['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
                index=dates
            )
            self.processed = self.original.copy()
            self.standardized = pd.DataFrame(
                np.random.randn(150, 3),
                columns=['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
                index=dates
            )
            # Metadata for monthly series detection
            self.metadata = pd.DataFrame({
                'Series_ID': ['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
                'Frequency': ['M', 'W', 'W']  # One monthly series
            })
    return MockDataLoader()


@pytest.mark.skipif(not TIMEMIXER_AVAILABLE, reason=f"TimeMixer not available: {IMPORT_ERROR if not TIMEMIXER_AVAILABLE else ''}")
def test_train_timemixer_basic(sample_data, mock_data_loader):
    """Test basic TimeMixer model training."""
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'prediction_length': 4,
        'context_length': 96,
        'n_layers': 2,
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
            model_name='timemixer_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        # Verify model was saved
        model_path = outputs_dir / "model.pkl"
        assert model_path.exists(), "TimeMixer model should be saved"


@pytest.mark.skipif(not TIMEMIXER_AVAILABLE, reason=f"TimeMixer not available: {IMPORT_ERROR if not TIMEMIXER_AVAILABLE else ''}")
def test_create_timemixer_model(mock_data_loader):
    """Test TimeMixer model creation."""
    model_params = {
        'prediction_length': 4,
        'context_length': 96,
        'n_layers': 2,
        'd_model': 32,
        'dropout': 0.1,
    }
    
    model = _create_timemixer_model(model_params, n_series=1, data_loader=mock_data_loader)
    
    assert isinstance(model, NeuralForecast)
    assert len(model.models) == 1
    assert isinstance(model.models[0], TimeMixer)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
