"""Tests for PatchTST model training."""

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
    from neuralforecast.models import PatchTST
    PATCHTST_AVAILABLE = True
except ImportError as e:
    PATCHTST_AVAILABLE = False
    IMPORT_ERROR = str(e)

from src.train.patchtst import train_patchtst_model, _create_patchtst_model


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
    """Create a mock data loader."""
    class MockDataLoader:
        def __init__(self):
            dates = pd.date_range('2020-01-01', periods=150, freq='W')
            self.original = pd.DataFrame(
                np.random.randn(150, 3) * 10 + 100,
                columns=['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
                index=dates
            )
            self.processed = self.original.copy()
    return MockDataLoader()


@pytest.mark.skipif(not PATCHTST_AVAILABLE, reason=f"PatchTST not available: {IMPORT_ERROR if not PATCHTST_AVAILABLE else ''}")
def test_train_patchtst_basic(sample_data, mock_data_loader):
    """Test basic PatchTST model training."""
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'prediction_length': 4,
        'context_length': 96,
        'num_layers': 2,
        'max_epochs': 1,
        'batch_size': 32,
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        train_patchtst_model(
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


@pytest.mark.skipif(not PATCHTST_AVAILABLE, reason=f"PatchTST not available: {IMPORT_ERROR if not PATCHTST_AVAILABLE else ''}")
def test_create_patchtst_model():
    """Test PatchTST model creation."""
    model_params = {
        'prediction_length': 4,
        'context_length': 96,
        'num_layers': 2,
        'd_model': 128,
        'dropout': 0.1,
    }
    
    model = _create_patchtst_model(model_params, n_series=1)
    
    assert isinstance(model, NeuralForecast)
    assert len(model.models) == 1
    assert isinstance(model.models[0], PatchTST)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
