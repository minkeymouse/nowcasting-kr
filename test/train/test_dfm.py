"""Tests for DFM model training pipeline with scaler support."""

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

from src.train.dfm import train_dfm_model


@pytest.fixture
def sample_data():
    """Create sample time series data (standardized)."""
    dates = pd.date_range('2020-01-01', periods=150, freq='W')
    # Create standardized data (mean≈0, std≈1)
    data = pd.DataFrame(
        np.random.randn(150, 3),
        columns=['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
        index=dates
    )
    return data


@pytest.fixture
def original_data():
    """Create original (unstandardized) data for scaler fitting."""
    dates = pd.date_range('2020-01-01', periods=150, freq='W')
    # Create non-standardized data (different scale)
    data = pd.DataFrame(
        np.random.randn(150, 3) * 100 + 50,  # Mean ~50, std ~100
        columns=['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
        index=dates
    )
    return data


@pytest.fixture
def mock_data_loader(original_data):
    """Create a mock data loader with training_data attribute."""
    class MockDataLoader:
        def __init__(self, training_data):
            self.training_data = training_data
            self.metadata = pd.DataFrame({
                'Series_ID': ['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
                'Frequency': ['M', 'W', 'W']
            })
    return MockDataLoader(original_data)


@pytest.mark.skipif(not DFM_AVAILABLE, reason=f"DFM not available: {IMPORT_ERROR if not DFM_AVAILABLE else ''}")
def test_train_dfm_with_scaler(sample_data, mock_data_loader):
    """Test DFM training with target_scaler creation and assignment."""
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'ar_lag': 2,
        'ar_order': 2,
        'threshold': 1e-3,
        'max_iter': 5,  # Small number for quick test
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
        'scaler': 'standard',
        'target_scaler': 'standard'
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        # Train model
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
        assert model_path.exists(), "Model checkpoint should be saved"
        
        # Verify dataset was saved
        dataset_path = outputs_dir / "dataset.pkl"
        assert dataset_path.exists(), "Dataset should be saved"
        
        # Load model and verify target_scaler is present
        model = DFM.load(model_path)
        assert hasattr(model, 'target_scaler'), "Model should have target_scaler attribute"
        assert model.target_scaler is not None, "target_scaler should not be None"
        assert isinstance(model.target_scaler, StandardScaler), "target_scaler should be StandardScaler"
        
        # Verify scaler was fitted on original data (not standardized)
        # Original data has mean ~50, std ~100, so scaler should reflect that
        scaler_mean = model.target_scaler.mean_
        scaler_scale = model.target_scaler.scale_
        assert len(scaler_mean) == 1, "Scaler should be fitted on 1 target series"
        assert len(scaler_scale) == 1, "Scaler should be fitted on 1 target series"
        # Mean should be around 50 (original data scale), not 0 (standardized)
        assert abs(scaler_mean[0]) > 10, f"Scaler mean should reflect original data scale, got {scaler_mean[0]}"
        assert scaler_scale[0] > 10, f"Scaler scale should reflect original data scale, got {scaler_scale[0]}"


@pytest.mark.skipif(not DFM_AVAILABLE, reason=f"DFM not available: {IMPORT_ERROR if not DFM_AVAILABLE else ''}")
def test_train_dfm_without_data_loader(sample_data):
    """Test DFM training without data_loader (no scaler)."""
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
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        # Train model without data_loader
        train_dfm_model(
            model_type='dfm',
            cfg=None,
            data=sample_data,
            model_name='dfm_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=None
        )
        
        # Verify model was saved
        model_path = outputs_dir / "model.pkl"
        assert model_path.exists(), "Model checkpoint should be saved"
        
        # Load model - target_scaler may be None if not provided
        model = DFM.load(model_path)
        # target_scaler can be None if data_loader was not provided
        # This is acceptable - model can still work without scaler


@pytest.mark.skipif(not DFM_AVAILABLE, reason=f"DFM not available: {IMPORT_ERROR if not DFM_AVAILABLE else ''}")
def test_dfm_dataset_scaler_assignment(sample_data, mock_data_loader):
    """Test that target_scaler is properly assigned to DFMDataset."""
    from dfm_python.config import DFMConfig
    
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'ar_lag': 2,
        'ar_order': 2,
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
        'scaler': 'standard',
        'target_scaler': 'standard'
    }
    
    config = DFMConfig.from_dict(model_params)
    dataset = DFMDataset(config=config, data=sample_data, target_series=['KOEQUIPTE'])
    
    # Create and fit scaler (mimicking train_dfm_model)
    original_targets = mock_data_loader.training_data[['KOEQUIPTE']]
    original_targets = original_targets.select_dtypes(include=[np.number]).reset_index(drop=True)
    
    target_scaler = StandardScaler()
    target_scaler.fit(original_targets.to_numpy(dtype=np.float64))
    
    # Assign scaler to dataset
    dataset.target_scaler = target_scaler
    
    # Verify scaler is accessible via get_initialization_params
    init_params = dataset.get_initialization_params()
    assert 'target_scaler' in init_params, "get_initialization_params should return target_scaler"
    assert init_params['target_scaler'] is target_scaler, "target_scaler should match assigned scaler"


@pytest.mark.skipif(not DFM_AVAILABLE, reason=f"DFM not available: {IMPORT_ERROR if not DFM_AVAILABLE else ''}")
def test_dfm_model_retrieves_scaler_from_dataset(sample_data, mock_data_loader):
    """Test that DFM.fit() retrieves target_scaler from dataset."""
    from dfm_python.config import DFMConfig
    
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'ar_lag': 2,
        'ar_order': 2,
        'threshold': 1e-3,
        'max_iter': 3,
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
        'scaler': 'standard',
        'target_scaler': 'standard'
    }
    
    config = DFMConfig.from_dict(model_params)
    dataset = DFMDataset(config=config, data=sample_data, target_series=['KOEQUIPTE'])
    
    # Create and fit scaler
    original_targets = mock_data_loader.training_data[['KOEQUIPTE']]
    original_targets = original_targets.select_dtypes(include=[np.number]).reset_index(drop=True)
    
    target_scaler = StandardScaler()
    target_scaler.fit(original_targets.to_numpy(dtype=np.float64))
    
    # Assign scaler to dataset
    dataset.target_scaler = target_scaler
    
    # Create and fit model
    model = DFM(config)
    X = dataset.get_processed_data()
    model.fit(X=X, dataset=dataset)
    
    # Verify model has target_scaler
    assert hasattr(model, 'target_scaler'), "Model should have target_scaler after fit()"
    assert model.target_scaler is target_scaler, "Model target_scaler should match dataset target_scaler"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
