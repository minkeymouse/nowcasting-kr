"""Tests for DFMDataset class."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from dfm_python.dataset.dfm_dataset import DFMDataset
    from dfm_python.config import DFMConfig
    DFM_AVAILABLE = True
except ImportError as e:
    DFM_AVAILABLE = False
    IMPORT_ERROR = str(e)


@pytest.fixture
def sample_data():
    """Create sample time series data."""
    dates = pd.date_range('2020-01-01', periods=100, freq='W')
    data = pd.DataFrame(
        np.random.randn(100, 3),
        columns=['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
        index=dates
    )
    data = data.reset_index()
    data.rename(columns={'index': 'date'}, inplace=True)
    return data


@pytest.fixture
def minimal_config():
    """Create minimal DFM config."""
    return {
        'clock': 'w',
        'blocks': {
            'Block_Global': {
                'num_factors': 1,
                'series': ['KOEQUIPTE', 'KOWRCCNSE', 'A001']
            }
        },
        'frequency': {
            'w': ['KOEQUIPTE', 'KOWRCCNSE', 'A001']
        }
    }


@pytest.mark.skipif(not DFM_AVAILABLE, reason=f"DFM not available: {IMPORT_ERROR if not DFM_AVAILABLE else ''}")
def test_dfm_dataset_initialization_basic(sample_data, minimal_config):
    """Test basic DFMDataset initialization."""
    config = DFMConfig.from_dict(minimal_config)
    dataset = DFMDataset(config=config, data=sample_data, time_index='date')
    
    assert dataset.config is not None
    assert dataset.variables is not None
    assert dataset.variables.shape[0] == len(sample_data)
    assert dataset.variables.shape[1] == 3
    assert dataset.time_index is not None


@pytest.mark.skipif(not DFM_AVAILABLE, reason=f"DFM not available: {IMPORT_ERROR if not DFM_AVAILABLE else ''}")
def test_dfm_dataset_time_index(sample_data, minimal_config):
    """Test DFMDataset time_index property."""
    config = DFMConfig.from_dict(minimal_config)
    dataset = DFMDataset(config=config, data=sample_data, time_index='date')
    
    assert dataset.time_index is not None
    assert len(dataset.time_index) == len(sample_data)


@pytest.mark.skipif(not DFM_AVAILABLE, reason=f"DFM not available: {IMPORT_ERROR if not DFM_AVAILABLE else ''}")
def test_dfm_dataset_variables(sample_data, minimal_config):
    """Test DFMDataset variables property."""
    config = DFMConfig.from_dict(minimal_config)
    dataset = DFMDataset(config=config, data=sample_data, time_index='date')
    
    variables = dataset.variables
    assert isinstance(variables, pd.DataFrame)
    assert list(variables.columns) == ['KOEQUIPTE', 'KOWRCCNSE', 'A001']


@pytest.mark.skipif(not DFM_AVAILABLE, reason=f"DFM not available: {IMPORT_ERROR if not DFM_AVAILABLE else ''}")
def test_dfm_dataset_missing_time_index(sample_data, minimal_config):
    """Test DFMDataset raises error when time_index column missing."""
    config = DFMConfig.from_dict(minimal_config)
    
    with pytest.raises(Exception):
        DFMDataset(config=config, data=sample_data, time_index='nonexistent')


@pytest.mark.skipif(not DFM_AVAILABLE, reason=f"DFM not available: {IMPORT_ERROR if not DFM_AVAILABLE else ''}")
def test_dfm_dataset_missing_data_error(minimal_config):
    """Test DFMDataset raises error when data not provided."""
    config = DFMConfig.from_dict(minimal_config)
    
    with pytest.raises(Exception):
        DFMDataset(config=config, data=None, time_index='date')


@pytest.mark.skipif(not DFM_AVAILABLE, reason=f"DFM not available: {IMPORT_ERROR if not DFM_AVAILABLE else ''}")
def test_dfm_dataset_column_order_consistency(sample_data, minimal_config):
    """Test that column order is preserved."""
    config = DFMConfig.from_dict(minimal_config)
    dataset = DFMDataset(config=config, data=sample_data, time_index='date')
    
    assert list(dataset.variables.columns) == ['KOEQUIPTE', 'KOWRCCNSE', 'A001']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
