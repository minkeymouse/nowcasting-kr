"""Tests for utility functions."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import (
    is_neuralforecast_model,
    convert_to_neuralforecast_format,
    interpolate_missing_values,
    load_model_checkpoint,
    get_monthly_series_from_metadata
)


def test_is_neuralforecast_model():
    """Test NeuralForecast model detection."""
    assert is_neuralforecast_model('patchtst') == True
    assert is_neuralforecast_model('tft') == True
    assert is_neuralforecast_model('itf') == True
    assert is_neuralforecast_model('itransformer') == True
    assert is_neuralforecast_model('timemixer') == True
    assert is_neuralforecast_model('dfm') == False
    assert is_neuralforecast_model('ddfm') == False


def test_convert_to_neuralforecast_format():
    """Test conversion to NeuralForecast format."""
    dates = pd.date_range('2020-01-01', periods=100, freq='W')
    data = pd.DataFrame(
        np.random.randn(100, 3),
        columns=['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
        index=dates
    )
    
    target_series = ['KOEQUIPTE', 'KOWRCCNSE']
    nf_df = convert_to_neuralforecast_format(data, target_series)
    
    assert 'unique_id' in nf_df.columns
    assert 'ds' in nf_df.columns
    assert 'y' in nf_df.columns
    assert len(nf_df) == 100 * len(target_series)
    assert set(nf_df['unique_id'].unique()) == set(target_series)


def test_interpolate_missing_values():
    """Test missing value interpolation."""
    dates = pd.date_range('2020-01-01', periods=100, freq='W')
    data = pd.DataFrame(
        np.random.randn(100, 3),
        columns=['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
        index=dates
    )
    
    # Add some NaNs
    data.loc[data.index[10:15], 'KOEQUIPTE'] = np.nan
    
    result = interpolate_missing_values(data, None)
    
    assert not result['KOEQUIPTE'].isna().any()
    assert result.shape == data.shape


def test_load_model_checkpoint(tmp_path):
    """Test loading model checkpoint."""
    import joblib
    
    class MockModel:
        def __init__(self):
            self.value = 42
    
    model = MockModel()
    model_path = tmp_path / "model.pkl"
    joblib.dump(model, model_path)
    
    loaded = load_model_checkpoint(model_path)
    assert loaded.value == 42


def test_get_monthly_series_from_metadata():
    """Test monthly series extraction from metadata."""
    metadata = pd.DataFrame({
        'Series_ID': ['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
        'Frequency': ['M', 'W', 'W']
    })
    
    class MockDataLoader:
        def __init__(self):
            self.metadata = metadata
    
    monthly_series = get_monthly_series_from_metadata(MockDataLoader())
    
    assert isinstance(monthly_series, set)
    assert 'KOEQUIPTE' in monthly_series
    assert 'KOWRCCNSE' not in monthly_series
    assert 'A001' not in monthly_series


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
