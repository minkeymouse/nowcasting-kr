"""Tests for helper functions."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.helper import (
    find_checkpoint_path,
    determine_experiment_type,
    parse_experiment_config,
    infer_frequency,
    extract_target_series_from_config
)


def test_find_checkpoint_path_pkl(tmp_path):
    """Test finding .pkl checkpoint."""
    checkpoint_path = tmp_path / "model.pkl"
    checkpoint_path.touch()
    
    result = find_checkpoint_path(tmp_path)
    assert result == checkpoint_path


def test_find_checkpoint_path_zip(tmp_path):
    """Test finding .zip checkpoint when .pkl doesn't exist."""
    checkpoint_path = tmp_path / "model.zip"
    checkpoint_path.touch()
    
    result = find_checkpoint_path(tmp_path)
    assert result == checkpoint_path


def test_find_checkpoint_path_not_found(tmp_path):
    """Test error when checkpoint not found."""
    with pytest.raises(FileNotFoundError):
        find_checkpoint_path(tmp_path)


def test_determine_experiment_type_short_term_string():
    """Test determining experiment type from string."""
    result = determine_experiment_type("short_term")
    assert result == "short_term"


def test_determine_experiment_type_long_term_string():
    """Test determining experiment type from string."""
    result = determine_experiment_type("long_term")
    assert result == "long_term"


def test_determine_experiment_type_short_term_dict():
    """Test determining experiment type from dict with horizon."""
    config = {'horizon': 1, 'start_date': '2020-01-01'}
    result = determine_experiment_type(config)
    assert result == "short_term"


def test_determine_experiment_type_long_term_dict():
    """Test determining experiment type from dict with horizons."""
    config = {'horizons': [4, 8, 12], 'start_date': '2020-01-01'}
    result = determine_experiment_type(config)
    assert result == "long_term"


def test_parse_experiment_config_short_term_string():
    """Test parsing short_term experiment config from string."""
    config = parse_experiment_config("short_term", "short_term")
    
    assert 'start_date' in config
    assert 'end_date' in config
    assert 'update_params' in config


def test_parse_experiment_config_long_term_dict():
    """Test parsing long_term experiment config from dict."""
    config_dict = {
        'start_date': '2020-01-01',
        'horizons': [4, 8, 12, 16, 20]
    }
    
    config = parse_experiment_config(config_dict, "long_term")
    
    assert config['start_date'] == '2020-01-01'
    assert config['horizons'] == [4, 8, 12, 16, 20]


def test_infer_frequency_from_dataframe():
    """Test frequency inference from DataFrame."""
    dates = pd.date_range('2020-01-01', periods=100, freq='W')
    df = pd.DataFrame(np.random.randn(100, 3), index=dates)
    
    freq = infer_frequency(df)
    assert freq == 'W' or 'W' in freq


def test_infer_frequency_default():
    """Test frequency inference with default."""
    df = pd.DataFrame(np.random.randn(100, 3))
    
    freq = infer_frequency(df, default='W')
    assert freq == 'W'


def test_extract_target_series_from_config_dict():
    """Test extracting target series from dict config."""
    cfg = {
        'model': {
            'target_series': ['KOEQUIPTE', 'KOWRCCNSE']
        }
    }
    
    result = extract_target_series_from_config(cfg)
    assert result == ['KOEQUIPTE', 'KOWRCCNSE']


def test_extract_target_series_from_config_none():
    """Test extracting target series when not present."""
    cfg = {'model': {}}
    
    result = extract_target_series_from_config(cfg)
    assert result is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
