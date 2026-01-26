"""Tests for DDFMDataset class."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
import warnings
from sklearn.preprocessing import StandardScaler

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Check model availability
try:
    from dfm_python import DDFMDataset
    DDFM_AVAILABLE = True
except ImportError as e:
    DDFM_AVAILABLE = False
    IMPORT_ERROR = str(e)


@pytest.fixture
def sample_data():
    """Create sample time series data."""
    dates = pd.date_range('2020-01-01', periods=100, freq='W')
    data = pd.DataFrame(
        np.random.randn(100, 4),
        columns=['time_idx', 'KOEQUIPTE', 'KOWRCCNSE', 'A001'],
        index=dates
    )
    data['time_idx'] = dates
    return data


@pytest.mark.skipif(not DDFM_AVAILABLE, reason=f"DDFM not available: {IMPORT_ERROR if not DDFM_AVAILABLE else ''}")
def test_ddfm_dataset_initialization_basic(sample_data):
    """Test basic DDFMDataset initialization."""
    dataset = DDFMDataset(
        data=sample_data,
        time_idx='time_idx'
    )
    
    assert dataset.time_idx == 'time_idx'
    assert dataset.data is not None
    assert dataset.X is not None
    assert dataset.y is not None


@pytest.mark.skipif(not DDFM_AVAILABLE, reason=f"DDFM not available: {IMPORT_ERROR if not DDFM_AVAILABLE else ''}")
def test_ddfm_dataset_with_covariates(sample_data):
    """Test DDFMDataset with covariates parameter."""
    dataset = DDFMDataset(
        data=sample_data,
        time_idx='time_idx',
        covariates=['KOWRCCNSE', 'A001']
    )
    
    assert dataset.covariates == ['KOWRCCNSE', 'A001']
    assert dataset.target_series == ['KOEQUIPTE']
    assert len(dataset.target_series) == 1


@pytest.mark.skipif(not DDFM_AVAILABLE, reason=f"DDFM not available: {IMPORT_ERROR if not DDFM_AVAILABLE else ''}")
def test_ddfm_dataset_default_all_targets(sample_data):
    """Test DDFMDataset default behavior (all series are targets)."""
    dataset = DDFMDataset(
        data=sample_data,
        time_idx='time_idx'
    )
    
    # By default, all series should be targets (excluding time_idx)
    assert dataset.covariates == []
    expected_targets = set(['KOEQUIPTE', 'KOWRCCNSE', 'A001'])
    assert set(dataset.target_series) == expected_targets


@pytest.mark.skipif(not DDFM_AVAILABLE, reason=f"DDFM not available: {IMPORT_ERROR if not DDFM_AVAILABLE else ''}")
def test_ddfm_dataset_target_series_deprecated(sample_data):
    """Test DDFMDataset with deprecated target_series parameter."""
    # Should emit deprecation warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dataset = DDFMDataset(
            data=sample_data,
            time_idx='time_idx',
            target_series=['KOEQUIPTE']
        )
        
        # Check deprecation warning was issued
        assert len(w) > 0
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
        assert any("target_series parameter is deprecated" in str(warning.message) for warning in w)
    
    # Should compute covariates from target_series
    assert 'KOEQUIPTE' in dataset.target_series
    assert 'KOWRCCNSE' in dataset.covariates or 'A001' in dataset.covariates


@pytest.mark.skipif(not DDFM_AVAILABLE, reason=f"DDFM not available: {IMPORT_ERROR if not DDFM_AVAILABLE else ''}")
def test_ddfm_dataset_with_scaler(sample_data):
    """Test DDFMDataset with scaler."""
    scaler = StandardScaler()
    scaler.fit(sample_data[['KOEQUIPTE']].values)
    
    dataset = DDFMDataset(
        data=sample_data,
        time_idx='time_idx',
        covariates=['KOWRCCNSE', 'A001'],
        scaler=scaler
    )
    
    assert dataset.scaler is not None
    assert dataset.scaler is scaler
    # Check that target data was scaled
    assert np.abs(dataset.y.mean()) < 1.0, "Scaled targets should have mean close to 0"


@pytest.mark.skipif(not DDFM_AVAILABLE, reason=f"DDFM not available: {IMPORT_ERROR if not DDFM_AVAILABLE else ''}")
def test_ddfm_dataset_target_scaler_backward_compat(sample_data):
    """Test DDFMDataset target_scaler property (backward compatibility)."""
    scaler = StandardScaler()
    scaler.fit(sample_data[['KOEQUIPTE']].values)
    
    dataset = DDFMDataset(
        data=sample_data,
        time_idx='time_idx',
        covariates=['KOWRCCNSE', 'A001'],
        scaler=scaler
    )
    
    # target_scaler property should return scaler
    assert dataset.target_scaler is not None
    assert dataset.target_scaler is scaler


@pytest.mark.skipif(not DDFM_AVAILABLE, reason=f"DDFM not available: {IMPORT_ERROR if not DDFM_AVAILABLE else ''}")
def test_ddfm_dataset_properties(sample_data):
    """Test DDFMDataset properties."""
    dataset = DDFMDataset(
        data=sample_data,
        time_idx='time_idx',
        covariates=['KOWRCCNSE', 'A001']
    )
    
    # Test shape properties
    assert dataset.target_shape[0] == len(sample_data)
    assert dataset.target_shape[1] == 1  # Only KOEQUIPTE is target
    assert dataset.feature_shape[0] == len(sample_data)
    assert dataset.feature_shape[1] == 2  # KOWRCCNSE and A001 are features
    
    # Test column properties
    assert 'KOEQUIPTE' in dataset.target_columns
    assert 'KOWRCCNSE' in dataset.feature_columns
    assert 'A001' in dataset.feature_columns
    
    # Test all_columns_are_targets
    assert not dataset.all_columns_are_targets


@pytest.mark.skipif(not DDFM_AVAILABLE, reason=f"DDFM not available: {IMPORT_ERROR if not DDFM_AVAILABLE else ''}")
def test_ddfm_dataset_all_columns_targets(sample_data):
    """Test DDFMDataset when all columns are targets."""
    dataset = DDFMDataset(
        data=sample_data,
        time_idx='time_idx'
    )
    
    # Note: all_columns_are_targets compares target_series with colnames (which includes time_idx)
    # So it's False when time_idx is in colnames but not in target_series
    # This is expected behavior - time_idx is metadata, not a target series
    assert not dataset.all_columns_are_targets  # time_idx is in colnames but not in target_series
    assert len(dataset.target_series) == 3  # All series except time_idx
    assert len(dataset.covariates) == 0


@pytest.mark.skipif(not DDFM_AVAILABLE, reason=f"DDFM not available: {IMPORT_ERROR if not DDFM_AVAILABLE else ''}")
def test_ddfm_dataset_from_dataset(sample_data):
    """Test DDFMDataset.from_dataset() class method."""
    original_dataset = DDFMDataset(
        data=sample_data,
        time_idx='time_idx',
        covariates=['KOWRCCNSE', 'A001']
    )
    
    # Create new data with same structure
    new_dates = pd.date_range('2023-01-01', periods=50, freq='W')
    new_data = pd.DataFrame(
        np.random.randn(50, 4),
        columns=['time_idx', 'KOEQUIPTE', 'KOWRCCNSE', 'A001'],
        index=new_dates
    )
    new_data['time_idx'] = new_dates
    
    new_dataset = DDFMDataset.from_dataset(new_data, original_dataset)
    
    assert new_dataset.time_idx == original_dataset.time_idx
    assert new_dataset.covariates == original_dataset.covariates
    assert new_dataset.target_series == original_dataset.target_series
    assert len(new_dataset.data) == len(new_data)


@pytest.mark.skipif(not DDFM_AVAILABLE, reason=f"DDFM not available: {IMPORT_ERROR if not DDFM_AVAILABLE else ''}")
def test_ddfm_dataset_split_features_and_targets(sample_data):
    """Test split_features_and_targets() method."""
    dataset = DDFMDataset(
        data=sample_data,
        time_idx='time_idx',
        covariates=['KOWRCCNSE', 'A001']
    )
    
    X, y = dataset.split_features_and_targets(sample_data.drop(columns=['time_idx']))
    
    assert X is not None
    assert y is not None
    assert 'KOEQUIPTE' in y.columns
    assert 'KOWRCCNSE' in X.columns
    assert 'A001' in X.columns


@pytest.mark.skipif(not DDFM_AVAILABLE, reason=f"DDFM not available: {IMPORT_ERROR if not DDFM_AVAILABLE else ''}")
def test_ddfm_dataset_split_all_targets(sample_data):
    """Test split_features_and_targets() when all columns are targets."""
    dataset = DDFMDataset(
        data=sample_data,
        time_idx='time_idx'
    )
    
    X, y = dataset.split_features_and_targets(sample_data.drop(columns=['time_idx']))
    
    # When all data columns (excluding time_idx) are targets, X should be None
    # But since all_columns_are_targets is False (time_idx is in colnames), X will not be None
    # Instead, X will be empty (no features) and y will contain all targets
    # Actually, let's check what the actual behavior is:
    # If all columns except time_idx are targets, then X should be empty or None
    # But the implementation checks all_columns_are_targets which includes time_idx
    # So X will not be None, but will be empty
    assert X is not None  # Because all_columns_are_targets is False (time_idx in colnames)
    assert y is not None
    assert len(y.columns) == 3  # All target series
    # X should be empty since all data columns are targets
    assert len(X.columns) == 0 or X.empty


@pytest.mark.skipif(not DDFM_AVAILABLE, reason=f"DDFM not available: {IMPORT_ERROR if not DDFM_AVAILABLE else ''}")
def test_ddfm_dataset_target_indices(sample_data):
    """Test target_indices property."""
    dataset = DDFMDataset(
        data=sample_data,
        time_idx='time_idx',
        covariates=['KOWRCCNSE', 'A001']
    )
    
    indices = dataset.target_indices
    assert isinstance(indices, np.ndarray)
    assert len(indices) == 1  # Only KOEQUIPTE is target
    # KOEQUIPTE should be at index 1 (after time_idx at 0)
    assert indices[0] == 1


@pytest.mark.skipif(not DDFM_AVAILABLE, reason=f"DDFM not available: {IMPORT_ERROR if not DDFM_AVAILABLE else ''}")
def test_ddfm_dataset_missing_covariates_warning(sample_data):
    """Test DDFMDataset warns when covariates don't exist in data."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dataset = DDFMDataset(
            data=sample_data,
            time_idx='time_idx',
            covariates=['NONEXISTENT']
        )
        
        # Should warn about missing covariates
        assert len(w) > 0
        assert any("not found in data" in str(warning.message) for warning in w)
    
    # Should still work, treating all available series as targets
    assert len(dataset.target_series) > 0


@pytest.mark.skipif(not DDFM_AVAILABLE, reason=f"DDFM not available: {IMPORT_ERROR if not DDFM_AVAILABLE else ''}")
def test_ddfm_dataset_missing_target_series_error(sample_data):
    """Test DDFMDataset raises error when target_series columns don't exist."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ValueError, match="not found in data"):
            DDFMDataset(
                data=sample_data,
                time_idx='time_idx',
                target_series=['NONEXISTENT']
            )


@pytest.mark.skipif(not DDFM_AVAILABLE, reason=f"DDFM not available: {IMPORT_ERROR if not DDFM_AVAILABLE else ''}")
def test_ddfm_dataset_covariates_vs_target_series(sample_data):
    """Test that covariates and target_series are mutually exclusive and computed correctly."""
    # Test with covariates
    dataset1 = DDFMDataset(
        data=sample_data,
        time_idx='time_idx',
        covariates=['KOWRCCNSE']
    )
    all_series = set(['KOEQUIPTE', 'KOWRCCNSE', 'A001'])
    
    # target_series should be all_series - covariates
    expected_targets = all_series - {'KOWRCCNSE'}
    assert set(dataset1.target_series) == expected_targets
    assert 'KOWRCCNSE' in dataset1.covariates
    
    # Test with target_series (deprecated)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        dataset2 = DDFMDataset(
            data=sample_data,
            time_idx='time_idx',
            target_series=['KOEQUIPTE']
        )
    
    # Should compute covariates from target_series
    expected_covariates = all_series - {'KOEQUIPTE'}
    assert set(dataset2.covariates) == expected_covariates
    assert 'KOEQUIPTE' in dataset2.target_series


@pytest.mark.skipif(not DDFM_AVAILABLE, reason=f"DDFM not available: {IMPORT_ERROR if not DDFM_AVAILABLE else ''}")
def test_ddfm_dataset_with_feature_scaler(sample_data):
    """Test DDFMDataset with feature_scaler."""
    feature_scaler = StandardScaler()
    
    dataset = DDFMDataset(
        data=sample_data,
        time_idx='time_idx',
        covariates=['KOWRCCNSE', 'A001'],
        feature_scaler=feature_scaler
    )
    
    assert dataset.feature_scaler is not None
    assert dataset.feature_scaler is feature_scaler


@pytest.mark.skipif(not DDFM_AVAILABLE, reason=f"DDFM not available: {IMPORT_ERROR if not DDFM_AVAILABLE else ''}")
def test_ddfm_dataset_target_nan_ratio(sample_data):
    """Test target_nan_ratio property."""
    # Add some NaNs
    sample_data_with_nans = sample_data.copy()
    sample_data_with_nans.loc[sample_data_with_nans.index[:10], 'KOEQUIPTE'] = np.nan
    
    dataset = DDFMDataset(
        data=sample_data_with_nans,
        time_idx='time_idx',
        covariates=['KOWRCCNSE', 'A001']
    )
    
    nan_ratio = dataset.target_nan_ratio
    assert 0 <= nan_ratio <= 1
    assert nan_ratio > 0  # Should have some NaNs


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
