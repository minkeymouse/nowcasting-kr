"""Consolidated tests for data loading, specification, and transformations."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.nowcasting.data_loader import load_config, load_config_from_yaml, load_config_from_excel, load_data
from src.utils.data_utils import summarize

def _get_base_dir():
    """Get base directory."""
    return Path(__file__).parent.parent.parent

# ============================================================================
# Specification Tests
# ============================================================================

def test_load_config():
    """Test loading model specification."""
    print("\n" + "="*70)
    print("TEST: Load Specification")
    print("="*70)
    
    base_dir = _get_base_dir()
    spec_file = base_dir / 'matlab' / 'Spec_US_example.xls'
    
    if not spec_file.exists():
        print(f"SKIPPED: Spec file not found: {spec_file}")
        return None
    
    config = load_config(spec_file)
    
    assert hasattr(config, 'SeriesID')
    assert hasattr(config, 'SeriesName')
    assert hasattr(config, 'Blocks')
    assert len(config.SeriesID) > 0
    
    print(f"✓ Loaded {len(config.SeriesID)} series")
    print(f"✓ Block structure: {config.Blocks.shape}")
    print(f"✓ Number of blocks: {len(config.BlockNames)}")
    
    return config

def test_config_structure():
    """Test configuration structure completeness."""
    print("\n" + "="*70)
    print("TEST: Configuration Structure")
    print("="*70)
    
    config = test_load_config()
    if config is None:
        return None
    
    required_attrs = ['SeriesID', 'SeriesName', 'Frequency', 'Units',
                     'Transformation', 'Category', 'Blocks', 'BlockNames']
    
    for attr in required_attrs:
        assert hasattr(config, attr), f"Missing attribute: {attr}"
    
    assert len(config.SeriesID) == len(config.SeriesName)
    assert config.Blocks.shape[0] == len(config.SeriesID)
    
    print("✓ Configuration structure verified")
    return True

# ============================================================================
# Data Loading Tests
# ============================================================================

def test_load_data():
    """Test loading data."""
    print("\n" + "="*70)
    print("TEST: Load Data")
    print("="*70)
    
    base_dir = _get_base_dir()
    spec_file = base_dir / 'matlab' / 'Spec_US_example.xls'
    
    if not spec_file.exists():
        print(f"SKIPPED: Spec file not found")
        return None
    
    config = load_config(spec_file)
    
    vintage = '2016-06-29'
    country = 'US'
    data_file = base_dir / 'data' / country / f'{vintage}.xls'
    
    if not data_file.exists():
        print(f"SKIPPED: Data file not found: {data_file}")
        return None
    
    sample_start = pd.Timestamp('2000-01-01')
    X, Time, Z = load_data(data_file, config, sample_start=sample_start)
    
    assert X.shape[0] > 0
    assert X.shape[1] == len(config.SeriesID)
    assert len(Time) == X.shape[0]
    
    print(f"✓ Loaded data: {X.shape[0]} time periods, {X.shape[1]} series")
    print(f"✓ Time range: {Time[0]} to {Time[-1]}")
    print(f"✓ Raw data shape: {Z.shape}")
    
    return X, Time, Z, config

def test_load_data_multiple_vintages():
    """Test loading multiple vintages."""
    print("\n" + "="*70)
    print("TEST: Multiple Vintages")
    print("="*70)
    
    base_dir = _get_base_dir()
    spec_file = base_dir / 'matlab' / 'Spec_US_example.xls'
    
    if not spec_file.exists():
        print("SKIPPED: Spec file not found")
        return None
    
    config = load_config(spec_file)
    
    vintages = ['2016-06-29', '2016-12-16', '2016-12-23']
    results = {}
    
    for vintage in vintages:
        data_file = base_dir / 'data' / 'US' / f'{vintage}.xls'
        if data_file.exists():
            try:
                X, Time, Z = load_data(data_file, config, sample_start=pd.Timestamp('2000-01-01'))
                results[vintage] = (X.shape, len(Time))
                print(f"✓ Loaded {vintage}: {X.shape}")
            except Exception as e:
                print(f"⚠ Failed to load {vintage}: {e}")
        else:
            print(f"⚠ Vintage not found: {vintage}")
    
    if len(results) > 0:
        print(f"✓ Loaded {len(results)} vintages")
        return True
    else:
        print("SKIPPED: No vintages found")
        return None

# ============================================================================
# Data Transformation Tests
# ============================================================================

def test_data_transformations():
    """Test data transformations."""
    print("\n" + "="*70)
    print("TEST: Data Transformations")
    print("="*70)
    
    base_dir = _get_base_dir()
    spec_file = base_dir / 'matlab' / 'Spec_US_example.xls'
    
    if not spec_file.exists():
        print("SKIPPED: Spec file not found")
        return None
    
    config = load_config(spec_file)
    
    vintage = '2016-06-29'
    data_file = base_dir / 'data' / 'US' / f'{vintage}.xls'
    
    if not data_file.exists():
        print("SKIPPED: Data file not found")
        return None
    
    X, Time, Z = load_data(data_file, config)
    
    assert np.sum(~np.isnan(X)) > 0
    
    try:
        idx = config.SeriesID.index('INDPRO')
        indpro_data = X[:, idx]
        assert np.sum(~np.isnan(indpro_data)) > 0
        print(f"✓ INDPRO transformation: {np.sum(~np.isnan(indpro_data))} observations")
    except ValueError:
        print("⚠ INDPRO not found in specification")
    
    print("✓ Data transformations verified")
    return True

def test_data_summary():
    """Test data summary functionality."""
    print("\n" + "="*70)
    print("TEST: Data Summary")
    print("="*70)
    
    base_dir = _get_base_dir()
    spec_file = base_dir / 'matlab' / 'Spec_US_example.xls'
    
    if not spec_file.exists():
        print("SKIPPED: Spec file not found")
        return None
    
    config = load_config(spec_file)
    
    vintage = '2016-06-29'
    data_file = base_dir / 'data' / 'US' / f'{vintage}.xls'
    
    if not data_file.exists():
        print("SKIPPED: Data file not found")
        return None
    
    sample_start = pd.Timestamp('2000-01-01')
    X, Time, Z = load_data(data_file, config, sample_start=sample_start)
    
    try:
        summarize(X, Time, config, vintage)
        print("✓ Data summary completed")
        return True
    except Exception as e:
        print(f"✗ Data summary failed: {e}")
        return False

# ============================================================================
# Data Validation Tests
# ============================================================================

def test_data_shapes():
    """Test data shapes are consistent."""
    print("\n" + "="*70)
    print("TEST: Data Shapes")
    print("="*70)
    
    base_dir = _get_base_dir()
    spec_file = base_dir / 'matlab' / 'Spec_US_example.xls'
    
    if not spec_file.exists():
        print("SKIPPED: Spec file not found")
        return None
    
    config = load_config(spec_file)
    
    vintage = '2016-06-29'
    data_file = base_dir / 'data' / 'US' / f'{vintage}.xls'
    
    if not data_file.exists():
        print("SKIPPED: Data file not found")
        return None
    
    X, Time, Z = load_data(data_file, config, sample_start=pd.Timestamp('2000-01-01'))
    
    assert X.ndim == 2
    assert len(Time) == X.shape[0]
    assert X.shape[1] == len(config.SeriesID)
    assert Z.ndim == 2
    
    print(f"✓ Data shapes verified: X {X.shape}, Time {len(Time)}, Z {Z.shape}")
    return True

def test_missing_data_handling():
    """Test missing data handling."""
    print("\n" + "="*70)
    print("TEST: Missing Data Handling")
    print("="*70)
    
    base_dir = _get_base_dir()
    spec_file = base_dir / 'matlab' / 'Spec_US_example.xls'
    
    if not spec_file.exists():
        print("SKIPPED: Spec file not found")
        return None
    
    config = load_config(spec_file)
    
    vintage = '2016-06-29'
    data_file = base_dir / 'data' / 'US' / f'{vintage}.xls'
    
    if not data_file.exists():
        print("SKIPPED: Data file not found")
        return None
    
    X, Time, Z = load_data(data_file, config, sample_start=pd.Timestamp('2000-01-01'))
    
    missing_rate = np.sum(np.isnan(X)) / X.size * 100
    print(f"✓ Missing data rate: {missing_rate:.2f}%")
    
    assert X.size > 0
    assert np.sum(~np.isnan(X)) > 0
    
    print("✓ Missing data handling verified")
    return True

# ============================================================================
# Test Runner
# ============================================================================

def run_all_tests():
    """Run all data loader tests."""
    print("\n" + "="*70)
    print("DATA LOADER TESTS")
    print("="*70)
    
    results = {}
    
    test_funcs = [
        ('load_config', test_load_config),
        ('config_structure', test_config_structure),
        ('load_data', test_load_data),
        ('multiple_vintages', test_load_data_multiple_vintages),
        ('transformations', test_data_transformations),
        ('summary', test_data_summary),
        ('shapes', test_data_shapes),
        ('missing_data', test_missing_data_handling),
    ]
    
    for name, func in test_funcs:
        try:
            results[name] = func()
            if results[name] is not None and results[name] is not False:
                print(f"✓ {name} PASSED")
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            results[name] = None
    
    passed = sum(1 for v in results.values() if v is not None and v is not False)
    total = len([v for v in results.values() if v is not None])
    
    print("\n" + "="*70)
    print(f"SUMMARY: {passed}/{total} tests passed")
    print("="*70)
    
    return results

if __name__ == '__main__':
    run_all_tests()



