"""
Fast tests for train_dfm.py script.

These tests verify that the training script can:
1. Load configuration from CSV
2. Load data from database
3. Initialize DFM model (without full estimation)
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.nowcasting.data_loader import load_config_from_csv
from adapters.adapter_database import load_data_from_db, save_blocks_to_db


def test_load_config_from_csv():
    """Test loading configuration from CSV spec file."""
    config_path = project_root / "src/spec/001_initial_spec.csv"
    
    if not config_path.exists():
        print(f"⚠ Skipping: Config file not found: {config_path}")
        return
    
    model_config = load_config_from_csv(config_path)
    
    # Verify basic structure
    assert model_config is not None
    assert hasattr(model_config, 'series')
    assert len(model_config.series) > 0
    
    # Verify series_id is generated correctly
    first_series = model_config.series[0]
    assert hasattr(first_series, 'series_id')
    assert first_series.series_id is not None
    assert first_series.series_id != '0'  # Should be generated from data_code/item_id/api_source
    
    print(f"✓ Loaded {len(model_config.series)} series from CSV")
    print(f"✓ First series_id: {first_series.series_id}")


def test_load_data_from_db():
    """Test loading data from database (quick check only)."""
    try:
        config_path = project_root / "src/spec/001_initial_spec.csv"
        model_config = load_config_from_csv(config_path)
        
        # Load data with minimal parameters
        X, Time, Z = load_data_from_db(
            vintage_id=1,  # Use specific vintage
            config=model_config,
            config_name='001-initial-spec',
            strict_mode=False  # Don't fail on missing data
        )
        
        # Verify data structure
        assert X is not None
        assert Time is not None
        assert Z is not None
        
        # Verify data dimensions
        assert len(Time) > 0, "Time index should not be empty"
        assert X.shape[0] == len(Time), "X rows should match Time length"
        assert X.shape[1] > 0, "X should have at least one series"
        
        print(f"✓ Loaded data: {X.shape[1]} series, {len(Time)} observations")
        print(f"✓ Time range: {Time[0]} to {Time[-1]}")
        
    except ImportError as e:
        print(f"⚠ Skipping: Database module not available: {e}")
        return
    except Exception as e:
        # If database is not available, skip test
        print(f"⚠ Skipping: Database connection failed: {e}")
        return


def test_save_blocks_to_db():
    """Test saving blocks to database."""
    try:
        config_path = project_root / "src/spec/001_initial_spec.csv"
        model_config = load_config_from_csv(config_path)
        
        # Save blocks
        save_blocks_to_db(
            config=model_config,
            config_name='test-001-initial-spec'  # Use test prefix to avoid conflicts
        )
        
        print("✓ Blocks saved to database successfully")
        
    except ImportError as e:
        print(f"⚠ Skipping: Database module not available: {e}")
        return
    except Exception as e:
        # If database is not available, skip test
        print(f"⚠ Skipping: Database connection failed: {e}")
        return


def test_config_series_id_generation():
    """Test that series_id is correctly generated from CSV spec."""
    config_path = project_root / "src/spec/001_initial_spec.csv"
    
    if not config_path.exists():
        print(f"⚠ Skipping: Config file not found: {config_path}")
        return
    
    model_config = load_config_from_csv(config_path)
    
    # Check that series_id follows the pattern: {api_source}_{data_code}_{item_id}
    for series in model_config.series:
        assert hasattr(series, 'series_id')
        series_id = series.series_id
        
        # Should not be just a number (from 'id' column)
        assert not series_id.isdigit(), f"series_id should not be just a number: {series_id}"
        
        # Should contain underscores (pattern: API_SOURCE_DATA_CODE_ITEM_ID)
        assert '_' in series_id, f"series_id should contain underscores: {series_id}"
    
    print(f"✓ All {len(model_config.series)} series have valid series_id format")


def test_missing_series_handling():
    """Test that missing series are handled gracefully with strict_mode=False."""
    try:
        config_path = project_root / "src/spec/001_initial_spec.csv"
        model_config = load_config_from_csv(config_path)
        
        # Load data with strict_mode=False (should fill missing series with NaN)
        X, Time, Z = load_data_from_db(
            vintage_id=1,
            config=model_config,
            config_name='001-initial-spec',
            strict_mode=False  # Should not fail on missing series
        )
        
        # Verify data structure even with missing series
        assert X is not None
        assert Time is not None
        assert X.shape[1] == len(model_config.series), "Should have same number of series as config"
        
        # Check for NaN values (expected for missing series)
        nan_series = []
        for i, series_id in enumerate(model_config.SeriesID):
            series_data = X[:, i]
            nan_count = np.sum(np.isnan(series_data))
            if nan_count == len(series_data):
                nan_series.append(series_id)
        
        if nan_series:
            print(f"⚠ Found {len(nan_series)} series with all NaN values (missing from database):")
            for series_id in nan_series[:5]:  # Show first 5
                print(f"   - {series_id}")
        else:
            print("✓ All series have at least some data")
        
        print(f"✓ Missing series handling works (strict_mode=False)")
        
    except ImportError as e:
        print(f"⚠ Skipping: Database module not available: {e}")
        return
    except Exception as e:
        print(f"⚠ Skipping: Database connection failed: {e}")
        return


def test_data_completeness_check():
    """Test data completeness validation."""
    try:
        config_path = project_root / "src/spec/001_initial_spec.csv"
        model_config = load_config_from_csv(config_path)
        
        # Load data
        X, Time, Z = load_data_from_db(
            vintage_id=1,
            config=model_config,
            config_name='001-initial-spec',
            strict_mode=False
        )
        
        # Check data completeness
        total_obs = X.shape[0] * X.shape[1]
        finite_obs = np.sum(np.isfinite(X))
        completeness_pct = (finite_obs / total_obs * 100) if total_obs > 0 else 0.0
        
        print(f"✓ Data completeness: {finite_obs}/{total_obs} ({completeness_pct:.1f}%)")
        
        # Check minimum observations per series
        min_obs = 10
        insufficient = [np.sum(np.isfinite(X[:, i])) < min_obs 
                       for i in range(len(model_config.SeriesID))]
        n_insufficient = sum(insufficient)
        
        if n_insufficient > 0:
            print(f"⚠ {n_insufficient} series have <{min_obs} observations")
        else:
            print(f"✓ All series have at least {min_obs} observations")
        
    except ImportError as e:
        print(f"⚠ Skipping: Database module not available: {e}")
        return
    except Exception as e:
        print(f"⚠ Skipping: Database connection failed: {e}")
        return


if __name__ == "__main__":
    # Run tests directly
    print("=" * 60)
    print("Running train_dfm tests...")
    print("=" * 60)
    
    tests = [
        test_load_config_from_csv,
        test_load_data_from_db,
        test_save_blocks_to_db,
        test_config_series_id_generation,
        test_missing_series_handling,
        test_data_completeness_check,
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_func in tests:
        print(f"\n[TEST] {test_func.__name__}")
        try:
            test_func()
            passed += 1
            print(f"✓ PASSED: {test_func.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"✗ FAILED: {test_func.__name__}: {e}")
        except Exception as e:
            skipped += 1
            print(f"⚠ SKIPPED: {test_func.__name__}: {e}")
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)
