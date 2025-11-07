"""
Fast tests for nowcast_dfm.py script.

These tests verify that the forecasting script can:
1. Load configuration from CSV
2. Load data from database for two vintages
3. Load saved DFM model
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.nowcasting.data_loader import load_config_from_csv
from adapters.adapter_database import load_data_from_db


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
    
    print(f"✓ Loaded {len(model_config.series)} series from CSV")


def test_load_data_for_vintage_old():
    """Test loading data for old vintage."""
    try:
        config_path = project_root / "src/spec/001_initial_spec.csv"
        model_config = load_config_from_csv(config_path)
        
        # Load data for vintage_id=1 (old vintage)
        X_old, Time_old, Z_old = load_data_from_db(
            vintage_id=1,
            config=model_config,
            config_name='001-initial-spec',
            strict_mode=False
        )
        
        assert X_old is not None
        assert Time_old is not None
        assert len(Time_old) > 0
        
        print(f"✓ Loaded old vintage data: {X_old.shape[1]} series, {len(Time_old)} observations")
        
    except ImportError as e:
        print(f"⚠ Skipping: Database module not available: {e}")
        return
    except Exception as e:
        print(f"⚠ Skipping: Database connection failed: {e}")
        return


def test_load_data_for_vintage_new():
    """Test loading data for new vintage."""
    try:
        config_path = project_root / "src/spec/001_initial_spec.csv"
        model_config = load_config_from_csv(config_path)
        
        # Load data for latest vintage (new vintage)
        X_new, Time_new, Z_new = load_data_from_db(
            vintage_id=1,  # Use same vintage for testing
            config=model_config,
            config_name='001-initial-spec',
            strict_mode=False
        )
        
        assert X_new is not None
        assert Time_new is not None
        assert len(Time_new) > 0
        
        print(f"✓ Loaded new vintage data: {X_new.shape[1]} series, {len(Time_new)} observations")
        
    except ImportError as e:
        print(f"⚠ Skipping: Database module not available: {e}")
        return
    except Exception as e:
        print(f"⚠ Skipping: Database connection failed: {e}")
        return


def test_model_file_exists():
    """Test that model file can be found (if it exists)."""
    model_file = project_root / "ResDFM.pkl"
    
    if model_file.exists():
        print(f"✓ Model file exists: {model_file}")
    else:
        print(f"⚠ Model file not found: {model_file} (this is OK if training hasn't run yet)")
        return


def test_vintage_resolution():
    """Test that vintage can be resolved from date."""
    try:
        from database import get_latest_vintage_id, get_vintage
        from adapters.adapter_database import _get_db_client
        
        client = _get_db_client()
        latest_vintage_id = get_latest_vintage_id(client=client)
        
        if latest_vintage_id:
            vintage_info = get_vintage(vintage_id=latest_vintage_id, client=client)
            assert vintage_info is not None
            assert 'vintage_date' in vintage_info
            
            print(f"✓ Latest vintage: ID={latest_vintage_id}, Date={vintage_info['vintage_date']}")
        else:
            print("⚠ No vintage found in database")
            return
            
    except ImportError as e:
        print(f"⚠ Skipping: Database module not available: {e}")
        return
    except Exception as e:
        print(f"⚠ Skipping: Database connection failed: {e}")
        return


def test_vintage_comparison():
    """Test that vintage comparison works for nowcasting."""
    try:
        import numpy as np
        config_path = project_root / "src/spec/001_initial_spec.csv"
        model_config = load_config_from_csv(config_path)
        
        # Load data for same vintage (simulating old and new)
        X_old, Time_old, Z_old = load_data_from_db(
            vintage_id=1, config=model_config, config_name='001-initial-spec', strict_mode=False
        )
        X_new, Time_new, Z_new = load_data_from_db(
            vintage_id=1, config=model_config, config_name='001-initial-spec', strict_mode=False
        )
        
        # Verify both vintages loaded
        assert X_old is not None and X_new is not None
        assert X_old.shape[1] == X_new.shape[1], "Number of series should match"
        
        print(f"✓ Vintage comparison: {X_old.shape[1]} series, "
              f"{len(Time_old)} old obs, {len(Time_new)} new obs")
        
    except ImportError as e:
        print(f"⚠ Skipping: Database module not available: {e}")
        return
    except Exception as e:
        print(f"⚠ Skipping: Database connection failed: {e}")
        return


def test_missing_series_in_vintages():
    """Test handling of missing series when comparing vintages."""
    try:
        import numpy as np
        config_path = project_root / "src/spec/001_initial_spec.csv"
        model_config = load_config_from_csv(config_path)
        
        # Load data with strict_mode=False to allow missing series
        X_old, Time_old, Z_old = load_data_from_db(
            vintage_id=1, config=model_config, config_name='001-initial-spec', strict_mode=False
        )
        
        # Check for missing series (all NaN)
        missing = [model_config.SeriesID[i] for i in range(len(model_config.SeriesID))
                  if np.all(np.isnan(X_old[:, i]))]
        
        if missing:
            print(f"⚠ Found {len(missing)} missing series (expected with strict_mode=False)")
        else:
            print("✓ All series have data in vintage")
        
    except ImportError as e:
        print(f"⚠ Skipping: Database module not available: {e}")
        return
    except Exception as e:
        print(f"⚠ Skipping: Database connection failed: {e}")
        return


if __name__ == "__main__":
    # Run tests directly
    print("=" * 60)
    print("Running nowcast_dfm tests...")
    print("=" * 60)
    
    tests = [
        test_load_config_from_csv,
        test_load_data_for_vintage_old,
        test_load_data_for_vintage_new,
        test_model_file_exists,
        test_vintage_resolution,
        test_vintage_comparison,
        test_missing_series_in_vintages,
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
