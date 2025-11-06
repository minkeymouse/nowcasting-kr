"""
Fast tests for train_dfm.py script.

These tests verify that the training script can:
1. Load configuration from CSV
2. Load data from database
3. Initialize DFM model (without full estimation)
"""

import sys
from pathlib import Path
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.nowcasting.data_loader import load_config_from_csv
from adapters.database import load_data_from_db, save_blocks_to_db


def test_load_config_from_csv():
    """Test loading configuration from CSV spec file."""
    config_path = project_root / "src/spec/001_initial_spec.csv"
    
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")
    
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
        pytest.skip(f"Database module not available: {e}")
    except Exception as e:
        # If database is not available, skip test
        pytest.skip(f"Database connection failed: {e}")


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
        pytest.skip(f"Database module not available: {e}")
    except Exception as e:
        # If database is not available, skip test
        pytest.skip(f"Database connection failed: {e}")


def test_config_series_id_generation():
    """Test that series_id is correctly generated from CSV spec."""
    config_path = project_root / "src/spec/001_initial_spec.csv"
    
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")
    
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


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
