"""
Test script for nowcasting_dfm.py - Quick test to populate database for dashboard.

This script runs a minimal nowcasting workflow to populate the database
with forecast results for dashboard testing.
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_nowcasting_workflow():
    """Run minimal nowcasting workflow to populate database."""
    try:
        from scripts.nowcast_dfm import main
        import hydra
        from omegaconf import DictConfig
        
        logger.info("=" * 60)
        logger.info("Running test nowcasting workflow")
        logger.info("=" * 60)
        
        # Run with minimal config
        with hydra.initialize(config_path="../configs", version_base=None):
            cfg = hydra.compose(
                config_name="config",
                overrides=[
                    "model.config_path=src/spec/001_initial_spec.csv",
                    "data.use_database=true",
                    "data.strict_mode=false",
                ]
            )
            main(cfg)
        
        logger.info("=" * 60)
        logger.info("Test nowcasting workflow completed")
        logger.info("=" * 60)
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


def verify_database_updates():
    """Verify that forecasts were saved to database."""
    try:
        from adapters.database import _get_db_client
        
        client = _get_db_client()
        
        # Check forecasts
        result = client.table('forecasts').select('*').order('created_at', desc=True).limit(10).execute()
        logger.info(f"✓ Forecasts in database: {len(result.data)}")
        
        if result.data:
            logger.info("Latest forecasts:")
            for f in result.data[:5]:
                logger.info(
                    f"  - {f.get('series_id')}: {f.get('forecast_value')} "
                    f"on {f.get('forecast_date')}"
                )
            return True
        else:
            logger.warning("⚠ No forecasts found in database")
            return False
            
    except Exception as e:
        logger.error(f"Verification failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Test Nowcasting DFM - Populate Database for Dashboard")
    print("=" * 60)
    
    # Run workflow
    success = test_nowcasting_workflow()
    
    if success:
        # Verify updates
        verify_database_updates()
        print("\n✓ Test completed successfully")
        sys.exit(0)
    else:
        print("\n✗ Test failed")
        sys.exit(1)

