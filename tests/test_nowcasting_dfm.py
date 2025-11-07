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
        import subprocess
        import sys
        import os
        
        logger.info("=" * 60)
        logger.info("Running test nowcasting workflow")
        logger.info("=" * 60)
        
        # Step 1: Check if model exists, if not run train_dfm first
        model_file = project_root / "ResDFM.pkl"
        if not model_file.exists():
            logger.info("Model file not found. Running train_dfm.py first...")
            train_script = project_root / "scripts" / "train_dfm.py"
            
            train_result = subprocess.run(
                [
                    sys.executable,
                    str(train_script),
                    "model.config_path=src/spec/001_initial_spec.csv",
                    "data.use_database=true",
                    "data.strict_mode=false",
                ],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes
            )
            
            if train_result.returncode != 0:
                logger.error(f"train_dfm failed with return code {train_result.returncode}")
                logger.error(f"Stdout: {train_result.stdout[-500:]}")
                logger.error(f"Stderr: {train_result.stderr[-500:]}")
                return False
            
            if not model_file.exists():
                logger.error("train_dfm completed but model file still not found")
                return False
            
            logger.info("✓ Model trained successfully")
        
        # Step 2: Run nowcast_dfm.py
        logger.info("Running nowcast_dfm.py...")
        nowcast_script = project_root / "scripts" / "nowcast_dfm.py"
        
        result = subprocess.run(
            [
                sys.executable,
                str(nowcast_script),
                "model.config_path=src/spec/001_initial_spec.csv",
                "data.use_database=true",
                "data.strict_mode=false",
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes
        )
        
        if result.returncode == 0:
            logger.info("=" * 60)
            logger.info("Test nowcasting workflow completed successfully")
            logger.info("=" * 60)
            return True
        else:
            logger.error(f"Script failed with return code {result.returncode}")
            logger.error(f"Stdout: {result.stdout[-500:]}")
            logger.error(f"Stderr: {result.stderr[-500:]}")
            return False
        
    except subprocess.TimeoutExpired:
        logger.error("Test timed out after 10 minutes")
        return False
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


def verify_database_updates():
    """Verify that forecasts were saved to database."""
    try:
        from adapters.adapter_database import _get_db_client
        
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

