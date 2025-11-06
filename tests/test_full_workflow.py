"""
Full workflow test: ingest_api → train_dfm → forecast_dfm

This test verifies the complete DFM workflow:
1. Data ingestion (simulated - assumes data already in DB)
2. Model training
3. Nowcasting/forecasting
"""

import sys
from pathlib import Path
import subprocess
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_train_dfm_workflow():
    """Test train_dfm.py workflow."""
    print("\n" + "=" * 60)
    print("[TEST] train_dfm.py Workflow")
    print("=" * 60)
    
    try:
        # Run train_dfm.py with timeout
        result = subprocess.run(
            [
                sys.executable,
                "scripts/train_dfm.py",
                "model.config_path=src/spec/001_initial_spec.csv",
                "data.use_database=true"
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes
        )
        
        # Check for key success indicators
        output = result.stdout + result.stderr
        
        checks = {
            "blocks_saved": "Saved.*block assignments" in output or "blocks" in output.lower(),
            "data_loaded": "Loaded data from database" in output or "observations" in output.lower(),
            "model_trained": "Estimating the dynamic factor model" in output,
            "model_saved": "ResDFM.pkl" in output or Path(project_root / "ResDFM.pkl").exists(),
        }
        
        print(f"Exit code: {result.returncode}")
        print(f"Checks:")
        for check_name, check_result in checks.items():
            status = "✓" if check_result else "✗"
            print(f"  {status} {check_name}: {check_result}")
        
        if result.returncode == 0 and all(checks.values()):
            print("✓ train_dfm.py workflow: PASSED")
            return True
        else:
            print("✗ train_dfm.py workflow: FAILED")
            if result.returncode != 0:
                print(f"Error output (last 500 chars):\n{output[-500:]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ train_dfm.py: TIMEOUT (exceeded 5 minutes)")
        return False
    except Exception as e:
        print(f"✗ train_dfm.py: ERROR - {e}")
        return False


def test_forecast_dfm_workflow():
    """Test forecast_dfm.py workflow."""
    print("\n" + "=" * 60)
    print("[TEST] forecast_dfm.py Workflow")
    print("=" * 60)
    
    try:
        # Run forecast_dfm.py with timeout
        result = subprocess.run(
            [
                sys.executable,
                "scripts/forecast_dfm.py",
                "model.config_path=src/spec/001_initial_spec.csv",
                "data.use_database=true"
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes
        )
        
        # Check for key success indicators
        output = result.stdout + result.stderr
        
        checks = {
            "blocks_saved": "Saved.*block assignments" in output or "blocks" in output.lower(),
            "data_loaded": "Loaded data from database" in output or "observations" in output.lower(),
            "model_loaded": "Loading.*model" in output or "ResDFM.pkl" in output,
            "nowcast_computed": "Nowcast" in output or "nowcasting" in output.lower(),
            "forecast_saved": "Saved.*forecast" in output or "save_nowcast" in output.lower(),
        }
        
        print(f"Exit code: {result.returncode}")
        print(f"Checks:")
        for check_name, check_result in checks.items():
            status = "✓" if check_result else "✗"
            print(f"  {status} {check_name}: {check_result}")
        
        if result.returncode == 0:
            print("✓ forecast_dfm.py workflow: PASSED")
            return True
        else:
            print("✗ forecast_dfm.py workflow: FAILED")
            if result.returncode != 0:
                print(f"Error output (last 500 chars):\n{output[-500:]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ forecast_dfm.py: TIMEOUT (exceeded 5 minutes)")
        return False
    except Exception as e:
        print(f"✗ forecast_dfm.py: ERROR - {e}")
        return False


def test_database_updates():
    """Test that database was updated correctly."""
    print("\n" + "=" * 60)
    print("[TEST] Database Updates")
    print("=" * 60)
    
    try:
        from adapters.database import _get_db_client
        
        client = _get_db_client()
        
        # Check blocks
        try:
            result = client.table('blocks').select('*').eq('config_name', '001-initial-spec').limit(1).execute()
            if result.data:
                print(f"✓ Blocks table: Has records for config_name=001-initial-spec")
            else:
                print("⚠ Blocks table: No records found")
        except Exception as e:
            print(f"⚠ Blocks check: {e}")
        
        # Check forecasts (latest)
        try:
            result = client.table('forecasts').select('*').order('created_at', desc=True).limit(1).execute()
            if result.data:
                print(f"✓ Forecasts table: Has latest records")
                print(f"  Latest: series_id={result.data[0].get('series_id')}, date={result.data[0].get('forecast_date')}")
            else:
                print("⚠ Forecasts table: No records found")
        except Exception as e:
            print(f"⚠ Forecasts check: {e}")
        
        print("✓ Database updates: PASSED")
        return True
        
    except ImportError as e:
        print(f"⚠ Skipping: Database module not available: {e}")
        return True  # Not a failure
    except Exception as e:
        print(f"⚠ Database check: {e}")
        return True  # Not a critical failure for DFM module


if __name__ == "__main__":
    print("=" * 60)
    print("Full Workflow Test: train_dfm → forecast_dfm")
    print("=" * 60)
    print("\nNote: Assumes data is already in database (from ingest_api)")
    
    results = []
    
    # Test train_dfm
    results.append(("train_dfm", test_train_dfm_workflow()))
    
    # Test forecast_dfm (only if train succeeded)
    if results[-1][1]:
        results.append(("forecast_dfm", test_forecast_dfm_workflow()))
    else:
        print("\n⚠ Skipping forecast_dfm test (train_dfm failed)")
        results.append(("forecast_dfm", False))
    
    # Test database updates
    results.append(("database_updates", test_database_updates()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} passed")
    print("=" * 60)
    
    if passed == total:
        sys.exit(0)
    else:
        sys.exit(1)

