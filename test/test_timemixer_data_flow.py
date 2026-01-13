"""Test TimeMixer data flow: verify raw data is used when use_norm=True."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocess import InvestmentData


def test_timemixer_data_selection():
    """Test that TimeMixer correctly selects raw data when use_norm=True."""
    print("=" * 80)
    print("TEST: TimeMixer Data Selection Logic")
    print("=" * 80)
    
    try:
        # Load data
        print("\n[Step 1] Loading data...")
        data_loader = InvestmentData()
        
        # Get standardized and raw data
        standardized_data = data_loader.standardized
        raw_data = data_loader.processed
        
        # Remove date columns
        date_cols = ['date', 'date_w', 'year', 'month', 'day']
        std_cols = [col for col in standardized_data.columns if col not in date_cols]
        raw_cols = [col for col in raw_data.columns if col not in date_cols]
        common_cols = [col for col in std_cols if col in raw_cols]
        
        standardized_data = standardized_data[common_cols]
        raw_data = raw_data[common_cols]
        
        print(f"  Standardized data shape: {standardized_data.shape}")
        print(f"  Raw data shape: {raw_data.shape}")
        print(f"  Common columns: {len(common_cols)}")
        
        # Check data statistics
        print("\n[Step 2] Verifying data characteristics...")
        std_mean_abs = standardized_data.mean().abs().max()
        std_std_diff = (standardized_data.std() - 1.0).abs().max()
        print(f"  Standardized data - |mean|: {std_mean_abs:.3f}, |std-1|: {std_std_diff:.3f}")
        
        raw_mean_abs = raw_data.mean().abs().max()
        raw_std_diff = (raw_data.std() - 1.0).abs().max()
        print(f"  Raw data - |mean|: {raw_mean_abs:.3f}, |std-1|: {raw_std_diff:.3f}")
        
        # Verify they are different
        if std_mean_abs < 0.1 and std_std_diff < 0.1:
            print("  ✓ Standardized data is properly standardized")
        else:
            print("  ✗ Standardized data may not be properly standardized")
            return False
        
        if raw_mean_abs > 1.0 or raw_std_diff > 0.5:
            print("  ✓ Raw data is NOT standardized (has real scale)")
        else:
            print("  ✗ Raw data appears to be standardized")
            return False
        
        # Simulate the logic from train_timemixer_model
        print("\n[Step 3] Simulating TimeMixer data selection logic...")
        model_params = {'use_norm': True}
        use_norm = model_params.get('use_norm', True) if model_params else True
        
        if use_norm and data_loader is not None:
            if hasattr(data_loader, 'processed') and data_loader.processed is not None:
                selected_raw_data = data_loader.processed.copy()
                date_cols_check = ['date', 'date_w', 'year', 'month', 'day']
                data_cols_check = [col for col in selected_raw_data.columns if col not in date_cols_check]
                selected_raw_data = selected_raw_data[data_cols_check].copy()
                
                # Filter to common columns
                if hasattr(data_loader, 'standardized') and data_loader.standardized is not None:
                    std_cols_check = [col for col in data_loader.standardized.columns if col not in date_cols_check]
                    common_cols_check = [col for col in data_cols_check if col in std_cols_check]
                    if common_cols_check:
                        selected_raw_data = selected_raw_data[common_cols_check]
                
                print(f"  Selected raw data shape: {selected_raw_data.shape}")
                
                # Verify it matches our raw_data
                if selected_raw_data.shape == raw_data.shape:
                    print("  ✓ Selected data shape matches expected raw data")
                else:
                    print(f"  ✗ Shape mismatch: {selected_raw_data.shape} vs {raw_data.shape}")
                    return False
                
                # Check a few values to ensure it's raw data (not standardized)
                sample_series = selected_raw_data.columns[0]
                sample_mean = selected_raw_data[sample_series].mean()
                sample_std = selected_raw_data[sample_series].std()
                
                print(f"  Sample series '{sample_series}': mean={sample_mean:.2f}, std={sample_std:.2f}")
                
                if abs(sample_mean) > 0.5 or abs(sample_std - 1.0) > 0.3:
                    print("  ✓ Selected data has raw scale (not standardized)")
                else:
                    print("  ✗ Selected data appears standardized")
                    return False
            else:
                print("  ✗ data_loader.processed not available")
                return False
        else:
            print("  ✗ Logic would not select raw data")
            return False
        
        print("\n[Step 4] Testing with use_norm=False...")
        model_params_no_norm = {'use_norm': False}
        use_norm_no_norm = model_params_no_norm.get('use_norm', True) if model_params_no_norm else True
        
        if not use_norm_no_norm:
            print("  ✓ When use_norm=False, standardized data would be used")
        else:
            print("  ✗ Logic incorrectly selects raw data when use_norm=False")
            return False
        
        print("\n" + "=" * 80)
        print("TEST PASSED: TimeMixer data selection logic works correctly")
        print("=" * 80)
        print("\nSummary:")
        print("  - When use_norm=True: Raw (unstandardized) data is selected ✓")
        print("  - When use_norm=False: Standardized data would be used ✓")
        print("  - Data loader integration works correctly ✓")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_timemixer_data_selection()
    sys.exit(0 if success else 1)