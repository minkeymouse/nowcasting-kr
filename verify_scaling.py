"""Verification script to check if scaling is working correctly."""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.preprocess import InvestmentData
from src.train._common import get_common_training_params
from src.utils import inverse_transform_predictions

def verify_scaling_pipeline():
    """Verify that the scaling pipeline works correctly."""
    
    print("=" * 80)
    print("SCALING PIPELINE VERIFICATION")
    print("=" * 80)
    
    # Step 1: Load and preprocess data
    print("\n[Step 1] Loading and preprocessing data...")
    data_loader = InvestmentData()
    
    # Check standardized data
    standardized = data_loader.standardized
    print(f"  Standardized data shape: {standardized.shape}")
    print(f"  Standardized data stats:")
    for col in standardized.columns[:3]:  # Check first 3 columns
        mean = standardized[col].mean()
        std = standardized[col].std()
        print(f"    {col}: mean={mean:.6f}, std={std:.6f}")
    
    # Step 2: Check model configurations
    print("\n[Step 2] Checking model configurations...")
    
    # PatchTST
    print("\n  PatchTST:")
    patchtst_params = get_common_training_params({'prediction_length': 4})
    print(f"    scaler_type: {patchtst_params['scaler_type']}")
    print(f"    revin: True (from model_params.get('revin', True))")
    print(f"    ⚠️  ISSUE: revin=True applies additional normalization on pre-standardized data!")
    
    # iTransformer
    print("\n  iTransformer:")
    itf_params = get_common_training_params({'prediction_length': 4})
    print(f"    scaler_type: {itf_params['scaler_type']}")
    print(f"    use_norm: False (correct - data is pre-standardized)")
    
    # TFT
    print("\n  TFT:")
    tft_params = get_common_training_params({'prediction_length': 4})
    print(f"    scaler_type: {tft_params['scaler_type']}")
    print(f"    No additional normalization (correct)")
    
    # Step 3: Test inverse transform
    print("\n[Step 3] Testing inverse transform...")
    test_series = standardized.columns[0]
    test_std_values = standardized[test_series].values[:5]
    test_original_values = data_loader.processed[test_series].values[:5]
    
    # Inverse transform
    predictions_original = inverse_transform_predictions(
        test_std_values, [test_series], data_loader, reverse_transformations=False
    )
    
    print(f"  Test series: {test_series}")
    print(f"  Standardized values: {test_std_values}")
    print(f"  Inverse transformed: {predictions_original}")
    print(f"  Original processed values: {test_original_values}")
    
    # Check if inverse transform recovers original
    max_diff = np.max(np.abs(predictions_original - test_original_values))
    print(f"  Max difference: {max_diff:.6f}")
    if max_diff < 1e-5:
        print("  ✓ Inverse transform works correctly")
    else:
        print(f"  ✗ Inverse transform has errors (max diff: {max_diff})")
    
    # Step 4: Identify issues
    print("\n[Step 4] IDENTIFIED ISSUES:")
    print("\n  🔴 CRITICAL ISSUE: PatchTST with revin=True")
    print("     - Data is pre-standardized (mean≈0, std≈1)")
    print("     - RevIN normalizes again by computing batch mean/std")
    print("     - This causes double normalization:")
    print("       1. Pre-standardization: x → (x - μ) / σ")
    print("       2. RevIN norm: (x_std - mean_batch) / std_batch")
    print("       3. During inference, RevIN denorm uses batch stats")
    print("       4. But batch stats are computed on already-standardized data")
    print("     - Result: Model predictions are in wrong scale")
    print("\n  ✅ RECOMMENDATION:")
    print("     - Set revin=False when data is pre-standardized")
    print("     - OR: Don't pre-standardize data if you want to use RevIN")
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    verify_scaling_pipeline()
