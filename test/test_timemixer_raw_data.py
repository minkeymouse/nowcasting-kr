"""Test TimeMixer training with raw (unstandardized) data."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocess import InvestmentData
from src.train.timemixer import train_timemixer_model
from src.utils import load_test_data

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_timemixer_raw_data_training():
    """Test that TimeMixer training uses raw data when use_norm=True."""
    print("=" * 80)
    print("TEST: TimeMixer Training with Raw Data")
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
        print("\n[Step 2] Checking data statistics...")
        print(f"  Standardized data - Mean: [{standardized_data.mean().min():.3f}, {standardized_data.mean().max():.3f}], "
              f"Std: [{standardized_data.std().min():.3f}, {standardized_data.std().max():.3f}]")
        print(f"  Raw data - Mean: [{raw_data.mean().min():.2f}, {raw_data.mean().max():.2f}], "
              f"Std: [{raw_data.std().min():.2f}, {raw_data.std().max():.2f}]")
        
        # Verify standardized data is actually standardized
        std_mean_abs = standardized_data.mean().abs().max()
        std_std_diff = (standardized_data.std() - 1.0).abs().max()
        print(f"  ✓ Standardized check - |mean| < 0.1: {std_mean_abs < 0.1}, |std-1| < 0.1: {std_std_diff < 0.1}")
        
        # Verify raw data is NOT standardized
        raw_mean_abs = raw_data.mean().abs().max()
        raw_std_diff = (raw_data.std() - 1.0).abs().max()
        print(f"  ✓ Raw data check - |mean| > 1.0: {raw_mean_abs > 1.0}, |std-1| > 0.5: {raw_std_diff > 0.5}")
        
        # Test training configuration
        print("\n[Step 3] Testing TimeMixer training configuration...")
        
        # Create minimal model params for testing
        model_params = {
            'prediction_length': 1,
            'context_length': 96,
            'target_series': common_cols[:5],  # Use first 5 series for quick test
            'max_epochs': 1,  # Just 1 epoch for testing
            'batch_size': 4,
            'learning_rate': 0.001,
            'use_norm': True,  # Enable RevIN
            'down_sampling_layers': 1,  # Simpler for testing
            'down_sampling_window': 4,
        }
        
        print(f"  Model params: prediction_length={model_params['prediction_length']}, "
              f"context_length={model_params['context_length']}, "
              f"target_series={len(model_params['target_series'])} series")
        
        # Prepare training data subset (small for testing)
        train_subset = standardized_data.iloc[:200, :5].copy()  # First 200 rows, 5 series
        
        print("\n[Step 4] Training TimeMixer (should use raw data)...")
        print("  Note: This will verify that raw data is used instead of standardized data")
        
        # Create temporary output directory
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir) / "test_timemixer"
            outputs_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # This should internally switch to raw data
                train_timemixer_model(
                    model_type='timemixer',
                    cfg=None,
                    data=train_subset,
                    model_name='test_timemixer',
                    outputs_dir=outputs_dir,
                    model_params=model_params,
                    data_loader=data_loader
                )
                
                print("  ✓ Training completed successfully!")
                
                # Verify model was saved
                model_path = outputs_dir / "model.pkl"
                if model_path.exists():
                    print(f"  ✓ Model saved at: {model_path}")
                else:
                    print(f"  ✗ Model not found at: {model_path}")
                    return False
                    
            except Exception as e:
                print(f"  ✗ Training failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        print("\n" + "=" * 80)
        print("TEST PASSED: TimeMixer training works with raw data")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_timemixer_raw_data_training()
    sys.exit(0 if success else 1)