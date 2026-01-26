"""Test to reproduce and diagnose DFM scaling issue.

This test uses small synthetic data to quickly reproduce the scaling issue
that causes DFM metrics to be orders of magnitude off.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
import tempfile
import joblib
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Check model availability
try:
    from dfm_python import DFM, DFMDataset
    from dfm_python.config import DFMConfig
    DFM_AVAILABLE = True
except ImportError as e:
    DFM_AVAILABLE = False
    IMPORT_ERROR = str(e)

from src.forecast.dfm import run_recursive_forecast


@pytest.fixture
def small_training_data():
    """Create small synthetic training data (3 series, 50 timesteps)."""
    dates = pd.date_range('2020-01-01', periods=50, freq='W')
    # Create data in transformed space (chg/logdiff) - NOT standardized
    # This simulates data_loader.processed which is transformed but not standardized
    np.random.seed(42)
    data = pd.DataFrame(
        np.random.randn(50, 3) * 0.1,  # Small values typical of differenced data
        columns=['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
        index=dates
    )
    return data


@pytest.fixture
def small_test_data():
    """Create small synthetic test data (3 series, 10 timesteps)."""
    dates = pd.date_range('2023-01-01', periods=10, freq='W')
    np.random.seed(123)
    data = pd.DataFrame(
        np.random.randn(10, 3) * 0.1,  # Same scale as training
        columns=['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
        index=dates
    )
    return data


@pytest.fixture
def trained_dfm_model_minimal(small_training_data, tmp_path):
    """Create and train a minimal DFM model (max_iter=1 for speed)."""
    model_params = {
        'covariates': ['KOWRCCNSE', 'A001'],  # Use covariates (new approach)
        'ar_lag': 2,
        'ar_order': 2,
        'threshold': 1e-3,
        'max_iter': 1,  # Fast training
        'blocks': {
            'Block_Global': {
                'num_factors': 1,
                'series': ['KOEQUIPTE', 'KOWRCCNSE', 'A001']
            }
        },
        'frequency': {
            'w': ['KOEQUIPTE', 'KOWRCCNSE', 'A001']
        },
        'clock': 'w',
        'mixed_freq': False,
        'scaler': 'standard'
    }
    
    config = DFMConfig.from_dict(model_params)
    dataset = DFMDataset(config=config, data=small_training_data, covariates=['KOWRCCNSE', 'A001'])
    
    # Train model
    model = DFM(config)
    X = dataset.get_processed_data()
    model.fit(X=X, dataset=dataset)
    
    # Save model and dataset
    checkpoint_path = tmp_path / "model.pkl"
    dataset_path = tmp_path / "dataset.pkl"
    model.save(checkpoint_path)
    joblib.dump(dataset, dataset_path)
    
    return checkpoint_path, dataset_path, model, dataset


@pytest.mark.skipif(not DFM_AVAILABLE, reason=f"DFM not available: {IMPORT_ERROR if not DFM_AVAILABLE else ''}")
def test_dfm_column_order_consistency(trained_dfm_model_minimal, small_test_data):
    """Test that column order is consistent between training and update data."""
    checkpoint_path, dataset_path, model, dataset = trained_dfm_model_minimal
    
    # Get training column order
    training_columns = None
    if hasattr(dataset, '_processed_columns'):
        training_columns = list(dataset._processed_columns)
    elif hasattr(model, 'scaler') and model.scaler is not None:
        if hasattr(model.scaler, 'feature_names_in_'):
            training_columns = list(model.scaler.feature_names_in_)
    
    # Get update data columns (simulating forecast pipeline)
    # All series should be included (covariates are used for factor extraction but all series are in data)
    update_columns = list(small_test_data.select_dtypes(include=[np.number]).columns)
    
    # Check column order match (all series should be in training data)
    assert training_columns is not None, "Training columns should be available"
    assert update_columns == training_columns, \
        f"Column order mismatch! Training: {training_columns}, Update: {update_columns}"
    
    # Verify that target_series is computed correctly from covariates
    if hasattr(dataset, 'covariates') and dataset.covariates:
        expected_targets = [c for c in training_columns if c not in dataset.covariates]
        assert dataset.target_series == expected_targets, \
            f"target_series should be computed from covariates. Expected: {expected_targets}, Got: {dataset.target_series}"


@pytest.mark.skipif(not DFM_AVAILABLE, reason=f"DFM not available: {IMPORT_ERROR if not DFM_AVAILABLE else ''}")
def test_dfm_scale_consistency(trained_dfm_model_minimal, small_training_data, small_test_data):
    """Test that update data scale matches training data scale."""
    checkpoint_path, dataset_path, model, dataset = trained_dfm_model_minimal
    
    # Get training data statistics (after internal standardization)
    if hasattr(model, 'data_processed') and model.data_processed is not None:
        training_mean = np.nanmean(model.data_processed, axis=0)
        training_std = np.nanstd(model.data_processed, axis=0)
    else:
        pytest.skip("Model data_processed not available")
    
    # Simulate what happens in forecast pipeline
    # Update data should be in same transformed space (chg/logdiff) as training
    update_data = small_test_data.select_dtypes(include=[np.number]).values
    
    # Check if scaler exists
    if hasattr(model, 'scaler') and model.scaler is not None:
        # Standardize update data using scaler (as done in model.update())
        update_standardized = model.scaler.transform(update_data)
        update_mean = np.nanmean(update_standardized, axis=0)
        update_std = np.nanstd(update_standardized, axis=0)
        
        # After standardization, means should be close to 0, stds close to 1
        # But we check that the standardization is applied correctly
        assert np.allclose(update_mean, 0, atol=1.0), \
            f"Update data after standardization should have mean≈0, got {update_mean}"
        # Standardization should produce std≈1
        assert np.allclose(update_std, 1, atol=0.5), \
            f"Update data after standardization should have std≈1, got {update_std}"
    else:
        # If scaler is missing, model.update() assumes data is already standardized
        pytest.skip("Scaler not available - model assumes data is already standardized")


@pytest.mark.skipif(not DFM_AVAILABLE, reason=f"DFM not available: {IMPORT_ERROR if not DFM_AVAILABLE else ''}")
def test_dfm_predictions_reasonable_range(trained_dfm_model_minimal, small_test_data):
    """Test that predictions are in reasonable range (not orders of magnitude off)."""
    checkpoint_path, dataset_path, model, dataset = trained_dfm_model_minimal
    
    # Run minimal recursive forecast (2 weeks)
    predictions, actuals, dates = run_recursive_forecast(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        test_data=small_test_data,
        start_date='2023-01-01',
        end_date='2023-01-15',  # 2 weeks
        covariates=['KOWRCCNSE', 'A001']  # Exclude these from targets
    )
    
    assert predictions is not None, "Predictions should not be None"
    assert actuals is not None, "Actuals should not be None"
    assert len(predictions) > 0, "Should have predictions"
    assert len(actuals) > 0, "Should have actuals"
    
    # Check that predictions are in reasonable range
    # Since data is in transformed space (chg/logdiff), values should be small
    pred_abs_max = np.abs(predictions).max()
    actual_abs_max = np.abs(actuals).max()
    
    # Predictions should not be orders of magnitude larger than actuals
    scale_ratio = pred_abs_max / (actual_abs_max + 1e-10)
    
    # If scale ratio > 100, we have a scaling issue
    assert scale_ratio < 100, \
        f"Predictions are orders of magnitude off! Scale ratio: {scale_ratio:.2f}, " \
        f"Pred max: {pred_abs_max:.2f}, Actual max: {actual_abs_max:.2f}"
    
    # Predictions should be in similar range as actuals (within 10x)
    assert scale_ratio < 10, \
        f"Predictions are too large! Scale ratio: {scale_ratio:.2f}"


@pytest.mark.skipif(not DFM_AVAILABLE, reason=f"DFM not available: {IMPORT_ERROR if not DFM_AVAILABLE else ''}")
def test_dfm_update_data_alignment(trained_dfm_model_minimal, small_test_data):
    """Test that update data is properly aligned with training data structure."""
    checkpoint_path, dataset_path, model, dataset = trained_dfm_model_minimal
    
    # Simulate forecast pipeline: get update data
    start_date = pd.Timestamp('2023-01-01')
    update_data_raw = small_test_data[small_test_data.index < start_date].copy()
    
    # Skip if no update data available
    if len(update_data_raw) == 0:
        pytest.skip("No update data available for this test")
    
    # Get numeric columns (as done in forecast pipeline)
    numeric_cols = update_data_raw.select_dtypes(include=[np.number]).columns.tolist()
    update_data = update_data_raw[numeric_cols].values
    
    # Check that update data has same number of columns as training
    if hasattr(model, 'data_processed') and model.data_processed is not None:
        n_training_cols = model.data_processed.shape[1]
        n_update_cols = update_data.shape[1]
        
        assert n_update_cols == n_training_cols, \
            f"Column count mismatch! Training: {n_training_cols}, Update: {n_update_cols}"
    
    # Check that update can be standardized (if scaler exists)
    if hasattr(model, 'scaler') and model.scaler is not None:
        try:
            update_standardized = model.scaler.transform(update_data)
            assert update_standardized.shape[0] == update_data.shape[0], \
                "Standardization should preserve number of rows"
            assert update_standardized.shape[1] == update_data.shape[1], \
                "Standardization should preserve number of columns"
        except Exception as e:
            pytest.fail(f"Failed to standardize update data: {e}")
    else:
        # If scaler is missing, model.update() assumes data is already standardized
        # This is acceptable and will be tested in the actual forecast test
        pass


@pytest.mark.skipif(not DFM_AVAILABLE, reason=f"DFM not available: {IMPORT_ERROR if not DFM_AVAILABLE else ''}")
def test_dfm_reproduces_scaling_issue(trained_dfm_model_minimal, small_test_data):
    """Test that reproduces the scaling issue if present.
    
    This test is designed to fail if the scaling issue exists,
    helping identify the root cause.
    """
    checkpoint_path, dataset_path, model, dataset = trained_dfm_model_minimal
    
    # Run recursive forecast
    predictions, actuals, dates = run_recursive_forecast(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        test_data=small_test_data,
        start_date='2023-01-01',
        end_date='2023-01-15',
        covariates=['KOWRCCNSE', 'A001']  # Exclude these from targets
    )
    
    if predictions is None or actuals is None:
        pytest.skip("Forecast failed, cannot test scaling")
    
    # Calculate scale ratios
    scale_ratios = np.abs(predictions) / (np.abs(actuals) + 1e-10)
    max_ratio = np.nanmax(scale_ratios)
    mean_ratio = np.nanmean(scale_ratios)
    
    # Log diagnostic information
    print(f"\n=== DFM Scaling Diagnostic ===")
    print(f"Predictions stats: min={np.nanmin(predictions):.4f}, max={np.nanmax(predictions):.4f}, "
          f"mean={np.nanmean(predictions):.4f}, std={np.nanstd(predictions):.4f}")
    print(f"Actuals stats: min={np.nanmin(actuals):.4f}, max={np.nanmax(actuals):.4f}, "
          f"mean={np.nanmean(actuals):.4f}, std={np.nanstd(actuals):.4f}")
    print(f"Scale ratios: mean={mean_ratio:.2f}, max={max_ratio:.2f}")
    
    # Check internal scaler
    if hasattr(model, 'scaler') and model.scaler is not None:
        scaler = model.scaler
        if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
            print(f"Internal scaler: mean range=[{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}], "
                  f"scale range=[{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")
    
    # Check column order
    if hasattr(dataset, '_processed_columns'):
        print(f"Dataset processed columns: {list(dataset._processed_columns)}")
    
    # The test passes if scale ratio is reasonable
    # If it fails, we've reproduced the scaling issue
    assert max_ratio < 100, \
        f"SCALING ISSUE REPRODUCED! Max scale ratio: {max_ratio:.2f} (expected < 100)"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
