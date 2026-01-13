"""Test to verify scaling pipeline consistency across training and forecasting."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.preprocess import InvestmentData
from src.utils import (
    standardize_data, 
    inverse_transform_predictions, 
    load_test_data
)


def test_scaling_pipeline_consistency():
    """Verify that scaling works correctly through the full pipeline.
    
    This test documents the current behavior and identifies issues:
    1. Training data standardization
    2. Test data standardization (during forecasting)
    3. Prediction inverse transformation
    """
    data_loader = InvestmentData()
    test_data = load_test_data('investment')
    training_data = data_loader.training_data
    scaler = data_loader.scaler
    
    # Check a series that has sufficient training data
    # Find a series with good coverage
    good_series = None
    for col in training_data.columns:
        if training_data[col].notna().sum() > 100:
            good_series = col
            break
    
    if good_series is None:
        pytest.skip("No series with sufficient training data found")
    
    print(f"\n=== Testing with series: {good_series} ===")
    
    # 1. Training data should be standardized (mean≈0, std≈1)
    train_vals = training_data[good_series].dropna().values
    train_mean = np.mean(train_vals)
    train_std = np.std(train_vals)
    print(f"Training data: mean={train_mean:.4f}, std={train_std:.4f}")
    assert abs(train_mean) < 1e-6, f"Training data should be standardized (mean≈0), got {train_mean}"
    assert abs(train_std - 1.0) < 0.5, f"Training data should be standardized (std≈1), got {train_std}"
    
    # 2. Test data standardization should work
    test_sample = test_data[[good_series]].head(10)
    test_original = test_sample[good_series].values
    
    test_standardized = standardize_data(test_sample, [good_series], scaler, data_loader)
    test_std_vals = test_standardized[good_series].values
    
    print(f"Test data (original): mean={np.mean(test_original):.2f}, std={np.std(test_original):.2f}")
    print(f"Test data (standardized): mean={np.mean(test_std_vals):.4f}, std={np.std(test_std_vals):.4f}")
    
    # Standardized test data should have mean≈0, std≈1
    # (may not be exact if test distribution differs from training)
    assert abs(np.mean(test_std_vals)) < 10, f"Standardized test data mean should be close to 0, got {np.mean(test_std_vals)}"
    
    # 3. Inverse transform should recover original scale
    std_prediction = np.array([0.5, 1.0, -0.5])  # Standardized predictions
    original_prediction = inverse_transform_predictions(std_prediction, [good_series], data_loader)
    
    # Check scaler indices
    idx = list(training_data.columns).index(good_series)
    mean = scaler.mean_[idx]
    scale = scaler.scale_[idx]
    
    manual_inverse = std_prediction * scale + mean
    print(f"\nInverse transform test:")
    print(f"  Input (std): {std_prediction}")
    print(f"  Output (original): {original_prediction}")
    print(f"  Manual calc: {manual_inverse}")
    print(f"  Scaler mean: {mean:.2f}, scale: {scale:.2f}")
    
    assert np.allclose(original_prediction, manual_inverse, rtol=1e-5), \
        "Inverse transform should match manual calculation"
    
    # 4. Round-trip test: original -> standardize -> inverse should recover original
    original_values = test_original[:3]
    standardized = standardize_data(
        pd.DataFrame({good_series: original_values}), 
        [good_series], 
        scaler, 
        data_loader
    )[good_series].values
    
    recovered = inverse_transform_predictions(standardized, [good_series], data_loader)
    
    print(f"\nRound-trip test:")
    print(f"  Original: {original_values}")
    print(f"  Standardized: {standardized}")
    print(f"  Recovered: {recovered}")
    
    assert np.allclose(original_values, recovered, rtol=1e-5), \
        "Round-trip (original -> standardize -> inverse) should recover original values"


def test_insufficient_training_data_issue():
    """Document the issue when training data has insufficient values.
    
    When a series has very few non-null values (e.g., KOEQUIPTE with only 1 value),
    the scaler may have scale=1, which makes standardization ineffective.
    """
    data_loader = InvestmentData()
    test_data = load_test_data('investment')
    training_data = data_loader.training_data
    scaler = data_loader.scaler
    
    # Check KOEQUIPTE specifically
    if 'KOEQUIPTE' not in training_data.columns:
        pytest.skip("KOEQUIPTE not in training data")
    
    train_vals = training_data['KOEQUIPTE'].dropna().values
    idx = list(training_data.columns).index('KOEQUIPTE')
    mean = scaler.mean_[idx]
    scale = scaler.scale_[idx]
    
    print(f"\n=== Issue with insufficient training data: KOEQUIPTE ===")
    print(f"Training data: {len(train_vals)} non-null values out of {len(training_data)}")
    print(f"Training values: {train_vals}")
    print(f"Scaler mean: {mean:.2f}, scale: {scale:.2f}")
    
    # Test standardization
    test_vals = test_data['KOEQUIPTE'].dropna().values[:5]
    test_df = pd.DataFrame({'KOEQUIPTE': test_vals})
    standardized = standardize_data(test_df, ['KOEQUIPTE'], scaler, data_loader)
    std_vals = standardized['KOEQUIPTE'].values
    
    print(f"Test data: {test_vals}")
    print(f"After standardization: {std_vals}")
    print(f"Expected: mean~0, std~1")
    print(f"Actual: mean={np.mean(std_vals):.4f}, std={np.std(std_vals):.4f}")
    
    # With scale=1, standardization formula: (x - mean) / 1 = x - mean
    # If mean=0, then (x - 0) / 1 = x (no change!)
    if scale == 1.0 and abs(mean) < 1e-6:
        print("WARNING: Scale=1, mean=0 means standardization does nothing!")
        print("Test data stays in original scale (~470), but model expects standardized scale (~0)")
    
    # Test inverse transform
    std_pred = np.array([0.0])  # Model predicts 0 in standardized scale
    original = inverse_transform_predictions(std_pred, ['KOEQUIPTE'], data_loader)
    
    print(f"\nInverse transform:")
    print(f"  Input (std pred): {std_pred[0]:.2f}")
    print(f"  Output: {original[0]:.2f}")
    print(f"  Actual test value: ~{np.mean(test_vals):.2f}")
    
    if abs(original[0]) < 1e-6 and np.mean(test_vals) > 100:
        print("ISSUE: Inverse transform returns ~0, but actuals are ~470")
        print("This causes huge scale mismatch in metrics!")


if __name__ == '__main__':
    test_scaling_pipeline_consistency()
    test_insufficient_training_data_issue()
