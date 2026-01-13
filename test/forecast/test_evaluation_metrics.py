"""Tests for evaluation metrics computation."""

import sys
from pathlib import Path
import numpy as np
import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import compute_smse, compute_smae


def test_smse_perfect_prediction():
    """Test sMSE with perfect predictions (should be 0)."""
    y_true = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
    y_pred = y_true.copy()
    
    smse = compute_smse(y_true, y_pred)
    assert abs(smse) < 1e-10  # Should be essentially zero


def test_smse_manual_calculation():
    """Test sMSE matches manual calculation."""
    # Simple case: 2 series, 3 time points
    y_true = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    y_pred = np.array([[1.1, 2.2], [2.1, 4.2], [3.1, 6.2]])
    
    # Manual calculation
    errors = y_true - y_pred
    mse_per_series = np.mean(errors ** 2, axis=0)
    var_per_series = np.var(y_true, axis=0, ddof=0)
    smse_manual = np.mean(mse_per_series / var_per_series)
    
    smse = compute_smse(y_true, y_pred)
    
    assert abs(smse - smse_manual) < 1e-10


def test_smae_perfect_prediction():
    """Test sMAE with perfect predictions."""
    y_true = np.array([[1.0, 2.0], [2.0, 3.0]])
    y_pred = y_true.copy()
    
    smae = compute_smae(y_true, y_pred)
    assert abs(smae) < 1e-10


def test_smae_manual_calculation():
    """Test sMAE matches manual calculation."""
    y_true = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    y_pred = np.array([[1.1, 2.2], [2.1, 4.2], [3.1, 6.2]])
    
    # Manual calculation
    mae_per_series = np.mean(np.abs(y_true - y_pred), axis=0)
    std_per_series = np.std(y_true, axis=0, ddof=0)
    smae_manual = np.mean(mae_per_series / std_per_series)
    
    smae = compute_smae(y_true, y_pred)
    
    assert abs(smae - smae_manual) < 1e-10


def test_metrics_with_nan():
    """Test metrics handle NaN values."""
    y_true = np.array([[1.0, 2.0], [np.nan, 3.0], [3.0, 4.0]])
    y_pred = np.array([[1.1, 2.1], [2.1, 3.1], [3.1, 4.1]])
    
    # Should not crash
    smse = compute_smse(y_true, y_pred)
    smae = compute_smae(y_true, y_pred)
    
    assert not np.isnan(smse)
    assert not np.isnan(smae)
