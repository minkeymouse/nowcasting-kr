"""Tests for metric computation functions."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.metric import (
    compute_smse,
    compute_smae,
    compute_test_data_std,
    validate_std_for_metrics,
    save_experiment_results
)


@pytest.fixture
def sample_predictions():
    """Create sample predictions and actuals."""
    np.random.seed(42)
    n_samples = 100
    n_series = 3
    
    y_true = np.random.randn(n_samples, n_series) * 10 + 100
    y_pred = y_true + np.random.randn(n_samples, n_series) * 2  # Add some noise
    
    return y_true, y_pred


@pytest.fixture
def sample_test_data():
    """Create sample test data."""
    dates = pd.date_range('2020-01-01', periods=200, freq='W')
    return pd.DataFrame(
        np.random.randn(200, 3) * 10 + 100,
        columns=['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
        index=dates
    )


def test_compute_smse_basic(sample_predictions):
    """Test basic sMSE computation."""
    y_true, y_pred = sample_predictions
    
    smse = compute_smse(y_true, y_pred)
    
    assert isinstance(smse, float)
    assert smse >= 0
    assert not np.isnan(smse)
    assert not np.isinf(smse)


def test_compute_smse_with_test_std(sample_predictions):
    """Test sMSE computation with test_data_std."""
    y_true, y_pred = sample_predictions
    test_std = np.array([10.0, 15.0, 20.0])
    
    smse = compute_smse(y_true, y_pred, test_data_std=test_std)
    
    assert isinstance(smse, float)
    assert smse >= 0


def test_compute_smse_per_series(sample_predictions):
    """Test sMSE computation per series."""
    y_true, y_pred = sample_predictions
    
    smse_per_series = compute_smse(y_true, y_pred, per_series=True)
    
    assert isinstance(smse_per_series, np.ndarray)
    assert len(smse_per_series) == y_true.shape[1]
    assert np.all(smse_per_series >= 0)


def test_compute_smae_basic(sample_predictions):
    """Test basic sMAE computation."""
    y_true, y_pred = sample_predictions
    
    smae = compute_smae(y_true, y_pred)
    
    assert isinstance(smae, float)
    assert smae >= 0
    assert not np.isnan(smae)
    assert not np.isinf(smae)


def test_compute_smae_with_test_std(sample_predictions):
    """Test sMAE computation with test_data_std."""
    y_true, y_pred = sample_predictions
    test_std = np.array([10.0, 15.0, 20.0])
    
    smae = compute_smae(y_true, y_pred, test_data_std=test_std)
    
    assert isinstance(smae, float)
    assert smae >= 0


def test_compute_smae_per_series(sample_predictions):
    """Test sMAE computation per series."""
    y_true, y_pred = sample_predictions
    
    smae_per_series = compute_smae(y_true, y_pred, per_series=True)
    
    assert isinstance(smae_per_series, np.ndarray)
    assert len(smae_per_series) == y_true.shape[1]
    assert np.all(smae_per_series >= 0)


def test_compute_test_data_std_weekly(sample_test_data):
    """Test test data std computation for weekly series."""
    target_series = ['KOEQUIPTE', 'KOWRCCNSE', 'A001']
    
    std = compute_test_data_std(sample_test_data, target_series, monthly_series=None)
    
    assert std is not None
    assert len(std) == len(target_series)
    assert np.all(std > 0)


def test_compute_test_data_std_monthly(sample_test_data):
    """Test test data std computation for monthly series."""
    target_series = ['KOEQUIPTE', 'KOWRCCNSE', 'A001']
    monthly_series = {'KOEQUIPTE'}  # One monthly series
    
    std = compute_test_data_std(sample_test_data, target_series, monthly_series=monthly_series)
    
    assert std is not None
    assert len(std) == len(target_series)
    assert np.all(std > 0)


def test_validate_std_for_metrics():
    """Test std validation for metrics."""
    n_series = 3
    
    # Valid std
    valid_std = np.array([10.0, 15.0, 20.0])
    result = validate_std_for_metrics(valid_std, n_series)
    assert result is not None
    assert np.array_equal(result, valid_std)
    
    # Invalid std (wrong length)
    invalid_std = np.array([10.0, 15.0])
    result = validate_std_for_metrics(invalid_std, n_series)
    assert result is None
    
    # None std
    result = validate_std_for_metrics(None, n_series)
    assert result is None


def test_save_experiment_results(tmp_path):
    """Test saving experiment results."""
    predictions = np.random.randn(100, 3)
    actuals = np.random.randn(100, 3)
    dates = pd.date_range('2020-01-01', periods=100, freq='W')
    target_series = ['KOEQUIPTE', 'KOWRCCNSE', 'A001']
    metrics = {'smse': 0.5, 'smae': 0.3}
    
    save_experiment_results(
        output_dir=tmp_path,
        predictions=predictions,
        actuals=actuals,
        dates=dates,
        target_series=target_series,
        metrics=metrics
    )
    
    # Verify files were created
    assert (tmp_path / "predictions.csv").exists()
    assert (tmp_path / "actuals.csv").exists()
    assert (tmp_path / "metrics.json").exists()
    
    # Verify predictions file
    pred_df = pd.read_csv(tmp_path / "predictions.csv", index_col=0, parse_dates=True)
    assert pred_df.shape == (100, 3)
    assert list(pred_df.columns) == target_series


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
