"""Tests for experiment utilities."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import (
    get_experiment_dates,
    get_weekly_dates,
    split_data_by_date,
    compute_smse,
    compute_smae,
    save_experiment_results,
    load_model_checkpoint,
    preprocess_data_for_model,
    extract_forecast_values
)


def test_get_experiment_dates():
    """Test date range extraction for experiments."""
    # Short-term
    dates = get_experiment_dates("short_term")
    assert dates["start_date"] == "2024-01-01"
    assert dates["end_date"] == "2025-10-04"
    
    # Long-term
    dates = get_experiment_dates("long_term")
    assert dates["start_date"] == "2025-01-01"
    assert dates["end_date"] == "2025-10-04"
    
    # Invalid
    with pytest.raises(ValueError):
        get_experiment_dates("invalid")


def test_get_weekly_dates():
    """Test weekly date generation."""
    dates = get_weekly_dates("2024-01-01", "2024-01-31")
    assert len(dates) >= 4  # At least 4 weeks
    assert dates[0] == pd.Timestamp("2024-01-01")
    assert dates[-1] <= pd.Timestamp("2024-01-31")


def test_split_data_by_date():
    """Test temporal data splitting."""
    # Create test data
    dates = pd.date_range("2024-01-01", "2024-01-31", freq='W-MON')
    data = pd.DataFrame(
        np.random.randn(len(dates), 3),
        index=dates,
        columns=['series1', 'series2', 'series3']
    )
    
    cutoff = pd.Timestamp("2024-01-15")
    train, test = split_data_by_date(data, cutoff)
    
    assert len(train) > 0
    assert len(test) > 0
    assert train.index.max() < test.index.min()
    assert train.index.max() < cutoff
    assert test.index.min() >= cutoff


def test_compute_smse():
    """Test standardized MSE computation."""
    # Simple test case
    y_true = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y_pred = np.array([[1.1, 2.1], [2.1, 3.1], [3.1, 4.1]])
    
    smse = compute_smse(y_true, y_pred)
    assert smse >= 0
    assert not np.isnan(smse)
    
    # Test with zero variance (constant series)
    y_true_const = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    y_pred_const = np.array([[1.1, 1.1], [1.1, 1.1], [1.1, 1.1]])
    smse_const = compute_smse(y_true_const, y_pred_const)
    assert not np.isnan(smse_const)
    
    # Test 1D
    smse_1d = compute_smse(y_true[:, 0], y_pred[:, 0])
    assert smse_1d >= 0


def test_compute_smae():
    """Test standardized MAE computation."""
    y_true = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y_pred = np.array([[1.1, 2.1], [2.1, 3.1], [3.1, 4.1]])
    
    smae = compute_smae(y_true, y_pred)
    assert smae >= 0
    assert not np.isnan(smae)
    
    # Test 1D
    smae_1d = compute_smae(y_true[:, 0], y_pred[:, 0])
    assert smae_1d >= 0


def test_save_experiment_results(tmp_path):
    """Test saving experiment results."""
    predictions = np.array([[1.0, 2.0], [1.1, 2.1]])
    actuals = np.array([[1.05, 2.05], [1.15, 2.15]])
    dates = pd.DatetimeIndex(['2024-01-01', '2024-01-08'])
    target_series = ['series1', 'series2']
    metrics = {'smse': 0.01, 'smae': 0.1}
    
    output_dir = tmp_path / "test_results"
    save_experiment_results(
        output_dir=output_dir,
        predictions=predictions,
        actuals=actuals,
        dates=dates,
        target_series=target_series,
        metrics=metrics
    )
    
    # Check files exist
    assert (output_dir / "predictions.csv").exists()
    assert (output_dir / "actuals.csv").exists()
    assert (output_dir / "metrics.json").exists()
    
    # Check metrics file
    import json
    with open(output_dir / "metrics.json") as f:
        loaded_metrics = json.load(f)
    assert loaded_metrics == metrics
