"""Inference/Nowcasting module - supports both CLI and programmatic API.

This module provides inference and nowcasting functionality that can be used:
- As CLI: python src/infer.py nowcast --config-name experiment/kogdp_report
- As API: from src.infer import run_nowcasting_evaluation

Functions exported for programmatic use:
- run_nowcasting_evaluation: Run nowcasting evaluation with masked data
"""

from pathlib import Path
import sys
import argparse
from typing import Dict, Any, Optional, List

# Set up paths first before any relative imports
# This allows the script to be run directly as python3 src/infer.py
_script_dir = Path(__file__).parent.resolve()
_project_root = _script_dir.parent.resolve()  # src/ -> project root
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

# Now use absolute imports
from src.utils.config_parser import setup_paths, get_project_root
setup_paths(include_dfm_python=True, include_src=True, include_app=True)

from src.utils.config_parser import parse_experiment_config, extract_experiment_params, validate_experiment_config

# Import nowcasting functions (merged from nowcasting.py)
from typing import Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def mask_recent_observations(
    data: Union[pd.DataFrame, pd.Series, np.ndarray],
    mask_days: int,
    date_index: Optional[pd.DatetimeIndex] = None,
    target_date: Optional[Union[datetime, str]] = None,
    mask_columns: Optional[Union[list, str]] = None
) -> Tuple[Union[pd.DataFrame, pd.Series, np.ndarray], np.ndarray]:
    """Mask recent observations to simulate publication lag.
    
    This function masks the most recent `mask_days` days of data to simulate
    a nowcasting scenario where recent data is not yet available due to
    publication lags.
    
    Parameters
    ----------
    data : pd.DataFrame, pd.Series, or np.ndarray
        Time series data (T × N) or (T,)
    mask_days : int
        Number of days to mask from the end (simulates publication lag)
    date_index : pd.DatetimeIndex, optional
        Date index for the data. If None and data is DataFrame/Series with
        DatetimeIndex, uses data.index
    target_date : datetime or str, optional
        Target date to mask from. If None, uses the last date in data.
        Data after (target_date - mask_days) will be masked.
    mask_columns : list or str, optional
        Columns to mask (for DataFrame). If None, masks all columns.
        If string, masks only that column.
        
    Returns
    -------
    masked_data : pd.DataFrame, pd.Series, or np.ndarray
        Data with recent observations masked (set to NaN)
    mask : np.ndarray
        Boolean mask indicating which observations were masked (True = masked)
    """
    # Get date index
    if date_index is None:
        if isinstance(data, (pd.DataFrame, pd.Series)):
            if isinstance(data.index, pd.DatetimeIndex):
                date_index = data.index
            else:
                # Integer index - create dummy dates
                date_index = pd.date_range('2000-01-01', periods=len(data), freq='D')
        else:
            # Numpy array - create dummy dates
            date_index = pd.date_range('2000-01-01', periods=len(data), freq='D')
    
    # Determine target date
    if target_date is None:
        target_date = date_index[-1]
    elif isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    
    # Calculate cutoff date
    cutoff_date = target_date - timedelta(days=mask_days)
    
    # Create mask: True for dates after cutoff (to be masked)
    mask = date_index > cutoff_date
    
    # Apply mask to data
    if isinstance(data, pd.DataFrame):
        masked_data = data.copy()
        if mask_columns is None:
            # Mask all columns
            masked_data.loc[mask, :] = np.nan
        elif isinstance(mask_columns, str):
            # Mask single column
            masked_data.loc[mask, mask_columns] = np.nan
        else:
            # Mask specified columns
            masked_data.loc[mask, mask_columns] = np.nan
    elif isinstance(data, pd.Series):
        masked_data = data.copy()
        masked_data.loc[mask] = np.nan
    else:
        # Numpy array
        masked_data = data.copy()
        if data.ndim == 1:
            masked_data[mask] = np.nan
        else:
            masked_data[mask, :] = np.nan
    
    return masked_data, mask


def create_nowcasting_splits(
    data: Union[pd.DataFrame, pd.Series],
    target_dates: list,
    mask_days: int,
    train_size: Optional[int] = None,
    test_size: int = 1
) -> list:
    """Create train/test splits for nowcasting backtesting.
    
    For each target date, creates a split where:
    - Training data: All data available up to (target_date - mask_days)
    - Test data: Target date (to be nowcasted)
    
    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        Time series data with DatetimeIndex
    target_dates : list
        List of target dates (datetime or str) to nowcast
    mask_days : int
        Publication lag in days (data after target_date - mask_days is masked)
    train_size : int, optional
        Maximum number of training samples. If None, uses all available data.
    test_size : int, default=1
        Number of test samples (typically 1 for nowcasting)
        
    Returns
    -------
    list
        List of tuples (train_idx, test_idx) for each target date
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("data must have DatetimeIndex")
    
    splits = []
    
    for target_date in target_dates:
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        
        # Calculate cutoff date
        cutoff_date = target_date - timedelta(days=mask_days)
        
        # Find indices
        train_mask = data.index <= cutoff_date
        test_mask = data.index == target_date
        
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        
        # Limit training size if specified
        if train_size is not None and len(train_idx) > train_size:
            train_idx = train_idx[-train_size:]
        
        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))
    
    return splits


def simulate_nowcasting_evaluation(
    forecaster,
    data: Union[pd.DataFrame, pd.Series],
    target_dates: list,
    mask_days: int,
    horizons: Union[list, np.ndarray] = [1],
    target_series: Optional[Union[str, int]] = None
) -> Dict[str, Any]:
    """Simulate nowcasting evaluation with masked data.
    
    This function simulates a nowcasting scenario by:
    1. Masking recent observations (publication lag)
    2. Training on available data
    3. Evaluating on target dates
    4. Comparing with full-data forecasts
    
    Parameters
    ----------
    forecaster : sktime BaseForecaster or callable
        Forecaster to evaluate. If callable, should accept (y_train, fh) and return predictions.
    data : pd.DataFrame or pd.Series
        Full time series data with DatetimeIndex
    target_dates : list
        List of target dates to nowcast
    mask_days : int
        Publication lag in days
    horizons : list or np.ndarray, default=[1]
        Forecast horizons to evaluate
    target_series : str or int, optional
        Target series to evaluate (for multivariate data)
        
    Returns
    -------
    dict
        Dictionary with evaluation results:
        - 'nowcast_metrics': Metrics for nowcasting (with masked data)
        - 'full_metrics': Metrics for full-data forecasting (baseline)
        - 'improvement': Improvement of full-data over nowcasting
        - 'target_dates': List of evaluated target dates
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("data must have DatetimeIndex")
    
    # Create splits
    splits = create_nowcasting_splits(data, target_dates, mask_days)
    
    nowcast_results = []
    full_results = []
    
    for train_idx, test_idx in splits:
        if len(test_idx) == 0:
            continue
        
        # Get training and test data
        y_train = data.iloc[train_idx]
        y_test = data.iloc[test_idx]
        target_date = data.index[test_idx[0]]
        
        # Nowcasting: Use masked training data
        y_train_masked, _ = mask_recent_observations(
            y_train, mask_days=mask_days, target_date=target_date
        )
        
        # Remove rows that are all NaN
        y_train_masked = y_train_masked.dropna(how='all')
        
        if len(y_train_masked) == 0:
            continue
        
        # Fit and predict with masked data (nowcasting)
        try:
            if hasattr(forecaster, 'fit') and hasattr(forecaster, 'predict'):
                # sktime forecaster
                forecaster.fit(y_train_masked)
                fh = np.asarray(horizons)
                y_pred_nowcast = forecaster.predict(fh=fh)
            else:
                # Callable
                y_pred_nowcast = forecaster(y_train_masked, fh=horizons)
            
            # Calculate nowcasting metrics
            from src.eval.evaluation import calculate_metrics_per_horizon
            nowcast_metrics = calculate_metrics_per_horizon(
                y_test, y_pred_nowcast, horizons, y_train=y_train_masked,
                target_series=target_series
            )
            nowcast_results.append({
                'target_date': target_date,
                'metrics': nowcast_metrics
            })
        except Exception as e:
            print(f"Error in nowcasting evaluation for {target_date}: {e}")
            continue
        
        # Full data: Use all training data (baseline)
        try:
            if hasattr(forecaster, 'fit') and hasattr(forecaster, 'predict'):
                forecaster.fit(y_train)
                y_pred_full = forecaster.predict(fh=fh)
            else:
                y_pred_full = forecaster(y_train, fh=horizons)
            
            # Calculate full-data metrics
            full_metrics = calculate_metrics_per_horizon(
                y_test, y_pred_full, horizons, y_train=y_train,
                target_series=target_series
            )
            full_results.append({
                'target_date': target_date,
                'metrics': full_metrics
            })
        except Exception as e:
            print(f"Error in full-data evaluation for {target_date}: {e}")
            continue
    
    # Aggregate results across target dates
    # Average metrics across all target dates
    if len(nowcast_results) > 0 and len(full_results) > 0:
        # Get all horizons
        all_horizons = set()
        for result in nowcast_results + full_results:
            all_horizons.update(result['metrics'].keys())
        
        nowcast_avg = {}
        full_avg = {}
        
        for h in all_horizons:
            nowcast_vals = [r['metrics'].get(h, {}) for r in nowcast_results]
            full_vals = [r['metrics'].get(h, {}) for r in full_results]
            
            # Average each metric
            nowcast_avg[h] = {
                'sMSE': np.nanmean([v.get('sMSE', np.nan) for v in nowcast_vals]),
                'sMAE': np.nanmean([v.get('sMAE', np.nan) for v in nowcast_vals]),
                'sRMSE': np.nanmean([v.get('sRMSE', np.nan) for v in nowcast_vals]),
            }
            full_avg[h] = {
                'sMSE': np.nanmean([v.get('sMSE', np.nan) for v in full_vals]),
                'sMAE': np.nanmean([v.get('sMAE', np.nan) for v in full_vals]),
                'sRMSE': np.nanmean([v.get('sRMSE', np.nan) for v in full_vals]),
            }
        
        # Calculate improvement (negative = nowcasting worse, positive = better)
        improvement = {}
        for h in all_horizons:
            improvement[h] = {
                'sMSE': full_avg[h]['sMSE'] - nowcast_avg[h]['sMSE'],
                'sMAE': full_avg[h]['sMAE'] - nowcast_avg[h]['sMAE'],
                'sRMSE': full_avg[h]['sRMSE'] - nowcast_avg[h]['sRMSE'],
            }
    else:
        nowcast_avg = {}
        full_avg = {}
        improvement = {}
    
    return {
        'nowcast_metrics': nowcast_avg,
        'full_metrics': full_avg,
        'improvement': improvement,
        'target_dates': [r['target_date'] for r in nowcast_results],
        'n_evaluations': len(nowcast_results)
    }


def run_nowcasting_evaluation(
    config_name: str,
    config_dir: Optional[str] = None,
    mask_days: int = 30,
    overrides: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Run nowcasting evaluation with masked data (programmatic API).
    
    Parameters
    ----------
    config_name : str
        Experiment config name (e.g., 'experiment/kogdp_report')
    config_dir : str, optional
        Config directory path. If None, uses default config/ directory.
    mask_days : int, default 30
        Number of days to mask for nowcasting simulation
    overrides : list of str, optional
        Hydra config overrides
        
    Returns
    -------
    dict
        Nowcasting evaluation results
    """
    if config_dir is None:
        config_dir = str(get_project_root() / "config")
    
    cfg = parse_experiment_config(config_name, config_dir, overrides)
    validate_experiment_config(cfg, require_target=True, require_models=False)
    
    params = extract_experiment_params(cfg)
    
    if not params['data_path']:
        raise ValueError(f"Config {config_name} must specify 'data_path'")
    
    print("=" * 70)
    print(f"Running nowcasting evaluation for {params['target_series']}")
    print(f"Mask days: {mask_days} | Horizons: {params['horizons']}")
    print("=" * 70)
    
    # Nowcasting evaluation implementation
    # 1. Load trained model from outputs/
    # 2. Load data
    # 3. Run simulate_nowcasting_evaluation
    print("\nRunning nowcasting evaluation...")
    print(f"Config: {config_name}")
    print(f"Target: {params['target_series']}")
    print(f"Data: {params['data_path']}")
    
    # TODO: Implement full nowcasting evaluation
    # This requires:
    # 1. Load trained model from outputs/models/{target}_{model}/model.pkl
    # 2. Load data from params['data_path']
    # 3. Run simulate_nowcasting_evaluation() from nowcasting.py with target dates
    # 4. Calculate metrics per horizon using evaluation.py
    # 5. Return evaluation results (nowcast_metrics, full_metrics, improvement)
    # Note: Currently returns 'not_implemented' status. This is a known limitation.
    
    return {
        'status': 'not_implemented',
        'config_name': config_name,
        'target_series': params['target_series'],
        'mask_days': mask_days,
        'horizons': params['horizons']
    }


def main():
    """Main CLI entry point for inference/nowcasting."""
    parser = argparse.ArgumentParser(description="Run inference and nowcasting using Hydra config")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Nowcasting evaluation command
    nowcast_parser = subparsers.add_parser('nowcast', help='Run nowcasting evaluation (requires experiment config)')
    nowcast_parser.add_argument("--config-name", required=True, help="Experiment config name (e.g., experiment/kogdp_report)")
    nowcast_parser.add_argument("--override", action="append", help="Hydra config override")
    nowcast_parser.add_argument("--mask-days", type=int, default=30, help="Number of days to mask for nowcasting simulation")
    
    args = parser.parse_args()
    
    if args.command == 'nowcast':
        result = run_nowcasting_evaluation(
            config_name=args.config_name,
            mask_days=args.mask_days,
            overrides=args.override
        )
        print(f"\n✓ Nowcasting evaluation completed")
        if result.get('status') == 'not_implemented':
            print("  Note: Full implementation pending")


if __name__ == "__main__":
    main()

