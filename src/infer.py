"""Inference/Nowcasting module - supports both CLI and programmatic API.

This module provides inference and nowcasting functionality that can be used:
- As CLI: python src/infer.py nowcast --config-name experiment/investment_koequipte_report
- As API: from src.infer import run_nowcasting_evaluation, run_backtest_evaluation

Functions exported for programmatic use:
- run_nowcasting_evaluation: Run nowcasting evaluation with masked data
- run_backtest_evaluation: Run backtest evaluation for all models with multiple time points
"""
from pathlib import Path
import sys
import argparse
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Set up paths BEFORE importing from src
# Get script directory (e.g., src/)
script_dir = Path(__file__).parent.resolve()
# Get project root (parent of src/)
project_root = script_dir.parent.resolve()

# Add to sys.path if not already there (use insert(0) to prioritize)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# Change to project root to ensure relative imports work
import os
original_cwd = os.getcwd()
try:
    os.chdir(str(project_root))
    
    # Now we can import from src
    from src.utils.cli import setup_cli_environment
    setup_cli_environment()
finally:
    # Restore original working directory
    os.chdir(original_cwd)

# Now use absolute imports
from src.utils.config_parser import get_project_root

from src.utils.config_parser import (
    parse_experiment_config,
    extract_experiment_params,
    validate_experiment_config
)

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
    """Simulate nowcasting evaluation with publication lag masking.
    
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
                fh = np.asarray(horizons)
                y_pred_full = forecaster.predict(fh=fh)
            else:
                y_pred_full = forecaster(y_train, fh=horizons)
            
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
    
    return {
        'nowcast_metrics': nowcast_results,
        'full_metrics': full_results,
        'target_dates': target_dates
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
        Experiment config name (e.g., 'experiment/investment_koequipte_report')
    config_dir : str, optional
        Config directory path. If None, uses default config/ directory.
    mask_days : int, default=30
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
    validate_experiment_config(cfg, require_target=True, require_models=True)
    
    params = extract_experiment_params(cfg)
    target_series = params['target_series']
    models = params['models']
    
    print("=" * 70)
    print(f"Running nowcasting evaluation for {target_series}")
    print(f"Models: {', '.join(models)}")
    print(f"Mask days: {mask_days}")
    print("=" * 70)
    
    # Load data
    data_path = params.get('data_path') or str(get_project_root() / "data" / "sample_data.csv")
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Generate target dates (last 12 months)
    end_date = data.index[-1]
    target_dates = []
    current = end_date.replace(day=1)
    for _ in range(12):
        if current.month == 12:
            last_day = current.replace(day=31)
        else:
            next_month = current.replace(month=current.month + 1, day=1)
            last_day = next_month - timedelta(days=1)
        target_dates.append(last_day)
        current -= relativedelta(months=1)
    target_dates.reverse()
    
    results = {}
    for model_name in models:
        print(f"\nEvaluating {model_name.upper()}...")
        
        # Load model
        from pathlib import Path
        import pickle
        
        model_dir = None
        for comparison_dir in Path("outputs/comparisons").glob(f"{target_series}_*"):
            model_path = comparison_dir / model_name.lower() / "model.pkl"
            if model_path.exists():
                model_dir = comparison_dir / model_name.lower()
                break
        
        if model_dir is None:
            print(f"  ✗ Model not found, skipping")
            continue
        
        with open(model_dir / "model.pkl", 'rb') as f:
            model_data = pickle.load(f)
        
        forecaster = model_data.get('forecaster')
        if forecaster is None:
            print(f"  ✗ Forecaster not found, skipping")
            continue
        
        # Run evaluation
        eval_result = simulate_nowcasting_evaluation(
            forecaster=forecaster,
            data=data,
            target_dates=target_dates,
            mask_days=mask_days,
            horizons=[1],
            target_series=target_series
        )
        
        results[model_name] = eval_result
        print(f"  ✓ Completed")
    
    return results


def run_backtest_evaluation(
    config_name: str,
    model: str,
    train_start: str = "1985-01-01",
    train_end: str = "2019-12-31",
    nowcast_start: str = "2024-01-01",
    nowcast_end: str = "2024-12-31",
    weeks_before: Optional[List[int]] = None,
    config_dir: Optional[str] = None,
    overrides: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Run backtest evaluation with multiple nowcasting time points (benchmark report structure).
    
    This function implements the benchmark report's nowcasting structure:
    - For each target month (e.g., 2024-01, 2024-02, ..., 2024-12)
    - At multiple time points before month end (e.g., 4 weeks, 1 week)
    - Mask data based on release dates from series config
    - Generate 1 horizon forecast at each time point
    - Compare predictions across time points
    
    Parameters
    ----------
    config_name : str
        Experiment config name (e.g., 'experiment/investment_koequipte_report')
    model : str
        Model name ('arima', 'var', 'dfm', or 'ddfm')
    train_start : str
        Training period start date (YYYY-MM-DD)
    train_end : str
        Training period end date (YYYY-MM-DD)
    nowcast_start : str
        Nowcasting period start date (YYYY-MM-DD)
    nowcast_end : str
        Nowcasting period end date (YYYY-MM-DD)
    weeks_before : list of int, optional
        List of weeks before month end to perform nowcasting (e.g., [4, 1]).
        If None, defaults to [4, 1] (4 weeks and 1 week before).
    config_dir : str, optional
        Config directory path. If None, uses default config/ directory.
    overrides : list of str, optional
        Hydra config overrides
        
    Returns
    -------
    dict
        Backtest evaluation results with metrics by time point and month
    """
    if config_dir is None:
        config_dir = str(get_project_root() / "config")
    
    # Default weeks_before: 4 weeks and 1 week before (as per benchmark report)
    if weeks_before is None:
        weeks_before = [4, 1]
    
    # Backtest supports all models, but DFM/DDFM use nowcast manager
    # ARIMA/VAR use simulate_nowcasting_evaluation (with release-based masking)
    supports_nowcast_manager = model.lower() in ['dfm', 'ddfm']
    
    cfg = parse_experiment_config(config_name, config_dir, overrides)
    validate_experiment_config(cfg, require_target=True, require_models=False)
    
    params = extract_experiment_params(cfg)
    target_series = params['target_series']
    
    print("=" * 70)
    print(f"Running backtest for {target_series} - {model.upper()}")
    print(f"Train period: {train_start} to {train_end}")
    print(f"Nowcast period: {nowcast_start} to {nowcast_end}")
    print(f"Nowcasting time points: {weeks_before} weeks before month end")
    print("=" * 70)
    
    # Load trained model
    from pathlib import Path
    from src.model.dfm_models import DFM, DDFM
    import pickle
    
    # Find trained model - first check checkpoint/, then outputs/comparisons/
    model_dir = None
    checkpoint_path = Path("checkpoint") / f"{target_series}_{model.lower()}" / "model.pkl"
    if checkpoint_path.exists():
        model_dir = checkpoint_path.parent
    else:
        # Fallback to outputs/comparisons/
        for comparison_dir in Path("outputs/comparisons").glob(f"{target_series}_*"):
            model_path = comparison_dir / model.lower() / "model.pkl"
            if model_path.exists():
                model_dir = comparison_dir / model.lower()
                break
    
    if model_dir is None:
        raise FileNotFoundError(
            f"Trained {model.upper()} model not found for {target_series}. "
            f"Please run training first: bash run_train.sh"
        )
    
    # Load model
    with open(model_dir / "model.pkl", 'rb') as f:
        model_data = pickle.load(f)
    
    # For DFM/DDFM, get nowcast manager
    # For ARIMA/VAR, get forecaster
    nowcast_manager = None
    forecaster = None
    dfm_model = None
    
    if supports_nowcast_manager:
        dfm_model = model_data.get('dfm_model') or model_data.get('ddfm_model')
        if dfm_model is None:
            raise ValueError(f"Model file does not contain {model.upper()} model")
        nowcast_manager = dfm_model.nowcast
    else:
        # ARIMA/VAR: use forecaster directly
        forecaster = model_data.get('forecaster')
        if forecaster is None:
            raise ValueError(f"Model file does not contain forecaster for {model.upper()} model")
    
    # Generate monthly target periods
    start_date = datetime.strptime(nowcast_start, "%Y-%m-%d")
    end_date = datetime.strptime(nowcast_end, "%Y-%m-%d")
    
    # Generate target periods (last day of each month)
    target_periods = []
    current = start_date.replace(day=1)  # Start of month
    while current <= end_date:
        # Use last day of month as target period
        if current.month == 12:
            last_day = current.replace(day=31)
        else:
            next_month = current.replace(month=current.month + 1, day=1)
            last_day = next_month - timedelta(days=1)
        target_periods.append(last_day)
        current += relativedelta(months=1)
    
    print(f"\nTarget periods: {len(target_periods)} months")
    if len(target_periods) > 0:
        print(f"  From: {target_periods[0].strftime('%Y-%m-%d')}")
        print(f"  To: {target_periods[-1].strftime('%Y-%m-%d')}")
    print(f"  Nowcasting at: {weeks_before} weeks before each month end")
    print(f"  Horizon: 1 (1 horizon forecast at each time point)")
    
    # Run backtest for each time point (weeks_before)
    # For each target month, predict at multiple time points
    print("\nRunning backtest with release-based masking...")
    print("  - Using release dates from series config for masking")
    print("  - Generating 1 horizon forecast at each time point for each month")
    
    # Store results by time point
    results_by_timepoint = {}
    train_end_date = datetime.strptime(train_end, "%Y-%m-%d")
    
    # Get training data std for standardization
    from src.utils.config_parser import get_project_root
    data_path_file = get_project_root() / "data" / "data.csv"
    if not data_path_file.exists():
        data_path_file = get_project_root() / "data" / "sample_data.csv"
    train_data = pd.read_csv(data_path_file, index_col=0, parse_dates=True)
    train_start_ts = pd.Timestamp(train_start)
    train_end_ts = pd.Timestamp(train_end)
    train_data_filtered = train_data[(train_data.index >= train_start_ts) & (train_data.index <= train_end_ts)]
    if target_series in train_data_filtered.columns:
        train_std = train_data_filtered[target_series].std()
    else:
        train_std = train_data_filtered.iloc[:, 0].std() if len(train_data_filtered.columns) > 0 else 1.0
    
    for weeks in weeks_before:
        print(f"\n{'='*70}")
        print(f"Nowcasting at {weeks} weeks before month end")
        print(f"{'='*70}")
        
        # For each target month, calculate view_date and predict
        monthly_results = []
        
        for target_month_end in target_periods:
            # Calculate view_date: target_month_end - weeks
            view_date = target_month_end - timedelta(weeks=weeks)
            
            # Skip if view_date is before training period
            if view_date < train_end_date:
                continue
            
            print(f"\n  Target month: {target_month_end.strftime('%Y-%m')}")
            print(f"  View date: {view_date.strftime('%Y-%m-%d')} ({weeks} weeks before)")
            
            try:
                if supports_nowcast_manager:
                    # For DFM/DDFM, use nowcast manager with view_date
                    # This handles release-based masking automatically
                    nowcast_result = nowcast_manager(
                        target_series=target_series,
                        view_date=view_date,
                        target_period=target_month_end,
                        return_result=True
                    )
                    
                    forecast_value = nowcast_result.nowcast_value
                    
                    # Get actual value (if available in data)
                    actual_value = np.nan
                    if target_month_end in train_data.index:
                        if target_series in train_data.columns:
                            actual_value = train_data.loc[target_month_end, target_series]
                        else:
                            actual_value = train_data.loc[target_month_end].iloc[0]
                
                else:
                    # For ARIMA/VAR, use simulate_nowcasting_evaluation
                    # Load data up to view_date
                    data = pd.read_csv(data_path_file, index_col=0, parse_dates=True)
                    data = data[data.index <= pd.Timestamp(view_date)]
                    
                    # Use simulate_nowcasting_evaluation with release-based masking
                    # Note: This uses mask_days as approximation - full release-based masking
                    # would require config-based release dates
                    eval_result = simulate_nowcasting_evaluation(
                        forecaster=forecaster,
                        data=data,
                        target_dates=[pd.Timestamp(target_month_end)],
                        mask_days=30,  # Approximation - should use release dates from config
                        horizons=[1],
                        target_series=target_series
                    )
                    
                    # Extract forecast and actual
                    if eval_result['nowcast_metrics'] and len(eval_result['nowcast_metrics']) > 0:
                        metrics = eval_result['nowcast_metrics'][0].get('metrics', {})
                        horizon_1_metrics = metrics.get(1, {})
                        forecast_value = horizon_1_metrics.get('forecast', np.nan)
                        actual_value = horizon_1_metrics.get('actual', np.nan)
                    else:
                        forecast_value = np.nan
                        actual_value = np.nan
                
                # Skip if forecast or actual is NaN
                if np.isnan(forecast_value) or np.isnan(actual_value):
                    print(f"    ⚠ Skipping: forecast={forecast_value}, actual={actual_value}")
                    continue
                
                # Calculate error and standardized metrics
                error = forecast_value - actual_value
                if train_std and train_std > 0:
                    sMSE = (error ** 2) / (train_std ** 2)
                    sMAE = abs(error) / train_std
                else:
                    sMSE = error ** 2
                    sMAE = abs(error)
                
                monthly_results.append({
                    'month': target_month_end.strftime('%Y-%m'),
                    'view_date': view_date.strftime('%Y-%m-%d'),
                    'weeks_before': weeks,
                    'forecast_value': float(forecast_value),
                    'actual_value': float(actual_value),
                    'error': float(error),
                    'abs_error': float(abs(error)),
                    'squared_error': float(error ** 2),
                    'sMSE': float(sMSE),
                    'sMAE': float(sMAE)
                })
                
                print(f"    ✓ Forecast: {forecast_value:.4f}, Actual: {actual_value:.4f}, Error: {error:.4f}")
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Store results for this time point
        if len(monthly_results) > 0:
            results_by_timepoint[f"{weeks}weeks"] = {
                'weeks_before': weeks,
                'monthly_results': monthly_results,
                'overall_sMAE': float(np.nanmean([r['sMAE'] for r in monthly_results])),
                'overall_sMSE': float(np.nanmean([r['sMSE'] for r in monthly_results])),
                'overall_mae': float(np.nanmean([r['abs_error'] for r in monthly_results])),
                'overall_rmse': float(np.sqrt(np.nanmean([r['squared_error'] for r in monthly_results]))),
                'n_months': len(monthly_results)
            }
            print(f"\n  ✓ Completed: {len(monthly_results)} months evaluated")
            print(f"    Average sMAE: {results_by_timepoint[f'{weeks}weeks']['overall_sMAE']:.4f}")
            print(f"    Average sMSE: {results_by_timepoint[f'{weeks}weeks']['overall_sMSE']:.4f}")
        else:
            print(f"\n  ⚠ No valid results for {weeks} weeks before")
    
    # Aggregate results across all time points
    if len(results_by_timepoint) > 0:
        results = {
            'target_series': target_series,
            'model': model.upper(),
            'train_period': f"{train_start} to {train_end}",
            'nowcast_period': f"{nowcast_start} to {nowcast_end}",
            'weeks_before': weeks_before,
            'results_by_timepoint': results_by_timepoint,
            'horizon': 1
        }
    else:
        results = {
            'target_series': target_series,
            'model': model.upper(),
            'status': 'no_results',
            'error': 'No valid results generated for any time point'
        }
    
    # Save results
    output_dir = Path("outputs/backtest")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{target_series}_{model}_backtest.json"
    
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"✓ Backtest completed")
    print(f"  Results saved to: {output_file}")
    print(f"  Time points evaluated: {list(results_by_timepoint.keys())}")
    print(f"{'='*70}")
    
    return results


def main():
    """Main CLI entry point for inference/nowcasting."""
    parser = argparse.ArgumentParser(description="Run inference and nowcasting using Hydra config")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Nowcasting evaluation command
    nowcast_parser = subparsers.add_parser('nowcast', help='Run nowcasting evaluation (requires experiment config)')
    nowcast_parser.add_argument("--config-name", required=True, help="Experiment config name (e.g., experiment/investment_koequipte_report)")
    nowcast_parser.add_argument("--override", action="append", help="Hydra config override")
    nowcast_parser.add_argument("--mask-days", type=int, default=30, help="Number of days to mask for nowcasting simulation")
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest evaluation for all models (requires experiment config)')
    backtest_parser.add_argument("--config-name", required=True, help="Experiment config name (e.g., experiment/investment_koequipte_report)")
    backtest_parser.add_argument("--model", required=True, choices=['arima', 'var', 'dfm', 'ddfm'], help="Model to backtest")
    backtest_parser.add_argument("--train-start", default="1985-01-01", help="Training period start (YYYY-MM-DD)")
    backtest_parser.add_argument("--train-end", default="2019-12-31", help="Training period end (YYYY-MM-DD)")
    backtest_parser.add_argument("--nowcast-start", default="2024-01-01", help="Nowcasting period start (YYYY-MM-DD)")
    backtest_parser.add_argument("--nowcast-end", default="2024-12-31", help="Nowcasting period end (YYYY-MM-DD)")
    backtest_parser.add_argument("--weeks-before", nargs="+", type=int, help="Weeks before month end (e.g., --weeks-before 4 1). Default: [4, 1]")
    backtest_parser.add_argument("--override", action="append", help="Hydra config override")
    
    args = parser.parse_args()
    
    config_path = str(get_project_root() / "config")
    
    if args.command == 'nowcast':
        result = run_nowcasting_evaluation(
            config_name=args.config_name,
            config_dir=config_path,
            mask_days=args.mask_days,
            overrides=args.override
        )
        print(f"\n✓ Nowcasting evaluation completed")
    
    elif args.command == 'backtest':
        result = run_backtest_evaluation(
            config_name=args.config_name,
            model=args.model,
            train_start=args.train_start,
            train_end=args.train_end,
            nowcast_start=args.nowcast_start,
            nowcast_end=args.nowcast_end,
            weeks_before=args.weeks_before,
            config_dir=config_path,
            overrides=args.override
        )
        print(f"\n✓ Backtest completed")


if __name__ == "__main__":
    main()
