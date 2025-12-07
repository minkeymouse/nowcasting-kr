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
        # For DFM/DDFM, the forecaster wraps the underlying model
        # Extract from forecaster's _dfm_model or _ddfm_model attribute
        forecaster = model_data.get('forecaster')
        if forecaster is None:
            raise ValueError(f"Model file does not contain forecaster for {model.upper()} model")
        
        # Extract underlying DFM/DDFM model from forecaster
        if hasattr(forecaster, '_dfm_model'):
            dfm_model = forecaster._dfm_model
        elif hasattr(forecaster, '_ddfm_model'):
            dfm_model = forecaster._ddfm_model
        else:
            raise ValueError(f"Forecaster does not contain {model.upper()} model (missing _dfm_model or _ddfm_model attribute)")
        
        if dfm_model is None:
            raise ValueError(f"{model.upper()} model is None - model may not be trained")
        
        # Ensure model result is available (may be None after unpickling)
        # Try to recompute result if training_state is available
        if hasattr(dfm_model, '_model') and dfm_model._model is not None:
            underlying_model = dfm_model._model
            if hasattr(underlying_model, '_result') and underlying_model._result is None:
                # Try to recompute result if training_state exists
                if hasattr(underlying_model, 'training_state') and underlying_model.training_state is not None:
                    try:
                        # Recompute result from training state
                        underlying_model._result = underlying_model.get_result()
                    except (RuntimeError, AttributeError, Exception) as e:
                        # If recomputation fails, raise informative error
                        raise RuntimeError(
                            f"Cannot access nowcast manager: model result is None and cannot be recomputed. "
                            f"Error: {e}. Model may need to be retrained."
                        )
        
        # Ensure data_module is available for nowcast manager
        # Data module may not be preserved after pickling, so recreate if needed
        if not hasattr(dfm_model, '_data_module') or dfm_model._data_module is None:
            # Recreate data module from config and data file
            from src.utils.config_parser import get_project_root
            from src.model.sktime_forecaster import create_data_module_from_dataframe
            from dfm_python.lightning import DFMDataModule
            from src.preprocess.utils import create_transformer_from_config
            
            # Load training data
            data_path_file = get_project_root() / "data" / "data.csv"
            if not data_path_file.exists():
                data_path_file = get_project_root() / "data" / "sample_data.csv"
            
            train_data = pd.read_csv(data_path_file, index_col=0, parse_dates=True)
            train_start_ts = pd.Timestamp(train_start)
            train_end_ts = pd.Timestamp(train_end)
            train_data_filtered = train_data[(train_data.index >= train_start_ts) & (train_data.index <= train_end_ts)]
            
            # Recreate data module
            try:
                data_module = create_data_module_from_dataframe(
                    model=dfm_model,
                    data=train_data_filtered,
                    dfm_data_module=DFMDataModule,
                    create_transformer_func=create_transformer_from_config
                )
                dfm_model._data_module = data_module
            except (ImportError, ValueError, AttributeError, Exception) as e:
                raise RuntimeError(
                    f"Cannot recreate data module for nowcast manager: {e}. "
                    f"Model may need to be retrained with data_module stored."
                )
        
        # Get nowcast manager from underlying model
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
    # Load full data once (will be reused for actual value lookup in backtests)
    from src.utils.config_parser import get_project_root
    data_path_file = get_project_root() / "data" / "data.csv"
    if not data_path_file.exists():
        data_path_file = get_project_root() / "data" / "sample_data.csv"
    full_data = pd.read_csv(data_path_file, index_col=0, parse_dates=True)
    train_start_ts = pd.Timestamp(train_start)
    train_end_ts = pd.Timestamp(train_end)
    train_data_filtered = full_data[(full_data.index >= train_start_ts) & (full_data.index <= train_end_ts)]
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
            
            # Skip if view_date is at or before training period end
            # For nowcasting, we need view_date > train_end_date to have new data beyond training
            if view_date <= train_end_date:
                continue
            
            print(f"\n  Target month: {target_month_end.strftime('%Y-%m')}")
            print(f"  View date: {view_date.strftime('%Y-%m-%d')} ({weeks} weeks before)")
            
            try:
                if supports_nowcast_manager:
                    # For DFM/DDFM, use nowcast manager with view_date
                    # This handles release-based masking automatically
                    # CRITICAL: Update data_module with full data up to view_date for nowcasting
                    # The data_module was created with training data only (1985-2019), but nowcasting
                    # needs data up to view_date (2024) to have Time_view that includes 2024 dates
                    if hasattr(dfm_model, '_data_module') and dfm_model._data_module is not None:
                        # Update data_module with full data up to view_date
                        from src.model.sktime_forecaster import create_data_module_from_dataframe
                        from dfm_python.lightning import DFMDataModule
                        from src.preprocess.utils import create_transformer_from_config
                        
                        # Get data up to view_date (not just training period)
                        data_up_to_view = full_data[full_data.index <= pd.Timestamp(view_date)]
                        if len(data_up_to_view) > 0:
                            try:
                                updated_data_module = create_data_module_from_dataframe(
                                    model=dfm_model,
                                    data=data_up_to_view,
                                    dfm_data_module=DFMDataModule,
                                    create_transformer_func=create_transformer_from_config
                                )
                                dfm_model._data_module = updated_data_module
                                # Update nowcast manager's data_module reference
                                nowcast_manager.data_module = updated_data_module
                            except (ImportError, ValueError, AttributeError, Exception) as e:
                                print(f"    ⚠ Warning: Could not update data_module with data up to view_date: {e}")
                                # Continue with existing data_module (may fail, but try anyway)
                        else:
                            print(f"    ⚠ Skipping: No data available up to view_date {view_date}")
                            continue
                    
                    # Find closest available date at or before target_month_end for nowcast manager
                    # (data is weekly, so exact month-end dates may not exist)
                    target_period_for_nowcast = target_month_end
                    if target_month_end <= full_data.index.max():
                        available_dates = full_data.index[full_data.index <= target_month_end]
                        if len(available_dates) > 0:
                            # Use the closest date at or before target_month_end
                            target_period_for_nowcast = available_dates.max().to_pydatetime()
                        else:
                            print(f"    ⚠ Skipping: No data available at or before target_month_end {target_month_end}")
                            continue
                    else:
                        # target_month_end is after data range, use latest available
                        target_period_for_nowcast = full_data.index.max().to_pydatetime()
                    
                    try:
                        nowcast_result = nowcast_manager(
                            target_series=target_series,
                            view_date=view_date,
                            target_period=target_period_for_nowcast,
                            return_result=True
                        )
                        
                        if nowcast_result is None:
                            print(f"    ⚠ Skipping: Nowcast manager returned None")
                            continue
                        
                        forecast_value = nowcast_result.nowcast_value
                        
                        # Check if forecast is NaN or invalid
                        if np.isnan(forecast_value) or not np.isfinite(forecast_value):
                            print(f"    ⚠ Skipping: Nowcast value is NaN or invalid: {forecast_value}")
                            continue
                    except Exception as e:
                        print(f"    ✗ Error in nowcast manager: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                    
                    # Get actual value from full data (target months are in 2024, after training period)
                    # Use full_data loaded before loop (contains all dates including 2024)
                    actual_value = np.nan
                    if target_month_end <= full_data.index.max():
                        # Find closest date at or before target_month_end
                        available_dates = full_data.index[full_data.index <= target_month_end]
                        if len(available_dates) > 0:
                            closest_date = available_dates.max()
                            if target_series in full_data.columns:
                                actual_value = float(full_data.loc[closest_date, target_series])
                            else:
                                actual_value = float(full_data.loc[closest_date].iloc[0])
                    else:
                        # target_month_end is after data range
                        actual_value = np.nan
                
                else:
                    # For ARIMA/VAR, use direct prediction approach
                    # Use full_data loaded before loop (reuse for efficiency)
                    # Filter data up to view_date for training (nowcasting scenario)
                    train_data_filtered = full_data[full_data.index <= pd.Timestamp(view_date)]
                    
                    # Skip if no training data
                    if len(train_data_filtered) == 0:
                        print(f"    ⚠ Skipping: No training data available up to view_date")
                        continue
                    
                    # Calculate forecast horizon: days from view_date to target_month_end
                    horizon_days = (target_month_end - view_date).days
                    if horizon_days <= 0:
                        print(f"    ⚠ Skipping: view_date ({view_date}) is not before target_month_end ({target_month_end})")
                        continue
                    
                    # Fit forecaster on data up to view_date
                    try:
                        # Clone forecaster to avoid modifying the original
                        from sklearn.base import clone
                        try:
                            forecaster_clone = clone(forecaster)
                        except (TypeError, AttributeError, Exception):
                            # If clone fails, use original (some forecasters don't support cloning)
                            forecaster_clone = forecaster
                        
                        # Prepare data for fitting - use only target series for univariate models
                        # Check if forecaster expects univariate or multivariate data
                        # First check the forecaster's tags
                        is_multivariate = False
                        if hasattr(forecaster_clone, 'get_tag'):
                            try:
                                scitype_y = forecaster_clone.get_tag('scitype:y', 'univariate')
                                is_multivariate = (scitype_y == 'multivariate')
                            except (AttributeError, KeyError, Exception):
                                pass
                        
                        # Also check if forecaster was originally trained on multiple columns
                        # by checking the _y attribute (training data)
                        if not is_multivariate and hasattr(forecaster_clone, '_y'):
                            try:
                                if isinstance(forecaster_clone._y, pd.DataFrame):
                                    is_multivariate = len(forecaster_clone._y.columns) > 1
                            except (AttributeError, Exception):
                                pass
                        
                        # Fallback: assume multivariate if data has multiple columns
                        if not is_multivariate:
                            is_multivariate = len(train_data_filtered.columns) > 1
                        
                        if is_multivariate:
                            fit_data = train_data_filtered.copy()
                            # Drop rows that are all NaN
                            fit_data = fit_data.dropna(how='all')
                        else:
                            # Univariate: use only target series, but maintain as DataFrame
                            # Some forecasters expect DataFrame even for univariate
                            if target_series in train_data_filtered.columns:
                                fit_data = train_data_filtered[[target_series]].copy()
                            else:
                                # Use first column, but keep as DataFrame with proper column name
                                first_col = train_data_filtered.columns[0]
                                fit_data = train_data_filtered[[first_col]].copy()
                            
                            # For univariate, drop NaN values in target series
                            # Use forward-fill to handle missing values, then drop remaining NaNs
                            target_col_name = fit_data.columns[0]
                            if fit_data[target_col_name].isna().any():
                                # Forward-fill missing values
                                fit_data = fit_data.ffill()
                                # Drop any remaining NaNs (leading NaNs that couldn't be forward-filled)
                                fit_data = fit_data.dropna()
                            
                            # Ensure fit_data is a DataFrame (not Series) for consistency
                            if isinstance(fit_data, pd.Series):
                                fit_data = fit_data.to_frame()
                        
                        # Skip if insufficient data after cleaning
                        if len(fit_data) < 10:  # Minimum data points needed for ARIMA/VAR
                            print(f"    ⚠ Skipping: Insufficient data after cleaning ({len(fit_data)} rows, need at least 10)")
                            continue
                        
                        # Check if target series has any non-null values
                        if is_multivariate:
                            if target_series in fit_data.columns:
                                target_col = fit_data[target_series]
                            else:
                                target_col = fit_data.iloc[:, 0]
                        else:
                            target_col = fit_data.iloc[:, 0]
                        
                        if target_col.isna().all():
                            print(f"    ⚠ Skipping: Target series has no non-null values")
                            continue
                        
                        # Check if there's recent data (within last 90 days of view_date for monthly data)
                        # For monthly series, data might only be available at month end, so use 90 days (3 months)
                        # This ensures we have enough recent information for nowcasting
                        recent_cutoff = pd.Timestamp(view_date) - pd.Timedelta(days=90)
                        recent_data = fit_data[fit_data.index >= recent_cutoff]
                        if is_multivariate:
                            if target_series in recent_data.columns:
                                recent_target = recent_data[target_series]
                            else:
                                recent_target = recent_data.iloc[:, 0]
                        else:
                            recent_target = recent_data.iloc[:, 0]
                        
                        if recent_target.notna().sum() == 0:
                            print(f"    ⚠ Skipping: No recent data (last 90 days) available for prediction")
                            continue
                        
                        # Check if last non-null value is too far from view_date (more than 120 days for monthly data)
                        # For monthly series, allow up to 120 days (4 months) since last data point
                        last_valid_idx = target_col.last_valid_index()
                        if last_valid_idx is not None:
                            days_since_last = (pd.Timestamp(view_date) - pd.Timestamp(last_valid_idx)).days
                            if days_since_last > 120:
                                print(f"    ⚠ Skipping: Last valid data point is {days_since_last} days before view_date (too old, >120 days)")
                                continue
                        
                        # Fit on training data
                        try:
                            forecaster_clone.fit(fit_data)
                        except Exception as e:
                            print(f"    ✗ Error fitting forecaster: {e}")
                            import traceback
                            traceback.print_exc()
                            continue
                        
                        # Predict for target_month_end (horizon = horizon_days)
                        try:
                            fh = np.asarray([horizon_days])
                            y_pred = forecaster_clone.predict(fh=fh)
                        except Exception as e:
                            print(f"    ✗ Error in prediction: {e}")
                            import traceback
                            traceback.print_exc()
                            # Set forecast_value to NaN and continue to skip check below
                            forecast_value = np.nan
                            y_pred = None
                        
                        # Extract forecast value
                        if y_pred is None:
                            forecast_value = np.nan
                        elif isinstance(y_pred, pd.DataFrame):
                            if len(y_pred) == 0:
                                forecast_value = np.nan
                            elif target_series in y_pred.columns:
                                forecast_value = float(y_pred[target_series].iloc[0])
                            elif len(y_pred.columns) > 0:
                                # Use first column if target_series not found
                                # Handle both string and integer column indices
                                try:
                                    forecast_value = float(y_pred.iloc[0, 0])
                                except (KeyError, IndexError, TypeError) as e:
                                    # If iloc fails, try accessing by column name/index
                                    first_col = y_pred.columns[0]
                                    if isinstance(first_col, (int, np.integer)):
                                        forecast_value = float(y_pred.iloc[0, first_col])
                                    else:
                                        forecast_value = float(y_pred[first_col].iloc[0])
                            else:
                                forecast_value = np.nan
                        elif isinstance(y_pred, pd.Series):
                            if len(y_pred) > 0:
                                forecast_value = float(y_pred.iloc[0])
                            else:
                                forecast_value = np.nan
                        elif hasattr(y_pred, '__len__') and len(y_pred) > 0:
                            try:
                                forecast_value = float(y_pred[0] if hasattr(y_pred, '__getitem__') else y_pred)
                            except (TypeError, IndexError, KeyError):
                                forecast_value = np.nan
                        else:
                            try:
                                forecast_value = float(y_pred) if not np.isnan(y_pred) else np.nan
                            except (TypeError, ValueError):
                                forecast_value = np.nan
                        
                        # Check if forecast is NaN
                        if np.isnan(forecast_value):
                            print(f"    ⚠ Prediction returned NaN (horizon={horizon_days} days)")
                        
                        # Get actual value from full data
                        # Handle cases where target_month_end might not be exactly in index
                        if target_month_end <= full_data.index.max():
                            # Find closest date at or before target_month_end
                            available_dates = full_data.index[full_data.index <= target_month_end]
                            if len(available_dates) > 0:
                                closest_date = available_dates.max()
                                if target_series in full_data.columns:
                                    actual_value = float(full_data.loc[closest_date, target_series])
                                else:
                                    actual_value = float(full_data.loc[closest_date].iloc[0])
                            else:
                                actual_value = np.nan
                        else:
                            # target_month_end is after data range
                            actual_value = np.nan
                        
                    except Exception as e:
                        print(f"    ✗ Error in prediction: {e}")
                        import traceback
                        traceback.print_exc()
                        forecast_value = np.nan
                        # Try to get actual value even if prediction failed
                        if target_month_end <= full_data.index.max():
                            available_dates = full_data.index[full_data.index <= target_month_end]
                            if len(available_dates) > 0:
                                closest_date = available_dates.max()
                                if target_series in full_data.columns:
                                    actual_value = float(full_data.loc[closest_date, target_series])
                                else:
                                    actual_value = float(full_data.loc[closest_date].iloc[0])
                            else:
                                actual_value = np.nan
                        else:
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
