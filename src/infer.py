"""Nowcasting module - supports both CLI and programmatic API.

This module provides NOWCASTING functionality only (not forecasting).
Forecasting is handled separately via train.py compare (see run_forecast.sh).

NOWCAST vs FORECAST:
- NOWCAST: Estimate current period using incomplete data (publication lag considered)
- FORECAST: Predict future periods using complete historical data

This module can be used:
- As CLI: python src/infer.py backtest --config-name experiment/investment_koequipte_report --model dfm
- As API: from src.infer import run_backtest_evaluation

Functions exported for programmatic use:
- run_backtest_evaluation: Run nowcasting backtest for all models with multiple time points
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
# Minimal path setup to allow importing setup_cli_environment
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.resolve()

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# Now we can import from src
from src.utils import setup_cli_environment
setup_cli_environment()  # This sets up all paths properly

# Now use absolute imports
from src.utils import (
    get_project_root,
    parse_experiment_config,
    extract_experiment_params,
    validate_experiment_config
)

# Removed unused functions: mask_recent_observations, create_nowcasting_splits, simulate_nowcasting_evaluation, run_nowcasting_evaluation
# These were only used by the 'nowcast' CLI command which is not used in the actual pipeline (run_backtest.sh uses 'backtest' command)

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
    # ARIMA/VAR use direct prediction with release-based masking
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
    from src.models import DFM, DDFM
    import pickle
    
    # Find trained model - first check checkpoint/, then outputs/comparisons/
    # Use absolute paths to avoid issues when script is run from different directory
    project_root = get_project_root()
    model_dir = None
    checkpoint_path = project_root / "checkpoint" / f"{target_series}_{model.lower()}" / "model.pkl"
    if checkpoint_path.exists():
        model_dir = checkpoint_path.parent
    else:
        # Fallback to outputs/comparisons/
        comparisons_dir = project_root / "outputs" / "comparisons"
        if comparisons_dir.exists():
            for comparison_dir in comparisons_dir.glob(f"{target_series}_*"):
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
                # Try multiple methods to restore result
                result_restored = False
                
                # Method 1: Try to recompute result from training_state
                if hasattr(underlying_model, 'training_state') and underlying_model.training_state is not None:
                    try:
                        # Recompute result from training state
                        underlying_model._result = underlying_model.get_result()
                        if underlying_model._result is not None:
                            result_restored = True
                            print(f"    ✓ Restored model result from training_state")
                    except (RuntimeError, AttributeError, Exception) as e:
                        print(f"    ⚠ Failed to restore result from training_state: {e}")
                
                # Method 2: Try using wrapper's get_result() method
                if not result_restored:
                    try:
                        result = dfm_model.get_result()
                        if result is not None:
                            underlying_model._result = result
                            result_restored = True
                            print(f"    ✓ Restored model result from wrapper get_result()")
                    except (RuntimeError, AttributeError, Exception) as e:
                        print(f"    ⚠ Failed to restore result from wrapper: {e}")
                
                # Method 3: Check if result is stored in model_data
                if not result_restored and 'result' in model_data:
                    try:
                        stored_result = model_data['result']
                        if stored_result is not None:
                            underlying_model._result = stored_result
                            result_restored = True
                            print(f"    ✓ Restored model result from saved model_data")
                    except (RuntimeError, AttributeError, Exception) as e:
                        print(f"    ⚠ Failed to restore result from model_data: {e}")
                
                # Final check: if result is still None, raise error
                if not result_restored and (not hasattr(underlying_model, '_result') or underlying_model._result is None):
                    raise RuntimeError(
                        f"Cannot access nowcast manager: model result is None and all restoration attempts failed. "
                        f"Model may need to be retrained. Check if model was properly saved with result data."
                    )
        
        # Ensure data_module is available for nowcast manager
        # Data module may not be preserved after pickling, so recreate if needed
        if not hasattr(dfm_model, '_data_module') or dfm_model._data_module is None:
            # Recreate data module from config and data file
            from src.utils import get_project_root
            from src.models import create_data_module_from_dataframe
            from dfm_python.lightning import DFMDataModule
            from src.preprocessing import create_transformer_from_config
            
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
        
        # Final verification: check if result is available before accessing nowcast
        if hasattr(dfm_model, '_model') and dfm_model._model is not None:
            underlying_model = dfm_model._model
            if hasattr(underlying_model, '_result') and underlying_model._result is None:
                raise RuntimeError(
                    f"Cannot access nowcast manager: model result is None. "
                    f"This indicates the model was not properly trained or the result was not saved. "
                    f"Please retrain the model."
                )
        
        # Get nowcast manager from underlying model
        try:
            nowcast_manager = dfm_model.nowcast
        except RuntimeError as e:
            if "Model must be trained" in str(e):
                raise RuntimeError(
                    f"Cannot access nowcast manager: {e}. "
                    f"Model result restoration failed. Please retrain the model."
                ) from e
            raise
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
    from src.utils import get_project_root
    from src.preprocessing import resample_to_monthly
    data_path_file = get_project_root() / "data" / "data.csv"
    if not data_path_file.exists():
        data_path_file = get_project_root() / "data" / "sample_data.csv"
    full_data = pd.read_csv(data_path_file, index_col=0, parse_dates=True)
    print(f"\nData loaded: {len(full_data)} rows")
    print(f"  Date range: {full_data.index.min()} to {full_data.index.max()}")
    print(f"  Columns: {list(full_data.columns)}")
    
    # CRITICAL FIX: Aggregate to monthly for actual value lookup (targets are monthly)
    # Weekly data needs to be aggregated to monthly to match target_month_end dates
    full_data_monthly = resample_to_monthly(full_data)
    print(f"Monthly data: {len(full_data_monthly)} rows")
    print(f"  Date range: {full_data_monthly.index.min()} to {full_data_monthly.index.max()}")
    
    # Check if data extends to nowcasting period
    nowcast_start_ts = pd.Timestamp(nowcast_start)
    nowcast_end_ts = pd.Timestamp(nowcast_end)
    if full_data_monthly.index.max() < nowcast_end_ts:
        print(f"\n⚠ WARNING: Data only extends to {full_data_monthly.index.max()}, but nowcasting period is {nowcast_start} to {nowcast_end}")
        print(f"  Some target months may not have actual values available")
    else:
        print(f"✓ Data extends to {full_data_monthly.index.max()}, covering nowcasting period {nowcast_start} to {nowcast_end}")
    
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
            # CRITICAL FIX: Convert target_month_end to pd.Timestamp at the start for consistent comparisons
            target_month_end_ts = pd.Timestamp(target_month_end) if not isinstance(target_month_end, pd.Timestamp) else target_month_end
            
            # Calculate view_date: target_month_end - weeks
            view_date = target_month_end_ts - timedelta(weeks=weeks)
            
            # Skip if view_date is at or before training period end
            # For nowcasting, we need view_date > train_end_date to have new data beyond training
            # However, for early months in 2024 with 4 weeks before, view_date might be in late 2023
            # which is still after train_end (2019-12-31), so this should be fine
            # CRITICAL FIX: Use date comparison that handles timezone-aware dates correctly
            view_date_ts = pd.Timestamp(view_date) if not isinstance(view_date, pd.Timestamp) else view_date
            train_end_ts = pd.Timestamp(train_end_date) if not isinstance(train_end_date, pd.Timestamp) else train_end_date
            if view_date_ts <= train_end_ts:
                print(f"    ⚠ Skipping: view_date ({view_date.strftime('%Y-%m-%d')}) <= train_end_date ({train_end_date.strftime('%Y-%m-%d')})")
                print(f"      This means we're trying to nowcast using data from training period, which is not valid for nowcasting")
                continue
            
            # Also check if target_month_end has actual value available
            # For nowcasting, we need actual values to compare against
            # CRITICAL FIX: This check should not skip if data exists before target_month_end
            # The actual value lookup code (lines 947-1010) handles finding the closest available date
            # So we only skip if there's NO data at all before target_month_end
            if len(full_data_monthly) > 0:
                available_before = full_data_monthly.index[full_data_monthly.index <= target_month_end_ts]
                if len(available_before) == 0:
                    print(f"    ⚠ Skipping: No monthly data available at or before target month {target_month_end_ts.strftime('%Y-%m')}")
                    print(f"      Data range: {full_data_monthly.index.min()} to {full_data_monthly.index.max()}")
                    continue
                # Check if exact month match exists (for diagnostic purposes only)
                available_for_target = full_data_monthly.index[
                    (full_data_monthly.index.year == target_month_end_ts.year) &
                    (full_data_monthly.index.month == target_month_end_ts.month)
                ]
                if len(available_for_target) == 0:
                    print(f"    ℹ Note: No exact month match for {target_month_end_ts.strftime('%Y-%m')}, will use closest available date")
            else:
                print(f"    ⚠ Skipping: full_data_monthly is empty - cannot get actual values")
                continue
            
            print(f"\n  Target month: {target_month_end_ts.strftime('%Y-%m')}")
            print(f"  View date: {view_date.strftime('%Y-%m-%d')} ({weeks} weeks before)")
            
            try:
                if supports_nowcast_manager:
                    # For DFM/DDFM, use nowcast manager with view_date
                    # This handles release-based masking automatically
                    # CRITICAL: Update data_module with full data up to view_date for nowcasting
                    # The data_module was created with training data only (1985-2019), but nowcasting
                    # needs data up to view_date (2024) to have Time_view that includes 2024 dates
                    if hasattr(dfm_model, '_data_module') and dfm_model._data_module is not None:
                        # Update data_module with full data up to target_month_end (not just view_date)
                        # CRITICAL: The nowcast manager needs TimeIndex to include target_period (target_month_end),
                        # not just view_date. We use view_date for masking (release dates), but TimeIndex must
                        # include target_period for the nowcast manager to find the target period.
                        from src.models import create_data_module_from_dataframe
                        from dfm_python.lightning import DFMDataModule
                        from src.preprocessing import create_transformer_from_config
                        
                        # Get data up to target_month_end (not just view_date) so TimeIndex includes target_period
                        # The nowcast manager uses view_date for masking but needs target_period in TimeIndex
                        # CRITICAL: Resample to monthly first so TimeIndex has monthly dates that match target periods
                        from src.preprocessing import resample_to_monthly
                        data_up_to_target = full_data[full_data.index <= target_month_end_ts]
                        # Resample to monthly to ensure TimeIndex has monthly dates
                        data_up_to_target_monthly = resample_to_monthly(data_up_to_target)
                        print(f"    Debug: Filtered data up to {target_month_end_ts}: {len(data_up_to_target)} rows (weekly), {len(data_up_to_target_monthly)} rows (monthly)")
                        print(f"    Debug: Monthly data range: {data_up_to_target_monthly.index.min()} to {data_up_to_target_monthly.index.max()}")
                        if len(data_up_to_target_monthly) > 0:
                            try:
                                print(f"    Debug: Updating data_module with monthly data up to {target_month_end_ts}...")
                                updated_data_module = create_data_module_from_dataframe(
                                    model=dfm_model,
                                    data=data_up_to_target_monthly,
                                    dfm_data_module=DFMDataModule,
                                    create_transformer_func=create_transformer_from_config
                                )
                                dfm_model._data_module = updated_data_module
                                # Update nowcast manager's data_module reference
                                nowcast_manager.data_module = updated_data_module
                                
                                # CRITICAL FIX: Validate that data_module was updated successfully
                                # Check if time_index has data after update (nowcast manager uses data_module.time_index)
                                if hasattr(updated_data_module, 'time_index') and updated_data_module.time_index is not None:
                                    time_index = updated_data_module.time_index
                                    # TimeIndex has 'dates' attribute or can be iterated
                                    if hasattr(time_index, 'dates') and len(time_index.dates) > 0:
                                        first_date = time_index.dates[0]
                                        last_date = time_index.dates[-1]
                                        print(f"    ✓ Data module updated: {len(time_index.dates)} time points (from {first_date} to {last_date})")
                                        # Check if target_period is in the time_index
                                        target_in_index = any(
                                            (isinstance(t, datetime) and t.year == target_month_end_ts.year and t.month == target_month_end_ts.month) or
                                            (hasattr(t, 'year') and hasattr(t, 'month') and t.year == target_month_end_ts.year and t.month == target_month_end_ts.month)
                                            for t in time_index.dates
                                        )
                                        if not target_in_index:
                                            print(f"    ⚠ Warning: Target period {target_month_end_ts} (year={target_month_end_ts.year}, month={target_month_end_ts.month}) not found in TimeIndex dates")
                                    elif hasattr(time_index, '__len__') and len(time_index) > 0:
                                        print(f"    ✓ Data module updated: {len(time_index)} time points")
                                        # Try to check if target_period is in time_index
                                        try:
                                            time_list = list(time_index)
                                            target_in_index = any(
                                                (isinstance(t, datetime) and t.year == target_month_end_ts.year and t.month == target_month_end_ts.month) or
                                                (hasattr(t, 'year') and hasattr(t, 'month') and t.year == target_month_end_ts.year and t.month == target_month_end_ts.month)
                                                for t in time_list
                                            )
                                            if not target_in_index:
                                                print(f"    ⚠ Warning: Target period {target_month_end_ts} (year={target_month_end_ts.year}, month={target_month_end_ts.month}) not found in TimeIndex")
                                        except Exception as e:
                                            print(f"    ⚠ Warning: Could not check if target_period is in TimeIndex: {e}")
                                    else:
                                        print(f"    ⚠ Warning: Data module updated but time_index appears empty")
                                else:
                                    print(f"    ⚠ Warning: Data module updated but time_index attribute not available")
                            except (ImportError, ValueError, AttributeError, Exception) as e:
                                print(f"    ✗ Error: Could not update data_module with data up to target_month_end: {e}")
                                import traceback
                                traceback.print_exc()
                                print(f"    ⚠ Skipping: Data module update failed - cannot perform nowcast without updated data")
                                continue  # Skip this month if data_module update fails
                        else:
                                print(f"    ⚠ Skipping: No data available up to target_month_end {target_month_end_ts}")
                            continue
                    
                    # Find closest available date at or before target_month_end for nowcast manager
                    # Use monthly aggregated data for monthly targets, but also check TimeIndex
                    target_period_for_nowcast = target_month_end_ts
                    if target_month_end_ts <= full_data_monthly.index.max():
                        available_dates = full_data_monthly.index[full_data_monthly.index <= target_month_end_ts]
                        if len(available_dates) > 0:
                            # Use the closest date at or before target_month_end (monthly data)
                            target_period_for_nowcast = available_dates.max().to_pydatetime()
                        else:
                            print(f"    ⚠ Skipping: No monthly data available at or before target_month_end {target_month_end_ts}")
                            continue
                    else:
                        # target_month_end is after data range, use latest available
                        target_period_for_nowcast = full_data_monthly.index.max().to_pydatetime()
                    
                    # CRITICAL: Also check if target_period_for_nowcast exists in data_module.time_index
                    # If not, find the closest date in the same month
                    if hasattr(nowcast_manager, 'data_module') and nowcast_manager.data_module is not None:
                        data_module = nowcast_manager.data_module
                        if hasattr(data_module, 'time_index') and data_module.time_index is not None:
                            time_index = data_module.time_index
                            # Try to find a date in the same month as target_period_for_nowcast
                            from dfm_python.utils.time import find_time_index
                            t_idx = find_time_index(time_index, target_period_for_nowcast)
                            if t_idx is None:
                                # Find closest date in the same month
                                target_year = target_period_for_nowcast.year
                                target_month = target_period_for_nowcast.month
                                closest_date = None
                                min_diff = None
                                
                                # Get dates from TimeIndex
                                if hasattr(time_index, 'dates'):
                                    time_dates = time_index.dates
                                elif hasattr(time_index, '__iter__'):
                                    time_dates = list(time_index)
                                else:
                                    time_dates = []
                                
                                for t in time_dates:
                                    if not isinstance(t, datetime):
                                        try:
                                            if isinstance(t, pd.Timestamp):
                                                t = t.to_pydatetime()
                                            elif hasattr(t, 'to_pydatetime'):
                                                t = t.to_pydatetime()
                                            else:
                                                continue
                                        except (ValueError, TypeError):
                                            continue
                                    
                                    if isinstance(t, datetime) and t.year == target_year and t.month == target_month:
                                        diff = abs((target_period_for_nowcast - t).total_seconds())
                                        if min_diff is None or diff < min_diff:
                                            min_diff = diff
                                            closest_date = t
                                
                                if closest_date is not None:
                                    print(f"    Debug: Using closest date in TimeIndex: {closest_date} (target was {target_period_for_nowcast})")
                                    target_period_for_nowcast = closest_date
                                else:
                                    # CRITICAL FIX: If no date found in same month, try to find closest date at or before target_period
                                    # This handles cases where TimeIndex has different dates than expected
                                    closest_before = None
                                    min_diff_before = None
                                    for t in time_dates:
                                        if not isinstance(t, datetime):
                                            try:
                                                if isinstance(t, pd.Timestamp):
                                                    t = t.to_pydatetime()
                                                elif hasattr(t, 'to_pydatetime'):
                                                    t = t.to_pydatetime()
                                                else:
                                                    continue
                                            except (ValueError, TypeError):
                                                continue
                                        
                                        if isinstance(t, datetime) and t <= target_period_for_nowcast:
                                            diff = abs((target_period_for_nowcast - t).total_seconds())
                                            if min_diff_before is None or diff < min_diff_before:
                                                min_diff_before = diff
                                                closest_before = t
                                    
                                    if closest_before is not None:
                                        print(f"    Debug: Using closest date at or before target: {closest_before} (target was {target_period_for_nowcast})")
                                        target_period_for_nowcast = closest_before
                                    else:
                                        print(f"    ⚠ Skipping: No date found in TimeIndex for month {target_year}-{target_month:02d} or before")
                                        continue  # Skip this month if no valid date found
                    
                    try:
                        # CRITICAL FIX: Check if data_module has valid time_index before calling nowcast manager
                        # The nowcast manager uses data_module.time_index (lowercase), not data_module.Time
                        if hasattr(nowcast_manager, 'data_module') and nowcast_manager.data_module is not None:
                            data_module = nowcast_manager.data_module
                            if hasattr(data_module, 'time_index') and data_module.time_index is not None:
                                time_index = data_module.time_index
                                # TimeIndex has 'dates' attribute or can be iterated
                                if hasattr(time_index, 'dates'):
                                    time_index_len = len(time_index.dates) if time_index.dates else 0
                                    if time_index_len > 0:
                                        print(f"    Debug: data_module.time_index has {time_index_len} dates (from {time_index.dates[0]} to {time_index.dates[-1]})")
                                elif hasattr(time_index, '__len__'):
                                    time_index_len = len(time_index)
                                    if time_index_len > 0:
                                        print(f"    Debug: data_module.time_index has {time_index_len} dates")
                                else:
                                    time_index_len = 0
                                
                                if time_index_len == 0:
                                    print(f"    ⚠ Skipping: Data module time_index is empty - cannot perform nowcast")
                                    continue
                            else:
                                print(f"    ⚠ Warning: data_module.time_index is None or not available")
                        else:
                            print(f"    ⚠ Warning: nowcast_manager.data_module is None or not available")
                        
                        # Final verification: Check if target_period_for_nowcast exists in time_index
                        # Use the same logic that nowcast manager uses to avoid "not found in Time index" errors
                        target_found = False
                        if hasattr(nowcast_manager, 'data_module') and nowcast_manager.data_module is not None:
                            data_module = nowcast_manager.data_module
                            if hasattr(data_module, 'time_index') and data_module.time_index is not None:
                                time_index = data_module.time_index
                                # Try to find target_period in time_index using find_time_index (same as nowcast manager)
                                try:
                                    from dfm_python.utils.time import find_time_index
                                    t_idx = find_time_index(time_index, target_period_for_nowcast)
                                    if t_idx is not None:
                                        target_found = True
                                        print(f"    ✓ Verified: target_period {target_period_for_nowcast} found in time_index at index {t_idx}")
                                    else:
                                        # Try to find any date in the same month
                                        target_year = target_period_for_nowcast.year
                                        target_month = target_period_for_nowcast.month
                                        if hasattr(time_index, 'dates'):
                                            time_dates = time_index.dates
                                        elif hasattr(time_index, '__iter__'):
                                            time_dates = list(time_index)
                                        else:
                                            time_dates = []
                                        
                                        for t in time_dates:
                                            if not isinstance(t, datetime):
                                                try:
                                                    if isinstance(t, pd.Timestamp):
                                                        t = t.to_pydatetime()
                                                    elif hasattr(t, 'to_pydatetime'):
                                                        t = t.to_pydatetime()
                                                except (ValueError, TypeError):
                                                    continue
                                            
                                            if isinstance(t, datetime) and t.year == target_year and t.month == target_month:
                                                target_found = True
                                                print(f"    ✓ Verified: Found date {t} in same month as target {target_period_for_nowcast}")
                                                target_period_for_nowcast = t  # Update to use the found date
                                                break
                                        
                                        if not target_found:
                                            print(f"    ⚠ Skipping: target_period {target_period_for_nowcast} not found in time_index and no date in same month")
                                            continue
                                except (ImportError, AttributeError, Exception) as e:
                                    print(f"    ⚠ Warning: Could not verify target_period in time_index: {e}")
                                    # Don't continue - this is a critical check, skip if verification fails
                                    print(f"    ⚠ Skipping: Target period verification failed")
                                    continue
                        else:
                            # If data_module or time_index is not available, cannot verify target_period
                            print(f"    ⚠ Skipping: Cannot verify target_period - data_module or time_index not available")
                            continue
                        
                        # CRITICAL: Only proceed if target_found is True
                        if not target_found:
                            print(f"    ⚠ Skipping: Target period verification failed - target_period not found in time_index")
                            continue
                        
                        # Clear the data view cache to force recalculation with updated data_module
                        if hasattr(nowcast_manager, '_data_view_cache'):
                            nowcast_manager._data_view_cache.clear()
                            print(f"    Debug: Cleared data view cache to force recalculation")
                        
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
                        else:
                            print(f"    ✓ DFM/DDFM nowcast value: {forecast_value:.4f}")
                    except ValueError as e:
                        # Handle ValueError from _prepare_target when Time_view is empty
                        if "Time_view is empty" in str(e) or "not found in Time index" in str(e):
                            print(f"    ⚠ Skipping: {e}")
                            continue
                        else:
                            print(f"    ✗ Error in nowcast manager: {e}")
                            import traceback
                            traceback.print_exc()
                            continue
                    except Exception as e:
                        print(f"    ✗ Error in nowcast manager: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                    
                    # Get actual value from full data (target months are in 2024, after training period)
                    # Use full_data loaded before loop (contains all dates including 2024)
                    actual_value = np.nan
                    if target_month_end_ts <= full_data_monthly.index.max():
                        # Find closest date at or before target_month_end (use monthly data)
                        available_dates = full_data_monthly.index[full_data_monthly.index <= target_month_end_ts]
                        if len(available_dates) > 0:
                            closest_date = available_dates.max()
                            if target_series in full_data_monthly.columns:
                                raw_value = full_data_monthly.loc[closest_date, target_series]
                                # Check if value is NaN in data
                                if pd.isna(raw_value):
                                    print(f"    ⚠ Warning: Actual value at {closest_date} is NaN in monthly data")
                                    actual_value = np.nan
                                else:
                                    actual_value = float(raw_value)
                            else:
                                raw_value = full_data_monthly.loc[closest_date].iloc[0]
                                if pd.isna(raw_value):
                                    print(f"    ⚠ Warning: Actual value at {closest_date} is NaN in monthly data")
                                    actual_value = np.nan
                                else:
                                    actual_value = float(raw_value)
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
                        print(f"    ⚠ Skipping: No training data available up to view_date {view_date}")
                        print(f"      Data range: {full_data.index.min()} to {full_data.index.max()}")
                        continue
                    else:
                        print(f"    ✓ Training data available: {len(train_data_filtered)} rows (from {train_data_filtered.index.min()} to {train_data_filtered.index.max()})")
                    
                    # CRITICAL FIX: Resample to monthly to match model training frequency
                    # Models are trained on monthly data (resample_to_monthly), so backtest must use monthly data too
                    from src.preprocessing import resample_to_monthly
                    original_count = len(train_data_filtered)
                    train_data_filtered = resample_to_monthly(train_data_filtered)
                    # Check if resampling removed all data
                    if len(train_data_filtered) == 0:
                        print(f"    ⚠ Skipping: Monthly resampling removed all data (original had {original_count} rows)")
                        continue
                    print(f"    ✓ Resampled to monthly: {len(train_data_filtered)} rows (from {train_data_filtered.index.min()} to {train_data_filtered.index.max()})")
                    
                    # Calculate nowcast horizon: days from view_date to target_month_end
                    # For nowcasting, we predict the current period (target_month_end) using data available at view_date
                    # CRITICAL FIX: Use target_month_end_ts for type consistency (pd.Timestamp)
                    horizon_days = (target_month_end_ts - view_date).days
                    if horizon_days <= 0:
                        print(f"    ⚠ Skipping: view_date ({view_date}) is not before target_month_end ({target_month_end_ts}), horizon={horizon_days} days")
                        continue
                    else:
                        print(f"    ✓ Horizon: {horizon_days} days (from {view_date} to {target_month_end_ts})")
                    
                    # Fit forecaster on data up to view_date (nowcasting: incomplete data scenario)
                    try:
                        # Clone forecaster to avoid modifying the original
                        from sklearn.base import clone
                        try:
                            forecaster_clone = clone(forecaster)
                            # CRITICAL FIX: For VAR models, reset internal state to allow refitting with new data structure
                            # VAR's internal transformation pipeline may have been fitted with different column structure
                            # Reset the forecaster's internal state to allow refitting
                            if hasattr(forecaster_clone, 'reset'):
                                try:
                                    forecaster_clone.reset()
                                except (AttributeError, Exception):
                                    pass
                            # Also reset internal _y and _X attributes if they exist
                            if hasattr(forecaster_clone, '_y'):
                                forecaster_clone._y = None
                            if hasattr(forecaster_clone, '_X'):
                                forecaster_clone._X = None
                            # Reset internal fitted state
                            if hasattr(forecaster_clone, '_is_fitted'):
                                forecaster_clone._is_fitted = False
                            # CRITICAL FIX: For VAR models, also reset any internal transformers or pipeline state
                            # VAR models may have internal transformers that need to be reset for column structure changes
                            if hasattr(forecaster_clone, 'forecaster_'):
                                # Check if forecaster_ has transformers that need resetting
                                forecaster_internal = forecaster_clone.forecaster_
                                if hasattr(forecaster_internal, 'transformers_'):
                                    # Reset transformers if they exist
                                    try:
                                        forecaster_internal.transformers_ = None
                                    except (AttributeError, Exception):
                                        pass
                                # Reset fitted state of internal forecaster
                                if hasattr(forecaster_internal, '_is_fitted'):
                                    forecaster_internal._is_fitted = False
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
                        training_columns = None
                        if hasattr(forecaster_clone, '_y'):
                            try:
                                if isinstance(forecaster_clone._y, pd.DataFrame):
                                    is_multivariate = len(forecaster_clone._y.columns) > 1
                                    training_columns = forecaster_clone._y.columns.tolist()
                            except (AttributeError, Exception):
                                pass
                        
                        # Fallback: assume multivariate if data has multiple columns
                        if not is_multivariate:
                            is_multivariate = len(train_data_filtered.columns) > 1
                        
                        if is_multivariate:
                            # CRITICAL FIX: For VAR, ensure we use the same columns as training data
                            # VAR's transformation pipeline expects the same column structure
                            # The pipeline may have been fitted with integer column indices, so we need to
                            # ensure the DataFrame columns match exactly (same order and same names)
                            if training_columns is not None:
                                # Use only columns that exist in both training and backtest data
                                available_columns = [col for col in training_columns if col in train_data_filtered.columns]
                                if len(available_columns) < len(training_columns):
                                    print(f"    ⚠ Warning: Some training columns missing in backtest data. Training: {training_columns}, Available: {available_columns}")
                                if len(available_columns) < 2:
                                    print(f"    ⚠ Skipping: VAR requires at least 2 columns, but only {len(available_columns)} available after column matching")
                                    continue
                                # CRITICAL: Maintain exact column order from training to match pipeline expectations
                                # The pipeline may use integer indices internally, so column order matters
                                fit_data = train_data_filtered[available_columns].copy()
                                # Ensure columns are in the exact same order as training_columns
                                # This is critical because VAR's internal ColumnEnsembleTransformer may use integer indices
                                fit_data = fit_data.reindex(columns=available_columns)
                                
                                # CRITICAL FIX: VAR's internal transformation pipeline may have been fitted with integer column indices
                                # If the pipeline uses integer indices, we need to ensure the DataFrame structure matches
                                # Check if forecaster has a transformation pipeline and if it uses integer indices
                                # If so, we may need to reset the pipeline or ensure column structure matches
                                # For now, ensure columns are in the exact same order as training to match pipeline expectations
                            else:
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
                            print(f"      Original train_data_filtered: {len(train_data_filtered)} rows")
                            continue
                        else:
                            print(f"    ✓ Data after cleaning: {len(fit_data)} rows")
                        
                        # Check if target series has any non-null values
                        if is_multivariate:
                            if target_series in fit_data.columns:
                                target_col = fit_data[target_series]
                            else:
                                target_col = fit_data.iloc[:, 0]
                        else:
                            target_col = fit_data.iloc[:, 0]
                        
                        non_null_count = target_col.notna().sum()
                        if target_col.isna().all():
                            print(f"    ⚠ Skipping: Target series has no non-null values (0/{len(target_col)} non-null)")
                            continue
                        else:
                            print(f"    ✓ Target series has {non_null_count}/{len(target_col)} non-null values")
                        
                        # Check if there's recent data (within last 6 months for monthly data)
                        # For monthly data, use 6 months to ensure we have enough recent information for nowcasting
                        # However, for nowcasting scenarios where view_date is early in a month, we need to be more lenient
                        # Use 730 days (24 months) to account for monthly data gaps and early-month view dates
                        # This is more lenient to avoid skipping valid months due to data gaps
                        recent_cutoff = pd.Timestamp(view_date) - pd.Timedelta(days=730)
                        # Use train_data_filtered (now monthly) for recent data check
                        recent_data = train_data_filtered[train_data_filtered.index >= recent_cutoff]
                        if is_multivariate:
                            if target_series in recent_data.columns:
                                recent_target = recent_data[target_series]
                            else:
                                recent_target = recent_data.iloc[:, 0]
                        else:
                            if target_series in recent_data.columns:
                                recent_target = recent_data[target_series]
                            else:
                                recent_target = recent_data.iloc[:, 0]
                        
                        recent_count = recent_target.notna().sum()
                        # For monthly data, require at least 1 recent data point (within last 24 months)
                        # This is more lenient than requiring data within 365 days, which might be too strict
                        # for monthly data when view_date is early in a month or when there are data gaps
                        if recent_count == 0:
                            print(f"    ⚠ Skipping: No recent data (last 730 days) available for prediction (checked {len(recent_data)} rows)")
                            continue
                        else:
                            print(f"    ✓ Found {recent_count} recent data points (last 730 days)")
                        
                        # Check if last non-null value is too far from view_date
                        # For monthly data, the last valid index should be within reasonable range
                        # Since we're resampling to monthly, the last index will be a month-end date
                        # For nowcasting, we're predicting the current month using data from previous months
                        # Use fit_data (after cleaning) to get last valid index, as it represents the actual data used for fitting
                        if is_multivariate:
                            if target_series in fit_data.columns:
                                target_col_for_check = fit_data[target_series]
                            else:
                                target_col_for_check = fit_data.iloc[:, 0]
                        else:
                            target_col_for_check = fit_data.iloc[:, 0]
                        last_valid_idx = target_col_for_check.last_valid_index()
                        if last_valid_idx is not None:
                            days_since_last = (pd.Timestamp(view_date) - pd.Timestamp(last_valid_idx)).days
                            # For monthly data, allow up to 180 days (6 months) since last data point
                            # This accounts for the fact that monthly data naturally has gaps
                            # and for nowcasting in early January, December data is still recent
                            # CRITICAL FIX: For monthly data, if view_date is early in a month (e.g., Jan 3),
                            # the last monthly data point might be from the previous month (Dec 31), which is only
                            # a few days ago. This is acceptable for nowcasting. However, if the gap is > 180 days,
                            # it means we're missing recent data, which is a problem.
                            # Also, for monthly data resampled from weekly, the last valid index should be
                            # at or very close to view_date (within the same month or previous month).
                            # Increased from 90 to 180 days to be more lenient and avoid skipping valid months
                            max_days_allowed = 180
                            if days_since_last > max_days_allowed:
                                print(f"    ⚠ Skipping: Last valid data point ({last_valid_idx}) is {days_since_last} days before view_date ({view_date}) (too old, >{max_days_allowed} days)")
                                continue
                            else:
                                print(f"    ✓ Last valid data point ({last_valid_idx}) is {days_since_last} days before view_date (acceptable)")
                        else:
                            print(f"    ⚠ Warning: No valid data point found in fit_data after cleaning")
                            # Don't skip here - let the fitting attempt proceed, it will fail if there's truly no data
                        
                        # CRITICAL FIX: For VAR models, store original column names for mapping back after prediction
                        # VAR's internal transformation pipeline may use integer column indices
                        original_columns = None
                        if model == 'var' and isinstance(fit_data, pd.DataFrame):
                            original_columns = fit_data.columns.tolist()
                        
                        # Fit on training data
                        try:
                            # CRITICAL FIX: For VAR models, the internal transformation pipeline may have been fitted
                            # with integer column indices, but we're passing data with string column names.
                            # The ColumnEnsembleTransformer in VAR's pipeline uses integer indices internally.
                            # Solution: Convert DataFrame columns to integer indices before fitting, then map back after prediction.
                            fit_data_for_fit = fit_data.copy()
                            
                            if model == 'var' and isinstance(fit_data, pd.DataFrame):
                                
                                # Check if forecaster has a pipeline with ColumnEnsembleTransformer
                                # If so, convert DataFrame columns to integer indices to match pipeline expectations
                                has_column_ensemble = False
                                if hasattr(forecaster_clone, 'forecaster_'):
                                    forecaster_internal = forecaster_clone.forecaster_
                                    # Check if it's a ForecastingPipeline
                                    if hasattr(forecaster_internal, 'steps'):
                                        for step_name, step_transformer in forecaster_internal.steps:
                                            # Check if transformer is ColumnEnsembleTransformer
                                            if hasattr(step_transformer, '__class__'):
                                                class_name = step_transformer.__class__.__name__
                                                if 'ColumnEnsemble' in class_name or 'ColumnTransformer' in class_name:
                                                    has_column_ensemble = True
                                                    # Convert DataFrame columns to integer indices
                                                    fit_data_for_fit.columns = range(len(fit_data_for_fit.columns))
                                                    print(f"    → VAR: Converted DataFrame columns to integer indices (0-{len(original_columns)-1}) for pipeline compatibility")
                                                    break
                                
                                # If no ColumnEnsembleTransformer found, try to reset pipeline transformers
                                if not has_column_ensemble:
                                    if hasattr(forecaster_clone, 'forecaster_') and hasattr(forecaster_clone.forecaster_, 'steps'):
                                        # This is a ForecastingPipeline - try to reset transformers
                                        try:
                                            for step_name, step_transformer in forecaster_clone.forecaster_.steps:
                                                if hasattr(step_transformer, 'reset'):
                                                    try:
                                                        step_transformer.reset()
                                                    except (AttributeError, Exception):
                                                        pass
                                                # Reset fitted state of transformers
                                                if hasattr(step_transformer, '_is_fitted'):
                                                    step_transformer._is_fitted = False
                                        except (AttributeError, Exception):
                                            pass
                            
                            forecaster_clone.fit(fit_data_for_fit)
                        except Exception as e:
                            print(f"    ✗ Error fitting forecaster: {e}")
                            import traceback
                            traceback.print_exc()
                            continue
                        
                        # Nowcast for target_month_end (horizon = horizon_days)
                        # This is nowcasting: predicting current period using incomplete data
                        # CRITICAL FIX: Convert horizon_days to periods based on data frequency
                        # Data is now monthly (resampled), so convert to months
                        # For nowcasting, after refitting on data up to view_date, we predict from the last data point
                        # to target_month_end. The horizon should be calculated from the last month in fit_data
                        # (which is the data the model was just refit on) to target_month_end.
                        # Use relativedelta for accurate month difference calculation
                        from dateutil.relativedelta import relativedelta
                        # Get the last month in the refitted data (fit_data is monthly)
                        last_fit_month = fit_data.index.max()
                        # Convert to datetime if needed
                        if isinstance(last_fit_month, pd.Timestamp):
                            last_fit_dt = last_fit_month.to_pydatetime()
                        else:
                            last_fit_dt = last_fit_month
                        # target_month_end_ts is already pd.Timestamp
                        target_dt = target_month_end_ts.to_pydatetime()
                        # Calculate months difference from last data point to target month
                        # For nowcasting, we're predicting the target month using data up to view_date
                        delta = relativedelta(target_dt, last_fit_dt)
                        months_ahead = delta.years * 12 + delta.months
                        # CRITICAL FIX: For nowcasting, we always predict the target_month_end month.
                        # The horizon should be the number of months from the last data point in fit_data to target_month_end.
                        # If months_ahead is 0 (same month), we need horizon=1 to predict that month (1 period ahead).
                        # If months_ahead is negative (target is before last data point, shouldn't happen), use 1 as fallback.
                        # If months_ahead is positive, use that value (predict months_ahead periods ahead).
                        # For nowcasting, months_ahead should typically be 0 or 1 (predicting current or next month).
                        if months_ahead < 0:
                            print(f"    ⚠ Warning: months_ahead is negative ({months_ahead}), using horizon=1 as fallback")
                            horizon_periods = 1
                        elif months_ahead == 0:
                            # Same month: predict 1 period ahead to get the target month
                            horizon_periods = 1
                        else:
                            # Future month: use months_ahead
                            horizon_periods = months_ahead
                        try:
                            fh = np.asarray([horizon_periods])
                            print(f"    → Predicting with horizon={horizon_periods} periods ({horizon_periods} months, from last data point {last_fit_month} to {target_month_end_ts})")
                            y_pred = forecaster_clone.predict(fh=fh)
                            
                            # CRITICAL FIX: If VAR was fitted with integer column indices, map prediction columns back to original names
                            if model == 'var' and original_columns is not None and isinstance(y_pred, pd.DataFrame):
                                # Prediction will have integer column indices (0, 1, 2, ...)
                                # Map them back to original column names
                                if len(y_pred.columns) == len(original_columns):
                                    y_pred.columns = original_columns
                                    print(f"    → VAR: Mapped prediction columns back to original names: {original_columns}")
                            
                            print(f"    → Prediction result type: {type(y_pred)}, shape: {getattr(y_pred, 'shape', 'N/A')}")
                        except Exception as e:
                            print(f"    ✗ Error in prediction: {e}")
                            import traceback
                            traceback.print_exc()
                            # Set forecast_value to NaN and continue to skip check below
                            forecast_value = np.nan
                            y_pred = None
                        
                        # Extract nowcast value
                        if y_pred is None:
                            forecast_value = np.nan
                            print(f"    ⚠ Prediction returned None")
                        elif isinstance(y_pred, pd.DataFrame):
                            if len(y_pred) == 0:
                                forecast_value = np.nan
                                print(f"    ⚠ Prediction DataFrame is empty")
                            elif target_series in y_pred.columns:
                                raw_val = y_pred[target_series].iloc[0]
                                if pd.isna(raw_val):
                                    forecast_value = np.nan
                                    print(f"    ⚠ Prediction value for {target_series} is NaN in DataFrame")
                                else:
                                    forecast_value = float(raw_val)
                            elif len(y_pred.columns) > 0:
                                # Use first column if target_series not found
                                # Handle both string and integer column indices
                                try:
                                    raw_val = y_pred.iloc[0, 0]
                                    if pd.isna(raw_val):
                                        forecast_value = np.nan
                                        print(f"    ⚠ Prediction value at iloc[0,0] is NaN")
                                    else:
                                        forecast_value = float(raw_val)
                                except (KeyError, IndexError, TypeError) as e:
                                    # If iloc fails, try accessing by column name/index
                                    first_col = y_pred.columns[0]
                                    if isinstance(first_col, (int, np.integer)):
                                        raw_val = y_pred.iloc[0, first_col]
                                    else:
                                        raw_val = y_pred[first_col].iloc[0]
                                    if pd.isna(raw_val):
                                        forecast_value = np.nan
                                        print(f"    ⚠ Prediction value from column {first_col} is NaN")
                                    else:
                                        forecast_value = float(raw_val)
                            else:
                                forecast_value = np.nan
                                print(f"    ⚠ Prediction DataFrame has no columns")
                        elif isinstance(y_pred, pd.Series):
                            if len(y_pred) > 0:
                                raw_val = y_pred.iloc[0]
                                if pd.isna(raw_val):
                                    forecast_value = np.nan
                                    print(f"    ⚠ Prediction value in Series is NaN")
                                else:
                                    forecast_value = float(raw_val)
                            else:
                                forecast_value = np.nan
                                print(f"    ⚠ Prediction Series is empty")
                        elif hasattr(y_pred, '__len__') and len(y_pred) > 0:
                            try:
                                raw_val = y_pred[0] if hasattr(y_pred, '__getitem__') else y_pred
                                if pd.isna(raw_val) if hasattr(pd, 'isna') else (raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val))):
                                    forecast_value = np.nan
                                    print(f"    ⚠ Prediction value from array-like is NaN")
                                else:
                                    forecast_value = float(raw_val)
                            except (TypeError, IndexError, KeyError) as e:
                                forecast_value = np.nan
                                print(f"    ⚠ Error extracting prediction from array-like: {e}")
                        else:
                            try:
                                if pd.isna(y_pred) if hasattr(pd, 'isna') else (y_pred is None or (isinstance(y_pred, float) and np.isnan(y_pred))):
                                    forecast_value = np.nan
                                    print(f"    ⚠ Prediction scalar value is NaN")
                                else:
                                    forecast_value = float(y_pred)
                            except (TypeError, ValueError) as e:
                                forecast_value = np.nan
                                print(f"    ⚠ Error converting prediction to float: {e}")
                        
                        # Check if nowcast is NaN
                        if np.isnan(forecast_value):
                            print(f"    ⚠ Nowcast returned NaN (horizon={horizon_days} days)")
                        else:
                            print(f"    ✓ Nowcast value extracted: {forecast_value:.4f} (horizon={horizon_days} days)")
                        
                    # Get actual value from monthly aggregated data (targets are monthly)
                    # Handle cases where target_month_end might not be exactly in index
                    # CRITICAL FIX: For nowcasting, we need actual values from the full dataset (including 2024)
                    # Check if full_data_monthly has data up to target_month_end
                    if len(full_data_monthly) > 0 and target_month_end_ts <= full_data_monthly.index.max():
                        # Find closest date at or before target_month_end (use monthly data)
                        available_dates = full_data_monthly.index[full_data_monthly.index <= target_month_end_ts]
                        if len(available_dates) > 0:
                            closest_date = available_dates.max()
                            # For nowcasting, we want the actual value for the target month
                            # If closest_date is in the same month as target_month_end, use it
                            # Otherwise, try to find a date in the target month
                            if closest_date.year == target_month_end_ts.year and closest_date.month == target_month_end_ts.month:
                                # Perfect match - use this date
                                if target_series in full_data_monthly.columns:
                                    raw_value = full_data_monthly.loc[closest_date, target_series]
                                else:
                                    raw_value = full_data_monthly.loc[closest_date].iloc[0]
                                if pd.isna(raw_value):
                                    print(f"    ⚠ Warning: Actual value at {closest_date} is NaN in monthly data")
                                    actual_value = np.nan
                                else:
                                    actual_value = float(raw_value)
                                    print(f"    ✓ Found actual value at {closest_date}: {actual_value:.4f}")
                            else:
                                # Closest date is in a different month - try to find exact month match
                                target_month_dates = full_data_monthly.index[
                                    (full_data_monthly.index.year == target_month_end_ts.year) &
                                    (full_data_monthly.index.month == target_month_end_ts.month)
                                ]
                                if len(target_month_dates) > 0:
                                    exact_date = target_month_dates.max()
                                    if target_series in full_data_monthly.columns:
                                        raw_value = full_data_monthly.loc[exact_date, target_series]
                                    else:
                                        raw_value = full_data_monthly.loc[exact_date].iloc[0]
                                    if pd.isna(raw_value):
                                        print(f"    ⚠ Warning: Actual value at {exact_date} is NaN in monthly data")
                                        actual_value = np.nan
                                    else:
                                        actual_value = float(raw_value)
                                        print(f"    ✓ Found actual value at {exact_date}: {actual_value:.4f}")
                                else:
                                    # No exact month match, use closest date
                                    if target_series in full_data_monthly.columns:
                                        raw_value = full_data_monthly.loc[closest_date, target_series]
                                    else:
                                        raw_value = full_data_monthly.loc[closest_date].iloc[0]
                                    if pd.isna(raw_value):
                                        print(f"    ⚠ Warning: Actual value at {closest_date} is NaN in monthly data")
                                        actual_value = np.nan
                                    else:
                                        actual_value = float(raw_value)
                                        print(f"    ✓ Using closest available date {closest_date}: {actual_value:.4f}")
                        else:
                            print(f"    ⚠ Warning: No monthly data available at or before {target_month_end_ts}")
                            actual_value = np.nan
                    else:
                        # target_month_end is after data range or full_data_monthly is empty
                        if len(full_data_monthly) == 0:
                            print(f"    ⚠ Warning: full_data_monthly is empty - cannot get actual value")
                        else:
                            print(f"    ⚠ Warning: target_month_end {target_month_end_ts} is after data range (max: {full_data_monthly.index.max()})")
                        actual_value = np.nan
                        
                    except Exception as e:
                        print(f"    ✗ Error in prediction: {e}")
                        import traceback
                        traceback.print_exc()
                        forecast_value = np.nan
                        # Try to get actual value even if prediction failed (use monthly data)
                        if target_month_end_ts <= full_data_monthly.index.max():
                            available_dates = full_data_monthly.index[full_data_monthly.index <= target_month_end_ts]
                            if len(available_dates) > 0:
                                closest_date = available_dates.max()
                                if target_series in full_data_monthly.columns:
                                    raw_value = full_data_monthly.loc[closest_date, target_series]
                                    if pd.isna(raw_value):
                                        print(f"    ⚠ Warning: Actual value at {closest_date} is NaN in monthly data")
                                        actual_value = np.nan
                                    else:
                                        actual_value = float(raw_value)
                                else:
                                    raw_value = full_data_monthly.loc[closest_date].iloc[0]
                                    if pd.isna(raw_value):
                                        print(f"    ⚠ Warning: Actual value at {closest_date} is NaN in monthly data")
                                        actual_value = np.nan
                                    else:
                                        actual_value = float(raw_value)
                            else:
                                actual_value = np.nan
                        else:
                            actual_value = np.nan
                
                # Skip if nowcast or actual is NaN
                if np.isnan(forecast_value) or np.isnan(actual_value):
                    if np.isnan(forecast_value):
                        print(f"    ⚠ Skipping: forecast_value is NaN (prediction failed or returned NaN)")
                    if np.isnan(actual_value):
                        print(f"    ⚠ Skipping: actual_value is NaN (value not found or is NaN in data)")
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
                    'month': target_month_end_ts.strftime('%Y-%m'),
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
                
                print(f"    ✓ Nowcast: {forecast_value:.4f}, Actual: {actual_value:.4f}, Error: {error:.4f}")
                
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
            print(f"    Total target months: {len(target_periods)}")
            print(f"    This means all {len(target_periods)} months were skipped due to validation failures or NaN predictions/actuals")
    
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
    
    # Nowcasting backtest command (NOT forecasting - this is for nowcasting backtesting)
    backtest_parser = subparsers.add_parser('backtest', help='Run nowcasting backtest evaluation for all models (requires experiment config)')
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
    
    if args.command == 'backtest':
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
        print(f"\n✓ Nowcasting backtest completed (this is nowcasting, not forecasting)")


if __name__ == "__main__":
    main()
