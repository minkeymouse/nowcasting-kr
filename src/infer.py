"""Nowcasting module - generates JSON/CSV results only."""
from pathlib import Path
import sys
import argparse
import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pickle
import json

# Set up paths
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from src.utils import setup_cli_environment
setup_cli_environment()

from src.utils import (
    get_project_root,
    parse_experiment_config,
    extract_experiment_params,
    validate_experiment_config
)
try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def _validate_checkpoint(model_data: dict, target_series: str, model: str) -> None:
    """Validate checkpoint contains required data.
    
    Raises:
        ValueError: If checkpoint is missing required fields
    """
    if not isinstance(model_data, dict):
        raise ValueError(f"Checkpoint for {target_series}_{model} is not a dictionary")
    
    if 'forecaster' not in model_data:
        raise ValueError(f"Checkpoint for {target_series}_{model} does not contain 'forecaster'")
    
    forecaster = model_data.get('forecaster')
    if forecaster is None:
        raise ValueError(f"Checkpoint for {target_series}_{model} has None forecaster")
    
    # For DFM/DDFM models, check if data_module is present (optional but recommended)
    if model.lower() in ['dfm', 'ddfm']:
        if hasattr(forecaster, '_dfm_model') or hasattr(forecaster, '_ddfm_model'):
            dfm_model = forecaster._dfm_model if hasattr(forecaster, '_dfm_model') else forecaster._ddfm_model
            if dfm_model is not None:
                if not hasattr(dfm_model, '_data_module') or dfm_model._data_module is None:
                    logger.warning(f"Checkpoint for {target_series}_{model} missing data_module - may need to recreate during inference")


def _load_model_for_inference(
    project_root: Path,
    target_series: str,
    model: str,
    supports_nowcast_manager: bool,
    train_start: str,
    train_end: str
) -> Tuple[Any, Optional[Any]]:
    """Load trained model from checkpoint with validation."""
    checkpoint_path = project_root / "checkpoint" / f"{target_series}_{model.lower()}" / "model.pkl"
    if not checkpoint_path.exists():
        comparisons_dir = project_root / "outputs" / "comparisons"
        if comparisons_dir.exists():
            for comparison_dir in comparisons_dir.glob(f"{target_series}_*"):
                model_path = comparison_dir / model.lower() / "model.pkl"
                if model_path.exists():
                    checkpoint_path = model_path
                    logger.info(f"Found checkpoint in comparisons directory: {checkpoint_path}")
                    break
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Trained {model.upper()} model not found for {target_series}. "
            f"Expected path: {project_root / 'checkpoint' / f'{target_series}_{model.lower()}' / 'model.pkl'}"
        )
    
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    try:
        with open(checkpoint_path, 'rb') as f:
            model_data = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, IOError) as e:
        raise ValueError(f"Failed to load checkpoint for {target_series}_{model}: {type(e).__name__}: {str(e)}") from e
    except Exception as e:
        raise ValueError(f"Unexpected error loading checkpoint for {target_series}_{model}: {type(e).__name__}: {str(e)}") from e
    
    # Validate checkpoint structure
    try:
        _validate_checkpoint(model_data, target_series, model)
    except ValueError as e:
        raise ValueError(f"Checkpoint validation failed for {target_series}_{model}: {str(e)}") from e
    
    forecaster = model_data.get('forecaster')
    if forecaster is None:
        raise ValueError(f"Model file does not contain forecaster for {target_series}_{model}")
    
    logger.info(f"Successfully loaded forecaster for {target_series}_{model}")
    
    if not supports_nowcast_manager:
        return forecaster, None
    
    # Extract DFM/DDFM model from forecaster
    # DFMForecaster/DDFMForecaster store DFMBase/DDFMBase instances directly
    if hasattr(forecaster, '_dfm_model'):
        dfm_model = forecaster._dfm_model
    elif hasattr(forecaster, '_ddfm_model'):
        dfm_model = forecaster._ddfm_model
    else:
        raise ValueError(f"Forecaster does not contain {model.upper()} model for {target_series}")
    
    if dfm_model is None:
        raise ValueError(f"{model.upper()} model is None for {target_series}")
    
    logger.info(f"Successfully extracted {model.upper()} model from forecaster")
    
    # Restore result from checkpoint if needed
    # DFMBase/DDFMBase are the actual models (not wrappers), so access _result directly
    if hasattr(dfm_model, '_result') and dfm_model._result is None:
        # Try to restore from checkpoint first (safest, no training)
        if 'result' in model_data:
            try:
                dfm_model._result = model_data['result']
                logger.debug(f"Restored result from checkpoint for {target_series}_{model}")
            except Exception as e:
                logger.debug(f"Could not restore result from checkpoint: {str(e)}")
        # Only restore from training_state if result is already computed (no EM)
        if dfm_model._result is None and hasattr(dfm_model, 'training_state') and dfm_model.training_state is not None:
            # Check if result is already computed in training_state
            if hasattr(dfm_model.training_state, '_result'):
                try:
                    dfm_model._result = dfm_model.training_state._result
                    logger.debug(f"Restored result from training_state for {target_series}_{model}")
                except Exception as e:
                    logger.debug(f"Could not restore result from training_state: {str(e)}")
    
    # Check for data_module (required for nowcasting)
    if not hasattr(dfm_model, '_data_module') or dfm_model._data_module is None:
        logger.warning(f"data_module not found in checkpoint for {target_series}_{model}. Model may not work correctly for nowcasting.")
        # Try to restore from checkpoint if available
        if 'data_module' in model_data:
            try:
                dfm_model._data_module = model_data['data_module']
                logger.info(f"Restored data_module from checkpoint for {target_series}_{model}")
            except Exception as e:
                logger.warning(f"Could not restore data_module from checkpoint: {str(e)}")
    
    # Return forecaster and dfm_model
    return forecaster, dfm_model


def _extract_target_forecast(
    X_pred: np.ndarray,
    pred_step: int,
    dfm_model: Any,
    target_series: str
) -> float:
    """Extract target series forecast value from prediction array.
    
    Args:
        X_pred: Prediction array (horizon x n_series)
        pred_step: Time step index to extract
        dfm_model: DFM/DDFM model instance
        target_series: Target series name
        
    Returns:
        Forecast value for target series
    """
    # Ensure pred_step is valid
    if pred_step >= X_pred.shape[0]:
        pred_step = X_pred.shape[0] - 1
    if pred_step < 0:
        pred_step = 0
    
    # Try to find target series index
    if hasattr(dfm_model, 'config'):
        try:
            from dfm_python.utils.helpers import find_series_index
            series_idx = find_series_index(dfm_model.config, target_series)
            if series_idx is not None and series_idx < X_pred.shape[1]:
                return float(X_pred[pred_step, series_idx])
        except (ValueError, IndexError, KeyError, AttributeError):
            pass
    
    # Fallback to first column
    return float(X_pred[pred_step, 0] if X_pred.shape[1] > 0 else 0.0)


def _save_json_results(
    output_file: Path,
    results: Dict[str, Any],
    logger: logging.Logger
) -> None:
    """Save results to JSON file with validation.
    
    Args:
        output_file: Path to output file
        results: Results dictionary to save
        logger: Logger instance
        
    Raises:
        IOError: If file cannot be created or is empty
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        
        # Validate file was created and is non-empty
        if not output_file.exists():
            raise IOError(f"Output file was not created: {output_file}")
        
        file_size = output_file.stat().st_size
        if file_size == 0:
            raise IOError(f"Output file is empty: {output_file}")
        
        logger.info(f"Results saved successfully to: {output_file} (size: {file_size} bytes)")
    except (IOError, OSError) as e:
        logger.error(f"Failed to save results to {output_file}: {type(e).__name__}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error saving results to {output_file}: {type(e).__name__}: {str(e)}")
        raise


def _update_data_module_for_nowcasting(
    dfm_model: Any,
    data_up_to_target: pd.DataFrame,
    target_month_end_ts: pd.Timestamp
) -> None:
    """Update data_module with data up to target month end.
    
    Args:
        dfm_model: DFMBase or DDFMBase model instance
        data_up_to_target: DataFrame with data up to view date
        target_month_end_ts: Target month end timestamp
        
    Raises:
        RuntimeError: If data_module is not available
        ValueError: If data is empty or invalid
    """
    if not hasattr(dfm_model, '_data_module'):
        raise RuntimeError(f"dfm_model does not have _data_module attribute")
    
    if dfm_model._data_module is None:
        raise RuntimeError(f"data_module is None - cannot update for nowcasting")
    
    # Validate input data
    if data_up_to_target is None:
        raise ValueError(f"data_up_to_target is None")
    
    if not isinstance(data_up_to_target, pd.DataFrame):
        raise ValueError(f"data_up_to_target must be a DataFrame, got {type(data_up_to_target)}")
    
    if len(data_up_to_target) == 0:
        raise ValueError(f"No data available in data_up_to_target (empty DataFrame)")
    
    if len(data_up_to_target.columns) == 0:
        raise ValueError(f"No columns in data_up_to_target")
    
    from src.preprocessing import resample_to_monthly
    from dfm_python.utils.time import TimeIndex
    
    try:
        data_monthly = resample_to_monthly(data_up_to_target)
    except Exception as e:
        raise ValueError(f"Failed to resample data to monthly: {type(e).__name__}: {str(e)}") from e
    
    if len(data_monthly) == 0:
        raise ValueError(f"No data available after resampling to monthly (target: {target_month_end_ts.strftime('%Y-%m')})")
    
    if len(data_monthly.columns) == 0:
        raise ValueError(f"No columns in resampled monthly data")
    
    # Validate index
    if data_monthly.index is None or len(data_monthly.index) == 0:
        raise ValueError(f"Invalid index in resampled monthly data")
    
    try:
        time_index = TimeIndex(data_monthly.index.to_pydatetime().tolist())
    except Exception as e:
        raise ValueError(f"Failed to create TimeIndex from data: {type(e).__name__}: {str(e)}") from e
    
    if len(time_index) == 0:
        raise ValueError(f"TimeIndex is empty after creation")
    
    # Update data_module
    try:
        existing_data_module = dfm_model._data_module
        existing_data_module.data = data_monthly
        existing_data_module.time_index = time_index
        dfm_model._data_module = existing_data_module
        logger.debug(f"Successfully updated data_module with {len(data_monthly)} monthly observations")
    except Exception as e:
        raise RuntimeError(f"Failed to update data_module: {type(e).__name__}: {str(e)}") from e


def run_backtest_evaluation(
    config_name: str,
    model: str,
    train_start: str = "1985-01-01",
    train_end: str = "2019-12-31",
    nowcast_start: str = "2024-01-01",
    nowcast_end: str = "2025-10-31",
    weeks_before: Optional[List[int]] = None,
    config_dir: Optional[str] = None,
    overrides: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Run backtest evaluation and generate JSON/CSV results."""
    project_root = get_project_root()
    
    if config_dir is None:
        config_dir = str(project_root / "config")
    
    if weeks_before is None:
        weeks_before = [4, 1]
    
    supports_nowcast_manager = model.lower() in ['dfm', 'ddfm']
    
    cfg = parse_experiment_config(config_name, config_dir, overrides)
    validate_experiment_config(cfg, require_target=True, require_models=False)
    
    params = extract_experiment_params(cfg)
    target_series = params['target_series']
    exp_cfg = params.get('exp_cfg', cfg)
    
    # Load release dates
    series_release_dates = {}
    if hasattr(exp_cfg, 'get'):
        series_ids_raw = exp_cfg.get('series', [])
    else:
        series_ids_raw = getattr(exp_cfg, 'series', [])
    
    if series_ids_raw:
        import yaml
        series_config_dir = project_root / "dfm-python" / "config" / "series"
        series_ids = OmegaConf.to_container(series_ids_raw, resolve=True) if hasattr(OmegaConf, 'to_container') else list(series_ids_raw)
        if not isinstance(series_ids, list):
            series_ids = []
        
        for series_id in series_ids:
            series_config_path = series_config_dir / f"{series_id}.yaml"
            if series_config_path.exists():
                try:
                    with open(series_config_path, 'r', encoding='utf-8') as f:
                        series_cfg = yaml.safe_load(f) or {}
                    release_date = series_cfg.get('release') or series_cfg.get('release_date')
                    if release_date is not None:
                        series_release_dates[series_id] = release_date
                except Exception:
                    pass
    
    # Load model
    forecaster, dfm_model = _load_model_for_inference(
        project_root, target_series, model, supports_nowcast_manager, train_start, train_end
    )
    
    # Generate target periods
    start_date = datetime.strptime(nowcast_start, "%Y-%m-%d")
    end_date = datetime.strptime(nowcast_end, "%Y-%m-%d")
    target_periods = []
    current = start_date.replace(day=1)
    while current <= end_date:
        if current.month == 12:
            last_day = current.replace(day=31)
        else:
            next_month = current.replace(month=current.month + 1, day=1)
            last_day = next_month - timedelta(days=1)
        target_periods.append(last_day)
        current += relativedelta(months=1)
    
    # Load data
    from src.preprocessing import resample_to_monthly
    data_path_file = project_root / "data" / "data.csv"
    if not data_path_file.exists():
        data_path_file = project_root / "data" / "sample_data.csv"
    full_data = pd.read_csv(data_path_file, index_col=0, parse_dates=True)
    full_data_monthly = resample_to_monthly(full_data)
    
    train_start_ts = pd.Timestamp(train_start)
    train_end_ts = pd.Timestamp(train_end)
    train_data_filtered = full_data[(full_data.index >= train_start_ts) & (full_data.index <= train_end_ts)]
    train_std = train_data_filtered[target_series].std() if target_series in train_data_filtered.columns else train_data_filtered.iloc[:, 0].std() if len(train_data_filtered.columns) > 0 else 1.0
    
    results_by_timepoint = {}
    train_end_date = datetime.strptime(train_end, "%Y-%m-%d")
    
    logger.info(f"Starting nowcasting backtest for {target_series} with {model.upper()}")
    logger.info(f"Target periods: {len(target_periods)} months from {nowcast_start} to {nowcast_end}")
    logger.info(f"Time points: {weeks_before} weeks before month end")
    
    for weeks in weeks_before:
        monthly_results = []
        logger.info(f"Processing {weeks} weeks before time point...")
        
        for target_month_end in target_periods:
            target_month_end_ts = pd.Timestamp(target_month_end) if not isinstance(target_month_end, pd.Timestamp) else target_month_end
            view_date = target_month_end_ts - timedelta(weeks=weeks)
            view_date_ts = pd.Timestamp(view_date)
            
            if view_date_ts <= pd.Timestamp(train_end_date):
                continue
            
            if len(full_data_monthly) == 0:
                continue
            
            try:
                if supports_nowcast_manager:
                    # Use predict method directly (simpler and faster)
                    # Update data_module with data up to view_date for nowcasting
                    data_up_to_view = full_data[full_data.index <= pd.Timestamp(view_date)].copy()
                    try:
                        _update_data_module_for_nowcasting(dfm_model, data_up_to_view, pd.Timestamp(view_date))
                    except (RuntimeError, ValueError) as e:
                        logger.warning(f"Failed to update data_module for {target_month_end_ts.strftime('%Y-%m')} (view_date={view_date.strftime('%Y-%m-%d')}): {str(e)}")
                        continue
                    except Exception as e:
                        logger.error(f"Unexpected error updating data_module for {target_month_end_ts.strftime('%Y-%m')} (view_date={view_date.strftime('%Y-%m-%d')}): {type(e).__name__}: {str(e)}")
                        continue
                    
                    # Use predict method - can use any horizon (1 for nowcasting, 22 for forecasting)
                    # Calculate horizon: number of months from view_date to target_month_end
                    view_date_ts = pd.Timestamp(view_date)
                    months_ahead = (target_month_end_ts.year - view_date_ts.year) * 12 + (target_month_end_ts.month - view_date_ts.month)
                    # For nowcasting, we want to predict the target period, so horizon should be based on months difference
                    # But if target is in the past relative to view_date, use horizon=1 (current period)
                    horizon = max(1, min(months_ahead, 22)) if months_ahead >= 0 else 1
                    
                    try:
                        # Use dfm_model.predict() directly (it's DFMBase/DDFMBase instance)
                        X_pred, Z_pred = dfm_model.predict(horizon=horizon, return_series=True, return_factors=True)
                        
                        # Extract target series value
                        # If horizon > 1, we need to select the appropriate time step
                        # For nowcasting, we typically want the last step (horizon-1) if months_ahead matches
                        pred_step = min(horizon - 1, max(0, months_ahead - 1)) if months_ahead > 0 else 0
                        
                        # Extract target series value from predictions
                        forecast_value = _extract_target_forecast(
                            X_pred, pred_step, dfm_model, target_series
                        )
                        
                        if not np.isfinite(forecast_value):
                            logger.debug(f"Non-finite forecast value for {target_month_end_ts.strftime('%Y-%m')} (view_date={view_date.strftime('%Y-%m-%d')}): {forecast_value}")
                            continue
                    except (ValueError, RuntimeError, AttributeError) as e:
                        logger.warning(f"Failed to predict for {target_month_end_ts.strftime('%Y-%m')} (view_date={view_date.strftime('%Y-%m-%d')}): {type(e).__name__}: {str(e)}")
                        continue
                    except Exception as e:
                        logger.error(f"Unexpected error during prediction for {target_month_end_ts.strftime('%Y-%m')} (view_date={view_date.strftime('%Y-%m-%d')}): {type(e).__name__}: {str(e)}")
                        continue
                    
                    # Get actual value
                    actual_value = np.nan
                    if target_month_end_ts <= full_data_monthly.index.max():
                        available_dates = full_data_monthly.index[full_data_monthly.index <= target_month_end_ts]
                        if len(available_dates) > 0:
                            closest_date = available_dates.max()
                            if target_series in full_data_monthly.columns:
                                raw_value = full_data_monthly.loc[closest_date, target_series]
                            else:
                                raw_value = full_data_monthly.loc[closest_date].iloc[0]
                            if not pd.isna(raw_value):
                                actual_value = float(raw_value)
                
                else:
                    # ARIMA/VAR
                    train_data_filtered = full_data[full_data.index <= pd.Timestamp(view_date)].copy()
                    if len(train_data_filtered) == 0:
                        continue
                    
                    from src.preprocessing import resample_to_monthly
                    train_data_filtered = resample_to_monthly(train_data_filtered)
                    if len(train_data_filtered) == 0:
                        continue
                    
                    # Apply release date masking
                    if series_release_dates:
                        # Import calculate_release_date (defined early in nowcasting.py, doesn't depend on para_const)
                        from src.nowcasting import calculate_release_date
                        view_date_dt = view_date if isinstance(view_date, datetime) else pd.Timestamp(view_date).to_pydatetime()
                        for series_id, release_offset in series_release_dates.items():
                            if series_id not in train_data_filtered.columns:
                                continue
                            for period_idx in train_data_filtered.index:
                                period_dt = period_idx.to_pydatetime() if isinstance(period_idx, pd.Timestamp) else period_idx
                                release_date_dt = calculate_release_date(release_offset, period_dt)
                                if release_date_dt > view_date_dt:
                                    train_data_filtered.loc[period_idx, series_id] = np.nan
                    
                    # Impute missing values after masking (ARIMA/VAR cannot handle NaN)
                    from src.preprocessing import impute_missing_values
                    try:
                        train_data_filtered = impute_missing_values(train_data_filtered, model_type=model.lower())
                        if train_data_filtered.isnull().any().any():
                            # If imputation failed, drop rows with remaining NaN
                            nan_count = train_data_filtered.isnull().sum().sum()
                            logger.warning(f"{model.upper()}: {nan_count} NaN values remain after imputation for {target_month_end_ts.strftime('%Y-%m')} (view_date={view_date.strftime('%Y-%m-%d')}). Dropping rows with NaN...")
                            train_data_filtered = train_data_filtered.dropna()
                            if len(train_data_filtered) == 0:
                                logger.warning(f"{model.upper()}: All data was dropped after imputation for {target_month_end_ts.strftime('%Y-%m')} (view_date={view_date.strftime('%Y-%m-%d')}). Skipping this timepoint.")
                                continue
                    except Exception as e:
                        logger.warning(f"{model.upper()}: Imputation failed for {target_month_end_ts.strftime('%Y-%m')} (view_date={view_date.strftime('%Y-%m-%d')}): {type(e).__name__}: {str(e)}. Skipping this timepoint.")
                        continue
                    
                    # Prepare data
                    is_multivariate = False
                    training_columns = None
                    if hasattr(forecaster, 'get_tag'):
                        try:
                            scitype_y = forecaster.get_tag('scitype:y', 'univariate')
                            is_multivariate = (scitype_y == 'multivariate')
                        except Exception:
                            pass
                    
                    if hasattr(forecaster, '_y'):
                        try:
                            if isinstance(forecaster._y, pd.DataFrame):
                                is_multivariate = len(forecaster._y.columns) > 1
                                training_columns = forecaster._y.columns.tolist()
                        except Exception:
                            pass
                    
                    if not is_multivariate:
                        is_multivariate = len(train_data_filtered.columns) > 1
                    
                    original_columns = None
                    if is_multivariate:
                        if training_columns is not None:
                            available_columns = [col for col in training_columns if col in train_data_filtered.columns]
                            original_columns = available_columns
                            if len(available_columns) < 2:
                                continue
                            fit_data = train_data_filtered[available_columns].copy()
                            fit_data = fit_data.reindex(columns=available_columns)
                        else:
                            fit_data = train_data_filtered.copy()
                            original_columns = list(fit_data.columns)
                        fit_data = fit_data.dropna(how='all')
                    else:
                        if target_series in train_data_filtered.columns:
                            fit_data = train_data_filtered[[target_series]].copy()
                        else:
                            fit_data = train_data_filtered[[train_data_filtered.columns[0]]].copy()
                        target_col_name = fit_data.columns[0]
                        if fit_data[target_col_name].isna().any():
                            fit_data = fit_data.ffill()
                        fit_data = fit_data.dropna()
                        if isinstance(fit_data, pd.Series):
                            fit_data = fit_data.to_frame()
                    
                    if len(fit_data) < 10:
                        continue
                    
                    # Re-fit model on masked and imputed data for nowcasting
                    # This is necessary because the checkpoint model was trained on full data,
                    # but nowcasting requires predictions based on masked data
                    try:
                        forecaster.fit(fit_data)
                        logger.debug(f"{model.upper()}: Re-fitted on masked data for {target_month_end_ts.strftime('%Y-%m')} (view_date={view_date.strftime('%Y-%m-%d')})")
                    except Exception as e:
                        logger.warning(f"{model.upper()}: Failed to re-fit for {target_month_end_ts.strftime('%Y-%m')} (view_date={view_date.strftime('%Y-%m-%d')}): {type(e).__name__}: {str(e)}")
                        continue
                    
                    # Predict
                    last_fit_month = fit_data.index.max()
                    if last_fit_month is None:
                        continue
                    
                    if isinstance(last_fit_month, pd.Timestamp):
                        last_fit_dt = last_fit_month.to_pydatetime()
                    else:
                        last_fit_dt = last_fit_month
                    target_dt = target_month_end_ts.to_pydatetime()
                    delta = relativedelta(target_dt, last_fit_dt)
                    months_ahead = delta.years * 12 + delta.months
                    if months_ahead < 0:
                        horizon_periods = 1
                    elif months_ahead == 0:
                        horizon_periods = 1
                    else:
                        horizon_periods = months_ahead
                    
                    try:
                        fh = np.asarray([horizon_periods])
                        y_pred = forecaster.predict(fh=fh)
                        
                        if model == 'var' and original_columns is not None and isinstance(y_pred, pd.DataFrame):
                            if len(y_pred.columns) == len(original_columns):
                                y_pred.columns = original_columns
                    except (ValueError, RuntimeError, AttributeError) as e:
                        logger.warning(f"Failed to predict for {target_month_end_ts.strftime('%Y-%m')} (view_date={view_date.strftime('%Y-%m-%d')}) with {model.upper()}: {type(e).__name__}: {str(e)}")
                        continue
                    except Exception as e:
                        logger.error(f"Unexpected error during prediction for {target_month_end_ts.strftime('%Y-%m')} (view_date={view_date.strftime('%Y-%m-%d')}) with {model.upper()}: {type(e).__name__}: {str(e)}")
                        continue
                    
                    # Extract forecast value
                    if y_pred is None:
                        forecast_value = np.nan
                    elif isinstance(y_pred, pd.DataFrame):
                        if len(y_pred) == 0:
                            forecast_value = np.nan
                        elif target_series in y_pred.columns:
                            raw_val = y_pred[target_series].iloc[0]
                            forecast_value = float(raw_val) if not pd.isna(raw_val) else np.nan
                        elif len(y_pred.columns) > 0:
                            raw_val = y_pred.iloc[0, 0]
                            forecast_value = float(raw_val) if not pd.isna(raw_val) else np.nan
                        else:
                            forecast_value = np.nan
                    elif isinstance(y_pred, pd.Series):
                        if len(y_pred) > 0:
                            raw_val = y_pred.iloc[0]
                            forecast_value = float(raw_val) if not pd.isna(raw_val) else np.nan
                        else:
                            forecast_value = np.nan
                    else:
                        try:
                            raw_val = y_pred[0] if hasattr(y_pred, '__getitem__') else y_pred
                            if pd.isna(raw_val) if hasattr(pd, 'isna') else (raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val))):
                                forecast_value = np.nan
                            else:
                                forecast_value = float(raw_val)
                        except Exception:
                            forecast_value = np.nan
                    
                    # Get actual value
                    actual_value = np.nan
                    if len(full_data_monthly) > 0 and target_month_end_ts <= full_data_monthly.index.max():
                        available_dates = full_data_monthly.index[full_data_monthly.index <= target_month_end_ts]
                        if len(available_dates) > 0:
                            closest_date = available_dates.max()
                            if target_series in full_data_monthly.columns:
                                raw_value = full_data_monthly.loc[closest_date, target_series]
                            else:
                                raw_value = full_data_monthly.loc[closest_date].iloc[0]
                            if not pd.isna(raw_value):
                                actual_value = float(raw_value)
                
                if np.isnan(forecast_value) or np.isnan(actual_value):
                    if np.isnan(forecast_value):
                        logger.debug(f"NaN forecast value for {target_month_end_ts.strftime('%Y-%m')} (view_date={view_date.strftime('%Y-%m-%d')})")
                    if np.isnan(actual_value):
                        logger.debug(f"NaN actual value for {target_month_end_ts.strftime('%Y-%m')} (view_date={view_date.strftime('%Y-%m-%d')})")
                    continue
                
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
                
            except (ValueError, RuntimeError, AttributeError, KeyError, IndexError) as e:
                logger.warning(f"Error processing {target_month_end_ts.strftime('%Y-%m')} (view_date={view_date.strftime('%Y-%m-%d')}): {type(e).__name__}: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error processing {target_month_end_ts.strftime('%Y-%m')} (view_date={view_date.strftime('%Y-%m-%d')}): {type(e).__name__}: {str(e)}")
                continue
        
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
            logger.info(f"Completed {weeks} weeks before: {len(monthly_results)} successful predictions out of {len(target_periods)} target months")
        else:
            logger.warning(f"No successful predictions for {weeks} weeks before time point")
    
    # Save results (always save, even if empty, to indicate status)
    total_timepoints = len(weeks_before)
    successful_timepoints = len(results_by_timepoint)
    
    if len(results_by_timepoint) > 0:
        results = {
            'target_series': target_series,
            'model': model.upper(),
            'train_period': f"{train_start} to {train_end}",
            'nowcast_period': f"{nowcast_start} to {nowcast_end}",
            'weeks_before': weeks_before,
            'results_by_timepoint': results_by_timepoint,
            'horizon': 1,
            'status': 'completed',
            'summary': {
                'total_timepoints': total_timepoints,
                'successful_timepoints': successful_timepoints,
                'failed_timepoints': total_timepoints - successful_timepoints
            }
        }
        logger.info(f"Generated results for {successful_timepoints}/{total_timepoints} time points")
    else:
        results = {
            'target_series': target_series,
            'model': model.upper(),
            'train_period': f"{train_start} to {train_end}",
            'nowcast_period': f"{nowcast_start} to {nowcast_end}",
            'weeks_before': weeks_before,
            'status': 'no_results',
            'error': 'No valid results generated for any time point',
            'summary': {
                'total_timepoints': total_timepoints,
                'successful_timepoints': 0,
                'failed_timepoints': total_timepoints
            }
        }
        logger.warning(f"No results generated for any time point - all predictions failed")
    
    output_file = project_root / "outputs" / "backtest" / f"{target_series}_{model}_backtest.json"
    _save_json_results(output_file, results, logger)
    
    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Run nowcasting backtest")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    backtest_parser = subparsers.add_parser('backtest', help='Run nowcasting backtest')
    backtest_parser.add_argument("--config-name", required=True)
    backtest_parser.add_argument("--model", required=True, choices=['arima', 'var', 'dfm', 'ddfm'])
    backtest_parser.add_argument("--train-start", default="1985-01-01")
    backtest_parser.add_argument("--train-end", default="2019-12-31")
    backtest_parser.add_argument("--nowcast-start", default="2024-01-01")
    backtest_parser.add_argument("--nowcast-end", default="2025-10-31")
    backtest_parser.add_argument("--weeks-before", nargs="+", type=int)
    backtest_parser.add_argument("--override", action="append")
    
    args = parser.parse_args()
    config_path = str(get_project_root() / "config")
    
    if args.command == 'backtest':
        run_backtest_evaluation(
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


if __name__ == "__main__":
    main()
