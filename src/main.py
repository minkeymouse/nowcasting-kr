from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Any, List, Optional, Tuple
import hydra
import logging
import pandas as pd
import numpy as np

from src.preprocess import InvestmentData, ProductionData
from src.utils import setup_logging, get_project_root
from src.train.patchtst import train_patchtst_model
from src.train.tft import train_tft_model
from src.train.itf import train_itf_model
from src.train.dfm import train_dfm_model
from src.train.ddfm import train_ddfm_model
from src.train.timemixer import train_timemixer_model
from src.utils import (
    load_test_data,
    get_monthly_series_from_metadata,
    aggregate_weekly_to_monthly_tent_kernel,
    extract_monthly_actuals
)
from src.metric import (
    compute_smse,
    compute_smae,
    save_experiment_results,
    compute_test_data_std,
    validate_std_for_metrics
)
from src.helper import (
    find_checkpoint_path,
    determine_experiment_type,
    parse_experiment_config,
    extract_target_series_from_config
)
from src.forecast.dfm import run_recursive_forecast as run_recursive_forecast_dfm
from src.forecast.dfm import run_multi_horizon_forecast as run_multi_horizon_forecast_dfm
from src.forecast.ddfm import run_recursive_forecast as run_recursive_forecast_ddfm
from src.forecast.ddfm import run_multi_horizon_forecast as run_multi_horizon_forecast_ddfm
from src.forecast.neuralforecast import (
    run_recursive_forecast as run_recursive_forecast_neuralforecast,
    run_multi_horizon_forecast as run_multi_horizon_forecast_neuralforecast,
    forecast as forecast_neuralforecast
)

logger = logging.getLogger(__name__)

# Model function mappings for cleaner code
TRAIN_FUNCTIONS = {
    'dfm': train_dfm_model,
    'ddfm': train_ddfm_model,
    'timemixer': train_timemixer_model,
    'patchtst': train_patchtst_model,
    'tft': train_tft_model,
    'itf': train_itf_model,
    'itransformer': train_itf_model,
}

def _create_recursive_forecast_wrapper(bound_model_type: str):
    """Create a wrapper that binds model_type for NeuralForecast models."""
    def wrapper(
        checkpoint_path: Path, test_data: pd.DataFrame, start_date: str, end_date: str,
        model_type: str, target_series: Optional[List[str]] = None,
        data_loader: Optional[Any] = None, update_params: bool = False
    ):
        # Use the bound model_type from closure (ignore the parameter)
        return run_recursive_forecast_neuralforecast(
            checkpoint_path, test_data, start_date, end_date, bound_model_type,
            target_series, data_loader, update_params
        )
    return wrapper

RECURSIVE_FORECAST_FUNCTIONS = {
    'dfm': run_recursive_forecast_dfm,
    'ddfm': run_recursive_forecast_ddfm,
    'patchtst': _create_recursive_forecast_wrapper('patchtst'),
    'tft': _create_recursive_forecast_wrapper('tft'),
    'itf': _create_recursive_forecast_wrapper('itf'),
    'itransformer': _create_recursive_forecast_wrapper('itf'),
    'timemixer': _create_recursive_forecast_wrapper('timemixer'),
}

def _create_multi_horizon_forecast_wrapper(bound_model_type: str):
    """Create a wrapper that binds model_type for NeuralForecast models."""
    def wrapper(
        checkpoint_path: Path, horizons: List[int], start_date: str,
        test_data: Optional[pd.DataFrame] = None, target_series: Optional[List[str]] = None,
        data_loader: Optional[Any] = None, return_weekly_forecasts: bool = False
    ):
        # Use the bound model_type from closure
        return run_multi_horizon_forecast_neuralforecast(
            checkpoint_path, horizons, start_date, test_data, bound_model_type,
            target_series, data_loader, return_weekly_forecasts
        )
    return wrapper

MULTI_HORIZON_FORECAST_FUNCTIONS = {
    'dfm': run_multi_horizon_forecast_dfm,
    'ddfm': run_multi_horizon_forecast_ddfm,
    'patchtst': _create_multi_horizon_forecast_wrapper('patchtst'),
    'tft': _create_multi_horizon_forecast_wrapper('tft'),
    'itf': _create_multi_horizon_forecast_wrapper('itf'),
    'itransformer': _create_multi_horizon_forecast_wrapper('itf'),
    'timemixer': _create_multi_horizon_forecast_wrapper('timemixer'),
}


def _extract_model_config(model_config: Any) -> tuple[str, Any]:
    """Extract model name and config from Hydra config structure.
    
    Handles both flat (cfg.model.name) and nested (cfg.model.{model_name}) structures.
    
    Parameters
    ----------
    model_config : Any
        Model configuration object from Hydra
        
    Returns
    -------
    tuple[str, Any]
        (model_name, model_cfg) tuple
    """
    # Check for flat structure (has 'name' attribute)
    if hasattr(model_config, 'name'):
        return model_config.name.lower(), model_config
    
    # Try nested structure with known model names
    known_models = ['dfm', 'ddfm', 'itf', 'itransformer', 'patchtst', 'tft', 'timemixer']
    for key in known_models:
        if hasattr(model_config, key):
            return key.lower(), getattr(model_config, key)
    
    # Fallback: infer from dict keys (single key = model name)
    model_dict = OmegaConf.to_container(model_config, resolve=True)
    if isinstance(model_dict, dict) and len(model_dict) == 1:
        model_name = list(model_dict.keys())[0].lower()
        return model_name, model_config[model_name]
    
    raise ValueError(
        f"Could not determine model name from config structure. "
        f"Expected flat structure (has 'name') or nested structure (one of {known_models}), "
        f"but got keys: {list(model_dict.keys()) if isinstance(model_dict, dict) else 'non-dict'}"
    )

@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> None:
    """Main entry point for training and forecasting.
    
    Uses Hydra's nested config groups:
    - DFM with data variants: model=dfm/investment, model=dfm/production
    - Other models: model=ddfm, model=itf, etc. (uses model/{name}/default.yaml)
    - Override via CLI: python -m src.main model=dfm/investment data=investment
    """
    setup_logging(log_dir=get_project_root() / "log", force=True)
    
    # Get data model type from config (investment or production)
    data_model = cfg.get('data', 'investment').lower()
    
    # Get model name and config - handles flat and nested structures
    model_name, model_cfg = _extract_model_config(cfg.model)
    
    # Warn if DFM model variant doesn't match data type (helpful for debugging)
    if model_name == 'dfm':
        target_series = model_cfg.get('target_series', [])
        if target_series:
            if 'KOEQUIPTE' in target_series and data_model != 'investment':
                logger.warning(
                    f"DFM config targets KOEQUIPTE (investment) but data={data_model}. "
                    f"Consider using: model=dfm/investment data=investment"
                )
            elif 'KOIPALL.G' in target_series and data_model != 'production':
                logger.warning(
                    f"DFM config targets KOIPALL.G (production) but data={data_model}. "
                    f"Consider using: model=dfm/production data=production"
                )
    
    # Select appropriate data class based on config
    data_class_map = {
        'investment': InvestmentData,
        'production': ProductionData,
    }
    
    if data_model not in data_class_map:
        raise ValueError(
            f"Unknown data model: {data_model}. "
            f"Must be one of: {list(data_class_map.keys())}"
        )
    
    # Load preprocessed data (all preprocessing done automatically)
    logger.info(f"Loading and preprocessing {data_model} model data...")
    data_loader = data_class_map[data_model]()
    training_data = data_loader.training_data
    logger.info(f"Training data shape: {training_data.shape}")
    logger.info(f"Number of series: {len(training_data.columns)}")
    
    outputs_dir = get_project_root() / "checkpoints" / data_model / model_name
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {outputs_dir.resolve()}")
    
    # Training
    if cfg.train:
        logger.info(f"Training {model_name} model...")
        
        if model_name not in TRAIN_FUNCTIONS:
            raise ValueError(f"Unknown model: {model_name}")
        
        train_func = TRAIN_FUNCTIONS[model_name]
        train_func(
            model_type=model_name,
            cfg=cfg,
            data=training_data,
            model_name=model_name,
            outputs_dir=outputs_dir,
            model_params=OmegaConf.to_container(model_cfg, resolve=True),
            data_loader=data_loader
        )
    
    # Experiment mode handling
    # Note: For attention-based models, one trained model is used for experiments.
    # - Short-term: Model trained with prediction_length for 1-week ahead forecasting
    # - Long-term: Model trained with prediction_length >= max(horizons) can predict all horizons
    #   (4w, 8w, ..., 40w) by extracting appropriate time steps from full prediction output
    #
    # For factor models (DFM/DDFM):
    # - No prediction_length parameter - model structure defines forecast dynamics
    # - Short-term: Updates factors at each step, predicts 1 step ahead
    # - Long-term: Updates factors once, predicts with max horizon, extracts all horizons
    experiment_cfg = cfg.get('experiment', None)
    if experiment_cfg and experiment_cfg != "null":
        experiment_type = determine_experiment_type(experiment_cfg)
        logger.info(f"Running {experiment_type} experiment with {model_name} model...")
        
        # Load the trained model checkpoint.
        # One model is trained once and used for all experiments (short-term and long-term).
        # For long-term: model trained with prediction_length >= max(horizons) can predict all horizons.
        checkpoint_path = find_checkpoint_path(
            outputs_dir,
            error_msg=f"Model checkpoint not found at {outputs_dir}. "
                      f"Train the model first with train=true. "
                      f"For long-term experiments, ensure model is trained with prediction_length >= max(horizons)."
        )
        
        dataset_path = outputs_dir / "dataset.pkl"
        
        if experiment_type == "short_term":
            run_short_term_experiment(
                cfg, model_name, data_model, data_loader, 
                checkpoint_path, dataset_path, outputs_dir
            )
        elif experiment_type == "long_term":
            run_long_term_experiment(
                cfg, model_name, data_model, data_loader,
                checkpoint_path, dataset_path, outputs_dir
            )
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}. Must be 'short_term' or 'long_term'")
    
    # Forecasting
    elif cfg.forecast:
        logger.info(f"Forecasting with {model_name} model...")
        
        # Extract horizon for forecast mode
        horizon = model_cfg.get('horizon', cfg.get('horizon', 1))  # Default: 1 week for short-term experiments
        
        checkpoint_path = find_checkpoint_path(outputs_dir)
        
        # Import forecast functions when needed
        from src.forecast.dfm import forecast as forecast_dfm
        from src.forecast.ddfm import forecast as forecast_ddfm
        
        def _forecast_neuralforecast_wrapper(model_type: str):
            """Wrapper that binds model_type for NeuralForecast models."""
            def wrapper(checkpoint_path: Path, horizon: int, model_type: str = None):
                return forecast_neuralforecast(
                    checkpoint_path=checkpoint_path, 
                    horizon=horizon, 
                    model_type=model_type
                )
            return wrapper
        
        FORECAST_FUNCTIONS = {
            'patchtst': lambda **kwargs: forecast_neuralforecast(**kwargs, model_type='patchtst'),
            'tft': lambda **kwargs: forecast_neuralforecast(**kwargs, model_type='tft'),
            'itf': lambda **kwargs: forecast_neuralforecast(**kwargs, model_type='itf'),
            'itransformer': lambda **kwargs: forecast_neuralforecast(**kwargs, model_type='itf'),
            'timemixer': lambda **kwargs: forecast_neuralforecast(**kwargs, model_type='timemixer'),
            'dfm': forecast_dfm,
            'ddfm': forecast_ddfm,
        }
        
        if model_name not in FORECAST_FUNCTIONS:
            raise ValueError(f"Unknown model for forecasting: {model_name}")
        
        FORECAST_FUNCTIONS[model_name](
            checkpoint_path=checkpoint_path, horizon=horizon, model_type=model_name
        )


def run_short_term_experiment(
    cfg: DictConfig,
    model_name: str,
    data_model: str,
    data_loader: Any,
    checkpoint_path: Path,
    dataset_path: Path,
    outputs_dir: Path
) -> None:
    """Run short-term recursive forecasting experiment.
    
    Weekly recursive forecasting from start_date to end_date.
    
    For attention-based models (PatchTST, TFT, iTransformer, TimeMixer):
    - Optionally retrains model parameters with update_params=True
    - Or just moves cutoff forward with update_params=False
    
    For factor models (DFM, DDFM):
    - Always updates factors at each step (via model.update())
    - Factors are updated via Kalman filtering (DFM) or neural forward pass (DDFM)
    - Model parameters (A, C, Q, R) remain fixed - only factor state is updated
    - This is the key difference: factor models maintain and update latent factors
    """
    # Parse experiment config with defaults
    exp_cfg = cfg.get('experiment', {})
    config = parse_experiment_config(exp_cfg, "short_term")
    start_date = config['start_date']
    end_date = config['end_date']
    update_params = config['update_params']
    
    logger.info(f"Short-term experiment: {start_date} to {end_date}")
    
    # Load test data
    test_data = load_test_data(data_model)
    
    # Get target series from model config
    target_series = extract_target_series_from_config(cfg)
    
    # Run recursive forecast
    if model_name not in RECURSIVE_FORECAST_FUNCTIONS:
        raise ValueError(f"Unknown model: {model_name}")
    
    forecast_func = RECURSIVE_FORECAST_FUNCTIONS[model_name]
    
    # Factor models (dfm, ddfm) return 3 values; others return 4 (including actual_target_series)
    common_args = {
        'checkpoint_path': checkpoint_path,
        'test_data': test_data,
        'start_date': start_date,
        'end_date': end_date,
        'model_type': model_name,
        'target_series': target_series,
        'data_loader': data_loader,
    }
    
    if model_name in ['dfm', 'ddfm']:
        common_args['dataset_path'] = dataset_path
        predictions, actuals, dates = forecast_func(**common_args)
        actual_target_series = target_series
    else:
        common_args['update_params'] = update_params
        predictions, actuals, dates, actual_target_series = forecast_func(**common_args)
    
    # Identify monthly series for aggregation and metric normalization
    monthly_series = get_monthly_series_from_metadata(data_loader)
    
    # Compute test data standard deviation for consistent metric normalization
    # Aggregate monthly series to monthly before computing std (since evaluation is monthly)
    test_data_std = compute_test_data_std(test_data, actual_target_series, monthly_series=monthly_series)
    
    # Aggregate weekly forecasts to monthly using tent kernel weights [0.1, 0.2, 0.3, 0.4]
    # Only aggregate monthly series; weekly series remain weekly
    if monthly_series:
        logger.info(f"Found {len(monthly_series)} monthly series. Aggregating weekly forecasts to monthly.")
        predictions_monthly, dates_monthly = aggregate_weekly_to_monthly_tent_kernel(
            predictions, dates, actual_target_series, monthly_series=monthly_series
        )
        
        # Extract monthly actuals from test_data (simple monthly aggregation - no tent kernel)
        test_data_full = load_test_data(data_model)
        actuals_monthly, actuals_monthly_dates = extract_monthly_actuals(
            test_data_full, dates_monthly, actual_target_series, monthly_series=monthly_series
        )
        
        # Align predictions and actuals by date and compute metrics
        actuals_monthly_dates = pd.DatetimeIndex(actuals_monthly_dates)
        common_dates = actuals_monthly_dates.intersection(dates_monthly)
        
        if len(common_dates) > 0:
            pred_indices = [i for i, d in enumerate(dates_monthly) if d in common_dates]
            actual_indices = [i for i, d in enumerate(actuals_monthly_dates) if d in common_dates]
            
            predictions_aligned = predictions_monthly[pred_indices]
            actuals_aligned = actuals_monthly[actual_indices]
            
            _save_monthly_metrics(
                predictions_aligned, actuals_aligned, common_dates, actual_target_series,
                test_data_std, data_model, model_name, len(common_dates)
            )
        else:
            # Fallback to weekly if no monthly alignment possible
            logger.warning("No common dates found between monthly predictions and actuals. Saving weekly results.")
            _save_weekly_metrics(
                actuals, predictions, dates, actual_target_series,
                test_data_std, data_model, model_name
            )
    else:
        # No monthly series: use weekly data directly
        logger.info("No monthly series found. Using weekly forecasts directly.")
        _save_weekly_metrics(
            actuals, predictions, dates, actual_target_series,
            test_data_std, data_model, model_name
        )


def _extract_actuals_for_horizon(
    test_data: Optional[pd.DataFrame],
    target_series: List[str],
    start_date: str,
    horizon: int,
    monthly_series: set,
    logger: logging.Logger
) -> Tuple[Optional[np.ndarray], Optional[pd.Timestamp]]:
    """Extract actuals for a given horizon, handling monthly aggregation.
    
    Parameters
    ----------
    test_data : pd.DataFrame or None
        Full test dataset
    target_series : list
        Target series names
    start_date : str
        Start date (YYYY-MM-DD)
    horizon : int
        Horizon in weeks
    monthly_series : set
        Set of monthly series IDs
    logger : logging.Logger
        Logger instance
    
    Returns
    -------
    tuple
        (actuals_array or None, horizon_date or None)
    """
    if test_data is None or not target_series:
        return None, None
    
    start_ts = pd.Timestamp(start_date)
    horizon_date = start_ts + pd.Timedelta(weeks=horizon)
    available_targets = [t for t in target_series if t in test_data.columns]
    
    if not available_targets:
        return None, None
    
    # For monthly series: aggregate weekly actuals in the month by averaging
    if monthly_series:
        month_mask = (test_data.index.year == horizon_date.year) & \
                     (test_data.index.month == horizon_date.month)
        month_data = test_data[month_mask][available_targets]
        
        if len(month_data) == 0:
            logger.warning(f"No test data found in month {horizon_date.year}-{horizon_date.month:02d} for horizon {horizon}w")
            return None, None
        
        actuals_values = month_data.mean(axis=0)
        actuals = pd.to_numeric(actuals_values, errors='coerce').values.reshape(1, -1)
        month_end = pd.Timestamp(year=horizon_date.year, month=horizon_date.month, day=1) + pd.offsets.MonthEnd(0)
        return actuals, month_end
    
    # For weekly series: use horizon_date directly
    if horizon_date not in test_data.index:
        logger.warning(f"No test data found at {horizon_date.date()} for horizon {horizon}w")
        return None, None
    
    actuals_values = test_data.loc[horizon_date, available_targets]
    actuals = pd.to_numeric(actuals_values, errors='coerce').values.reshape(1, -1)
    return actuals, horizon_date


def _compute_and_save_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    dates: pd.DatetimeIndex,
    target_series: List[str],
    test_data_std: Optional[np.ndarray],
    output_dir: Path,
    period_type: str = "weeks"
) -> None:
    """Helper to compute and save metrics."""
    std_for_metrics = validate_std_for_metrics(test_data_std, len(target_series))
    smse = compute_smse(actuals, predictions, test_data_std=std_for_metrics)
    smae = compute_smae(actuals, predictions, test_data_std=std_for_metrics)
    metrics = {"smse": float(smse), "smae": float(smae)}
    logger.info(f"Short-term experiment metrics ({period_type}, {len(dates)} periods): sMSE={smse:.6f}, sMAE={smae:.6f}")
    
    save_experiment_results(
        output_dir=output_dir,
        predictions=predictions,
        actuals=actuals,
        dates=dates,
        target_series=target_series,
        metrics=metrics
    )


def _save_weekly_metrics(
    actuals: np.ndarray,
    predictions: np.ndarray,
    dates: pd.DatetimeIndex,
    target_series: List[str],
    test_data_std: Optional[np.ndarray],
    data_model: str,
    model_name: str
) -> None:
    """Helper to compute and save weekly metrics."""
    output_dir = get_project_root() / "outputs" / "short_term" / data_model / model_name
    _compute_and_save_metrics(
        predictions, actuals, dates, target_series, test_data_std, output_dir, "weeks"
    )


def _save_monthly_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    dates: pd.DatetimeIndex,
    target_series: List[str],
    test_data_std: Optional[np.ndarray],
    data_model: str,
    model_name: str,
    n_months: int
) -> None:
    """Helper to compute and save monthly metrics."""
    output_dir = get_project_root() / "outputs" / "short_term" / data_model / model_name
    _compute_and_save_metrics(
        predictions, actuals, dates, target_series, test_data_std, output_dir, "months"
    )


def run_long_term_experiment(
    cfg: DictConfig,
    model_name: str,
    data_model: str,
    data_loader: Any,
    checkpoint_path: Path,
    dataset_path: Path,
    outputs_dir: Path
) -> None:
    """Run long-term multi-horizon forecasting experiment.
    
    Fixed start point with multiple forecast horizons (4, 8, 12, ..., 40 weeks).
    
    For deep learning models (PatchTST, TFT, iTransformer, TimeMixer):
    - Uses ONE model trained with max horizon (prediction_length >= 40 weeks)
    - The same model predicts all horizons (4w, 8w, ..., 40w) by extracting
      the appropriate time step from the full prediction output
    - This matches the experiment design: train with max horizon, use for all horizons
    
    For DFM/DDFM (structural factor models):
    - Uses one model (trained once) - no horizon parameter in training
    - Updates factors once with data up to start_date (via model.update())
    - Predicts once with max(horizons) - the structural AR dynamics naturally define all horizons
    - Extracts different time steps from the single structural forecast
    - Key difference: Factor models update latent factors, not model parameters
    """
    # Parse experiment config with defaults
    exp_cfg = cfg.get('experiment', {})
    config = parse_experiment_config(exp_cfg, "long_term")
    start_date = config['start_date']
    horizons = config['horizons']
    
    logger.info(f"Long-term experiment: start_date={start_date}, horizons={horizons}")
    
    # Load test data
    test_data = load_test_data(data_model)
    
    # Get target series from model config
    target_series = extract_target_series_from_config(cfg)
    
    # Run multi-horizon forecast
    if model_name not in MULTI_HORIZON_FORECAST_FUNCTIONS:
        raise ValueError(f"Unknown model: {model_name}")
    
    forecast_func = MULTI_HORIZON_FORECAST_FUNCTIONS[model_name]
    
    common_args = {
        'checkpoint_path': checkpoint_path,
        'horizons': horizons,
        'start_date': start_date,
        'test_data': test_data,
        'target_series': target_series,
        'data_loader': data_loader,
    }
    
    if model_name in ['dfm', 'ddfm']:
        common_args['dataset_path'] = dataset_path
        common_args['model_type'] = model_name
        forecasts_dict = forecast_func(**common_args)
        actual_target_series = target_series
    else:
        result = forecast_func(**common_args)
        forecasts_dict, actual_target_series = result
    
    # Identify monthly series and compute test data std
    monthly_series = get_monthly_series_from_metadata(data_loader)
    test_data_std = compute_test_data_std(test_data, actual_target_series, monthly_series=monthly_series)
    
    # For monthly series, re-run forecasts with return_weekly_forecasts=True if needed
    if monthly_series and model_name in ['patchtst', 'tft', 'itransformer', 'itf', 'timemixer']:
        logger.info(f"Found {len(monthly_series)} monthly series. Re-running forecasts to get weekly values for aggregation.")
        forecast_func = MULTI_HORIZON_FORECAST_FUNCTIONS[model_name]
        forecasts_dict, _ = forecast_func(
            checkpoint_path=checkpoint_path,
            horizons=horizons,
            start_date=start_date,
            test_data=test_data,
            target_series=target_series,
            data_loader=data_loader,
            return_weekly_forecasts=True
        )
    
    # Save results for each horizon
    base_output_dir = get_project_root() / "outputs" / "long_term" / data_model / model_name
    
    for horizon, forecast_data in forecasts_dict.items():
        horizon_dir = base_output_dir / f"horizon_{horizon}w"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract actuals if available
        actuals, horizon_date = _extract_actuals_for_horizon(
            test_data, actual_target_series, start_date, horizon, monthly_series, logger
        )
        
        # Handle forecast aggregation for monthly series
        if monthly_series and isinstance(forecast_data, dict) and 'weekly_forecasts' in forecast_data:
            # Aggregate weekly forecasts to monthly using tent kernel
            weekly_forecasts = forecast_data['weekly_forecasts']
            weekly_dates = forecast_data['dates']
            
            if len(weekly_forecasts) > 0:
                aggregated_forecasts, aggregated_dates = aggregate_weekly_to_monthly_tent_kernel(
                    weekly_forecasts, weekly_dates, actual_target_series, 
                    monthly_series=monthly_series
                )
                
                if len(aggregated_forecasts) > 0:
                    forecast_values = aggregated_forecasts[0]  # First (and should be only) month
                    horizon_date = aggregated_dates[0] if len(aggregated_dates) > 0 else horizon_date
                else:
                    logger.warning(f"No aggregated forecasts for horizon {horizon}w")
                    forecast_values = np.full(len(actual_target_series), np.nan)
            else:
                logger.warning(f"No weekly forecasts for horizon {horizon}w")
                forecast_values = np.full(len(actual_target_series), np.nan)
        else:
            # Single forecast value (weekly series or no weekly forecasts available)
            forecast_values = forecast_data if isinstance(forecast_data, np.ndarray) else np.array(forecast_data)
        
        # Compute metrics if actuals available
        metrics = None
        if actuals is not None:
            forecast_array = np.asarray(forecast_values).reshape(1, -1)
            actuals_array = np.asarray(actuals)
            std_for_metrics = validate_std_for_metrics(test_data_std, len(actual_target_series))
            smse = compute_smse(actuals_array, forecast_array, test_data_std=std_for_metrics)
            smae = compute_smae(actuals_array, forecast_array, test_data_std=std_for_metrics)
            metrics = {"smse": float(smse), "smae": float(smae)}
            logger.info(f"Horizon {horizon}w: sMSE={smse:.6f}, sMAE={smae:.6f}")
        
        # Save results using actual target series
        save_experiment_results(
            output_dir=horizon_dir,
            predictions=forecast_values.reshape(1, -1),
            actuals=actuals,
            dates=pd.DatetimeIndex([horizon_date]) if horizon_date is not None else None,
            target_series=actual_target_series,
            metrics=metrics
        )


if __name__ == "__main__":
    main()
