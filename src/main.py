from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Any, List, Optional, Tuple
import hydra
import logging
import pandas as pd
import numpy as np
import joblib

from src.preprocess import InvestmentData, ProductionData
from src.utils import setup_logging, get_project_root
from src.train.dfm import train_dfm_model
from src.train.ddfm import train_ddfm_model
from src.train.mamba import train_mamba_model
from src.utils import (
    load_test_data,
    get_monthly_series_from_metadata,
    aggregate_weekly_to_monthly_tent_kernel,
    extract_monthly_actuals,
    inverse_transform_predictions,
    apply_inverse_transformations_with_accumulation,
    _extract_last_value_in_month,
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
from src.forecast.mamba import (
    run_recursive_forecast as run_recursive_forecast_mamba,
    run_multi_horizon_forecast as run_multi_horizon_forecast_mamba,
    forecast as forecast_mamba
)

logger = logging.getLogger(__name__)

# Optional NeuralForecast-based models (PatchTST/TFT/iTransformer/TimeMixer)
# Keep these optional so running DFM/DDFM/Mamba does not require neuralforecast installed.
def _missing_optional_dependency(model_name: str, dependency: str, err: Exception):
    raise ImportError(
        f"Model '{model_name}' requires optional dependency '{dependency}', "
        f"but it is not available in this environment."
    ) from err

try:
    from src.train.patchtst import train_patchtst_model
    from src.train.tft import train_tft_model
    from src.train.itf import train_itf_model
    from src.train.timemixer import train_timemixer_model
    from src.forecast.neuralforecast import (
        run_recursive_forecast as run_recursive_forecast_neuralforecast,
        run_multi_horizon_forecast as run_multi_horizon_forecast_neuralforecast,
        forecast as forecast_neuralforecast,
    )
    _HAS_NEURALFORECAST = True
except Exception as e:
    _HAS_NEURALFORECAST = False
    _NEURALFORECAST_IMPORT_ERROR = e
    train_patchtst_model = None
    train_tft_model = None
    train_itf_model = None
    train_timemixer_model = None
    run_recursive_forecast_neuralforecast = None
    run_multi_horizon_forecast_neuralforecast = None
    forecast_neuralforecast = None

# Model function mappings for cleaner code
TRAIN_FUNCTIONS = {
    'dfm': train_dfm_model,
    'ddfm': train_ddfm_model,
    'timemixer': train_timemixer_model if _HAS_NEURALFORECAST else (lambda *a, **k: _missing_optional_dependency("timemixer", "neuralforecast", _NEURALFORECAST_IMPORT_ERROR)),
    'patchtst': train_patchtst_model if _HAS_NEURALFORECAST else (lambda *a, **k: _missing_optional_dependency("patchtst", "neuralforecast", _NEURALFORECAST_IMPORT_ERROR)),
    'tft': train_tft_model if _HAS_NEURALFORECAST else (lambda *a, **k: _missing_optional_dependency("tft", "neuralforecast", _NEURALFORECAST_IMPORT_ERROR)),
    'itf': train_itf_model if _HAS_NEURALFORECAST else (lambda *a, **k: _missing_optional_dependency("itf", "neuralforecast", _NEURALFORECAST_IMPORT_ERROR)),
    'itransformer': train_itf_model if _HAS_NEURALFORECAST else (lambda *a, **k: _missing_optional_dependency("itransformer", "neuralforecast", _NEURALFORECAST_IMPORT_ERROR)),
    'mamba': train_mamba_model,
}

def _create_neuralforecast_wrapper(bound_model_type: str, is_recursive: bool = True):
    """Create a wrapper that binds model_type for NeuralForecast models."""
    if is_recursive:
        def wrapper(
            checkpoint_path: Path, test_data: pd.DataFrame, start_date: str, end_date: str,
            model_type: str, target_series: Optional[List[str]] = None,
            data_loader: Optional[Any] = None, update_params: bool = False
        ):
            if not _HAS_NEURALFORECAST or run_recursive_forecast_neuralforecast is None:
                return _missing_optional_dependency(bound_model_type, "neuralforecast", _NEURALFORECAST_IMPORT_ERROR)
            return run_recursive_forecast_neuralforecast(
                checkpoint_path, test_data, start_date, end_date, bound_model_type,
                target_series, data_loader, update_params
            )
    else:
        def wrapper(
            checkpoint_path: Path, horizons: List[int], start_date: str,
            test_data: Optional[pd.DataFrame] = None, target_series: Optional[List[str]] = None,
            data_loader: Optional[Any] = None, return_weekly_forecasts: bool = False
        ):
            if not _HAS_NEURALFORECAST or run_multi_horizon_forecast_neuralforecast is None:
                return _missing_optional_dependency(bound_model_type, "neuralforecast", _NEURALFORECAST_IMPORT_ERROR)
            return run_multi_horizon_forecast_neuralforecast(
                checkpoint_path, horizons, start_date, test_data, bound_model_type,
                target_series, data_loader, return_weekly_forecasts
            )
    return wrapper

RECURSIVE_FORECAST_FUNCTIONS = {
    'dfm': run_recursive_forecast_dfm,
    'ddfm': run_recursive_forecast_ddfm,
    'mamba': run_recursive_forecast_mamba,
}

MULTI_HORIZON_FORECAST_FUNCTIONS = {
    'dfm': run_multi_horizon_forecast_dfm,
    'ddfm': run_multi_horizon_forecast_ddfm,
    'mamba': run_multi_horizon_forecast_mamba,
}

# Add NeuralForecast model wrappers
if _HAS_NEURALFORECAST:
    for model_type in ['patchtst', 'tft', 'itf', 'itransformer', 'timemixer']:
        bound_type = 'itf' if model_type == 'itransformer' else model_type
        RECURSIVE_FORECAST_FUNCTIONS[model_type] = _create_neuralforecast_wrapper(bound_type, is_recursive=True)
        MULTI_HORIZON_FORECAST_FUNCTIONS[model_type] = _create_neuralforecast_wrapper(bound_type, is_recursive=False)


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
    known_models = ['dfm', 'ddfm', 'itf', 'itransformer', 'patchtst', 'tft', 'timemixer', 'mamba']
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
    
    # Get model name and config - handles flat and nested structures
    model_name, model_cfg = _extract_model_config(cfg.model)
    
    # Get data model type from config (investment or production)
    data_model = cfg.get('data', 'investment').lower()
    logger.info(f"Initial data_model from config: {data_model}")
    
    # CRITICAL: Auto-correct data_model based on target_series if there's a mismatch
    # This handles the case where model=dfm/production is used but data=investment (default)
    if model_name == 'dfm':
        target_series = model_cfg.get('target_series', [])
        if target_series:
            if 'KOIPALL.G' in target_series and data_model != 'production':
                logger.warning(f"Auto-correcting data_model from {data_model} to production (target_series contains KOIPALL.G)")
                data_model = 'production'
            elif 'KOEQUIPTE' in target_series and data_model != 'investment':
                logger.warning(f"Auto-correcting data_model from {data_model} to investment (target_series contains KOEQUIPTE)")
                data_model = 'investment'
    
    # Warn if DFM model variant doesn't match data type
    if model_name == 'dfm':
        target_series = model_cfg.get('target_series', [])
        expected_data = {'KOEQUIPTE': 'investment', 'KOIPALL.G': 'production'}
        for target, expected in expected_data.items():
            if target in target_series and data_model != expected:
                logger.warning(
                    f"DFM config targets {target} ({expected}) but data={data_model}. "
                    f"Consider using: model=dfm/{expected} data={expected}"
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
        
        # Dataset metadata is now stored in model.pkl, so dataset.pkl is no longer needed
        if experiment_type == "short_term":
            run_short_term_experiment(
                cfg, model_name, data_model, data_loader, 
                checkpoint_path, outputs_dir
            )
        elif experiment_type == "long_term":
            run_long_term_experiment(
                cfg, model_name, data_model, data_loader,
                checkpoint_path, outputs_dir
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
            'mamba': forecast_mamba,
            'dfm': forecast_dfm,
            'ddfm': forecast_ddfm,
        }
        
        if model_name not in FORECAST_FUNCTIONS:
            raise ValueError(f"Unknown model for forecasting: {model_name}")
        
        # Call forecast function (model_type already bound for neuralforecast models)
        forecast_kwargs = {'checkpoint_path': checkpoint_path, 'horizon': horizon}
        if model_name not in ['patchtst', 'tft', 'itf', 'itransformer', 'timemixer']:
            forecast_kwargs['model_type'] = model_name
        # Pass data_loader and window_size to DFM/DDFM for latest data context
        if model_name in ['dfm', 'ddfm']:
            forecast_kwargs['data_loader'] = data_loader
            forecast_kwargs['window_size'] = model_cfg.get('window_size', 52)  # Default: 52 weeks
        FORECAST_FUNCTIONS[model_name](**forecast_kwargs)


def run_short_term_experiment(
    cfg: DictConfig,
    model_name: str,
    data_model: str,
    data_loader: Any,
    checkpoint_path: Path,
    outputs_dir: Path
) -> None:
    """Run short-term recursive forecasting experiment.
    
    Weekly recursive forecasting from start_date to end_date.
    All models follow consistent pipeline: predictions → inverse transform → aggregate (if monthly) → metrics.
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
    
    # Prepare common forecast arguments
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
        # Remove target_series from args (DDFM/DFM don't accept it - they infer from dataset/covariates)
        forecast_args = {k: v for k, v in common_args.items() if k != 'target_series'}
        predictions, actuals, dates = forecast_func(**forecast_args)
        # Get target_series from model checkpoint metadata if not in config
        if not target_series:
            try:
                if model_name == 'dfm':
                    from dfm_python import DFM
                    model = DFM.load(checkpoint_path)
                    if hasattr(model, '_checkpoint_metadata') and model._checkpoint_metadata:
                        metadata = model._checkpoint_metadata.get('dataset_metadata')
                        if metadata and metadata.get('target_series'):
                            target_series = metadata['target_series']
                elif model_name == 'ddfm':
                    from dfm_python import DDFM
                    model = DDFM.load(checkpoint_path, dataset=None)
                    if hasattr(model, '_checkpoint_metadata') and model._checkpoint_metadata:
                        metadata = model._checkpoint_metadata.get('dataset_metadata')
                        if metadata and metadata.get('target_series'):
                            target_series = metadata['target_series']
                    logger.info(f"Using target_series from dataset: {len(target_series)} series")
            except Exception as e:
                logger.warning(f"Could not extract target_series from dataset: {e}")
        # For DFM/DDFM: predictions contain all series, but we should filter to target_series for evaluation
        # Fallback: infer from predictions shape only if target_series not specified
        if not target_series and predictions is not None and len(predictions) > 0 and test_data is not None:
            # Get all numeric columns (exclude date columns)
            numeric_cols = [c for c in test_data.columns if c not in ['date', 'date_w', 'year', 'month', 'day']]
            # Match predictions shape
            n_series = predictions.shape[1] if predictions.ndim > 1 else 1
            target_series = numeric_cols[:n_series] if len(numeric_cols) >= n_series else numeric_cols
            logger.warning(f"Inferred {len(target_series)} target series from test_data columns (predictions shape: {predictions.shape})")
        
        # Use target_series from config for evaluation (even though DFM predicts all series)
        actual_target_series = target_series if target_series else []
        
        # For DFM/DDFM: filter predictions/actuals to target_series if specified
        # (Factor models predict all series, but we evaluate only on target_series)
        if model_name in ['dfm', 'ddfm'] and actual_target_series and predictions is not None and test_data is not None:
            # CRITICAL: Predictions are in training column order, NOT test_data column order
            # Get training column order from model to correctly map target_series to prediction indices
            date_cols = ['date', 'date_w', 'year', 'month', 'day']
            training_numeric_cols = None
            
            try:
                if model_name == 'dfm':
                    from dfm_python import DFM
                    from src.forecast.dfm import _get_training_column_order
                    model = DFM.load(checkpoint_path)
                    training_column_order = _get_training_column_order(model, test_data, logger)
                    if training_column_order:
                        training_numeric_cols = [c for c in training_column_order if c not in date_cols]
                elif model_name == 'ddfm':
                    from dfm_python import DDFM
                    model = DDFM.load(checkpoint_path, dataset=None)
                    if hasattr(model, 'scaler') and model.scaler is not None:
                        if hasattr(model.scaler, 'feature_names_in_') and model.scaler.feature_names_in_ is not None:
                            training_numeric_cols = [c for c in model.scaler.feature_names_in_ if c not in date_cols]
            except Exception as e:
                logger.warning(f"Could not get training column order from model: {e}. Falling back to test_data column order.")
            
            # Use training column order if available, otherwise fall back to test_data
            if training_numeric_cols:
                prediction_cols = training_numeric_cols
                logger.debug(f"Using training column order ({len(prediction_cols)} series) to filter predictions")
            else:
                # Fallback: use test_data column order (may cause wrong filtering if order differs)
                prediction_cols = [c for c in test_data.columns if c not in date_cols and pd.api.types.is_numeric_dtype(test_data[c])]
                logger.warning(f"Using test_data column order to filter predictions (may be incorrect if order differs from training)")
            
            # Find indices of target_series in predictions (using correct column order)
            target_indices = []
            for ts in actual_target_series:
                if ts in prediction_cols:
                    target_indices.append(prediction_cols.index(ts))
                else:
                    logger.warning(f"Target series {ts} not found in prediction columns")
            
            if target_indices and len(target_indices) == len(actual_target_series):
                # Filter predictions and actuals to target_series columns
                if predictions.ndim > 1 and predictions.shape[1] > len(target_indices):
                    # #region agent log
                    if 'KOEQUIPTE' in actual_target_series or 'KOIPALL.G' in actual_target_series:
                        koequipte_idx = actual_target_series.index('KOEQUIPTE') if 'KOEQUIPTE' in actual_target_series else None
                        koipall_idx = actual_target_series.index('KOIPALL.G') if 'KOIPALL.G' in actual_target_series else None
                        target_idx_for_koequipte = target_indices[koequipte_idx] if koequipte_idx is not None else None
                        logger.info(f"DEBUG: Filtering predictions - KOEQUIPTE at index {target_idx_for_koequipte} in predictions (target_series index: {koequipte_idx})")
                        logger.info(f"DEBUG: Prediction column order (first 5): {prediction_cols[:5]}")
                        logger.info(f"DEBUG: Predictions shape before filtering: {predictions.shape}, first 3 values for KOEQUIPTE index: {predictions[0, target_idx_for_koequipte] if target_idx_for_koequipte is not None and target_idx_for_koequipte < predictions.shape[1] else 'N/A'}")
                    # #endregion
                    predictions = predictions[:, target_indices]
                    logger.info(f"Filtered DFM predictions to {len(actual_target_series)} target series: {actual_target_series}")
                    # #region agent log
                    if 'KOEQUIPTE' in actual_target_series:
                        koequipte_idx_in_filtered = actual_target_series.index('KOEQUIPTE')
                        logger.info(f"DEBUG: After filtering - KOEQUIPTE at index {koequipte_idx_in_filtered}, first 3 values: {predictions[:3, koequipte_idx_in_filtered].tolist()}")
                    # #endregion
                if actuals is not None and actuals.ndim > 1 and actuals.shape[1] > len(target_indices):
                    # For actuals, use test_data column order (actuals come from test_data)
                    test_numeric_cols = [c for c in test_data.columns if c not in date_cols and pd.api.types.is_numeric_dtype(test_data[c])]
                    actuals_target_indices = [test_numeric_cols.index(ts) for ts in actual_target_series if ts in test_numeric_cols]
                    if len(actuals_target_indices) == len(actual_target_series):
                        actuals = actuals[:, actuals_target_indices]
                        logger.info(f"Filtered DFM actuals to {len(actual_target_series)} target series")
            elif not target_indices:
                logger.warning(f"None of target_series {actual_target_series} found in predictions. Using all series.")
                # Fallback: use all series
                actual_target_series = prediction_cols[:predictions.shape[1]] if predictions.ndim > 1 else prediction_cols
        elif predictions is not None and test_data is not None:
            # For non-DFM models: ensure shape matches
            n_series = predictions.shape[1] if predictions.ndim > 1 else 1
            if len(actual_target_series) != n_series:
                # Get all numeric columns from test_data
                numeric_cols = [c for c in test_data.columns if c not in ['date', 'date_w', 'year', 'month', 'day']]
                if len(numeric_cols) >= n_series:
                    actual_target_series = numeric_cols[:n_series]
                    logger.warning(f"Adjusted target_series to match predictions shape: {len(actual_target_series)} series from test_data")
                else:
                    logger.error(f"Cannot match predictions shape {n_series} with available columns {len(numeric_cols)}")
                    actual_target_series = numeric_cols  # Use what we have
    else:
        common_args['update_params'] = update_params
        predictions, actuals, dates, actual_target_series = forecast_func(**common_args)
    
    # Apply inverse transformations to all models (handles chg/logdiff consistently)
    # Pipeline: Model predictions → Inverse transform (chg/logdiff → levels) → Aggregate to monthly (if needed) → Compare with actuals
    # CRITICAL: For DFM/DDFM, predictions are in TRANSFORMED space (chg/logdiff) and MUST be accumulated
    #   - DFM/DDFM scaler is fitted on transformed data (chg), so inverse transform returns chg values
    #   - We MUST apply accumulation to convert chg → levels
    # CRITICAL: For attention-based models (PatchTST/TFT/iTransformer), predictions are already in ORIGINAL levels (no transformations applied)
    #   - These models use original data (no preprocessing transformations)
    #   - Model outputs are already in original scale
    #   - We should NOT apply inverse transformations
    if data_loader is not None and actual_target_series is not None and len(actual_target_series) > 0:
        # Skip inverse transformation for attention-based models and Mamba - they already output in original levels
        # DFM/DDFM need accumulation (they return chg values)
        if model_name in ['patchtst', 'tft', 'itf', 'itransformer', 'timemixer', 'mamba']:
            logger.info(f"Skipping inverse transformation for {model_name} - predictions already in original levels")
        else:
            # Ensure predictions shape matches target_series length
            if predictions is not None and predictions.ndim > 1:
                if predictions.shape[1] != len(actual_target_series):
                    predictions = predictions[:, :len(actual_target_series)]
                    logger.warning(f"Truncated predictions to match target_series length: {predictions.shape}")
            
            try:
                # Get experiment start date for proper initial level lookup in accumulation
                exp_cfg = cfg.get('experiment', {})
                config = parse_experiment_config(exp_cfg, "short_term")
                experiment_start_date = pd.Timestamp(config['start_date']) if 'start_date' in config else None
                
                predictions = apply_inverse_transformations_with_accumulation(
                    predictions, pd.DatetimeIndex(dates), actual_target_series, data_loader, test_data,
                    experiment_start_date=experiment_start_date
                )
                
                logger.info(f"Applied inverse transformations to {model_name} predictions (transformed space → raw levels).")
            except Exception as e:
                logger.error(
                    f"Failed to inverse-transform predictions for {model_name}. "
                    f"Error: {e}",
                    exc_info=True
                )
                # Don't proceed with raw outputs - this will cause wrong metrics
                raise RuntimeError(f"Inverse transformation failed for {model_name}: {e}") from e

    # Identify monthly series for aggregation and metric normalization
    monthly_series = get_monthly_series_from_metadata(data_loader) if data_loader is not None else set()
    
    # Compute test data std for metric normalization (filter to experiment period)
    test_data_for_std = test_data[
        (test_data.index >= pd.Timestamp(start_date)) & 
        (test_data.index <= pd.Timestamp(end_date))
    ] if test_data is not None else None
    test_data_std = compute_test_data_std(test_data_for_std, actual_target_series, monthly_series=monthly_series)
    
    # Aggregate weekly forecasts to monthly (inverse transform already done above)
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
        
        output_dir = get_project_root() / "outputs" / "short_term" / data_model / model_name
        
        if len(common_dates) > 0:
            pred_indices = [i for i, d in enumerate(dates_monthly) if d in common_dates]
            actual_indices = [i for i, d in enumerate(actuals_monthly_dates) if d in common_dates]
            
            predictions_aligned = predictions_monthly[pred_indices]
            actuals_aligned = actuals_monthly[actual_indices]
            
            # Ensure predictions and actuals have same shape (truncate predictions if needed)
            if predictions_aligned.shape[1] != actuals_aligned.shape[1]:
                min_cols = min(predictions_aligned.shape[1], actuals_aligned.shape[1])
                predictions_aligned = predictions_aligned[:, :min_cols]
                actuals_aligned = actuals_aligned[:, :min_cols]
                actual_target_series = actual_target_series[:min_cols] if len(actual_target_series) > min_cols else actual_target_series
                logger.warning(f"Truncated predictions/actuals to match shape: {predictions_aligned.shape}")
            
            # Use test_data_std computed from full test period for normalization (not actuals_std)
            # This ensures consistent normalization across all periods, not just the aligned subset
            # test_data_std was already computed above from the full test period
            if test_data_std is None or len(test_data_std) != len(actual_target_series):
                # Fallback: use actuals std if test_data_std is not available
                actuals_std = np.std(actuals_aligned, axis=0, ddof=0) if actuals_aligned.size > 0 else None
                if actuals_std is not None and len(actuals_std) == len(actual_target_series):
                    test_data_std = actuals_std
                    logger.warning("Using actuals_std as fallback - test_data_std not available")
            
            _save_metrics(predictions_aligned, actuals_aligned, common_dates, actual_target_series, test_data_std, output_dir, "months")
        else:
            logger.warning("No common dates found between monthly predictions and actuals. Saving weekly results.")
            _save_metrics(predictions, actuals, dates, actual_target_series, test_data_std, output_dir, "weeks")
    else:
        # No monthly series: use weekly data directly
        logger.info("No monthly series found. Using weekly forecasts directly.")
        output_dir = get_project_root() / "outputs" / "short_term" / data_model / model_name
        
        # Ensure predictions and actuals have same shape (truncate predictions if needed)
        if predictions is not None and actuals is not None:
            if predictions.ndim > 1 and actuals.ndim > 1:
                if predictions.shape[1] != actuals.shape[1]:
                    min_cols = min(predictions.shape[1], actuals.shape[1])
                    predictions = predictions[:, :min_cols]
                    actuals = actuals[:, :min_cols]
                    actual_target_series = actual_target_series[:min_cols] if len(actual_target_series) > min_cols else actual_target_series
                    logger.warning(f"Truncated predictions/actuals to match shape: {predictions.shape}")
        
        # Use test_data_std computed from full test period for normalization (not actuals_std)
        # This ensures consistent normalization across all periods
        # test_data_std was already computed above from the full test period
        if test_data_std is None or len(test_data_std) != len(actual_target_series):
            # Fallback: use actuals std if test_data_std is not available
            if actuals is not None and len(actuals) > 0:
                actuals_std = np.std(actuals, axis=0, ddof=0) if actuals.ndim > 1 else np.array([np.std(actuals, ddof=0)])
                if actuals_std is not None and len(actuals_std) == len(actual_target_series):
                    test_data_std = actuals_std
                    logger.warning("Using actuals_std as fallback - test_data_std not available (weekly)")
        
        _save_metrics(predictions, actuals, dates, actual_target_series, test_data_std, output_dir, "weeks")


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
    
    # For monthly series: extract last non-NaN value in the month (consistent with short-term)
    if monthly_series:
        month_mask = (test_data.index.year == horizon_date.year) & \
                     (test_data.index.month == horizon_date.month)
        month_data = test_data[month_mask][available_targets]
        
        if len(month_data) == 0:
            # Try to find data in adjacent months (within 1 month before/after)
            # This handles edge cases where forecast date is at month boundary
            for month_offset in [-1, 1]:
                adj_date = horizon_date + pd.DateOffset(months=month_offset)
                adj_mask = (test_data.index.year == adj_date.year) & \
                          (test_data.index.month == adj_date.month)
                adj_data = test_data[adj_mask][available_targets]
                if len(adj_data) > 0:
                    month_data = adj_data
                    horizon_date = adj_date  # Update horizon_date to match found month
                    logger.debug(f"Using data from adjacent month {adj_date.year}-{adj_date.month:02d} for horizon {horizon}w")
                    break
            
            if len(month_data) == 0:
                logger.warning(f"No test data found in month {horizon_date.year}-{horizon_date.month:02d} (or adjacent months) for horizon {horizon}w")
                return None, None
        
        # Extract last non-NaN value per series in the month
        actuals_values = _extract_last_value_in_month(month_data, available_targets)
        actuals = actuals_values.reshape(1, -1)
        month_end = pd.Timestamp(year=horizon_date.year, month=horizon_date.month, day=1) + pd.offsets.MonthEnd(0)
        return actuals, month_end
    
    # For weekly series: find nearest date within tolerance (3 days)
    # This handles cases where forecast dates don't exactly match test data dates
    if horizon_date not in test_data.index:
        # Find nearest date within 3 days tolerance
        date_diff = (test_data.index - horizon_date).abs()
        nearest_idx = date_diff.idxmin()
        nearest_diff = date_diff.loc[nearest_idx]
        
        if nearest_diff <= pd.Timedelta(days=3):
            # Use nearest date if within tolerance
            horizon_date = nearest_idx
            logger.debug(f"Using nearest date {horizon_date.date()} (diff: {nearest_diff.days} days) for horizon {horizon}w")
        else:
            logger.warning(f"No test data found near {horizon_date.date()} (nearest: {nearest_idx.date()}, diff: {nearest_diff.days} days) for horizon {horizon}w")
            return None, None
    
    actuals_values = test_data.loc[horizon_date, available_targets]
    actuals = pd.to_numeric(actuals_values, errors='coerce').values.reshape(1, -1)
    return actuals, horizon_date


def _save_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    dates: pd.DatetimeIndex,
    target_series: List[str],
    test_data_std: Optional[np.ndarray],
    output_dir: Path,
    period_type: str = "weeks"
) -> None:
    """Compute and save metrics for predictions and actuals."""
    std_for_metrics = validate_std_for_metrics(test_data_std, len(target_series))
    smse = compute_smse(actuals, predictions, test_data_std=std_for_metrics)
    smae = compute_smae(actuals, predictions, test_data_std=std_for_metrics)
    metrics = {"smse": float(smse), "smae": float(smae)}
    logger.info(f"Metrics ({period_type}, {len(dates)} periods): sMSE={smse:.6f}, sMAE={smae:.6f}")
    
    save_experiment_results(
        output_dir=output_dir,
        predictions=predictions,
        actuals=actuals,
        dates=dates,
        target_series=target_series,
        metrics=metrics
    )


def run_long_term_experiment(
    cfg: DictConfig,
    model_name: str,
    data_model: str,
    data_loader: Any,
    checkpoint_path: Path,
    outputs_dir: Path
) -> None:
    """Run long-term multi-horizon forecasting experiment.
    
    Fixed start point with multiple forecast horizons (4, 8, 12, ..., 40 weeks).
    All models follow consistent pipeline: predictions → inverse transform → aggregate (if monthly) → metrics.
    """
    # Parse experiment config with defaults
    exp_cfg = cfg.get('experiment', {})
    config = parse_experiment_config(exp_cfg, "long_term")
    start_date = config['start_date']
    horizons = config['horizons']
    
    logger.info(f"Long-term experiment: start_date={start_date}, horizons={horizons}")
    
    # Load test data
    test_data = load_test_data(data_model)
    
    # Mamba uses raw-level data with internal scaling; keep test_data as-is
    
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
        common_args['model_type'] = model_name
        # Remove target_series from args (DDFM/DFM don't accept it - they infer from dataset/covariates)
        forecast_args = {k: v for k, v in common_args.items() if k != 'target_series'}
        forecasts_dict = forecast_func(**forecast_args)
        actual_target_series = target_series
    else:
        result = forecast_func(**common_args)
        forecasts_dict, actual_target_series = result
    
    # Identify monthly series and compute test data std
    monthly_series = get_monthly_series_from_metadata(data_loader)
    test_data_std = compute_test_data_std(test_data, actual_target_series, monthly_series=monthly_series)
    
    # For monthly series, re-run forecasts with return_weekly_forecasts=True if needed
    if monthly_series and model_name in ['patchtst', 'tft', 'itransformer', 'itf', 'timemixer', 'mamba', 'ddfm']:
        logger.info(f"Found {len(monthly_series)} monthly series. Re-running forecasts to get weekly values for aggregation.")
        forecast_func = MULTI_HORIZON_FORECAST_FUNCTIONS[model_name]
        forecast_args = {
            'checkpoint_path': checkpoint_path,
            'horizons': horizons,
            'start_date': start_date,
            'test_data': test_data,
            'data_loader': data_loader,
            'return_weekly_forecasts': True
        }
        # Add target_series only for non-DFM/DDFM models
        if model_name not in ['dfm', 'ddfm']:
            forecast_args['target_series'] = target_series
        if model_name in ['dfm', 'ddfm']:
            forecast_args['model_type'] = model_name
            forecasts_dict = forecast_func(**forecast_args)
        else:
            forecasts_dict, _ = forecast_func(**forecast_args)
    
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
                aggregation_targets = forecast_data.get('_available_targets', actual_target_series)
                primary_target = forecast_data.get('_primary_target', actual_target_series[0] if actual_target_series else None)

                # Apply inverse transformations before monthly aggregation
                # CRITICAL: DFM/DDFM need accumulation (they return chg values)
                if data_loader is not None:
                    try:
                        # Get experiment start date for proper initial level lookup in accumulation
                        exp_cfg = cfg.get('experiment', {})
                        config = parse_experiment_config(exp_cfg, "long_term")
                        experiment_start_date = pd.Timestamp(config['start_date']) if 'start_date' in config else None
                        
                        weekly_forecasts = apply_inverse_transformations_with_accumulation(
                            weekly_forecasts, pd.DatetimeIndex(weekly_dates), aggregation_targets, data_loader, test_data,
                            experiment_start_date=experiment_start_date
                        )
                    except Exception as e:
                        logger.warning(f"Failed to inverse-transform weekly forecasts for {model_name} (horizon {horizon}w): {e}")
                
                aggregated_forecasts, aggregated_dates = aggregate_weekly_to_monthly_tent_kernel(
                    weekly_forecasts, weekly_dates, aggregation_targets, monthly_series=monthly_series
                )
                
                if len(aggregated_forecasts) > 0:
                    primary_idx = aggregation_targets.index(primary_target) if (primary_target and primary_target in aggregation_targets) else 0
                    forecast_values = aggregated_forecasts[0, primary_idx:primary_idx+1]
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
            
            # For DFM/DDFM: filter forecast to target_series BEFORE inverse transform
            # (Factor models predict all series, but we need only target_series for inverse transform)
            if model_name in ['dfm', 'ddfm'] and actual_target_series and forecast_values is not None:
                forecast_array = np.asarray(forecast_values)
                # If forecast has more series than targets, filter to target series
                if forecast_array.ndim > 0 and len(forecast_array) > len(actual_target_series):
                    # Forecast has all series, need to filter to target_series
                    # Get training column order to map target_series to forecast indices
                    if data_loader is not None and hasattr(data_loader, 'processed') and data_loader.processed is not None:
                        # Use processed data columns to find target series indices
                        processed_cols = [c for c in data_loader.processed.columns if pd.api.types.is_numeric_dtype(data_loader.processed[c])]
                        target_indices = [processed_cols.index(ts) for ts in actual_target_series if ts in processed_cols]
                        if len(target_indices) == len(actual_target_series) and max(target_indices) < len(forecast_array):
                            forecast_values = forecast_array[target_indices]
                            logger.debug(f"Filtered DFM forecast from {len(forecast_array)} to {len(actual_target_series)} target series before inverse transform")
                        else:
                            logger.warning(f"Could not map target_series to forecast indices. Using first {len(actual_target_series)} series.")
                            forecast_values = forecast_array[:len(actual_target_series)]
                    else:
                        logger.warning(f"Could not filter forecast to target_series. Using first {len(actual_target_series)} series.")
                        forecast_values = forecast_array[:len(actual_target_series)]

            # Apply inverse transformations for all models (consistent with short-term)
            if data_loader is not None and horizon_date is not None:
                try:
                    inv_pred = inverse_transform_predictions(
                        np.asarray(forecast_values), actual_target_series, data_loader,
                        reverse_transformations=True, test_data=test_data,
                        cutoff_date=pd.Timestamp(horizon_date)
                    )
                    if inv_pred is not None:
                        forecast_values = inv_pred
                except Exception as e:
                    logger.warning(f"Failed to inverse-transform forecast for {model_name} (horizon {horizon}w): {e}")
        
        # Compute metrics if actuals available
        metrics = None
        if actuals is not None:
            forecast_array = np.asarray(forecast_values).reshape(1, -1)
            actuals_array = np.asarray(actuals)
            
            if not np.isnan(actuals_array).all():
                std_for_metrics = validate_std_for_metrics(test_data_std, len(actual_target_series))
                smse = compute_smse(actuals_array, forecast_array, test_data_std=std_for_metrics)
                smae = compute_smae(actuals_array, forecast_array, test_data_std=std_for_metrics)
                
                if not (np.isnan(smse) and np.isnan(smae)):
                    metrics = {
                        "smse": float(smse) if not np.isnan(smse) else None,
                        "smae": float(smae) if not np.isnan(smae) else None
                    }
                    # Format values separately (can't use conditionals in format specifiers)
                    smse_str = f"{smse:.6f}" if not np.isnan(smse) else "NaN"
                    smae_str = f"{smae:.6f}" if not np.isnan(smae) else "NaN"
                    logger.info(f"Horizon {horizon}w: sMSE={smse_str}, sMAE={smae_str}")
                else:
                    logger.warning(f"Horizon {horizon}w: All metrics are NaN, skipping metric save")
            else:
                logger.warning(f"Horizon {horizon}w: All actuals are NaN, skipping metric computation")
        
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
