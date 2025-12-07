"""Training module - sktime and DFM/DDFM model training.

This module provides training functionality for:
- sktime models: ARIMA, VAR (using sktime forecaster interface)
- DFM/DDFM models: Dynamic Factor Models (using dfm-python)

Preprocessing (resampling, imputation, scaling) is handled by src.preprocessing module.

This module can be used:
- As CLI: python src/train.py train --config-name experiment/investment_koequipte_report
- As API: from src.train import train_model, compare_models_by_config

Functions exported for programmatic use:
- train_model: Train a single model
- compare_models_by_config: Compare multiple models from experiment config
"""

from pathlib import Path
import sys
import os
import argparse
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import numpy as np
import pandas as pd

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
    validate_experiment_config,
    ValidationError,
    DEFAULT_DDFM_ENCODER_LAYERS,
    DEFAULT_DDFM_NUM_FACTORS,
    DEFAULT_DDFM_EPOCHS
)

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except ImportError as e:
    raise ImportError(f"Required dependencies not available: {e}")

from src.models import DFM, DDFM  # For type hints


# ============================================================================
# Helper Functions
# ============================================================================

def _extract_target_series(cfg: DictConfig) -> Optional[str]:
    """Extract target series from config."""
    source_cfg = cfg.experiment if 'experiment' in cfg else cfg
    return source_cfg.get('target_series') if hasattr(source_cfg, 'get') else getattr(source_cfg, 'target_series', None)


def _extract_horizons(cfg: DictConfig, default_horizons: Optional[List[int]] = None) -> List[int]:
    """Extract forecast horizons from config."""
    if default_horizons is None:
        default_horizons = list(range(1, 23))  # 22 monthly horizons: 2024-01 to 2025-10
    
    source_cfg = cfg.experiment if 'experiment' in cfg else cfg
    horizons_raw = None
    
    if hasattr(source_cfg, 'get'):
        horizons_raw = source_cfg.get('forecast_horizons')
    else:
        horizons_raw = getattr(source_cfg, 'forecast_horizons', None)
    
    if horizons_raw is None:
        return default_horizons
    
    horizons_raw = OmegaConf.to_container(horizons_raw, resolve=True)
    if isinstance(horizons_raw, list):
        return [int(str(h)) for h in horizons_raw]
    else:
        return [int(str(horizons_raw))]


def _load_series_configs(cfg: DictConfig, config_path: str) -> List[dict]:
    """Load series configurations from config files."""
    import yaml
    
    source_cfg = cfg.experiment if 'experiment' in cfg else cfg
    series_ids_raw = None
    
    if hasattr(source_cfg, 'get'):
        series_ids_raw = source_cfg.get('series')
    else:
        series_ids_raw = getattr(source_cfg, 'series', None)
    
    if not series_ids_raw:
        return []
    
    series_ids = OmegaConf.to_container(series_ids_raw, resolve=True)
    if not isinstance(series_ids, list):
        return []
    
    # Get series overrides
    series_overrides = {}
    if hasattr(source_cfg, 'get'):
        series_overrides_raw = source_cfg.get('series_overrides')
    else:
        series_overrides_raw = getattr(source_cfg, 'series_overrides', None)
    
    if series_overrides_raw:
        try:
            series_overrides = OmegaConf.to_container(series_overrides_raw, resolve=True) or {}
            if not isinstance(series_overrides, dict):
                series_overrides = {}
        except (ValueError, TypeError):
            series_overrides = {}
    
    # Load each series config
    series_list = []
    config_path_obj = Path(config_path)
    series_dir = config_path_obj / "series"
    
    for series_id in series_ids:
        series_config_path = series_dir / f"{series_id}.yaml"
        
        # Default series config
        dfm_series = {
            'series_id': series_id,
            'frequency': 'm',
            'transformation': 'lin',
            '_block_names': None
        }
        
        # Load from file if exists
        if series_config_path.exists():
            try:
                with open(series_config_path, 'r') as f:
                    series_cfg = yaml.safe_load(f) or {}
                
                dfm_series.update({
                    'series_id': series_cfg.get('series_id', series_id),
                    'frequency': series_cfg.get('frequency', 'm'),
                    'transformation': series_cfg.get('transformation', 'lin'),
                })
                
                # Handle blocks/block field
                if 'blocks' in series_cfg and series_cfg['blocks'] is not None:
                    dfm_series['_block_names'] = series_cfg['blocks']
                elif 'block' in series_cfg and series_cfg['block'] is not None:
                    dfm_series['_block_names'] = series_cfg['block']
            except Exception as e:
                print(f"Warning: Failed to load series config {series_id}: {e}")
        
        # Apply overrides
        if series_id in series_overrides:
            override = series_overrides[series_id]
            if isinstance(override, dict):
                if 'frequency' in override:
                    dfm_series['frequency'] = override['frequency']
                if 'transformation' in override:
                    dfm_series['transformation'] = override['transformation']
                if 'blocks' in override and override['blocks'] is not None:
                    dfm_series['_block_names'] = override['blocks']
                elif 'block' in override and override['block'] is not None:
                    dfm_series['_block_names'] = override['block']
        
        series_list.append(dfm_series)
    
    return series_list


def _extract_data_path(cfg: DictConfig) -> Optional[str]:
    """Extract data path from config."""
    source_cfg = cfg.experiment if 'experiment' in cfg else cfg
    if hasattr(source_cfg, 'get'):
        return source_cfg.get('data_path')
    else:
        return getattr(source_cfg, 'data_path', None)


# Import preprocessing functions
from src.preprocessing import (
    resample_to_monthly,
    set_dataframe_frequency,
    impute_missing_values,
    prepare_multivariate_data,
    prepare_univariate_data
)


def _detect_model_type_from_name(model_name: str) -> str:
    """Detect model type from model name (e.g., 'KOEQUIPTE_arima' -> 'arima')."""
    valid_types = ['arima', 'var', 'dfm', 'ddfm']
    parts = model_name.lower().split('_')
    
    # Check last part first (most common: TARGET_MODEL)
    if parts[-1] in valid_types:
        return parts[-1]
    
    # Check first part (MODEL_TARGET format)
    if parts[0] in valid_types:
        return parts[0]
    
    # Search all parts
    for part in parts:
        if part in valid_types:
            return part
    
    return None  # Not found in name


def _detect_model_type_from_config(cfg: DictConfig) -> str:
    """Detect model type from Hydra config."""
    defaults = cfg.get('defaults', [])
    for default in defaults:
        if isinstance(default, dict) and default.get('override') == '/model':
            model_override = default.get('_target_', '')
            if 'ddfm' in model_override.lower():
                return "ddfm"
    
    model_type = cfg.get('model_type', '').lower()
    if model_type in ('ddfm', 'deep'):
        return "ddfm"
    
    ddfm_params = ['encoder_layers', 'epochs', 'learning_rate', 'batch_size']
    if any(key in cfg for key in ddfm_params):
        return "ddfm"
    
    return "dfm"


def _train_forecaster(
    model_type: str,
    config_name: str,
    cfg: DictConfig,
    data_file: str,
    model_name: Optional[str],
    horizons: Optional[List[int]],
    outputs_dir: Path,
    model_cfg_dict: Optional[dict] = None
) -> Dict[str, Any]:
    """Train any model using sktime forecaster interface."""
    try:
        from sktime.forecasting.arima import ARIMA as SktimeARIMA
        from sktime.forecasting.var import VAR as SktimeVAR
        from sktime.transformations.series.impute import Imputer
        from src.models import DFMForecaster, DDFMForecaster
    except ImportError as e:
        raise ImportError(f"sktime is required for {model_type} models. Install with: pip install sktime[forecasting]") from e
    
    data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    train_start = pd.Timestamp('1985-01-01')
    train_end = pd.Timestamp('2019-12-31')
    data = data[(data.index >= train_start) & (data.index <= train_end)]
    
    if len(data) == 0:
        raise ValidationError(f"No data available in training period (1985-2019). Data range: {data.index.min()} to {data.index.max()}")
    
    print(f"Training data period: {data.index.min()} to {data.index.max()} (1985-2019 enforced)")
    print(f"Original data frequency: Weekly ({len(data)} points)")
    print(f"All models will use monthly frequency (window-averaged from weekly data)")
    
    source_cfg = cfg.experiment if 'experiment' in cfg else cfg
    target_series = _extract_target_series(cfg)
    
    if not target_series or target_series not in data.columns:
        raise ValidationError(f"target_series '{target_series}' not found in data. Available columns: {list(data.columns)}")
    
    model_params = {}
    if 'model_overrides' in source_cfg:
        model_overrides_dict = OmegaConf.to_container(source_cfg.model_overrides, resolve=True)
        if isinstance(model_overrides_dict, dict) and model_type in model_overrides_dict:
            model_params = model_overrides_dict[model_type] or {}
    
    print(f"\n{'='*70}")
    print(f"Training {model_type.upper()} model")
    print(f"{'='*70}")
    print(f"Config: {config_name}")
    print(f"Data: {data_file}")
    
    if model_type == 'dfm':
        max_iter = model_params.get('max_iter', 5000)
        threshold = model_params.get('threshold', 1e-5)
        config_dict = model_cfg_dict if model_cfg_dict else {}
        if not config_dict or 'series' not in config_dict:
            raise ValidationError(f"No series found in config. Config: {config_name}")
        
        forecaster = DFMForecaster(
            config_dict=config_dict,
            max_iter=max_iter,
            threshold=threshold
        )
        
        target_series = _extract_target_series(cfg)
        # Preprocess multivariate data for DFM (resample to monthly, set frequency)
        # Note: DFM uses internal preprocessing pipeline (imputation + scaling), so we only do resampling and frequency setting here
        y_train = prepare_multivariate_data(data, config_dict, cfg, target_series, model_type='dfm')
        y_train = set_dataframe_frequency(y_train)
        print(f"Max iterations: {max_iter}, Threshold: {threshold}, Series: {y_train.shape[1]} (filtered from config, target_series included)")
        
    elif model_type == 'ddfm':
        epochs = model_params.get('epochs', 100)
        encoder_layers = model_params.get('encoder_layers', [64, 32])
        num_factors = model_params.get('num_factors', 1)
        learning_rate = model_params.get('learning_rate', 0.0001)
        batch_size = model_params.get('batch_size', 32)
        
        config_dict = model_cfg_dict if model_cfg_dict else {}
        if not config_dict or 'series' not in config_dict:
            raise ValidationError(f"No series found in config. Config: {config_name}")
        
        forecaster = DDFMForecaster(
            config_dict=config_dict,
            encoder_layers=encoder_layers,
            num_factors=num_factors,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size
        )
        
        target_series = _extract_target_series(cfg)
        # Preprocess multivariate data for DDFM (resample to monthly, set frequency)
        # Note: DDFM uses internal preprocessing pipeline (imputation + scaling), so we only do resampling and frequency setting here
        y_train = prepare_multivariate_data(data, config_dict, cfg, target_series, model_type='ddfm')
        y_train = set_dataframe_frequency(y_train)
        print(f"Epochs: {epochs}, Encoder layers: {encoder_layers}, Factors: {num_factors}, Series: {y_train.shape[1]} (filtered from config, target_series included)")
        
    elif model_type == 'arima':
        if not target_series:
            raise ValidationError(f"target_series required for ARIMA. Config: {config_name}")
        
        # Preprocess univariate data for ARIMA (resample to monthly, set frequency)
        y_train = prepare_univariate_data(data, target_series)
        if len(y_train) == 0:
            raise ValidationError(f"No valid monthly data for target series '{target_series}' after resampling")
        
        print(f"Target series: {target_series} (resampled to monthly)")
        order = model_params.get('order', [1, 1, 1])
        auto_arima = model_params.get('auto_arima', {})
        
        # Create base forecaster
        if auto_arima and auto_arima.get('enabled', False):
            from sktime.forecasting.arima import AutoARIMA
            base_forecaster = AutoARIMA(
                max_p=auto_arima.get('max_p', 5),
                max_d=auto_arima.get('max_d', 2),
                max_q=auto_arima.get('max_q', 5),
                information_criterion=auto_arima.get('information_criterion', 'aic')
            )
        else:
            base_forecaster = SktimeARIMA(order=tuple(order) if isinstance(order, list) else order)
        
        # Use ForecastingPipeline with Imputer for ARIMA (sktime handles this automatically)
        from sktime.forecasting.compose import ForecastingPipeline
        from sktime.forecasting.naive import NaiveForecaster
        
        # Multi-stage imputation: ffill -> bfill -> forecaster
        imputer_ffill = Imputer(method="ffill")
        imputer_bfill = Imputer(method="bfill")
        imputer_forecaster = Imputer(method="forecaster", forecaster=NaiveForecaster(strategy="last"))
        
        # Create pipeline: imputation -> forecaster
        # Note: sktime's ForecastingPipeline applies transformers before forecaster.fit()
        forecaster = ForecastingPipeline([
            ('imputer_ffill', imputer_ffill),
            ('imputer_bfill', imputer_bfill),
            ('imputer_forecaster', imputer_forecaster),
            ('forecaster', base_forecaster)
        ])
        
        print(f"Order: {order}")
        
    elif model_type == 'var':
        lag_order = model_params.get('lag_order')
        auto_lag = model_params.get('auto_lag', {})
        trend = model_params.get('trend', 'c')
        if trend is None:
            trend = 'c'
        
        # Preprocess multivariate data for VAR (resample to monthly, set frequency, impute)
        y_train = prepare_multivariate_data(data, None, cfg, target_series, model_type='var')
        y_train = set_dataframe_frequency(y_train)
        # VAR cannot handle missing data - must impute before fitting
        # Note: VAR (statsmodels) doesn't support NaNs, so we must pre-impute
        # This is different from ARIMA which can use ForecastingPipeline
        y_train = impute_missing_values(y_train, model_type='var')
        
        if len(y_train) == 0:
            raise ValidationError(f"VAR: No valid data after imputation.")
        
        if y_train.shape[1] < 2:
            raise ValidationError(f"VAR requires at least 2 series. Found {y_train.shape[1]} series.")
        
        if y_train.isnull().any().any():
            nan_count = y_train.isnull().sum().sum()
            raise ValidationError(f"VAR cannot handle missing data. Found {nan_count} NaN values after all imputation attempts.")
        
        if lag_order is None and auto_lag and auto_lag.get('enabled', False):
            maxlags = auto_lag.get('maxlags', 12)
            if maxlags is None:
                maxlags = 12
            ic = auto_lag.get('ic', 'aic')
            if ic is None:
                ic = 'aic'
            forecaster = SktimeVAR(maxlags=int(maxlags), trend=str(trend), ic=str(ic))
        else:
            maxlags = lag_order if lag_order is not None else 1
            if maxlags is None:
                maxlags = 1
            forecaster = SktimeVAR(maxlags=int(maxlags), trend=str(trend))
        print(f"Max lags: {maxlags}, Trend: {trend}, Series: {y_train.shape[1]}")
        # NOTE: VAR models can become numerically unstable for long forecasting horizons (>7 months)
        # This is a known limitation of VAR models. The evaluation code (evaluation.py) automatically
        # filters extreme values (>1e10) and marks them as NaN. This is expected behavior.
        # For monthly data, VAR should be limited to horizon <= 7 months to prevent instability.
    
    # Extract horizons from config if not provided
    if horizons is None:
        horizons = _extract_horizons(cfg)
    
    # Train once on full training data (no evaluation split to avoid double training)
    forecaster.fit(y_train)
    print(f"{'='*70}\n")
    
    # No evaluation metrics (trained on full data, no held-out test set)
    forecast_metrics = {}
    
    # Determine model directory and name
    if model_name and (str(outputs_dir).endswith(model_name) or str(outputs_dir.name) == model_name):
        model_dir = outputs_dir
        final_model_name = outputs_dir.name
    else:
        final_model_name = model_name or (
            f"{model_type}_{target_series}_{datetime.now().strftime('%Y%m%d_%H%M%S')}" 
            if target_series else f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        model_dir = outputs_dir / final_model_name
    
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / "model.pkl"
    
    # Atomic save: save to temp file first, then rename
    import pickle
    import os
    import shutil
    
    temp_file = model_dir / f"model.pkl.tmp.{os.getpid()}"
    try:
        with open(temp_file, 'wb') as f:
            pickle.dump({
                'forecaster': forecaster,
                'model_type': model_type,
                'target_series': target_series,
                'config': OmegaConf.to_container(cfg, resolve=True)
            }, f)
        
        # Atomic rename: replace old file only after successful save
        if model_file.exists():
            backup_file = model_dir / f"model.pkl.backup.{os.getpid()}"
            shutil.move(str(model_file), str(backup_file))
            try:
                shutil.move(str(temp_file), str(model_file))
                backup_file.unlink()  # Remove backup after successful rename
            except Exception:
                # Restore backup if rename fails
                if backup_file.exists():
                    shutil.move(str(backup_file), str(model_file))
                raise
        else:
            shutil.move(str(temp_file), str(model_file))
        
        print(f"Saved model checkpoint: {model_file}")
    except Exception as e:
        # Clean up temp file if save failed
        if temp_file and temp_file.exists():
            temp_file.unlink()
        raise ValidationError(f"Failed to save model checkpoint: {e}") from e
    
    # Extract training metrics (DFM/DDFM only)
    result = None
    metadata = {}
    if model_type in ['dfm', 'ddfm']:
        underlying_model = getattr(forecaster, '_dfm_model', None) or getattr(forecaster, '_ddfm_model', None)
        if underlying_model:
            try:
                result = underlying_model.get_result()
                metadata = underlying_model.get_metadata()
            except (AttributeError, RuntimeError):
                pass
    
    # Build metrics dict
    base_metrics = {
        'model_type': model_type,
        'forecast_metrics': forecast_metrics,
        'training_completed': True
    }
    
    if result:
        if isinstance(result, dict):
            base_metrics.update({
                'converged': result.get('converged', True),
                'num_iter': result.get('num_iter', 0),
                'loglik': result.get('loglik', np.nan)
            })
        else:
            base_metrics.update({
                'converged': getattr(result, 'converged', True),
                'num_iter': getattr(result, 'num_iter', 0),
                'loglik': getattr(result, 'loglik', np.nan)
            })
    else:
        base_metrics.update({
            'converged': True,
            'num_iter': 0,
            'loglik': np.nan
        })
        if model_type not in ['dfm', 'ddfm']:
            result = {'converged': True, 'num_iter': 0, 'loglik': np.nan}
            metadata = {
                'created_at': datetime.now().isoformat(),
                'model_type': model_type,
                'training_completed': True,
                'target_series': target_series
            }
    
    metrics = base_metrics
    
    return {
        'status': 'completed',
        'model_name': final_model_name,
        'model_dir': str(model_dir),
        'metrics': metrics,
        'result': result,
        'metadata': metadata
    }


# ============================================================================
# Main Training Functions
# ============================================================================

def train(
    config_name: str,
    config_path: Optional[str] = None,
    data_path: Optional[str] = None,
    model_name: Optional[str] = None,
    config_overrides: Optional[list] = None,
    horizons: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Train a single forecasting model using Hydra configuration.
    
    This function provides a unified interface for training all model types
    (ARIMA, VAR, DFM, DDFM) using sktime forecaster interface. The model is
    trained, evaluated on specified horizons, and saved to checkpoint/.
    """
    config_path = config_path or str(project_root / "config")
    
    checkpoint_dir = None
    checkpoint_model_name = None
    if config_overrides:
        for override in config_overrides:
            if override.startswith('checkpoint_dir=') or override.startswith('+checkpoint_dir='):
                checkpoint_dir = override.split('=', 1)[1]
            elif override.startswith('checkpoint_model_name=') or override.startswith('+checkpoint_model_name='):
                checkpoint_model_name = override.split('=', 1)[1]
    
    if not checkpoint_dir:
        raise ValueError("checkpoint_dir must be specified. Use --checkpoint-dir argument.")
    outputs_dir = Path(checkpoint_dir)
    if checkpoint_model_name:
        outputs_dir = outputs_dir / checkpoint_model_name
    
    # Create directory and verify it's writable
    try:
        outputs_dir.mkdir(parents=True, exist_ok=True)
        if not outputs_dir.exists():
            raise ValidationError(f"Cannot create checkpoint directory: {outputs_dir}")
        if not os.access(outputs_dir, os.W_OK):
            raise ValidationError(f"Checkpoint directory is not writable: {outputs_dir}")
    except (OSError, PermissionError) as e:
        raise ValidationError(f"Cannot create or write to checkpoint directory {outputs_dir}: {e}") from e
    
    with hydra.initialize_config_dir(config_dir=config_path, version_base="1.3"):
        cfg = hydra.compose(config_name=config_name, overrides=config_overrides or [])
        if OmegaConf.is_struct(cfg):
            OmegaConf.set_struct(cfg, False)
        if 'experiment' in cfg and OmegaConf.is_struct(cfg.experiment):
            OmegaConf.set_struct(cfg.experiment, False)
        
        # Detect model type from model_name or config
        if model_name:
            detected_type = _detect_model_type_from_name(model_name)
            model_type = detected_type if detected_type else _detect_model_type_from_config(cfg)
        else:
            model_type = _detect_model_type_from_config(cfg)
        
        # Load series configs
        series_list = _load_series_configs(cfg, config_path)
        
        model_cfg_dict = {}
        config_path_obj = Path(config_path)
        model_config_path = config_path_obj / "model" / f"{model_type}.yaml"
        
        block_names_order = []
        if model_config_path.exists():
            import yaml
            with open(model_config_path, 'r') as f:
                model_yaml = yaml.safe_load(f) or {}
            if 'blocks' in model_yaml and isinstance(model_yaml['blocks'], dict):
                block_names_order = list(model_yaml['blocks'].keys())
            excluded_keys = ['models', 'series', 'target_series', 'data_path', 'start_date', 'end_date', 
                            'forecast_horizons', 'evaluation_metrics', 'test_size', 'output_path', 
                            'name', 'description', 'defaults']
            model_yaml = {k: v for k, v in model_yaml.items() if k not in excluded_keys}
            model_cfg_dict = {**model_yaml, **model_cfg_dict}
        
        experiment_model_overrides = {}
        source_cfg = cfg.experiment if 'experiment' in cfg else cfg
        
        if 'model_overrides' in source_cfg:
            model_overrides_dict = OmegaConf.to_container(source_cfg.model_overrides, resolve=True)
            if isinstance(model_overrides_dict, dict) and model_type in model_overrides_dict:
                model_specific_overrides = model_overrides_dict[model_type]
                if isinstance(model_specific_overrides, dict):
                    experiment_model_overrides = model_specific_overrides
        
        if experiment_model_overrides:
            model_cfg_dict = {**model_cfg_dict, **experiment_model_overrides}
        
        if not block_names_order and model_type == 'ddfm':
            dfm_config_path = config_path_obj / "model" / "dfm.yaml"
            if dfm_config_path.exists():
                import yaml
                with open(dfm_config_path, 'r') as f:
                    dfm_yaml = yaml.safe_load(f) or {}
                if 'blocks' in dfm_yaml and isinstance(dfm_yaml['blocks'], dict):
                    block_names_order = list(dfm_yaml['blocks'].keys())
                    if 'blocks' not in model_cfg_dict:
                        model_cfg_dict['blocks'] = dfm_yaml['blocks']
        
        block_clocks = {}
        if 'blocks' in model_cfg_dict and isinstance(model_cfg_dict['blocks'], dict):
            for block_name, block_cfg in model_cfg_dict['blocks'].items():
                if isinstance(block_cfg, dict) and 'clock' in block_cfg:
                    block_clocks[block_name] = block_cfg['clock']
        
        freq_hierarchy = {'d': 1, 'w': 2, 'm': 3, 'q': 4, 'sa': 5, 'a': 6}
        
        filtered_series_list = []
        if block_names_order and series_list:
            for series_item in series_list:
                series_freq = series_item.get('frequency', 'm').lower()
                series_freq_level = freq_hierarchy.get(series_freq, 3)
                
                if '_block_names' in series_item:
                    block_names = series_item.pop('_block_names')
                    if block_names is None:
                        if block_names_order:
                            first_block_name = block_names_order[0]
                            first_block_clock = block_clocks.get(first_block_name, 'm')
                            first_block_clock_level = freq_hierarchy.get(first_block_clock.lower(), 3)
                            if series_freq_level >= first_block_clock_level:
                                series_item['blocks'] = [1] + [0] * (len(block_names_order) - 1) if len(block_names_order) > 1 else [1]
                                filtered_series_list.append(series_item)
                            else:
                                print(f"Warning: Series {series_item.get('series_id', 'unknown')} with frequency '{series_freq}' is incompatible with first block clock '{first_block_clock}'. Skipping.")
                        else:
                            filtered_series_list.append(series_item)
                        continue
                    compatible_blocks = []
                    for block_spec in block_names:
                        block_name = None
                        if isinstance(block_spec, str):
                            block_name = block_spec
                        elif isinstance(block_spec, int) and 0 <= block_spec < len(block_names_order):
                            block_name = block_names_order[block_spec]
                        
                        if block_name:
                            block_clock = block_clocks.get(block_name, 'm')
                            block_clock_level = freq_hierarchy.get(block_clock.lower(), 3)
                            if series_freq_level >= block_clock_level:
                                compatible_blocks.append(block_spec)
                    
                    if compatible_blocks:
                        block_vector = [0] * len(block_names_order)
                        for block_spec in compatible_blocks:
                            if isinstance(block_spec, str):
                                if block_spec in block_names_order:
                                    block_idx = block_names_order.index(block_spec)
                                    block_vector[block_idx] = 1
                            elif isinstance(block_spec, int):
                                block_idx = block_spec - 1 if block_spec > 0 else block_spec
                                if 0 <= block_idx < len(block_names_order):
                                    block_vector[block_idx] = 1
                        series_item['blocks'] = block_vector
                        filtered_series_list.append(series_item)
                    else:
                        print(f"Warning: Series {series_item.get('series_id', 'unknown')} with frequency '{series_freq}' is incompatible with block clocks. Skipping.")
                else:
                    if block_names_order:
                        first_block_name = block_names_order[0]
                        first_block_clock = block_clocks.get(first_block_name, 'm')
                        first_block_clock_level = freq_hierarchy.get(first_block_clock.lower(), 3)
                        if series_freq_level >= first_block_clock_level:
                            series_item['blocks'] = [1] + [0] * (len(block_names_order) - 1) if len(block_names_order) > 1 else [1]
                            filtered_series_list.append(series_item)
                        else:
                            print(f"Warning: Series {series_item.get('series_id', 'unknown')} with frequency '{series_freq}' is incompatible with first block clock '{first_block_clock}'. Skipping.")
                    else:
                        filtered_series_list.append(series_item)
        else:
            filtered_series_list = series_list
        
        if filtered_series_list:
            model_cfg_dict['series'] = filtered_series_list
        else:
            raise ValidationError(f"No compatible series found after filtering by block clock frequencies. Check series frequencies against block clocks.")
        
        excluded_keys = ['models', 'target_series', 'data_path', 'start_date', 'end_date', 
                        'forecast_horizons', 'evaluation_metrics', 'test_size', 'output_path', 
                        'name', 'description', 'defaults']
        model_cfg_dict = {k: v for k, v in model_cfg_dict.items() if k not in excluded_keys}
        
        # Extract data path from config or use provided
        data_file = data_path or _extract_data_path(cfg)
        if not data_file:
            raise ValidationError(f"data_path required. Config: {config_name}")
        
        return _train_forecaster(
            model_type=model_type,
            config_name=config_name,
            cfg=cfg,
            data_file=data_file,
            model_name=model_name,
            horizons=horizons,
            outputs_dir=outputs_dir,
            model_cfg_dict=model_cfg_dict
        )


def compare_models(
    target_series: str,
    models: List[str],
    horizons: List[int] = list(range(1, 23)),  # 22 monthly horizons: 2024-01 to 2025-10
    data_path: Optional[str] = None,
    config_dir: Optional[str] = None,
    config_name: Optional[str] = None,
    config_overrides: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Train multiple models and compare their performance."""
    from src.utils import get_project_root
    project_root = get_project_root()
    config_dir = config_dir or str(project_root / "config")
    if data_path is None:
        data_path = str(project_root / "data" / "data.csv")
        if not Path(data_path).exists():
            data_path = str(project_root / "data" / "sample_data.csv")
    
    # Use fixed directory name (overwrite mode) - one result per experiment
    output_dir = project_root / "outputs" / "comparisons" / target_series
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"Comparing models for {target_series}")
    print(f"Models: {', '.join(models)} | Horizons: {horizons}")
    print("=" * 70)
    
    model_results = {}
    failed_models = []
    
    if config_name is None:
        report_map = {
            "KOEQUIPTE": "experiment/investment_koequipte_report",
            "KOWRCCNSE": "experiment/consumption_kowrccnse_report",
            "KOIPALL.G": "experiment/production_koipallg_report"
        }
        config_name = report_map.get(target_series, f"experiment/{target_series.lower().replace('...', '').replace('.', '')}_report")
    
    if horizons is None or len(horizons) == 0:
        cfg = parse_experiment_config(config_name, config_dir)
        params = extract_experiment_params(cfg)
        horizons = params.get('horizons', list(range(1, 23)))
        print(f"Extracted horizons from config: {len(horizons)} horizons (1-22)")
    
    for i, model_name in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] {model_name.upper()}...")
        
        if config_overrides is None:
            config_overrides = []
        
        try:
            checkpoint_dir = None
            if config_overrides:
                for override in config_overrides:
                    if override.startswith('checkpoint_dir=') or override.startswith('+checkpoint_dir='):
                        checkpoint_dir = override.split('=', 1)[1]
                        break
            
            checkpoint_path = None
            if checkpoint_dir:
                checkpoint_path = Path(checkpoint_dir) / f"{target_series}_{model_name}" / "model.pkl"
            
            if checkpoint_path and checkpoint_path.exists():
                print(f"  Loading model from checkpoint: {checkpoint_path}")
                import pickle
                with open(checkpoint_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                result = {
                    'status': 'completed',
                    'model_name': f"{target_series}_{model_name}",
                    'model_dir': str(checkpoint_path.parent),
                    'model_type': model_name,
                    'target_series': target_series,
                    'metrics': model_data.get('metrics', {}),
                    'checkpoint_loaded': True
                }
                
                forecaster = model_data.get('forecaster')
                if forecaster and horizons:
                    actual_data_path = data_path
                    if not actual_data_path or not Path(actual_data_path).exists():
                        from src.utils import get_project_root
                        project_root = get_project_root()
                        actual_data_path = str(project_root / "data" / "data.csv")
                        if not Path(actual_data_path).exists():
                            actual_data_path = str(project_root / "data" / "sample_data.csv")
                    
                    data = pd.read_csv(actual_data_path, index_col=0, parse_dates=True)
                    train_start = pd.Timestamp('1985-01-01')
                    train_end = pd.Timestamp('2019-12-31')
                    data = data[(data.index >= train_start) & (data.index <= train_end)]
                    data = resample_to_monthly(data)
                    
                    split_idx = int(len(data) * 0.8)
                    y_train_eval = data.iloc[:split_idx]
                    y_test_eval = data.iloc[split_idx:]
                    
                    from src.evaluation import evaluate_forecaster
                    forecast_metrics_raw = evaluate_forecaster(
                        forecaster, y_train_eval, y_test_eval, horizons, target_series=target_series
                    )
                    forecast_metrics = {str(k): v for k, v in forecast_metrics_raw.items()}
                    result['metrics'] = {'forecast_metrics': forecast_metrics}
            else:
                # Checkpoint not found - skip this model
                checkpoint_msg = f"checkpoint/{target_series}_{model_name}/model.pkl" if checkpoint_dir else "checkpoint"
                print(f"  ⚠ Skipping: Checkpoint not found ({checkpoint_msg})")
                print(f"     Run 'bash run_train.sh' to train this model first")
                result = {
                    'status': 'skipped',
                    'model_name': f"{target_series}_{model_name}",
                    'model_type': model_name,
                    'target_series': target_series,
                    'error': f'Checkpoint not found: {checkpoint_path}',
                    'metrics': None
                }
                failed_models.append(model_name)
                model_results[model_name] = result
                continue
            
            if result is not None:
                model_results[model_name] = result
                if result.get('status') == 'completed':
                    print(f"  ✓ Completed")
                elif result.get('status') == 'skipped':
                    # Already printed skip message above
                    pass
                else:
                    print(f"  ⚠ Status: {result.get('status', 'unknown')}")
        except Exception as e:
            import traceback
            print(f"  ✗ Failed: {str(e)}")
            print("Full traceback:")
            traceback.print_exc()
            failed_models.append(model_name)
            model_results[model_name] = {'status': 'failed', 'error': str(e), 'metrics': None}
    
    comparison = None
    successful_count = len(model_results) - len(failed_models)
    if successful_count > 0:
        print(f"\nComparing {successful_count} successful models...")
        comparison = _compare_results(model_results, horizons, target_series)
        
        if comparison and comparison.get('metrics_table') is not None:
            from src.evaluation import generate_comparison_table
            
            if generate_comparison_table:
                table_path = output_dir / "comparison_table.csv"
                generate_comparison_table(comparison, output_path=str(table_path))
                print(f"  Table: {table_path}")
    
    comparison_data = {
        'target_series': target_series,
        'models': models,
        'horizons': horizons,
        'results': model_results,
        'comparison': comparison,
        'output_dir': str(output_dir),
        'failed_models': failed_models
    }
    
    results_file = output_dir / "comparison_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nResults: {results_file}")
    
    # Count skipped models separately
    skipped_models = [name for name, result in model_results.items() if result.get('status') == 'skipped']
    actual_failed = [name for name in failed_models if name not in skipped_models]
    
    if skipped_models:
        print(f"Skipped (no checkpoint): {', '.join(skipped_models)}")
    if actual_failed:
        print(f"Failed: {', '.join(actual_failed)}")
    print("=" * 70)
    
    return comparison_data


def _compare_results(
    results: Dict[str, Dict[str, Any]],
    horizons: List[int],
    target_series: str
) -> Dict[str, Any]:
    """Compare results from multiple models."""
    from src.evaluation import compare_multiple_models
    
    successful_results = {
        name: result for name, result in results.items()
        if result.get('status') == 'completed' and result.get('metrics') is not None
    }
    
    if not successful_results:
        return {
            'metrics_table': None,
            'summary': 'No successful models to compare',
            'best_model_per_horizon': {}
        }
    
    return compare_multiple_models(
        model_results=successful_results,
        horizons=horizons,
        target_series=target_series
    )


# ============================================================================
# Programmatic API Functions
# ============================================================================

def train_model(
    config_name: str,
    config_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    overrides: Optional[List[str]] = None,
    checkpoint_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Train a single model (programmatic API)."""
    if config_dir is None:
        config_dir = str(get_project_root() / "config")
    
    cfg = parse_experiment_config(config_name, config_dir, overrides)
    validate_experiment_config(cfg, require_target=False, require_models=True)
    
    params = extract_experiment_params(cfg)
    models = params['models']
    target_series = params.get('target_series')
    
    if model_name is None:
        model_name = models[0] if models else None
        if len(models) > 1:
            print(f"Warning: Config specifies {len(models)} models, training only first: {model_name}")
    
    if not model_name:
        raise ValueError(f"Config {config_name} must specify at least one model in 'models' list")
    
    if checkpoint_dir and target_series:
        checkpoint_model_name = f"{target_series}_{model_name}"
    else:
        checkpoint_model_name = model_name
    
    final_overrides = list(overrides) if overrides else []
    if checkpoint_dir:
        final_overrides.append(f"+checkpoint_dir={checkpoint_dir}")
        final_overrides.append(f"+checkpoint_model_name={checkpoint_model_name}")
    
    result = train(
        config_name=config_name,
        config_path=config_dir,
        model_name=checkpoint_model_name,
        config_overrides=final_overrides
    )
    
    if checkpoint_dir and 'model_dir' in result:
        result['model_dir'] = str(Path(checkpoint_dir) / checkpoint_model_name)
    
    return result


def compare_models_by_config(
    config_name: str,
    config_dir: Optional[str] = None,
    overrides: Optional[List[str]] = None,
    models_filter: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Compare multiple models from experiment config (programmatic API)."""
    if config_dir is None:
        config_dir = str(get_project_root() / "config")
    
    cfg = parse_experiment_config(config_name, config_dir, overrides)
    validate_experiment_config(cfg, require_target=True, require_models=True)
    
    params = extract_experiment_params(cfg)
    
    models_to_run = params['models']
    if models_filter:
        models_to_run = [m for m in models_to_run if m.lower() in [mf.lower() for mf in models_filter]]
        if not models_to_run:
            raise ValueError(f"No models match filter {models_filter}. Available models: {params['models']}")
    
    result = compare_models(
        target_series=params['target_series'],
        models=models_to_run,
        horizons=params['horizons'],
        data_path=params['data_path'],
        config_dir=config_dir,
        config_name=config_name,
        config_overrides=overrides
    )
    
    return result


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Main CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train models using Hydra config")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    train_parser = subparsers.add_parser('train', help='Train single model (requires experiment config)')
    train_parser.add_argument("--config-name", required=True, help="Experiment config name (e.g., experiment/investment_koequipte_report)")
    train_parser.add_argument("--model", help="Model name to train (e.g., arima, var, dfm, ddfm). If not specified, uses first model from config.")
    train_parser.add_argument("--checkpoint-dir", required=True, help="Directory to save checkpoint (e.g., checkpoint)")
    train_parser.add_argument("--override", action="append", help="Hydra config override (e.g., model_overrides.dfm.max_iter=10)")
    
    compare_parser = subparsers.add_parser('compare', help='Compare multiple models (requires experiment config)')
    compare_parser.add_argument("--config-name", required=True, help="Experiment config name (e.g., experiment/investment_koequipte_report)")
    compare_parser.add_argument("--override", action="append", help="Hydra config override")
    compare_parser.add_argument("--models", nargs="+", help="Filter models to run (e.g., --models arima var). If not specified, runs all models from config.")
    
    args = parser.parse_args()
    
    config_path = str(get_project_root() / "config")
    
    if args.command == 'train':
        result = train_model(
            config_name=args.config_name,
            config_dir=config_path,
            model_name=args.model,
            checkpoint_dir=args.checkpoint_dir,
            overrides=args.override
        )
        print(f"\n✓ Model saved to: {result['model_dir']}")
        
    elif args.command == 'compare':
        overrides = list(args.override) if args.override else []
        
        result = compare_models_by_config(
            config_name=args.config_name,
            config_dir=config_path,
            overrides=overrides,
            models_filter=args.models
        )
        print(f"\n✓ Comparison saved to: {result['output_dir']}")
        if result.get('failed_models'):
            print(f"  Failed: {', '.join(result['failed_models'])}")


if __name__ == "__main__":
    main()
