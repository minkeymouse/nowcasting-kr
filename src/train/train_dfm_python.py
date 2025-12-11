"""Training module for dfm-python-based models (DFM, DDFM).

This module handles training for Dynamic Factor Model (DFM) and Deep Dynamic
Factor Model (DDFM) using dfm-python package.
"""

from pathlib import Path
import sys
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np
import pandas as pd

from src.utils import (
    ValidationError,
    get_experiment_cfg,
    get_project_root,
    get_config_path,
    set_forecaster_attributes,
    extract_model_metrics,
    resolve_data_path,
    load_and_filter_data,
    setup_paths,
    TRAIN_START,
    TRAIN_END,
    SCALER_ROBUST,
    SCALER_STANDARD
)

# Setup paths
setup_paths(include_dfm_python=True, include_src=True)
project_root = get_project_root()

try:
    from omegaconf import DictConfig, OmegaConf
except ImportError as e:
    raise ImportError(f"Required dependencies not available: {e}")

from src.train.preprocess import (
    set_dataframe_frequency,
    prepare_multivariate_data
)

logger = logging.getLogger(__name__)

# Import dfm-python config classes
try:
    from dfm_python.config import DFMConfig, SeriesConfig
except ImportError:
    DFMConfig = None
    SeriesConfig = None


# ============================================================================
# Helper Functions
# ============================================================================

def _build_dfm_config(
    cfg: DictConfig,
    available_series: List[str],
    model_config: dict,
    config_path: str,
    clock: str = 'w',
    mixed_freq: bool = True
) -> Any:
    """Build dfm-python DFMConfig from experiment config."""
    if DFMConfig is None:
        raise ImportError("dfm-python package not available. Install with: pip install dfm-python")
    
    exp_cfg = get_experiment_cfg(cfg)
    series_ids_raw = exp_cfg.get('series', [])
    series_ids = OmegaConf.to_container(series_ids_raw, resolve=True) if series_ids_raw else []
    if not isinstance(series_ids, list):
        series_ids = []
    
    available_series_set = set(available_series)
    filtered_series_ids = [s for s in series_ids if s in available_series_set]
    
    if not filtered_series_ids:
        raise ValidationError(
            f"No series found in available_series. "
            f"Available: {available_series[:10]}..."
        )
    
    config_dict = {
        'series': filtered_series_ids,
        'clock': clock,
        'max_iter': model_config.get('max_iter', 5000),
        'threshold': model_config.get('threshold', 1e-5),
        'nan_method': model_config.get('nan_method', 2),
        'nan_k': model_config.get('nan_k', 3),
        'scaler': model_config.get('scaler', SCALER_ROBUST)
    }
    
    blocks_dict = model_config.get('blocks', {})
    if not blocks_dict:
        blocks_dict = {
            'block1': {
                'factors': 3,
                'ar_lag': 1,
                'clock': clock,
                'series': []
            }
        }
    config_dict['blocks'] = blocks_dict
    
    return DFMConfig.from_hydra(config_dict, config_path=config_path)


def _create_time_index_from_data(data: pd.DataFrame) -> Any:
    """Create dfm-python TimeIndex from DataFrame."""
    try:
        from dfm_python.utils.time import TimeIndex, parse_timestamp
    except ImportError:
        raise ImportError("dfm-python package not available. Install with: pip install dfm-python")
    
    if isinstance(data.index, pd.DatetimeIndex):
        dates = data.index
    elif 'date' in data.columns:
        dates = pd.to_datetime(data['date'])
    else:
        dates = pd.date_range(start=TRAIN_START, periods=len(data), freq='W')
    
    time_list = [parse_timestamp(d.strftime('%Y-%m-%d')) for d in dates]
    return TimeIndex(time_list)


def _apply_target_preprocessing(
    df: pd.DataFrame,
    target_series: str,
    clip_quantiles: Optional[List[float]] = None
) -> pd.DataFrame:
    """Apply target-specific preprocessing (e.g., quantile clipping)."""
    if clip_quantiles and len(clip_quantiles) == 2:
        try:
            lower, upper = float(clip_quantiles[0]), float(clip_quantiles[1])
            lower = max(0.0, min(lower, 1.0))
            upper = max(lower, min(upper, 1.0))
            df = df.clip(lower=df.quantile(lower), upper=df.quantile(upper), axis=1)
            logger.info(
                f"Applied quantile clipping for {target_series}: "
                f"lower={lower:.3f}, upper={upper:.3f}"
            )
        except Exception as e:
            logger.warning(f"Quantile clipping skipped for {target_series}: {e}")
    return df


def _create_preprocessing_pipeline(
    model: Any,
    scaler_type: str = SCALER_ROBUST,
    config: Optional[Any] = None,
    model_type: Optional[str] = None
) -> Any:
    """Create preprocessing pipeline for DFM/DDFM model.
    
    IMPORTANT: For DDFM, this matches the original implementation which only standardizes data.
    The original DDFM does NOT apply differencing or other transformations - only standardization.
    Therefore, we create a simple pipeline with imputation and scaler only (no transformations).
    
    For DFM, we can optionally include transformations, but for DDFM we match the original.
    """
    try:
        from sktime.transformations.compose import TransformerPipeline
        from sktime.transformations.series.impute import Imputer
        from sklearn.preprocessing import StandardScaler, RobustScaler
    except ImportError as e:
        raise ImportError("sktime is required. Install with: pip install sktime[forecasting]") from e
    
    # For DDFM: Match original implementation - only standardization, no transformations
    # Original DDFM: self.data = (data - self.mean_z) / self.sigma_z (no differencing)
    # Determine model type from parameter or model class
    if model_type is None:
        if hasattr(model, '__class__'):
            model_type = model.__class__.__name__.lower()
        elif config is not None and hasattr(config, 'model_type'):
            model_type = str(config.model_type).lower()
    
    # Check if this is DDFM model
    is_ddfm = (model_type == 'ddfm' or 
               (hasattr(model, '__class__') and 'ddfm' in model.__class__.__name__.lower()))
    
    if is_ddfm:
        # For DDFM: Match original implementation - only standardization (no transformations)
        # Original DDFM does: (data - mean_z) / sigma_z
        # We use StandardScaler which does the same: (X - mean) / std
        logger.info("Creating DDFM preprocessing pipeline (standardization only, matching original implementation)")
        
        # #region agent log
        import json
        log_path = "/data/nowcasting-kr/.cursor/debug.log"
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "H1",
                "location": "train_dfm_python.py:184",
                "message": "Creating DDFM preprocessing pipeline",
                "data": {
                    "is_ddfm": True,
                    "scaler_type": scaler_type,
                    "has_transformations": False,
                    "pipeline_steps": ["impute_ffill", "impute_bfill", "scaler"]
                },
                "timestamp": int(__import__("time").time() * 1000)
            }) + "\n")
        # #endregion
        
        imputation_steps = [
            ('impute_ffill', Imputer(method="ffill")),
            ('impute_bfill', Imputer(method="bfill"))
        ]
        
        scaler_type_lower = scaler_type.lower() if scaler_type else SCALER_STANDARD  # Use StandardScaler to match original
        if scaler_type_lower == SCALER_ROBUST:
            scaler = RobustScaler()
        elif scaler_type_lower == SCALER_STANDARD:
            scaler = StandardScaler()
        else:
            scaler = StandardScaler()  # Default to StandardScaler to match original
        
        return TransformerPipeline(steps=imputation_steps + [('scaler', scaler)])
    
    # For DFM: Can include transformations if config provided
    if config is not None and hasattr(config, 'series') and config.series:
        try:
            from src.train.preprocess import create_transformer_from_config
            pipeline = create_transformer_from_config(config)
            logger.info("Created preprocessing pipeline with transformations from config (DFM)")
            return pipeline
        except Exception as e:
            logger.warning(f"Failed to create pipeline with transformations from config: {e}. Using simple pipeline.")
    
    # Fallback: simple pipeline without transformations
    imputation_steps = [
        ('impute_ffill', Imputer(method="ffill")),
        ('impute_bfill', Imputer(method="bfill"))
    ]
    
    scaler_type_lower = scaler_type.lower() if scaler_type else SCALER_ROBUST
    if scaler_type_lower == SCALER_ROBUST:
        scaler = RobustScaler()
    elif scaler_type_lower == SCALER_STANDARD:
        scaler = StandardScaler()
    else:
        logger.warning(f"Unknown scaler type '{scaler_type}', using RobustScaler")
        scaler = RobustScaler()
    
    return TransformerPipeline(steps=imputation_steps + [('scaler', scaler)])


# ============================================================================
# Training Functions
# ============================================================================

def train_dfm_python_model(
    model_type: str,
    config_name: str,
    cfg: DictConfig,
    data_file: str,
    model_name: Optional[str],
    horizons: Optional[List[int]],
    outputs_dir: Path,
    model_cfg_dict: Optional[dict] = None
) -> Dict[str, Any]:
    """Train dfm-python-based model (DFM or DDFM)."""
    # Load and filter data
    from src.utils import load_and_filter_data
    data_path = resolve_data_path(data_file) if data_file else resolve_data_path()
    data = load_and_filter_data(data_path, TRAIN_START, TRAIN_END)
    
    if len(data) == 0:
        raise ValidationError(f"No data available in training period (1985-2019). Data range: {data.index.min()} to {data.index.max()}")
    
    logger.info(f"Training data period: {data.index.min()} to {data.index.max()} (1985-2019 enforced)")
    logger.info(f"Data frequency: Weekly ({len(data)} points)")
    
    exp_cfg = get_experiment_cfg(cfg)
    target_series = exp_cfg.get('target_series')
    
    if not target_series or target_series not in data.columns:
        raise ValidationError(f"target_series '{target_series}' not found in data. Available columns: {list(data.columns)}")
    
    model_params = {}
    if 'model_overrides' in exp_cfg:
        model_overrides_dict = OmegaConf.to_container(exp_cfg.model_overrides, resolve=True)
        if isinstance(model_overrides_dict, dict) and model_type in model_overrides_dict:
            model_params = model_overrides_dict[model_type] or {}
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Training {model_type.upper()} model")
    logger.info(f"{'='*70}")
    logger.info(f"Config: {config_name}")
    logger.info(f"Data: {data_file}")
    
    if model_type == 'dfm':
        max_iter = model_params.get('max_iter', 5000)
        threshold = model_params.get('threshold', 1e-5)
        mixed_freq = model_params.get('mixed_freq', False)
        clock = model_params.get('clock', 'w')
        config_dict = model_cfg_dict if model_cfg_dict else {}
        if not config_dict or 'series' not in config_dict:
            raise ValidationError(f"No series found in config. Config: {config_name}")
        
        if clock is not None:
            config_dict['clock'] = clock
        
        y_train, available_series = prepare_multivariate_data(data, config_dict, cfg, target_series, model_type='dfm')
        clip_quantiles = model_params.get('clip_quantiles') if isinstance(model_params, dict) else None
        y_train = _apply_target_preprocessing(y_train, target_series, clip_quantiles)
        y_train = set_dataframe_frequency(y_train)
        logger.info(f"Max iterations: {max_iter}, Threshold: {threshold}, Series: {y_train.shape[1]}")
        
        try:
            from dfm_python.config import DFMConfig
            exp_cfg_for_dfm = get_experiment_cfg(cfg)
            if model_params:
                cfg_dict = OmegaConf.to_container(exp_cfg_for_dfm, resolve=True)
                cfg_dict.update(model_params)
                exp_cfg_for_dfm = OmegaConf.create(cfg_dict)
            
            dfm_config = DFMConfig.from_hydra(exp_cfg_for_dfm, config_path=str(project_root / "config"))
        except Exception as e:
            raise ValidationError(f"Failed to create DFMConfig: {e}") from e
        
        available_series_set = set(available_series)
        filtered_series = [s for s in dfm_config.series if s.series_id in available_series_set]
        if len(filtered_series) != len(dfm_config.series):
            dfm_config = type(dfm_config)(
                series=filtered_series,
                blocks=dfm_config.blocks,
                clock=clock,
                max_iter=dfm_config.max_iter if hasattr(dfm_config, 'max_iter') else max_iter,
                threshold=dfm_config.threshold if hasattr(dfm_config, 'threshold') else threshold,
                nan_method=dfm_config.nan_method if hasattr(dfm_config, 'nan_method') else 2,
                nan_k=dfm_config.nan_k if hasattr(dfm_config, 'nan_k') else 3,
                scaler=dfm_config.scaler if hasattr(dfm_config, 'scaler') else model_cfg_dict.get('scaler', SCALER_ROBUST) if model_cfg_dict else SCALER_ROBUST
            )
        
        time_index = _create_time_index_from_data(y_train)
        
        try:
            from dfm_python import DFM, DFMDataModule, DFMTrainer
        except ImportError as e:
            raise ImportError(f"dfm-python package not available: {e}") from e
        
        model = DFM(config=dfm_config, mixed_freq=mixed_freq, max_iter=max_iter, threshold=threshold)
        scaler_type = model_cfg_dict.get('scaler', SCALER_ROBUST) if model_cfg_dict else SCALER_ROBUST
        preprocessing_pipeline = _create_preprocessing_pipeline(model, scaler_type=scaler_type, config=dfm_config, model_type='dfm')
        
        data_module = DFMDataModule(
            config=dfm_config,
            pipeline=preprocessing_pipeline,
            data=y_train.values,
            time_index=time_index
        )
        data_module.setup()
        
        trainer = DFMTrainer(max_epochs=max_iter)
        trainer.fit(model, data_module)
        
        # Store entire preprocessing pipeline (not just scaler) so inverse_transform can handle transformations
        # The pipeline includes transformations (chg, log, etc.) that need to be inverse transformed
        if preprocessing_pipeline is not None:
            model.preprocess = preprocessing_pipeline  # Store full preprocessing pipeline
            logger.info(f"Stored preprocessing pipeline ({type(preprocessing_pipeline).__name__}) in DFM model for inverse transform")
        else:
            logger.warning("No preprocessing pipeline to store for DFM model")
        
        forecaster = model
        set_forecaster_attributes(forecaster, y_train)
    
    elif model_type == 'ddfm':
        epochs = model_params.get('epochs', 100)
        encoder_layers = model_params.get('encoder_layers', [16, 4])
        num_factors = model_params.get('num_factors', 1)
        learning_rate = model_params.get('learning_rate', 0.005)
        batch_size = model_params.get('batch_size', 100)
        activation = model_params.get('activation', 'relu')
        loss_function = model_params.get('loss_function', 'mse')
        huber_delta = model_params.get('huber_delta', 1.0)
        weight_decay = model_params.get('weight_decay', 0.0)
        grad_clip_val = model_params.get('grad_clip_val', 1.0)
        factor_order = model_params.get('factor_order', 1)
        mult_epoch_pretrain = model_params.get('mult_epoch_pretrain', 1)
        
        # Target-specific adjustments for KOEQUIPTE
        if target_series == 'KOEQUIPTE':
            if encoder_layers == [16, 4] or encoder_layers is None:
                encoder_layers = [64, 32, 16]
            if activation == 'relu' and 'activation' not in model_params:
                activation = 'tanh'
            if epochs == 100:
                epochs = 150
            if weight_decay == 0.0:
                weight_decay = 1e-4
            if mult_epoch_pretrain == 1:
                mult_epoch_pretrain = 2
            if batch_size >= 100:
                batch_size = 64
            logger.info(f"Applied KOEQUIPTE-specific settings: encoder={encoder_layers}, activation={activation}, epochs={epochs}, weight_decay={weight_decay}")
        
        config_dict = model_cfg_dict if model_cfg_dict else {}
        if not config_dict or 'series' not in config_dict:
            raise ValidationError(f"No series found in config. Config: {config_name}")
        
        y_train, available_series = prepare_multivariate_data(data, config_dict, cfg, target_series, model_type='ddfm')
        clip_quantiles = model_params.get('clip_quantiles') if isinstance(model_params, dict) else None
        y_train = _apply_target_preprocessing(y_train, target_series, clip_quantiles)
        y_train = set_dataframe_frequency(y_train)
        logger.info(f"Epochs: {epochs}, Encoder: {encoder_layers}, Factors: {num_factors}, Series: {y_train.shape[1]}")
        
        config_path = str(get_config_path())
        clock = model_params.get('clock', 'w')
        mixed_freq = model_params.get('mixed_freq', False)
        dfm_config = _build_dfm_config(
            cfg=cfg,
            available_series=available_series,
            model_config=config_dict,
            config_path=config_path,
            clock=clock,
            mixed_freq=mixed_freq
        )
        
        time_index = _create_time_index_from_data(y_train)
        
        try:
            from dfm_python import DDFM, DFMDataModule, DDFMTrainer
        except ImportError as e:
            raise ImportError(f"dfm-python package not available: {e}") from e
        
        ddfm_model = DDFM(
            config=dfm_config,
            encoder_layers=encoder_layers,
            num_factors=num_factors,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            loss_function=loss_function,
            huber_delta=huber_delta,
            activation=activation,
            weight_decay=weight_decay,
            grad_clip_val=grad_clip_val,
            factor_order=factor_order,
            mult_epoch_pretrain=mult_epoch_pretrain
        )
        
        scaler_type = model_cfg_dict.get('scaler', SCALER_ROBUST) if model_cfg_dict else SCALER_ROBUST
        preprocessing_pipeline = _create_preprocessing_pipeline(ddfm_model, scaler_type=scaler_type, config=dfm_config, model_type='ddfm')
        
        data_module = DFMDataModule(
            config=dfm_config,
            pipeline=preprocessing_pipeline,
            data=y_train.values,
            time_index=time_index
        )
        data_module.setup()
        
        # DDFM training happens entirely in on_train_start() via MCMC.
        # PyTorch Lightning should not run additional epochs, so set max_epochs=1.
        # The epochs parameter is used internally by MCMC (epochs_per_iter), not by Lightning.
        trainer = DDFMTrainer(max_epochs=1)
        logger.info(f"DDFM trainer created with max_epochs={trainer.max_epochs} (epochs={epochs} used internally as epochs_per_iter)")
        trainer.fit(ddfm_model, data_module)
        
        # Store entire preprocessing pipeline (not just scaler) so inverse_transform can handle transformations
        # The pipeline includes transformations (chg, log, etc.) that need to be inverse transformed
        if preprocessing_pipeline is not None:
            ddfm_model.preprocess = preprocessing_pipeline  # Store full preprocessing pipeline
            logger.info(f"Stored preprocessing pipeline ({type(preprocessing_pipeline).__name__}) in DDFM model for inverse transform")
        else:
            logger.warning("No preprocessing pipeline to store for DDFM model")
        
        forecaster = ddfm_model
        set_forecaster_attributes(forecaster, y_train)
    
    else:
        raise ValidationError(f"Unknown dfm-python model type: {model_type}")
    
    logger.info(f"{'='*70}\n")
    
    # Save checkpoint
    # Check if outputs_dir already ends with model_name to avoid redundant subdirectory
    # outputs_dir is already checkpoint_dir / checkpoint_model_name (e.g., checkpoints/KOWRCCNSE_dfm)
    # If model_name matches the directory name, use outputs_dir directly
    if model_name and (str(outputs_dir).endswith(model_name) or outputs_dir.name == model_name):
        model_dir = outputs_dir
        final_model_name = outputs_dir.name
    else:
        final_model_name = model_name or f"{model_type}_{target_series}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_dir = outputs_dir / final_model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / "model.pkl"
    
    # Prepare metadata (include scaler if available)
    metadata = {
        'model_type': model_type,
        'target_series': target_series,
        'config': OmegaConf.to_container(cfg, resolve=True),
        'training_data_index': list(y_train.index) if hasattr(y_train, 'index') else []
    }
    
    # Add preprocessing pipeline to metadata if available (for checkpoint loading)
    if hasattr(forecaster, 'preprocess') and forecaster.preprocess is not None:
        metadata['preprocess'] = forecaster.preprocess
        logger.info(f"Added preprocessing pipeline to metadata: {type(forecaster.preprocess).__name__}")
    
    # Save checkpoint using unified function
    from src.models import save_model_checkpoint
    try:
        save_model_checkpoint(forecaster, model_file, metadata)
    except Exception as e:
        raise ValidationError(f"Failed to save model checkpoint: {e}") from e
    
    # Extract metrics
    try:
        result = forecaster.get_result()
        result_metadata = forecaster.get_metadata()
    except (AttributeError, RuntimeError):
        result = None
        result_metadata = {}
    
    model_metrics = extract_model_metrics(forecaster)
    metrics = {
        'model_type': model_type,
        'forecast_metrics': {},
        'training_completed': True,
        **model_metrics
    }
    
    return {
        'status': 'completed',
        'model_name': final_model_name,
        'model_dir': str(model_dir),
        'metrics': metrics,
        'result': result,
        'metadata': result_metadata
    }
