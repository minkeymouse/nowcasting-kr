"""Training module - sktime and DFM/DDFM model training.

Supports ARIMA, VAR, DFM, DDFM models using sktime forecaster interface.
Preprocessing is handled by src.preprocessing module.
"""

from pathlib import Path
import sys
import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import numpy as np
import pandas as pd
import torch

# Path setup
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add dfm-python to path
dfm_python_path = project_root / "dfm-python" / "src"
if dfm_python_path.exists() and str(dfm_python_path) not in sys.path:
    sys.path.insert(0, str(dfm_python_path))

# Now use absolute imports
from src.utils import (
    extract_experiment_params,
    validate_experiment_config,
    ValidationError
)

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except ImportError as e:
    raise ImportError(f"Required dependencies not available: {e}")

# DFM and DDFM are imported from dfm_python when needed

# Set up logging (centralized configuration)
from src.utils import setup_logging
# Log directory will be set in main() function
setup_logging()

# Set up logger
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

def _get_experiment_cfg(cfg: DictConfig) -> DictConfig:
    """Extract experiment config section from Hydra config.
    
    Handles both cases where config is wrapped in 'experiment' key or is the config itself.
    
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object
        
    Returns
    -------
    DictConfig
        Experiment configuration section
    """
    return cfg.experiment if 'experiment' in cfg else cfg


def _get_cfg_value(cfg: DictConfig, key: str, default=None):
    """Get value from config (handles both DictConfig and dict).
    
    Note: For extracting multiple values, prefer using extract_experiment_params() from src.utils.
    """
    source_cfg = _get_experiment_cfg(cfg)
    if hasattr(source_cfg, 'get'):
        return source_cfg.get(key, default)
    return getattr(source_cfg, key, default)


def _convert_experiment_to_dfm_config_fallback(
    cfg: DictConfig,
    available_series_list: List[str],
    model_cfg_dict: dict,
    config_path: str,
    clock: str = 'w',
    mixed_freq: bool = True
) -> Any:
    """Convert experiment config to dfm-python DFMConfig format.
    
    This function creates a minimal config dict and uses DFMConfig.from_hydra()
    to handle series loading from individual config files when series are specified as strings.
    
    Parameters
    ----------
    cfg : DictConfig
        Experiment configuration
    available_series_list : List[str]
        List of series IDs that are actually in the data after filtering
    model_cfg_dict : dict
        Model configuration dictionary with series and blocks
    config_path : str
        Path to config directory (for loading series configs)
    clock : str, default 'w'
        Clock frequency for DFM
    mixed_freq : bool, default True
        Whether to use mixed frequency mode (unused, kept for compatibility)
        
    Returns
    -------
    DFMConfig
        dfm-python DFMConfig object ready for use
    """
    if DFMConfig is None:
        raise ImportError("dfm-python package not available. Install with: pip install dfm-python")
    
    # Get series list from cfg (Hydra already composed it)
    source_cfg = _get_experiment_cfg(cfg)
    series_ids_raw = source_cfg.get('series', [])
    # Convert ListConfig to regular list
    series_ids = OmegaConf.to_container(series_ids_raw, resolve=True) if series_ids_raw else []
    if not isinstance(series_ids, list):
        series_ids = []
    
    # Filter to only available series
    available_series_set = set(available_series_list)
    filtered_series_ids = [s for s in series_ids if s in available_series_set]
    
    if not filtered_series_ids:
        raise ValidationError(
            f"No series found in available_series_list. "
            f"Available: {available_series_list[:10]}..."
        )
    
    # Create config dict for DFMConfig.from_hydra()
    # series as list of strings will be loaded by _parse_series_list using config_path
    config_dict = {
        'series': filtered_series_ids,
        'clock': clock,
        'max_iter': model_cfg_dict.get('max_iter', 5000),
        'threshold': model_cfg_dict.get('threshold', 1e-5),
        'nan_method': model_cfg_dict.get('nan_method', 2),
        'nan_k': model_cfg_dict.get('nan_k', 3),
        'scaler': model_cfg_dict.get('scaler', 'robust')
    }
    
    # Get blocks from model_cfg_dict or create default
    blocks_dict = model_cfg_dict.get('blocks', {})
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
    
    # Use from_hydra with config_path (required when series is a list of strings)
    return DFMConfig.from_hydra(config_dict, config_path=config_path)


def _create_time_index_from_data(data: pd.DataFrame) -> Any:
    """Create dfm-python TimeIndex from DataFrame.
    
    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed data with DatetimeIndex or date column
        
    Returns
    -------
    TimeIndex
        dfm-python TimeIndex object
    """
    try:
        from dfm_python.utils.time import TimeIndex, parse_timestamp
    except ImportError:
        raise ImportError("dfm-python package not available. Install with: pip install dfm-python")
    
    # Extract dates from index or date column
    if isinstance(data.index, pd.DatetimeIndex):
        dates = data.index
    elif 'date' in data.columns:
        dates = pd.to_datetime(data['date'])
    else:
        # Create weekly dates starting from a default date
        n_periods = len(data)
        start_date = pd.Timestamp('1985-01-01')
        dates = pd.date_range(start=start_date, periods=n_periods, freq='W')
    
    # Convert to TimeIndex format
    time_list = [parse_timestamp(d.strftime('%Y-%m-%d')) for d in dates]
    return TimeIndex(time_list)


# Target-specific preprocessing (supports optional quantile clipping)
def _apply_target_specific_preprocessing(
    df: pd.DataFrame,
    target_series: str,
    clip_quantiles: Optional[List[float]] = None
) -> pd.DataFrame:
    """Apply extra preprocessing for specific targets."""
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


# Import preprocessing functions
from src.preprocessing import (
    resample_to_monthly,
    set_dataframe_frequency,
    impute_missing_values,
    prepare_multivariate_data,
    prepare_univariate_data
)


def _detect_model_type(model_name: str) -> Optional[str]:
    """Detect model type from model name."""
    valid_types = ['arima', 'var', 'dfm', 'ddfm']
    parts = model_name.lower().split('_')
    for part in parts:
        if part in valid_types:
            return part
    return None


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
    """Train model using sktime forecaster interface."""
    try:
        from sktime.forecasting.arima import ARIMA as SktimeARIMA
        from sktime.forecasting.var import VAR as SktimeVAR
        from sktime.transformations.series.impute import Imputer
    except ImportError as e:
        raise ImportError(f"sktime is required for {model_type} models. Install with: pip install sktime[forecasting]") from e
    
    data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    train_start = pd.Timestamp('1985-01-01')
    train_end = pd.Timestamp('2019-12-31')
    data = data[(data.index >= train_start) & (data.index <= train_end)]
    
    if len(data) == 0:
        raise ValidationError(f"No data available in training period (1985-2019). Data range: {data.index.min()} to {data.index.max()}")
    
    logger.info(f"Training data period: {data.index.min()} to {data.index.max()} (1985-2019 enforced)")
    logger.info(f"Data frequency: Weekly ({len(data)} points)")
    logger.info(f"All models will use weekly frequency (clock='w')")
    
    source_cfg = _get_experiment_cfg(cfg)
    # Use extract_experiment_params for consistent config extraction
    exp_params = extract_experiment_params(cfg)
    target_series = exp_params.get('target_series')
    
    if not target_series or target_series not in data.columns:
        raise ValidationError(f"target_series '{target_series}' not found in data. Available columns: {list(data.columns)}")
    
    model_params = {}
    if 'model_overrides' in source_cfg:
        model_overrides_dict = OmegaConf.to_container(source_cfg.model_overrides, resolve=True)
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
        clock = model_params.get('clock', 'w')  # Default to 'w' for weekly
        config_dict = model_cfg_dict if model_cfg_dict else {}
        if not config_dict or 'series' not in config_dict:
            raise ValidationError(f"No series found in config. Config: {config_name}")
        
        # Add clock to config_dict if provided (for prepare_multivariate_data to check)
        if clock is not None:
            config_dict['clock'] = clock
        
        exp_params = extract_experiment_params(cfg)
        target_series = exp_params.get('target_series')
        # Preprocess multivariate data for DFM (weekly if clock='w', monthly otherwise)
        y_train, available_series_list = prepare_multivariate_data(data, config_dict, cfg, target_series, model_type='dfm')
        clip_quantiles = model_params.get('clip_quantiles') if isinstance(model_params, dict) else None
        y_train = _apply_target_specific_preprocessing(y_train, target_series, clip_quantiles)
        y_train = set_dataframe_frequency(y_train)
        logger.info(f"Max iterations: {max_iter}, Threshold: {threshold}, Series: {y_train.shape[1]} (filtered from config, target_series included)")
        
        # Use @hydra.main injected cfg directly (Hydra already composed config with defaults)
        # Convert to dfm-python DFMConfig format
        try:
            from dfm_python.config import DFMConfig
            # Extract model_overrides.dfm section and merge with base config
            source_cfg = _get_experiment_cfg(cfg)
            model_overrides_raw = source_cfg.get('model_overrides', {})
            model_overrides = OmegaConf.to_container(model_overrides_raw, resolve=True) if model_overrides_raw else {}
            if not isinstance(model_overrides, dict):
                model_overrides = {}
            dfm_overrides = model_overrides.get('dfm', {}) or {}
            
            # Use DFMConfig.from_hydra() directly with Hydra DictConfig
            # from_hydra() accepts DictConfig and handles conversion internally
            # Extract only experiment config (not the full Hydra cfg which includes 'experiment' key)
            source_cfg = _get_experiment_cfg(cfg)
            
            # Merge dfm_overrides into source_cfg
            if dfm_overrides:
                # Convert to dict, merge, then convert back to DictConfig
                source_dict = OmegaConf.to_container(source_cfg, resolve=True)
                source_dict.update(dfm_overrides)
                source_cfg = OmegaConf.create(source_dict)
            
            # Convert using DFMConfig.from_hydra() - accepts DictConfig directly
            # This will handle series: [strings] via _parse_series_list with config_path
            dfm_config = DFMConfig.from_hydra(
                source_cfg,  # Pass DictConfig directly
                config_path=str(project_root / "config")
            )
        except Exception as e:
            raise ValidationError(f"Failed to create DFMConfig: {e}") from e
        
        # Filter series to only available ones (after conversion)
        available_series_set = set(available_series_list)
        filtered_series = [
            s for s in dfm_config.series 
            if s.series_id in available_series_set
        ]
        if len(filtered_series) != len(dfm_config.series):
            # Create new config with filtered series
            dfm_config = type(dfm_config)(
                series=filtered_series,
                blocks=dfm_config.blocks,
                clock=clock,
                max_iter=dfm_config.max_iter if hasattr(dfm_config, 'max_iter') else max_iter,
                threshold=dfm_config.threshold if hasattr(dfm_config, 'threshold') else threshold,
                nan_method=dfm_config.nan_method if hasattr(dfm_config, 'nan_method') else 2,
                nan_k=dfm_config.nan_k if hasattr(dfm_config, 'nan_k') else 3,
                scaler=dfm_config.scaler if hasattr(dfm_config, 'scaler') else model_cfg_dict.get('scaler', 'robust')
            )
        
        # Create time index
        time_index = _create_time_index_from_data(y_train)
        
        # Use dfm-python standard pattern
        try:
            from dfm_python import DFM, DFMDataModule, DFMTrainer
        except ImportError as e:
            raise ImportError(f"dfm-python package not available: {e}") from e
        
        # Create model with config
        # Note: We convert experiment config to dfm-python format because:
        # 1. Experiment config has different structure (experiment/ wrapper, series in separate files)
        # 2. dfm-python expects dfm-python config format (series list, blocks dict)
        # 3. We need to filter series based on available data
        # Alternative: Could use model.load_config(hydra=cfg) but would still need conversion
        model = DFM(
            config=dfm_config,
            mixed_freq=mixed_freq,
            max_iter=max_iter,
            threshold=threshold
        )
        
        # Create preprocessing pipeline with scaler (for Mx/Wx extraction and inverse transform)
        from src.models.models_forecasters import _create_preprocessing_pipeline
        # Get scaler type from config (default: 'robust')
        scaler_type = model_cfg_dict.get('scaler', 'robust') if model_cfg_dict else 'robust'
        preprocessing_pipeline = _create_preprocessing_pipeline(model, scaler_type=scaler_type)
        
        # Create DataModule with pipeline (for scaler statistics extraction)
        # Data is already preprocessed, but pipeline is needed for Mx/Wx extraction
        data_module = DFMDataModule(
            config=dfm_config,
            pipeline=preprocessing_pipeline,
            data=y_train.values,
            time_index=time_index
        )
        data_module.setup()
        
        # Use trainer.fit() - now safe because training_step uses torch.no_grad() for EM algorithm
        # This is the proper way to use PyTorch Lightning while avoiding memory issues
        # Root cause fix: training_step now wraps self.em() in torch.no_grad() to prevent
        # PyTorch from creating gradient graphs even when automatic_optimization=False
        trainer = DFMTrainer(max_epochs=max_iter)
        trainer.fit(model, data_module)
        
        # Extract scaler from pipeline and store in model for inverse transform
        # This enables predict() to use scaler.inverse_transform() for unstandardization
        try:
            from dfm_python.lightning.data_module import _get_scaler
            scaler = _get_scaler(preprocessing_pipeline)
            if scaler is not None:
                model.scaler = scaler
                logger.debug(f"Stored scaler ({type(scaler).__name__}) in DFM model for inverse transform")
        except Exception as e:
            logger.warning(f"Could not extract scaler from pipeline for DFM model: {e}")
        
        # Use model directly as forecaster (DFM model has predict method)
        forecaster = model
        # Add compatibility attributes for evaluation code
        if not hasattr(forecaster, 'is_fitted'):
            forecaster.is_fitted = True
        if not hasattr(forecaster, '_y'):
            forecaster._y = y_train
        
    elif model_type == 'ddfm':
        epochs = model_params.get('epochs', 100)
        encoder_layers = model_params.get('encoder_layers', [16, 4])  # Updated to match original DDFM default
        num_factors = model_params.get('num_factors', 1)
        learning_rate = model_params.get('learning_rate', 0.005)  # Updated to match original DDFM default
        batch_size = model_params.get('batch_size', 100)  # Updated to match original DDFM default
        
        # Target-specific encoder architecture improvements
        # KOEQUIPTE shows identical performance to DFM, suggesting encoder needs more capacity
        # Use larger/deeper encoder for targets that may benefit from more nonlinear capacity
        exp_params = extract_experiment_params(cfg)
        target_series = exp_params.get('target_series')
        activation = model_params.get('activation', 'relu')  # Default activation function
        
        if target_series == 'KOEQUIPTE':
            # KOEQUIPTE: Use deeper encoder and different activation to capture nonlinear relationships
            # Current encoder may be learning only linear features (identical to DFM performance)
            # Improvements:
            # 1. Use deeper encoder [64, 32, 16] instead of default [16, 4] for more capacity
            # 2. Use 'tanh' activation instead of 'relu' to better capture negative correlations
            #    (ReLU zeros negative values, which may prevent learning negative factor relationships)
            if encoder_layers == [16, 4] or encoder_layers is None:
                encoder_layers = [64, 32, 16]
                logger.info(f"Using target-specific encoder architecture for {target_series}: {encoder_layers}")
            # Use tanh activation for KOEQUIPTE (unless explicitly overridden in config)
            if activation == 'relu' and 'activation' not in model_params:
                activation = 'tanh'
                logger.info(f"Using tanh activation for {target_series} to better capture negative correlations")
            # Also increase epochs slightly for more complex architecture
            if epochs == 100:
                epochs = 150
                logger.info(f"Increased epochs to {epochs} for {target_series} with deeper encoder")
        
        # Loss function configuration (default: 'mse', can use 'huber' for robustness to outliers)
        loss_function = model_params.get('loss_function', 'mse')
        huber_delta = model_params.get('huber_delta', 1.0)
        
        # Regularization and training stability improvements
        # Weight decay helps prevent encoder from collapsing to linear behavior
        weight_decay = model_params.get('weight_decay', 0.0)
        if target_series == 'KOEQUIPTE':
            # Use weight decay for KOEQUIPTE to prevent linear collapse
            if weight_decay == 0.0:
                weight_decay = 1e-4  # Small weight decay to encourage nonlinear features
                logger.info(f"Using weight_decay={weight_decay} for {target_series} to prevent linear collapse")
        
        # Gradient clipping for training stability (default: 1.0)
        grad_clip_val = model_params.get('grad_clip_val', 1.0)
        
        # Factor order configuration (VAR lag order for factor dynamics: 1 or 2)
        # VAR(2) can capture longer-term dependencies but requires more data
        # For KOEQUIPTE with deeper encoder, VAR(2) might help capture more complex dynamics
        factor_order = model_params.get('factor_order', 1)
        if target_series == 'KOEQUIPTE':
            # Consider VAR(2) for KOEQUIPTE to capture longer-term dependencies
            # This may help if the series has complex multi-period dynamics
            if factor_order == 1:
                # Keep VAR(1) as default, but allow override via config
                # VAR(2) requires more data and may not always improve performance
                pass  # Use default VAR(1) for now, can be overridden in config if needed
        
        # Pre-training configuration
        # Increased pre-training helps encoder learn better nonlinear features before MCMC
        mult_epoch_pretrain = model_params.get('mult_epoch_pretrain', 1)
        if target_series == 'KOEQUIPTE':
            # Increase pre-training for KOEQUIPTE to help encoder learn nonlinear features
            # More pre-training epochs give encoder more time to learn before MCMC starts
            if mult_epoch_pretrain == 1:
                mult_epoch_pretrain = 2  # Double pre-training epochs for KOEQUIPTE
                logger.info(f"Increased mult_epoch_pretrain to {mult_epoch_pretrain} for {target_series} to improve encoder learning")
        
        # Batch size optimization for KOEQUIPTE
        # Smaller batch sizes can improve gradient diversity and help escape linear solutions
        if target_series == 'KOEQUIPTE' and batch_size >= 100:
            # Use smaller batch size for KOEQUIPTE to improve gradient diversity
            # This can help encoder learn nonlinear features instead of collapsing to linear
            batch_size = 64  # Smaller batch size for better gradient diversity
            logger.info(f"Using smaller batch_size={batch_size} for {target_series} to improve gradient diversity")
        
        config_dict = model_cfg_dict if model_cfg_dict else {}
        if not config_dict or 'series' not in config_dict:
            raise ValidationError(f"No series found in config. Config: {config_name}")
        
        exp_params = extract_experiment_params(cfg)
        target_series = exp_params.get('target_series')
        # Preprocess multivariate data for DDFM (resample to monthly, set frequency)
        y_train, available_series_list = prepare_multivariate_data(data, config_dict, cfg, target_series, model_type='ddfm')
        clip_quantiles = model_params.get('clip_quantiles') if isinstance(model_params, dict) else None
        y_train = _apply_target_specific_preprocessing(y_train, target_series, clip_quantiles)
        y_train = set_dataframe_frequency(y_train)
        logger.info(f"Epochs: {epochs}, Encoder layers: {encoder_layers}, Activation: {activation}, Factors: {num_factors}, Loss: {loss_function}, Series: {y_train.shape[1]} (filtered from config, target_series included)")
        
        # Convert experiment config to dfm-python DFMConfig (DDFM uses same config structure)
        config_path = str(project_root / "config")
        clock = model_params.get('clock', 'w')  # Default to weekly, but should come from Hydra config
        mixed_freq = model_params.get('mixed_freq', False)
        dfm_config = _convert_experiment_to_dfm_config_fallback(
            cfg=cfg,
            available_series_list=available_series_list,
            model_cfg_dict=config_dict,
            config_path=config_path,
            clock=clock,
            mixed_freq=mixed_freq
        )
        
        # Create time index
        time_index = _create_time_index_from_data(y_train)
        
        # Use dfm-python standard pattern
        try:
            from dfm_python import DDFM, DFMDataModule, DDFMTrainer
        except ImportError as e:
            raise ImportError(f"dfm-python package not available: {e}") from e
        
        # Create DDFM model
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
        
        # Create preprocessing pipeline with scaler (for Mx/Wx extraction and inverse transform)
        from src.models.models_forecasters import _create_preprocessing_pipeline
        # Get scaler type from config (default: 'robust')
        scaler_type = model_cfg_dict.get('scaler', 'robust') if model_cfg_dict else 'robust'
        preprocessing_pipeline = _create_preprocessing_pipeline(ddfm_model, scaler_type=scaler_type)
        
        # Create DataModule with pipeline (for scaler statistics extraction)
        # Data is already preprocessed, but pipeline is needed for Mx/Wx extraction
        data_module = DFMDataModule(
            config=dfm_config,
            pipeline=preprocessing_pipeline,
            data=y_train.values,
            time_index=time_index
        )
        data_module.setup()
        
        # Create trainer and fit
        trainer = DDFMTrainer(max_epochs=epochs)
        trainer.fit(ddfm_model, data_module)
        
        # Extract scaler from pipeline and store in model for inverse transform
        # This enables predict() to use scaler.inverse_transform() for unstandardization
        try:
            from dfm_python.lightning.data_module import _get_scaler
            scaler = _get_scaler(preprocessing_pipeline)
            if scaler is not None:
                ddfm_model.scaler = scaler
                logger.debug(f"Stored scaler ({type(scaler).__name__}) in DDFM model for inverse transform")
        except Exception as e:
            logger.warning(f"Could not extract scaler from pipeline for DDFM model: {e}")
        
        # Use model directly as forecaster (DDFM model has predict method)
        forecaster = ddfm_model
        # Add compatibility attributes for evaluation code
        if not hasattr(forecaster, 'is_fitted'):
            forecaster.is_fitted = True
        if not hasattr(forecaster, '_y'):
            forecaster._y = y_train
        
    elif model_type == 'arima':
        # Use models/arima module
        from src.models import train_arima
        
        checkpoint_dir_temp = outputs_dir / "temp_arima"
        checkpoint_dir_temp.mkdir(parents=True, exist_ok=True)
        
        forecaster, _ = train_arima(
            data=data,
            target_series=target_series,
            config=model_params,
            checkpoint_dir=checkpoint_dir_temp
        )
        
        # Remove temp checkpoint (will save properly later)
        import shutil
        if checkpoint_dir_temp.exists():
            shutil.rmtree(checkpoint_dir_temp)
        
    elif model_type == 'var':
        # Use models/var module
        from src.models import train_var
        
        checkpoint_dir_temp = outputs_dir / "temp_var"
        checkpoint_dir_temp.mkdir(parents=True, exist_ok=True)
        
        forecaster, _ = train_var(
            data=data,
            target_series=target_series,
            config=model_params,
            cfg=cfg,
            checkpoint_dir=checkpoint_dir_temp
        )
        
        # Remove temp checkpoint (will save properly later)
        import shutil
        if checkpoint_dir_temp.exists():
            shutil.rmtree(checkpoint_dir_temp)
    
    # Extract horizons from config if not provided
    if horizons is None:
        exp_params = extract_experiment_params(cfg)
        horizons = exp_params.get('horizons', list(range(1, 23)))
    
    # For ARIMA/VAR, models are already trained by train_arima/train_var
    # For DFM/DDFM, forecaster is already fitted
    # Only fit if forecaster is not fitted yet
    if not hasattr(forecaster, 'is_fitted') or not forecaster.is_fitted:
        # Get y_train for fitting (needed for DFM/DDFM which create forecaster wrapper)
        if model_type in ['dfm', 'ddfm']:
            # y_train already prepared in DFM/DDFM blocks above
            if 'y_train' in locals():
                forecaster.fit(y_train)
        else:
            # For ARIMA/VAR, should not reach here as they're already fitted
            logger.warning(f"Forecaster for {model_type} not fitted, but should be already fitted")
    
    logger.info(f"{'='*70}\n")
    
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
        
        logger.info(f"Saved model checkpoint: {model_file}")
    except Exception as e:
        # Clean up temp file if save failed
        if temp_file and temp_file.exists():
            temp_file.unlink()
        raise ValidationError(f"Failed to save model checkpoint: {e}") from e
    
    # Extract training metrics (DFM/DDFM only)
    result = None
    metadata = {}
    if model_type in ['dfm', 'ddfm']:
        # For DFM/DDFM, forecaster is the model itself
        try:
            result = forecaster.get_result()
            metadata = forecaster.get_metadata()
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
    cfg: DictConfig,
    model_name: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    horizons: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Train a single forecasting model using Hydra configuration.
    
    This function provides a unified interface for training all model types
    (ARIMA, VAR, DFM, DDFM) using sktime forecaster interface. The model is
    trained, evaluated on specified horizons, and saved to checkpoint/.
    
    Parameters
    ----------
    cfg : DictConfig
        Hydra config (can be injected by @hydra.main or passed directly)
    model_name : Optional[str]
        Model name to train (e.g., arima, var, dfm, ddfm)
    checkpoint_dir : Optional[str]
        Directory to save checkpoint
    horizons : Optional[List[int]]
        Forecast horizons (if not specified, uses config value)
    """
    # Extract parameters from cfg (Hydra automatically loads experiment config)
    source_cfg = _get_experiment_cfg(cfg)
    config_name = getattr(cfg, '_name_', None) or cfg.get('_name_', 'experiment/consumption_kowrccnse_report')
    checkpoint_model_name = cfg.get('checkpoint_model_name')
    
    # Get model_name from cfg if not provided as parameter
    if not model_name:
        model_name = cfg.get('model')
    
    # Get checkpoint_dir from cfg if not provided as parameter
    if not checkpoint_dir:
        checkpoint_dir = cfg.get('checkpoint_dir', 'checkpoint')
    if not checkpoint_dir:
        raise ValueError("checkpoint_dir must be specified in config or via override.")
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
    
    # Config is already loaded by @hydra.main, no need for hydra.compose()
    # cfg already contains the composed config with defaults resolved
    if OmegaConf.is_struct(cfg):
        OmegaConf.set_struct(cfg, False)
    if 'experiment' in cfg and OmegaConf.is_struct(cfg.experiment):
        OmegaConf.set_struct(cfg.experiment, False)
    
    config_path = str(project_root / "config")
    
    # Detect model type from model_name
    if model_name:
        model_type = _detect_model_type(model_name)
        if not model_type:
            raise ValueError(f"Cannot detect model type from name: {model_name}")
    else:
        raise ValueError("model_name is required (set via parameter or cfg.model)")
    
    # Extract model config from Hydra-composed cfg
    # Hydra already merged defaults (e.g., /model: dfm) with experiment config
    source_cfg = _get_experiment_cfg(cfg)
    model_overrides_raw = source_cfg.get('model_overrides', {})
    model_overrides = OmegaConf.to_container(model_overrides_raw, resolve=True) if model_overrides_raw else {}
    if not isinstance(model_overrides, dict):
        model_overrides = {}
    
    # Get model-specific overrides
    model_specific_overrides = model_overrides.get(model_type, {}) or {}
    
    # Build model_cfg_dict from overrides (Hydra already composed defaults)
    # Include series from experiment config (needed for DFM/DDFM)
    model_cfg_dict = model_specific_overrides.copy()
    # Add series from experiment config (required for DFM/DDFM)
    # Convert ListConfig to regular list using OmegaConf.to_container
    if 'series' in source_cfg:
        series_raw = source_cfg.get('series', [])
        model_cfg_dict['series'] = OmegaConf.to_container(series_raw, resolve=True) if series_raw else []
    
    # Extract data path from config
    data_file = source_cfg.get('data_path')
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
    horizons: List[int] = list(range(1, 25)),  # 24 monthly horizons: 1 horizon = 1 month (4 weeks), 24 months = 2 years
    data_path: Optional[str] = None,
    config_dir: Optional[str] = None,
    config_name: Optional[str] = None,
    config_overrides: Optional[List[str]] = None,
    checkpoint_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Train multiple models and compare their performance."""
    project_root = Path(__file__).parent.parent.resolve()
    config_dir = config_dir or str(project_root / "config")
    if data_path is None:
        data_path = str(project_root / "data" / "data.csv")
        if not Path(data_path).exists():
            data_path = str(project_root / "data" / "sample_data.csv")
    
    # Use fixed directory name (overwrite mode) - one result per experiment
    output_dir = project_root / "outputs" / "comparisons" / target_series
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info(f"Comparing models for {target_series}")
    logger.info(f"Models: {', '.join(models)} | Horizons: {horizons}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 70)
    
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
        # Use Hydra to load config
        import hydra
        with hydra.initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            cfg = hydra.compose(config_name=config_name, overrides=config_overrides or [])
            if OmegaConf.is_struct(cfg):
                OmegaConf.set_struct(cfg, False)
            source_cfg = _get_experiment_cfg(cfg)
            horizons = source_cfg.get('forecast_horizons', list(range(1, 25)))  # Default: 24 months (2024-01 to 2025-12)
        logger.info(f"Extracted horizons from config: {len(horizons)} horizons ({min(horizons)}-{max(horizons)})")
    
    for i, model_name in enumerate(models, 1):
        logger.info(f"[{i}/{len(models)}] {model_name.upper()}...")
        
        if config_overrides is None:
            config_overrides = []
        
        try:
            # Get checkpoint_dir from parameter or config_overrides
            checkpoint_dir_local = checkpoint_dir
            if checkpoint_dir_local is None and config_overrides:
                for override in config_overrides:
                    if override.startswith('checkpoint_dir=') or override.startswith('+checkpoint_dir='):
                        checkpoint_dir_local = override.split('=', 1)[1]
                        break
            
            # Default to 'checkpoints' if not specified
            if checkpoint_dir_local is None:
                checkpoint_dir_local = 'checkpoints'
            
            checkpoint_path = None
            if checkpoint_dir_local:
                checkpoint_path = Path(checkpoint_dir_local) / f"{target_series}_{model_name}" / "model.pkl"
            
            if checkpoint_path and checkpoint_path.exists():
                logger.info(f"Loading model from checkpoint: {checkpoint_path}")
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
                if forecaster is None:
                    logger.warning(f"Forecaster is None in checkpoint for {target_series}_{model_name}")
                    result['status'] = 'failed'
                    result['error'] = 'Forecaster is None in checkpoint'
                    failed_models.append(model_name)
                    model_results[model_name] = result
                    continue
                
                # Check if forecaster is already fitted (should be from checkpoint)
                is_fitted = False
                if hasattr(forecaster, 'is_fitted'):
                    # Check if it's a method (callable) or property
                    if callable(forecaster.is_fitted):
                        try:
                            is_fitted = forecaster.is_fitted()
                        except Exception:
                            # If calling fails, try as property/attribute
                            is_fitted = bool(forecaster.is_fitted) if not callable(forecaster.is_fitted) else False
                    else:
                        is_fitted = bool(forecaster.is_fitted)
                elif hasattr(forecaster, '_is_fitted'):
                    is_fitted = bool(forecaster._is_fitted)
                elif hasattr(forecaster, '_fitted_forecaster'):
                    is_fitted = forecaster._fitted_forecaster is not None
                elif hasattr(forecaster, '_y'):
                    is_fitted = forecaster._y is not None
                
                if not is_fitted:
                    logger.warning(f"Forecaster from checkpoint for {target_series}_{model_name} appears not to be fitted. This may cause issues during evaluation.")
                
                if forecaster and horizons:
                    logger.info(f"Evaluating {model_name.upper()} model on {len(horizons)} horizons")
                    actual_data_path = data_path
                    if not actual_data_path or not Path(actual_data_path).exists():
                        actual_data_path = str(project_root / "data" / "data.csv")
                        if not Path(actual_data_path).exists():
                            actual_data_path = str(project_root / "data" / "sample_data.csv")
                    
                    # Load full data to get both training and test periods
                    full_data = pd.read_csv(actual_data_path, index_col=0, parse_dates=True)
                    
                    # Training period: 1985-2019
                    train_start = pd.Timestamp('1985-01-01')
                    train_end = pd.Timestamp('2019-12-31')
                    train_data = full_data[(full_data.index >= train_start) & (full_data.index <= train_end)]
                    train_data = resample_to_monthly(train_data)
                    
                    # Test period: 2024-2025 (actual future data, not split from training)
                    # 2024-01 to 2025-10 = 22 months, but we predict 24 months for extended forecast
                    test_start = pd.Timestamp('2024-01-01')
                    test_end = pd.Timestamp('2025-10-31')  # Actual data available until here
                    test_data = full_data[(full_data.index >= test_start) & (full_data.index <= test_end)]
                    test_data = resample_to_monthly(test_data)
                    
                    # Validate train-test split to prevent data leakage
                    if len(train_data) == 0:
                        raise ValidationError(f"No training data available in period 1985-2019. Data range: {full_data.index.min()} to {full_data.index.max()}")
                    if len(test_data) == 0:
                        raise ValidationError(f"No test data available in period 2024-2025. Data range: {full_data.index.min()} to {full_data.index.max()}")
                    if train_data.index.max() >= test_data.index.min():
                        raise ValidationError(f"Data leakage detected: Training period ends at {train_data.index.max()} but test period starts at {test_data.index.min()}. There must be a gap between training and test periods.")
                    
                    logger.info(f"Train period: {train_data.index.min()} to {train_data.index.max()} ({len(train_data)} points)")
                    logger.info(f"Test period: {test_data.index.min()} to {test_data.index.max()} ({len(test_data)} points)")
                    
                    # VAR: no automatic imputation (was masking signal). Drop rows with NaN.
                    if model_name.lower() == 'var':
                        train_nan = train_data.isnull().sum().sum()
                        test_nan = test_data.isnull().sum().sum()
                        if train_nan > 0:
                            logger.warning(f"VAR: Dropping {train_nan} missing values from training data (row-wise dropna).")
                            train_data = train_data.dropna()
                        if test_nan > 0:
                            logger.warning(f"VAR: Dropping {test_nan} missing values from test data (row-wise dropna).")
                            test_data = test_data.dropna()
                        if len(train_data) == 0:
                            raise ValidationError("VAR: Training data empty after dropna; cannot evaluate.")
                        if len(test_data) == 0:
                            raise ValidationError("VAR: Test data empty after dropna; cannot evaluate.")
                    
                    # For ARIMA/VAR: These models are not state-based, so they only need
                    # the last few periods (model specification: ARIMA(p,d,q) needs p+d lags,
                    # VAR(p) needs p lags) before the forecast start point.
                    # Recursive forecasting is handled automatically by sktime.
                    # No bridge data needed - gaps in timeline are not a problem.
                    if model_name.lower() in ['arima', 'var']:
                        y_train_eval = train_data
                        logger.info(f"{model_name.upper()} will use recursive forecasting from training data end ({train_data.index.max()}) to test period start ({test_start})")
                    else:
                        y_train_eval = train_data
                    
                    y_test_eval = test_data
                    
                    # For DFM/DDFM, extract recent data (2020-2023) for factor state update
                    # This period is between training (1985-2019) and test (2024-2025)
                    # Use window=None (history=None) to include full period for state update
                    # IMPORTANT: Apply the same preprocessing pipeline as training
                    y_recent_eval = None
                    if model_name.lower() in ['dfm', 'ddfm']:
                        recent_start = pd.Timestamp('2020-01-01')
                        recent_end = pd.Timestamp('2023-12-31')
                        recent_data = full_data[(full_data.index >= recent_start) & (full_data.index <= recent_end)]
                        if len(recent_data) > 0:
                            # Get the actual DFM/DDFM model to access its config
                            dfm_model = forecaster
                            if hasattr(forecaster, '_dfm_model'):
                                dfm_model = forecaster._dfm_model
                            elif hasattr(forecaster, '_ddfm_model'):
                                dfm_model = forecaster._ddfm_model
                            
                            # Get model config to determine which series and preprocessing were used
                            if hasattr(dfm_model, 'config') and hasattr(dfm_model.config, 'series'):
                                # Get series IDs that the model was trained on
                                trained_series_ids = [s.series_id for s in dfm_model.config.series]
                                
                                # Get clock frequency from model config
                                clock = getattr(dfm_model.config, 'clock', 'm')
                                should_resample = (clock != 'w')
                                
                                # Filter to trained series only
                                available_series = [s for s in trained_series_ids if s in recent_data.columns]
                                if target_series and target_series in recent_data.columns and target_series not in available_series:
                                    available_series.append(target_series)
                                
                                if len(available_series) > 0:
                                    selected_data = recent_data[available_series].dropna(how='all')
                                    
                                    # Apply transformations from series config files (same as training)
                                    from src.preprocessing import apply_transformations
                                    # Use project_root from outer scope or get it
                                    if 'project_root' not in locals() and 'project_root' not in globals():
                                        project_root = Path(__file__).parent.parent.resolve()
                                    config_path = str(project_root / "config")
                                    selected_data = apply_transformations(selected_data, config_path=config_path, series_ids=available_series)
                                    
                                    # Resample if needed (same logic as training)
                                    # For clock='w' with monthly series (mixed_freq=True):
                                    # Monthly data needs to be converted to weekly frequency
                                    # with monthly values placed at appropriate weekly positions
                                    if should_resample:
                                        processed_data = resample_to_monthly(selected_data)
                                    else:
                                        # clock='w' but series are monthly - need to convert to weekly
                                        # Check if data is monthly and needs conversion
                                        if hasattr(dfm_model, 'mixed_freq') and dfm_model.mixed_freq:
                                            # Check if series are monthly
                                            monthly_series = [s.series_id for s in dfm_model.config.series 
                                                            if hasattr(s, 'frequency') and s.frequency == 'm']
                                            if len(monthly_series) > 0 and clock == 'w':
                                                # Convert monthly data to weekly frequency
                                                from src.preprocessing import resample_to_weekly
                                                processed_data = resample_to_weekly(selected_data)
                                                logger.info(f"Converted monthly data to weekly frequency for {len(monthly_series)} monthly series")
                                            else:
                                                processed_data = selected_data
                                        else:
                                            processed_data = selected_data
                                    
                                    # Apply target-specific preprocessing (same as training)
                                    # Load config if not already loaded (use cached version if available)
                                    if 'cfg' not in locals():
                                        try:
                                            import hydra
                                            from hydra.core.global_hydra import GlobalHydra
                                            # Clear existing Hydra instance if needed
                                            if GlobalHydra.instance().is_initialized():
                                                GlobalHydra.instance().clear()
                                            with hydra.initialize_config_dir(config_dir=config_dir, version_base="1.3"):
                                                cfg = hydra.compose(config_name=config_name, overrides=config_overrides or [])
                                                if OmegaConf.is_struct(cfg):
                                                    OmegaConf.set_struct(cfg, False)
                                        except Exception as e:
                                            logger.warning(f"Failed to load config for preprocessing: {e}, using defaults")
                                            cfg = None
                                    
                                    if cfg is not None:
                                        source_cfg = _get_experiment_cfg(cfg)
                                    else:
                                        source_cfg = None
                                    
                                    if source_cfg is not None:
                                        model_overrides = OmegaConf.to_container(source_cfg.get('model_overrides', {}), resolve=True) if source_cfg else {}
                                        dfm_overrides = model_overrides.get(model_name.lower(), {}) or {}
                                        clip_quantiles = dfm_overrides.get('clip_quantiles')
                                    else:
                                        clip_quantiles = None
                                    processed_data = _apply_target_specific_preprocessing(processed_data, target_series, clip_quantiles)
                                    
                                    # Ensure column order matches model config series order
                                    # Model expects series in the same order as config.series
                                    final_series_order = []
                                    for s in dfm_model.config.series:
                                        if s.series_id in processed_data.columns:
                                            final_series_order.append(s.series_id)
                                    # Add any remaining columns (e.g., target_series if not in config)
                                    for col in processed_data.columns:
                                        if col not in final_series_order:
                                            final_series_order.append(col)
                                    
                                    y_recent_eval = processed_data[final_series_order].copy()
                                    y_recent_eval = set_dataframe_frequency(y_recent_eval)
                                    
                                    logger.info(f"Extracted {len(y_recent_eval)} recent periods ({recent_start} to {recent_end}) for {model_name.upper()} state update")
                                    logger.info(f"  Applied same preprocessing as training: {len(available_series)} series, clock={clock}, transformations applied")
                                else:
                                    logger.warning(f"No matching series found in recent data for {model_name.upper()}. Trained series: {trained_series_ids[:5]}...")
                            else:
                                # Fallback: Use simple resampling and column matching (old behavior)
                                logger.warning(f"{model_name.upper()} model missing config, using fallback preprocessing")
                                recent_data_monthly = resample_to_monthly(recent_data)
                                if isinstance(y_train_eval, pd.DataFrame):
                                    available_cols = [col for col in y_train_eval.columns if col in recent_data_monthly.columns]
                                    if len(available_cols) > 0:
                                        y_recent_eval = recent_data_monthly[available_cols].copy()
                                        y_recent_eval = y_recent_eval[y_train_eval.columns]
                        else:
                            logger.warning(f"No recent data available ({recent_start} to {recent_end}) for {model_name.upper()} update")
                    
                    logger.info(f"Evaluation data: train={len(y_train_eval)} points ({y_train_eval.index.min()} to {y_train_eval.index.max()}), test={len(y_test_eval)} points ({test_start} to {test_end})")
                    if y_recent_eval is not None:
                        logger.info(f"Recent data for update: {len(y_recent_eval)} points (2020-01-01 to 2023-12-31, window=None)")
                    
                    # For DFM/DDFM: Update model state with recent data (2020-2023) before forecasting
                    # Training period (2019) and forecast period (2024-2025) are disconnected,
                    # so we need to update the factor state using the gap period data
                    if model_name.lower() in ['dfm', 'ddfm'] and y_recent_eval is not None:
                        try:
                            # Get the actual DFM/DDFM model (forecaster might be a wrapper)
                            dfm_model = forecaster
                            if hasattr(forecaster, '_dfm_model'):
                                dfm_model = forecaster._dfm_model
                            elif hasattr(forecaster, '_ddfm_model'):
                                dfm_model = forecaster._ddfm_model
                            
                            # Check if model has update() method (dfm-python models have this)
                            if hasattr(dfm_model, 'update'):
                                # Standardize recent data using Mx/Wx from model result
                                if hasattr(dfm_model, 'result') and hasattr(dfm_model.result, 'Mx') and hasattr(dfm_model.result, 'Wx'):
                                    import numpy as np
                                    Mx = np.asarray(dfm_model.result.Mx)
                                    Wx = np.asarray(dfm_model.result.Wx)
                                    y_recent_array = y_recent_eval.values if isinstance(y_recent_eval, pd.DataFrame) else y_recent_eval
                                    
                                    # Handle shape mismatch
                                    if y_recent_array.ndim == 2:
                                        Mx_use = Mx[:y_recent_array.shape[1]] if len(Mx) >= y_recent_array.shape[1] else Mx
                                        Wx_use = Wx[:y_recent_array.shape[1]] if len(Wx) >= y_recent_array.shape[1] else Wx
                                        Wx_use = np.where(Wx_use == 0, 1.0, Wx_use)
                                        y_recent_std = (y_recent_array - Mx_use) / Wx_use
                                    else:
                                        y_recent_std = y_recent_array
                                    
                                    # Update model state with recent data (history=None to use full period)
                                    dfm_model.update(y_recent_std, history=None)
                                    logger.info(f"Updated {model_name.upper()} state with {len(y_recent_array)} periods (2020-2023) using update() method")
                                else:
                                    logger.warning(f"{model_name.upper()} model missing Mx/Wx, cannot standardize recent data for update()")
                            else:
                                logger.warning(f"{model_name.upper()} model does not have update() method")
                        except Exception as e:
                            logger.warning(f"Failed to update {model_name.upper()} state with recent data: {e}")
                    
                    from src.evaluate import evaluate_forecaster
                    try:
                        # evaluate_forecaster will call fit() internally, but since forecaster is already fitted,
                        # it should use the existing fitted state. However, to be safe, we check if it's fitted first.
                        # For DFM/DDFM, state is already updated above, so y_recent is not needed here
                        forecast_metrics_raw = evaluate_forecaster(
                            forecaster, y_train_eval, y_test_eval, horizons, 
                            target_series=target_series, y_recent=None  # Already updated above
                        )
                        forecast_metrics = {str(k): v for k, v in forecast_metrics_raw.items()}
                        result['metrics'] = {'forecast_metrics': forecast_metrics}
                        logger.info(f"Successfully evaluated {model_name.upper()} on {len([k for k in forecast_metrics.keys() if forecast_metrics[k].get('n_valid', 0) > 0])} horizons")
                    except Exception as e:
                        logger.error(f"Failed to evaluate {model_name.upper()}: {type(e).__name__}: {str(e)}")
                        result['status'] = 'failed'
                        result['error'] = f"Evaluation failed: {str(e)}"
                        failed_models.append(model_name)
                        model_results[model_name] = result
                        continue
            else:
                # Checkpoint not found - skip this model
                checkpoint_msg = f"{checkpoint_dir_local}/{target_series}_{model_name}/model.pkl" if checkpoint_dir_local else "checkpoint"
                logger.warning(f"Skipping: Checkpoint not found ({checkpoint_msg})")
                logger.warning(f"  Run 'bash run_train.sh' to train this model first")
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
                    logger.info("Completed")
                elif result.get('status') == 'skipped':
                    # Already logged skip message above
                    pass
                else:
                    logger.warning(f"Status: {result.get('status', 'unknown')}")
        except Exception as e:
            import traceback
            logger.error(f"Failed: {str(e)}")
            logger.debug("Full traceback:", exc_info=True)
            failed_models.append(model_name)
            model_results[model_name] = {'status': 'failed', 'error': str(e), 'metrics': None}
    
    comparison = None
    successful_count = len(model_results) - len(failed_models)
    if successful_count > 0:
        logger.info(f"Comparing {successful_count} successful models...")
        comparison = _compare_results(model_results, horizons, target_series)
        
        if comparison and comparison.get('metrics_table') is not None:
            # Save comparison table if available
            table_path = output_dir / "comparison_table.csv"
            metrics_table = comparison.get('metrics_table')
            if metrics_table is not None:
                try:
                    metrics_table.to_csv(table_path, index=False)
                    logger.info(f"Table: {table_path}")
                except Exception as e:
                    logger.warning(f"Failed to save comparison table: {e}")
    
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
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False, default=str)
        
        # Validate file was created and is non-empty
        if not results_file.exists():
            raise IOError(f"Results file was not created: {results_file}")
        
        file_size = results_file.stat().st_size
        if file_size == 0:
            raise IOError(f"Results file is empty: {results_file}")
        
        logger.info(f"Results saved successfully to: {results_file} (size: {file_size} bytes)")
    except (IOError, OSError) as e:
        logger.error(f"Failed to save results to {results_file}: {type(e).__name__}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error saving results to {results_file}: {type(e).__name__}: {str(e)}")
        raise
    
    # Count skipped models separately
    skipped_models = [name for name, result in model_results.items() if result.get('status') == 'skipped']
    actual_failed = [name for name in failed_models if name not in skipped_models]
    
    if skipped_models:
        logger.warning(f"Skipped (no checkpoint): {', '.join(skipped_models)}")
    if actual_failed:
        logger.error(f"Failed: {', '.join(actual_failed)}")
    logger.info("=" * 70)
    
    return comparison_data


def _compare_results(
    results: Dict[str, Dict[str, Any]],
    horizons: List[int],
    target_series: str
) -> Dict[str, Any]:
    """Compare results from multiple models."""
    from src.evaluate import compare_multiple_models
    
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
    cfg: DictConfig,
    model_name: Optional[str] = None,
    checkpoint_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Train a single model using Hydra config (injected by @hydra.main).
    
    Parameters
    ----------
    cfg : DictConfig
        Hydra config injected by @hydra.main decorator (already composed with defaults)
    model_name : Optional[str]
        Model name to train (e.g., arima, var, dfm, ddfm). If not specified, uses first model from config.
    checkpoint_dir : Optional[str]
        Directory to save checkpoint (e.g., checkpoint)
    """
    # Extract params from cfg (Hydra already composed config)
    source_cfg = _get_experiment_cfg(cfg)
    models = source_cfg.get('models', [])
    target_series = source_cfg.get('target_series')
    
    if model_name is None:
        model_name = models[0] if models else None
        if len(models) > 1:
            logger.warning(f"Config specifies {len(models)} models, training only first: {model_name}")
    
    if not model_name:
        raise ValueError(f"Config must specify at least one model in 'models' list")
    
    if checkpoint_dir and target_series:
        checkpoint_model_name = f"{target_series}_{model_name}"
    else:
        checkpoint_model_name = model_name
    
    # Use train() function with cfg directly (no need for config_name/config_path)
    result = train(
        cfg=cfg,
        model_name=checkpoint_model_name,
        checkpoint_dir=checkpoint_dir
    )
    
    if checkpoint_dir and 'model_dir' in result:
        result['model_dir'] = str(Path(checkpoint_dir) / checkpoint_model_name)
    
    return result


def compare_models_by_config(
    cfg: DictConfig,
    models_filter: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Compare multiple models from Hydra config (injected by @hydra.main).
    
    Parameters
    ----------
    cfg : DictConfig
        Hydra config injected by @hydra.main decorator (already composed with defaults)
    models_filter : Optional[List[str]]
        Filter models to run (e.g., ['arima', 'var']). If not specified, runs all models from config.
    """
    # Extract params from cfg (Hydra already composed config)
    source_cfg = _get_experiment_cfg(cfg)
    config_name = getattr(cfg, '_name_', None) or cfg.get('_name_', 'experiment/consumption_kowrccnse_report')
    config_dir = str(project_root / "config")
    
    models_to_run = source_cfg.get('models', [])
    if models_filter:
        models_to_run = [m for m in models_to_run if m.lower() in [mf.lower() for mf in models_filter]]
        if not models_to_run:
            raise ValueError(f"No models match filter {models_filter}. Available models: {source_cfg.get('models', [])}")
    
    # Get checkpoint_dir from config
    checkpoint_dir = source_cfg.get('checkpoint_dir', 'checkpoints')
    
    result = compare_models(
        target_series=source_cfg.get('target_series'),
        models=models_to_run,
        horizons=source_cfg.get('forecast_horizons', []),
        data_path=source_cfg.get('data_path'),
        config_dir=config_dir,
        config_name=config_name,
        config_overrides=None,  # Already handled by Hydra
        checkpoint_dir=checkpoint_dir
    )
    
    return result


# ============================================================================
# Nowcasting (DFM/DDFM) entry point
# ============================================================================

def _load_checkpoint_forecaster(checkpoint_path: Path):
    """Load forecaster from checkpoint pickle."""
    import pickle
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
    # Check multiple known formats:
    # - {'forecaster': ..., 'config': ...} (expected by nowcast)
    # - {'model': ..., 'metadata': ...} (saved by save_model_checkpoint)
    # - direct model object (rare)
    if isinstance(data, dict):
        forecaster = data.get('forecaster') or data.get('model')
        cfg_saved = data.get('config') or data.get('metadata')
    else:
        forecaster = data
        cfg_saved = None
    return forecaster, cfg_saved


def _get_model_feature_cols_from_forecaster(forecaster, cfg_saved) -> Optional[List[str]]:
    """Derive feature columns used to train the DFM/DDFM forecaster."""
    if hasattr(forecaster, "_y") and isinstance(getattr(forecaster, "_y"), pd.DataFrame):
        return list(forecaster._y.columns)
    try:
        if cfg_saved and isinstance(cfg_saved, dict):
            exp_cfg = cfg_saved.get('experiment', cfg_saved)
            series = exp_cfg.get('series') or []
            if isinstance(series, list):
                return [str(s) for s in series]
    except Exception:
        pass
    return None


def nowcast(
    config_name: str,
    config_dir: Optional[str],
    checkpoint_dir: str,
    model_name: str,
    nowcast_start: str,
    nowcast_end: str,
    output_dir: str = "outputs/backtest",
    weeks_before: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Generate nowcasts using checkpointed DFM/DDFM models.
    
    Uses update() -> predict(horizon=1) pattern per month, reloading model
    for each month to avoid state accumulation.
    """
    config_dir = config_dir or str(project_root / "config")
    # Use Hydra to load config
    import hydra
    with hydra.initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = hydra.compose(config_name=config_name)
        if OmegaConf.is_struct(cfg):
            OmegaConf.set_struct(cfg, False)
        source_cfg = _get_experiment_cfg(cfg)
        target_series = source_cfg.get('target_series')
        data_path = source_cfg.get('data_path') or str(project_root / "data" / "data.csv")
    if not target_series:
        raise ValueError("target_series required for nowcast")
    # Default timepoints: 4 weeks and 1 week before
    if weeks_before is None:
        weeks_before = [4, 1]
    # Load and resample data
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    monthly = data
    # Drop non-numeric columns (e.g., string weekly dates) before resampling
    monthly = monthly.select_dtypes(include=[np.number])
    if isinstance(monthly.index, pd.DatetimeIndex):
        monthly = monthly.sort_index()
    monthly = monthly[(monthly.index >= pd.Timestamp('1985-01-01'))]
    monthly = monthly.resample('MS').mean()
    nowcast_start_ts = pd.to_datetime(nowcast_start)
    nowcast_end_ts = pd.to_datetime(nowcast_end)
    months = pd.date_range(nowcast_start_ts, nowcast_end_ts, freq='MS')
    checkpoint_path = Path(checkpoint_dir) / f"{target_series}_{model_name}" / "model.pkl"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Structure results by timepoint
    results_by_timepoint = {}
    for weeks in weeks_before:
        tp_key = f"{weeks}weeks"
        results_by_timepoint[tp_key] = {"monthly_results": []}
    
    # Also maintain flat results for backward compatibility
    flat_results = []
    
    for ts in months:
        month_str = ts.strftime("%Y-%m")
        # Get actual value for this month
        actual_val = None
        if target_series in monthly.columns and ts in monthly.index:
            actual_val = float(monthly.loc[ts, target_series]) if not pd.isna(monthly.loc[ts, target_series]) else None
        elif isinstance(monthly, pd.DataFrame) and ts in monthly.index:
            # Try first column if target_series not found
            actual_val = float(monthly.loc[ts].iloc[0]) if len(monthly.loc[ts]) > 0 and not pd.isna(monthly.loc[ts].iloc[0]) else None
        
        # Process each timepoint
        for weeks in weeks_before:
            tp_key = f"{weeks}weeks"
            # Reload forecaster fresh each month to avoid state carryover
            forecaster, cfg_saved = _load_checkpoint_forecaster(checkpoint_path)
            if forecaster is None:
                monthly_result = {
                    "month": month_str,
                    "status": "failed",
                    "error": "forecaster not found in checkpoint"
                }
                results_by_timepoint[tp_key]["monthly_results"].append(monthly_result)
                flat_results.append({"date": ts.strftime("%Y-%m-%d"), "status": "failed", "error": "forecaster not found in checkpoint"})
                continue
            
            model_feature_cols = _get_model_feature_cols_from_forecaster(forecaster, cfg_saved)
            # Prepare available data: subtract weeks from target month
            available_end = ts - pd.Timedelta(weeks=weeks)
            y_available = monthly.loc[:available_end]
            if isinstance(y_available, pd.DataFrame) and model_feature_cols:
                cols = [c for c in model_feature_cols if c in y_available.columns]
                y_available = y_available[cols]
            # Drop all-NaN rows to avoid empty updates
            if isinstance(y_available, pd.DataFrame):
                y_available = y_available.dropna(how='all')
            if len(y_available) == 0:
                monthly_result = {
                    "month": month_str,
                    "status": "failed",
                    "error": f"no available data before target month (cutoff: {available_end.strftime('%Y-%m-%d')})"
                }
                results_by_timepoint[tp_key]["monthly_results"].append(monthly_result)
                flat_results.append({"date": ts.strftime("%Y-%m-%d"), "status": "failed", "error": "no available data before target month"})
                continue
            
            # Update state then predict horizon=1
            try:
                if hasattr(forecaster, "_dfm_model") or hasattr(forecaster, "_ddfm_model"):
                    import torch  # Ensure torch is in scope even if module import was skipped
                    dfm_model = getattr(forecaster, "_dfm_model", None) or getattr(forecaster, "_ddfm_model", None)
                    # Move model to CUDA if available; otherwise CPU
                    target_device = "cuda" if torch.cuda.is_available() else "cpu"
                    try:
                        if hasattr(dfm_model, "to"):
                            dfm_model.to(target_device)
                    except Exception:
                        target_device = "cpu"
                        try:
                            if hasattr(dfm_model, "to"):
                                dfm_model.to(target_device)
                        except Exception:
                            pass
                    result = getattr(dfm_model, "result", None)
                    if result is None or not hasattr(result, "Mx") or not hasattr(result, "Wx"):
                        raise RuntimeError("Model result missing Mx/Wx")
                    Mx = np.asarray(result.Mx)
                    Wx = np.asarray(result.Wx)
                    y_vals = y_available.values
                    Mx_use = Mx[: y_vals.shape[1]]
                    Wx_use = np.where(Wx[: y_vals.shape[1]] == 0, 1.0, Wx[: y_vals.shape[1]])
                    X_std = (y_vals - Mx_use) / Wx_use
                    X_std = np.where(np.isfinite(X_std), X_std, np.nan)
                    # Move to target device as torch tensor
                    X_std_tensor = torch.tensor(X_std, dtype=torch.float32, device=target_device)
                    dfm_model.update(X_std_tensor, history=None)
                    raw_pred = dfm_model.predict(horizon=1)
                    from src.models.models_utils import _convert_predictions_to_dataframe
                    pred_df = _convert_predictions_to_dataframe(raw_pred, y_available, 1)
                    pred_val = None
                    if isinstance(pred_df, pd.DataFrame):
                        if target_series in pred_df.columns:
                            pred_val = float(pred_df[target_series].iloc[0])
                        elif pred_df.shape[1] > 0:
                            pred_val = float(pred_df.iloc[0, 0])
                    elif isinstance(pred_df, pd.Series) and len(pred_df) > 0:
                        pred_val = float(pred_df.iloc[0])
                    
                    # Calculate error if actual value is available
                    error = None
                    if actual_val is not None and pred_val is not None:
                        error = float(pred_val - actual_val)
                    
                    # Create monthly result with expected structure
                    monthly_result = {
                        "month": month_str,
                        "forecast_value": pred_val,
                        "actual_value": actual_val,
                        "error": error,
                        "status": "ok"
                    }
                    results_by_timepoint[tp_key]["monthly_results"].append(monthly_result)
                    flat_results.append({
                        "date": ts.strftime("%Y-%m-%d"),
                        "prediction": pred_val,
                        "actual": actual_val,
                        "error": error,
                        "n_features": y_vals.shape[1],
                        "status": "ok"
                    })
                else:
                    monthly_result = {
                        "month": month_str,
                        "status": "failed",
                        "error": "not a DFM/DDFM forecaster"
                    }
                    results_by_timepoint[tp_key]["monthly_results"].append(monthly_result)
                    flat_results.append({"date": ts.strftime("%Y-%m-%d"), "status": "failed", "error": "not a DFM/DDFM forecaster"})
            except Exception as e:
                monthly_result = {
                    "month": month_str,
                    "status": "failed",
                    "error": str(e)
                }
                results_by_timepoint[tp_key]["monthly_results"].append(monthly_result)
                flat_results.append({"date": ts.strftime("%Y-%m-%d"), "status": "failed", "error": str(e)})
    
    # Persist with both structures for backward compatibility
    out_file = output_dir_path / f"{target_series}_{model_name}_backtest.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "target_series": target_series,
            "model": model_name,
            "config": config_name,
            "nowcast_start": nowcast_start,
            "nowcast_end": nowcast_end,
            "weeks_before": weeks_before,
            "results": flat_results,  # Keep for backward compatibility
            "results_by_timepoint": results_by_timepoint  # New structure expected by table/plot code
        }, f, indent=2)
    return {"output": str(out_file), "results": flat_results, "results_by_timepoint": results_by_timepoint}

# ============================================================================
# CLI Entry Point
# ============================================================================

@hydra.main(config_path=str(project_root / "config"), config_name="experiment/consumption_kowrccnse_report", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """CLI entry point for training using Hydra decorator.
    
    Uses @hydra.main to automatically load experiment config with defaults.
    Config is automatically injected by Hydra.
    
    Usage:
        python train.py experiment/consumption_kowrccnse_report model=dfm checkpoint_dir=checkpoints
        python train.py experiment/investment_koequipte_report model=ddfm checkpoint_dir=checkpoints
        python train.py experiment/consumption_kowrccnse_report command=compare
        python train.py experiment/consumption_kowrccnse_report command=nowcast model=dfm nowcast_start=2024-01-01 nowcast_end=2024-12-01
    """
    # Setup logging to file (log directory)
    # Use experiment.log_dir if provided, otherwise default to project_root / "log"
    exp_cfg = cfg.experiment if 'experiment' in cfg else cfg
    log_dir = exp_cfg.get('log_dir', None)
    if log_dir:
        log_dir = project_root / log_dir if isinstance(log_dir, str) else Path(log_dir)
    else:
        log_dir = project_root / "log"
    setup_logging(log_dir=log_dir, force=True)
    
    # Extract command from cfg (Hydra automatically handles config_name override)
    # Hydra loads experiment configs under 'experiment' key
    exp_cfg = cfg.experiment if 'experiment' in cfg else cfg
    command = exp_cfg.get('command', 'train')
    model_name = exp_cfg.get('model')
    checkpoint_dir = exp_cfg.get('checkpoint_dir')
    models_filter = exp_cfg.get('models')
    nowcast_start = exp_cfg.get('nowcast_start')
    nowcast_end = exp_cfg.get('nowcast_end')
    output_dir = exp_cfg.get('output_dir', 'outputs/backtest')
    weeks_before = exp_cfg.get('weeks_before')
    
    # Get config_name from Hydra metadata (automatically set by @hydra.main)
    config_name = getattr(cfg, '_name_', None) or cfg.get('_name_', 'experiment/consumption_kowrccnse_report')
    config_path = str(project_root / "config")
    
    if command == 'train':
        if not checkpoint_dir:
            raise ValueError("checkpoint_dir must be specified in config or via override.")
        result = train_model(
            cfg=cfg,  # Pass cfg directly (Hydra already composed it)
            model_name=model_name,
            checkpoint_dir=checkpoint_dir
        )
        print(f"\n✓ Model saved to: {result['model_dir']}")
        
    elif command == 'compare':
        result = compare_models_by_config(
            cfg=cfg,  # Pass cfg directly (Hydra already composed it)
            models_filter=models_filter
        )
        print(f"\n✓ Comparison saved to: {result['output_dir']}")
        if result.get('failed_models'):
            print(f"  Failed: {', '.join(result['failed_models'])}")
    elif command == 'nowcast':
        if not model_name:
            raise ValueError("model is required for nowcast command.")
        result = nowcast(
            config_name=config_name,
            config_dir=config_path,
            checkpoint_dir=checkpoint_dir or 'checkpoint',
            model_name=model_name,
            nowcast_start=nowcast_start,
            nowcast_end=nowcast_end,
            output_dir=output_dir,
            weeks_before=weeks_before
        )
        print(f"\n✓ Nowcast saved to: {result['output']}")
        failed = [r for r in result['results'] if r.get('status') != 'ok']
        if failed:
            print(f"  Failed entries: {len(failed)}")


if __name__ == "__main__":
    main()
