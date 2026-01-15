"""Training function for TimeMixer model using NeuralForecast."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.models import TimeMixer

from src.train._common import (
    prepare_training_data,
    train_neuralforecast_model,
    get_common_training_params,
    get_processed_data_from_loader
)
from src.utils import get_monthly_series_from_metadata

logger = logging.getLogger(__name__)


def train_timemixer_model(
    model_type: str,
    cfg: Any,
    data: pd.DataFrame,
    model_name: str,
    outputs_dir: Path = None,
    model_params: Optional[Dict[str, Any]] = None,
    data_loader: Optional[Any] = None
) -> None:
    """Train TimeMixer model using NeuralForecast.
    
    Parameters
    ----------
    model_type : str
        Model type: 'timemixer'
    cfg : Any
        Hydra config object
    data : pd.DataFrame
        Preprocessed training data (standardized, without date columns)
        NOTE: For attention-based models, we now use processed (unstandardized) data
        and let the model handle scaling internally.
    model_name : str
        Model name for saving
    outputs_dir : Path
        Directory to save trained model
    model_params : dict, optional
        Model parameters dictionary
    data_loader : optional
        Data loader object for datetime index preservation
    """
    logger.info(f"Training TimeMixer model...")
    
    # Use processed (unstandardized) data instead of standardized
    # TimeMixer with use_norm=True (RevIN) handles normalization internally
    # Similar to PatchTST and iTransformer
    data = get_processed_data_from_loader(data, data_loader, "TimeMixer")
    logger.info(f"Training data shape: {data.shape}")
    
    # TimeMixer uses multivariate forecasting: all variables as targets
    # Prepare training data with all variables as targets
    target_data, covariate_data, available_targets, covariate_names = prepare_training_data(
        data, model_params, data_loader, use_covariates=True
    )
    
    # Combine all variables (target + covariates) as multivariate targets
    all_targets = available_targets + covariate_names
    all_target_data = pd.concat([target_data, covariate_data], axis=1)
    
    logger.info(f"Using multivariate forecasting: {len(all_targets)} variables as targets")
    logger.info(f"  Primary target: {available_targets[0] if available_targets else 'N/A'}")
    logger.info(f"  Additional targets: {len(covariate_names)} variables")
    
    # Create model with n_series = total number of variables
    # Pass data size for proper max_steps calculation
    n_samples = len(all_target_data)
    model = _create_timemixer_model(model_params, len(all_targets), data_loader, n_samples)
    
    # Train model with all variables as targets
    train_neuralforecast_model(
        model, all_target_data, all_targets, outputs_dir,
        covariate_data=None,  # No covariates - all are targets
        covariate_names=None
    )


def _create_timemixer_model(
    model_params: Dict[str, Any],
    n_series: int,
    data_loader: Optional[Any] = None,
    n_samples: Optional[int] = None
) -> NeuralForecast:
    """Create TimeMixer model instance.
    
    Configures multi-scale downsampling for mixed-frequency data:
    - For weekly + monthly data: Creates weekly -> monthly -> quarterly -> yearly hierarchy
      (3 downsampling layers with window=4, creating 4 scales total)
    - For uniform data: Uses default downsampling (1 layer with window=2, creating 2 scales)
    """
    common_params = get_common_training_params(model_params, n_samples=n_samples)
    
    # Get multiscale parameters for mixed-frequency data alignment
    # For weekly + monthly data, create frequency hierarchy: weekly -> monthly -> quarterly -> yearly
    monthly_series = get_monthly_series_from_metadata(data_loader) if data_loader else set()
    
    # Auto-detect downsampling parameters based on data characteristics
    # Config can override via 'down_sampling_layers' and 'down_sampling_window'
    if monthly_series and len(monthly_series) > 0:
        # Weekly -> Monthly hierarchy
        # Using window=4 for weekly->monthly downsampling (as per report: downsampling_window=4)
        # Using 1 layer to avoid dimension mismatch in upsampling layers
        #   Layer 0 (original): weekly (1 week)
        #   Layer 1: monthly (~4 weeks)
        # Note: More layers (quarterly, yearly) cause dimension mismatch in upsampling
        #       due to sequence length constraints with context_length=96
        default_window = model_params.get('down_sampling_window', 4)
        default_layers = model_params.get('down_sampling_layers', 1)  # 1 layer = 2 scales (weekly + monthly)
        
        logger.info(f"Detected {len(monthly_series)} monthly series. "
                   f"Configuring multi-scale downsampling: "
                   f"down_sampling_layers={default_layers}, down_sampling_window={default_window} "
                   f"(weekly -> monthly)")
    else:
        # Default for uniform data: simpler downsampling
        default_window = model_params.get('down_sampling_window', 2)
        default_layers = model_params.get('down_sampling_layers', 1)
    
    # TimeMixer's use_norm enables RevIN for internal normalization
    # When use_norm=True, raw (unstandardized) data is provided to the model
    # RevIN normalizes internally and denormalizes predictions back to raw scale
    use_norm = model_params.get('use_norm', True)  # Default True
    
    # Read model architecture parameters from config (with fallbacks)
    # Config files use 'n_layers', but also support 'e_layers' and 'num_layers' for compatibility
    e_layers = (model_params.get('e_layers') or 
                model_params.get('num_layers') or 
                model_params.get('n_layers', 4))
    
    # d_model and d_ff: allow config override, otherwise use defaults
    d_model = model_params.get('d_model', 32)
    d_ff = model_params.get('d_ff', d_model * 4)
    
    # moving_avg: use config value or default (TimeMixer handles edge cases internally)
    moving_avg = model_params.get('moving_avg', 25)
    
    model_instance = TimeMixer(
        **common_params,
        n_series=n_series,
        d_model=d_model,
        d_ff=d_ff,
        e_layers=e_layers,
        dropout=model_params.get('dropout', 0.1),
        # Multiscale parameters for mixed-frequency data
        # Creates hierarchical scales: weekly -> monthly -> quarterly -> yearly (when monthly series detected)
        # Can be overridden in config files
        down_sampling_layers=default_layers,
        down_sampling_window=default_window,
        down_sampling_method=model_params.get('down_sampling_method', 'avg'),
        decomp_method=model_params.get('decomp_method', 'moving_avg'),
        moving_avg=moving_avg,
        top_k=model_params.get('top_k', 5),
        channel_independence=model_params.get('channel_independence', 0),
        use_norm=use_norm,
        # No hist_exog_list - TimeMixer doesn't support covariates, uses multivariate targets instead
    )
    
    # Use default local_scaler_type='standard' to let NeuralForecast handle scaling
    # Similar to PatchTST (RevIN) and iTransformer (use_norm) - NeuralForecast scaling works with model's internal norm
    return NeuralForecast(models=[model_instance], freq='W')
