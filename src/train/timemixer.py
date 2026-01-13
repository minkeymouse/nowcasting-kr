"""Training function for TimeMixer model using NeuralForecast."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.models import TimeMixer

from src.train._common import (
    prepare_training_data,
    train_neuralforecast_model,
    get_common_training_params
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
    logger.info(f"Training data shape: {data.shape}")
    
    # TimeMixer with use_norm=True (RevIN) should receive raw (unstandardized) data
    # Check if we need to use raw data
    use_norm = model_params.get('use_norm', True) if model_params else True
    
    if use_norm and data_loader is not None:
        # Use raw processed data (not standardized) for TimeMixer with RevIN
        if hasattr(data_loader, 'processed') and data_loader.processed is not None:
            raw_data = data_loader.processed.copy()
            # Remove date columns if present (same as standardized data)
            date_cols = ['date', 'date_w', 'year', 'month', 'day']
            data_cols = [col for col in raw_data.columns if col not in date_cols]
            raw_data = raw_data[data_cols].copy()
            
            # Ensure same columns as standardized data (in case of filtering)
            if hasattr(data_loader, 'standardized') and data_loader.standardized is not None:
                std_cols = [col for col in data_loader.standardized.columns if col not in date_cols]
                # Filter to common columns
                common_cols = [col for col in data_cols if col in std_cols]
                if common_cols:
                    raw_data = raw_data[common_cols]
            
            logger.info(f"Using raw (unstandardized) data for TimeMixer with RevIN. "
                       f"Raw data shape: {raw_data.shape}")
            logger.info(f"Raw data stats - Mean range: [{raw_data.mean().min():.2f}, {raw_data.mean().max():.2f}], "
                       f"Std range: [{raw_data.std().min():.2f}, {raw_data.std().max():.2f}]")
            
            # Use raw data instead of standardized data
            data = raw_data
    
    # Prepare training data
    target_data, available_targets = prepare_training_data(data, model_params, data_loader)
    
    # Create model
    model = _create_timemixer_model(model_params, len(available_targets), data_loader)
    
    # Train model
    train_neuralforecast_model(model, target_data, available_targets, outputs_dir)


def _create_timemixer_model(
    model_params: Dict[str, Any],
    n_series: int,
    data_loader: Optional[Any] = None
) -> NeuralForecast:
    """Create TimeMixer model instance.
    
    Configures multi-scale downsampling for mixed-frequency data:
    - For weekly + monthly data: Creates weekly -> monthly -> quarterly -> yearly hierarchy
      (3 downsampling layers with window=4, creating 4 scales total)
    - For uniform data: Uses default downsampling (1 layer with window=2, creating 2 scales)
    """
    common_params = get_common_training_params(model_params)
    
    # Get multiscale parameters for mixed-frequency data alignment
    # For weekly + monthly data, create frequency hierarchy: weekly -> monthly -> quarterly -> yearly
    monthly_series = get_monthly_series_from_metadata(data_loader) if data_loader else set()
    
    if monthly_series and len(monthly_series) > 0:
        # Weekly -> Monthly -> Quarterly -> Yearly hierarchy
        # Using window=4 for sequential downsampling:
        #   Layer 0 (original): weekly (1 week)
        #   Layer 1: monthly (~4 weeks)
        #   Layer 2: quarterly (~16 weeks ≈ 4 months, close to 3-month quarter)
        #   Layer 3: yearly (~64 weeks ≈ 1.25 years, close to annual)
        default_window = model_params.get('down_sampling_window', 4)
        default_layers = model_params.get('down_sampling_layers', 3)  # 3 layers = 4 scales total
        
        logger.info(f"Detected {len(monthly_series)} monthly series. "
                   f"Configuring multi-scale downsampling: "
                   f"down_sampling_layers={default_layers}, down_sampling_window={default_window} "
                   f"(weekly -> monthly -> quarterly -> yearly)")
    else:
        # Default for uniform data: simpler downsampling
        default_window = model_params.get('down_sampling_window', 2)
        default_layers = model_params.get('down_sampling_layers', 1)
    
    # TimeMixer's use_norm enables RevIN for internal normalization
    # When use_norm=True, raw (unstandardized) data is provided to the model
    # RevIN will normalize internally and denormalize predictions back to raw scale
    use_norm = model_params.get('use_norm', True)  # Default True
    
    model_instance = TimeMixer(
        **common_params,
        n_series=n_series,
        d_model=model_params.get('d_model', 32),
        d_ff=model_params.get('d_ff', 32),
        e_layers=model_params.get('e_layers', model_params.get('num_layers', 4)),
        dropout=model_params.get('dropout', 0.1),
        # Multiscale parameters for mixed-frequency data
        # Creates hierarchical scales: weekly -> monthly -> quarterly -> yearly (when monthly series detected)
        down_sampling_layers=default_layers,
        down_sampling_window=default_window,
        down_sampling_method=model_params.get('down_sampling_method', 'avg'),
        decomp_method=model_params.get('decomp_method', 'moving_avg'),
        moving_avg=model_params.get('moving_avg', 25),
        top_k=model_params.get('top_k', 5),
        channel_independence=model_params.get('channel_independence', 0),
        use_norm=use_norm,
    )
    
    return NeuralForecast(models=[model_instance], freq='W')
