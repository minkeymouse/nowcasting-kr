"""Training function for iTransformer model using NeuralForecast."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.models import iTransformer

from src.train._common import (
    prepare_training_data,
    train_neuralforecast_model,
    get_common_training_params,
    get_processed_data_from_loader
)

logger = logging.getLogger(__name__)


def train_itf_model(
    model_type: str,
    cfg: Any,
    data: pd.DataFrame,
    model_name: str,
    outputs_dir: Path = None,
    model_params: Optional[Dict[str, Any]] = None,
    data_loader: Optional[Any] = None
) -> None:
    """Train iTransformer model using NeuralForecast.
    
    Parameters
    ----------
    model_type : str
        Model type: 'itf' or 'itransformer'
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
    logger.info(f"Training iTransformer model...")
    
    # Use processed (unstandardized) data instead of standardized
    # Let the model handle scaling internally
    data = get_processed_data_from_loader(data, data_loader, "iTransformer")
    logger.info(f"Training data shape: {data.shape}")
    
    # iTransformer uses multivariate forecasting: all variables as targets
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
    model = _create_itf_model(model_params, len(all_targets), n_samples)
    
    # Train model with all variables as targets
    train_neuralforecast_model(
        model, all_target_data, all_targets, outputs_dir,
        covariate_data=None,  # No covariates - all are targets
        covariate_names=None
    )


def _create_itf_model(
    model_params: Dict[str, Any], 
    n_series: int,
    n_samples: Optional[int] = None
) -> NeuralForecast:
    """Create iTransformer model instance.
    
    iTransformer uses multivariate forecasting where all variables are treated as targets.
    The n_series parameter represents the total number of variables (multivariate).
    iTransformer applies attention across variates (variables) to model interrelationships.
    
    Parameters
    ----------
    model_params : dict
        Model parameters dictionary
    n_series : int
        Total number of variables (all used as targets for multivariate forecasting)
    """
    common_params = get_common_training_params(model_params, n_samples=n_samples)
    
    # Data is now unstandardized, so use_norm should be enabled (like RevIN for PatchTST)
    use_norm = model_params.get('use_norm', True)  # Default to True when data is not pre-standardized
    
    model_instance = iTransformer(
        **common_params,
        n_series=n_series,  # Total number of variables (multivariate)
        n_heads=model_params.get('n_heads', 8),
        e_layers=model_params.get('e_layers', model_params.get('num_layers', 2)),
        d_layers=model_params.get('d_layers', 1),
        hidden_size=model_params.get('d_model', 512),
        d_ff=model_params.get('d_ff', 2048),
        dropout=model_params.get('dropout', 0.1),
        use_norm=use_norm,  # Enable normalization for unstandardized data
        # No hist_exog_list - iTransformer doesn't support covariates, uses multivariate targets instead
    )
    
    # Use default local_scaler_type='standard' to let NeuralForecast handle scaling
    # Similar to PatchTST (RevIN) and TFT - NeuralForecast scaling works with model's internal norm
    return NeuralForecast(models=[model_instance], freq='W')
