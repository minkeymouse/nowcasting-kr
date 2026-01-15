"""Training function for TFT model using NeuralForecast."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.models import TFT

from src.train._common import (
    prepare_training_data,
    train_neuralforecast_model,
    get_common_training_params,
    get_processed_data_from_loader
)

logger = logging.getLogger(__name__)


def train_tft_model(
    model_type: str,
    cfg: Any,
    data: pd.DataFrame,
    model_name: str,
    outputs_dir: Path = None,
    model_params: Optional[Dict[str, Any]] = None,
    data_loader: Optional[Any] = None
) -> None:
    """Train TFT model using NeuralForecast.
    
    Parameters
    ----------
    model_type : str
        Model type: 'tft'
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
    logger.info(f"Training TFT model...")
    
    # Use processed (unstandardized) data instead of standardized
    # Let the model handle scaling internally
    data = get_processed_data_from_loader(data, data_loader, "TFT")
    logger.info(f"Training data shape: {data.shape}")
    
    # Prepare training data (includes covariates)
    target_data, covariate_data, available_targets, covariate_names = prepare_training_data(
        data, model_params, data_loader, use_covariates=True
    )
    
    # Create model with hist_exog_list if covariates are available
    # Pass data size for proper max_steps calculation
    n_samples = len(target_data)
    model = _create_tft_model(model_params, len(available_targets), covariate_names, n_samples)
    
    # Train model with covariates
    train_neuralforecast_model(
        model, target_data, available_targets, outputs_dir,
        covariate_data=covariate_data,
        covariate_names=covariate_names
    )


def _create_tft_model(
    model_params: Dict[str, Any], 
    n_series: int,
    hist_exog_list: Optional[list[str]] = None,
    n_samples: Optional[int] = None
) -> NeuralForecast:
    """Create TFT model instance.
    
    Parameters
    ----------
    model_params : dict
        Model parameters dictionary
    n_series : int
        Number of target series
    hist_exog_list : list[str], optional
        List of historical exogenous variable names (covariates)
    """
    # Reduce batch size if using many covariates to avoid OOM (before calculating max_steps)
    batch_size = model_params.get('batch_size', 32)
    if hist_exog_list and len(hist_exog_list) > 30:
        batch_size = min(batch_size, 64)
        model_params = model_params.copy()  # Don't modify original
        model_params['batch_size'] = batch_size
        logger.info(f"Reduced batch_size to {batch_size} due to large number of covariates ({len(hist_exog_list)})")
    
    # Now calculate max_steps with correct batch_size
    common_params = get_common_training_params(model_params, n_samples=n_samples)
    
    model_instance = TFT(
        **common_params,
        tgt_size=model_params.get('tgt_size', 1),
        hidden_size=model_params.get('hidden_size', model_params.get('d_model', 128)),
        n_head=model_params.get('n_heads', model_params.get('n_head', 4)),
        attn_dropout=model_params.get('attn_dropout', model_params.get('dropout', 0.0)),
        grn_activation=model_params.get('grn_activation', 'ELU'),
        n_rnn_layers=model_params.get('n_rnn_layers', 1),
        rnn_type=model_params.get('rnn_type', 'lstm'),
        one_rnn_initial_state=model_params.get('one_rnn_initial_state', False),
        dropout=model_params.get('dropout', 0.1),
        hist_exog_list=hist_exog_list,  # Pass covariates
    )
    
    # Use default local_scaler_type='standard' to let NeuralForecast handle scaling
    return NeuralForecast(models=[model_instance], freq='W')
