"""Training function for PatchTST model using NeuralForecast."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST

from src.train._common import (
    prepare_training_data,
    train_neuralforecast_model,
    get_common_training_params,
    get_processed_data_from_loader
)

logger = logging.getLogger(__name__)


def train_patchtst_model(
    model_type: str,
    cfg: Any,
    data: pd.DataFrame,
    model_name: str,
    outputs_dir: Path = None,
    model_params: Optional[Dict[str, Any]] = None,
    data_loader: Optional[Any] = None
) -> None:
    """Train PatchTST model using NeuralForecast.
    
    Parameters
    ----------
    model_type : str
        Model type: 'patchtst'
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
    logger.info(f"Training PatchTST model...")
    
    # Use processed (unstandardized) data instead of standardized
    # Let the model handle scaling internally
    data = get_processed_data_from_loader(data, data_loader, "PatchTST")
    logger.info(f"Training data shape: {data.shape}")
    
    # Prepare training data
    target_data, available_targets = prepare_training_data(data, model_params, data_loader)
    
    # Create model
    model = _create_patchtst_model(model_params, len(available_targets))
    
    # Train model
    train_neuralforecast_model(model, target_data, available_targets, outputs_dir)


def _create_patchtst_model(model_params: Dict[str, Any], n_series: int) -> NeuralForecast:
    """Create PatchTST model instance."""
    common_params = get_common_training_params(model_params)
    
    model_instance = PatchTST(
        **common_params,
        encoder_layers=model_params.get('num_layers', model_params.get('encoder_layers', 3)),
        n_heads=model_params.get('n_heads', 16),
        hidden_size=model_params.get('d_model', model_params.get('hidden_size', 128)),
        linear_hidden_size=model_params.get('linear_hidden_size', model_params.get('d_model', 128) * 2),
        dropout=model_params.get('dropout', 0.2),
        fc_dropout=model_params.get('fc_dropout', model_params.get('dropout', 0.2)),
        head_dropout=model_params.get('head_dropout', 0.0),
        attn_dropout=model_params.get('attn_dropout', 0.0),
        patch_len=model_params.get('patch_len', 16),
        stride=model_params.get('stride', 8),
        # Data is now unstandardized, so RevIN can be enabled (default)
        revin=model_params.get('revin', True),  # Default to True when data is not pre-standardized
        revin_affine=model_params.get('revin_affine', False),
        revin_subtract_last=model_params.get('revin_subtract_last', True),
        activation=model_params.get('activation', 'gelu'),
        res_attention=model_params.get('res_attention', True),
        batch_normalization=model_params.get('batch_normalization', False),
        learn_pos_embed=model_params.get('learn_pos_embed', True),
    )
    
    # Use default local_scaler_type='standard' to let NeuralForecast handle scaling
    return NeuralForecast(models=[model_instance], freq='W')
