"""Training function for Mamba model using mamba-ssm."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib

from src.train._common import (
    prepare_training_data,
    get_processed_data_from_loader,
    save_model_checkpoint
)

logger = logging.getLogger(__name__)

try:
    from mamba_ssm import Mamba2
    _HAS_MAMBA_SSM = True
except ImportError:
    _HAS_MAMBA_SSM = False
    Mamba2 = None
    logger.error("mamba-ssm not available. Please install: pip install mamba-ssm")


class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting with sliding windows."""
    
    def __init__(self, data: np.ndarray, context_length: int, prediction_length: int):
        """
        Parameters
        ----------
        data : np.ndarray
            Time series data of shape (T, D) where T is time steps, D is features
        context_length : int
            Input sequence length
        prediction_length : int
            Output sequence length
        """
        self.data = torch.FloatTensor(data)
        self.context_length = context_length
        self.prediction_length = prediction_length
        # For training: input is context_length, output is prediction_length
        # We need context_length + prediction_length total length
        self.total_length = context_length + prediction_length
        
    def __len__(self):
        return max(0, len(self.data) - self.total_length + 1)
    
    def __getitem__(self, idx):
        start_idx = idx
        end_idx = start_idx + self.total_length
        
        sequence = self.data[start_idx:end_idx]
        x = sequence[:self.context_length]  # (context_length, D)
        y = sequence[self.context_length:]  # (prediction_length, D)
        
        return x, y


class MambaForecaster(nn.Module):
    """Mamba-based time series forecasting model using Mamba2 blocks.
    
    Similar to other models (PatchTST, TFT, iTransformer):
    - Input: (B, context_length, d_model)
    - Output: (B, prediction_length, d_model)
    """
    
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        context_length: int,
        prediction_length: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.device = device
        
        # Note: Input projection from n_features to d_model is handled externally
        # This model expects input of shape (B, context_length, d_model)
        
        # Stack of Mamba2 blocks
        # Mamba2 maintains input shape: (B, L, d_model) -> (B, L, d_model)
        # Set use_mem_eff_path=False to avoid requiring causal-conv1d
        self.mamba_blocks = nn.ModuleList([
            Mamba2(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                layer_idx=i,
                use_mem_eff_path=False,  # Disable to avoid causal-conv1d requirement
                device=device
            ) for i in range(n_layers)
        ])
        
        # Layer normalization after each block
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        # If input projection is used, we need to project back to original feature space
        # For now, we'll use a learnable projection
        self.output_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (B, context_length, d_model)
            
        Returns
        -------
        torch.Tensor
            Output of shape (B, prediction_length, d_model)
        """
        # Note: x should already be of shape (B, context_length, d_model)
        # Input projection from n_features to d_model is handled externally
        
        # Pass through Mamba2 blocks with residual connections
        for i, (mamba_block, norm) in enumerate(zip(self.mamba_blocks, self.norms)):
            residual = x
            x = mamba_block(x)  # (B, context_length, d_model)
            x = x + residual  # Residual connection
            x = norm(x)
            x = self.dropout(x)
        
        # Extract last prediction_length time steps
        # If prediction_length <= context_length, take last prediction_length steps
        # If prediction_length > context_length, we need to handle differently
        if self.prediction_length <= self.context_length:
            # Extract last prediction_length steps
            x = x[:, -self.prediction_length:, :]  # (B, prediction_length, d_model)
        else:
            # If prediction_length > context_length, repeat last step
            # This is a limitation - for long-term forecasting, we'd need autoregressive generation
            last_step = x[:, -1:, :]  # (B, 1, d_model)
            x = last_step.repeat(1, self.prediction_length, 1)  # (B, prediction_length, d_model)
            logger.warning(
                f"prediction_length ({self.prediction_length}) > context_length ({self.context_length}). "
                f"Repeating last step. Consider using autoregressive generation for better results."
            )
        
        # Final output projection
        x = self.output_proj(x)
        
        return x


def train_mamba_model(
    model_type: str,
    cfg: Any,
    data: pd.DataFrame,
    model_name: str,
    outputs_dir: Path = None,
    model_params: Optional[Dict[str, Any]] = None,
    data_loader: Optional[Any] = None
) -> None:
    """Train Mamba model using mamba-ssm.
    
    Parameters
    ----------
    model_type : str
        Model type: 'mamba'
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
    if not _HAS_MAMBA_SSM:
        raise ImportError("mamba-ssm not available. Please install: pip install mamba-ssm")
    
    logger.info(f"Training Mamba model...")
    
    # Use processed (unstandardized) data - similar to other attention-based models
    data = get_processed_data_from_loader(data, data_loader, "Mamba")
    logger.info(f"Training data shape: {data.shape}")
    
    # Prepare training data
    target_data, available_targets = prepare_training_data(data, model_params, data_loader)
    
    # Convert to numpy
    data_array = target_data.values.astype(np.float32)
    n_features = data_array.shape[1]
    
    # Model parameters
    d_model = model_params.get('d_model', model_params.get('hidden_size', 128))
    n_layers = model_params.get('n_layers', model_params.get('num_layers', 4))
    context_length = model_params.get('context_length', model_params.get('n_lags', 96))
    prediction_length = model_params.get('prediction_length', model_params.get('horizon', 1))
    d_state = model_params.get('d_state', 64)  # Mamba2 default is 64
    d_conv = model_params.get('d_conv', 4)
    expand = model_params.get('expand', 2)
    dropout = model_params.get('dropout', 0.1)
    
    batch_size = model_params.get('batch_size', 32)
    learning_rate = model_params.get('learning_rate', 0.001)
    max_epochs = model_params.get('max_epochs', 10)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Model config: d_model={d_model}, n_layers={n_layers}, "
                f"context_length={context_length}, prediction_length={prediction_length}")
    
    # Input/output projections: n_features <-> d_model
    # If n_features != d_model, we need projection layers
    if n_features != d_model:
        logger.info(f"Projecting input from {n_features} features to d_model={d_model}")
        input_proj = nn.Linear(n_features, d_model).to(device)
        output_proj = nn.Linear(d_model, n_features).to(device)
    else:
        input_proj = None
        output_proj = None
    
    # Create dataset
    dataset = TimeSeriesDataset(data_array, context_length, prediction_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    logger.info(f"Dataset size: {len(dataset)}, Batches per epoch: {len(dataloader)}")
    
    # Create model
    model = MambaForecaster(
        d_model=d_model,
        n_layers=n_layers,
        context_length=context_length,
        prediction_length=prediction_length,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        dropout=dropout,
        device=device
    ).to(device)
    
    # Optimizer and loss - include input/output projections if they exist
    params = list(model.parameters())
    if input_proj is not None:
        params.extend(list(input_proj.parameters()))
        params.extend(list(output_proj.parameters()))
    optimizer = optim.Adam(params, lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    logger.info(f"Starting training for {max_epochs} epochs...")
    model.train()
    for epoch in range(max_epochs):
        total_loss = 0.0
        n_batches = 0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)  # (B, context_length, n_features)
            batch_y = batch_y.to(device)  # (B, prediction_length, n_features)
            
            # Project input if needed
            if input_proj is not None:
                batch_x = input_proj(batch_x)  # (B, context_length, d_model)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(batch_x)  # (B, prediction_length, d_model)
            
            # Project output back to original feature space if needed
            if output_proj is not None:
                pred = output_proj(pred)  # (B, prediction_length, n_features)
            
            # Loss - compare with original batch_y (n_features)
            loss = criterion(pred, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        logger.info(f"Epoch {epoch+1}/{max_epochs}, Loss: {avg_loss:.6f}")
    
    logger.info("Training completed!")
    
    # Save model
    model_path = outputs_dir / "model.pkl"
    save_model_checkpoint(model, model_path, "Mamba")
    
    # Save input/output projections if used
    if input_proj is not None:
        input_proj_path = outputs_dir / "input_proj.pkl"
        joblib.dump(input_proj, input_proj_path)
        logger.info(f"Input projection saved: {input_proj_path}")
        
        output_proj_path = outputs_dir / "output_proj.pkl"
        joblib.dump(output_proj, output_proj_path)
        logger.info(f"Output projection saved: {output_proj_path}")
    
    # Save metadata
    metadata = {
        'd_model': d_model,
        'n_layers': n_layers,
        'context_length': context_length,
        'prediction_length': prediction_length,
        'd_state': d_state,
        'd_conv': d_conv,
        'expand': expand,
        'n_features': n_features,
        'has_input_proj': input_proj is not None,
        'has_output_proj': output_proj is not None,
        'available_targets': available_targets,
        'data_shape': data_array.shape
    }
    metadata_path = outputs_dir / "metadata.pkl"
    joblib.dump(metadata, metadata_path)
    logger.info(f"Metadata saved: {metadata_path}")
