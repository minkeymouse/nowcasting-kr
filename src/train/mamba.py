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
    
    def __init__(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        context_length: int,
        prediction_length: int
    ):
        """
        Parameters
        ----------
        x_data : np.ndarray
            Input time series data of shape (T, Dx)
        y_data : np.ndarray
            Target time series data of shape (T, Dy)
        context_length : int
            Input sequence length
        prediction_length : int
            Output sequence length
        """
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.total_length = context_length + prediction_length
        
    def __len__(self):
        return max(0, len(self.x_data) - self.total_length + 1)
    
    def __getitem__(self, idx):
        start_idx = idx
        end_idx = start_idx + self.total_length
        
        x_seq = self.x_data[start_idx:end_idx]
        y_seq = self.y_data[start_idx:end_idx]
        x = x_seq[:self.context_length]
        y = y_seq[self.context_length:]
        return x, y


class MambaForecaster(nn.Module):
    """Mamba-based time series forecasting model."""
    
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
        use_revin: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.device = device
        self.use_revin = use_revin
        
        self.mamba_blocks = nn.ModuleList([
            Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                   layer_idx=i, use_mem_eff_path=False, device=device)
            for i in range(n_layers)
        ])
        
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.prediction_head = nn.Linear(d_model, d_model * prediction_length)
        
    def forward(self, x):
        if self.use_revin:
            x_mean = x.mean(dim=1, keepdim=True)
            x_std = x.std(dim=1, keepdim=True)
            x_std = torch.clamp(x_std, min=1e-4) + 1e-5
            x = (x - x_mean) / x_std
        
        for mamba_block, norm in zip(self.mamba_blocks, self.norms):
            x = x + mamba_block(x)
            x = norm(x)
            x = self.dropout(x)
        
        last_hidden = x[:, -1, :]
        pred = self.prediction_head(last_hidden)
        pred = pred.reshape(-1, self.prediction_length, self.d_model)
        
        if self.use_revin:
            pred = pred * x_std + x_mean
        
        return pred


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
    
    data = get_processed_data_from_loader(data, data_loader, "Mamba")
    logger.info(f"Training data shape: {data.shape}")
    
    use_covariates = True
    if model_params is not None:
        use_covariates = model_params.get('use_covariates', True)
    target_data, covariate_data, available_targets, covariate_names = prepare_training_data(
        data, model_params, data_loader, use_covariates=use_covariates
    )
    covariates_as_inputs_only = False
    if model_params is not None:
        covariates_as_inputs_only = model_params.get('covariates_as_inputs_only', False)
    
    # Combine target and covariates for inputs; choose outputs by mode
    if len(covariate_names) > 0:
        input_data = pd.concat([target_data, covariate_data], axis=1)
        model_features = available_targets + covariate_names
        logger.info(
            f"Input features: {len(model_features)} variables "
            f"({len(available_targets)} targets + {len(covariate_names)} covariates)"
        )
        if covariates_as_inputs_only:
            output_data = target_data
            output_targets = available_targets
            logger.info("Outputs: target-only (covariates used as inputs only)")
        else:
            output_data = input_data
            output_targets = model_features
            logger.info("Outputs: targets + covariates (multi-output forecasting)")
    else:
        input_data = target_data
        output_data = target_data
        model_features = available_targets
        output_targets = available_targets
        logger.info(f"Using {len(available_targets)} target series (no covariates available)")
    
    x_array = input_data.values.astype(np.float32)
    y_array = output_data.values.astype(np.float32)
    n_features_in = x_array.shape[1]
    n_features_out = y_array.shape[1]
    logger.info(f"Total input features: {n_features_in}, output targets: {n_features_out}")
    d_model = model_params.get('d_model', model_params.get('hidden_size', 128))
    n_layers = model_params.get('n_layers', model_params.get('num_layers', 4))
    context_length = model_params.get('context_length', model_params.get('n_lags', 96))
    prediction_length = model_params.get('prediction_length', model_params.get('horizon', 1))
    d_state = model_params.get('d_state', 64)
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
    
    if n_features_in != d_model:
        input_proj = nn.Linear(n_features_in, d_model).to(device)
    else:
        input_proj = None
    output_proj = nn.Linear(d_model, n_features_out).to(device)
    
    # Create per-series StandardScaler for inputs and targets
    from sklearn.preprocessing import StandardScaler
    input_scaler = StandardScaler()
    target_scaler = StandardScaler()
    input_scaler.fit(x_array)
    target_scaler.fit(y_array)
    logger.info(
        f"Fitted input StandardScaler: mean={input_scaler.mean_[:5]}, scale={input_scaler.scale_[:5]}"
    )
    logger.info(
        f"Fitted target StandardScaler: mean={target_scaler.mean_[:5]}, scale={target_scaler.scale_[:5]}"
    )
    
    # Create dataset
    dataset = TimeSeriesDataset(x_array, y_array, context_length, prediction_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    logger.info(f"Dataset size: {len(dataset)}, Batches per epoch: {len(dataloader)}")
    
    use_revin = model_params.get('use_revin', True)
    model = MambaForecaster(
        d_model=d_model,
        n_layers=n_layers,
        context_length=context_length,
        prediction_length=prediction_length,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        dropout=dropout,
        use_revin=use_revin,
        device=device
    ).to(device)
    
    params = list(model.parameters())
    if input_proj is not None:
        params.extend(list(input_proj.parameters()))
        params.extend(list(output_proj.parameters()))
    optimizer = optim.Adam(params, lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop - two-stage scaling: StandardScaler (Stage 1) + RevIN (Stage 2)
    logger.info(f"Starting training for {max_epochs} epochs...")
    model.train()
    for epoch in range(max_epochs):
        total_loss = 0.0
        n_batches = 0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Stage 1: Apply per-series StandardScaler (training scale preservation)
            batch_x_np = batch_x.cpu().numpy()
            batch_x_scaled = input_scaler.transform(batch_x_np.reshape(-1, n_features_in)).reshape(batch_x.shape)
            batch_x_scaled = torch.FloatTensor(batch_x_scaled).to(device)
            
            if input_proj is not None:
                x = input_proj(batch_x_scaled)
            else:
                x = batch_x_scaled
            
            # Stage 2: RevIN handles context-based normalization internally
            pred = model(x)
            
            if output_proj is not None:
                pred = output_proj(pred)
            
            # Denormalize predictions using scaler (reverse Stage 1)
            # Keep gradient by doing operations in PyTorch
            target_mean = torch.FloatTensor(target_scaler.mean_).to(device)
            target_scale = torch.FloatTensor(target_scaler.scale_).to(device)
            pred_denorm = pred * target_scale.unsqueeze(0).unsqueeze(0) + target_mean.unsqueeze(0).unsqueeze(0)
            
            optimizer.zero_grad()
            loss = criterion(pred_denorm, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        logger.info(f"Epoch {epoch+1}/{max_epochs}, Loss: {total_loss/n_batches:.6f}")
    
    logger.info("Training completed!")
    
    model_path = outputs_dir / "model.pkl"
    save_model_checkpoint(model, model_path, "Mamba")
    
    if input_proj is not None:
        joblib.dump(input_proj, outputs_dir / "input_proj.pkl")
    joblib.dump(output_proj, outputs_dir / "output_proj.pkl")
    
    # Save per-series StandardScalers (Stage 1: Training scale preservation)
    joblib.dump(input_scaler, outputs_dir / "input_scaler.pkl")
    joblib.dump(target_scaler, outputs_dir / "target_scaler.pkl")
    logger.info(f"Saved input StandardScaler to {outputs_dir / 'input_scaler.pkl'}")
    logger.info(f"Saved target StandardScaler to {outputs_dir / 'target_scaler.pkl'}")
    
    # Save metadata
    metadata = {
        'd_model': d_model,
        'n_layers': n_layers,
        'context_length': context_length,
        'prediction_length': prediction_length,
        'd_state': d_state,
        'd_conv': d_conv,
        'expand': expand,
        'n_features_in': n_features_in,
        'n_features_out': n_features_out,
        'has_input_proj': input_proj is not None,
        'has_output_proj': True,
        'model_features': model_features,
        'available_targets': output_targets,
        'data_shape': x_array.shape,
        'covariates_as_inputs_only': covariates_as_inputs_only,
        'use_revin': use_revin
    }
    metadata_path = outputs_dir / "metadata.pkl"
    joblib.dump(metadata, metadata_path)
    logger.info(f"Metadata saved: {metadata_path}")
