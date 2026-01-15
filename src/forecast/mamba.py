"""Forecasting functions for Mamba model."""

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
import numpy as np
import torch
import joblib

from src.utils import (
    interpolate_missing_values,
    filter_and_prepare_test_data
)

logger = logging.getLogger(__name__)

try:
    from mamba_ssm import Mamba2
    _HAS_MAMBA_SSM = True
except ImportError:
    _HAS_MAMBA_SSM = False
    logger.error("mamba-ssm not available. Please install: pip install mamba-ssm")


def load_mamba_model(checkpoint_path: Path, device: str = None):
    """Load trained Mamba model from checkpoint.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to model checkpoint
    device : str, optional
        Device to load model on
    """
    if not _HAS_MAMBA_SSM:
        raise ImportError("mamba-ssm not available. Please install: pip install mamba-ssm")
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    metadata = joblib.load(checkpoint_path.parent / "metadata.pkl")
    from src.train.mamba import MambaForecaster
    
    model_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MambaForecaster(
        d_model=metadata['d_model'],
        n_layers=metadata['n_layers'],
        context_length=metadata['context_length'],
        prediction_length=metadata['prediction_length'],
        d_state=metadata.get('d_state', 64),
        d_conv=metadata.get('d_conv', 4),
        expand=metadata.get('expand', 2),
        dropout=metadata.get('dropout', 0.1),
        device=model_device
    )
    
    checkpoint = joblib.load(checkpoint_path)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        model = model.to(model_device).eval()
    elif hasattr(checkpoint, 'state_dict') and hasattr(checkpoint, 'load_state_dict'):
        model = checkpoint.to(model_device).eval()
    else:
        try:
            model.load_state_dict(checkpoint)
            model = model.to(model_device).eval()
        except Exception:
            model = checkpoint.to(model_device).eval()
    
    input_proj = None
    output_proj = None
    if (checkpoint_path.parent / "input_proj.pkl").exists():
        input_proj = joblib.load(checkpoint_path.parent / "input_proj.pkl").to(model_device)
    if (checkpoint_path.parent / "output_proj.pkl").exists():
        output_proj = joblib.load(checkpoint_path.parent / "output_proj.pkl").to(model_device)
    
    # Load per-series StandardScaler (Stage 1: Training scale preservation)
    scaler = None
    if (checkpoint_path.parent / "scaler.pkl").exists():
        scaler = joblib.load(checkpoint_path.parent / "scaler.pkl")
        logger.info(f"Loaded per-series StandardScaler from {checkpoint_path.parent / 'scaler.pkl'}")
    
    return model, metadata, input_proj, output_proj, scaler


def run_recursive_forecast(
    checkpoint_path: Path,
    test_data: pd.DataFrame,
    start_date: str,
    end_date: str,
    model_type: str,
    target_series: Optional[List[str]] = None,
    data_loader: Optional[Any] = None,
    update_params: bool = False
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, List[str]]:
    """Run recursive forecasting experiment with weekly updates for Mamba.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to trained model checkpoint
    test_data : pd.DataFrame
        Test data with datetime index
    start_date : str
        Start date for experiment (YYYY-MM-DD)
    end_date : str
        End date for experiment (YYYY-MM-DD)
    model_type : str
        Model type ('mamba')
    target_series : list, optional
        List of target series names
    data_loader : optional
        Data loader object
    update_params : bool, default False
        If True, retrain model at each step (not implemented yet)
    
    Returns
    -------
    tuple
        (predictions, actuals, dates, target_series)
    """
    if not _HAS_MAMBA_SSM:
        raise ImportError("mamba-ssm not available. Please install: pip install mamba-ssm")
    
    # Load model, metadata, and projections
    model, metadata, input_proj, output_proj, scaler = load_mamba_model(checkpoint_path)
    device = model.device
    context_length = metadata['context_length']
    prediction_length = metadata['prediction_length']
    n_features = metadata.get('n_features', len(target_series) if target_series else None)
    available_targets = metadata.get('available_targets', target_series or list(test_data.columns))
    
    # Filter available targets
    available_targets = [t for t in available_targets if t in test_data.columns]
    if not available_targets:
        raise ValueError("No target series found in test_data")
    
    test_data_filtered, _, _ = filter_and_prepare_test_data(
        test_data, start_date, end_date, available_targets
    )
    test_data_for_context = test_data[available_targets].copy() if available_targets else test_data.copy()
    
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    weekly_dates = test_data_filtered.index[
        (test_data_filtered.index >= start_ts) & (test_data_filtered.index <= end_ts)
    ].sort_values()
    
    if len(weekly_dates) < 2:
        raise ValueError(f"Not enough weekly dates. Found {len(weekly_dates)} dates.")
    
    logger.info(f"Running recursive Mamba forecast from {start_ts.date()} to {end_ts.date()} ({len(weekly_dates)} weeks)")
    
    predictions = []
    actuals = []
    forecast_dates = []
    
    # Loop over weekly cutoffs
    for i in range(len(weekly_dates) - 1):
        cutoff_date = weekly_dates[i]
        next_date = weekly_dates[i + 1]
        
        train_data_up_to_cutoff = test_data_for_context[test_data_for_context.index <= cutoff_date].copy()
        train_data_up_to_cutoff = interpolate_missing_values(train_data_up_to_cutoff, data_loader)
        
        if len(train_data_up_to_cutoff) < context_length:
            if len(train_data_up_to_cutoff) == 0:
                predictions.append(np.full(len(available_targets), np.nan))
                actuals.append(np.full(len(available_targets), np.nan))
                forecast_dates.append(next_date)
                continue
            padding_needed = context_length - len(train_data_up_to_cutoff)
            last_row = train_data_up_to_cutoff.iloc[-1:].copy()
            padding = pd.concat([last_row] * padding_needed, ignore_index=True)
            padding.index = pd.date_range(end=train_data_up_to_cutoff.index[-1], periods=padding_needed + 1, freq='W')[1:]
            train_data_up_to_cutoff = pd.concat([train_data_up_to_cutoff, padding])
        
        context_data = train_data_up_to_cutoff.iloc[-context_length:].values.astype(np.float32)
        
        # Stage 1: Apply per-series StandardScaler (training scale preservation)
        if scaler is not None:
            context_data_scaled = scaler.transform(context_data.reshape(-1, context_data.shape[1])).reshape(context_data.shape)
        else:
            context_data_scaled = context_data
        
        x = torch.FloatTensor(context_data_scaled).unsqueeze(0).to(device)
        
        if input_proj is not None:
            x = input_proj(x)
        
        with torch.no_grad():
            # Stage 2: RevIN handles context-based normalization internally
            pred = model(x)[0, 0, :]
            if output_proj is not None:
                pred = output_proj(pred.unsqueeze(0))[0]
            
        # Keep predictions in processed/transformed space (same as training)
        # Do NOT inverse transform - predictions stay in processed space for evaluation
        pred = pred.cpu().numpy()
        
        predictions.append(pred)
        forecast_dates.append(next_date)
        actuals.append(
            test_data_filtered.loc[next_date, available_targets].values
            if next_date in test_data_filtered.index
            else np.full(len(available_targets), np.nan)
        )
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    forecast_dates = pd.DatetimeIndex(forecast_dates)
    
    return predictions, actuals, forecast_dates, available_targets


def run_multi_horizon_forecast(
    checkpoint_path: Path,
    horizons: List[int],
    start_date: str,
    test_data: Optional[pd.DataFrame] = None,
    model_type: str = "mamba",
    target_series: Optional[List[str]] = None,
    data_loader: Optional[Any] = None,
    return_weekly_forecasts: bool = False
) -> Tuple[Dict[int, Any], List[str]]:
    """Run multi-horizon forecasting from fixed start point for Mamba.
    
    Uses ONE model (trained with max horizon) to predict all horizons.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to trained model checkpoint
    horizons : list
        List of forecast horizons (weeks), e.g. [4, 8, 12, ..., 40]
    start_date : str
        Fixed start date for all forecasts (YYYY-MM-DD)
    test_data : pd.DataFrame, optional
        Test data for extracting actuals
    model_type : str, default "mamba"
        Model type ('mamba')
    target_series : list, optional
        List of target series names
    data_loader : optional
        Data loader object
    return_weekly_forecasts : bool, default False
        If True, return weekly forecasts within each horizon month
    
    Returns
    -------
    tuple
        (horizon_forecasts dict, target_series list)
    """
    if not _HAS_MAMBA_SSM:
        raise ImportError("mamba-ssm not available. Please install: pip install mamba-ssm")
    
    # Load model, metadata, and projections
    model, metadata, input_proj, output_proj, scaler = load_mamba_model(checkpoint_path)
    device = model.device
    context_length = metadata['context_length']
    prediction_length = metadata['prediction_length']
    available_targets = metadata.get('available_targets', target_series or [])
    
    if test_data is not None:
        available_targets = [t for t in available_targets if t in test_data.columns]
    
    if not available_targets:
        raise ValueError("No target series available")
    
    start_date_ts = pd.Timestamp(start_date)
    forecasts = {}
    
    # Get context data up to start_date
    if test_data is not None:
        train_data = test_data[test_data.index < start_date_ts][available_targets].copy()
        train_data = interpolate_missing_values(train_data, data_loader)
        
        if len(train_data) < context_length:
            if len(train_data) == 0:
                raise ValueError(f"No data available before {start_date}")
            padding_needed = context_length - len(train_data)
            last_row = train_data.iloc[-1:].copy()
            padding = pd.concat([last_row] * padding_needed, ignore_index=True)
            padding.index = pd.date_range(end=train_data.index[-1], periods=padding_needed + 1, freq='W')[1:]
            train_data = pd.concat([train_data, padding])
    else:
        raise ValueError("test_data is required for multi-horizon forecasting")
    
    context_data = train_data.iloc[-context_length:].values.astype(np.float32)
    
    # Stage 1: Apply per-series StandardScaler (training scale preservation)
    if scaler is not None:
        context_data_scaled = scaler.transform(context_data.reshape(-1, context_data.shape[1])).reshape(context_data.shape)
    else:
        context_data_scaled = context_data
    
    x = torch.FloatTensor(context_data_scaled).unsqueeze(0).to(device)
    
    if input_proj is not None:
        x = input_proj(x)
    
    with torch.no_grad():
        # Stage 2: RevIN handles context-based normalization internally
        pred_all = model(x)[0]
        if output_proj is not None:
            pred_all = output_proj(pred_all)
        
        # Keep predictions in processed/transformed space (same as training)
        # Do NOT inverse transform - predictions stay in processed space for evaluation
        pred_all = pred_all.cpu().numpy()
    
    # Extract forecasts for each horizon
    for horizon in horizons:
        idx = min(horizon - 1, len(pred_all) - 1)
        forecasts[horizon] = pred_all[idx, :]
    
    return forecasts, available_targets


def forecast(
    checkpoint_path: Path,
    horizon: int,
    model_type: str = "mamba",
    recursive: bool = False,
    test_data: Optional[pd.DataFrame] = None,
    update_params: bool = False
) -> None:
    """Load trained Mamba model and generate forecasts."""
    if test_data is None:
        raise ValueError("test_data is required for forecasting")
    
    model, metadata, input_proj, output_proj, scaler = load_mamba_model(checkpoint_path)
    context_length = metadata['context_length']
    available_targets = metadata.get('available_targets', list(test_data.columns))
    
    context_data = test_data[available_targets].iloc[-context_length:].values.astype(np.float32)
    
    # Stage 1: Apply per-series StandardScaler (training scale preservation)
    if scaler is not None:
        context_data_scaled = scaler.transform(context_data.reshape(-1, context_data.shape[1])).reshape(context_data.shape)
    else:
        context_data_scaled = context_data
    
    x = torch.FloatTensor(context_data_scaled).unsqueeze(0).to(model.device)
    
    if input_proj is not None:
        x = input_proj(x)
    
    with torch.no_grad():
        # Stage 2: RevIN handles context-based normalization internally
        pred = model(x)[0, 0, :]
        if output_proj is not None:
            pred = output_proj(pred.unsqueeze(0))[0]
        
        # Keep predictions in processed/transformed space (same as training)
        # Do NOT inverse transform - predictions stay in processed space for evaluation
        pred = pred.cpu().numpy()
    
    output_dir = checkpoint_path.parent / "forecasts"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "predictions.npy", pred)
    logger.info(f"Forecasts saved to: {output_dir}")
