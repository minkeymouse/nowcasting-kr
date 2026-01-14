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
    """Load trained Mamba model from checkpoint."""
    if not _HAS_MAMBA_SSM:
        raise ImportError("mamba-ssm not available. Please install: pip install mamba-ssm")
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    # Load metadata
    metadata_path = checkpoint_path.parent / "metadata.pkl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    metadata = joblib.load(metadata_path)
    
    # Load model first to get device
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
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Assume it's the model itself
        model.load_state_dict(checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint)
    
    # Move model to device
    model = model.to(model_device)
    model.eval()
    
    # Load input/output projections if exist and move to device
    input_proj = None
    output_proj = None
    input_proj_path = checkpoint_path.parent / "input_proj.pkl"
    if input_proj_path.exists():
        input_proj = joblib.load(input_proj_path)
        input_proj = input_proj.to(model_device)
        logger.info("Loaded input projection layer")
    
    output_proj_path = checkpoint_path.parent / "output_proj.pkl"
    if output_proj_path.exists():
        output_proj = joblib.load(output_proj_path)
        output_proj = output_proj.to(model_device)
        logger.info("Loaded output projection layer")
    
    return model, metadata, input_proj, output_proj


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
    model, metadata, input_proj, output_proj = load_mamba_model(checkpoint_path)
    device = model.device
    context_length = metadata['context_length']
    prediction_length = metadata['prediction_length']
    n_features = metadata.get('n_features', len(target_series) if target_series else None)
    available_targets = metadata.get('available_targets', target_series or list(test_data.columns))
    
    # Filter available targets
    available_targets = [t for t in available_targets if t in test_data.columns]
    if not available_targets:
        raise ValueError("No target series found in test_data")
    
    # Filter test data
    test_data_filtered, _, _ = filter_and_prepare_test_data(
        test_data, start_date, end_date, available_targets
    )
    
    # Get weekly dates
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
        
        # Get data up to cutoff
        train_data_up_to_cutoff = test_data_filtered[test_data_filtered.index <= cutoff_date][available_targets].copy()
        
        # Impute missing values
        train_data_up_to_cutoff = interpolate_missing_values(train_data_up_to_cutoff, data_loader)
        
        # Ensure we have enough data
        if len(train_data_up_to_cutoff) < context_length:
            logger.warning(f"Not enough data at {cutoff_date.date()}. Need {context_length}, have {len(train_data_up_to_cutoff)}")
            if len(train_data_up_to_cutoff) == 0:
                predictions.append(np.full(len(available_targets), np.nan))
                actuals.append(np.full(len(available_targets), np.nan))
                forecast_dates.append(next_date)
                continue
            
            # Pad with last value
            padding_needed = context_length - len(train_data_up_to_cutoff)
            last_row = train_data_up_to_cutoff.iloc[-1:].copy()
            padding = pd.concat([last_row] * padding_needed, ignore_index=True)
            padding.index = pd.date_range(
                end=train_data_up_to_cutoff.index[-1],
                periods=padding_needed + 1,
                freq='W'
            )[1:]
            train_data_up_to_cutoff = pd.concat([train_data_up_to_cutoff, padding])
        
    # Get context window
    context_data = train_data_up_to_cutoff.iloc[-context_length:].values.astype(np.float32)
    
    # Convert to tensor and move to device first
    x = torch.FloatTensor(context_data).unsqueeze(0).to(device)  # (1, context_length, n_features)
    
    # Apply input projection if needed (after moving to device)
    if input_proj is not None:
        x = input_proj(x)  # (1, context_length, d_model)
    
    # Predict
    with torch.no_grad():
        pred = model(x)  # (1, prediction_length, d_model)
        
        # Extract first prediction step (for recursive forecasting, we predict 1 step ahead)
        pred_values = pred[0, 0, :].cpu().numpy()  # (d_model,)
        
        # Project output back to original feature space if needed
        if output_proj is not None:
            pred_tensor = torch.FloatTensor(pred_values).unsqueeze(0).to(device)  # (1, d_model)
            pred_values = output_proj(pred_tensor)[0].cpu().numpy()  # (n_features,)
    
    predictions.append(pred_values)
    forecast_dates.append(next_date)
    
    # Get actual values
    if next_date in test_data_filtered.index:
        actual_values = test_data_filtered.loc[next_date, available_targets].values
        actuals.append(actual_values)
    else:
        logger.warning(f"No actual data for {next_date.date()}")
        actuals.append(np.full(len(available_targets), np.nan))
    
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
    model, metadata, input_proj, output_proj = load_mamba_model(checkpoint_path)
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
            logger.warning(f"Not enough data. Need {context_length}, have {len(train_data)}")
            if len(train_data) > 0:
                padding_needed = context_length - len(train_data)
                last_row = train_data.iloc[-1:].copy()
                padding = pd.concat([last_row] * padding_needed, ignore_index=True)
                padding.index = pd.date_range(
                    end=train_data.index[-1],
                    periods=padding_needed + 1,
                    freq='W'
                )[1:]
                train_data = pd.concat([train_data, padding])
            else:
                raise ValueError(f"No data available before {start_date}")
    else:
        raise ValueError("test_data is required for multi-horizon forecasting")
    
    # Get context window
    context_data = train_data.iloc[-context_length:].values.astype(np.float32)
    x = torch.FloatTensor(context_data).unsqueeze(0).to(device)  # (1, context_length, n_features)
    
    # Apply input projection if needed (after moving to device)
    if input_proj is not None:
        x = input_proj(x)  # (1, context_length, d_model)
    
    # Predict all horizons at once (model outputs prediction_length steps)
    with torch.no_grad():
        pred_all = model(x)  # (1, prediction_length, d_model)
        pred_all = pred_all[0].cpu().numpy()  # (prediction_length, d_model)
        
        # Project output back to original feature space if needed
        if output_proj is not None:
            pred_tensor = torch.FloatTensor(pred_all).to(device)  # (prediction_length, d_model)
            # Ensure output_proj is on the same device
            if next(output_proj.parameters()).device != device:
                pred_tensor = pred_tensor.to(next(output_proj.parameters()).device)
            pred_all = output_proj(pred_tensor).cpu().numpy()  # (prediction_length, n_features)
    
    # Extract forecasts for each horizon
    for horizon in horizons:
        if horizon <= len(pred_all):
            # Extract the horizon-th step (0-indexed, so horizon-1)
            forecast_values = pred_all[horizon - 1, :]
        else:
            # If horizon exceeds model's prediction_length, use last step
            logger.warning(f"Horizon {horizon}w exceeds model prediction_length {len(pred_all)}. Using last step.")
            forecast_values = pred_all[-1, :]
        
        if return_weekly_forecasts:
            # For monthly aggregation, return weekly forecasts in the month
            horizon_date = start_date_ts + pd.Timedelta(weeks=horizon)
            month_start = pd.Timestamp(year=horizon_date.year, month=horizon_date.month, day=1)
            month_end = month_start + pd.offsets.MonthEnd(0)
            
            # Get all weeks in the month
            weekly_dates = []
            weekly_forecasts = []
            current_date = month_start
            while current_date <= month_end:
                weeks_from_start = (current_date - start_date_ts).days // 7
                if 0 <= weeks_from_start < len(pred_all):
                    weekly_forecasts.append(pred_all[weeks_from_start, :])
                    weekly_dates.append(current_date)
                current_date += pd.Timedelta(weeks=1)
                if weeks_from_start >= max(horizons):
                    break
            
            if weekly_forecasts:
                forecasts[horizon] = {
                    'weekly_forecasts': np.array(weekly_forecasts),
                    'dates': pd.DatetimeIndex(weekly_dates)
                }
            else:
                forecasts[horizon] = forecast_values
        else:
            forecasts[horizon] = forecast_values
    
    return forecasts, available_targets


def forecast(
    checkpoint_path: Path,
    horizon: int,
    model_type: str = "mamba",
    recursive: bool = False,
    test_data: Optional[pd.DataFrame] = None,
    update_params: bool = False
) -> None:
    """Load trained Mamba model and generate forecasts.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to model checkpoint
    horizon : int
        Forecast horizon
    model_type : str, default "mamba"
        Model type ('mamba')
    recursive : bool, default False
        Whether to use recursive forecasting
    test_data : pd.DataFrame, optional
        Test data for recursive forecasting
    update_params : bool, default False
        Whether to update model parameters (not implemented yet)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading Mamba model from: {checkpoint_path}")
    
    if recursive and test_data is None:
        raise ValueError("test_data is required when recursive=True")
    
    # For simple forecasting, we need test_data to get context
    if test_data is None:
        logger.warning("No test_data provided. Cannot generate forecasts without context.")
        return
    
    # Load model, metadata, and projections
    model, metadata, input_proj, output_proj = load_mamba_model(checkpoint_path)
    device = model.device
    context_length = metadata['context_length']
    available_targets = metadata.get('available_targets', list(test_data.columns))
    
    # Get context
    context_data = test_data[available_targets].iloc[-context_length:].values.astype(np.float32)
    x = torch.FloatTensor(context_data).unsqueeze(0).to(device)  # (1, context_length, n_features)
    
    # Apply input projection if needed (after moving to device)
    if input_proj is not None:
        x = input_proj(x)  # (1, context_length, d_model)
    
    # Predict
    with torch.no_grad():
        pred = model(x)  # (1, prediction_length, d_model)
        forecast_values = pred[0, 0, :].cpu().numpy()  # First step
        
        # Project output back to original feature space if needed
        if output_proj is not None:
            pred_tensor = torch.FloatTensor(forecast_values).unsqueeze(0).to(device)  # (1, d_model)
            forecast_values = output_proj(pred_tensor)[0].cpu().numpy()  # (n_features,)
    
    logger.info(f"Forecast generated for horizon={horizon}")
    logger.info(f"Forecast values: {forecast_values}")
    
    # Save forecasts
    output_dir = checkpoint_path.parent / "forecasts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / "predictions.npy", forecast_values)
    logger.info(f"Forecasts saved to: {output_dir}")
