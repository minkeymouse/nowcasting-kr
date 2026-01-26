"""Common utilities for training NeuralForecast models."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
import pickle

from src.utils import interpolate_missing_values, convert_to_neuralforecast_format

logger = logging.getLogger(__name__)


def prepare_training_data(
    data: pd.DataFrame,
    model_params: Optional[Dict[str, Any]],
    data_loader: Optional[Any],
    use_covariates: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Prepare training data and extract available target series and covariates.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data with all variables
    model_params : dict, optional
        Model parameters dictionary
    data_loader : optional
        Data loader object
    use_covariates : bool, default True
        Whether to include non-target variables as covariates
    
    Returns
    -------
    tuple
        (target_data, covariate_data, available_targets, covariate_names)
    """
    # Get target series from config
    target_series = model_params.get('target_series') if model_params else None
    
    # Filter to available target series
    if target_series and len(target_series) > 0:
        available_targets = [t for t in target_series if t in data.columns]
        if not available_targets:
            logger.warning(f"No specified target series found. Using all columns.")
            available_targets = list(data.columns)
        else:
            missing = [t for t in target_series if t not in data.columns]
            if missing:
                logger.warning(f"Some target series not found: {missing}")
    else:
        available_targets = list(data.columns)
        logger.info("No target_series specified. Using all available columns.")
    
    if not available_targets:
        raise ValueError("No target series available for training.")
    
    logger.info(f"Target series ({len(available_targets)}): {available_targets[:5]}{'...' if len(available_targets) > 5 else ''}")
    
    # Select target series
    target_data = data[available_targets].copy()
    
    # Get covariates (all non-target variables, excluding date/metadata columns)
    if use_covariates:
        # Exclude date/metadata columns from covariates
        date_cols = {'date', 'date_w', 'Date', 'Date_w', 'year', 'month', 'day', 'Year', 'Month', 'Day'}
        covariate_names = [col for col in data.columns 
                          if col not in available_targets and col not in date_cols]
        if covariate_names:
            covariate_data = data[covariate_names].copy()
            logger.info(f"Including {len(covariate_names)} covariates: {covariate_names[:5]}{'...' if len(covariate_names) > 5 else ''}")
        else:
            covariate_data = pd.DataFrame(index=data.index)
            logger.info("No covariates available (all columns are target series or date columns)")
    else:
        covariate_data = pd.DataFrame(index=data.index)
        covariate_names = []
        logger.info("Covariates disabled - using only target series")
    
    # Ensure datetime index
    if not isinstance(target_data.index, pd.DatetimeIndex):
        if data_loader is not None and hasattr(data_loader, 'processed'):
            processed_data = data_loader.processed
            if isinstance(processed_data.index, pd.DatetimeIndex):
                target_data.index = processed_data.index[:len(target_data)]
                covariate_data.index = processed_data.index[:len(covariate_data)]
                logger.info("Using datetime index from data_loader.processed")
            else:
                dates = pd.date_range('2020-01-01', periods=len(target_data), freq='W')
                target_data.index = dates
                covariate_data.index = dates
                logger.warning("Creating synthetic weekly datetime index.")
        else:
            dates = pd.date_range('2020-01-01', periods=len(target_data), freq='W')
            target_data.index = dates
            covariate_data.index = dates
            logger.warning("Creating synthetic weekly datetime index.")
    
    # Interpolate NaN values (required for attention-based models)
    target_data = interpolate_missing_values(target_data, data_loader)
    if not covariate_data.empty:
        covariate_data = interpolate_missing_values(covariate_data, data_loader)
    
    return target_data, covariate_data, available_targets, covariate_names


def save_model_checkpoint(model: Any, model_path: Path, model_type: str) -> None:
    """Save model checkpoint using joblib, pickle, or dill.
    
    Parameters
    ----------
    model : Any
        Trained model to save
    model_path : Path
        Path where to save the model
    model_type : str
        Model type name (for error messages)
    """
    # Ensure directory exists
    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving model checkpoint to: {model_path.resolve()}")
    
    try:
        joblib.dump(model, model_path)
        logger.info(f"Model checkpoint saved successfully: {model_path}")
    except Exception as e:
        logger.warning(f"joblib save failed: {e}. Trying pickle...")
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Model saved using pickle: {model_path}")
        except Exception as e2:
            # For models with unpicklable objects, try dill
            logger.warning(f"pickle save failed: {e2}. Trying dill...")
            try:
                import dill
                with open(model_path, 'wb') as f:
                    dill.dump(model, f)
                logger.info(f"Model saved using dill: {model_path}")
            except ImportError:
                logger.error(f"Cannot save {model_type} model - dill not available. Install with: pip install dill")
                logger.error(f"Training completed but model could not be saved. Error: {e2}")
                raise
            except Exception as e3:
                logger.error(f"All save methods failed for {model_type}. Last error: {e3}")
                logger.error(f"Training completed but model could not be saved.")
                raise


def get_common_training_params(model_params: Dict[str, Any], n_samples: Optional[int] = None) -> Dict[str, Any]:
    """Extract common training parameters for NeuralForecast models.
    
    Parameters
    ----------
    model_params : dict
        Model parameters dictionary
    n_samples : int, optional
        Number of training samples. If provided, calculates max_steps based on actual batches per epoch.
        If None, uses a conservative estimate (assumes ~20 batches per epoch).
        
    Returns
    -------
    dict
        Common parameters dictionary
    """
    h = model_params.get('prediction_length', model_params.get('horizon', 1))
    input_size = model_params.get('context_length', model_params.get('n_lags', 96))
    batch_size = model_params.get('batch_size', 32)

    # NeuralForecast trains by max_steps. Prefer config-provided max_steps.
    max_steps = model_params.get('max_steps', None)

    # Backward-compatible fallback (legacy configs may still specify max_epochs).
    if max_steps is None:
        max_epochs = model_params.get('max_epochs', 10)
        # Calculate max_steps based on actual data if available
        if n_samples is not None:
            # Estimate batches per epoch (accounting for validation split ~20%)
            train_samples = int(n_samples * 0.8)
            batches_per_epoch = max(1, train_samples // batch_size)
            max_steps = max_epochs * batches_per_epoch
            logger.info(f"Calculated max_steps: {max_epochs} epochs × {batches_per_epoch} batches/epoch = {max_steps} steps")
        else:
            # Conservative estimate: assume ~20 batches per epoch
            batches_per_epoch_estimate = 20
            max_steps = max_epochs * batches_per_epoch_estimate
            logger.warning(
                f"Using estimated max_steps: {max_epochs} epochs × {batches_per_epoch_estimate} "
                f"batches/epoch = {max_steps} steps (estimate)"
            )
    else:
        logger.info(f"Using configured max_steps={max_steps}")
    
    # NeuralForecast models accept max_steps directly
    # When max_steps is set, PyTorch Lightning uses step-based training
    # Note: Progress bar may still show "Epoch" but numbers will be correct (steps)
    return {
        'h': h,
        'input_size': input_size,
        'batch_size': batch_size,
        'learning_rate': model_params.get('learning_rate', 0.001),
        'max_steps': max_steps,  # This ensures step-based training with correct total
        'val_check_steps': 50,
        'early_stop_patience_steps': 40,
        # No trainer_kwargs - NeuralForecast stores it nested which causes errors
        # max_steps is sufficient for step-based training
        # No scaler_type - let models use default internal scaling
    }


def get_processed_data_from_loader(
    data: pd.DataFrame,
    data_loader: Optional[Any],
    model_name: str = "model"
) -> pd.DataFrame:
    """Extract ORIGINAL (raw levels) data from data_loader for attention-based models.
    
    For attention-based models (TFT, PatchTST, iTransformer), we use ORIGINAL (raw levels)
    data WITHOUT any transformations (no differencing, no log transforms, etc.).
    The models handle normalization internally.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data (may be ignored if data_loader is available)
    data_loader : Any, optional
        Data loader with original and processed attributes
    model_name : str
        Model name for logging
    
    Returns
    -------
    pd.DataFrame
        Raw original data without date columns (NO transformations applied)
    """
    if data_loader is not None and hasattr(data_loader, 'original') and data_loader.original is not None:
        # Use ORIGINAL (raw) data - no transformations applied
        original_data = data_loader.original.copy()
        
        # Set index if date_w column exists
        if 'date_w' in original_data.columns:
            original_data['date_w'] = pd.to_datetime(original_data['date_w'], errors='coerce')
            original_data = original_data.set_index('date_w')
        
        # Remove date columns if present
        date_cols = ['date', 'date_w', 'year', 'month', 'day']
        data_cols = [col for col in original_data.columns if col not in date_cols]
        original_data = original_data[data_cols].copy()
        
        # Ensure only numeric columns
        original_data = original_data.select_dtypes(include=[np.number])
        
        # Ensure same columns as input data (in case of filtering)
        if hasattr(data, 'columns'):
            common_cols = [col for col in original_data.columns if col in data.columns]
            if common_cols:
                original_data = original_data[common_cols]
        
        logger.info(f"Using ORIGINAL (raw levels, NO transformations) data for {model_name}. Shape: {original_data.shape}")
        if len(original_data.columns) > 0:
            logger.info(f"Data stats - Mean range: [{original_data.mean().min():.2f}, {original_data.mean().max():.2f}], "
                       f"Std range: [{original_data.std().min():.2f}, {original_data.std().max():.2f}]")
        logger.info(f"  → No differencing or transformations applied. Model handles normalization internally.")
        return original_data
    else:
        logger.warning("No original data available from data_loader, using provided data as-is")
        return data


def train_neuralforecast_model(
    model: Any,
    target_data: pd.DataFrame,
    available_targets: list[str],
    outputs_dir: Path,
    covariate_data: Optional[pd.DataFrame] = None,
    covariate_names: Optional[list[str]] = None
) -> None:
    """Train a NeuralForecast model.
    
    Parameters
    ----------
    model : Any
        NeuralForecast model instance
    target_data : pd.DataFrame
        Prepared training data (target series only)
    available_targets : list[str]
        List of target series names
    outputs_dir : Path
        Directory to save the trained model
    covariate_data : pd.DataFrame, optional
        Covariate data (non-target variables)
    covariate_names : list[str], optional
        List of covariate column names
    """
    # Convert to NeuralForecast format (includes covariates if provided)
    nf_df = convert_to_neuralforecast_format(
        target_data, 
        available_targets,
        covariate_data=covariate_data,
        covariate_names=covariate_names
    )
    
    logger.info(f"Fitting model...")
    if covariate_names:
        logger.info(f"  Using {len(covariate_names)} covariates: {covariate_names[:5]}{'...' if len(covariate_names) > 5 else ''}")
    val_size = max(int(len(nf_df) * 0.2 / len(available_targets)), 12) if len(nf_df) > 20 else 0
    model.fit(df=nf_df, val_size=val_size if val_size > 0 else None)
    
    logger.info("Model training completed!")
    
    # Save model
    model_path = outputs_dir / "model.pkl"
    save_model_checkpoint(model, model_path, "NeuralForecast")
