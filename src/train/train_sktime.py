"""Training module for sktime-based models (ARIMA, VAR, TFT, LSTM, Chronos).

This module handles training for models that use sktime forecaster interface
or are implemented in src.models modules.
"""

from pathlib import Path
import sys
import logging
import shutil
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np
import pandas as pd

# Path setup
from src.utils import get_project_root
project_root = get_project_root()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import (
    ValidationError,
    get_experiment_cfg,
    resolve_data_path,
    load_and_filter_data,
    TRAIN_START,
    TRAIN_END
)

try:
    from omegaconf import DictConfig, OmegaConf
except ImportError as e:
    raise ImportError(f"Required dependencies not available: {e}")

from src.train.preprocess import (
    set_dataframe_frequency,
    prepare_univariate_data,
    impute_missing_values
)

logger = logging.getLogger(__name__)


def train_sktime_model(
    model_type: str,
    config_name: str,
    cfg: DictConfig,
    data_file: str,
    model_name: Optional[str],
    horizons: Optional[List[int]],
    outputs_dir: Path,
    model_params: Optional[dict] = None
) -> Dict[str, Any]:
    """Train sktime-based model (ARIMA, VAR, TFT, LSTM, Chronos).
    
    Parameters
    ----------
    model_type : str
        Model type ('arima', 'var', 'tft', 'lstm', 'chronos')
    config_name : str
        Config name for logging
    cfg : DictConfig
        Hydra configuration
    data_file : str
        Path to data file
    model_name : Optional[str]
        Model name
    horizons : Optional[List[int]]
        Forecast horizons
    outputs_dir : Path
        Output directory for checkpoints
    model_params : Optional[dict]
        Model-specific parameters
        
    Returns
    -------
    Dict[str, Any]
        Training result with status, model_name, model_dir, metrics
    """
    # Load and filter data
    from src.utils import load_and_filter_data, TRAIN_START, TRAIN_END
    data_path = resolve_data_path(data_file) if data_file else resolve_data_path()
    data = load_and_filter_data(data_path, TRAIN_START, TRAIN_END)
    
    if len(data) == 0:
        raise ValidationError(f"No data available in training period (1985-2019). Data range: {data.index.min()} to {data.index.max()}")
    
    logger.info(f"Training data period: {data.index.min()} to {data.index.max()} (1985-2019 enforced)")
    logger.info(f"Data frequency: Weekly ({len(data)} points)")
    
    # Extract target series
    exp_cfg = get_experiment_cfg(cfg)
    target_series = exp_cfg.get('target_series')
    
    if not target_series or target_series not in data.columns:
        raise ValidationError(f"target_series '{target_series}' not found in data. Available columns: {list(data.columns)}")
    
    # Get model parameters
    if model_params is None:
        model_params = {}
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Training {model_type.upper()} model")
    logger.info(f"{'='*70}")
    logger.info(f"Config: {config_name}")
    logger.info(f"Target: {target_series}")
    logger.info(f"Data: {data_file}")
    
    # Unified preprocessing for univariate models
    # VAR is multivariate and handled separately below
    y_train = None
    if model_type.lower() != 'var':
        # Unified preprocessing: prepare univariate data for univariate sktime models
        # All models now train on weekly data (no resampling to monthly)
        y_train = prepare_univariate_data(data, target_series)
        if len(y_train) == 0:
            raise ValidationError(f"No valid weekly data for target series '{target_series}' after preprocessing")
        
        y_train = set_dataframe_frequency(y_train.to_frame()).iloc[:, 0]
        
        # Unified preprocessing already includes final imputation in prepare_univariate_data()
        # Verify no NaNs remain
        if y_train.isnull().any():
            nan_count = int(y_train.isnull().sum())
            raise ValidationError(f"{model_type.upper()}: Found {nan_count} NaN values after unified preprocessing. This should not happen.")
    
    # Train model based on type
    model_metadata = {}
    checkpoint_dir_temp = outputs_dir / f"temp_{model_type}"
    checkpoint_dir_temp.mkdir(parents=True, exist_ok=True)
    
    try:
        if model_type == 'var':
            # VAR is multivariate - needs original data for prepare_multivariate_data
            from src.models import train_var
            forecaster, model_metadata = train_var(
                data=data, target_series=target_series, config=model_params, cfg=cfg, checkpoint_dir=checkpoint_dir_temp
            )
        else:
            # Univariate models: use preprocessed y_train from unified preprocessing
            if model_type == 'arima':
                from src.models import train_arima
                forecaster, model_metadata = train_arima(
                    y_train=y_train, target_series=target_series, config=model_params, checkpoint_dir=checkpoint_dir_temp
                )
            elif model_type == 'tft':
                from src.models import train_tft
                # Prepare covariates: all other series except target
                X_train = None
                if len(data.columns) > 1:
                    # Get all series except target as covariates
                    covariate_series = [col for col in data.columns if col != target_series]
                    if covariate_series:
                        # Prepare covariates with same preprocessing as target
                        X_train = data[covariate_series].copy()
                        # Filter out non-numeric columns (e.g., date_w, date) that can't be scaled
                        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
                        X_train = X_train[numeric_cols]
                        if len(numeric_cols) > 0:
                            # Apply same preprocessing: set frequency, impute
                            X_train = set_dataframe_frequency(X_train)
                            X_train = impute_missing_values(X_train)
                            logger.info(f"Using {len(numeric_cols)} numeric covariates: {numeric_cols[:5]}{'...' if len(numeric_cols) > 5 else ''}")
                        else:
                            X_train = None
                            logger.warning("No numeric covariates available after filtering")
                # Get original data (before transformations) for scaler fitting
                y_train_original = None
                if target_series in data.columns:
                    y_train_original = data[target_series].dropna()
                    # Align with y_train index
                    if len(y_train_original) > 0:
                        y_train_original = y_train_original.reindex(y_train.index, method='ffill').bfill()
                forecaster, model_metadata = train_tft(
                    y_train=y_train, target_series=target_series, config=model_params, cfg=cfg, 
                    checkpoint_dir=checkpoint_dir_temp, X_train=X_train, y_train_original=y_train_original
                )
            elif model_type == 'lstm':
                from src.models import train_lstm
                # Prepare covariates: all other series except target
                X_train = None
                if len(data.columns) > 1:
                    covariate_series = [col for col in data.columns if col != target_series]
                    if covariate_series:
                        X_train = data[covariate_series].copy()
                        # Filter out non-numeric columns (e.g., date_w, date) that can't be scaled
                        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
                        X_train = X_train[numeric_cols]
                        if len(numeric_cols) > 0:
                            X_train = set_dataframe_frequency(X_train)
                            X_train = impute_missing_values(X_train)
                            logger.info(f"Using {len(numeric_cols)} numeric covariates: {numeric_cols[:5]}{'...' if len(numeric_cols) > 5 else ''}")
                        else:
                            X_train = None
                            logger.warning("No numeric covariates available after filtering")
                # Get original data (before transformations) for scaler fitting
                y_train_original = None
                if target_series in data.columns:
                    y_train_original = data[target_series].dropna()
                    # Align with y_train index
                    if len(y_train_original) > 0:
                        y_train_original = y_train_original.reindex(y_train.index, method='ffill').bfill()
                forecaster, model_metadata = train_lstm(
                    y_train=y_train, target_series=target_series, config=model_params, 
                    checkpoint_dir=checkpoint_dir_temp, X_train=X_train, y_train_original=y_train_original
                )
            elif model_type == 'chronos':
                from src.models import train_chronos
                # Get original data (before transformations) for scaler fitting
                # This ensures inverse transform returns to original scale
                y_train_original = None
                if target_series in data.columns:
                    y_train_original = data[target_series].dropna()
                    # Align with y_train index
                    if len(y_train_original) > 0:
                        y_train_original = y_train_original.reindex(y_train.index, method='ffill').bfill()
                forecaster, model_metadata = train_chronos(
                    y_train=y_train, target_series=target_series, config=model_params, 
                    checkpoint_dir=checkpoint_dir_temp, y_train_original=y_train_original
                )
            else:
                raise ValidationError(f"Unknown sktime model type: {model_type}")
    except ImportError as e:
        raise
    except Exception as e:
        raise
    finally:
        # Always clean up temp checkpoint
        if checkpoint_dir_temp.exists():
            shutil.rmtree(checkpoint_dir_temp)
    
    logger.info(f"{'='*70}\n")
    
    # Determine model directory and name
    # If outputs_dir already ends with model_name, use it directly
    # Otherwise, create subdirectory with model_name
    if model_name and (str(outputs_dir).endswith(model_name) or str(outputs_dir.name) == model_name):
        model_dir = outputs_dir
        final_model_name = outputs_dir.name
    else:
        # If outputs_dir already has the target_model structure, use it
        # Otherwise, append model_name
        if model_name and outputs_dir.name == model_name:
            model_dir = outputs_dir
            final_model_name = model_name
        else:
            final_model_name = model_name or (
                f"{model_type}_{target_series}_{datetime.now().strftime('%Y%m%d_%H%M%S')}" 
                if target_series else f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            model_dir = outputs_dir / final_model_name
    
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / "model.pkl"
    
    # Prepare metadata
    metadata = model_metadata.copy() if model_metadata else {}
    metadata.update({
        'model_type': model_type,
        'target_series': target_series,
        'config': OmegaConf.to_container(cfg, resolve=True)
    })
    
    # Save checkpoint using unified function
    from src.models import save_model_checkpoint
    try:
        save_model_checkpoint(forecaster, model_file, metadata)
    except Exception as e:
        raise ValidationError(f"Failed to save model checkpoint: {e}") from e
    
    # Build metrics dict (sktime models don't have convergence metrics)
    metrics = {
        'model_type': model_type,
        'forecast_metrics': {},
        'training_completed': True,
        'converged': True,
        'num_iter': 0,
        'loglik': np.nan
    }
    
    result = {'converged': True, 'num_iter': 0, 'loglik': np.nan}
    result_metadata = {
        'created_at': datetime.now().isoformat(),
        'model_type': model_type,
        'training_completed': True,
        'target_series': target_series
    }
    
    return {
        'status': 'completed',
        'model_name': final_model_name,
        'model_dir': str(model_dir),
        'metrics': metrics,
        'result': result,
        'metadata': result_metadata
    }
