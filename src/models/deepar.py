"""DeepAR model training and forecasting.

Uses sktime's DeepAR forecaster (if available) or pytorch-forecasting DeepAR.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from sktime.forecasting.compose import ForecastingPipeline
from sktime.transformations.series.impute import Imputer
from sktime.forecasting.naive import NaiveForecaster
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.utils import ValidationError, SCALER_ROBUST, SCALER_STANDARD
# Data preprocessing handled in train_sktime.py
from .base import save_model_checkpoint, load_model_checkpoint

logger = logging.getLogger(__name__)


def train_deepar(
    y_train: pd.Series,
    target_series: str,
    config: Dict[str, Any],
    checkpoint_dir: Path,
    X_train: Optional[pd.DataFrame] = None,
    y_train_original: Optional[pd.Series] = None
) -> Tuple[Any, Dict[str, Any]]:
    """Train DeepAR model.
    
    Parameters
    ----------
    y_train : pd.Series
        Preprocessed training data (already transformed, resampled, and imputed)
    target_series : str
        Target series name
    config : Dict[str, Any]
        Model configuration (hidden_size, epochs, etc.)
    checkpoint_dir : Path
        Directory to save checkpoint
        
    Returns
    -------
    Tuple[Any, Dict[str, Any]]
        (trained_model, metadata) tuple
    """
    # Try sktime's DeepAR first, fallback to pytorch-forecasting
    try:
        from sktime.forecasting.pytorchforecasting import PytorchForecastingDeepAR
        use_pytorch_forecasting = True
        
        # PyTorch 2.6 compatibility: Register pytorch-forecasting classes as safe globals
        try:
            import torch
            if hasattr(torch.serialization, 'add_safe_globals'):
                try:
                    from pytorch_forecasting.data.encoders import EncoderNormalizer
                    torch.serialization.add_safe_globals([EncoderNormalizer])
                except (ImportError, AttributeError):
                    pass
        except (ImportError, AttributeError):
            pass
    except ImportError:
        try:
            from sktime.forecasting.deep import DeepAR
            use_pytorch_forecasting = False
        except ImportError as e:
            raise ImportError(
                f"DeepAR forecaster is not available. "
                f"Install with: pip install 'sktime[all-extras]' or pip install pytorch-forecasting"
            ) from e
    
    # Data preprocessing is handled in train_sktime.py for unified preprocessing
    # y_train is already prepared (weekly data, transformed, imputed)
    
    # Get model parameters
    hidden_size = config.get('hidden_size', 40)
    num_layers = config.get('num_layers', 2)
    epochs = config.get('epochs', config.get('max_epochs', 100))
    batch_size = config.get('batch_size', 32)
    learning_rate = config.get('learning_rate', 0.001)
    
    # Get scaler type
    scaler_type = config.get('scaler', SCALER_ROBUST)
    
    # Create scaler transformer for pipeline
    scaler_transformer = None
    scaler = None
    if scaler_type and scaler_type != 'null':
        scaler_type_lower = scaler_type.lower()
        if scaler_type_lower == SCALER_ROBUST:
            sklearn_scaler = RobustScaler()
        elif scaler_type_lower == SCALER_STANDARD:
            sklearn_scaler = StandardScaler()
        else:
            logger.warning(f"Unknown scaler type '{scaler_type}', using RobustScaler")
            sklearn_scaler = RobustScaler()
        
        # Fit scaler to original data if available (for proper inverse transform)
        if y_train_original is not None and len(y_train_original) > 0:
            sklearn_scaler.fit(y_train_original.values.reshape(-1, 1))
            logger.info(f"Fitted {scaler_type} scaler to original data ({len(y_train_original)} observations)")
        else:
            # Fit to transformed data as fallback
            sklearn_scaler.fit(y_train.values.reshape(-1, 1))
            logger.info(f"Fitted {scaler_type} scaler to transformed data ({len(y_train)} observations)")
        
        scaler_transformer = TabularToSeriesAdaptor(sklearn_scaler)
        scaler = sklearn_scaler
        logger.info(f"Will apply {scaler_type} scaling in ForecastingPipeline")
    else:
        logger.info("No scaling applied (scaler='null')")
    
    logger.info(f"Training DeepAR on {target_series} ({len(y_train)} observations)")
    
    # Horizon is now in weeks in config (12 weeks - unified across all models)
    horizon = config.get('horizon', 12)  # Forecast horizon in weeks
    
    # Forecast horizon in weeks (12 weeks - unified across all models)
    # This must match max_prediction_length for pytorch-forecasting DeepAR
    forecast_horizon_weeks = 12
    
    # Validate parameters
    input_size = config.get('input_size', 96)
    # max_prediction_length must match forecast horizon for pytorch-forecasting DeepAR
    # The model can only predict up to max_prediction_length steps, so it must be set to forecast_horizon_weeks
    max_prediction_length_training = forecast_horizon_weeks  # Must match forecast horizon (12 weeks)
    total_required = input_size + max_prediction_length_training
    if len(y_train) < total_required:
        raise ValueError(
            f"DeepAR training failed: Insufficient data. "
            f"Need at least {total_required} observations (input_size={input_size} + max_prediction_length={max_prediction_length_training}), "
            f"but only have {len(y_train)} observations. "
            f"Reduce input_size or use more training data."
        )
    
    # Create base forecaster
    if use_pytorch_forecasting:
        # horizon is already in weeks (set above)
        
        # Model architecture parameters (for DeepAR model itself)
        model_params = {
            'hidden_size': hidden_size,
            'rnn_layers': num_layers,
            'dropout': config.get('dropout', 0.1),
            'learning_rate': learning_rate,
        }
        # Training parameters (for PyTorch Lightning trainer)
        # Use minimal logging to avoid hanging issues
        # Explicitly set GPU to ensure proper GPU utilization
        num_workers = config.get('num_workers', 4)  # Default to 4 for better GPU utilization
        trainer_params = {
            'max_epochs': epochs,
            'enable_progress_bar': True,
            'enable_model_summary': False,  # Disable to reduce overhead
            'logger': False,  # Disable Lightning logger to avoid hanging
            'enable_checkpointing': False,
            'log_every_n_steps': 10,  # Log every 10 steps
            'accelerator': 'gpu',  # Explicitly use GPU
            'devices': 1,  # Use single GPU (device 0)
        }
        # Note: max_prediction_length MUST match forecast horizon for pytorch-forecasting DeepAR
        # The model can only predict up to max_prediction_length steps, so it must be set to the desired forecast horizon
        # max_prediction_length_training is already set to forecast_horizon_weeks (12) above
        dataset_params = {
            'max_encoder_length': input_size,  # Default: 96 weeks = 2 years
            'max_prediction_length': max_prediction_length_training,  # Must match forecast horizon (12 weeks)
        }
        # DataLoader parameters
        # Set num_workers=4 to improve GPU utilization (parallel data loading)
        # Note: Increased from 0 to reduce CPU bottleneck and improve GPU utilization
        train_to_dataloader_params = {
            'train': True,
            'num_workers': num_workers,
        }
        validation_to_dataloader_params = {
            'train': False,
            'num_workers': num_workers,
        }
        
        base_forecaster = PytorchForecastingDeepAR(
            model_params=model_params,
            trainer_params=trainer_params,
            dataset_params=dataset_params,
            train_to_dataloader_params=train_to_dataloader_params,
            validation_to_dataloader_params=validation_to_dataloader_params
        )
    else:
        base_forecaster = DeepAR(
            hidden_size=hidden_size,
            num_layers=num_layers,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
    
    # Create pipeline with imputation and scaling
    pipeline_steps = [
        ('imputer_ffill', Imputer(method="ffill")),
        ('imputer_bfill', Imputer(method="bfill")),
        ('imputer_forecaster', Imputer(method="forecaster", forecaster=NaiveForecaster(strategy="last"))),
    ]
    
    # Add scaler to pipeline if specified
    if scaler_transformer is not None:
        pipeline_steps.append(('scaler', scaler_transformer))
    
    pipeline_steps.append(('forecaster', base_forecaster))
    
    forecaster = ForecastingPipeline(pipeline_steps)
    
    # Train
    # For PytorchForecastingDeepAR, set fh to forecast horizon (12 weeks)
    from sktime.forecasting.base import ForecastingHorizon
    # Set fh to forecast horizon (12 weeks)
    # Note: forecast_horizon_weeks is already defined above (12 weeks)
    training_fh = ForecastingHorizon(range(1, forecast_horizon_weeks + 1), is_relative=True)
    logger.info(f"Setting training fh to {forecast_horizon_weeks} weeks")
    
    # Prepare covariates if available
    X_train_aligned = None
    if use_pytorch_forecasting:
        if X_train is not None:
            # Remove duplicate indices before reindex (pytorch-forecasting can fail with duplicates)
            y_train_clean = y_train[~y_train.index.duplicated(keep='first')]
            X_train_clean = X_train[~X_train.index.duplicated(keep='first')]
            
            # Align covariates with target series index
            X_train_aligned = X_train_clean.reindex(y_train_clean.index, method='ffill').bfill()
            y_train = y_train_clean
            
            # Limit covariates if too many (pytorch-forecasting can hang with too many covariates)
            max_covariates = config.get('max_covariates', 50)  # Default: limit to 50 covariates
            if len(X_train_aligned.columns) > max_covariates:
                logger.warning(
                    f"Too many covariates ({len(X_train_aligned.columns)}). "
                    f"Limiting to {max_covariates} to avoid hanging issues. "
                    f"Using first {max_covariates} covariates."
                )
                X_train_aligned = X_train_aligned.iloc[:, :max_covariates]
            
            logger.info(f"Starting DeepAR training with {len(X_train_aligned.columns)} covariates (this may take several minutes for {len(y_train)} observations)...")
            logger.info(f"Data shapes - y_train: {y_train.shape}, X_train: {X_train_aligned.shape}")
            logger.info(f"Training parameters - epochs: {epochs}, batch_size: {batch_size}, num_workers: {num_workers}")
            logger.info("Preparing dataset (this may take 1-2 minutes with many covariates)...")
            import sys
            sys.stdout.flush()  # Force flush to see progress
            
            logger.info("Calling forecaster.fit()...")
            try:
                forecaster.fit(y_train, fh=training_fh, X=X_train_aligned)
                logger.info("forecaster.fit() completed successfully")
            except Exception as e:
                logger.error(f"forecaster.fit() failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise
        else:
            # Remove duplicate indices before fit (pytorch-forecasting can fail with duplicates)
            y_train_clean = y_train[~y_train.index.duplicated(keep='first')]
            y_train = y_train_clean
            
            logger.info(f"Starting DeepAR training (this may take several minutes for {len(y_train)} observations)...")
            logger.info(f"Data shapes - y_train: {y_train.shape}")
            logger.info(f"Training parameters - epochs: {epochs}, batch_size: {batch_size}, num_workers: {num_workers}")
            logger.info("Calling forecaster.fit()...")
            try:
                forecaster.fit(y_train, fh=training_fh)
                logger.info("forecaster.fit() completed successfully")
            except Exception as e:
                logger.error(f"forecaster.fit() failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise
    else:
        if X_train is not None:
            X_train_aligned = X_train.reindex(y_train.index, method='ffill').bfill()
            
            # Limit covariates if too many
            max_covariates = config.get('max_covariates', 50)
            if len(X_train_aligned.columns) > max_covariates:
                logger.warning(
                    f"Too many covariates ({len(X_train_aligned.columns)}). "
                    f"Limiting to {max_covariates} to avoid hanging issues."
                )
                X_train_aligned = X_train_aligned.iloc[:, :max_covariates]
            
            logger.info(f"Training DeepAR with {len(X_train_aligned.columns)} covariates")
            forecaster.fit(y_train, X=X_train_aligned)
        else:
            forecaster.fit(y_train)
    
    logger.info("DeepAR training completed")
    
    # Save checkpoint using sktime's built-in save method
    # ForecastingPipeline inherits from BaseForecaster and has save() method
    checkpoint_path = checkpoint_dir / "model.zip"  # sktime saves as .zip
    metadata_path = checkpoint_dir / "metadata.pkl"  # Save metadata separately
    
    # Ensure checkpoint directory exists
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if model.zip or model.pkl is a directory (from previous failed save) and remove it
    for path_to_check in [checkpoint_path, checkpoint_dir / "model.pkl"]:
        if path_to_check.exists() and path_to_check.is_dir():
            logger.warning(f"Removing existing directory at {path_to_check}")
            import shutil
            shutil.rmtree(path_to_check)
    
    # Prepare metadata
    metadata = {
        'model_type': 'deepar',
        'target_series': target_series,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'scaler': scaler,
        'scaler_type': scaler_type if scaler else None,
        'training_data_shape': y_train.shape,
        'training_data_index': list(y_train.index),
        'covariates_used': list(X_train_aligned.columns) if X_train_aligned is not None else None,
        'num_covariates': len(X_train_aligned.columns) if X_train_aligned is not None else 0,
        'training_fh_length': forecast_horizon_weeks  # Store actual training fh length (12 weeks)
    }
    
    # Use sktime's save method (handles pytorch-forecasting models better)
    try:
        # Try cloudpickle format first (better for pytorch-forecasting)
        forecaster.save(path=str(checkpoint_path), serialization_format='cloudpickle')
        logger.info(f"Saved forecaster using sktime.save() to {checkpoint_path}")
    except Exception as e:
        # Fallback to pickle
        try:
            forecaster.save(path=str(checkpoint_path), serialization_format='pickle')
            logger.info(f"Saved forecaster using sktime.save() (pickle) to {checkpoint_path}")
        except Exception as e2:
            logger.warning(f"sktime.save() failed: {e2}, falling back to custom save")
            # Fallback to custom save method
            # Ensure directory exists for custom save
            checkpoint_path_pkl = checkpoint_dir / "model.pkl"
            checkpoint_path_pkl.parent.mkdir(parents=True, exist_ok=True)
            save_model_checkpoint(forecaster, checkpoint_path_pkl, metadata)
            checkpoint_path = checkpoint_path_pkl
    
    # Save metadata separately
    try:
        import pickle
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Saved metadata to {metadata_path}")
    except Exception as e:
        logger.warning(f"Failed to save metadata separately: {e}")
    
    return forecaster, metadata


def forecast_deepar(
    model: Any,
    horizon: int,
    last_date: Optional[pd.Timestamp] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Generate DeepAR forecasts in weekly frequency using direct long-horizon forecasting.
    
    Parameters
    ----------
    model : Any
        Trained DeepAR forecaster (trained on weekly data)
    horizon : int
        Forecast horizon in weeks (12 weeks - unified across all models)
    last_date : Optional[pd.Timestamp]
        Last date in training data
        
    Returns
    -------
    pd.DataFrame
        Forecasted values with DatetimeIndex (weekly frequency)
    """
    from sktime.forecasting.base import ForecastingHorizon
    
    logger = logging.getLogger(__name__)
    
    # Horizon is already in weeks
    horizon_weeks = horizon
    
    # Get last date for index creation (needed before cutoff fix)
    if last_date is None and metadata and 'training_data_index' in metadata:
        try:
            last_date = pd.to_datetime(metadata['training_data_index'][-1])
            logger.info(f"Using last training date from metadata: {last_date}")
        except Exception as e:
            logger.debug(f"Could not parse last_date from metadata: {e}")
    
    # CRITICAL: Fix pipeline cutoff frequency BEFORE creating ForecastingHorizon
    # For models trained on weekly data, cutoff might have W-FRI or other weekday
    # We need to standardize to W-SUN to match ForecastingHorizon expectations
    # Fix cutoff first, then create ForecastingHorizon
    
    # Get cutoff date for adjustment
    cutoff_date = None
    cutoff_freq_original = None
    if hasattr(model, 'cutoff') and model.cutoff is not None and len(model.cutoff) > 0:
        cutoff_date = model.cutoff[0]
        if hasattr(model.cutoff, 'freq') and model.cutoff.freq is not None:
            cutoff_freq_original = model.cutoff.freq
    elif last_date is not None:
        cutoff_date = last_date
    else:
        cutoff_date = pd.Timestamp('2019-12-31')
    
    # Adjust to nearest Sunday (standardize to W-SUN)
    if cutoff_date is not None:
        days_since_sunday = cutoff_date.weekday()
        if days_since_sunday == 6:
            sunday_date = cutoff_date
        else:
            sunday_date = cutoff_date + pd.Timedelta(days=6-days_since_sunday)
    else:
        sunday_date = pd.Timestamp('2019-12-29')  # Fallback to a Sunday
    
    # CRITICAL: Replace pipeline cutoff with W-SUN to match ForecastingHorizon
    # This ensures no frequency mismatch during prediction
    try:
        cutoff_weekly = pd.DatetimeIndex([sunday_date], freq='W-SUN')
        # Directly replace the cutoff (this is what TFT does)
        model._cutoff = cutoff_weekly
        logger.info(f"Replaced pipeline cutoff with W-SUN: {cutoff_weekly} (original was {cutoff_freq_original})")
    except Exception as e:
        logger.warning(f"Failed to replace pipeline cutoff: {e}")
    
    # Also update forecaster step cutoff (CRITICAL: this is what sktime actually uses)
    try:
        if hasattr(model, 'steps'):
            # Update ALL forecaster steps' cutoff to W-SUN
            for name, step in model.steps:
                if 'forecaster' in name.lower():
                    # Set both _cutoff and cutoff if they exist
                    new_cutoff = pd.DatetimeIndex([sunday_date], freq='W-SUN')
                    if hasattr(step, '_cutoff'):
                        step._cutoff = new_cutoff
                        logger.info(f"Updated forecaster step '{name}' _cutoff to 'W-SUN': {new_cutoff}")
                    # Also try to set cutoff property if it exists
                    if hasattr(step, 'cutoff'):
                        try:
                            step.cutoff = new_cutoff
                            logger.info(f"Updated forecaster step '{name}' cutoff property to 'W-SUN': {new_cutoff}")
                        except:
                            # If setting cutoff property fails, try _cutoff
                            if hasattr(step, '_cutoff'):
                                step._cutoff = new_cutoff
        # Also check forecaster_ if it exists
        if hasattr(model, 'forecaster_') and hasattr(model.forecaster_, '_cutoff'):
            new_cutoff = pd.DatetimeIndex([sunday_date], freq='W-SUN')
            model.forecaster_._cutoff = new_cutoff
            logger.info(f"Updated model.forecaster_._cutoff to 'W-SUN': {new_cutoff}")
    except Exception as e:
        logger.warning(f"Failed to update forecaster cutoff with freq: {e}")
    
    # NOW create ForecastingHorizon (after cutoff is fixed to W-SUN)
    # Get training fh length from metadata (used during fit)
    training_fh_length = metadata.get('training_fh_length', 12) if metadata else 12
    
    logger.info(f"Training fh length: {training_fh_length} weeks, Target horizon: {horizon_weeks} weeks")
    
    # Create ForecastingHorizon WITHOUT freq first, then set freq after cutoff is fixed
    # This prevents sktime from trying to inherit freq from original cutoff
    training_fh = ForecastingHorizon(range(1, training_fh_length + 1), is_relative=True)
    
    # Now set freq to W-SUN to match the standardized cutoff
    # This must be done AFTER cutoff is fixed to W-SUN
    fh_freq = 'W-SUN'
    training_fh.freq = fh_freq
    logger.info(f"Created ForecastingHorizon and set freq={fh_freq} to match standardized cutoff")
    
    # For PytorchForecastingDeepAR, try direct prediction first using training fh
    # The model was trained with training_fh_length, so we should use that fh
    fh = training_fh if training_fh_length == horizon_weeks else training_fh
    
    # Check if model was trained with covariates - if so, we need to provide X from the start
    num_covariates = metadata.get('num_covariates', 0) if metadata else 0
    covariates_used = metadata.get('covariates_used', []) if metadata else []
    
    # Create dummy X if covariates were used during training
    X_dummy = None
    if num_covariates > 0:
        if last_date is not None:
            forecast_index = pd.date_range(
                start=last_date + pd.Timedelta(weeks=1),
                periods=horizon_weeks,
                freq='W'
            )
        else:
            forecast_index = pd.RangeIndex(start=0, stop=horizon_weeks)
        
        if covariates_used and len(covariates_used) == num_covariates:
            X_dummy = pd.DataFrame(
                {col: [0.0] * horizon_weeks for col in covariates_used},
                index=forecast_index
            )
        else:
            X_dummy = pd.DataFrame(
                {f'__dummy_{i}__': [0.0] * horizon_weeks for i in range(num_covariates)},
                index=forecast_index
            )
        logger.info(f"Model was trained with {num_covariates} covariates, using dummy X for prediction")
    
    # Final check: ensure cutoff is W-SUN before prediction
    # This must be done right before predict() call as sktime may read cutoff internally
    try:
        if hasattr(model, 'cutoff') and model.cutoff is not None:
            if hasattr(model.cutoff, 'freq') and model.cutoff.freq is not None:
                if str(model.cutoff.freq) != 'W-SUN' and 'W-SUN' not in str(model.cutoff.freq):
                    # Force cutoff to W-SUN
                    cutoff_weekly = pd.DatetimeIndex([sunday_date], freq='W-SUN')
                    model._cutoff = cutoff_weekly
                    logger.info(f"Final cutoff fix: set to W-SUN before predict()")
    except Exception as e:
        logger.warning(f"Failed final cutoff fix: {e}")
    
    try:
        logger.info(f"Attempting direct prediction for {horizon_weeks} weeks (using training fh length: {training_fh_length})...")
        # Use X_dummy if available (for models trained with covariates), otherwise try without X
        try:
            if X_dummy is not None:
                forecast = model.predict(fh=fh, X=X_dummy)
            else:
                forecast = model.predict(fh=fh)
            forecast_length = len(forecast) if hasattr(forecast, '__len__') else 0
            if forecast_length == 0:
                logger.warning(f"Model returned empty forecast (length=0), this may indicate a model issue")
                # Try to extract values to check if it's really empty
                if isinstance(forecast, pd.Series):
                    if len(forecast) == 0:
                        raise ValueError("Model returned empty Series forecast")
                elif isinstance(forecast, pd.DataFrame):
                    if len(forecast) == 0:
                        raise ValueError("Model returned empty DataFrame forecast")
                else:
                    raise ValueError(f"Model returned empty forecast of type {type(forecast)}")
            logger.info(f"Successfully generated {forecast_length} weeks of forecasts using direct prediction")
        except (AttributeError, TypeError, ValueError) as x_error:
            # If X parameter issue, try with dummy DataFrame matching training structure
            error_msg = str(x_error).lower()
            if "'nonetype' object has no attribute 'copy'" in error_msg or "nonetype" in error_msg or "0 feature" in error_msg or "robustscaler" in error_msg or "expecting" in error_msg:
                # Create dummy DataFrame matching the structure used during training
                # Get number of covariates from metadata
                num_covariates = metadata.get('num_covariates', 0) if metadata else 0
                if last_date is not None:
                    forecast_index = pd.date_range(
                        start=last_date + pd.Timedelta(weeks=1),
                        periods=horizon_weeks,
                        freq='W'
                    )
                else:
                    forecast_index = pd.RangeIndex(start=0, stop=horizon_weeks)
                
                if num_covariates > 0:
                    # Model was trained with covariates, create dummy DataFrame with matching number of columns
                    # Use column names from training if available
                    covariates_used = metadata.get('covariates_used', []) if metadata else []
                    if covariates_used and len(covariates_used) == num_covariates:
                        # Use same column names as training
                        X_dummy = pd.DataFrame(
                            {col: [0.0] * horizon_weeks for col in covariates_used},
                            index=forecast_index
                        )
                    else:
                        # Create dummy columns
                        X_dummy = pd.DataFrame(
                            {f'__dummy_{i}__': [0.0] * horizon_weeks for i in range(num_covariates)},
                            index=forecast_index
                        )
                    logger.debug(f"Created dummy X with {num_covariates} columns matching training structure")
                else:
                    # Model was trained without covariates, but pipeline still needs X
                    # Create single dummy column
                    X_dummy = pd.DataFrame({'__dummy__': [0.0] * horizon_weeks}, index=forecast_index)
                    logger.debug("Created dummy X with single column (model trained without covariates)")
                
                try:
                    # Use training fh (which matches what the model was trained with)
                    forecast = model.predict(fh=training_fh, X=X_dummy)
                    
                    # Extract forecast values properly
                    if isinstance(forecast, pd.Series):
                        forecast_values = forecast.values
                        forecast_length = len(forecast_values)
                    elif isinstance(forecast, pd.DataFrame):
                        forecast_values = forecast.iloc[:, 0].values if len(forecast.columns) > 0 else forecast.values.flatten()
                        forecast_length = len(forecast_values)
                    else:
                        forecast_values = np.array(forecast).flatten()
                        forecast_length = len(forecast_values)
                    
                    # Check if forecast is empty
                    if forecast_length == 0:
                        logger.warning(f"Model returned empty forecast, trying with different fh...")
                        # Try with exact horizon fh (with explicit weekly freq matching cutoff)
                        fh_exact = ForecastingHorizon(range(1, horizon_weeks + 1), is_relative=True, freq=fh_freq)
                        forecast = model.predict(fh=fh_exact, X=X_dummy)
                        if isinstance(forecast, pd.Series):
                            forecast_values = forecast.values
                            forecast_length = len(forecast_values)
                        elif isinstance(forecast, pd.DataFrame):
                            forecast_values = forecast.iloc[:, 0].values if len(forecast.columns) > 0 else forecast.values.flatten()
                            forecast_length = len(forecast_values)
                        else:
                            forecast_values = np.array(forecast).flatten()
                            forecast_length = len(forecast_values)
                    
                    # Reconstruct forecast Series/DataFrame if needed
                    if forecast_length > 0:
                        if isinstance(forecast, pd.Series):
                            forecast = forecast.iloc[:horizon_weeks] if forecast_length > horizon_weeks else forecast
                        elif isinstance(forecast, pd.DataFrame):
                            forecast = forecast.iloc[:horizon_weeks] if forecast_length > horizon_weeks else forecast
                        else:
                            # Create Series from values
                            if last_date is not None:
                                forecast_index = pd.date_range(
                                    start=last_date + pd.Timedelta(weeks=1),
                                    periods=min(forecast_length, horizon_weeks),
                                    freq='W'
                                )
                            else:
                                forecast_index = pd.RangeIndex(start=0, stop=min(forecast_length, horizon_weeks))
                            forecast = pd.Series(forecast_values[:horizon_weeks], index=forecast_index[:horizon_weeks])
                    
                    # Final check: ensure forecast has proper length
                    if forecast_length == 0:
                        # If still empty, try one more time with exact fh matching horizon
                        logger.warning(f"Forecast is still empty after retry, attempting final prediction with exact fh={horizon_weeks}...")
                        fh_final = ForecastingHorizon(range(1, horizon_weeks + 1), is_relative=True, freq=fh_freq)
                        forecast_final = model.predict(fh=fh_final, X=X_dummy)
                        if isinstance(forecast_final, pd.Series):
                            forecast = forecast_final
                            forecast_length = len(forecast)
                        elif isinstance(forecast_final, pd.DataFrame):
                            forecast = forecast_final
                            forecast_length = len(forecast)
                        else:
                            forecast_values_final = np.array(forecast_final).flatten()
                            forecast_length = len(forecast_values_final)
                            if forecast_length > 0:
                                if last_date is not None:
                                    forecast_index_final = pd.date_range(
                                        start=last_date + pd.Timedelta(weeks=1),
                                        periods=forecast_length,
                                        freq='W'
                                    )
                                else:
                                    forecast_index_final = pd.RangeIndex(start=0, stop=forecast_length)
                                forecast = pd.Series(forecast_values_final, index=forecast_index_final)
                    
                    # Final check: ensure forecast has proper length
                    forecast_length = len(forecast) if hasattr(forecast, '__len__') else 0
                    
                    # If forecast is still empty after all retries, raise error
                    if forecast_length == 0:
                        raise ValueError(f"DeepAR prediction failed: model returned empty forecast after all retries. This may indicate a model compatibility issue.")
                    
                    logger.info(f"Successfully generated {forecast_length} weeks of forecasts using direct prediction (with dummy X, fh length: {training_fh_length})")
                    
                    # Ensure forecast has correct length (trim or pad if needed)
                    if forecast_length != horizon_weeks:
                        logger.warning(f"Forecast length ({forecast_length}) doesn't match requested horizon ({horizon_weeks}), adjusting...")
                        if isinstance(forecast, pd.Series):
                            if forecast_length > horizon_weeks:
                                forecast = forecast.iloc[:horizon_weeks]
                            elif forecast_length < horizon_weeks:
                                # Pad with last value
                                last_value = forecast.iloc[-1] if forecast_length > 0 else 0.0
                                padding = pd.Series([last_value] * (horizon_weeks - forecast_length))
                                if last_date is not None:
                                    padding_index = pd.date_range(
                                        start=forecast.index[-1] + pd.Timedelta(weeks=1),
                                        periods=horizon_weeks - forecast_length,
                                        freq='W'
                                    )
                                    padding.index = padding_index
                                forecast = pd.concat([forecast, padding])
                        elif isinstance(forecast, pd.DataFrame):
                            if forecast_length > horizon_weeks:
                                forecast = forecast.iloc[:horizon_weeks]
                            elif forecast_length < horizon_weeks:
                                # Pad with last row
                                last_row = forecast.iloc[-1:] if forecast_length > 0 else pd.DataFrame({col: [0.0] for col in forecast.columns})
                                padding = pd.concat([last_row] * (horizon_weeks - forecast_length), ignore_index=True)
                                if last_date is not None:
                                    padding_index = pd.date_range(
                                        start=forecast.index[-1] + pd.Timedelta(weeks=1),
                                        periods=horizon_weeks - forecast_length,
                                        freq='W'
                                    )
                                    padding.index = padding_index
                                forecast = pd.concat([forecast, padding])
                    
                    # Update forecast_length after adjustment
                    forecast_length = len(forecast) if hasattr(forecast, '__len__') else horizon_weeks
                    logger.info(f"Final forecast length: {forecast_length} weeks")
                except Exception as e2:
                    error_msg2 = str(e2).lower()
                    if "forecast length mismatch" in error_msg2:
                        # This is expected, will be handled by recursive prediction
                        raise
                    logger.warning(f"Failed to predict with dummy X: {e2}")
                    # If dummy X also fails, this might be a different issue
                    raise x_error
            else:
                raise
    except ValueError as e:
        error_msg = str(e).lower()
        if "different forecasting horizon" in error_msg or "forecast length mismatch" in error_msg:
            # Model was not trained with 12 weeks, or model only generates partial forecasts
            # Need recursive prediction
            logger.warning(f"Direct prediction failed or incomplete: {e}")
            
            # Detect actual prediction length from error message or use default
            # If forecast_length was 0, use training_fh_length instead
            actual_prediction_length = training_fh_length
            if "forecast length mismatch" in error_msg:
                # Extract the actual length from error message
                import re
                match = re.search(r'got (\d+)', error_msg)
                if match:
                    detected_length = int(match.group(1))
                    # Only use detected length if it's greater than 0
                    if detected_length > 0:
                        actual_prediction_length = detected_length
                        logger.info(f"Detected actual prediction length: {actual_prediction_length} weeks")
                    else:
                        logger.warning(f"Detected prediction length is 0, using training_fh_length: {training_fh_length} weeks")
                        actual_prediction_length = training_fh_length
                else:
                    logger.warning(f"Could not extract prediction length from error, using training_fh_length: {training_fh_length} weeks")
                    actual_prediction_length = training_fh_length
            
            # Ensure actual_prediction_length is at least 1
            if actual_prediction_length <= 0:
                actual_prediction_length = training_fh_length
                logger.warning(f"actual_prediction_length is {actual_prediction_length}, forcing to training_fh_length: {training_fh_length}")
            
            logger.info(f"Falling back to recursive prediction using prediction length: {actual_prediction_length} weeks")
            
            # Prepare dummy X structure for recursive prediction
            num_covariates = metadata.get('num_covariates', 0) if metadata else 0
            covariates_used = metadata.get('covariates_used', []) if metadata else []
            
            # Use recursive prediction
            if horizon_weeks > actual_prediction_length:
                logger.info(f"Using recursive prediction: {actual_prediction_length} weeks per step to reach {horizon_weeks} weeks")
                
                # Use training fh for recursive prediction steps
                step_fh = training_fh
                
                # Collect all forecasts
                all_forecasts = []
                remaining_horizon = horizon_weeks
                current_model = model
                current_last_date = last_date
                
                iteration = 0
                while remaining_horizon > 0:
                    iteration += 1
                    # Determine how many steps to predict in this iteration
                    steps_this_iteration = min(actual_prediction_length, remaining_horizon)
                    
                    logger.info(f"Iteration {iteration}: Predicting {steps_this_iteration} weeks (remaining: {remaining_horizon} weeks)...")
                    
                    try:
                        # Use training fh for prediction (which matches what the model was trained with)
                        iter_fh = training_fh
                        
                        # Predict using the actual prediction length
                        # For ForecastingPipeline with PytorchForecastingDeepAR, we need to handle X carefully
                        # Create dummy X matching training structure for each iteration
                        if current_last_date is not None:
                            iter_forecast_index = pd.date_range(
                                start=current_last_date + pd.Timedelta(weeks=1),
                                periods=steps_this_iteration,
                                freq='W'
                            )
                        else:
                            iter_forecast_index = pd.RangeIndex(start=0, stop=steps_this_iteration)
                        
                        # Create dummy X with same structure as training (reuse variables from outer scope)
                        if num_covariates > 0:
                            if covariates_used and len(covariates_used) == num_covariates:
                                X_dummy_iter = pd.DataFrame(
                                    {col: [0.0] * steps_this_iteration for col in covariates_used},
                                    index=iter_forecast_index
                                )
                            else:
                                X_dummy_iter = pd.DataFrame(
                                    {f'__dummy_{i}__': [0.0] * steps_this_iteration for i in range(num_covariates)},
                                    index=iter_forecast_index
                                )
                        else:
                            X_dummy_iter = pd.DataFrame({'__dummy__': [0.0] * steps_this_iteration}, index=iter_forecast_index)
                        
                        try:
                            # Use training fh for each step (which matches what the model was trained with)
                            forecast_step = current_model.predict(fh=training_fh, X=X_dummy_iter)
                        except (AttributeError, TypeError, ValueError) as e:
                            error_msg = str(e).lower()
                            if "'nonetype' object has no attribute 'copy'" in error_msg or "nonetype" in error_msg:
                                # Pipeline transformer expects X but we don't have it
                                # The issue is that ForecastingPipeline's transformers were fit with specific X columns
                                # We need to provide X with the same structure (columns) as during training
                                logger.warning("Pipeline requires X parameter, but model was trained without covariates.")
                                logger.warning("This is a known limitation of ForecastingPipeline with PytorchForecastingDeepAR.")
                                logger.warning("Recursive prediction may not work correctly. Consider retraining with fh=12.")
                                raise ValueError(
                                    f"DeepAR recursive prediction failed: ForecastingPipeline requires X parameter "
                                    f"but the model was trained without covariates. "
                                    f"Please retrain the model with the desired forecast horizon (12 weeks) "
                                    f"or modify the pipeline to handle X=None. Original error: {e}"
                                )
                            elif "0 feature(s)" in error_msg or "robustscaler" in error_msg:
                                # Empty DataFrame caused scaler error - this shouldn't happen if we bypass transformers
                                logger.warning("Scaler error, this should not occur with direct forecaster access")
                                raise
                            else:
                                raise
                        
                        # Extract forecast values
                        if isinstance(forecast_step, pd.Series):
                            forecast_values = forecast_step.values
                        elif isinstance(forecast_step, pd.DataFrame):
                            forecast_values = forecast_step.iloc[:, 0].values
                        else:
                            forecast_values = np.array(forecast_step).flatten()
                        
                        # Take only the number of steps we need (model might generate more than requested)
                        forecast_values = forecast_values[:steps_this_iteration]
                        if len(forecast_values) > 0:
                            all_forecasts.append(forecast_values)
                        else:
                            logger.warning(f"  No forecast values generated in iteration {iteration}")
                            break
                        
                        logger.info(f"  Generated {len(forecast_values)} forecast values")
                        logger.debug(f"  Forecast values sample: {forecast_values[:3] if len(forecast_values) >= 3 else forecast_values}")
                        
                        # Update remaining horizon
                        remaining_horizon -= steps_this_iteration
                        logger.debug(f"  Updated remaining horizon: {remaining_horizon}")
                        
                        # Update current_last_date for next iteration (for index calculation)
                        if current_last_date is not None:
                            current_last_date = current_last_date + pd.Timedelta(weeks=steps_this_iteration)
                        
                        # Note: We don't update the model with predicted values because:
                        # 1. PytorchForecastingDeepAR's update() may not work correctly with ForecastingPipeline
                        # 2. Each prediction step uses the original trained model state
                        # 3. This is a simpler approach, though less optimal than true recursive forecasting
                        # The model will use its internal state for each prediction step
                        
                    except ValueError as e:
                        error_msg = str(e).lower()
                        if "different forecasting horizon" in error_msg:
                            # If fh mismatch, try with the exact prediction length
                            logger.warning(f"  FH mismatch, trying with exact prediction length: {actual_prediction_length}")
                            try:
                                # Create X_dummy for step_fh (uses actual_prediction_length)
                                if current_last_date is not None:
                                    step_forecast_index = pd.date_range(
                                        start=current_last_date + pd.Timedelta(weeks=1),
                                        periods=actual_prediction_length,
                                        freq='W'
                                    )
                                else:
                                    step_forecast_index = pd.RangeIndex(start=0, stop=actual_prediction_length)
                                
                                if num_covariates > 0:
                                    if covariates_used and len(covariates_used) == num_covariates:
                                        X_dummy_step = pd.DataFrame(
                                            {col: [0.0] * actual_prediction_length for col in covariates_used},
                                            index=step_forecast_index
                                        )
                                    else:
                                        X_dummy_step = pd.DataFrame(
                                            {f'__dummy_{i}__': [0.0] * actual_prediction_length for i in range(num_covariates)},
                                            index=step_forecast_index
                                        )
                                else:
                                    X_dummy_step = pd.DataFrame({'__dummy__': [0.0] * actual_prediction_length}, index=step_forecast_index)
                                
                                # Use model.predict with step_fh
                                forecast_step = current_model.predict(fh=step_fh, X=X_dummy_step)
                                if isinstance(forecast_step, pd.Series):
                                    forecast_values = forecast_step.values
                                elif isinstance(forecast_step, pd.DataFrame):
                                    forecast_values = forecast_step.iloc[:, 0].values
                                else:
                                    forecast_values = np.array(forecast_step).flatten()
                                
                                # Take only what we need
                                forecast_values = forecast_values[:steps_this_iteration]
                                all_forecasts.append(forecast_values)
                                remaining_horizon -= steps_this_iteration
                            except Exception as e2:
                                logger.error(f"  Failed to predict with training fh: {e2}")
                                raise
                        else:
                            raise
                
                # Combine all forecasts (after while loop)
                if len(all_forecasts) > 0:
                    forecast_values = np.concatenate(all_forecasts)[:horizon_weeks]  # Ensure exact length
                    
                    # Create forecast Series/DataFrame with proper index
                    if last_date is not None:
                        forecast_index = pd.date_range(
                            start=last_date + pd.Timedelta(weeks=1),
                            periods=len(forecast_values),
                            freq='W'
                        )
                    else:
                        forecast_index = pd.RangeIndex(start=0, stop=len(forecast_values))
                    
                    forecast = pd.Series(forecast_values, index=forecast_index, name='forecast')
                    logger.info(f"Generated {len(forecast)} weeks of forecasts using {iteration} recursive prediction steps")
                else:
                    raise ValueError("No forecasts generated during recursive prediction")
            else:
                # If horizon <= training_fh_length, use direct prediction with training fh (with explicit weekly freq matching cutoff)
                step_fh = ForecastingHorizon(range(1, training_fh_length + 1), is_relative=True, freq=fh_freq)
                logger.info(f"Using direct prediction with training fh length: {training_fh_length} weeks...")
                try:
                    forecast = model.predict(fh=step_fh)
                    # Take only what we need
                    if isinstance(forecast, pd.Series):
                        forecast = forecast.iloc[:horizon_weeks]
                    elif isinstance(forecast, pd.DataFrame):
                        forecast = forecast.iloc[:horizon_weeks]
                except ValueError as e2:
                    logger.error(f"Failed to predict even with training fh: {e2}")
                    raise
        else:
            # Other error, re-raise
            raise
    
    # Create weekly index using pandas date_range
    # Only update index if forecast length matches expected horizon
    if last_date is not None:
        # Use actual forecast length, not horizon_weeks (model might return different length)
        forecast_length = len(forecast) if hasattr(forecast, '__len__') else horizon_weeks
        forecast_index = pd.date_range(
            start=last_date + pd.Timedelta(weeks=1),
            periods=forecast_length,
            freq='W'
        )
        if isinstance(forecast, pd.Series):
            if len(forecast) == len(forecast_index):
                forecast.index = forecast_index
            else:
                logger.warning(f"Forecast length ({len(forecast)}) doesn't match index length ({len(forecast_index)}), using existing index")
        elif isinstance(forecast, pd.DataFrame):
            if len(forecast) == len(forecast_index):
                forecast.index = forecast_index
            else:
                logger.warning(f"Forecast length ({len(forecast)}) doesn't match index length ({len(forecast_index)}), using existing index")
    
    # Apply inverse transform if scaler is available in metadata
    # ForecastingPipeline should handle this, but we ensure it's done explicitly
    if metadata and 'scaler' in metadata and metadata['scaler'] is not None:
        scaler = metadata['scaler']
        try:
            # Inverse transform the forecast
            if isinstance(forecast, pd.Series):
                forecast_values = forecast.values.reshape(-1, 1)
                forecast_inverse = scaler.inverse_transform(forecast_values)
                forecast = pd.Series(
                    forecast_inverse.flatten(),
                    index=forecast.index,
                    name=forecast.name
                )
                logger.info("Applied inverse transform to return to original scale")
            elif isinstance(forecast, pd.DataFrame):
                forecast_values = forecast.values
                forecast_inverse = scaler.inverse_transform(forecast_values)
                forecast = pd.DataFrame(
                    forecast_inverse,
                    index=forecast.index,
                    columns=forecast.columns
                )
                logger.info("Applied inverse transform to return to original scale")
        except Exception as e:
            logger.warning(f"Failed to apply inverse transform: {e}")
    
    return forecast if isinstance(forecast, pd.DataFrame) else forecast.to_frame()
