"""Training functions for sktime forecasting models."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)


def train_sktime_model(
    model_type: str,
    config_name: str,
    cfg: Any,
    data: pd.DataFrame,
    model_name: str,
    horizons: Optional[int] = None,
    outputs_dir: Path = None,
    model_params: Optional[Dict[str, Any]] = None,
    data_loader: Optional[Any] = None,
    metadata: Optional[pd.DataFrame] = None
) -> None:
    """Train sktime forecasting model.
    
    Parameters
    ----------
    model_type : str
        Model type: 'itf', 'itransformer', 'patchtst', 'tft', 'timemixer'
    config_name : str
        Config name (unused, kept for compatibility)
    cfg : Any
        Hydra config object
    data : pd.DataFrame
        Preprocessed training data (standardized, without date columns)
    model_name : str
        Model name for saving
    horizons : int, optional
        Forecast horizons (unused for training)
    outputs_dir : Path
        Directory to save trained model
    model_params : dict, optional
        Model parameters dictionary
    """
    try:
        from sktime.forecasting.base import ForecastingHorizon
        import joblib
        import pickle
    except ImportError as e:
        raise ImportError(f"sktime not available: {e}")
    
    logger.info(f"Training {model_type.upper()} model...")
    logger.info(f"Training data shape: {data.shape}")
    
    # Get target series from config, or determine from metadata/config
    target_series = model_params.get('target_series') if model_params else None
    
    # Filter to available target series
    if target_series is not None and len(target_series) > 0:
        # Filter to available target series
        available_targets = [t for t in target_series if t in data.columns]
        if not available_targets:
            logger.warning(f"No specified target series found in data: {target_series}. Using all columns as fallback.")
            available_targets = list(data.columns)
        else:
            # Log if some targets were missing
            missing = [t for t in target_series if t not in data.columns]
            if missing:
                logger.warning(f"Some target series not found in data: {missing}. Using available ones: {available_targets}")
    else:
        # No target series specified - use all columns (consistent with DFM behavior)
        available_targets = list(data.columns)
        logger.info("No target_series specified in config. Using all available columns.")
    
    if not available_targets:
        raise ValueError("No target series available for training. Data has no columns.")
    
    logger.info(f"Target series ({len(available_targets)}): {available_targets[:5]}{'...' if len(available_targets) > 5 else ''}")
    
    # Select target series for training
    target_data = data[available_targets].copy()
    
    # Get model-specific parameters
    model_params = model_params or {}
    
    # Create model based on type
    model_type_lower = model_type.lower()
    
    # Convert data to neuralforecast format if needed
    # neuralforecast expects: unique_id, ds (datetime), y (target) format
    use_neuralforecast = model_type_lower in ['itransformer', 'itf', 'tft', 'timemixer', 'patchtst']
    
    if use_neuralforecast:
        # Convert pandas DataFrame to neuralforecast format
        # Create long format: each series becomes rows with unique_id, ds, y
        nf_data_list = []
        
        # Ensure we have a datetime index
        # Preserve original index from data_loader if available (has proper datetime from date_w)
        if not isinstance(data.index, pd.DatetimeIndex):
            if data_loader is not None and hasattr(data_loader, 'processed'):
                # Use index from processed data which has proper DatetimeIndex from date_w
                processed_data = data_loader.processed
                if isinstance(processed_data.index, pd.DatetimeIndex):
                    # Align with training_data (which drops date columns)
                    data = data.copy()
                    data.index = processed_data.index[:len(data)]
                    logger.info("Using datetime index from data_loader.processed")
                else:
                    # Fallback: create weekly datetime index
                    logger.warning("data_loader.processed doesn't have DatetimeIndex. Creating synthetic weekly index.")
                    dates = pd.date_range('2020-01-01', periods=len(data), freq='W')
                    data = data.copy()
                    data.index = dates
            else:
                # Fallback: create weekly datetime index
                logger.warning("No data_loader provided and data doesn't have DatetimeIndex. Creating synthetic weekly index.")
                dates = pd.date_range('2020-01-01', periods=len(data), freq='W')
                data = data.copy()
                data.index = dates
        
        for col in available_targets:
            series_data = pd.DataFrame({
                'unique_id': col,
                'ds': data.index,
                'y': data[col].values
            })
            nf_data_list.append(series_data)
        
        nf_df = pd.concat(nf_data_list, ignore_index=True)
        
        # Determine forecast horizon
        h = model_params.get('prediction_length', model_params.get('n_forecasts', 24))
        input_size = model_params.get('context_length', model_params.get('n_lags', 96))
        
        if model_type_lower in ['itf', 'itransformer']:
            from neuralforecast import NeuralForecast
            from neuralforecast.models import iTransformer
            
            model_nf = iTransformer(
                h=h,
                input_size=input_size,
                n_series=len(available_targets),  # Number of series (required)
                n_heads=model_params.get('n_heads', 8),  # Use n_heads (not n_head)
                e_layers=model_params.get('e_layers', 2),
                d_layers=model_params.get('d_layers', 1),
                hidden_size=model_params.get('d_model', 512),  # Use hidden_size (not d_model)
                d_ff=model_params.get('d_ff', 2048),
                dropout=model_params.get('dropout', 0.1),
                batch_size=model_params.get('batch_size', 32),
                learning_rate=model_params.get('learning_rate', 0.001),
                max_steps=model_params.get('max_epochs', 10) * 100,  # Convert epochs to steps
                val_check_steps=50,
                early_stop_patience_steps=3
            )
            nf = NeuralForecast(models=[model_nf], freq='W')
            logger.info(f"Fitting ITransformer model...")
            # Split data: use 20% for validation if early stopping is enabled
            val_size = max(int(len(nf_df) * 0.2 / len(available_targets)), 12) if len(nf_df) > 20 else 0
            nf.fit(df=nf_df, val_size=val_size if val_size > 0 else None)
            model = nf  # Save NeuralForecast wrapper
        
        elif model_type_lower == 'patchtst':
            from neuralforecast import NeuralForecast
            from neuralforecast.models import PatchTST
            
            model_nf = PatchTST(
                h=h,
                input_size=input_size,
                patch_len=model_params.get('patch_len', 16),
                stride=model_params.get('stride', 8),
                hidden_size=model_params.get('d_model', 128),
                n_heads=model_params.get('n_heads', 8),
                encoder_layers=model_params.get('num_layers', 3),
                dropout=model_params.get('dropout', 0.1),
                batch_size=model_params.get('batch_size', 32),
                learning_rate=model_params.get('learning_rate', 0.001),
                max_steps=model_params.get('max_epochs', 10) * 100,
                val_check_steps=50,
                early_stop_patience_steps=3
            )
            nf = NeuralForecast(models=[model_nf], freq='W')
            logger.info(f"Fitting PatchTST model...")
            val_size = max(int(len(nf_df) * 0.2 / len(available_targets)), 12) if len(nf_df) > 20 else 0
            nf.fit(df=nf_df, val_size=val_size if val_size > 0 else None)
            model = nf
        
        elif model_type_lower == 'tft':
            from neuralforecast import NeuralForecast
            from neuralforecast.models import TFT
            
            model_nf = TFT(
                h=h,
                input_size=input_size,
                hidden_size=model_params.get('hidden_size', 64),
                n_head=model_params.get('num_heads', 4),
                dropout=model_params.get('dropout', 0.1),
                batch_size=model_params.get('batch_size', 256),
                learning_rate=model_params.get('learning_rate', 0.001),
                max_steps=model_params.get('max_epochs', 10) * 100,
                val_check_steps=50,
                early_stop_patience_steps=3
            )
            nf = NeuralForecast(models=[model_nf], freq='W')
            logger.info(f"Fitting TFT model...")
            val_size = max(int(len(nf_df) * 0.2 / len(available_targets)), 12) if len(nf_df) > 20 else 0
            nf.fit(df=nf_df, val_size=val_size if val_size > 0 else None)
            model = nf
        
        elif model_type_lower == 'timemixer':
            from neuralforecast import NeuralForecast
            from neuralforecast.models import TimeMixer
            
            model_nf = TimeMixer(
                h=h,
                input_size=input_size,
                n_series=len(available_targets),  # Number of series (required)
                e_layers=model_params.get('n_layers', 2),
                dropout=model_params.get('dropout', 0.1),
                batch_size=model_params.get('batch_size', 32),
                learning_rate=model_params.get('learning_rate', 0.001),
                max_steps=model_params.get('max_epochs', 10) * 100,
                val_check_steps=50,
                early_stop_patience_steps=3
            )
            nf = NeuralForecast(models=[model_nf], freq='W')
            logger.info(f"Fitting TimeMixer model...")
            val_size = max(int(len(nf_df) * 0.2 / len(available_targets)), 12) if len(nf_df) > 20 else 0
            nf.fit(df=nf_df, val_size=val_size if val_size > 0 else None)
            model = nf
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    else:
        # Fallback to sktime if available (for PatchTST)
        if model_type_lower == 'patchtst':
            try:
                from sktime.forecasting.patch_tst import PatchTSTForecaster
                model = PatchTSTForecaster(
                    prediction_length=model_params.get('prediction_length', 24),
                    context_length=model_params.get('context_length', 96),
                    patch_len=model_params.get('patch_len', 16),
                    stride=model_params.get('stride', 8),
                    d_model=model_params.get('d_model', 128),
                    n_heads=model_params.get('n_heads', 8),
                    num_layers=model_params.get('num_layers', 3),
                    dropout=model_params.get('dropout', 0.1),
                    batch_size=model_params.get('batch_size', 32),
                    max_epochs=model_params.get('max_epochs', 10),
                    learning_rate=model_params.get('learning_rate', 0.001)
                )
                logger.info(f"Using sktime PatchTSTForecaster (neuralforecast not available)")
            except ImportError:
                raise ImportError("neuralforecast required for PatchTST training")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    # Fit model (already fitted for neuralforecast models)
    if not use_neuralforecast:
        logger.info(f"Fitting {model_type.upper()} model on {len(available_targets)} target series...")
        model.fit(target_data)
        logger.info("Model training completed!")
    else:
        logger.info("Model training completed!")
    
    # Save model
    model_path = outputs_dir / "model.pkl"
    logger.info(f"Saving model checkpoint to: {model_path.resolve()}")
    
    if use_neuralforecast:
        # NeuralForecast models use different saving mechanism
        try:
            # NeuralForecast wrapper can be saved with joblib
            joblib.dump(model, model_path)
            logger.info(f"Model checkpoint saved successfully: {model_path}")
        except Exception as e:
            logger.warning(f"joblib save failed: {e}. Trying pickle...")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Model saved to {model_path} using pickle")
        
        # Also save each model in the wrapper
        try:
            for i, model_obj in enumerate(model.models):
                model_obj_path = outputs_dir / f"model_{i}.pkl"
                try:
                    joblib.dump(model_obj, model_obj_path)
                except:
                    pass  # Some models may not pickle
        except Exception as e:
            logger.debug(f"Could not save individual models: {e}")
    else:
        # Standard sktime models
        try:
            # Try joblib first
            joblib.dump(model, model_path)
            logger.info(f"Model checkpoint saved successfully: {model_path}")
        except Exception as e:
            logger.warning(f"joblib save failed: {e}. Trying pickle...")
            # Fallback to pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Model checkpoint saved successfully (pickle): {model_path}")
        
        # Save model as zip if sktime supports it
        try:
            model_zip_path = outputs_dir / "model.zip"
            model.save(model_zip_path)
            logger.info(f"Model also saved to {model_zip_path}")
        except Exception as e:
            logger.debug(f"Could not save as zip: {e}")
