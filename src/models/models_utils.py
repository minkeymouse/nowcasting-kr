"""Utility functions for model wrappers.

This module provides common utilities for DFM and DDFM model operations.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Callable, Union, List, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import pickle

from src.utils import ValidationError

try:
    from dfm_python.lightning import DFMDataModule
except ImportError:
    DFMDataModule = None

# Import helper functions from models_forecasters
from .models_forecasters import _create_preprocessing_pipeline, _train_factor_model

# ============================================================================
# Common Utilities for Model Wrappers
# ============================================================================

def _convert_horizon(fh: Union[int, List, np.ndarray]) -> int:
    """Convert forecast horizon to integer.
    
    Args:
        fh: Forecast horizon (int, list, or array)
        
    Returns:
        Integer horizon value. For lists/arrays, returns the maximum value (for fh=[1,2,3] returns 3)
        or the single value if list has one element (for fh=[5] returns 5).
    """
    if isinstance(fh, (list, np.ndarray)):
        if len(fh) == 0:
            return 1
        # For list/array, take the maximum value (e.g., fh=[1,2,3] -> 3, fh=[5] -> 5)
        # This handles both single values and ranges
        fh_array = np.asarray(fh)
        return int(np.max(fh_array))
    return int(fh)


def _handle_prediction_error(e: Exception, horizon: int, model_type: str) -> None:
    """Handle prediction errors with appropriate logging.
    
    Args:
        e: Exception that occurred
        horizon: Forecast horizon
        model_type: 'DFM' or 'DDFM'
    """
    import logging
    logger = logging.getLogger(__name__)
    error_msg = str(e).lower()
    
    # For long horizons (>= 28), this often indicates numerical instability
    if horizon >= 28:
        logger.warning(
            f"{model_type} predict() failed for long horizon={horizon}: {type(e).__name__}: {e}. "
            f"This typically indicates numerical instability, convergence issues, or horizon exceeding data limits."
        )
    elif "nan" in error_msg or "inf" in error_msg:
        logger.warning(
            f"{model_type} predict() failed due to NaN/Inf values at horizon={horizon}: {e}. "
            f"This may indicate model convergence issues or numerical instability."
        )
    else:
        logger.error(f"{model_type} predict() failed for horizon={horizon}: {type(e).__name__}: {e}")


def _validate_predictions(predictions: Any, horizon: int, model_type: str) -> None:
    """Validate that predictions are not empty.
    
    Args:
        predictions: Model predictions (can be array or tuple)
        horizon: Forecast horizon
        model_type: 'DFM' or 'DDFM'
        
    Raises:
        ValueError: If predictions are None or empty
    """
    if predictions is None:
        raise ValueError(f"{model_type} predict() returned None for horizon={horizon}")
    
    # Extract array from tuple if needed
    if isinstance(predictions, tuple):
        pred_array = predictions[0] if len(predictions) > 0 else None
    else:
        pred_array = predictions
    
    if pred_array is not None:
        # Convert torch tensors to numpy array (handle CUDA tensors)
        try:
            import torch
            if isinstance(pred_array, torch.Tensor):
                pred_array = pred_array.cpu().numpy()
            else:
                pred_array = np.asarray(pred_array)
        except ImportError:
            pred_array = np.asarray(pred_array)
        if pred_array.size == 0:
            raise ValueError(
                f"{model_type} predict() returned empty array for horizon={horizon}. "
                f"This may indicate numerical instability or horizon exceeding data limits."
            )


def _convert_predictions_to_dataframe(
    predictions: Union[np.ndarray, Tuple],
    y_train: pd.DataFrame,
    horizon: int
) -> pd.DataFrame:
    """Convert model predictions to DataFrame with proper index and columns.
    
    This helper function handles the common pattern of converting predictions
    from DFM/DDFM models to a pandas DataFrame with appropriate index and columns.
    Used by both DFMForecaster and DDFMForecaster.
    
    Args:
        predictions: Model predictions (can be array or tuple)
        y_train: Training data DataFrame (for column names and index)
        horizon: Forecast horizon
        
    Returns:
        DataFrame with predictions, matching y_train structure
    """
    # Extract forecast array from tuple if needed
    if isinstance(predictions, tuple):
        X_forecast = predictions[0]
    else:
        X_forecast = predictions
    
    # Validate X_forecast is not empty
    if X_forecast is None:
        raise ValueError(f"_convert_predictions_to_dataframe: predictions is None for horizon={horizon}")
    
    # Convert torch tensors to numpy array (handle CUDA tensors)
    try:
        import torch
        if isinstance(X_forecast, torch.Tensor):
            X_forecast = X_forecast.cpu().numpy()
        else:
            X_forecast = np.asarray(X_forecast)
    except ImportError:
        # torch not available, use standard conversion
        X_forecast = np.asarray(X_forecast)
    
    # Validate shape
    if X_forecast.size == 0:
        raise ValueError(f"_convert_predictions_to_dataframe: predictions array is empty for horizon={horizon}")
    
    if len(X_forecast.shape) < 2:
        # Reshape to 2D if 1D
        X_forecast = X_forecast.reshape(-1, 1) if len(X_forecast.shape) == 1 else X_forecast
    
    # Get series order from model config or training data
    try:
        from dfm_python.utils.helpers import get_series_ids
        # Try to get config from the model (if available in context)
        # This is a fallback - actual implementation should pass config
        series_ids = list(y_train.columns)
    except (ImportError, AttributeError, Exception):
        series_ids = list(y_train.columns)
    
    # Ensure series_ids matches the number of columns in X_forecast
    if len(series_ids) != X_forecast.shape[1]:
        series_ids = list(y_train.columns)
    
    # Create forecast index
    last_train_idx = y_train.index[-1]
    if isinstance(last_train_idx, pd.Timestamp):
        freq = pd.infer_freq(y_train.index)
        forecast_index = pd.date_range(
            start=last_train_idx + pd.Timedelta(days=1),
            periods=horizon,
            freq=freq
        )
    else:
        forecast_index = np.arange(
            int(last_train_idx) + 1,
            int(last_train_idx) + 1 + horizon
        )
    
    # Create DataFrame with series_ids as columns
    y_pred = pd.DataFrame(
        X_forecast,
        index=forecast_index,
        columns=series_ids[:X_forecast.shape[1]] if len(series_ids) >= X_forecast.shape[1] else list(y_train.columns)
    )
    
    # Reorder columns to match training data order
    if set(y_pred.columns) == set(y_train.columns):
        y_pred = y_pred[y_train.columns]
    
    return y_pred

def load_config(
    model: Any,
    metadata: Dict[str, Any],
    source: Any = None,
    *,
    yaml: Optional[Union[str, Path]] = None,
    mapping: Optional[Dict[str, Any]] = None,
    hydra: Optional[Any] = None
) -> None:
    """Load configuration from various sources.
    
    This function handles the common logic for loading configuration from
    various sources (YAML, dict, Hydra config, etc.) and updating metadata.
    
    Args:
        model: The underlying model instance (DFM or DDFM)
        metadata: The metadata dictionary to update
        source: Configuration source (YAML path, dict, DFMConfig, or ConfigSource)
        yaml: Path to YAML config file
        mapping: Dictionary config
        hydra: Hydra DictConfig object
        
    Raises:
        ValidationError: If no configuration source is provided
        
    Note: If multiple keyword arguments are provided, only one should be used.
    The underlying model will validate this.
    """
    # Pass through to underlying model with keyword support
    if hydra is not None:
        model.load_config(hydra=hydra)
    elif yaml is not None:
        model.load_config(yaml=yaml)
    elif mapping is not None:
        model.load_config(mapping=mapping)
    elif source is not None:
        model.load_config(source)
    else:
        raise ValidationError("No configuration source provided")
    metadata["config_loaded"] = True




def create_data_module_from_dataframe(
    model: Any,
    data: pd.DataFrame,
    dfm_data_module: Optional[Any],
    create_transformer_func: Optional[Any]
) -> Any:
    """Create DFMDataModule from in-memory DataFrame (preprocessed data).
    
    This function creates a DFMDataModule instance from an already-preprocessed
    pandas DataFrame, avoiding the need for temporary files.
    
    Args:
        model: The model wrapper instance (DFM or DDFM) that has get_config() method
        data: Preprocessed pandas DataFrame (already standardized, no missing values)
        dfm_data_module: DFMDataModule class (or None if not available)
        create_transformer_func: Function to create transformer pipeline from config
        
    Returns:
        DFMDataModule instance ready for training (setup() will be called)
        
    Raises:
        ValidationError: If config is not loaded or transformer creation fails
        ImportError: If required dependencies are not available
    """
    # Validate dependencies
    if dfm_data_module is None:
        raise ImportError("DFMDataModule not available. Install dfm-python with PyTorch support: pip install dfm-python[deep]")
    if create_transformer_func is None:
        raise ImportError("create_transformer_from_config not available. Check preprocess module.")
    
    # Get config from model
    if hasattr(model, 'config'):
        config = model.config
    elif hasattr(model, '_config'):
        config = model._config
    elif hasattr(model, 'get_config'):
        config = model.get_config()
    else:
        config = None
    
    if config is None:
        raise ValidationError("Configuration not loaded. Call load_config() first to load the model configuration.")
    
    # Create preprocessing pipeline from config for statistics extraction (Mx/Wx)
    # Data is already preprocessed, pipeline is only for extracting statistics
    try:
        pipeline = create_transformer_func(config)
        # Test if pipeline can handle the data columns
        if isinstance(data, pd.DataFrame):
            test_data = data.iloc[:1]
            try:
                pipeline.fit_transform(test_data)
            except (ValueError, TypeError, AttributeError, KeyError):
                # Pipeline can't handle the data (column mismatch)
                pipeline = None
    except (ValueError, TypeError, AttributeError, ImportError, KeyError):
        pipeline = None
    if isinstance(data, pd.DataFrame):
        # Ensure index is sktime-compatible (RangeIndex, DatetimeIndex, or PeriodIndex)
        data_to_pass = data.copy()
        if not isinstance(data_to_pass.index, (pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex)):
            # Convert to RangeIndex if not compatible
            data_to_pass.index = pd.RangeIndex(start=0, stop=len(data_to_pass))
        
        # Convert pandas Index to TimeIndex object for DFMDataModule
        try:
            from dfm_python.utils.time import TimeIndex
            from datetime import datetime
            
            # Convert pandas Index to list of datetime objects for TimeIndex
            if isinstance(data_to_pass.index, pd.DatetimeIndex):
                # DatetimeIndex: convert to list of datetime objects
                time_index = TimeIndex(data_to_pass.index.to_pydatetime().tolist())
            elif isinstance(data_to_pass.index, pd.PeriodIndex):
                # PeriodIndex: convert to datetime
                time_index = TimeIndex(data_to_pass.index.to_timestamp().to_pydatetime().tolist())
            elif isinstance(data_to_pass.index, pd.RangeIndex):
                raise ValueError(
                    f"RangeIndex not supported for nowcasting - data must have DatetimeIndex. "
                    f"Data index type: {type(data_to_pass.index)}"
                )
            else:
                # Other index types: try to convert to datetime
                try:
                    # Try to convert index values to datetime
                    datetime_index = pd.to_datetime(data_to_pass.index)
                    time_index = TimeIndex(datetime_index.to_pydatetime().tolist())
                except (ValueError, TypeError) as e:
                    # If conversion fails, raise error - we need dates for nowcasting
                    raise ValueError(
                        f"Cannot convert index to TimeIndex for nowcasting: {e}. "
                        f"Data must have DatetimeIndex or PeriodIndex."
                    ) from e
        except (ImportError, AttributeError, Exception) as e:
            raise ValueError(f"Failed to create TimeIndex for data_module: {e}") from e
    else:
        # If numpy array, convert to DataFrame using data columns if available
        data_array = np.asarray(data)
        time_index = None
        # Try to get column names from config, but this may not match if target_series was added
        try:
            from dfm_python.utils.helpers import get_series_ids
            series_ids = get_series_ids(config)
            # Adjust if shape doesn't match
            if len(series_ids) != data_array.shape[1]:
                series_ids = [f"series_{i}" for i in range(data_array.shape[1])]
            data_to_pass = pd.DataFrame(data_array, columns=series_ids)
        except (ImportError, AttributeError):
            data_to_pass = pd.DataFrame(data_array)
    
    data_module = dfm_data_module(
        config=config,
        pipeline=pipeline,
        data=data_to_pass,
        time_index=time_index
    )
    data_module.setup()
    
    return data_module


def create_data_module(
    model: Any,
    data_path: str,
    dfm_data_module: Optional[Any],
    create_transformer_func: Optional[Any]
) -> Any:
    """Create DFMDataModule from data_path and config.
    
    This function handles the common logic for creating a DFMDataModule instance
    that is used by both DFM and DDFM wrappers.
    
    **dfm-python 0.4.5+ pattern**:
    - Users must preprocess data BEFORE passing to DFMDataModule
    - The pipeline parameter is ONLY for extracting statistics (Mx/Wx) for forecasting/nowcasting
    - DFMDataModule expects preprocessed data (standardized, no missing values)
    - The pipeline is fitted on preprocessed data to extract statistics, not to preprocess
    
    Args:
        model: The model wrapper instance (DFM or DDFM) that has get_config() method
        data_path: Path to data file (raw data)
        dfm_data_module: DFMDataModule class (or None if not available)
        create_transformer_func: Function to create transformer pipeline from config
        
    Returns:
        DFMDataModule instance ready for training (setup() will be called)
        
    Raises:
        ValidationError: If config is not loaded or transformer creation fails
        ImportError: If required dependencies are not available
    """
    import pandas as pd
    
    # Validate dependencies
    if dfm_data_module is None:
        raise ImportError("DFMDataModule not available. Install dfm-python with PyTorch support: pip install dfm-python[deep]")
    if create_transformer_func is None:
        raise ImportError("create_transformer_from_config not available. Check preprocess module.")
    
    # Get config from model
    config = model.get_config()
    if config is None:
        raise ValidationError("Configuration not loaded. Call load_config() first to load the model configuration.")
    
    # Create preprocessing pipeline from config
    # This pipeline will be used to preprocess raw data AND extract statistics (Mx/Wx)
    pipeline = create_transformer_func(config)
    
    # Load raw data from file
    # Use dfm-python's data loading utility to get raw data
    try:
        from dfm_python.data.utils import load_data as dfm_load_data
        X_raw, Time, _ = dfm_load_data(data_path, config)
    except (ImportError, AttributeError):
        # Fallback: use our own data loading
        from src.preprocessing import read_data
        X_raw, Time, _ = read_data(data_path)
    
    # Convert to pandas DataFrame
    # Get series IDs from config
    try:
        series_ids = config.get_series_ids()
    except AttributeError:
        # Fallback: generate series IDs
        n_series = X_raw.shape[1] if len(X_raw.shape) > 1 else 1
        series_ids = [f"series_{i}" for i in range(n_series)]
    
    # Ensure we have the right number of series
    if len(series_ids) != X_raw.shape[1]:
        # Adjust series_ids to match data
        series_ids = [f"series_{i}" for i in range(X_raw.shape[1])]
    
    # Convert to pandas DataFrame (raw data)
    # TimeIndex has 'dates' attribute (list of datetime objects)
    # Ensure index is DatetimeIndex, PeriodIndex, or RangeIndex for sktime compatibility
    if hasattr(Time, 'dates'):
        # TimeIndex object - convert dates list to DatetimeIndex
        try:
            time_index = pd.DatetimeIndex(Time.dates)
        except (ValueError, TypeError):
            # If conversion fails, use RangeIndex
            time_index = pd.RangeIndex(start=0, stop=len(Time.dates))
    elif hasattr(Time, 'series'):
        # Fallback: use series if available
        time_index = Time.series
        if not isinstance(time_index, (pd.DatetimeIndex, pd.PeriodIndex, pd.RangeIndex)):
            try:
                time_index = pd.to_datetime(time_index)
            except (ValueError, TypeError):
                time_index = pd.RangeIndex(start=0, stop=len(time_index))
    elif hasattr(Time, 'to_pandas'):
        # Fallback: try to_pandas() method
        time_index = Time.to_pandas()
        if not isinstance(time_index, (pd.DatetimeIndex, pd.PeriodIndex, pd.RangeIndex)):
            try:
                time_index = pd.to_datetime(time_index)
            except (ValueError, TypeError):
                time_index = pd.RangeIndex(start=0, stop=len(time_index))
    else:
        # No time index available - use RangeIndex for sktime compatibility
        time_index = pd.RangeIndex(start=0, stop=X_raw.shape[0])
    
    X_df = pd.DataFrame(X_raw, columns=series_ids, index=time_index)
    
    # Preprocess data using the pipeline
    # This is required: dfm-python expects preprocessed data (standardized, no missing values)
    # Note: We fit_transform here to avoid index issues in DFMDataModule.setup()
    # The pipeline applies transformations specified in series config files, then standardization
    X_processed = pipeline.fit_transform(X_df)
    
    # Convert to numpy array to avoid index compatibility issues with sktime
    # DFMDataModule can handle numpy arrays directly, and this avoids ColumnEnsembleTransformer index issues
    if isinstance(X_processed, pd.DataFrame):
        # Extract values and preserve column names if needed
        X_processed_array = X_processed.values
        # Store column names for reference (DFMDataModule will use config series IDs)
        X_processed = X_processed_array
    elif isinstance(X_processed, np.ndarray):
        # Already an array, use as-is
        pass
    else:
        # Try to convert to array
        X_processed = np.asarray(X_processed)
    
    # Create DFMDataModule with preprocessed data
    # Pipeline is None to avoid index issues - DFMDataModule computes statistics from data
    data_module = dfm_data_module(
        config=config,
        pipeline=None,  # Set to None to avoid index issues - data is already preprocessed
        data=X_processed,  # PREPROCESSED data (required by dfm-python)
        time_index=Time
    )
    
    # Setup the data module - this fits the pipeline on preprocessed data to extract statistics
    # (Mx/Wx) for forecasting/nowcasting operations
    data_module.setup()
    
    return data_module


def train_model(
    model_wrapper: Any,
    model_instance: Any,
    metadata: Dict[str, Any],
    data_module: Optional[Any],
    data_path: Optional[str],
    create_data_module_method: Callable[[str], Any],
    trainer_class: Optional[Any] = None,
    **kwargs
) -> Any:
    """Train model using PyTorch Lightning trainer.
    
    This function handles the common logic for training models that is shared
    between DFM and DDFM wrappers. It handles data module determination,
    model training using PyTorch Lightning trainers, and returns the result for metadata extraction.
    
    Args:
        model_wrapper: The model wrapper instance (self) for accessing methods
        model_instance: The underlying model instance (self._model)
        metadata: The metadata dictionary to update
        data_module: Optional DFMDataModule instance (if provided, used directly)
        data_path: Optional path to data file (used if data_module is None)
        create_data_module_method: Method to create data module from data_path
        trainer_class: Trainer class to use (DFMTrainer or DDFMTrainer). If None, auto-detects from model type.
        **kwargs: Additional training parameters passed to trainer (e.g., max_epochs, max_iter, threshold for DFM, epochs for DDFM)
        
    Returns:
        Model result object (from model.get_result())
        
    Raises:
        ValidationError: If neither data_module nor data_path is provided
        ImportError: If trainer classes are not available
    """
    # Import trainers
    try:
        from dfm_python import DFMTrainer, DDFMTrainer
    except ImportError:
        raise ImportError(
            "DFMTrainer/DDFMTrainer not available. Install dfm-python with PyTorch Lightning support."
        )
    
    # Determine data module - prioritize provided data_module, fall back to data_path
    if data_module is None:
        if data_path is None:
            raise ValidationError(
                "Either data_module or data_path must be provided. "
                "Use train(data_module=...) or train(data_path=...)."
            )
        
        # Create data module from data_path
        # This automatically handles data loading, preprocessing, and transformer setup
        data_module = create_data_module_method(data_path)
        metadata["data_path"] = data_path
    
    # Determine model type first (needed for DDFM-specific fixes)
    model_type = metadata.get("model_type", "dfm")
    
    # Determine trainer class if not provided
    if trainer_class is None:
        if model_type == "ddfm":
            trainer_class = DDFMTrainer
        else:
            trainer_class = DFMTrainer
    
    # Extract trainer parameters from kwargs
    # For DFM: max_iter -> max_epochs, threshold is handled by model
    # For DDFM: epochs -> max_epochs
    trainer_kwargs = {}
    if "max_epochs" in kwargs:
        trainer_kwargs["max_epochs"] = kwargs.pop("max_epochs")
    elif "max_iter" in kwargs:
        # DFM uses max_iter, convert to max_epochs for trainer
        trainer_kwargs["max_epochs"] = kwargs.pop("max_iter")
    elif "epochs" in kwargs:
        # DDFM uses epochs, convert to max_epochs for trainer
        trainer_kwargs["max_epochs"] = kwargs.pop("epochs")
    else:
        # Default max_epochs
        trainer_kwargs["max_epochs"] = 100
    
    # DDFM-specific fixes for numerical stability
    if model_type == "ddfm":
        # Enable gradient clipping to prevent gradient explosion
        if "gradient_clip_val" in kwargs:
            trainer_kwargs["gradient_clip_val"] = kwargs.pop("gradient_clip_val")
        else:
            trainer_kwargs["gradient_clip_val"] = 1.0  # Default: clip gradients at 1.0
    
    # Pass through other trainer kwargs (enable_progress_bar, etc.)
    # Default to True for progress visibility
    if "enable_progress_bar" in kwargs:
        trainer_kwargs["enable_progress_bar"] = kwargs.pop("enable_progress_bar")
    else:
        trainer_kwargs["enable_progress_bar"] = True  # Default: show progress
    
    if "enable_model_summary" in kwargs:
        trainer_kwargs["enable_model_summary"] = kwargs.pop("enable_model_summary")
    
    # Create trainer
    trainer = trainer_class(**trainer_kwargs)
    
    # Train the model using PyTorch Lightning pattern
    # trainer.fit() handles the training loop
    trainer.fit(model_instance, data_module)
    
    # Get result after training
    result = model_instance.get_result()
    
    # Update common metadata
    metadata["training_completed"] = datetime.now().isoformat()
    
    return result


def save_to_outputs(
    model_wrapper: Any,
    model_name: str,
    outputs_dir: Path,
    config_path: Optional[str] = None
) -> Path:
    """Save model to outputs directory structure.
    
    This function handles the common logic for saving models to the outputs/
    directory structure that is shared between DFM and DDFM wrappers.
    
    Args:
        model_wrapper: The model wrapper instance (self) for accessing methods
        model_name: Name of the model
        outputs_dir: Base outputs directory (Path object)
        config_path: Optional path to config file to copy
        
    Returns:
        Path to the model directory
    """
    import pickle
    
    # Create model directory structure
    model_dir = outputs_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect model components for saving
    # Result contains factor estimates, loadings, and training metrics
    result = model_wrapper.get_result()
    # Config contains all model configuration parameters
    config = model_wrapper.get_config()
    # Time index may not always be available (e.g., if model not trained)
    time_index = None
    try:
        time_index = model_wrapper.get_time()
    except (AttributeError, RuntimeError):
        # Silently handle case where time index is not available
        # This can happen if model hasn't been trained or time index not set
        time_index = None
    
    # Save model, result, config, and time index as a single pickle file
    # This allows complete model reconstruction later
    with open(model_dir / "model.pkl", 'wb') as f:
        pickle.dump({
            "model": model_wrapper,
            "result": result,
            "config": config,
            "time": time_index
        }, f)
    
    # Save config copy if provided (optional)
    # This creates a human-readable YAML copy of the config for reference
    if config_path:
        config_path_obj = Path(config_path)
        if config_path_obj.exists():
            with open(model_dir / "config.yaml", 'w') as f:
                with open(config_path, 'r') as src:
                    f.write(src.read())
    
    return model_dir

