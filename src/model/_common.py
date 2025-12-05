"""Common utilities and shared functionality for model wrappers.

This module provides shared functionality used by both DFM and DDFM wrappers
to reduce code duplication and improve maintainability.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union, Callable
from datetime import datetime
import numpy as np
import pandas as pd

# Set up paths using centralized utility (relative import since we're in src/)
from ..utils.path_setup import setup_paths
setup_paths(include_app=True)

# Import custom exceptions for error handling
from app.utils import ValidationError


def load_config_impl(
    model: Any,
    metadata: Dict[str, Any],
    source: Any = None,
    *,
    yaml: Optional[Union[str, Path]] = None,
    mapping: Optional[Dict[str, Any]] = None,
    hydra: Optional[Any] = None
) -> None:
    """Shared implementation for load_config method.
    
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


def create_deprecation_warning_message(method_name: str, replacement: str) -> str:
    """Create a standardized deprecation warning message.
    
    Args:
        method_name: Name of the deprecated method
        replacement: What to use instead
        
    Returns:
        Formatted deprecation warning message
    """
    return (
        f"{method_name}() is deprecated. Use {replacement} instead. "
        f"The new API requires DFMDataModule which is automatically created from data_path."
    )


def validate_data_module_requirements(
    dfm_data_module: Optional[Any],
    create_transformer_func: Optional[Any]
) -> None:
    """Validate that requirements for creating DFMDataModule are available.
    
    Args:
        dfm_data_module: DFMDataModule class (or None if not available)
        create_transformer_func: Function to create transformer (or None if not available)
        
    Raises:
        ImportError: If required dependencies are not available
    """
    if dfm_data_module is None:
        raise ImportError(
            "DFMDataModule not available. Install dfm-python with PyTorch support: "
            "pip install dfm-python[deep]"
        )
    if create_transformer_func is None:
        raise ImportError(
            "create_transformer_from_config not available. Check preprocess module."
        )


def create_standard_error_message(
    operation: str,
    reason: str,
    suggestion: Optional[str] = None
) -> str:
    """Create a standardized, actionable error message.
    
    Args:
        operation: What operation was being attempted
        reason: Why it failed
        suggestion: Optional suggestion for how to fix it
        
    Returns:
        Formatted error message
    """
    msg = f"{operation} failed: {reason}"
    if suggestion:
        msg += f"\nSuggestion: {suggestion}"
    return msg


def create_data_module_impl(
    model: Any,
    data_path: str,
    dfm_data_module: Optional[Any],
    create_transformer_func: Optional[Any]
) -> Any:
    """Shared implementation for creating DFMDataModule from data_path and config.
    
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
    
    # Validate that required dependencies are available
    validate_data_module_requirements(dfm_data_module, create_transformer_func)
    
    # Get config from model - must be loaded before creating data module
    config = model.get_config()
    if config is None:
        raise ValidationError(
            create_standard_error_message(
                operation="Creating data module",
                reason="Configuration not loaded",
                suggestion="Call load_config() first to load the model configuration"
            )
        )
    
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
        from preprocess.utils import read_data
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
    
    # Note: Statistics (Mx/Wx) extraction removed - DFMDataModule will handle this
    # Pipeline is set to None to avoid index issues, and DFMDataModule will compute statistics from data
    
    # Create DFMDataModule with config, PREPROCESSED data, and NO pipeline
    # Pipeline is set to None to avoid index issues in DFMDataModule.setup()
    # Statistics (Mx/Wx) will be computed from data if needed
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


def train_impl(
    model_wrapper: Any,
    model_instance: Any,
    metadata: Dict[str, Any],
    data_module: Optional[Any],
    data_path: Optional[str],
    create_data_module_method: Callable[[str], Any],
    trainer_class: Optional[Any] = None,
    **kwargs
) -> Any:
    """Shared implementation for train method.
    
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
            # Check if data was loaded via deprecated load_data() method
            # This supports backward compatibility with old API
            if metadata.get("data_loaded") and "data_path" in metadata:
                data_path = metadata["data_path"]
            else:
                raise ValidationError(
                    "Either data_module or data_path must be provided. "
                    "Use train(data_module=...) or train(data_path=...)."
                )
        
        # Create data module from data_path
        # This automatically handles data loading, preprocessing, and transformer setup
        data_module = create_data_module_method(data_path)
        metadata["data_path"] = data_path
    
    # Determine trainer class if not provided
    if trainer_class is None:
        model_type = metadata.get("model_type", "dfm")
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


def save_to_outputs_impl(
    model_wrapper: Any,
    model_name: str,
    outputs_dir: Path,
    config_path: Optional[str] = None
) -> Path:
    """Shared implementation for save_to_outputs method.
    
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
    except:
        # Silently handle case where time index is not available
        # This can happen if model hasn't been trained or time index not set
        pass
    
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
    
    # Create subdirectories for logs, plots, and results
    # These are created even if empty to maintain consistent structure
    (model_dir / "logs").mkdir(exist_ok=True)
    (model_dir / "plots").mkdir(exist_ok=True)
    (model_dir / "results").mkdir(exist_ok=True)
    
    return model_dir
