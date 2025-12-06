"""sktime-compatible Forecaster wrappers for DFM and DDFM models.

This module provides sktime Forecaster interface implementations for DFM and DDFM models,
enabling integration with sktime's forecasting pipeline, splitters, and evaluation tools.
"""

from typing import Optional, Union, Any, Dict, Callable
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Path setup is handled in entry points (train.py, infer.py)

try:
    from sktime.forecasting.base import BaseForecaster
    from sktime.forecasting.base._base import ForecastingHorizon
    HAS_SKTIME = True
except ImportError:
    HAS_SKTIME = False
    BaseForecaster = None
    ForecastingHorizon = None

from ..utils.config_parser import ValidationError

# Note: DFM and DDFM are imported inside methods to avoid circular import
# (dfm.py imports utilities from this module)


def _check_sktime_available():
    """Check if sktime is available and raise ImportError if not (internal use)."""
    if not HAS_SKTIME:
        raise ImportError(
            "sktime is required for sktime forecasters. "
            "Install it with: pip install sktime[forecasting]"
        )


class DFMForecaster(BaseForecaster):
    """sktime-compatible Forecaster wrapper for DFM models.
    
    This class wraps the DFM model to work with sktime's forecasting API,
    enabling use with splitters, evaluation functions, and forecasting pipelines.
    
    Parameters
    ----------
    config_path : str or Path, optional
        Path to DFM configuration file (YAML)
    config_dict : dict, optional
        Configuration dictionary
    max_iter : int, default=5000
        Maximum EM iterations
    threshold : float, default=1e-5
        Convergence threshold
    **kwargs
        Additional parameters passed to DFM wrapper
        
    Examples
    --------
    >>> from src.model.sktime_forecaster import DFMForecaster
    >>> from sktime.split import ExpandingWindowSplitter
    >>> import numpy as np
    >>> 
    >>> # Create forecaster
    >>> forecaster = DFMForecaster(config_path="config/experiment/myexp.yaml")
    >>> 
    >>> # Define forecasting horizon
    >>> fh = np.arange(1, 13)  # Next 12 steps
    >>> 
    >>> # Fit on training data
    >>> forecaster.fit(y_train, fh=fh)
    >>> 
    >>> # Predict
    >>> y_pred = forecaster.predict(fh=fh)
    """
    
    _tags = {
        "requires-fh-in-fit": False,  # Can fit without forecasting horizon
        "handles-missing-data": True,  # DFM handles missing data via Kalman filter
        "y_inner_mtype": "pd.DataFrame",  # Multi-variate time series
        "X_inner_mtype": "pd.DataFrame",  # Optional exogenous variables
        "scitype:y": "both",  # Can handle univariate or multivariate
    }
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config_dict: Optional[dict] = None,
        max_iter: int = 5000,
        threshold: float = 1e-5,
        **kwargs
    ):
        _check_sktime_available()
        super().__init__()
        
        self.config_path = config_path
        self.config_dict = config_dict
        self.max_iter = max_iter
        self.threshold = threshold
        self._dfm_model = None
        self._is_fitted = False
        
    def _fit(self, y, X=None, fh=None):
        """Fit the DFM model to training data.
        
        Parameters
        ----------
        y : pd.DataFrame
            Training time series data (T x N), where T is time periods and N is number of series
        X : pd.DataFrame, optional
            Exogenous variables (not used by DFM, but kept for API compatibility)
        fh : ForecastingHorizon, optional
            Forecasting horizon (not required for fitting, but can be used for validation)
            
        Returns
        -------
        self
            Returns self for method chaining
        """
        # Import here to avoid circular import
        from .dfm import DFM
        # Initialize DFM model
        self._dfm_model = DFM()
        
        # Load configuration
        if self.config_path:
            self._dfm_model.load_config(yaml=self.config_path)
        elif self.config_dict:
            self._dfm_model.load_config(mapping=self.config_dict)
        else:
            raise ValidationError(
                "Either config_path or config_dict must be provided to DFMForecaster"
            )
        
        # Convert y to DataFrame if needed
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)
        
        # Create data module from in-memory DataFrame (avoiding temporary files)
        try:
            from dfm_python.lightning import DFMDataModule
            from ..preprocess.utils import create_transformer_from_config
            
            data_module = create_data_module_from_dataframe(
                model=self._dfm_model,
                data=y,
                dfm_data_module=DFMDataModule,
                create_transformer_func=create_transformer_from_config
            )
            
            # Train the model using in-memory data module
            self._dfm_model.train(data_module=data_module, max_iter=self.max_iter, threshold=self.threshold)
        except (ImportError, ValueError, AttributeError) as e:
            # Fallback to temporary file when in-memory DataFrame handling fails
            # This handles cases where DFMDataModule creation fails due to missing dependencies
            # or DataFrame-to-DataModule conversion issues
            import tempfile
            import os
            
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            y.to_csv(temp_file.name)
            temp_file.close()
            
            try:
                # Train the model using temporary file
                self._dfm_model.train(data_path=temp_file.name, max_iter=self.max_iter, threshold=self.threshold)
            finally:
                # Clean up temporary file to avoid disk space issues
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
        
        # Store training data for prediction
        self._y = y
        
        self._is_fitted = True
        return self
    
    def _predict(self, fh, X=None):
        """Generate forecasts.
        
        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon (steps ahead to forecast)
        X : pd.DataFrame, optional
            Exogenous variables (not used by DFM)
            
        Returns
        -------
        pd.DataFrame
            Forecasted values with same columns as training data
        """
        if not self._is_fitted or self._dfm_model is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        # Convert fh to integer horizon
        if isinstance(fh, ForecastingHorizon):
            horizon = len(fh)
        elif isinstance(fh, (list, np.ndarray)):
            horizon = len(fh)
        else:
            horizon = int(fh)
        
        # Get predictions from DFM model
        predictions = self._dfm_model.predict(horizon=horizon)
        
        # Convert to DataFrame with appropriate index
        if isinstance(predictions, tuple):
            # DFM returns (X_forecast, Z_forecast) tuple
            X_forecast = predictions[0]
        else:
            X_forecast = predictions
        
        # Get series order from DFM model config (predictions are in this order)
        try:
            from dfm_python.utils.helpers import get_series_ids
            config = self._dfm_model.get_config()
            series_ids = get_series_ids(config)
            # Ensure series_ids matches the number of columns in X_forecast
            if len(series_ids) != X_forecast.shape[1]:
                # Fallback: use training data columns if count doesn't match
                series_ids = list(self._y.columns)
        except (ImportError, AttributeError, Exception):
            # Fallback: use training data columns if we can't get series_ids
            series_ids = list(self._y.columns)
        
        # Create index for forecast period
        last_train_idx = self._y.index[-1]
        if isinstance(last_train_idx, pd.Timestamp):
            # Create future timestamps based on frequency
            freq = pd.infer_freq(self._y.index)
            forecast_index = pd.date_range(
                start=last_train_idx + pd.Timedelta(days=1),
                periods=horizon,
                freq=freq
            )
        else:
            # Integer index
            forecast_index = np.arange(
                int(last_train_idx) + 1,
                int(last_train_idx) + 1 + horizon
            )
        
        # Create DataFrame with series_ids as columns (matching prediction order)
        y_pred = pd.DataFrame(
            X_forecast,
            index=forecast_index,
            columns=series_ids[:X_forecast.shape[1]] if len(series_ids) >= X_forecast.shape[1] else list(self._y.columns)
        )
        
        # Reorder columns to match training data order (for consistency with evaluation)
        # This ensures target_series extraction works correctly
        if set(y_pred.columns) == set(self._y.columns):
            y_pred = y_pred[self._y.columns]
        
        return y_pred
    
    def get_fitted_params(self):
        """Get fitted parameters from the underlying DFM model.
        
        Returns
        -------
        dict
            Dictionary with fitted parameters and metadata
        """
        if not self._is_fitted or self._dfm_model is None:
            return {}
        
        result = self._dfm_model.get_result()
        metadata = self._dfm_model.get_metadata()
        
        return {
            'converged': getattr(result, 'converged', None),
            'num_iter': getattr(result, 'num_iter', None),
            'loglik': getattr(result, 'loglik', None),
            **metadata
        }


class DDFMForecaster(BaseForecaster):
    """sktime-compatible Forecaster wrapper for DDFM models.
    
    This class wraps the DDFM model to work with sktime's forecasting API.
    
    Parameters
    ----------
    config_path : str or Path, optional
        Path to DDFM configuration file (YAML)
    config_dict : dict, optional
        Configuration dictionary
    encoder_layers : list of int, optional
        Hidden layer dimensions for encoder (default: [64, 32])
    num_factors : int, optional
        Number of factors (inferred from config if None)
    epochs : int, default=100
        Number of training epochs
    learning_rate : float, default=0.001
        Learning rate for Adam optimizer
    batch_size : int, default=32
        Batch size for training
    **kwargs
        Additional parameters passed to DDFM wrapper
    """
    
    _tags = {
        "requires-fh-in-fit": False,
        "handles-missing-data": True,
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "both",
    }
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config_dict: Optional[dict] = None,
        encoder_layers: Optional[list] = None,
        num_factors: Optional[int] = None,
        epochs: int = 100,
        learning_rate: float = 0.0001,  # Reduced from 0.001 for numerical stability
        batch_size: int = 32,
        **kwargs
    ):
        _check_sktime_available()
        super().__init__()
        
        self.config_path = config_path
        self.config_dict = config_dict
        self.encoder_layers = encoder_layers
        self.num_factors = num_factors
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self._ddfm_model = None
        self._is_fitted = False
        
    def _fit(self, y, X=None, fh=None):
        """Fit the DDFM model to training data."""
        # Initialize DDFM model
        # Import here to avoid circular import
        from .ddfm import DDFM
        ddfm_params = {}
        if self.encoder_layers:
            ddfm_params['encoder_layers'] = self.encoder_layers
        if self.num_factors:
            ddfm_params['num_factors'] = self.num_factors
        
        self._ddfm_model = DDFM(**ddfm_params)
        
        # Load configuration
        if self.config_path:
            self._ddfm_model.load_config(yaml=self.config_path)
        elif self.config_dict:
            self._ddfm_model.load_config(mapping=self.config_dict)
        else:
            raise ValidationError(
                "Either config_path or config_dict must be provided to DDFMForecaster"
            )
        
        # Convert y to DataFrame if needed
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)
        
        # Create data module from in-memory DataFrame (avoiding temporary files)
        try:
            from dfm_python.lightning import DFMDataModule
            from ..preprocess.utils import create_transformer_from_config
            
            data_module = create_data_module_from_dataframe(
                model=self._ddfm_model,
                data=y,
                dfm_data_module=DFMDataModule,
                create_transformer_func=create_transformer_from_config
            )
            
            # Train the model using in-memory data module
            self._ddfm_model.train(
                data_module=data_module,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size
            )
        except (ImportError, ValueError, AttributeError) as e:
            # Fallback to temporary file when in-memory DataFrame handling fails
            # This handles cases where DFMDataModule creation fails due to missing dependencies
            # or DataFrame-to-DataModule conversion issues
            import tempfile
            import os
            
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            y.to_csv(temp_file.name)
            temp_file.close()
            
            try:
                # Train the model using temporary file
                self._ddfm_model.train(
                    data_path=temp_file.name,
                    epochs=self.epochs,
                    learning_rate=self.learning_rate,
                    batch_size=self.batch_size
                )
            finally:
                # Clean up temporary file to avoid disk space issues
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
        
        # Store training data for prediction
        self._y = y
        
        self._is_fitted = True
        return self
    
    def _predict(self, fh, X=None):
        """Generate forecasts."""
        if not self._is_fitted or self._ddfm_model is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        # Convert fh to integer horizon
        if isinstance(fh, ForecastingHorizon):
            horizon = len(fh)
        elif isinstance(fh, (list, np.ndarray)):
            horizon = len(fh)
        else:
            horizon = int(fh)
        
        # Get predictions
        predictions = self._ddfm_model.predict(horizon=horizon)
        
        # Convert to DataFrame
        if isinstance(predictions, tuple):
            X_forecast = predictions[0]
        else:
            X_forecast = predictions
        
        # Get series order from DDFM model config (predictions are in this order)
        try:
            from dfm_python.utils.helpers import get_series_ids
            config = self._ddfm_model.get_config()
            series_ids = get_series_ids(config)
            # Ensure series_ids matches the number of columns in X_forecast
            if len(series_ids) != X_forecast.shape[1]:
                # Fallback: use training data columns if count doesn't match
                series_ids = list(self._y.columns)
        except (ImportError, AttributeError, Exception):
            # Fallback: use training data columns if we can't get series_ids
            series_ids = list(self._y.columns)
        
        # Create forecast index
        last_train_idx = self._y.index[-1]
        if isinstance(last_train_idx, pd.Timestamp):
            freq = pd.infer_freq(self._y.index)
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
        
        # Create DataFrame with series_ids as columns (matching prediction order)
        y_pred = pd.DataFrame(
            X_forecast,
            index=forecast_index,
            columns=series_ids[:X_forecast.shape[1]] if len(series_ids) >= X_forecast.shape[1] else list(self._y.columns)
        )
        
        # Reorder columns to match training data order (for consistency with evaluation)
        # This ensures target_series extraction works correctly
        if set(y_pred.columns) == set(self._y.columns):
            y_pred = y_pred[self._y.columns]
        
        return y_pred
    
    def get_fitted_params(self):
        """Get fitted parameters from the underlying DDFM model."""
        if not self._is_fitted or self._ddfm_model is None:
            return {}
        
        result = self._ddfm_model.get_result()
        metadata = self._ddfm_model.get_metadata()
        
        return {
            'converged': getattr(result, 'converged', None),
            'num_iter': getattr(result, 'num_iter', None),
            'loglik': getattr(result, 'loglik', None),
            **metadata
        }



# ============================================================================
# Common Utilities for Model Wrappers
# ============================================================================

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
    
    # IMPORTANT: Ensure data is passed as DataFrame with correct columns
    # DFMDataModule.setup() will use DataFrame columns directly if data is DataFrame
    # So we don't need to worry about series_ids mismatch when passing DataFrame
    
    # Create preprocessing pipeline from config for statistics extraction (Mx/Wx)
    # Note: Data is already preprocessed, pipeline is only for extracting statistics
    # IMPORTANT: If target_series was added to data but not in config, pipeline may fail
    # Use passthrough transformer (None) to avoid series_ids mismatch issues
    # Statistics (Mx/Wx) will be computed from data directly if needed
    try:
        pipeline = create_transformer_func(config)
        # Test if pipeline can handle the data columns
        if isinstance(data, pd.DataFrame):
            test_data = data.iloc[:1]  # Test with first row
            try:
                pipeline.fit_transform(test_data)
            except (ValueError, TypeError, AttributeError):
                # Pipeline can't handle the data (likely due to column mismatch)
                # Use passthrough transformer instead
                pipeline = None
    except (ValueError, TypeError, AttributeError, ImportError):
        # If pipeline creation fails, use passthrough
        pipeline = None
    
    # IMPORTANT: Pass DataFrame directly (not numpy array) to preserve column names
    # DFMDataModule.setup() will use DataFrame columns directly, avoiding series_ids mismatch
    # This is critical when target_series is added to training data but not in config's series_ids
    if isinstance(data, pd.DataFrame):
        # Ensure index is sktime-compatible (RangeIndex, DatetimeIndex, or PeriodIndex)
        data_to_pass = data.copy()
        if not isinstance(data_to_pass.index, (pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex)):
            # Convert to RangeIndex if not compatible
            data_to_pass.index = pd.RangeIndex(start=0, stop=len(data_to_pass))
        time_index = data_to_pass.index
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
    
    # Create DFMDataModule with config, preprocessed data, and pipeline for statistics
    # Pipeline will be fitted in setup() to extract statistics (Mx/Wx) for forecasting
    data_module = dfm_data_module(
        config=config,
        pipeline=pipeline,  # For extracting statistics (Mx/Wx)
        data=data_to_pass,  # Pass DataFrame directly to preserve column names
        time_index=time_index
    )
    
    # Setup the data module - this fits the pipeline on preprocessed data to extract statistics
    # (Mx/Wx) for forecasting/nowcasting operations
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
        from ..preprocess.utils import read_data
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
    
    # Create subdirectories for logs, plots, and results
    # These are created even if empty to maintain consistent structure
    (model_dir / "logs").mkdir(exist_ok=True)
    (model_dir / "plots").mkdir(exist_ok=True)
    (model_dir / "results").mkdir(exist_ok=True)
    
    return model_dir
