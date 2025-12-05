"""sktime-compatible Forecaster wrappers for DFM and DDFM models.

This module provides sktime Forecaster interface implementations for DFM and DDFM models,
enabling integration with sktime's forecasting pipeline, splitters, and evaluation tools.
"""

from typing import Optional, Union, Any
import numpy as np
import pandas as pd
from pathlib import Path

# Set up paths
from ..utils.path_setup import setup_paths
setup_paths(include_dfm_python=True, include_src=True, include_app=True)

try:
    from sktime.forecasting.base import BaseForecaster
    from sktime.forecasting.base._base import ForecastingHorizon
    HAS_SKTIME = True
except ImportError:
    HAS_SKTIME = False
    BaseForecaster = None
    ForecastingHorizon = None

from app.utils import ValidationError

# Import model wrappers
from .dfm import DFM
from .ddfm import DDFM


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
        
        # Convert y to numpy array if needed
        if isinstance(y, pd.DataFrame):
            data_array = y.values
            data_index = y.index
            data_columns = y.columns
        else:
            data_array = np.array(y)
            data_index = None
            data_columns = None
        
        # Save data to temporary file for DFM (DFM expects file path)
        # This is a workaround until DFM wrapper supports in-memory arrays
        import tempfile
        import os
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_df = pd.DataFrame(data_array, index=data_index, columns=data_columns)
        temp_df.to_csv(temp_file.name)
        temp_file.close()
        
        try:
            # Train the model using temporary file
            self._dfm_model.train(data_path=temp_file.name, max_iter=self.max_iter, threshold=self.threshold)
        finally:
            # Clean up temporary file
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
        
        # Create DataFrame with same column names as training data
        y_pred = pd.DataFrame(
            X_forecast,
            index=forecast_index,
            columns=self._y.columns
        )
        
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
        learning_rate: float = 0.001,
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
        
        # Convert y to numpy array if needed
        if isinstance(y, pd.DataFrame):
            data_array = y.values
            data_index = y.index
            data_columns = y.columns
        else:
            data_array = np.array(y)
            data_index = None
            data_columns = None
        
        # Save data to temporary file for DDFM
        import tempfile
        import os
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_df = pd.DataFrame(data_array, index=data_index, columns=data_columns)
        temp_df.to_csv(temp_file.name)
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
            # Clean up temporary file
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
        
        y_pred = pd.DataFrame(
            X_forecast,
            index=forecast_index,
            columns=self._y.columns
        )
        
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

