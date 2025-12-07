"""DFM and DDFM model wrappers - direct interface to dfm-python.

This module provides thin wrappers around dfm-python's DFM and DDFM classes
for metadata tracking and checkpoint saving. The wrappers use dfm-python's
native interface directly without sktime compatibility layers.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple, Callable
from datetime import datetime

# Path setup is handled in entry points (train.py, infer.py)

# Import custom exceptions for error handling
from src.utils import ValidationError

try:
    from dfm_python import DFM as DFMBase, DDFM as DDFMBase
    from dfm_python.config.results import DFMResult
    from dfm_python.config import DFMConfig
    from dfm_python.lightning import DFMDataModule
except ImportError:
    DFMBase = None
    DDFMBase = None
    DFMResult = None
    DFMConfig = None
    DFMDataModule = None

# Shared utilities from sktime_forecaster are imported inside methods to avoid circular imports


class DFM:
    """Wrapper around dfm-python DFM with metadata tracking and checkpoint/ structure support.
    
    This wrapper provides:
    - Metadata tracking (creation time, training metrics, etc.)
    - save_to_outputs() method for saving to checkpoint/{model_name}/ structure
    - All original dfm-python DFM functionality
    """
    
    def __init__(self):
        """Initialize DFM wrapper."""
        if DFMBase is None:
            raise ImportError("dfm-python package not available")
        self._model = DFMBase()
        self._metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "model_type": "dfm"
        }
    
    def load_config(self, source: Any = None, *, yaml: Optional[Union[str, Path]] = None, 
                    mapping: Optional[Dict[str, Any]] = None, hydra: Optional[Any] = None):
        """Load configuration from various sources.
        
        Args:
            source: Configuration source (YAML path, dict, DFMConfig, or ConfigSource)
            yaml: Path to YAML config file
            mapping: Dictionary config
            hydra: Hydra DictConfig object
            
        Note: If multiple keyword arguments are provided, only one should be used.
        The underlying model will validate this.
        """
        # Use the load_config helper function (defined later in this file)
        load_config(self._model, self._metadata, source, yaml=yaml, mapping=mapping, hydra=hydra)
    
    def _create_data_module(self, data_path: str) -> Any:
        """Create DFMDataModule from data_path and config.
        
        Args:
            data_path: Path to data file
            
        Returns:
            DFMDataModule instance ready for training
            
        Raises:
            ValidationError: If config is not loaded or transformer creation fails
            ImportError: If required dependencies are not available
        """
        # Use the create_data_module helper function (defined later in this file)
        return create_data_module(
            model=self,
            data_path=data_path,
            dfm_data_module=DFMDataModule,
            create_transformer_func=create_transformer_from_config
        )
    
    def train(
        self,
        data_module: Optional[Any] = None,
        data_path: Optional[str] = None,
        **kwargs
    ):
        """Train the model.
        
        Args:
            data_module: DFMDataModule instance (preferred). If None, data_path must be provided.
            data_path: Path to data file. Used only if data_module is None.
            **kwargs: Additional training parameters:
                - max_iter or max_epochs: Maximum number of EM iterations (default: 100)
                - threshold: Convergence threshold for DFM (passed to model, default: 1e-5)
                - enable_progress_bar: Show training progress (default: True)
                - enable_model_summary: Print model summary (default: False)
            
        Raises:
            ValidationError: If neither data_module nor data_path is provided
            ImportError: If required dependencies are not available
        """
        # Import DFMTrainer for type hint
        try:
            from dfm_python import DFMTrainer
        except ImportError:
            DFMTrainer = None
        
        # Use local train_model function (now in same file)
        result = train_model(
            model_wrapper=self,
            model_instance=self._model,
            metadata=self._metadata,
            data_module=data_module,
            data_path=data_path,
            create_data_module_method=self._create_data_module,
            trainer_class=DFMTrainer,
            **kwargs
        )
        
        # Add DFM-specific training metadata
        # DFM results always have these attributes
        self._metadata["converged"] = result.converged
        self._metadata["num_iter"] = result.num_iter
        self._metadata["loglik"] = float(result.loglik)
    
    def predict(self, horizon: Optional[int] = None) -> Any:
        """Generate forecasts using the trained model.
        
        Args:
            horizon: Number of periods ahead to forecast. If None, uses default
                    horizon from model configuration.
            
        Returns:
            Forecast results (type depends on underlying dfm-python implementation).
            Typically returns an array or DataFrame with forecasted values.
            
        Raises:
            RuntimeError: If model has not been trained yet. Call train() first.
        """
        return self._model.predict(horizon=horizon)
    
    def get_result(self) -> Any:
        """Get model training result.
        
        Returns:
            DFMResult object containing:
            - Factor estimates (Z)
            - Loadings (Lambda)
            - Convergence status (converged)
            - Number of iterations (num_iter)
            - Log-likelihood (loglik)
            - Other model-specific results
            
        Raises:
            RuntimeError: If model has not been trained yet. Call train() first.
        """
        return self._model.get_result()
    
    def get_config(self) -> Any:
        """Get model configuration.
        
        Returns:
            DFMConfig object containing all model configuration parameters
            (blocks, series, factors, etc.).
            
        Raises:
            RuntimeError: If config has not been loaded yet. Call load_config() first.
        """
        # dfm-python uses config attribute, not get_config method
        if hasattr(self._model, 'config') and self._model.config is not None:
            return self._model.config
        elif hasattr(self._model, '_config') and self._model._config is not None:
            return self._model._config
        elif hasattr(self._model, 'get_config'):
            return self._model.get_config()
        else:
            raise RuntimeError("Config not loaded. Call load_config() first.")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata.
        
        Returns:
            Dictionary containing model metadata:
            - created_at: ISO timestamp of model creation
            - model_type: "dfm" or "ddfm"
            - training_completed: ISO timestamp when training finished (if trained)
            - converged: Whether training converged (if trained)
            - num_iter: Number of training iterations (if trained)
            - loglik: Final log-likelihood value (if trained)
            - data_path: Path to training data (if provided)
            - Other model-specific metadata
            
        Note:
            Returns a copy of the metadata dictionary to prevent external modifications.
        """
        return self._metadata.copy()
    
    def get_time(self) -> Any:
        """Get time index from the model.
        
        Returns:
            Time index array or Series (type depends on underlying dfm-python
            implementation). Represents the time periods corresponding to the
            model's data.
            
        Raises:
            RuntimeError: If model has not been trained yet or time index is not available.
        """
        return self._model.get_time()
    
    def save_to_outputs(
        self,
        model_name: str,
        outputs_dir: Path,
        config_path: Optional[str] = None
    ) -> Path:
        """Save model to outputs/{model_name}/ structure.
        
        Creates the following structure:
        outputs/
          {model_name}/
            model.pkl
            config.yaml (if config_path provided)
        
        Args:
            model_name: Name of the model
            outputs_dir: Base checkpoint directory (Path object)
            config_path: Optional path to config file to copy
            
        Returns:
            Path to the model directory
        """
        # Use local save_to_outputs function (now in same file)
        return save_to_outputs(
            model_wrapper=self,
            model_name=model_name,
            outputs_dir=outputs_dir,
            config_path=config_path
        )
    
    @property
    def nowcast(self) -> Any:
        """Get nowcast manager for generating nowcasts.
        
        Returns:
            Nowcast manager object from dfm-python that provides methods for:
            - Generating nowcasts (forecasts for current period)
            - Computing revision impacts
            - Analyzing data releases
            
        Raises:
            RuntimeError: If model has not been trained yet. Call train() first.
        """
        if self._model._result is None:
            raise RuntimeError("Model must be trained before accessing nowcast")
        from src.nowcasting import Nowcast
        # Note: data_module is not stored, pass None - Nowcast will handle it
        return Nowcast(self._model, data_module=None)


class DDFM:
    """Wrapper around dfm-python DDFM with metadata tracking and checkpoint/ structure support.
    
    This wrapper provides:
    - Metadata tracking (creation time, training metrics, etc.)
    - save_to_outputs() method for saving to checkpoint/{model_name}/ structure
    - All original dfm-python DDFM functionality
    """
    
    def __init__(
        self,
        encoder_layers: Optional[List[int]] = None,
        num_factors: Optional[int] = None,
        **kwargs
    ):
        """Initialize DDFM wrapper.
        
        Args:
            encoder_layers: List of integers specifying the size of each encoder layer.
                          Defaults to [64, 32] if not provided.
                          Example: [128, 64, 32] creates a 3-layer encoder.
            num_factors: Number of latent factors to extract. Defaults to 1 if not provided.
            **kwargs: Additional arguments passed to underlying DDFMBase model.
            
        Raises:
            ImportError: If dfm-python[deep] package is not installed (PyTorch required).
        """
        if DDFMBase is None:
            raise ImportError("DDFM requires PyTorch. Install with: pip install dfm-python[deep]")
        
        # Initialize underlying DDFM model with encoder architecture and factor count
        self._model = DDFMBase(
            encoder_layers=encoder_layers or [64, 32],
            num_factors=num_factors or 1,
            **kwargs
        )
        # Store metadata including architecture parameters for later reference
        self._metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "model_type": "ddfm",
            "encoder_layers": encoder_layers or [64, 32],
            "num_factors": num_factors or 1
        }
    
    def load_config(self, source: Any = None, *, yaml: Optional[Union[str, Path]] = None, 
                    mapping: Optional[Dict[str, Any]] = None, hydra: Optional[Any] = None):
        """Load configuration from various sources.
        
        Args:
            source: Configuration source (YAML path, dict, DFMConfig, or ConfigSource)
            yaml: Path to YAML config file
            mapping: Dictionary config
            hydra: Hydra DictConfig object
            
        Note: If multiple keyword arguments are provided, only one should be used.
        The underlying model will validate this.
        """
        # Use the load_config helper function (defined later in this file)
        load_config(self._model, self._metadata, source, yaml=yaml, mapping=mapping, hydra=hydra)
    
    def _create_data_module(self, data_path: str) -> Any:
        """Create DFMDataModule from data_path and config.
        
        Args:
            data_path: Path to data file
            
        Returns:
            DFMDataModule instance ready for training
            
        Raises:
            ValidationError: If config is not loaded or transformer creation fails
            ImportError: If required dependencies are not available
        """
        # Use the create_data_module helper function (defined later in this file)
        return create_data_module(
            model=self,
            data_path=data_path,
            dfm_data_module=DFMDataModule,
            create_transformer_func=create_transformer_from_config
        )
    
    def train(
        self,
        data_module: Optional[Any] = None,
        data_path: Optional[str] = None,
        **kwargs
    ):
        """Train the model.
        
        Args:
            data_module: DFMDataModule instance (preferred). If None, data_path must be provided.
            data_path: Path to data file. Used only if data_module is None.
            **kwargs: Additional training parameters:
                - epochs or max_epochs: Maximum number of training epochs (default: 100)
                - learning_rate: Learning rate for optimizer (passed to model)
                - batch_size: Batch size for training (passed to data module)
                - enable_progress_bar: Show training progress (default: True)
                - enable_model_summary: Print model summary (default: True)
            
        Raises:
            ValidationError: If neither data_module nor data_path is provided
            ImportError: If required dependencies are not available
        """
        # Import DDFMTrainer for type hint
        try:
            from dfm_python import DDFMTrainer
        except ImportError:
            DDFMTrainer = None
        
        # Use local train_model function (now in same file)
        result = train_model(
            model_wrapper=self,
            model_instance=self._model,
            metadata=self._metadata,
            data_module=data_module,
            data_path=data_path,
            create_data_module_method=self._create_data_module,
            trainer_class=DDFMTrainer,
            **kwargs
        )
        
        # Add DDFM-specific training metadata
        # DDFM results may not have all attributes, so check before accessing
        import numpy as np
        if hasattr(result, 'converged'):
            self._metadata["converged"] = result.converged
        if hasattr(result, 'num_iter'):
            self._metadata["num_iter"] = result.num_iter
        if hasattr(result, 'loglik'):
            self._metadata["loglik"] = float(result.loglik) if result.loglik is not None else np.nan
    
    def predict(self, horizon: Optional[int] = None) -> Any:
        """Generate forecasts using the trained model.
        
        Args:
            horizon: Number of periods ahead to forecast. If None, uses default
                    horizon from model configuration.
            
        Returns:
            Forecast results (type depends on underlying dfm-python implementation).
            Typically returns an array or DataFrame with forecasted values.
            
        Raises:
            RuntimeError: If model has not been trained yet. Call train() first.
        """
        return self._model.predict(horizon=horizon)
    
    def get_result(self) -> Any:
        """Get model training result.
        
        Returns:
            DFMResult object containing:
            - Factor estimates (Z)
            - Loadings (Lambda)
            - Convergence status (converged, if available)
            - Number of iterations (num_iter, if available)
            - Log-likelihood (loglik, if available)
            - Other model-specific results
            
        Raises:
            RuntimeError: If model has not been trained yet. Call train() first.
        """
        return self._model.get_result()
    
    def get_config(self) -> Any:
        """Get model configuration.
        
        Returns:
            DFMConfig object containing all model configuration parameters
            (blocks, series, factors, etc.).
            
        Raises:
            RuntimeError: If config has not been loaded yet. Call load_config() first.
        """
        # dfm-python uses config attribute, not get_config method
        if hasattr(self._model, 'config') and self._model.config is not None:
            return self._model.config
        elif hasattr(self._model, '_config') and self._model._config is not None:
            return self._model._config
        elif hasattr(self._model, 'get_config'):
            return self._model.get_config()
        else:
            raise RuntimeError("Config not loaded. Call load_config() first.")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata.
        
        Returns:
            Dictionary containing model metadata:
            - created_at: ISO timestamp of model creation
            - model_type: "dfm" or "ddfm"
            - encoder_layers: Neural encoder layer sizes (DDFM only)
            - num_factors: Number of factors (DDFM only)
            - training_completed: ISO timestamp when training finished (if trained)
            - converged: Whether training converged (if trained and available)
            - num_iter: Number of training iterations (if trained and available)
            - loglik: Final log-likelihood value (if trained and available)
            - data_path: Path to training data (if provided)
            - Other model-specific metadata
            
        Note:
            Returns a copy of the metadata dictionary to prevent external modifications.
        """
        return self._metadata.copy()
    
    def get_time(self) -> Any:
        """Get time index from the model.
        
        Returns:
            Time index array or Series (type depends on underlying dfm-python
            implementation). Represents the time periods corresponding to the
            model's data.
            
        Raises:
            RuntimeError: If model has not been trained yet or time index is not available.
        """
        return self._model.get_time()
    
    def save_to_outputs(
        self,
        model_name: str,
        outputs_dir: Path,
        config_path: Optional[str] = None
    ) -> Path:
        """Save model to outputs/{model_name}/ structure.
        
        Creates the following structure:
        outputs/
          {model_name}/
            model.pkl
            config.yaml (if config_path provided)
        
        Args:
            model_name: Name of the model
            outputs_dir: Base checkpoint directory (Path object)
            config_path: Optional path to config file to copy
            
        Returns:
            Path to the model directory
        """
        # Use local save_to_outputs function (now in same file)
        return save_to_outputs(
            model_wrapper=self,
            model_name=model_name,
            outputs_dir=outputs_dir,
            config_path=config_path
        )
    
    @property
    def nowcast(self) -> Any:
        """Get nowcast manager for generating nowcasts.
        
        Returns:
            Nowcast manager object from dfm-python that provides methods for:
            - Generating nowcasts (forecasts for current period)
            - Computing revision impacts
            - Analyzing data releases
            
        Raises:
            RuntimeError: If model has not been trained yet. Call train() first.
        """
        if self._model._result is None:
            raise RuntimeError("Model must be trained before accessing nowcast")
        from src.nowcasting import Nowcast
        # Note: data_module is not stored, pass None - Nowcast will handle it
        return Nowcast(self._model, data_module=None)


# ============================================================================
# Simple sktime-compatible wrappers for evaluation (minimal interface)
# ============================================================================

import numpy as np
import pandas as pd
# Import LinAlgError for numerical stability error handling
from numpy.linalg import LinAlgError


def _create_preprocessing_pipeline(model_config: Any) -> Any:
    """Create preprocessing pipeline following dfm-python documentation pattern.
    
    This helper function creates a TransformerPipeline with imputation and scaling
    steps, matching the pattern recommended in dfm-python documentation.
    
    Args:
        model_config: Model config object with config property (DFMBase or DDFMBase)
        
    Returns:
        TransformerPipeline instance ready for use with DFMDataModule
    """
    from dfm_python.lightning.scaling import create_scaling_transformer_from_config
    from sktime.transformations.compose import TransformerPipeline
    from sktime.transformations.series.impute import Imputer
    
    # Get scaler from model config (unified scaling for all series)
    scaler = create_scaling_transformer_from_config(model_config)
    
    # Create pipeline matching docs pattern: impute (ffill) -> impute (bfill) -> scale
    pipeline = TransformerPipeline(
        steps=[
            ('impute_ffill', Imputer(method="ffill")),
            ('impute_bfill', Imputer(method="bfill")),
            ('scaler', scaler)
        ]
    )
    
    return pipeline


def _train_factor_model(
    model: Any,
    data: pd.DataFrame,
    trainer_class: Any,
    max_epochs: int
) -> None:
    """Train a factor model (DFM or DDFM) using the standard PyTorch Lightning pattern.
    
    This helper function implements the standard training pattern from dfm-python
    documentation:
    1. Create preprocessing pipeline
    2. Create DFMDataModule with pipeline and data
    3. Setup data module
    4. Create trainer and fit
    
    Args:
        model: DFMBase or DDFMBase model instance (must have config property)
        data: Training data as pandas DataFrame
        trainer_class: DFMTrainer or DDFMTrainer class
        max_epochs: Maximum number of training epochs/iterations
    """
    from dfm_python.lightning import DFMDataModule
    
    # Create preprocessing pipeline
    pipeline = _create_preprocessing_pipeline(model.config)
    
    # Create DataModule with pipeline and data (matching docs pattern)
    data_module = DFMDataModule(
        config=model.config,
        pipeline=pipeline,
        data=data
    )
    data_module.setup()  # Pipeline is applied here
    
    # Create trainer and fit (matching docs pattern)
    trainer = trainer_class(max_epochs=max_epochs, enable_progress_bar=False)
    trainer.fit(model, data_module)

class DFMForecaster:
    """Minimal sktime-compatible wrapper for DFM models (for evaluation only).
    
    This is a thin wrapper that provides fit() and predict() methods compatible
    with evaluate_forecaster(), but uses dfm-python's native interface directly.
    """
    
    def __init__(
        self,
        config_dict: Optional[dict] = None,
        max_iter: int = 5000,
        threshold: float = 1e-5,
        **kwargs
    ):
        self.config_dict = config_dict
        self.max_iter = max_iter
        self.threshold = threshold
        self._dfm_model = None
        self._is_fitted = False
        
    def fit(self, y, X=None, fh=None):
        """Fit the DFM model to training data.
        
        Uses the standard PyTorch Lightning pattern from dfm-python documentation:
        1. Create model and load config
        2. Create preprocessing pipeline
        3. Create DFMDataModule with pipeline and data
        4. Setup data module
        5. Create trainer and fit
        """
        # Initialize DFMBase (from dfm-python) with max_iter and threshold
        if DFMBase is None:
            raise ImportError("dfm-python package not available")
        self._dfm_model = DFMBase(max_iter=self.max_iter, threshold=self.threshold)
        
        if self.config_dict:
            self._dfm_model.load_config(mapping=self.config_dict)
        else:
            raise ValidationError("config_dict must be provided to DFMForecaster")
        
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)
        
        # Use the standard pattern from dfm-python documentation
        from dfm_python import DFMTrainer
        
        # Train using helper function (matches docs pattern)
        _train_factor_model(
            model=self._dfm_model,
            data=y,
            trainer_class=DFMTrainer,
            max_epochs=self.max_iter
        )
        
        self._y = y
        self._is_fitted = True
        return self
    
    def predict(self, fh):
        """Generate forecasts."""
        if not self._is_fitted or self._dfm_model is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        horizon = _convert_horizon(fh)
        
        # Get predictions from DFM model with error handling
        try:
            predictions = self._dfm_model.predict(horizon=horizon)
        except (ValueError, RuntimeError, LinAlgError) as e:
            _handle_prediction_error(e, horizon, "DFM")
            raise
        except Exception as e:
            _handle_prediction_error(e, horizon, "DFM")
            raise
        
        # Validate predictions
        _validate_predictions(predictions, horizon, "DFM")
        
        # Convert to DataFrame with error handling
        try:
            return _convert_predictions_to_dataframe(predictions, self._y, horizon)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"DFM _convert_predictions_to_dataframe() failed for horizon={horizon}: {type(e).__name__}: {e}")
            raise


class DDFMForecaster:
    """Minimal sktime-compatible wrapper for DDFM models (for evaluation only).
    
    This is a thin wrapper that provides fit() and predict() methods compatible
    with evaluate_forecaster(), but uses dfm-python's native interface directly.
    """
    
    def __init__(
        self,
        config_dict: Optional[dict] = None,
        encoder_layers: Optional[list] = None,
        num_factors: Optional[int] = None,
        epochs: int = 100,
        learning_rate: float = 0.0001,
        batch_size: int = 32,
        **kwargs
    ):
        self.config_dict = config_dict
        self.encoder_layers = encoder_layers
        self.num_factors = num_factors
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self._ddfm_model = None
        self._is_fitted = False
        
    def fit(self, y, X=None, fh=None):
        """Fit the DDFM model to training data.
        
        Uses the standard PyTorch Lightning pattern from dfm-python documentation:
        1. Create model and load config
        2. Create preprocessing pipeline
        3. Create DFMDataModule with pipeline and data
        4. Setup data module
        5. Create trainer and fit
        """
        # Create DDFM model with parameters
        ddfm_params = {
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size
        }
        if self.encoder_layers:
            ddfm_params['encoder_layers'] = self.encoder_layers
        if self.num_factors:
            ddfm_params['num_factors'] = self.num_factors
        
        self._ddfm_model = DDFMBase(**ddfm_params)
        
        if self.config_dict:
            self._ddfm_model.load_config(mapping=self.config_dict)
        else:
            raise ValidationError("config_dict must be provided to DDFMForecaster")
        
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)
        
        # Use the standard pattern from dfm-python documentation
        from dfm_python import DDFMTrainer
        
        # Train using helper function (matches docs pattern)
        _train_factor_model(
            model=self._ddfm_model,
            data=y,
            trainer_class=DDFMTrainer,
            max_epochs=self.epochs
        )
        
        self._y = y
        self._is_fitted = True
        return self
    
    def predict(self, fh):
        """Generate forecasts."""
        if not self._is_fitted or self._ddfm_model is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        horizon = _convert_horizon(fh)
        
        # Get predictions with error handling
        try:
            predictions = self._ddfm_model.predict(horizon=horizon)
        except (ValueError, RuntimeError, LinAlgError) as e:
            _handle_prediction_error(e, horizon, "DDFM")
            raise
        except Exception as e:
            _handle_prediction_error(e, horizon, "DDFM")
            raise
        
        # Validate predictions
        _validate_predictions(predictions, horizon, "DDFM")
        
        # Convert to DataFrame with error handling
        try:
            return _convert_predictions_to_dataframe(predictions, self._y, horizon)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"DDFM _convert_predictions_to_dataframe() failed for horizon={horizon}: {type(e).__name__}: {e}")
            raise



# ============================================================================
# Common Utilities for Model Wrappers
# ============================================================================

def _convert_horizon(fh: Union[int, List, np.ndarray]) -> int:
    """Convert forecast horizon to integer.
    
    Args:
        fh: Forecast horizon (int, list, or array)
        
    Returns:
        Integer horizon value
    """
    if isinstance(fh, (list, np.ndarray)):
        return len(fh) if len(fh) > 0 else int(fh[0]) if len(fh) == 1 else 1
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
    
    # Convert to numpy array if needed
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
    # DFMBase has a 'config' property, not 'get_config()' method
    if hasattr(model, 'config'):
        config = model.config
    elif hasattr(model, '_config'):
        config = model._config
    elif hasattr(model, 'get_config'):
        config = model.get_config()
    else:
        config = None
    
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
                # CRITICAL FIX: ColumnEnsembleTransformer may have been fitted with integer column indices,
                # but data has string column names. Try to match column structure.
                # If pipeline uses integer indices, ensure test_data has matching structure
                pipeline.fit_transform(test_data)
            except (ValueError, TypeError, AttributeError, KeyError) as e:
                # Pipeline can't handle the data (likely due to column mismatch)
                # This can happen if pipeline was fitted with integer indices but data has string column names
                # or vice versa. Use passthrough transformer instead.
                # KeyError specifically catches "None of [Index([1], dtype='int64')] are in the [columns]"
                pipeline = None
    except (ValueError, TypeError, AttributeError, ImportError, KeyError):
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
        
        # CRITICAL FIX: Convert pandas Index to TimeIndex object for DFMDataModule
        # DFMDataModule expects TimeIndex object, not pandas Index
        # The nowcast manager uses data_module.time_index which must be a TimeIndex
        # This is critical for nowcasting - the Time index must include all dates up to view_date
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
                # RangeIndex: This is problematic for nowcasting as we need actual dates
                # For nowcasting, we need real dates, so this should not happen
                # But if it does, try to infer dates from data or use a default range
                # This is a fallback - ideally data should have DatetimeIndex
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
            # If TimeIndex import fails or conversion fails, raise error
            # This is critical for nowcasting - we cannot proceed without proper TimeIndex
            raise ValueError(
                f"Failed to create TimeIndex for data_module: {e}. "
                f"This is required for nowcasting operations."
            ) from e
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

