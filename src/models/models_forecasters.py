"""Sktime-compatible forecasters for DFM and DDFM models.

This module provides minimal sktime-compatible wrappers for evaluation.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Union
from numpy.linalg import LinAlgError
from src.utils import ValidationError

try:
    from dfm_python import DFM as DFMBase, DDFM as DDFMBase
except ImportError:
    DFMBase = None
    DDFMBase = None

# Helper functions are defined in this file

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
    
    # Validate model parameters after training to detect numerical instability
    import numpy as np
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        result = model.get_result()
        if result is not None:
            # Check A matrix (factor transition matrix) for extreme values
            if hasattr(result, 'A') and result.A is not None:
                A = np.array(result.A)
                if np.any(~np.isfinite(A)):
                    nan_count = np.sum(~np.isfinite(A))
                    logger.warning(
                        f"DFM training: A matrix contains {nan_count} non-finite values (NaN/Inf). "
                        f"This indicates numerical instability during training."
                    )
                else:
                    A_max = np.max(np.abs(A))
                    if A_max > 1e6:
                        logger.warning(
                            f"DFM training: A matrix contains extreme values (max abs: {A_max:.2e}). "
                            f"This may indicate numerical instability. Consider checking convergence or regularization."
                        )
            
            # Check C matrix (factor loadings) for extreme values
            if hasattr(result, 'C') and result.C is not None:
                C = np.array(result.C)
                if np.any(~np.isfinite(C)):
                    nan_count = np.sum(~np.isfinite(C))
                    logger.warning(
                        f"DFM training: C matrix contains {nan_count} non-finite values (NaN/Inf). "
                        f"This indicates numerical instability during training."
                    )
                else:
                    C_max = np.max(np.abs(C))
                    if C_max > 1e6:
                        logger.warning(
                            f"DFM training: C matrix contains extreme values (max abs: {C_max:.2e}). "
                            f"This may indicate numerical instability. Consider checking convergence or regularization."
                        )
            
            # Check convergence status
            if hasattr(result, 'converged'):
                if not result.converged:
                    logger.warning(
                        f"DFM training: Model did not converge (max_epochs={max_epochs} reached). "
                        f"This may lead to poor predictions or numerical instability."
                    )
    except Exception as e:
        # Don't fail training if validation fails, just log warning
        logger.debug(f"Could not validate model parameters after training: {type(e).__name__}: {str(e)}")

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
        mixed_freq: Optional[bool] = None,
        **kwargs
    ):
        self.config_dict = config_dict
        self.max_iter = max_iter
        self.threshold = threshold
        # Get mixed_freq from config_dict if not explicitly provided
        if mixed_freq is None and config_dict and isinstance(config_dict, dict):
            mixed_freq = config_dict.get('mixed_freq', False)
        self.mixed_freq = mixed_freq if mixed_freq is not None else False
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
        
        # Get tent_weights_dict from config_dict (legacy support, mixed_freq takes precedence)
        tent_weights = None
        if self.config_dict and isinstance(self.config_dict, dict):
            tent_weights = self.config_dict.get("tent_weights_dict")
        
        self._dfm_model = DFMBase(
            max_iter=self.max_iter,
            threshold=self.threshold,
            tent_weights_dict=tent_weights,  # Deprecated, kept for backward compatibility
            mixed_freq=self.mixed_freq
        )
        
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
    
    def predict(self, fh, history=None):
        """Generate forecasts.
        
        Parameters
        ----------
        fh : int or list
            Forecast horizon
        history : int, optional
            Number of historical periods to use for factor state update before prediction.
            If None, uses training state only (default).
        """
        if not self._is_fitted or self._dfm_model is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        horizon = _convert_horizon(fh)
        
        # Get predictions from DFM model with error handling
        # Pass history parameter to update factor state with recent data
        try:
            if history is not None and history > 0:
                predictions = self._dfm_model.predict(horizon=horizon, history=history)
            else:
                predictions = self._dfm_model.predict(horizon=horizon)
        except (ValueError, RuntimeError, LinAlgError) as e:
            _handle_prediction_error(e, horizon, "DFM")
            raise
        except Exception as e:
            _handle_prediction_error(e, horizon, "DFM")
            raise
        
        # Validate predictions
        from .models_utils import _validate_predictions
        _validate_predictions(predictions, horizon, "DFM")
        
        # Convert to DataFrame with error handling
        try:
            from .models_utils import _convert_predictions_to_dataframe
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
        learning_rate: float = 0.005,  # Updated to match original DDFM default
        batch_size: int = 100,  # Updated to match original DDFM default
        loss_function: str = 'mse',
        huber_delta: float = 1.0,
        activation: str = 'relu',  # Activation function for encoder ('relu', 'tanh', 'sigmoid')
        weight_decay: float = 0.0,  # Weight decay (L2 regularization) for optimizer
        grad_clip_val: float = 1.0,  # Maximum gradient norm for gradient clipping
        factor_order: int = 1,  # VAR lag order for factor dynamics (1 or 2, default: 1)
        mult_epoch_pretrain: int = 1,  # Multiplier for pre-training epochs (default: 1)
        **kwargs
    ):
        self.config_dict = config_dict
        self.encoder_layers = encoder_layers
        self.num_factors = num_factors
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.huber_delta = huber_delta
        self.activation = activation
        self.weight_decay = weight_decay
        self.grad_clip_val = grad_clip_val
        self.factor_order = factor_order
        self.mult_epoch_pretrain = mult_epoch_pretrain
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
            'batch_size': self.batch_size,
            'loss_function': self.loss_function,
            'huber_delta': self.huber_delta,
            'activation': self.activation,
            'weight_decay': self.weight_decay,
            'grad_clip_val': self.grad_clip_val,
            'factor_order': self.factor_order,
            'mult_epoch_pretrain': self.mult_epoch_pretrain
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
    
    def predict(self, fh, history=None):
        """Generate forecasts.
        
        Parameters
        ----------
        fh : int or list
            Forecast horizon
        history : int, optional
            Number of historical periods to use for factor state update before prediction.
            If None, uses training state only (default).
        """
        if not self._is_fitted or self._ddfm_model is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        horizon = _convert_horizon(fh)
        
        # Get predictions with error handling
        # Pass history parameter to update factor state with recent data
        try:
            if history is not None and history > 0:
                predictions = self._ddfm_model.predict(horizon=horizon, history=history)
            else:
                predictions = self._ddfm_model.predict(horizon=horizon)
        except (ValueError, RuntimeError, LinAlgError) as e:
            _handle_prediction_error(e, horizon, "DDFM")
            raise
        except Exception as e:
            _handle_prediction_error(e, horizon, "DDFM")
            raise
        
        # Validate predictions
        from .models_utils import _validate_predictions
        _validate_predictions(predictions, horizon, "DDFM")
        
        # Convert to DataFrame with error handling
        try:
            from .models_utils import _convert_predictions_to_dataframe
            return _convert_predictions_to_dataframe(predictions, self._y, horizon)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"DDFM _convert_predictions_to_dataframe() failed for horizon={horizon}: {type(e).__name__}: {e}")
            raise


