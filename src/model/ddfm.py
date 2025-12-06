"""DDFM model wrapper with metadata tracking and outputs/ structure support."""

from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime

# Path setup is handled in entry points (train.py, infer.py)

# Import custom exceptions for error handling
from ..utils.config_parser import ValidationError

try:
    from dfm_python import DDFM as DDFMBase
    from dfm_python.config.results import DFMResult
    from dfm_python.config import DFMConfig
    from dfm_python.lightning import DFMDataModule
except ImportError:
    DDFMBase = None
    DFMResult = None
    DFMConfig = None
    DFMDataModule = None

from ..preprocess.utils import create_transformer_from_config

# Import shared utilities
from .sktime_forecaster import (
    load_config,
    validate_data_module_requirements,
    create_standard_error_message,
    create_data_module,
    train_model,
    save_to_outputs
)


class DDFM:
    """Wrapper around dfm-python DDFM with metadata tracking and outputs/ structure support.
    
    This wrapper provides:
    - Metadata tracking (creation time, training metrics, etc.)
    - save_to_outputs() method for saving to outputs/{model_name}/ structure
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
        
        # Use shared training implementation
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
            logs/
            plots/
            results/
        
        Args:
            model_name: Name of the model
            outputs_dir: Base outputs directory (Path object)
            config_path: Optional path to config file to copy
            
        Returns:
            Path to the model directory
        """
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
        return self._model.nowcast

