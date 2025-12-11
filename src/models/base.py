"""Base utilities for model training and forecasting."""

from pathlib import Path
from typing import Dict, Any, Tuple
import pickle
import logging

logger = logging.getLogger(__name__)


def save_model_checkpoint(
    model: Any,
    checkpoint_path: Path,
    metadata: Dict[str, Any]
) -> None:
    """Save model checkpoint with metadata.
    
    Supports both 'model' and 'forecaster' keys for backward compatibility.
    Uses dill for models that can't be pickled with standard pickle (e.g., PyTorch Forecasting models).
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use 'forecaster' key for compatibility with existing code
    checkpoint_data = {
        'forecaster': model,
        'model': model,  # Also include 'model' for compatibility
        'metadata': metadata,
        **metadata  # Flatten metadata for easier access
    }
    
    # Try standard pickle first, fall back to dill for complex objects
    # For pytorch-forecasting models, skip pickle and go straight to dill
    use_dill = False
    # Check if model is a pytorch-forecasting model or contains one (e.g., in ForecastingPipeline)
    model_str = str(type(model))
    if 'pytorch_forecasting' in model_str or 'PytorchForecasting' in model_str:
        use_dill = True
        logger.debug("Detected pytorch-forecasting model, using dill for serialization")
    elif hasattr(model, 'steps') or hasattr(model, '_steps'):
        # Check if it's a pipeline (e.g., ForecastingPipeline) containing pytorch-forecasting
        steps = getattr(model, 'steps', None) or getattr(model, '_steps', None)
        if steps:
            for step_name, step_obj in steps:
                step_str = str(type(step_obj))
                if 'pytorch_forecasting' in step_str or 'PytorchForecasting' in step_str:
                    use_dill = True
                    logger.debug(f"Detected pytorch-forecasting model in pipeline step '{step_name}', using dill for serialization")
                    break
    
    if use_dill:
        # Use dill directly for pytorch-forecasting models
        try:
            import dill
            with open(checkpoint_path, 'wb') as f:
                dill.dump(checkpoint_data, f)
            logger.info(f"Saved checkpoint using dill to {checkpoint_path}")
        except ImportError:
            raise RuntimeError(
                "dill is required for pytorch-forecasting models. "
                "Install with: pip install dill"
            )
        except Exception as e:
            # If dill fails, try joblib as last resort
            try:
                import joblib
                with open(checkpoint_path, 'wb') as f:
                    joblib.dump(checkpoint_data, f)
                logger.info(f"Saved checkpoint using joblib to {checkpoint_path}")
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to save pytorch-forecasting model checkpoint. "
                    f"Dill failed: {e}, Joblib failed: {e2}. "
                    "Try installing dill: pip install dill"
                ) from e2
    else:
        # Standard pickle for other models
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
        except (AttributeError, pickle.PicklingError) as e:
            # Some models can't be pickled with standard pickle
            # Try dill which handles more complex objects
            try:
                import dill
                with open(checkpoint_path, 'wb') as f:
                    dill.dump(checkpoint_data, f)
                logger.debug(f"Used dill to save checkpoint (standard pickle failed: {e})")
            except ImportError:
                # If dill is not available, try joblib
                try:
                    import joblib
                    with open(checkpoint_path, 'wb') as f:
                        joblib.dump(checkpoint_data, f)
                    logger.debug(f"Used joblib to save checkpoint (standard pickle failed: {e})")
                except ImportError:
                    # If both fail, re-raise the original error
                    raise RuntimeError(
                        f"Failed to save model checkpoint. Standard pickle failed: {e}. "
                        "Install 'dill' or 'joblib' for better support of complex models: "
                        "pip install dill joblib"
                    ) from e
    
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def load_model_checkpoint(checkpoint_path: Path) -> Tuple[Any, Dict[str, Any]]:
    """Load model checkpoint with metadata.
    
    Supports both sktime's load_from_path (for .zip files) and custom pickle/dill/joblib formats.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Check if this is a sktime .zip file
    if checkpoint_path.suffix == '.zip':
        try:
            from sktime.forecasting.base import BaseForecaster
            model = BaseForecaster.load_from_path(str(checkpoint_path))
            logger.info(f"Loaded sktime forecaster from {checkpoint_path}")
            
            # Try to load metadata from separate file
            metadata_path = checkpoint_path.parent / "metadata.pkl"
            metadata = {}
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'rb') as f:
                        metadata = pickle.load(f)
                    logger.info(f"Loaded metadata from {metadata_path}")
                except Exception as e:
                    logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
            
            return model, metadata
        except Exception as e:
            logger.warning(f"Failed to load as sktime checkpoint: {e}, trying custom format")
    
    # Try standard pickle first, fall back to dill/joblib if needed
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
    except (AttributeError, pickle.UnpicklingError, EOFError):
        # Try dill
        try:
            import dill
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = dill.load(f)
        except ImportError:
            # Try joblib
            try:
                import joblib
                with open(checkpoint_path, 'rb') as f:
                    checkpoint_data = joblib.load(f)
            except ImportError:
                raise RuntimeError(
                    "Failed to load checkpoint. Install 'dill' or 'joblib' for better support: "
                    "pip install dill joblib"
                )
    
    # Handle both formats: {'model': ...} and {'forecaster': ...}
    model = checkpoint_data.get('forecaster') or checkpoint_data.get('model')
    
    # Extract metadata - could be in 'metadata' key or flattened
    if 'metadata' in checkpoint_data:
        metadata = checkpoint_data['metadata']
    else:
        # Extract metadata from flattened keys (exclude model keys)
        metadata = {k: v for k, v in checkpoint_data.items() 
                   if k not in ['forecaster', 'model']}
    
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return model, metadata

