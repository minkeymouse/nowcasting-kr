"""Model service for loading models (simplified - inference removed)."""

import pickle
from pathlib import Path
from typing import Dict, Optional, Any

from app.utils import (
    OUTPUTS_DIR,
    ModelNotFoundError,
    TrainingError,
    ValidationError,
    MODEL_PKL_FILE,
    validate_model_file,
    validate_model_structure
)
from .registry import ModelRegistry


class ModelService:
    """Service for model operations (simplified)."""
    
    def __init__(self):
        self._registry = ModelRegistry()
    
    def load_model(self, model_name: str) -> Dict[str, Any]:
        """Load a trained model.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Dictionary containing model, result, config, and time
            
        Raises:
            ModelNotFoundError: If model is not in registry or file doesn't exist
            TrainingError: If model file is corrupted or invalid
        """
        if not model_name:
            raise ModelNotFoundError("Model name cannot be empty")
        
        if not self._registry.model_exists(model_name):
            raise ModelNotFoundError(f"Model '{model_name}' not found in registry")
        
        # src.training.train saves to outputs/models/{model_name}/
        model_path = OUTPUTS_DIR / "models" / model_name / MODEL_PKL_FILE
        if not model_path.exists():
            # Fallback to old location (for backward compatibility)
            model_path = OUTPUTS_DIR / model_name / MODEL_PKL_FILE
        
        validate_model_file(model_path)
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            validate_model_structure(model_data)
            return model_data
            
        except pickle.UnpicklingError as e:
            raise TrainingError(f"Failed to unpickle model file: {str(e)}") from e
        except (IOError, OSError) as e:
            raise TrainingError(f"File I/O error loading model: {str(e)}") from e
        except ValidationError as e:
            raise TrainingError(str(e)) from e
        except Exception as e:
            raise TrainingError(f"Unexpected error loading model: {str(e)}") from e

