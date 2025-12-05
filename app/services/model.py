"""Model service for loading models and running inference."""

import pickle
from pathlib import Path
from typing import Dict, Optional, Any

from app.utils import (
    OUTPUTS_DIR,
    ModelNotFoundError,
    TrainingError,
    ValidationError,
    MODEL_PKL_FILE,
    CONFIG_YAML_FILE,
    validate_model_file,
    validate_model_structure
)
from .registry import ModelRegistry


class ModelService:
    """Service for model operations."""
    
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
        
        model_path = OUTPUTS_DIR / model_name / MODEL_PKL_FILE
        validate_model_file(model_path)
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Validate model structure (raises ValidationError, convert to TrainingError)
            try:
                validate_model_structure(model_data)
            except ValidationError as e:
                raise TrainingError(str(e)) from e
            
            return model_data
            
        except pickle.UnpicklingError as e:
            raise TrainingError(f"Failed to unpickle model file: {str(e)}") from e
        except (IOError, OSError) as e:
            raise TrainingError(f"File I/O error loading model: {str(e)}") from e
        except TrainingError:
            # Re-raise TrainingError as-is
            raise
        except Exception as e:
            raise TrainingError(f"Unexpected error loading model: {str(e)}") from e
    
    def save_model(self, model: Any, model_name: str) -> None:
        """Save a model."""
        model_dir = OUTPUTS_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        with open(model_dir / MODEL_PKL_FILE, 'wb') as f:
            pickle.dump(model, f)
    
    def _get_nowcast_manager(self, model: Any, model_name: str) -> Any:
        """Get nowcast manager from model.
        
        Args:
            model: Model instance
            model_name: Name of the model (for error messages)
            
        Returns:
            Nowcast manager instance
            
        Raises:
            TrainingError: If nowcast manager cannot be retrieved
        """
        if not hasattr(model, 'nowcast'):
            raise TrainingError(f"Model '{model_name}' does not have nowcast attribute")
        
        nowcast = model.nowcast
        if nowcast is None:
            raise TrainingError(f"Nowcast manager is None for model '{model_name}'")
        
        return nowcast
    
    def _extract_nowcast_value(self, result: Any) -> float:
        """Extract nowcast value from result object.
        
        Args:
            result: Nowcast result object
            
        Returns:
            Nowcast value as float
        """
        if hasattr(result, 'nowcast_value'):
            return float(result.nowcast_value)
        elif isinstance(result, (int, float)):
            return float(result)
        else:
            return float(result) if result else 0.0
    
    def _format_date(self, date_obj: Any, fallback: str) -> str:
        """Format date object to ISO string.
        
        Args:
            date_obj: Date object (may have isoformat method)
            fallback: Fallback string if formatting fails
            
        Returns:
            ISO formatted date string
        """
        if hasattr(date_obj, 'isoformat'):
            return date_obj.isoformat()
        return str(date_obj) if date_obj else fallback
    
    def run_inference(
        self,
        model_name: str,
        target_series: str,
        view_date: str,
        target_period: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run inference (nowcast) on a trained model.
        
        Args:
            model_name: Name of the trained model
            target_series: Name of the target series to nowcast
            view_date: Date string for the view date
            target_period: Optional target period string
            
        Returns:
            Dictionary with nowcast results
            
        Raises:
            ModelNotFoundError: If model doesn't exist
            TrainingError: If inference fails
        """
        if not target_series:
            raise TrainingError("target_series cannot be empty")
        if not view_date:
            raise TrainingError("view_date cannot be empty")
        
        model_data = self.load_model(model_name)
        model = model_data["model"]
        
        # Get nowcast manager
        try:
            nowcast = self._get_nowcast_manager(model, model_name)
        except AttributeError as e:
            raise TrainingError(f"Model '{model_name}' does not support nowcasting: {str(e)}") from e
        except Exception as e:
            raise TrainingError(f"Failed to get nowcast manager for model '{model_name}': {str(e)}") from e
        
        # Run nowcast
        try:
            nowcast_kwargs = {
                "target_series": target_series,
                "view_date": view_date,
                "return_result": True
            }
            if target_period:
                nowcast_kwargs["target_period"] = target_period
            
            result = nowcast(**nowcast_kwargs)
            
            # Extract values
            nowcast_value = self._extract_nowcast_value(result)
            target_period_str = self._format_date(
                getattr(result, 'target_period', None),
                target_period or ""
            )
            view_date_str = self._format_date(
                getattr(result, 'view_date', None),
                view_date
            )
            
            # Extract factors if available
            factors_at_view = None
            if hasattr(result, 'factors_at_view') and result.factors_at_view is not None:
                try:
                    factors_at_view = result.factors_at_view.tolist()
                except AttributeError:
                    factors_at_view = result.factors_at_view
            
            return {
                "nowcast_value": nowcast_value,
                "target_series": target_series,
                "target_period": target_period_str,
                "view_date": view_date_str,
                "data_availability": getattr(result, 'data_availability', None),
                "factors_at_view": factors_at_view
            }
        except KeyError as e:
            raise TrainingError(f"Inference failed: target series '{target_series}' not found in model: {str(e)}") from e
        except ValueError as e:
            raise TrainingError(f"Inference failed: invalid date or period format: {str(e)}") from e
        except AttributeError as e:
            raise TrainingError(f"Inference failed: missing required attribute: {str(e)}") from e
        except Exception as e:
            raise TrainingError(f"Inference failed for model '{model_name}': {str(e)}") from e

