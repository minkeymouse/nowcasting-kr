"""Model service for loading models and running inference."""

import pickle
from pathlib import Path
from typing import Dict, Optional, Any

from utils import OUTPUTS_DIR, ModelNotFoundError, TrainingError
from .registry import ModelRegistry


class ModelService:
    """Service for model operations."""
    
    def __init__(self):
        self._registry = ModelRegistry()
    
    def load_model(self, model_name: str):
        """Load a trained model."""
        if not self._registry.model_exists(model_name):
            raise ModelNotFoundError(f"Model not found: {model_name}")
        
        model_path = OUTPUTS_DIR / model_name / "model.pkl"
        if not model_path.exists():
            raise ModelNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def save_model(self, model: Any, model_name: str):
        """Save a model."""
        model_dir = OUTPUTS_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        with open(model_dir / "model.pkl", 'wb') as f:
            pickle.dump(model, f)
    
    def run_inference(
        self,
        model_name: str,
        target_series: str,
        view_date: str,
        target_period: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run inference (nowcast) on a trained model."""
        model_data = self.load_model(model_name)
        model = model_data["model"]
        
        # Get nowcast
        try:
            nowcast = model.nowcast
        except Exception as e:
            raise TrainingError(f"Failed to get nowcast manager: {str(e)}") from e
        
        try:
            if target_period:
                result = nowcast(
                    target_series,
                    view_date=view_date,
                    target_period=target_period,
                    return_result=True
                )
            else:
                result = nowcast(
                    target_series,
                    view_date=view_date,
                    return_result=True
                )
            
            # Handle different result types
            if hasattr(result, 'nowcast_value'):
                nowcast_value = result.nowcast_value
            elif isinstance(result, (int, float)):
                nowcast_value = float(result)
            else:
                nowcast_value = float(result) if result else 0.0
            
            target_period_str = None
            if hasattr(result, 'target_period'):
                target_period_str = result.target_period.isoformat() if hasattr(result.target_period, 'isoformat') else str(result.target_period)
            elif target_period:
                target_period_str = target_period
            
            view_date_str = None
            if hasattr(result, 'view_date'):
                view_date_str = result.view_date.isoformat() if hasattr(result.view_date, 'isoformat') else str(result.view_date)
            else:
                view_date_str = view_date
            
            return {
                "nowcast_value": nowcast_value,
                "target_series": target_series,
                "target_period": target_period_str,
                "view_date": view_date_str,
                "data_availability": getattr(result, 'data_availability', None),
                "factors_at_view": result.factors_at_view.tolist() if hasattr(result, 'factors_at_view') and result.factors_at_view is not None else None
            }
        except Exception as e:
            raise TrainingError(f"Inference failed: {str(e)}") from e

