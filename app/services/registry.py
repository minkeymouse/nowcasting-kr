"""Model registry service for managing trained models."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from app.utils import REGISTRY_PATH, ModelType, TrainingError


class ModelRegistry:
    """Manages the model registry JSON file."""
    
    def __init__(self, registry_path: Path = REGISTRY_PATH):
        self.registry_path = registry_path
        self._ensure_registry_exists()
    
    def _ensure_registry_exists(self):
        """Create registry file if it doesn't exist."""
        if not self.registry_path.exists():
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_json_file({"models": []})
    
    def _read_json_file(self) -> Dict[str, Any]:
        """Read registry JSON file.
        
        Returns:
            Parsed JSON dictionary
            
        Raises:
            TrainingError: If file read fails
        """
        try:
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (IOError, OSError) as e:
            raise TrainingError(f"Failed to read registry file: {str(e)}") from e
        except json.JSONDecodeError as e:
            raise TrainingError(f"Invalid JSON in registry file: {str(e)}") from e
    
    def _write_json_file(self, data: Dict[str, Any]) -> None:
        """Write registry JSON file.
        
        Args:
            data: Dictionary to write
            
        Raises:
            TrainingError: If file write fails
        """
        try:
            with open(self.registry_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except (IOError, OSError) as e:
            raise TrainingError(f"Failed to write registry file: {str(e)}") from e
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        registry = self._read_json_file()
        return registry.get("models", [])
    
    def register_model(
        self,
        model_name: str,
        timestamp: str,
        config_path: str,
        model_type: str,
        training_metrics: Optional[Dict[str, Any]] = None,
        training_duration_seconds: Optional[float] = None,
        model_pkl_path: Optional[str] = None,
        config_yaml_path: Optional[str] = None,
        experiment_id: Optional[str] = None
    ):
        """Register a new model in the registry.
        
        Args:
            model_name: Name of the model
            timestamp: ISO format timestamp when model was created
            config_path: Original path to config file used for training
            model_type: Type of model ("dfm" or "ddfm")
            training_metrics: Optional dictionary with training metrics (converged, num_iter, loglik)
            training_duration_seconds: Optional training duration in seconds
            model_pkl_path: Optional path to saved model.pkl file
            config_yaml_path: Optional path to saved config.yaml file
            experiment_id: Optional experiment ID. If not provided, extracted from config_path for backward compatibility.
        """
        registry = self._read_json_file()
        
        # Extract experiment_id from config_path if not provided (backward compatibility)
        if experiment_id is None:
            if "experiment/" in config_path:
                # Extract from path like "config/experiment/exp1.yaml"
                experiment_id = config_path.split("experiment/")[1].split(".yaml")[0]
            elif "default.yaml" in config_path or config_path.endswith("default.yaml"):
                experiment_id = "default"
            else:
                # Fallback: try to extract from any path structure
                config_path_obj = Path(config_path)
                if config_path_obj.parent.name == "experiment":
                    experiment_id = config_path_obj.stem
                else:
                    experiment_id = None
        
        model_info = {
            "model_name": model_name,
            "timestamp": timestamp,
            "config_path": config_path,
            "model_type": model_type
        }
        
        # Add experiment_id if available
        if experiment_id is not None:
            model_info["experiment_id"] = experiment_id
        
        # Add training metrics if provided
        if training_metrics:
            model_info["training_metrics"] = training_metrics
        
        # Add training duration if provided
        if training_duration_seconds is not None:
            model_info["training_duration_seconds"] = training_duration_seconds
        
        # Add file paths if provided
        if model_pkl_path:
            model_info["model_pkl_path"] = model_pkl_path
        if config_yaml_path:
            model_info["config_yaml_path"] = config_yaml_path
        
        registry["models"].append(model_info)
        self._write_json_file(registry)
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        models = self.list_models()
        for model in models:
            if model["model_name"] == model_name:
                return model
        return None
    
    def model_exists(self, model_name: str) -> bool:
        """Check if a model exists in the registry."""
        return self.get_model_info(model_name) is not None
    
    def get_model_counts_by_type(self) -> Dict[str, int]:
        """Get count of models by type."""
        models = self.list_models()
        counts = {ModelType.DFM: 0, ModelType.DDFM: 0}
        for model in models:
            model_type = model.get("model_type", ModelType.DFM)
            if model_type in counts:
                counts[model_type] += 1
        return counts
    
    def get_experiment_usage(self) -> Dict[str, int]:
        """Get experiment usage statistics."""
        models = self.list_models()
        usage = {}
        for model in models:
            # Use experiment_id if available (new models), fallback to config_path parsing for backward compatibility
            experiment_id = model.get("experiment_id")
            if experiment_id:
                usage[experiment_id] = usage.get(experiment_id, 0) + 1
            else:
                # Backward compatibility: extract from config_path
                config_path = model.get("config_path", "")
                if "experiment/" in config_path:
                    exp_id = config_path.split("experiment/")[1].split(".yaml")[0]
                    usage[exp_id] = usage.get(exp_id, 0) + 1
                else:
                    usage["default"] = usage.get("default", 0) + 1
        return usage

