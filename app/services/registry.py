"""Model registry service for managing trained models."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from utils import REGISTRY_PATH


class ModelRegistry:
    """Manages the model registry JSON file."""
    
    def __init__(self, registry_path: Path = REGISTRY_PATH):
        self.registry_path = registry_path
        self._ensure_registry_exists()
    
    def _ensure_registry_exists(self):
        """Create registry file if it doesn't exist."""
        if not self.registry_path.exists():
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.registry_path, 'w') as f:
                json.dump({"models": []}, f, indent=2)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        with open(self.registry_path, 'r') as f:
            registry = json.load(f)
        return registry.get("models", [])
    
    def register_model(
        self,
        model_name: str,
        timestamp: str,
        config_path: str,
        model_type: str
    ):
        """Register a new model in the registry."""
        with open(self.registry_path, 'r') as f:
            registry = json.load(f)
        
        model_info = {
            "model_name": model_name,
            "timestamp": timestamp,
            "config_path": config_path,
            "model_type": model_type
        }
        
        registry["models"].append(model_info)
        
        with open(self.registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
    
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
        counts = {"dfm": 0, "ddfm": 0}
        for model in models:
            model_type = model.get("model_type", "dfm")
            if model_type in counts:
                counts[model_type] += 1
        return counts
    
    def get_experiment_usage(self) -> Dict[str, int]:
        """Get experiment usage statistics."""
        models = self.list_models()
        usage = {}
        for model in models:
            config_path = model.get("config_path", "")
            if "experiment/" in config_path:
                exp_id = config_path.split("experiment/")[1].split(".yaml")[0]
                usage[exp_id] = usage.get(exp_id, 0) + 1
            else:
                usage["default"] = usage.get("default", 0) + 1
        return usage

