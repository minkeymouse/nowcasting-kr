"""Utility functions and constants for the nowcasting app."""

from pathlib import Path
from typing import Optional
from datetime import datetime

# Constants
OUTPUTS_DIR = Path("outputs")
CONFIG_DIR = Path("config")
REGISTRY_PATH = OUTPUTS_DIR / "models_registry.json"
DATA_DIR = Path("app/data")
DEFAULT_MODEL_NAME_FORMAT = "%Y%m%d_%H%M%S"

# Ensure directories exist
OUTPUTS_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


class CustomException(Exception):
    """Base exception class for the application."""
    pass


class TrainingError(CustomException):
    """Raised when training fails."""
    pass


class ModelNotFoundError(CustomException):
    """Raised when a model is not found."""
    pass


class ConfigError(CustomException):
    """Raised when there's a configuration error."""
    pass


class ValidationError(CustomException):
    """Raised when validation fails."""
    pass


def generate_model_name(prefix: Optional[str] = None) -> str:
    """Generate a model name with timestamp.
    
    Args:
        prefix: Optional prefix for the model name
        
    Returns:
        Model name string
    """
    timestamp = datetime.now().strftime(DEFAULT_MODEL_NAME_FORMAT)
    if prefix:
        return f"{prefix}_{timestamp}"
    return f"model_{timestamp}"

