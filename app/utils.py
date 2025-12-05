"""Utility functions and constants for the nowcasting app."""

from pathlib import Path
from typing import Optional, Type, TypeVar, Dict, Any, List
from datetime import datetime
from enum import Enum

T = TypeVar('T')


# ===== Directory Constants =====
OUTPUTS_DIR = Path("outputs")
CONFIG_DIR = Path("config")
REGISTRY_PATH = OUTPUTS_DIR / "models_registry.json"
DATA_DIR = Path("app/data")
DEFAULT_MODEL_NAME_FORMAT = "%Y%m%d_%H%M%S"


# ===== Model Type Constants =====
class ModelType(str, Enum):
    """Model type enumeration."""
    DFM = "dfm"
    DDFM = "ddfm"


# ===== Job Status Constants =====
class JobStatus(str, Enum):
    """Job status enumeration."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ===== File Name Constants =====
MODEL_PKL_FILE = "model.pkl"
CONFIG_YAML_FILE = "config.yaml"
TRAINING_LOG_FILE = "training.log"
METRICS_JSON_FILE = "metrics.json"
ERRORS_LOG_FILE = "errors.log"


# ===== DDFM Default Parameters =====
DEFAULT_DDFM_ENCODER_LAYERS = [64, 32]
DEFAULT_DDFM_NUM_FACTORS = 1
DEFAULT_DDFM_EPOCHS = 100

# ===== Directory Name Constants =====
LOGS_DIR_NAME = "logs"
PLOTS_DIR_NAME = "plots"
RESULTS_DIR_NAME = "results"

# ===== Model Data Structure Constants =====
REQUIRED_MODEL_KEYS = ["model", "result", "config"]

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


class AgentError(CustomException):
    """Raised when agent-related operations fail."""
    pass


def format_error_message(
    operation: str,
    reason: str,
    suggestion: Optional[str] = None
) -> str:
    """Format error message consistently.
    
    Creates standardized error messages with the format:
    "{operation} failed: {reason}" with optional suggestion.
    
    Args:
        operation: Description of the operation that failed (e.g., "Training", "Config loading")
        reason: Why the operation failed (e.g., "Model type mismatch", "File not found")
        suggestion: Optional suggestion for how to fix the issue
        
    Returns:
        Formatted error message string
        
    Examples:
        >>> format_error_message("Training", "Model type mismatch", "Use correct model type")
        "Training failed: Model type mismatch. Suggestion: Use correct model type"
        
        >>> format_error_message("Config loading", "File not found")
        "Config loading failed: File not found"
    """
    message = f"{operation} failed: {reason}"
    if suggestion:
        message += f". Suggestion: {suggestion}"
    return message


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


# ===== File Validation Utilities =====

def validate_file_exists(
    file_path: Path,
    error_class: Type[Exception],
    error_message: str
) -> Path:
    """Validate that a file exists.
    
    Args:
        file_path: Path to the file to validate
        error_class: Exception class to raise if file doesn't exist
        error_message: Error message to include in exception
        
    Returns:
        Path object (for chaining)
        
    Raises:
        error_class: If file doesn't exist
    """
    if not file_path.exists():
        raise error_class(error_message)
    return file_path


def validate_config_file(file_path: Path) -> Path:
    """Validate that a config file exists.
    
    Args:
        file_path: Path to the config file
        
    Returns:
        Path object (for chaining)
        
    Raises:
        ConfigError: If file doesn't exist
    """
    return validate_file_exists(
        file_path,
        ConfigError,
        f"Config file not found: {file_path}"
    )


def validate_model_file(file_path: Path) -> Path:
    """Validate that a model file exists.
    
    Args:
        file_path: Path to the model file
        
    Returns:
        Path object (for chaining)
        
    Raises:
        ModelNotFoundError: If file doesn't exist
    """
    return validate_file_exists(
        file_path,
        ModelNotFoundError,
        f"Model file not found: {file_path}"
    )


def validate_data_file(file_path: Path) -> Path:
    """Validate that a data file exists.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        Path object (for chaining)
        
    Raises:
        ValidationError: If file doesn't exist
    """
    return validate_file_exists(
        file_path,
        ValidationError,
        f"Data file not found: {file_path}"
    )


def validate_model_structure(
    model_data: Dict[str, Any],
    required_keys: Optional[List[str]] = None
) -> None:
    """Validate model data structure.
    
    Args:
        model_data: Model data dictionary to validate
        required_keys: List of required keys. If None, uses REQUIRED_MODEL_KEYS
        
    Raises:
        ValidationError: If model data is not a dict or missing required keys
    """
    if not isinstance(model_data, dict):
        raise ValidationError(f"Invalid model file format: expected dict, got {type(model_data)}")
    
    if required_keys is None:
        required_keys = REQUIRED_MODEL_KEYS
    
    missing_keys = [key for key in required_keys if key not in model_data]
    if missing_keys:
        raise ValidationError(f"Model file missing required keys: {missing_keys}")


def load_model_file(model_path: Path) -> Dict[str, Any]:
    """Load and validate model file.
    
    Loads a model from pickle file, validates file existence and structure.
    
    Args:
        model_path: Path to model.pkl file
        
    Returns:
        Dictionary containing validated model data
        
    Raises:
        ModelNotFoundError: If file doesn't exist or can't be read
        ValidationError: If model structure is invalid
    """
    import pickle
    
    validate_model_file(model_path)
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
    except (IOError, OSError) as e:
        raise ModelNotFoundError(f"Failed to read model file {model_path}: {str(e)}") from e
    except pickle.UnpicklingError as e:
        raise ModelNotFoundError(f"Failed to unpickle model file {model_path}: {str(e)}") from e
    except Exception as e:
        raise ModelNotFoundError(f"Unexpected error loading model file {model_path}: {str(e)}") from e
    
    validate_model_structure(model_data)
    
    return model_data


# ===== File Upload Utilities =====

def validate_csv_file(filename: Optional[str]) -> None:
    """Validate that filename is a CSV file.
    
    Args:
        filename: Filename to validate
        
    Raises:
        ValidationError: If filename is not a CSV file
    """
    if not filename or not filename.endswith('.csv'):
        raise ValidationError("File must be a CSV file")


def ensure_csv_extension(filename: str) -> str:
    """Ensure filename has .csv extension.
    
    Args:
        filename: Filename to check
        
    Returns:
        Filename with .csv extension
    """
    return filename if filename.endswith('.csv') else f"{filename}.csv"


def validate_csv_columns(file_path: Path, required_columns: List[str]) -> None:
    """Validate that CSV file has required columns.
    
    Args:
        file_path: Path to CSV file
        required_columns: List of required column names
        
    Raises:
        ValidationError: If required columns are missing
    """
    import pandas as pd
    
    try:
        df = pd.read_csv(file_path, nrows=1)
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            available = ', '.join(df.columns)
            raise ValidationError(
                f"Missing required columns: {', '.join(missing_cols)}. "
                f"Available columns: {available}"
            )
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(f"Failed to validate CSV: {str(e)}") from e

