"""Training manager service for handling model training jobs."""

import json
import logging
import uuid
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Type, Union, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from src.model.dfm import DFM
    from src.model.ddfm import DDFM

# Set up paths using centralized utility
# Import directly from project root (app/services/ is in project root, src/ is sibling)
from src.utils.path_setup import setup_paths
setup_paths(include_dfm_python=True, include_src=True)

try:
    from src.model.dfm import DFM
    from src.model.ddfm import DDFM
except ImportError:
    DFM = None
    DDFM = None

from app.utils import (
    OUTPUTS_DIR,
    TrainingError,
    ValidationError,
    ModelType,
    JobStatus,
    MODEL_PKL_FILE,
    CONFIG_YAML_FILE,
    TRAINING_LOG_FILE,
    METRICS_JSON_FILE,
    ERRORS_LOG_FILE,
    DEFAULT_DDFM_ENCODER_LAYERS,
    DEFAULT_DDFM_NUM_FACTORS,
    DEFAULT_DDFM_EPOCHS,
    LOGS_DIR_NAME
)
from .registry import ModelRegistry

# Set up logging
logger = logging.getLogger(__name__)


# Type aliases for better readability
ProgressCallback = Callable[[int, str], None]
"""Type alias for progress logging callback function.
    
Callback signature: (progress: int, message: str) -> None
- progress: Progress value (0-100)
- message: Progress message string
"""

ModelInstance = Union["DFM", "DDFM"]
"""Type alias for model instance (DFM or DDFM wrapper)."""

ModelClass = Union[Type["DFM"], Type["DDFM"]]
"""Type alias for model class (DFM or DDFM class)."""

TrainingMetrics = Dict[str, Any]
"""Type alias for training metrics dictionary.
    
Structure:
- converged: bool - Whether training converged
- num_iter: int - Number of iterations
- loglik: float - Log-likelihood value
- training_completed: Optional[str] - ISO timestamp of completion
- model_type: str - Model type ("dfm" or "ddfm")
- encoder_layers: List[int] - (DDFM only) Encoder layer sizes
- num_factors: int - (DDFM only) Number of factors
"""


# Training progress milestones
class TrainingProgress:
    """Training progress milestones as constants."""
    INIT = 0
    LOAD_CONFIG = 10
    INIT_MODEL = 30
    LOAD_DATA = 20
    TRAINING = 40
    SAVE_MODEL = 80
    REGISTER = 90
    COMPLETE = 100


class TrainingManager:
    """Manages training jobs and progress."""
    
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._registry = ModelRegistry()
    
    def _get_logs_dir(self, model_name: str) -> Path:
        """Get logs directory for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to logs directory
        """
        return OUTPUTS_DIR / model_name / LOGS_DIR_NAME
    
    def _validate_list_of_ints(self, param_name: str, value: Any) -> List[int]:
        """Validate that value is a list of integers.
        
        Args:
            param_name: Name of parameter (for error messages)
            value: Value to validate
            
        Returns:
            Validated list of integers
            
        Raises:
            ValidationError: If value is not a list of integers
        """
        if not isinstance(value, list) or not all(isinstance(x, int) for x in value):
            raise ValidationError(
                f"Invalid {param_name} in config: must be a list of integers, "
                f"got {type(value).__name__ if not isinstance(value, list) else value}"
            )
        return value
    
    def _validate_positive_int(self, param_name: str, value: Any) -> int:
        """Validate that value is a positive integer.
        
        Args:
            param_name: Name of parameter (for error messages)
            value: Value to validate
            
        Returns:
            Validated positive integer
            
        Raises:
            ValidationError: If value is not a positive integer
        """
        if not isinstance(value, int) or value <= 0:
            raise ValidationError(
                f"Invalid {param_name} in config: must be a positive integer, "
                f"got {type(value).__name__ if not isinstance(value, int) else value}"
            )
        return value
    
    def _extract_params_from_config(
        self,
        config_path: str,
        default_params: Dict[str, Any],
        param_extractors: Dict[str, Callable[[Any], Any]]
    ) -> Dict[str, Any]:
        """Shared helper to extract parameters from config file.
        
        Args:
            config_path: Path to config YAML file
            default_params: Dictionary with default parameter values
            param_extractors: Dictionary mapping parameter names to extraction functions
                Each function takes the config_data dict and returns the extracted value
                Functions should handle validation internally or return None to use default
        
        Returns:
            Dictionary with extracted parameters (using defaults if extraction fails)
        """
        params = default_params.copy()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
            
            # Extract each parameter using its extractor function
            for param_name, extractor_func in param_extractors.items():
                try:
                    extracted_value = extractor_func(config_data)
                    if extracted_value is not None:
                        params[param_name] = extracted_value
                except Exception as e:
                    logger.debug(
                        f"Could not extract parameter '{param_name}' from config '{config_path}': {e}. "
                        f"Using default value: {params[param_name]}"
                    )
        
        except (IOError, OSError, yaml.YAMLError) as e:
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            logger.warning(
                f"Could not read/parse config file '{config_path}': {e}. "
                f"Using default parameters: {param_str}"
            )
        
        return params
    
    def _extract_dfm_params_from_config(self, config_path: str) -> Dict[str, Any]:
        """Extract DFM training parameters from config file.
        
        Args:
            config_path: Path to config YAML file
            
        Returns:
            Dictionary with DFM training parameters (max_iter, threshold)
            Uses defaults if parameters not found in config
            Defaults: max_iter=5000, threshold=1e-5
        """
        def extract_threshold(config_data: Dict[str, Any]) -> Optional[float]:
            """Extract and validate threshold parameter (can be float or string)."""
            if "threshold" not in config_data:
                return None
            threshold_value = config_data["threshold"]
            if isinstance(threshold_value, str):
                try:
                    return float(threshold_value)
                except ValueError:
                    return None
            return float(threshold_value) if isinstance(threshold_value, (int, float)) else None
        
        return self._extract_params_from_config(
            config_path,
            default_params={"max_iter": 5000, "threshold": 1e-5},
            param_extractors={
                "max_iter": lambda d: self._validate_positive_int("max_iter", d["max_iter"]) if "max_iter" in d else None,
                "threshold": extract_threshold
            }
        )
    
    def _extract_ddfm_params_from_config(self, config_path: str) -> Dict[str, Any]:
        """Extract DDFM parameters from config file.
        
        Args:
            config_path: Path to config YAML file
            
        Returns:
            Dictionary with DDFM parameters (encoder_layers, num_factors, epochs)
            Uses defaults if parameters not found in config
        """
        return self._extract_params_from_config(
            config_path,
            default_params={
                "encoder_layers": DEFAULT_DDFM_ENCODER_LAYERS,
                "num_factors": DEFAULT_DDFM_NUM_FACTORS,
                "epochs": DEFAULT_DDFM_EPOCHS
            },
            param_extractors={
                "encoder_layers": lambda d: self._validate_list_of_ints("encoder_layers", d["encoder_layers"]) if "encoder_layers" in d else None,
                "num_factors": lambda d: self._validate_positive_int("num_factors", d["num_factors"]) if "num_factors" in d else None,
                "epochs": lambda d: self._validate_positive_int("epochs", d["epochs"]) if "epochs" in d else None
            }
        )
    
    def _extract_metrics_from_metadata(self, metadata: Dict[str, Any], model_type: str) -> TrainingMetrics:
        """Extract training metrics from model metadata.
        
        Args:
            metadata: Model metadata dictionary containing training information
            model_type: Type of model ("dfm" or "ddfm")
            
        Returns:
            TrainingMetrics: Dictionary of training metrics with the following structure:
                - converged: bool - Whether training converged
                - num_iter: int - Number of iterations
                - loglik: float - Log-likelihood value
                - training_completed: Optional[str] - ISO timestamp of completion
                - model_type: str - Model type ("dfm" or "ddfm")
                - encoder_layers: List[int] - (DDFM only) Encoder layer sizes
                - num_factors: int - (DDFM only) Number of factors
        """
        metrics = {
            "converged": metadata.get("converged", False),
            "num_iter": metadata.get("num_iter", 0),
            "loglik": metadata.get("loglik", 0.0),
            "training_completed": metadata.get("training_completed"),
            "model_type": model_type
        }
        
        # Add DDFM-specific metrics
        if model_type == ModelType.DDFM:
            metrics.update({
                "encoder_layers": metadata.get("encoder_layers", DEFAULT_DDFM_ENCODER_LAYERS),
                "num_factors": metadata.get("num_factors", DEFAULT_DDFM_NUM_FACTORS)
            })
        
        return metrics
    
    def _format_error_message(
        self,
        operation: str,
        model_name: str,
        error: Exception,
        config_path: Optional[str] = None,
        data_path: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> str:
        """Format error message consistently with context.
        
        Args:
            operation: Description of the operation that failed
            model_name: Name of the model being trained
            error: The exception that occurred
            config_path: Optional path to config file (for context)
            data_path: Optional path to data file (for context)
            model_type: Optional model type (for context)
            
        Returns:
            Formatted error message with available context
        """
        def _short_path(path: str) -> str:
            """Return filename if path is long, full path otherwise."""
            return Path(path).name if len(path) > 50 else path
        
        context_parts = []
        if model_type:
            context_parts.append(f"type: {model_type}")
        if config_path:
            context_parts.append(f"config: {_short_path(config_path)}")
        if data_path:
            context_parts.append(f"data: {_short_path(data_path)}")
        
        context_str = f" ({', '.join(context_parts)})" if context_parts else ""
        return f"Training failed during {operation} for model '{model_name}'{context_str}: {str(error)}"
    
    def _handle_training_error(
        self,
        error: Exception,
        operation: str,
        model_name: str,
        log_progress: ProgressCallback,
        current_progress: int,
        config_path: Optional[str] = None,
        data_path: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> TrainingError:
        """Handle training errors consistently with context.
        
        Args:
            error: The exception that occurred
            operation: Description of the operation that failed
            model_name: Name of the model being trained
            log_progress: Progress logging callback
            current_progress: Current progress value
            config_path: Optional path to config file (for context)
            data_path: Optional path to data file (for context)
            model_type: Optional model type (for context)
            
        Returns:
            TrainingError: Wrapped training error with context (never returns, always raises)
        """
        error_msg = self._format_error_message(
            operation, model_name, error,
            config_path=config_path, data_path=data_path, model_type=model_type
        )
        log_progress(current_progress, error_msg)
        raise TrainingError(error_msg) from error
    
    def _save_and_register_model(
        self,
        model: ModelInstance,
        model_name: str,
        config_path: str,
        model_type: str,
        log_progress: ProgressCallback,
        training_metrics: Optional[TrainingMetrics] = None,
        training_duration_seconds: Optional[float] = None,
        experiment_id: Optional[str] = None
    ) -> None:
        """Save model and register it in the registry.
        
        Args:
            model: Trained model instance
            model_name: Name of the model
            config_path: Path to config file
            model_type: Type of model ("dfm" or "ddfm")
            log_progress: Progress logging callback
            training_metrics: Optional dictionary with training metrics
            training_duration_seconds: Optional training duration in seconds
        """
        log_progress(TrainingProgress.SAVE_MODEL, "Saving model...")
        
        # Save model and get the model directory path
        model_dir = model.save_to_outputs(
            model_name=model_name,
            outputs_dir=OUTPUTS_DIR,
            config_path=config_path
        )
        
        # Construct file paths
        model_pkl_path = str(model_dir / MODEL_PKL_FILE)
        config_yaml_path = str(model_dir / CONFIG_YAML_FILE) if (model_dir / CONFIG_YAML_FILE).exists() else None
        
        log_progress(TrainingProgress.REGISTER, "Registering model...")
        
        timestamp = datetime.now().isoformat()
        
        # Extract training metrics from model metadata if not provided
        if training_metrics is None:
            metadata = model.get_metadata()
            training_metrics = {
                "converged": metadata.get("converged", False),
                "num_iter": metadata.get("num_iter", 0),
                "loglik": metadata.get("loglik", 0.0)
            }
        
        self._registry.register_model(
            model_name=model_name,
            timestamp=timestamp,
            config_path=config_path,
            model_type=model_type,
            training_metrics=training_metrics,
            training_duration_seconds=training_duration_seconds,
            model_pkl_path=model_pkl_path,
            config_yaml_path=config_yaml_path,
            experiment_id=experiment_id
        )
    
    def _save_training_logs(
        self,
        model_name: str,
        progress_messages: List[Dict[str, Any]],
        metrics: Optional[Dict[str, Any]] = None,
        errors: Optional[List[str]] = None
    ):
        """Save training logs to outputs/{model_name}/logs/.
        
        Creates:
        - training.log: Progress messages with timestamps
        - metrics.json: Training metrics
        - errors.log: Error messages (if any)
        
        Args:
            model_name: Name of the model
            progress_messages: List of progress messages with timestamps
            metrics: Training metrics dictionary
            errors: List of error messages
        """
        logs_dir = self._get_logs_dir(model_name)
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Save training.log
        training_log_path = logs_dir / TRAINING_LOG_FILE
        with open(training_log_path, 'w', encoding='utf-8') as f:
            for msg in progress_messages:
                timestamp = msg.get("timestamp", "")
                message = msg.get("message", "")
                progress = msg.get("progress", 0)
                f.write(f"[{timestamp}] [{progress}%] {message}\n")
        
        # Save metrics.json
        if metrics:
            metrics_path = logs_dir / METRICS_JSON_FILE
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        # Save errors.log if there are errors
        if errors:
            errors_log_path = logs_dir / ERRORS_LOG_FILE
            with open(errors_log_path, 'w', encoding='utf-8') as f:
                for error in errors:
                    f.write(f"{error}\n")
    
    def start_training(
        self,
        model_name: str,
        model_type: str,  # Can be ModelType enum or string
        config_path: str,
        data_path: str,
        experiment_id: Optional[str] = None
    ) -> str:
        """Start a training job and return job_id."""
        job_id = str(uuid.uuid4())
        progress_messages = []
        errors = []
        
        self._jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.RUNNING,
            "progress": 0,
            "message": "Initializing training...",
            "error": None,
            "model_name": model_name,
            "timestamp": datetime.now().isoformat()
        }
        
        # Initialize progress tracking
        def log_progress(progress: int, message: str) -> None:
            timestamp = datetime.now().isoformat()
            progress_messages.append({
                "timestamp": timestamp,
                "progress": progress,
                "message": message
            })
            self._update_progress(job_id, progress, message)
        
        try:
            log_progress(TrainingProgress.INIT, "Initializing training...")
            
            # Run training synchronously
            if model_type == ModelType.DFM:
                metrics = self._train_dfm(model_name, config_path, data_path, log_progress, experiment_id=experiment_id)
            elif model_type == ModelType.DDFM:
                metrics = self._train_ddfm(model_name, config_path, data_path, log_progress, experiment_id=experiment_id)
            else:
                raise ValidationError(f"Unknown model type: {model_type}. Must be '{ModelType.DFM}' or '{ModelType.DDFM}'")
            
            log_progress(TrainingProgress.COMPLETE, "Training completed successfully")
            
            self._jobs[job_id]["status"] = JobStatus.COMPLETED
            self._jobs[job_id]["progress"] = TrainingProgress.COMPLETE
            self._jobs[job_id]["message"] = "Training completed successfully"
            
            # Save logs
            self._save_training_logs(model_name, progress_messages, metrics, errors if errors else None)
            
        except Exception as e:
            # Handle all exceptions uniformly
            current_progress = self._jobs[job_id].get("progress", 0)
            error_msg = self._format_error_message(
                "training", model_name, e,
                config_path=config_path,
                data_path=data_path,
                model_type=model_type
            )
            errors.append(error_msg)
            log_progress(current_progress, error_msg)
            
            self._jobs[job_id]["status"] = JobStatus.FAILED
            self._jobs[job_id]["error"] = str(e)
            self._jobs[job_id]["message"] = error_msg
            
            # Save logs including errors
            self._save_training_logs(model_name, progress_messages, None, errors)
            
            raise TrainingError(error_msg) from e
        
        return job_id
    
    def _train_model(
        self,
        model_name: str,
        config_path: str,
        data_path: str,
        model_type: str,  # Can be ModelType enum or string
        log_progress: ProgressCallback,
        model_class: Optional[ModelClass],
        model_kwargs: Optional[Dict[str, Any]] = None,
        training_kwargs: Optional[Dict[str, Any]] = None,
        experiment_id: Optional[str] = None
    ) -> TrainingMetrics:
        """Generic model training method.
        
        Args:
            model_name: Name of the model
            config_path: Path to config file
            data_path: Path to data file
            model_type: Type of model (ModelType.DFM, ModelType.DDFM, or string)
            log_progress: Progress logging callback
            model_class: Model class to instantiate
            model_kwargs: Optional kwargs for model initialization (e.g., encoder_layers for DDFM)
            training_kwargs: Optional kwargs for model.train() call (e.g., max_iter, threshold for DFM)
            
        Returns:
            TrainingMetrics: Dictionary of training metrics (see TrainingMetrics type alias for structure)
            
        Raises:
            ImportError: If model class is not available
            TrainingError: If training fails
        """
        if model_class is None:
            raise ImportError(f"{model_type.upper()} package not available")
        
        # Track training start time for duration calculation
        training_start_time = datetime.now()
        
        # Load configuration with error context
        try:
            log_progress(TrainingProgress.LOAD_CONFIG, "Loading configuration...")
            
            # Initialize model
            if model_kwargs:
                model = model_class(**model_kwargs)
                if model_type == ModelType.DDFM:
                    log_progress(TrainingProgress.INIT_MODEL, "Initializing DDFM model...")
            else:
                model = model_class()
            
            model.load_config(yaml=config_path)
        except Exception as e:
            raise self._handle_training_error(
                error=e,
                operation="configuration loading",
                model_name=model_name,
                log_progress=log_progress,
                current_progress=TrainingProgress.LOAD_CONFIG,
                config_path=config_path,
                data_path=data_path,
                model_type=model_type
            )
        
        # Load data and train model with error context
        try:
            log_progress(TrainingProgress.LOAD_DATA, "Loading data...")
            
            model_type_display = model_type.upper() if isinstance(model_type, str) else model_type.value.upper()
            log_progress(TrainingProgress.TRAINING, f"Training {model_type_display} model (this may take a while)...")
            
            # Use new API: train() accepts data_path directly
            # This will automatically create DFMDataModule from data_path
            # Pass training_kwargs (e.g., max_iter, threshold for DFM) if provided
            if training_kwargs:
                model.train(data_path=data_path, **training_kwargs)
            else:
                model.train(data_path=data_path)
        except Exception as e:
            raise self._handle_training_error(
                error=e,
                operation="data loading and model training",
                model_name=model_name,
                log_progress=log_progress,
                current_progress=TrainingProgress.TRAINING,
                config_path=config_path,
                data_path=data_path,
                model_type=model_type
            )
        
        # Calculate training duration
        training_end_time = datetime.now()
        training_duration = (training_end_time - training_start_time).total_seconds()
        
        # Extract metrics
        metadata = model.get_metadata()
        metrics = self._extract_metrics_from_metadata(metadata, model_type)
        
        # Save and register (pass duration for registry)
        try:
            self._save_and_register_model(
                model, 
                model_name, 
                config_path, 
                model_type, 
                log_progress,
                training_metrics=metrics,
                training_duration_seconds=training_duration,
                experiment_id=experiment_id
            )
        except Exception as e:
            raise self._handle_training_error(
                error=e,
                operation="model saving and registration",
                model_name=model_name,
                log_progress=log_progress,
                current_progress=TrainingProgress.SAVE_MODEL,
                config_path=config_path,
                data_path=data_path,
                model_type=model_type
            )
        
        return metrics
    
    def _train_dfm(
        self,
        model_name: str,
        config_path: str,
        data_path: str,
        log_progress: ProgressCallback,
        experiment_id: Optional[str] = None
    ) -> TrainingMetrics:
        """Train a DFM model.
        
        DFM training parameters (max_iter, threshold) are read from the config file.
        If not found in config, defaults are used (max_iter=5000, threshold=1e-5).
        
        Returns:
            TrainingMetrics: Dictionary of training metrics (see TrainingMetrics type alias for structure)
            
        Raises:
            TrainingError: If training fails
        """
        # Extract DFM training parameters from config file
        training_kwargs = self._extract_dfm_params_from_config(config_path)
        
        return self._train_model(
            model_name=model_name,
            config_path=config_path,
            data_path=data_path,
            model_type=ModelType.DFM,
            log_progress=log_progress,
            model_class=DFM,
            training_kwargs=training_kwargs,
            experiment_id=experiment_id
        )
    
    def _train_ddfm(
        self,
        model_name: str,
        config_path: str,
        data_path: str,
        log_progress: ProgressCallback,
        experiment_id: Optional[str] = None
    ) -> TrainingMetrics:
        """Train a DDFM model.
        
        DDFM parameters (encoder_layers, num_factors, epochs) are read from the config file.
        If not found in config, defaults are used.
        
        Returns:
            TrainingMetrics: Dictionary of training metrics (see TrainingMetrics type alias for structure)
            
        Raises:
            TrainingError: If training fails
        """
        # Extract DDFM parameters from config file
        ddfm_params = self._extract_ddfm_params_from_config(config_path)
        
        return self._train_model(
            model_name=model_name,
            config_path=config_path,
            data_path=data_path,
            model_type=ModelType.DDFM,
            log_progress=log_progress,
            model_class=DDFM,
            model_kwargs=ddfm_params,
            experiment_id=experiment_id
        )
    
    def _update_progress(self, job_id: str, progress: int, message: str) -> None:
        """Update training progress."""
        if job_id in self._jobs:
            self._jobs[job_id]["progress"] = progress
            self._jobs[job_id]["message"] = message
    
    def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get training job status."""
        return self._jobs.get(job_id)
    
    def get_recent_jobs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent training jobs, sorted by most recent first."""
        jobs = list(self._jobs.values())
        # Sort by timestamp if available, otherwise by insertion order (reversed)
        jobs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return jobs[:limit]

