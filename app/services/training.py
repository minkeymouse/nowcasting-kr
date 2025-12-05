"""Training manager service for handling model training jobs.

This service wraps src.training functions to provide progress tracking and job management
for the web application.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Set up paths using centralized utility
from src.utils.path_setup import setup_paths
setup_paths(include_dfm_python=True, include_src=True, include_app=True)

# Import training functions from src
from src.training import train
from app.utils import (
    OUTPUTS_DIR,
    TrainingError,
    JobStatus,
    TRAINING_LOG_FILE,
    METRICS_JSON_FILE,
    ERRORS_LOG_FILE,
    LOGS_DIR_NAME,
    format_error_message
)
from .registry import ModelRegistry

# Set up logging
logger = logging.getLogger(__name__)


# Training progress milestones
class TrainingProgress:
    """Training progress milestones as constants."""
    INIT = 0
    LOAD_CONFIG = 10
    TRAINING = 50
    SAVE_MODEL = 80
    COMPLETE = 100


class TrainingManager:
    """Manages training jobs and progress using src.training functions."""
    
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._registry = ModelRegistry()
    
    def _get_logs_dir(self, model_name: str) -> Path:
        """Get logs directory for a model."""
        return OUTPUTS_DIR / model_name / LOGS_DIR_NAME
    
    def _save_training_logs(
        self,
        model_name: str,
        progress_messages: List[Dict[str, Any]],
        metrics: Optional[Dict[str, Any]] = None,
        errors: Optional[List[str]] = None
    ):
        """Save training logs to outputs/{model_name}/logs/."""
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
        model_type: str,
        config_path: str,
        data_path: str,
        experiment_id: Optional[str] = None
    ) -> str:
        """Start a training job using src.training.train.
        
        Args:
            model_name: Name of the model
            model_type: Model type ("dfm" or "ddfm")
            config_path: Path to config file
            data_path: Path to data file
            experiment_id: Optional experiment ID
            
        Returns:
            Job ID for tracking the training job
        """
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
            
            # Extract experiment ID from config path if not provided
            if experiment_id is None:
                config_path_obj = Path(config_path)
                if config_path_obj.parent.name == "experiment":
                    experiment_id = config_path_obj.stem
                elif config_path_obj.name == "default.yaml":
                    experiment_id = "default"
                else:
                    experiment_id = config_path_obj.stem
            
            log_progress(TrainingProgress.LOAD_CONFIG, f"Loading configuration: {experiment_id}")
            
            log_progress(TrainingProgress.TRAINING, f"Training {model_type.upper()} model...")
            
            config_dir = Path(config_path).parent
            if config_dir.name == "experiment":
                config_dir = config_dir.parent
            
            result = train(
                config_name=experiment_id,
                config_path=str(config_dir),
                data_path=data_path,
                model_name=model_name,
                config_overrides=[f"model_name={model_name}"]
            )
            
            # Extract metrics from result
            metrics = result.get('metrics', {})
            
            # Register model in registry
            log_progress(TrainingProgress.SAVE_MODEL, "Registering model...")
            timestamp = datetime.now().isoformat()
            
            self._registry.register_model(
                model_name=model_name,
                timestamp=timestamp,
                config_path=config_path,
                model_type=model_type,
                training_metrics=metrics,
                experiment_id=experiment_id
            )
            
            log_progress(TrainingProgress.COMPLETE, "Training completed successfully")
            
            self._jobs[job_id]["status"] = JobStatus.COMPLETED
            self._jobs[job_id]["progress"] = TrainingProgress.COMPLETE
            self._jobs[job_id]["message"] = "Training completed successfully"
            
            # Save logs
            self._save_training_logs(model_name, progress_messages, metrics, errors if errors else None)
            
        except Exception as e:
            # Handle all exceptions uniformly
            current_progress = self._jobs[job_id].get("progress", 0)
            error_msg = format_error_message(
                "Training",
                f"model '{model_name}': {str(e)}",
                f"Check config: {config_path}, data: {data_path}"
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
        jobs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return jobs[:limit]
