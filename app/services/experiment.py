"""Experiment service for running experiments using src.training functions."""

import json
import uuid
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime

# Set up paths
from src.utils.path_setup import setup_paths
setup_paths(include_dfm_python=True, include_src=True, include_app=True)

from src.train import train_model
from app.utils import (
    TrainingError, ConfigError, JobStatus, format_error_message, validate_config_file,
    CONFIG_DIR, OUTPUTS_DIR, LOGS_DIR_NAME, METRICS_JSON_FILE, ERRORS_LOG_FILE
)
from .registry import ModelRegistry


class ExperimentService:
    """Service for running experiments using src.training.train."""
    
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._registry = ModelRegistry()
        self._config_dir = Path(CONFIG_DIR)
        self._experiment_config_dir = self._config_dir / "experiment"
    
    def _find_config_path(self, experiment_id: str) -> Path:
        """Find config path for experiment."""
        config_path = self._experiment_config_dir / f"{experiment_id}.yaml"
        try:
            validate_config_file(config_path)
            return config_path
        except ConfigError:
            config_path = self._config_dir / "default.yaml"
            try:
                validate_config_file(config_path)
                return config_path
            except ConfigError:
                raise ConfigError(f"Experiment config not found: {experiment_id}")
    
    def _save_job_logs(self, job_id: str, job: Dict[str, Any]):
        """Save job logs if model_name is available."""
        model_name = job.get("model_name")
        if not model_name:
            return
        
        # src.training.train saves to outputs/models/{model_name}/
        logs_dir = OUTPUTS_DIR / "models" / model_name / LOGS_DIR_NAME
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics if available
        if job.get("result") and job["result"].get("metrics"):
            metrics_path = logs_dir / METRICS_JSON_FILE
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(job["result"]["metrics"], f, indent=2, ensure_ascii=False)
        
        # Save errors if failed
        if job.get("status") == JobStatus.FAILED and job.get("error"):
            errors_path = logs_dir / ERRORS_LOG_FILE
            with open(errors_path, 'w', encoding='utf-8') as f:
                f.write(f"{job['error']}\n")
    
    def run_experiment(
        self,
        experiment_id: str,
        data_path: Optional[str] = None,
        model_name: Optional[str] = None,
        config_overrides: Optional[List[str]] = None,
        horizons: Optional[List[int]] = None
    ) -> str:
        """Run an experiment using src.training.train.
        
        Args:
            experiment_id: Experiment ID (config name)
            data_path: Optional data path override
            model_name: Optional model name (auto-generated if not provided)
            config_overrides: Optional Hydra config overrides
            horizons: Optional forecast horizons for evaluation (default: [1, 7, 28])
            
        Returns:
            Job ID for tracking
        """
        job_id = str(uuid.uuid4())
        self._find_config_path(experiment_id)
        
        job = {
            "job_id": job_id,
            "status": JobStatus.RUNNING,
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "error": None,
            "model_name": model_name
        }
        self._jobs[job_id] = job
        
        try:
            result = train_model(
                config_name=experiment_id,
                config_dir=str(self._config_dir),
                model_name=model_name,
                overrides=config_overrides
            )
            
            job["status"] = JobStatus.COMPLETED
            job["result"] = result
            final_model_name = result.get("model_name") or model_name
            job["model_name"] = final_model_name
            
            # Register model in registry
            if final_model_name:
                self._registry.register_model(
                    model_name=final_model_name,
                    timestamp=job["timestamp"],
                    config_path=str(self._find_config_path(experiment_id)),
                    model_type=result.get("metadata", {}).get("model_type", "dfm"),
                    training_metrics=result.get("metrics"),
                    experiment_id=experiment_id
                )
            
            self._save_job_logs(job_id, job)
            
        except Exception as e:
            job["status"] = JobStatus.FAILED
            error_msg = format_error_message("Experiment execution", str(e))
            job["error"] = error_msg
            self._save_job_logs(job_id, job)
            raise TrainingError(error_msg) from e
        
        return job_id
    
    def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment job status."""
        return self._jobs.get(job_id)
    
    def get_recent_jobs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent experiment jobs."""
        jobs = list(self._jobs.values())
        jobs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return jobs[:limit]
