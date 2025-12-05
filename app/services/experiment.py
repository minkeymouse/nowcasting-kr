"""Experiment service for running experiments using src.training functions."""

import uuid
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime

# Set up paths
from src.utils.path_setup import setup_paths
setup_paths(include_dfm_python=True, include_src=True, include_app=True)

from src.training import train
from app.utils import (
    TrainingError, ConfigError, JobStatus, format_error_message, validate_config_file,
    CONFIG_DIR
)


class ExperimentService:
    """Service for running experiments using src.training functions."""
    
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._config_dir = Path(CONFIG_DIR)
        self._experiment_config_dir = self._config_dir / "experiment"
    
    def _find_config_path(self, experiment_id: str) -> Path:
        """Find config path for experiment."""
        # Try experiment-specific config first
        config_path = self._experiment_config_dir / f"{experiment_id}.yaml"
        try:
            validate_config_file(config_path)
            return config_path
        except ConfigError:
            # Fall back to default config
            config_path = self._config_dir / "default.yaml"
            try:
                validate_config_file(config_path)
                return config_path
            except ConfigError:
                raise ConfigError(f"Experiment config not found: {experiment_id}")
    
    def _create_job(self, job_id: str, experiment_id: str) -> Dict[str, Any]:
        """Create a new job entry."""
        job = {
            "job_id": job_id,
            "status": JobStatus.RUNNING,
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "error": None
        }
        self._jobs[job_id] = job
        return job
    
    def run_hydra_experiment(
        self,
        experiment_id: str,
        config_overrides: Optional[List[str]] = None,
        output_dir: Optional[str] = None
    ) -> str:
        """Run an experiment using src.training.train."""
        job_id = str(uuid.uuid4())
        self._find_config_path(experiment_id)
        self._create_job(job_id, experiment_id)
        
        try:
            result = train(
                config_name=experiment_id,
                config_path=str(self._config_dir),
                config_overrides=config_overrides
            )
            
            self._jobs[job_id]["status"] = JobStatus.COMPLETED
            self._jobs[job_id]["result"] = result
            
        except Exception as e:
            self._jobs[job_id]["status"] = JobStatus.FAILED
            error_msg = format_error_message(
                "Experiment execution",
                str(e)
            )
            self._jobs[job_id]["error"] = error_msg
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

