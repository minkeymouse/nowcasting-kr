"""Experiment service for running Hydra-based experiments via src/main.py."""

import subprocess
import uuid
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime

from app.utils import (
    TrainingError, ConfigError, JobStatus, format_error_message, validate_config_file
)


# Constants
EXPERIMENT_TIMEOUT_SECONDS = 3600  # 1 hour
PYTHON_COMMAND = "python3"
CONFIG_DIR_NAME = "config"
EXPERIMENT_CONFIG_DIR = "experiment"
DEFAULT_CONFIG_NAME = "default"


class ExperimentService:
    """Service for running Hydra-based experiments."""
    
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._project_root = Path(__file__).parent.parent.parent
        self._config_dir = self._project_root / CONFIG_DIR_NAME
        self._experiment_config_dir = self._config_dir / EXPERIMENT_CONFIG_DIR
    
    def _find_config_path(self, experiment_id: str) -> Path:
        """Find config path for experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Path to config file
            
        Raises:
            ConfigError: If config not found
        """
        # Try experiment-specific config first
        config_path = self._experiment_config_dir / f"{experiment_id}.yaml"
        try:
            validate_config_file(config_path)
            return config_path
        except ConfigError:
            # Fall back to default config
            config_path = self._config_dir / f"{DEFAULT_CONFIG_NAME}.yaml"
            try:
                validate_config_file(config_path)
                return config_path
            except ConfigError:
                raise ConfigError(f"Experiment config not found: {experiment_id}")
    
    def _build_hydra_command(
        self,
        experiment_id: str,
        config_overrides: Optional[List[str]] = None,
        output_dir: Optional[str] = None
    ) -> List[str]:
        """Build Hydra command for experiment execution.
        
        Args:
            experiment_id: Experiment ID
            config_overrides: Optional config overrides
            output_dir: Optional output directory override
            
        Returns:
            Command list for subprocess
        """
        cmd = [PYTHON_COMMAND, str(self._project_root / "src" / "main.py")]
        cmd.append(f"config_name={experiment_id}")
        
        if config_overrides:
            cmd.extend(config_overrides)
        
        if output_dir:
            cmd.append(f"hydra.run.dir={output_dir}")
        
        return cmd
    
    def _create_job(self, job_id: str, experiment_id: str) -> Dict[str, Any]:
        """Create a new job entry.
        
        Args:
            job_id: Job ID
            experiment_id: Experiment ID
            
        Returns:
            Job dictionary
        """
        job = {
            "job_id": job_id,
            "status": JobStatus.RUNNING,
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "output": [],
            "error": None
        }
        self._jobs[job_id] = job
        return job
    
    def _process_subprocess_result(self, job_id: str, result: subprocess.CompletedProcess):
        """Process subprocess result and update job status.
        
        Args:
            job_id: Job ID
            result: Completed subprocess result
        """
        output_lines = result.stdout.split('\n') if result.stdout else []
        error_lines = result.stderr.split('\n') if result.stderr else []
        
        self._jobs[job_id]["output"] = output_lines
        self._jobs[job_id]["error_output"] = error_lines
        
        if result.returncode == 0:
            self._jobs[job_id]["status"] = JobStatus.COMPLETED
        else:
            self._jobs[job_id]["status"] = JobStatus.FAILED
            error_message = "\n".join(error_lines) if error_lines else (result.stderr or "Unknown error")
            self._jobs[job_id]["error"] = error_message
            raise TrainingError(f"Experiment failed: {error_message}")
    
    def run_hydra_experiment(
        self,
        experiment_id: str,
        config_overrides: Optional[List[str]] = None,
        output_dir: Optional[str] = None
    ) -> str:
        """Run a Hydra-based experiment via src/main.py.
        
        Args:
            experiment_id: Experiment ID (config name)
            config_overrides: Optional list of Hydra config overrides (e.g., ["data_path=path/to/data.csv"])
            output_dir: Optional output directory override
            
        Returns:
            Job ID for tracking the experiment
            
        Raises:
            ConfigError: If experiment config doesn't exist
            TrainingError: If experiment execution fails
        """
        job_id = str(uuid.uuid4())
        
        # Validate experiment config exists
        self._find_config_path(experiment_id)
        
        # Create job entry
        self._create_job(job_id, experiment_id)
        
        try:
            # Build and run Hydra command
            cmd = self._build_hydra_command(experiment_id, config_overrides, output_dir)
            
            result = subprocess.run(
                cmd,
                cwd=self._project_root,
                capture_output=True,
                text=True,
                timeout=EXPERIMENT_TIMEOUT_SECONDS
            )
            
            self._process_subprocess_result(job_id, result)
            
        except subprocess.TimeoutExpired:
            self._jobs[job_id]["status"] = JobStatus.FAILED
            error_msg = format_error_message(
                operation="Experiment execution",
                reason=f"Timed out after {EXPERIMENT_TIMEOUT_SECONDS} seconds",
                suggestion="Try increasing the timeout or optimizing the experiment configuration"
            )
            self._jobs[job_id]["error"] = error_msg
            raise TrainingError(error_msg)
        except TrainingError:
            # Re-raise TrainingError as-is
            raise
        except (subprocess.SubprocessError, OSError) as e:
            self._jobs[job_id]["status"] = JobStatus.FAILED
            error_msg = format_error_message(
                operation="Experiment execution",
                reason=f"Subprocess error: {str(e)}",
                suggestion="Check system resources and experiment configuration"
            )
            self._jobs[job_id]["error"] = error_msg
            raise TrainingError(error_msg) from e
        except Exception as e:
            self._jobs[job_id]["status"] = JobStatus.FAILED
            error_msg = format_error_message(
                operation="Experiment execution",
                reason=str(e),
                suggestion="Check experiment configuration and logs for details"
            )
            self._jobs[job_id]["error"] = error_msg
            raise TrainingError(error_msg) from e
        
        return job_id
    
    def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment job status."""
        return self._jobs.get(job_id)
    
    def get_output(self, job_id: str) -> List[str]:
        """Get experiment output."""
        job = self._jobs.get(job_id)
        if job:
            return job.get("output", [])
        return []
    
    def get_recent_jobs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent experiment jobs."""
        jobs = list(self._jobs.values())
        jobs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return jobs[:limit]

