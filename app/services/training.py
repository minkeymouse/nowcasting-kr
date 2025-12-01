"""Training manager service for handling model training jobs."""

import pickle
import uuid
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import sys

# Add dfm-python to path
dfm_python_path = Path(__file__).parent.parent.parent / "dfm-python" / "src"
if dfm_python_path.exists():
    sys.path.insert(0, str(dfm_python_path))

try:
    from dfm_python import DFM, DDFM
except ImportError:
    DFM = None
    DDFM = None

from utils import OUTPUTS_DIR
from .registry import ModelRegistry
from utils import TrainingError


class TrainingManager:
    """Manages training jobs and progress."""
    
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._registry = ModelRegistry()
    
    def start_training(
        self,
        model_name: str,
        model_type: str,
        config_path: str,
        data_path: str
    ) -> str:
        """Start a training job and return job_id."""
        job_id = str(uuid.uuid4())
        
        self._jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "progress": 0,
            "message": "Initializing training...",
            "error": None,
            "model_name": model_name,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Run training synchronously
            if model_type == "dfm":
                self._train_dfm(job_id, model_name, config_path, data_path)
            elif model_type == "ddfm":
                self._train_ddfm(job_id, model_name, config_path, data_path)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            self._jobs[job_id]["status"] = "completed"
            self._jobs[job_id]["progress"] = 100
            self._jobs[job_id]["message"] = "Training completed successfully"
            
        except Exception as e:
            self._jobs[job_id]["status"] = "failed"
            self._jobs[job_id]["error"] = str(e)
            self._jobs[job_id]["message"] = f"Training failed: {str(e)}"
            raise TrainingError(f"Training failed: {str(e)}") from e
        
        return job_id
    
    def _train_dfm(
        self,
        job_id: str,
        model_name: str,
        config_path: str,
        data_path: str
    ):
        """Train a DFM model."""
        if DFM is None:
            raise ImportError("dfm-python package not available")
        
        self._update_progress(job_id, 10, "Loading configuration...")
        
        # Create and train model
        model = DFM()
        model.load_config(yaml=config_path)
        
        self._update_progress(job_id, 20, "Loading data...")
        
        model.load_data(data_path)
        
        self._update_progress(job_id, 40, "Training model (this may take a while)...")
        
        # Train
        model.train()
        
        self._update_progress(job_id, 80, "Saving model...")
        
        # Save model
        model_dir = OUTPUTS_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model result
        result = model.get_result()
        config = model.get_config()
        
        # Get time index if available
        time_index = None
        try:
            time_index = model.get_time()
        except:
            pass
        
        with open(model_dir / "model.pkl", 'wb') as f:
            pickle.dump({
                "model": model,
                "result": result,
                "config": config,
                "time": time_index
            }, f)
        
        # Save config copy
        config_path_obj = Path(config_path)
        if config_path_obj.exists():
            with open(model_dir / "config.yaml", 'w') as f:
                with open(config_path, 'r') as src:
                    f.write(src.read())
        
        # Create subdirectories
        (model_dir / "logs").mkdir(exist_ok=True)
        (model_dir / "plots").mkdir(exist_ok=True)
        (model_dir / "results").mkdir(exist_ok=True)
        
        self._update_progress(job_id, 90, "Registering model...")
        
        # Register model
        timestamp = datetime.now().isoformat()
        self._registry.register_model(
            model_name=model_name,
            timestamp=timestamp,
            config_path=config_path,
            model_type="dfm"
        )
    
    def _train_ddfm(
        self,
        job_id: str,
        model_name: str,
        config_path: str,
        data_path: str
    ):
        """Train a DDFM model."""
        if DDFM is None:
            raise ImportError("DDFM requires PyTorch. Install with: pip install dfm-python[deep]")
        
        self._update_progress(job_id, 10, "Loading configuration...")
        
        self._update_progress(job_id, 30, "Initializing DDFM model...")
        
        # Create DDFM model (with default parameters, can be made configurable)
        model = DDFM(
            encoder_layers=[64, 32],
            num_factors=1,
            epochs=100
        )
        model.load_config(yaml=config_path)
        
        self._update_progress(job_id, 20, "Loading data...")
        
        model.load_data(data_path)
        
        self._update_progress(job_id, 40, "Training DDFM model (this may take a while)...")
        
        # Train
        model.train()
        
        self._update_progress(job_id, 80, "Saving model...")
        
        # Save model
        model_dir = OUTPUTS_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model result
        result = model.get_result()
        config = model.get_config()
        
        # Get time index if available
        time_index = None
        try:
            time_index = model.get_time()
        except:
            pass
        
        with open(model_dir / "model.pkl", 'wb') as f:
            pickle.dump({
                "model": model,
                "result": result,
                "config": config,
                "time": time_index
            }, f)
        
        # Save config copy
        config_path_obj = Path(config_path)
        if config_path_obj.exists():
            with open(model_dir / "config.yaml", 'w') as f:
                with open(config_path, 'r') as src:
                    f.write(src.read())
        
        # Create subdirectories
        (model_dir / "logs").mkdir(exist_ok=True)
        (model_dir / "plots").mkdir(exist_ok=True)
        (model_dir / "results").mkdir(exist_ok=True)
        
        self._update_progress(job_id, 90, "Registering model...")
        
        # Register model
        timestamp = datetime.now().isoformat()
        self._registry.register_model(
            model_name=model_name,
            timestamp=timestamp,
            config_path=config_path,
            model_type="ddfm"
        )
    
    def _update_progress(self, job_id: str, progress: int, message: str):
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

