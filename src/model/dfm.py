"""DFM model wrapper with metadata tracking."""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Add dfm-python to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "dfm-python" / "src"))

try:
    from dfm_python import DFM as DFMBase
    from dfm_python.core.results import DFMResult
    from dfm_python.config import DFMConfig
except ImportError:
    DFMBase = None
    DFMResult = None
    DFMConfig = None


class DFM:
    """Wrapper around dfm-python DFM with metadata tracking."""
    
    def __init__(self):
        """Initialize DFM wrapper."""
        if DFMBase is None:
            raise ImportError("dfm-python package not available")
        self._model = DFMBase()
        self._metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "model_type": "dfm"
        }
    
    def load_config(self, source: Any):
        """Load configuration."""
        self._model.load_config(source)
        self._metadata["config_loaded"] = True
    
    def load_data(self, data_path: str):
        """Load data."""
        self._model.load_data(data_path)
        self._metadata["data_path"] = data_path
        self._metadata["data_loaded"] = True
    
    def train(self, **kwargs):
        """Train the model."""
        self._model.train(**kwargs)
        result = self._model.get_result()
        
        # Add training metadata
        self._metadata["training_completed"] = datetime.now().isoformat()
        self._metadata["converged"] = result.converged
        self._metadata["num_iter"] = result.num_iter
        self._metadata["loglik"] = float(result.loglik)
    
    def predict(self, horizon: Optional[int] = None):
        """Generate forecasts."""
        return self._model.predict(horizon=horizon)
    
    def get_result(self) -> DFMResult:
        """Get model result."""
        return self._model.get_result()
    
    def get_config(self) -> DFMConfig:
        """Get model configuration."""
        return self._model.get_config()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        return self._metadata.copy()
    
    @property
    def nowcast(self):
        """Get nowcast manager."""
        return self._model.nowcast

