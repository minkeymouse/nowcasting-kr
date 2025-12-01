"""DDFM model wrapper with metadata tracking."""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

# Add dfm-python to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "dfm-python" / "src"))

try:
    from dfm_python import DDFM as DDFMBase
    from dfm_python.core.results import DFMResult
    from dfm_python.config import DFMConfig
except ImportError:
    DDFMBase = None
    DFMResult = None
    DFMConfig = None


class DDFM:
    """Wrapper around dfm-python DDFM with metadata tracking."""
    
    def __init__(
        self,
        encoder_layers: Optional[List[int]] = None,
        num_factors: Optional[int] = None,
        **kwargs
    ):
        """Initialize DDFM wrapper."""
        if DDFMBase is None:
            raise ImportError("DDFM requires PyTorch. Install with: pip install dfm-python[deep]")
        
        self._model = DDFMBase(
            encoder_layers=encoder_layers or [64, 32],
            num_factors=num_factors or 1,
            **kwargs
        )
        self._metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "model_type": "ddfm",
            "encoder_layers": encoder_layers or [64, 32],
            "num_factors": num_factors or 1
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
        if hasattr(result, 'converged'):
            self._metadata["converged"] = result.converged
        if hasattr(result, 'num_iter'):
            self._metadata["num_iter"] = result.num_iter
        if hasattr(result, 'loglik'):
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

