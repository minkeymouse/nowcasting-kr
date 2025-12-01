"""Data loading utilities wrapping dfm-python dataloader."""

import sys
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime

# Add dfm-python to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "dfm-python" / "src"))

try:
    from dfm_python.dataloader import load_data as dfm_load_data
    from dfm_python.config import DFMConfig
except ImportError:
    dfm_load_data = None
    DFMConfig = None


def load_data(
    data_path: str,
    config: DFMConfig,
    sample_start: Optional[datetime] = None,
    sample_end: Optional[datetime] = None
) -> Tuple:
    """Load data using dfm-python dataloader with project-specific validation.
    
    Args:
        data_path: Path to CSV data file
        config: DFMConfig object
        sample_start: Optional start date for data sampling
        sample_end: Optional end date for data sampling
        
    Returns:
        Tuple of (X, Time, Z) where:
        - X: Data matrix (T x N)
        - Time: Time index
        - Z: Original data (optional)
        
    Raises:
        ImportError: If dfm-python is not available
        ValueError: If data validation fails
    """
    if dfm_load_data is None:
        raise ImportError("dfm-python package not available")
    
    # Validate data path
    data_path_obj = Path(data_path)
    if not data_path_obj.exists():
        raise ValueError(f"Data file not found: {data_path}")
    
    if not data_path_obj.suffix == '.csv':
        raise ValueError(f"Data file must be CSV, got: {data_path_obj.suffix}")
    
    # Load data using dfm-python
    X, Time, Z = dfm_load_data(
        data_path,
        config,
        sample_start=sample_start,
        sample_end=sample_end
    )
    
    # Project-specific validation
    if X.shape[0] == 0:
        raise ValueError("Data file is empty")
    
    if X.shape[1] == 0:
        raise ValueError("No series found in data file")
    
    return X, Time, Z

