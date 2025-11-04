"""Configuration models for DFM nowcasting using Pydantic and Hydra."""

import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
import warnings

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    from pydantic import ConfigDict
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    from dataclasses import dataclass
    BaseModel = None

try:
    from hydra.core.config_store import ConfigStore
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    ConfigStore = None
    DictConfig = None
    OmegaConf = None

# Valid frequency codes
_VALID_FREQUENCIES = {'d', 'w', 'm', 'q', 'sa', 'a'}

# Valid transformation codes
_VALID_TRANSFORMATIONS = {
    'lin', 'chg', 'ch1', 'pch', 'pc1', 'pca', 
    'cch', 'cca', 'log'
}

# Transformation to readable units mapping
_TRANSFORM_UNITS_MAP = {
    'lin': 'Levels (No Transformation)',
    'chg': 'Change (Difference)',
    'ch1': 'Year over Year Change (Difference)',
    'pch': 'Percent Change',
    'pc1': 'Year over Year Percent Change',
    'pca': 'Percent Change (Annual Rate)',
    'cch': 'Continuously Compounded Rate of Change',
    'cca': 'Continuously Compounded Annual Rate of Change',
    'log': 'Natural Log'
}


if PYDANTIC_AVAILABLE:
    class SeriesConfig(BaseModel):
        """Configuration for a single time series."""
        series_id: str = Field(..., description="Unique identifier for the series")
        series_name: str = Field(..., description="Human-readable name")
        frequency: str = Field(..., description="Frequency code: d, w, m, q, sa, a")
        units: str = Field(..., description="Original units of measurement")
        transformation: str = Field(..., description="Transformation code")
        category: str = Field(..., description="Category classification")
        blocks: List[int] = Field(..., description="Block loading structure (binary list)")
        
        @field_validator('frequency')
        @classmethod
        def validate_frequency(cls, v):
            if v not in _VALID_FREQUENCIES:
                raise ValueError(f"Invalid frequency: {v}. Must be one of {_VALID_FREQUENCIES}")
            return v
        
        @field_validator('transformation')
        @classmethod
        def validate_transformation(cls, v):
            if v not in _VALID_TRANSFORMATIONS:
                warnings.warn(f"Unknown transformation code: {v}. Will use untransformed data.")
            return v


    class ModelConfig(BaseModel):
        """Model specification structure with validation.
        
        This class defines the complete schema for DFM model specifications,
        including runtime validation of all fields.
        """
        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            validate_assignment=True,
        )
        
        series: List[SeriesConfig] = Field(..., description="List of time series configurations")
        block_names: List[str] = Field(..., description="Names of factor blocks")
        
        @model_validator(mode='after')
        def validate_blocks(self):
            """Validate blocks structure and consistency."""
            if not self.series:
                raise ValueError("At least one series must be specified")
            
            # Extract blocks matrix
            n_series = len(self.series)
            n_blocks = len(self.block_names)
            
            # Check all series have same number of blocks
            for i, s in enumerate(self.series):
                if len(s.blocks) != n_blocks:
                    raise ValueError(
                        f"Series {i} ({s.series_id}) has {len(s.blocks)} blocks, "
                        f"but expected {n_blocks} (from block_names)"
                    )
            
            # Check first column (global block) is all 1s
            for i, s in enumerate(self.series):
                if s.blocks[0] != 1:
                    raise ValueError(
                        f"Series {i} ({s.series_id}) must load on global block "
                        f"(first block must be 1)"
                    )
            
            return self
        
        # Convenience properties for backward compatibility
        @property
        def SeriesID(self) -> List[str]:
            """Backward compatibility: SeriesID property."""
            return [s.series_id for s in self.series]
        
        @property
        def SeriesName(self) -> List[str]:
            """Backward compatibility: SeriesName property."""
            return [s.series_name for s in self.series]
        
        @property
        def Frequency(self) -> List[str]:
            """Backward compatibility: Frequency property."""
            return [s.frequency for s in self.series]
        
        @property
        def Units(self) -> List[str]:
            """Backward compatibility: Units property."""
            return [s.units for s in self.series]
        
        @property
        def Transformation(self) -> List[str]:
            """Backward compatibility: Transformation property."""
            return [s.transformation for s in self.series]
        
        @property
        def Category(self) -> List[str]:
            """Backward compatibility: Category property."""
            return [s.category for s in self.series]
        
        @property
        def Blocks(self) -> np.ndarray:
            """Backward compatibility: Blocks property as numpy array."""
            blocks_list = [s.blocks for s in self.series]
            return np.array(blocks_list, dtype=int)
        
        @property
        def BlockNames(self) -> List[str]:
            """Backward compatibility: BlockNames property."""
            return self.block_names
        
        @property
        def UnitsTransformed(self) -> List[str]:
            """Backward compatibility: UnitsTransformed property."""
            return [_TRANSFORM_UNITS_MAP.get(t, t) for t in self.Transformation]
        
        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
            """Create ModelConfig from dictionary (legacy format).
            
            Handles both legacy format (separate lists) and new format (series list).
            """
            # Handle legacy format with separate lists
            if 'SeriesID' in data or 'series_id' in data:
                # Legacy format - convert to new format
                series_list = []
                n = len(data.get('SeriesID', data.get('series_id', [])))
                
                # Handle Blocks - can be numpy array or list of lists
                blocks_data = data.get('Blocks', data.get('blocks', []))
                if isinstance(blocks_data, np.ndarray):
                    blocks_data = blocks_data.tolist()
                elif not isinstance(blocks_data, list):
                    blocks_data = []
                
                for i in range(n):
                    # Extract blocks for this series
                    if blocks_data and i < len(blocks_data):
                        if isinstance(blocks_data[i], (list, np.ndarray)):
                            series_blocks = list(blocks_data[i]) if isinstance(blocks_data[i], np.ndarray) else blocks_data[i]
                        else:
                            series_blocks = [blocks_data[i]]
                    else:
                        series_blocks = []
                    
                    series_list.append(SeriesConfig(
                        series_id=data.get('SeriesID', data.get('series_id', []))[i],
                        series_name=data.get('SeriesName', data.get('series_name', []))[i],
                        frequency=data.get('Frequency', data.get('frequency', []))[i],
                        units=data.get('Units', data.get('units', []))[i],
                        transformation=data.get('Transformation', data.get('transformation', []))[i],
                        category=data.get('Category', data.get('category', []))[i],
                        blocks=series_blocks
                    ))
                
                return cls(
                    series=series_list,
                    block_names=data.get('BlockNames', data.get('block_names', []))
                )
            else:
                # New format with series list
                return cls(**data)


    class DataConfig(BaseModel):
        """Data loading configuration."""
        vintage: Optional[str] = Field(None, description="Data vintage date (YYYY-MM-DD)")
        country: str = Field("US", description="Country code")
        sample_start: Optional[str] = Field(None, description="Sample start date (YYYY-MM-DD)")
        data_path: Optional[str] = Field(None, description="Custom data file path")
        load_excel: bool = Field(False, description="Force loading from Excel even if cache exists")


    class DFMConfig(BaseModel):
        """DFM estimation configuration."""
        threshold: float = Field(1e-5, description="EM convergence threshold")
        max_iter: int = Field(5000, description="Maximum EM iterations")
        nan_method: int = Field(2, description="NaN handling method (1-5)")
        nan_k: int = Field(3, description="NaN filter parameter")


    class AppConfig(BaseModel):
        """Root application configuration combining all sub-configs."""
        model: ModelConfig = Field(..., description="Model specification")
        data: DataConfig = Field(..., description="Data loading configuration")
        dfm: DFMConfig = Field(default_factory=DFMConfig, description="DFM estimation configuration")
        
        # Optional: experiment metadata
        experiment_name: Optional[str] = Field(None, description="Experiment name")
        output_dir: Optional[str] = Field(None, description="Output directory for results")

else:
    # Fallback to dataclass if pydantic not available
    from dataclasses import dataclass
    
    @dataclass
    class SeriesConfig:
        series_id: str
        series_name: str
        frequency: str
        units: str
        transformation: str
        category: str
        blocks: List[int]
    
    @dataclass
    class ModelConfig:
        series: List[SeriesConfig]
        block_names: List[str]
    
    @dataclass
    class DataConfig:
        vintage: Optional[str] = None
        country: str = "US"
        sample_start: Optional[str] = None
        data_path: Optional[str] = None
        load_excel: bool = False
    
    @dataclass
    class DFMConfig:
        threshold: float = 1e-5
        max_iter: int = 5000
        nan_method: int = 2
        nan_k: int = 3
    
    @dataclass
    class AppConfig:
        model: ModelConfig
        data: DataConfig
        dfm: DFMConfig = None


# Register with Hydra ConfigStore if available
if HYDRA_AVAILABLE and PYDANTIC_AVAILABLE:
    cs = ConfigStore.instance()
    
    # Register individual configs
    cs.store(name="model_config", node=ModelConfig)
    cs.store(name="data_config", node=DataConfig)
    cs.store(name="dfm_config", node=DFMConfig)
    cs.store(name="app_config", node=AppConfig)
    
    # Register SeriesConfig for nested usage
    cs.store(name="series_config", node=SeriesConfig)

