"""Configuration models for DFM nowcasting using OmegaConf and Hydra."""

import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
import warnings
from dataclasses import dataclass, field

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


def validate_frequency(frequency: str) -> str:
    """Validate frequency code."""
    if frequency not in _VALID_FREQUENCIES:
        raise ValueError(f"Invalid frequency: {frequency}. Must be one of {_VALID_FREQUENCIES}")
    return frequency


def validate_transformation(transformation: str) -> str:
    """Validate transformation code."""
    if transformation not in _VALID_TRANSFORMATIONS:
        warnings.warn(f"Unknown transformation code: {transformation}. Will use untransformed data.")
    return transformation


@dataclass
class SeriesConfig:
    """Configuration for a single time series."""
    series_id: str
    series_name: str
    frequency: str
    units: str
    transformation: str
    category: str
    blocks: List[int]
    api_code: Optional[str] = None
    api_source: Optional[str] = None
    
    def __post_init__(self):
        """Validate fields after initialization."""
        self.frequency = validate_frequency(self.frequency)
        self.transformation = validate_transformation(self.transformation)


@dataclass
class ModelConfig:
    """Model specification structure for DFM models."""
    series: List[SeriesConfig]
    block_names: List[str]
    _cached_blocks: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
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
        """Backward compatibility: Blocks property as numpy array (cached)."""
        if self._cached_blocks is None:
            blocks_list = [s.blocks for s in self.series]
            self._cached_blocks = np.array(blocks_list, dtype=int)
        return self._cached_blocks
    
    @property
    def BlockNames(self) -> List[str]:
        """Backward compatibility: BlockNames property (alias for block_names)."""
        return self.block_names
    
    @property
    def UnitsTransformed(self) -> List[str]:
        """Backward compatibility: UnitsTransformed property."""
        return [_TRANSFORM_UNITS_MAP.get(t, t) for t in self.Transformation]
    
    @classmethod
    def _from_legacy_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Convert legacy format (separate lists) to new format (series list)."""
        series_list = []
        n = len(data.get('SeriesID', data.get('series_id', [])))
        
        # Handle Blocks - can be numpy array or list of lists
        blocks_data = data.get('Blocks', data.get('blocks', []))
        if isinstance(blocks_data, np.ndarray):
            blocks_data = blocks_data.tolist()
        elif not isinstance(blocks_data, list):
            blocks_data = []
        
        # Helper to get list value with index fallback
        def get_list_value(key: str, index: int, default=None):
            """Get value from list, handling both camelCase and snake_case keys."""
            val = data.get(key, data.get(key.lower(), default))
            if isinstance(val, list) and index < len(val):
                return val[index]
            return default
        
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
                series_id=get_list_value('SeriesID', i, ''),
                series_name=get_list_value('SeriesName', i, ''),
                frequency=get_list_value('Frequency', i, 'm'),
                units=get_list_value('Units', i, ''),
                transformation=get_list_value('Transformation', i, 'lin'),
                category=get_list_value('Category', i, ''),
                blocks=series_blocks,
                api_code=get_list_value('api_code', i),
                api_source=get_list_value('api_source', i)
            ))
        
        return cls(
            series=series_list,
            block_names=data.get('BlockNames', data.get('block_names', []))
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create ModelConfig from dictionary.
        
        Handles both legacy format (separate lists) and new format (series list).
        
        Legacy format: {'SeriesID': [...], 'Frequency': [...], 'Blocks': [[...]], ...}
        New format: {'series': [{'series_id': ..., ...}], 'block_names': [...]}
        """
        # Detect legacy format (has SeriesID or series_id as lists)
        if 'SeriesID' in data or 'series_id' in data:
            return cls._from_legacy_dict(data)
        
        # New format with series list
        if 'series' in data:
            series_list = [
                SeriesConfig(**s) if isinstance(s, dict) else s 
                for s in data['series']
            ]
            return cls(
                series=series_list,
                block_names=data.get('block_names', [])
            )
        
        # Direct instantiation (shouldn't happen often, but handle it)
        return cls(**data)


@dataclass
class DataConfig:
    """Data loading configuration."""
    vintage: Optional[str] = None
    country: str = "US"
    sample_start: Optional[str] = None
    data_path: Optional[str] = None  # CSV file path
    # Database-related fields
    use_database: bool = False
    vintage_id: Optional[int] = None
    config_name: Optional[str] = None
    config_id: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    strict_mode: bool = False


@dataclass
class DFMConfig:
    """DFM estimation configuration."""
    threshold: float = 1e-5
    max_iter: int = 5000
    nan_method: int = 2
    nan_k: int = 3


@dataclass
class AppConfig:
    """Root application configuration combining all sub-configs."""
    model: ModelConfig
    data: DataConfig
    dfm: DFMConfig = field(default_factory=DFMConfig)


# Register with Hydra ConfigStore following the Structured Config schema pattern
# This enables validation of YAML config files while keeping our full dataclass
# with @property methods for runtime use.
# 
# Pattern: Schema validation (from Hydra docs)
# - YAML files extend schemas via defaults list
# - Schemas provide type checking and validation
# - Runtime uses full ModelConfig with @property methods
if HYDRA_AVAILABLE:
    cs = ConfigStore.instance()
    
    try:
        # Create schema versions without @property methods for Hydra validation
        # These match our dataclass structure exactly for schema validation.
        # We'll still use the full ModelConfig/DataConfig/DFMConfig classes with
        # @property methods at runtime via from_dict() conversion.
        from dataclasses import dataclass as schema_dataclass
        
        @schema_dataclass
        class SeriesConfigSchema:
            """Schema for SeriesConfig validation in Hydra."""
            series_id: str
            series_name: str
            frequency: str
            units: str
            transformation: str
            category: str
            blocks: List[int]
            api_code: Optional[str] = None
            api_source: Optional[str] = None
        
        @schema_dataclass
        class ModelConfigSchema:
            """Schema for ModelConfig validation in Hydra."""
            series: List[SeriesConfigSchema]
            block_names: List[str]
        
        @schema_dataclass
        class DataConfigSchema:
            """Schema for DataConfig validation in Hydra."""
            vintage: Optional[str] = None
            country: str = "US"
            sample_start: Optional[str] = None
            data_path: Optional[str] = None
            use_database: bool = False
            vintage_id: Optional[int] = None
            config_name: Optional[str] = None
            config_id: Optional[int] = None
            start_date: Optional[str] = None
            end_date: Optional[str] = None
            strict_mode: bool = False
        
        @schema_dataclass
        class DFMConfigSchema:
            """Schema for DFMConfig validation in Hydra."""
            threshold: float = 1e-5
            max_iter: int = 5000
            nan_method: int = 2
            nan_k: int = 3
        
        # Register schemas in config groups (following Hydra docs pattern)
        # These can be referenced in YAML defaults lists for validation
        # Format: defaults: [base_model_config, _self_]
        cs.store(group="model", name="base_model_config", node=ModelConfigSchema)
        cs.store(group="data", name="base_data_config", node=DataConfigSchema)
        cs.store(group="dfm", name="base_dfm_config", node=DFMConfigSchema)
        
        # Also register standalone for direct use
        cs.store(name="model_config_schema", node=ModelConfigSchema)
        cs.store(name="data_config_schema", node=DataConfigSchema)
        cs.store(name="dfm_config_schema", node=DFMConfigSchema)
        
    except Exception as e:
        # If registration fails, continue without schema validation
        # Configs will still work via from_dict() in config_loader.py
        warnings.warn(f"Could not register Hydra structured config schemas: {e}. "
                     f"Configs will still work via from_dict() but without schema validation.")
