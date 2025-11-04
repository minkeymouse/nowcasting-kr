"""Core nowcasting modules for DFM estimation and forecasting."""

from .config import ModelConfig, DataConfig, DFMConfig, AppConfig
from .data_loader import load_config, load_config_from_yaml, load_config_from_excel, load_data, transform_data
from .dfm import DFMResult, dfm
from .kalman import run_kf, skf, fis, miss_data
from .news import update_nowcast, news_dfm, para_const

# Backward compatibility aliases
ModelSpec = ModelConfig  # Deprecated: use ModelConfig
load_spec = load_config  # Deprecated: use load_config

__all__ = [
    'ModelConfig', 'DataConfig', 'DFMConfig', 'AppConfig',
    'load_config', 'load_config_from_yaml', 'load_config_from_excel',
    'load_data', 'transform_data',
    'DFMResult', 'dfm',
    'run_kf', 'skf', 'fis', 'miss_data',
    'update_nowcast', 'news_dfm', 'para_const',
    # Deprecated aliases
    'ModelSpec', 'load_spec',
]

