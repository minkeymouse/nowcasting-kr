"""Pytest configuration and fixtures for testing with Hydra configs."""

import sys
from pathlib import Path
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def hydra_config_dir():
    """Return the config directory path for Hydra."""
    return project_root / "config"


@pytest.fixture
def create_hydra_config():
    """Fixture to create a temporary Hydra-compatible config dict."""
    def _create_config(model_type: str, data_type: str = "investment", **overrides):
        """Create a config dict compatible with Hydra structure.
        
        Parameters
        ----------
        model_type : str
            Model type: 'dfm', 'ddfm', 'itf', etc.
            For DFM, use nested path: 'dfm/investment', 'dfm/consumption', 'dfm/production'
        data_type : str
            Data type: 'investment', 'consumption', 'production'
        **overrides
            Additional config overrides
            
        Returns
        -------
        dict
            Config dictionary compatible with main.py structure
        """
        from omegaconf import OmegaConf
        
        base_config = {
            'data': data_type,
            'train': True,
            'forecast': False,
        }
        
        # Model config depends on model type
        if model_type.startswith('dfm/'):
            # Nested DFM config (e.g., dfm/investment)
            model_name = 'dfm'
            variant = model_type.split('/')[1] if '/' in model_type else 'investment'
            
            # Load from actual config file if it exists
            config_file = project_root / "config" / "model" / "dfm" / f"{variant}.yaml"
            if config_file.exists():
                import yaml
                with open(config_file, 'r') as f:
                    model_config = yaml.safe_load(f)
            else:
                # Fallback: create minimal config
                model_config = {
                    'name': 'dfm',
                    'target_series': ['KOEQUIPTE'] if variant == 'investment' else 
                                   ['KOWRCCNSE'] if variant == 'consumption' else 
                                   ['KOIPALL.G']
                }
        elif model_type == 'ddfm':
            model_name = 'ddfm'
            model_config = {
                'name': 'ddfm',
                'target_series': None,
                'encoder_layers': [16, 8],
                'num_factors': 2,
                'max_epoch': 2,
            }
        else:
            # Other models (itf, patchtst, etc.)
            model_name = model_type
            model_config = {
                'name': model_type,
                'target_series': ['KOEQUIPTE'],
                'horizon': 88,
            }
        
        # Apply overrides
        base_config.update(overrides)
        model_config.update(overrides.get('model', {}))
        if 'model' in overrides:
            del overrides['model']
        
        config = {
            **base_config,
            'model': model_config
        }
        
        return OmegaConf.create(config)
    
    return _create_config
