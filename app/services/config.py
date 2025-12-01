"""Configuration manager service for handling YAML config files."""

import yaml
from pathlib import Path
from typing import Dict, List

from utils import CONFIG_DIR, ConfigError, ValidationError


class ConfigManager:
    """Manages YAML configuration files with hierarchical structure."""
    
    def __init__(self, config_dir: Path = CONFIG_DIR):
        self.config_dir = config_dir
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_dir = self.config_dir / "experiment"
        self.series_dir = self.config_dir / "series"
        self.blocks_dir = self.config_dir / "blocks"
        self.experiment_dir.mkdir(exist_ok=True)
        self.series_dir.mkdir(exist_ok=True)
        self.blocks_dir.mkdir(exist_ok=True)
    
    def get_config(self, config_name: str) -> str:
        """Get config file content (legacy method for backward compatibility)."""
        config_path = self.config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            raise ConfigError(f"Config file not found: {config_name}")
        
        with open(config_path, 'r') as f:
            return f.read()
    
    def update_config(self, config_name: str, content: str):
        """Update config file (legacy method for backward compatibility)."""
        # Validate YAML
        self.validate_config(content)
        
        config_path = self.config_dir / f"{config_name}.yaml"
        with open(config_path, 'w') as f:
            f.write(content)
    
    def list_configs(self) -> List[str]:
        """List available config files (legacy method)."""
        configs = []
        for path in self.config_dir.glob("*.yaml"):
            configs.append(path.stem)
        return configs
    
    def validate_config(self, content: str):
        """Validate YAML content."""
        try:
            yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise ValidationError(f"Invalid YAML: {str(e)}") from e
    
    # Experiment management methods
    def list_experiments(self) -> List[str]:
        """List all experiment IDs (including 'default')."""
        experiments = ["default"]
        for path in self.experiment_dir.glob("*.yaml"):
            experiments.append(path.stem)
        return sorted(experiments)
    
    def get_experiment(self, experiment_id: str) -> Dict[str, str]:
        """Get experiment config and metadata."""
        if experiment_id == "default":
            config_path = self.config_dir / "default.yaml"
            if not config_path.exists():
                raise ConfigError("Default config not found")
            with open(config_path, 'r') as f:
                content = f.read()
            config_dict = yaml.safe_load(content)
            # Try to infer model type from config
            model_type = "dfm"  # Default
            if any(key.startswith("ddfm_") for key in config_dict.keys() if isinstance(key, str)):
                model_type = "ddfm"
            return {
                "experiment_id": experiment_id,
                "model_type": model_type,
                "content": content
            }
        else:
            config_path = self.experiment_dir / f"{experiment_id}.yaml"
            if not config_path.exists():
                raise ConfigError(f"Experiment not found: {experiment_id}")
            with open(config_path, 'r') as f:
                content = f.read()
            config_dict = yaml.safe_load(content)
            # Try to infer model type from config
            model_type = "dfm"  # Default
            if any(key.startswith("ddfm_") for key in config_dict.keys() if isinstance(key, str)):
                model_type = "ddfm"
            return {
                "experiment_id": experiment_id,
                "model_type": model_type,
                "content": content
            }
    
    def create_experiment(self, experiment_id: str, model_type: str, content: str):
        """Create a new experiment config."""
        if experiment_id == "default":
            raise ConfigError("Cannot create 'default' experiment. Use update_experiment instead.")
        
        self.validate_config(content)
        
        config_path = self.experiment_dir / f"{experiment_id}.yaml"
        if config_path.exists():
            raise ConfigError(f"Experiment already exists: {experiment_id}")
        
        with open(config_path, 'w') as f:
            f.write(content)
    
    def update_experiment(self, experiment_id: str, content: str):
        """Update experiment config."""
        self.validate_config(content)
        
        if experiment_id == "default":
            config_path = self.config_dir / "default.yaml"
        else:
            config_path = self.experiment_dir / f"{experiment_id}.yaml"
        
        with open(config_path, 'w') as f:
            f.write(content)
    
    # Series config management
    def list_series_configs(self) -> List[str]:
        """List all series config names."""
        series_configs = []
        for path in self.series_dir.glob("*.yaml"):
            series_configs.append(path.stem)
        return sorted(series_configs)
    
    def get_series_config(self, series_name: str) -> str:
        """Get series config content."""
        config_path = self.series_dir / f"{series_name}.yaml"
        if not config_path.exists():
            raise ConfigError(f"Series config not found: {series_name}")
        
        with open(config_path, 'r') as f:
            return f.read()
    
    def update_series_config(self, series_name: str, content: str):
        """Update series config."""
        self.validate_config(content)
        
        config_path = self.series_dir / f"{series_name}.yaml"
        with open(config_path, 'w') as f:
            f.write(content)
    
    # Block config management
    def list_block_configs(self) -> List[str]:
        """List all block config names."""
        block_configs = []
        for path in self.blocks_dir.glob("*.yaml"):
            block_configs.append(path.stem)
        return sorted(block_configs)
    
    def get_block_config(self, block_name: str) -> str:
        """Get block config content."""
        config_path = self.blocks_dir / f"{block_name}.yaml"
        if not config_path.exists():
            raise ConfigError(f"Block config not found: {block_name}")
        
        with open(config_path, 'r') as f:
            return f.read()
    
    def update_block_config(self, block_name: str, content: str):
        """Update block config."""
        self.validate_config(content)
        
        config_path = self.blocks_dir / f"{block_name}.yaml"
        with open(config_path, 'w') as f:
            f.write(content)
    
    def get_experiment_config_path(self, experiment_id: str) -> str:
        """Get the config path for an experiment (for training)."""
        if experiment_id == "default":
            return str(self.config_dir / "default.yaml")
        else:
            return str(self.experiment_dir / f"{experiment_id}.yaml")

