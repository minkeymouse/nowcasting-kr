"""Configuration manager service for handling YAML config files.

This service integrates with src.config module for CSV import and config management.
"""

import warnings
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Set up paths using centralized utility
# Import directly from project root (app/services/ is in project root, src/ is sibling)
from src.utils.path_setup import setup_paths
setup_paths(include_src=True)

from src.config import ConfigManager as BaseConfigManager, ConfigUpdater
from app.utils import (
    CONFIG_DIR, ConfigError, ValidationError, ModelType, 
    validate_config_file, validate_file_exists, format_error_message
)


class ConfigManager:
    """Manages YAML configuration files with hierarchical structure.
    
    Integrates with src.config module for CSV import and advanced config management.
    """
    
    def __init__(self, config_dir: Path = CONFIG_DIR):
        self.config_dir = config_dir
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_dir = self.config_dir / "experiment"
        self.series_dir = self.config_dir / "series"
        self.model_dir = self.config_dir / "model"
        # Removed blocks_dir - blocks are not used in the current architecture
        self.experiment_dir.mkdir(exist_ok=True)
        self.series_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize base config manager from src.config
        self._base_manager = BaseConfigManager(str(self.config_dir))
        self._updater: Optional[ConfigUpdater] = None
    
    def _read_yaml_file(self, file_path: Path) -> str:
        """Read YAML file content.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            File content as string
            
        Raises:
            ConfigError: If file doesn't exist or I/O error occurs
        """
        validate_config_file(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except (IOError, OSError) as e:
            error_msg = format_error_message(
                operation="Reading config file",
                reason=f"I/O error reading file '{file_path}': {str(e)}",
                suggestion="Verify the file exists and has read permissions"
            )
            raise ConfigError(error_msg) from e
    
    def _write_yaml_file(self, file_path: Path, content: str) -> None:
        """Write YAML file content.
        
        Args:
            file_path: Path to YAML file
            content: Content to write
            
        Raises:
            ConfigError: If I/O error occurs
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except (IOError, OSError) as e:
            # Determine config type from path for better error context
            # This helps users understand which type of config file failed to write
            # (experiment, series, block, or model config)
            config_type = "unknown"
            if "experiment" in str(file_path):
                config_type = "experiment"
            elif "series" in str(file_path):
                config_type = "series"
            elif "blocks" in str(file_path) or "block" in str(file_path):
                config_type = "block"
            elif "model" in str(file_path):
                config_type = "model"
            
            error_msg = format_error_message(
                operation=f"Writing {config_type} config file",
                reason=f"I/O error writing to file '{file_path}': {str(e)}",
                suggestion="Verify the directory exists and has write permissions"
            )
            raise ConfigError(error_msg) from e
    
    def _infer_model_type(self, config_dict: Dict[str, Any]) -> str:
        """Infer model type from config dictionary.
        
        Args:
            config_dict: Parsed YAML config dictionary
            
        Returns:
            Model type string (ModelType.DFM or ModelType.DDFM)
        """
        if any(key.startswith("ddfm_") for key in config_dict.keys() if isinstance(key, str)):
            return ModelType.DDFM
        return ModelType.DFM
    
    def get_config(self, config_name: str) -> str:
        """Get config file content (legacy method - deprecated).
        
        .. deprecated:: 0.1.0
            Use :meth:`get_experiment` for experiment configs or access BaseConfigManager
            methods for series/block configs.
        """
        warnings.warn(
            f"get_config() is deprecated. Use get_experiment('{config_name}') for experiments "
            "or use BaseConfigManager methods for series/block configs.",
            DeprecationWarning,
            stacklevel=2
        )
        config_path = self.config_dir / f"{config_name}.yaml"
        return self._read_yaml_file(config_path)
    
    def update_config(self, config_name: str, content: str):
        """Update config file (legacy method - deprecated).
        
        .. deprecated:: 0.1.0
            Use :meth:`update_experiment` for experiment configs or access BaseConfigManager
            methods for series/block configs.
        """
        warnings.warn(
            f"update_config() is deprecated. Use update_experiment('{config_name}', content) "
            "for experiments or use BaseConfigManager methods for series/block configs.",
            DeprecationWarning,
            stacklevel=2
        )
        # Validate YAML
        self.validate_config(content, config_type="config")
        
        config_path = self.config_dir / f"{config_name}.yaml"
        self._write_yaml_file(config_path, content)
    
    def list_configs(self) -> List[str]:
        """List available config files (legacy method - deprecated).
        
        .. deprecated:: 0.1.0
            Use :meth:`list_experiments` for experiment configs or access BaseConfigManager
            methods for series/block configs.
        """
        warnings.warn(
            "list_configs() is deprecated. Use list_experiments() for experiments "
            "or use BaseConfigManager methods for series/block configs.",
            DeprecationWarning,
            stacklevel=2
        )
        configs = []
        for path in self.config_dir.glob("*.yaml"):
            configs.append(path.stem)
        return configs
    
    def validate_config(self, content: str, config_type: Optional[str] = None):
        """Validate YAML content.
        
        Args:
            content: YAML content to validate
            config_type: Optional config type (experiment/series/block) for better error context
        """
        try:
            yaml.safe_load(content)
        except yaml.YAMLError as e:
            config_context = f" in {config_type} config" if config_type else ""
            error_msg = format_error_message(
                operation="Validating YAML content",
                reason=f"Invalid YAML syntax{config_context}: {str(e)}",
                suggestion="Check YAML syntax, indentation, and ensure all brackets/quotes are properly closed"
            )
            raise ValidationError(error_msg) from e
    
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
        else:
            config_path = self.experiment_dir / f"{experiment_id}.yaml"
        
        content = self._read_yaml_file(config_path)
        config_dict = yaml.safe_load(content)
        model_type = self._infer_model_type(config_dict)
        
        return {
            "experiment_id": experiment_id,
            "model_type": model_type,
            "content": content
        }
    
    def get_experiment_unified(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment config as unified structure (parsed dict).
        
        Returns the config in the unified structure format:
        {
            "experiment_id": str,
            "model_type": str,
            "config": {
                "name": str,
                "description": str,
                "model_type": str,
                "preprocess": {
                    "metadata": {...},
                    "series": [...]
                },
                ...
            }
        }
        """
        if experiment_id == "default":
            config_path = self.config_dir / "default.yaml"
        else:
            config_path = self.experiment_dir / f"{experiment_id}.yaml"
        
        content = self._read_yaml_file(config_path)
        config_dict = yaml.safe_load(content)
        model_type = config_dict.get("model_type") or self._infer_model_type(config_dict)
        
        return {
            "experiment_id": experiment_id,
            "model_type": model_type,
            "config": config_dict
        }
    
    def update_experiment_unified(self, experiment_id: str, config: Dict[str, Any]) -> None:
        """Update experiment config from unified structure (dict).
        
        Args:
            experiment_id: Experiment ID
            config: Config dictionary in unified structure format
        """
        # Convert dict to YAML
        content = yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)
        self.update_experiment(experiment_id, content)
    
    def create_experiment(
        self,
        experiment_id: Optional[str] = None,
        model_type: str = "dfm",
        content: Optional[str] = None,
        auto_generate_name: bool = True
    ) -> str:
        """Create a new experiment config with unique ID.
        
        Args:
            experiment_id: Base experiment ID. If None, generates unique ID.
            model_type: Model type ("dfm" or "ddfm")
            content: YAML content. If None, creates minimal config.
            auto_generate_name: Whether to auto-generate unique ID.
            
        Returns:
            Unique experiment ID that was created
        """
        if experiment_id == "default":
            error_msg = format_error_message(
                operation="Creating experiment",
                reason="Cannot create 'default' experiment (reserved name)",
                suggestion="Use update_experiment('default', content) to modify the default experiment"
            )
            raise ConfigError(error_msg)
        
        # Generate unique name
        if auto_generate_name or experiment_id is None:
            unique_id = self._base_manager.generate_unique_experiment_name(experiment_id)
        else:
            unique_id = experiment_id
        
        # Validate content if provided
        if content:
            self.validate_config(content, config_type="experiment")
        else:
            # Create minimal config
            minimal_config = {
                'defaults': [
                    {'override /model': model_type},
                    '_self_'
                ],
                'name': unique_id,
                'description': f'Auto-generated experiment: {unique_id}',
                'data': {
                    'path': 'data/sample_data.csv'
                },
                'output': {
                    'path': f'outputs/{unique_id}',
                    'save_model': True,
                    'save_plots': True,
                    'save_results': True
                },
                'series': [],
                'target': None
            }
            content = yaml.dump(minimal_config, default_flow_style=False, allow_unicode=True)
        
        # Update name in content if it's YAML
        try:
            config_dict = yaml.safe_load(content)
            if isinstance(config_dict, dict):
                config_dict['name'] = unique_id
                content = yaml.dump(config_dict, default_flow_style=False, allow_unicode=True)
        except Exception:
            pass
        
        config_path = self.experiment_dir / f"{unique_id}.yaml"
        if config_path.exists():
            error_msg = format_error_message(
                operation="Creating experiment",
                reason=f"Experiment '{unique_id}' already exists at '{config_path}'",
                suggestion="Use update_experiment() to modify existing experiment or choose a different experiment ID"
            )
            raise ConfigError(error_msg)
        
        self._write_yaml_file(config_path, content)
        return unique_id
    
    def update_experiment(self, experiment_id: str, content: str):
        """Update experiment config."""
        self.validate_config(content, config_type="experiment")
        
        if experiment_id == "default":
            config_path = self.config_dir / "default.yaml"
        else:
            config_path = self.experiment_dir / f"{experiment_id}.yaml"
        
        self._write_yaml_file(config_path, content)
    
    # Model config management
    def list_model_configs(self) -> List[str]:
        """List available model config types.
        
        Returns:
            List of model type names (e.g., ['dfm', 'ddfm', 'arma', 'tft', 'xgboost'])
        """
        model_types = []
        for path in self.model_dir.glob("*.yaml"):
            model_types.append(path.stem)
        return sorted(model_types)
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get model config as dictionary.
        
        Args:
            model_type: Model type (dfm, ddfm, arma, tft, xgboost)
            
        Returns:
            Model config as dictionary
        """
        config_path = self.model_dir / f"{model_type}.yaml"
        if not config_path.exists():
            return {}
        
        content = self._read_yaml_file(config_path)
        return yaml.safe_load(content) or {}
    
    def update_model_config(self, model_type: str, config: Dict[str, Any]) -> None:
        """Update model config.
        
        Args:
            model_type: Model type (dfm, ddfm, arma, tft, xgboost)
            config: Config dictionary
        """
        config_path = self.model_dir / f"{model_type}.yaml"
        content = yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)
        self._write_yaml_file(config_path, content)
    
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
        return self._read_yaml_file(config_path)
    
    def update_series_config(self, series_name: str, content: str):
        """Update series config."""
        self.validate_config(content, config_type="series")
        
        config_path = self.series_dir / f"{series_name}.yaml"
        self._write_yaml_file(config_path, content)
    
    # Block config management - REMOVED
    # Blocks are not used in the current architecture
    # Block information is stored directly in series configs
    
    def get_experiment_config_path(self, experiment_id: str) -> str:
        """Get the config path for an experiment (for training)."""
        if experiment_id == "default":
            return str(self.config_dir / "default.yaml")
        else:
            return str(self.experiment_dir / f"{experiment_id}.yaml")
    
    # ========================================================================
    # CSV Import and Update Methods
    # ========================================================================
    
    def import_configs_from_csv(
        self,
        csv_path: Optional[Union[str, Path]] = None,
        overwrite: bool = True,
        update_series: bool = True,
        update_experiments: bool = True
    ) -> Dict[str, Any]:
        """Import configurations from metadata CSV file.
        
        Args:
            csv_path: Path to metadata.csv. If None, uses data/metadata.csv
            overwrite: Whether to overwrite existing configs
            update_series: Whether to update series configs
            update_experiments: Whether to update experiment configs
            
        Returns:
            Dictionary with update statistics
        """
        if csv_path is None:
            csv_path = self.config_dir.parent / "data" / "metadata.csv"
        
        validate_file_exists(
            Path(csv_path),
            ConfigError,
            f"CSV file not found: {csv_path}"
        )
        
        if self._updater is None:
            self._updater = ConfigUpdater(str(self.config_dir), str(csv_path))
        
        return self._updater.update_all_from_csv(
            overwrite=overwrite,
            update_series=update_series,
            update_experiments=update_experiments
        )
    
    def sync_configs_from_csv(
        self,
        csv_path: Optional[Union[str, Path]] = None,
        remove_orphaned: bool = False
    ) -> Dict[str, Any]:
        """Sync config files with CSV metadata.
        
        Args:
            csv_path: Path to metadata.csv. If None, uses data/metadata.csv
            remove_orphaned: Whether to remove configs not in CSV
            
        Returns:
            Dictionary with sync statistics
        """
        if csv_path is None:
            csv_path = self.config_dir.parent / "data" / "metadata.csv"
        
        validate_file_exists(
            Path(csv_path),
            ConfigError,
            f"CSV file not found: {csv_path}"
        )
        
        if self._updater is None:
            self._updater = ConfigUpdater(str(self.config_dir), str(csv_path))
        
        return self._updater.sync_from_csv(remove_orphaned=remove_orphaned)
    
    def update_series_from_csv(
        self,
        series_ids: Optional[List[str]] = None,
        csv_path: Optional[Union[str, Path]] = None,
        overwrite: bool = True
    ) -> Dict[str, int]:
        """Update specific series configs from CSV.
        
        Args:
            series_ids: List of series IDs to update. If None, updates all.
            csv_path: Path to metadata.csv. If None, uses data/metadata.csv
            overwrite: Whether to overwrite existing configs
            
        Returns:
            Dictionary with update statistics
        """
        if csv_path is None:
            csv_path = self.config_dir.parent / "data" / "metadata.csv"
        
        validate_file_exists(
            Path(csv_path),
            ConfigError,
            f"CSV file not found: {csv_path}"
        )
        
        if self._updater is None:
            self._updater = ConfigUpdater(str(self.config_dir), str(csv_path))
        
        return self._updater.update_series_from_csv(series_ids=series_ids, overwrite=overwrite)
    
    def update_experiments_from_csv(
        self,
        experiment_names: Optional[List[str]] = None,
        csv_path: Optional[Union[str, Path]] = None,
        overwrite: bool = True
    ) -> Dict[str, int]:
        """Update specific experiment configs from CSV.
        
        Args:
            experiment_names: List of experiment IDs to update. If None, updates all.
                Note: Parameter name kept for backward compatibility with base manager.
            csv_path: Path to metadata.csv. If None, uses data/metadata.csv
            overwrite: Whether to overwrite existing configs
            
        Returns:
            Dictionary with update statistics
        """
        if csv_path is None:
            csv_path = self.config_dir.parent / "data" / "metadata.csv"
        
        validate_file_exists(
            Path(csv_path),
            ConfigError,
            f"CSV file not found: {csv_path}"
        )
        
        if self._updater is None:
            self._updater = ConfigUpdater(str(self.config_dir), str(csv_path))
        
        return self._updater.update_experiments_from_csv(
            experiment_names=experiment_names,
            overwrite=overwrite
        )

