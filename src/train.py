"""Training module - supports both CLI and programmatic API.

This module provides training functionality that can be used:
- As CLI: python src/train.py train --config-name experiment/kogdp_report
- As API: from src.train import train_model, compare_models_by_config

Functions exported for programmatic use:
- train_model: Train a single model
- compare_models_by_config: Compare multiple models from experiment config
"""

from pathlib import Path
import sys
import argparse
from typing import Dict, Any, Optional, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "dfm-python" / "src"))

# Set up paths
try:
    from src.utils.path_setup import setup_paths
except ImportError:
    from utils.path_setup import setup_paths

setup_paths(include_dfm_python=True, include_src=True, include_app=True)

try:
    from omegaconf import OmegaConf
except ImportError as e:
    raise ImportError(f"Required dependencies not available: {e}")

from src.core.training import train, compare_models
from src.utils.config_parser import parse_experiment_config, extract_experiment_params, validate_experiment_config


def train_model(
    config_name: str,
    config_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    overrides: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Train a single model (programmatic API).
    
    Parameters
    ----------
    config_name : str
        Experiment config name (e.g., 'experiment/kogdp_report')
    config_dir : str, optional
        Config directory path. If None, uses default config/ directory.
    model_name : str, optional
        Model name. If None, uses first model from config or auto-generates.
    overrides : list of str, optional
        Hydra config overrides (e.g., ['model_overrides.dfm.max_iter=10'])
        
    Returns
    -------
    dict
        Training result dictionary with keys: status, model_name, model_dir, metrics, etc.
    """
    if config_dir is None:
        config_dir = str(project_root / "config")
    
    cfg = parse_experiment_config(config_name, config_dir, overrides)
    validate_experiment_config(cfg, require_target=False, require_models=True)
    
    params = extract_experiment_params(cfg)
    models = params['models']
    
    # Use first model if model_name not specified
    if model_name is None:
        model_name = models[0] if models else None
        if len(models) > 1:
            print(f"Warning: Config specifies {len(models)} models, training only first: {model_name}")
    
    if not model_name:
        raise ValueError(f"Config {config_name} must specify at least one model in 'models' list")
    
    result = train(
        config_name=config_name,
        config_path=config_dir,
        model_name=model_name,
        config_overrides=overrides or []
    )
    
    return result


def compare_models_by_config(
    config_name: str,
    config_dir: Optional[str] = None,
    overrides: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Compare multiple models from experiment config (programmatic API).
    
    Parameters
    ----------
    config_name : str
        Experiment config name (e.g., 'experiment/kogdp_report')
    config_dir : str, optional
        Config directory path. If None, uses default config/ directory.
    overrides : list of str, optional
        Hydra config overrides
        
    Returns
    -------
    dict
        Comparison result dictionary with keys: target_series, models, horizons, results, comparison, etc.
    """
    if config_dir is None:
        config_dir = str(project_root / "config")
    
    cfg = parse_experiment_config(config_name, config_dir, overrides)
    validate_experiment_config(cfg, require_target=True, require_models=True)
    
    params = extract_experiment_params(cfg)
    
    result = compare_models(
        target_series=params['target_series'],
        models=params['models'],
        horizons=params['horizons'],
        data_path=params['data_path'],
        config_dir=config_dir,
        config_name=config_name,
        config_overrides=overrides
    )
    
    return result


def main():
    """Main CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train models using Hydra config")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    train_parser = subparsers.add_parser('train', help='Train single model (requires experiment config)')
    train_parser.add_argument("--config-name", required=True, help="Experiment config name (e.g., experiment/kogdp_report)")
    train_parser.add_argument("--override", action="append", help="Hydra config override (e.g., model_overrides.dfm.max_iter=10)")
    
    compare_parser = subparsers.add_parser('compare', help='Compare multiple models (requires experiment config)')
    compare_parser.add_argument("--config-name", required=True, help="Experiment config name (e.g., experiment/kogdp_report)")
    compare_parser.add_argument("--override", action="append", help="Hydra config override")
    
    args = parser.parse_args()
    
    config_path = str(project_root / "config")
    
    if args.command == 'train':
        result = train_model(
            config_name=args.config_name,
            config_dir=config_path,
            overrides=args.override
        )
        print(f"\n✓ Model saved to: {result['model_dir']}")
        
    elif args.command == 'compare':
        result = compare_models_by_config(
            config_name=args.config_name,
            config_dir=config_path,
            overrides=args.override
        )
        print(f"\n✓ Comparison saved to: {result['output_dir']}")
        if result.get('failed_models'):
            print(f"  Failed: {', '.join(result['failed_models'])}")


if __name__ == "__main__":
    main()

