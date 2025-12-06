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

# Set up paths first before any relative imports
# This allows the script to be run directly as python3 src/train.py
_script_dir = Path(__file__).parent.resolve()
_project_root = _script_dir.parent.resolve()  # src/ -> project root
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

# Now use absolute imports
from src.utils.config_parser import setup_paths, get_project_root
setup_paths(include_dfm_python=True, include_src=True, include_app=True)

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
        config_dir = str(get_project_root() / "config")
    
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
    overrides: Optional[List[str]] = None,
    models_filter: Optional[List[str]] = None
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
    models_filter : list of str, optional
        Filter models to run (e.g., ['var', 'dfm']). If None, runs all models from config.
        
    Returns
    -------
    dict
        Comparison result dictionary with keys: target_series, models, horizons, results, comparison, etc.
    """
    if config_dir is None:
        config_dir = str(get_project_root() / "config")
    
    cfg = parse_experiment_config(config_name, config_dir, overrides)
    validate_experiment_config(cfg, require_target=True, require_models=True)
    
    params = extract_experiment_params(cfg)
    
    # Filter models if models_filter is provided
    models_to_run = params['models']
    if models_filter:
        models_to_run = [m for m in models_to_run if m.lower() in [mf.lower() for mf in models_filter]]
        if not models_to_run:
            raise ValueError(f"No models match filter {models_filter}. Available models: {params['models']}")
    
    result = compare_models(
        target_series=params['target_series'],
        models=models_to_run,
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
    compare_parser.add_argument("--models", nargs="+", help="Filter models to run (e.g., --models arima var). If not specified, runs all models from config.")
    
    args = parser.parse_args()
    
    config_path = str(get_project_root() / "config")
    
    if args.command == 'train':
        result = train_model(
            config_name=args.config_name,
            config_dir=config_path,
            overrides=args.override
        )
        print(f"\n✓ Model saved to: {result['model_dir']}")
        
    elif args.command == 'compare':
        # Filter models if --models flag is provided
        overrides = list(args.override) if args.override else []
        
        result = compare_models_by_config(
            config_name=args.config_name,
            config_dir=config_path,
            overrides=overrides,
            models_filter=args.models
        )
        print(f"\n✓ Comparison saved to: {result['output_dir']}")
        if result.get('failed_models'):
            print(f"  Failed: {', '.join(result['failed_models'])}")


if __name__ == "__main__":
    main()

