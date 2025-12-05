"""Inference/Nowcasting module - supports both CLI and programmatic API.

This module provides inference and nowcasting functionality that can be used:
- As CLI: python src/infer.py nowcast --config-name experiment/kogdp_report
- As API: from src.infer import run_nowcasting_evaluation

Functions exported for programmatic use:
- run_nowcasting_evaluation: Run nowcasting evaluation with masked data
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

from src.nowcasting import (
    mask_recent_observations,
    create_nowcasting_splits,
    simulate_nowcasting_evaluation
)
from src.utils.config_parser import parse_experiment_config, extract_experiment_params, validate_experiment_config


def run_nowcasting_evaluation(
    config_name: str,
    config_dir: Optional[str] = None,
    mask_days: int = 30,
    overrides: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Run nowcasting evaluation with masked data (programmatic API).
    
    Parameters
    ----------
    config_name : str
        Experiment config name (e.g., 'experiment/kogdp_report')
    config_dir : str, optional
        Config directory path. If None, uses default config/ directory.
    mask_days : int, default 30
        Number of days to mask for nowcasting simulation
    overrides : list of str, optional
        Hydra config overrides
        
    Returns
    -------
    dict
        Nowcasting evaluation results
    """
    if config_dir is None:
        config_dir = str(project_root / "config")
    
    cfg = parse_experiment_config(config_name, config_dir, overrides)
    validate_experiment_config(cfg, require_target=True, require_models=False)
    
    params = extract_experiment_params(cfg)
    
    if not params['data_path']:
        raise ValueError(f"Config {config_name} must specify 'data_path'")
    
    print("=" * 70)
    print(f"Running nowcasting evaluation for {params['target_series']}")
    print(f"Mask days: {mask_days} | Horizons: {params['horizons']}")
    print("=" * 70)
    
    # TODO: Implement nowcasting evaluation using loaded model
    # This is a placeholder - actual implementation would:
    # 1. Load trained model from outputs/
    # 2. Load data
    # 3. Run simulate_nowcasting_evaluation
    print("\nNowcasting evaluation not yet fully implemented.")
    print("This will use simulate_nowcasting_evaluation() from nowcasting.py")
    print(f"Config: {config_name}")
    print(f"Target: {params['target_series']}")
    print(f"Data: {params['data_path']}")
    
    return {
        'status': 'not_implemented',
        'config_name': config_name,
        'target_series': params['target_series'],
        'mask_days': mask_days,
        'horizons': params['horizons']
    }


def main():
    """Main CLI entry point for inference/nowcasting."""
    parser = argparse.ArgumentParser(description="Run inference and nowcasting using Hydra config")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Nowcasting evaluation command
    nowcast_parser = subparsers.add_parser('nowcast', help='Run nowcasting evaluation (requires experiment config)')
    nowcast_parser.add_argument("--config-name", required=True, help="Experiment config name (e.g., experiment/kogdp_report)")
    nowcast_parser.add_argument("--override", action="append", help="Hydra config override")
    nowcast_parser.add_argument("--mask-days", type=int, default=30, help="Number of days to mask for nowcasting simulation")
    
    args = parser.parse_args()
    
    if args.command == 'nowcast':
        result = run_nowcasting_evaluation(
            config_name=args.config_name,
            mask_days=args.mask_days,
            overrides=args.override
        )
        print(f"\n✓ Nowcasting evaluation completed")
        if result.get('status') == 'not_implemented':
            print("  Note: Full implementation pending")


if __name__ == "__main__":
    main()

