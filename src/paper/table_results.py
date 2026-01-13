"""Aggregate experiment results into CSV format.

Scans output directories and creates a CSV file with metrics (sMSE, sMAE)
for each model, dataset, experiment type, and horizon.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import re

from src.utils import get_project_root

logger = logging.getLogger(__name__)


def extract_horizon_from_path(path: Path) -> Optional[str]:
    """Extract horizon from path like 'horizon_12w' or 'horizon_4w'.
    
    Parameters
    ----------
    path : Path
        Path object containing horizon info
    
    Returns
    -------
    str or None
        Horizon string (e.g., "4w", "12w") or None if not found
    """
    path_str = str(path)
    match = re.search(r'horizon_(\d+w)', path_str)
    if match:
        return match.group(1)
    return None


def load_metrics_from_experiment(
    output_base: Path,
    experiment_type: str,
    data_model: str,
    model_name: str,
    horizon: Optional[str] = None
) -> Optional[Dict[str, float]]:
    """Load metrics from a specific experiment.
    
    Parameters
    ----------
    output_base : Path
        Base output directory (usually outputs/)
    experiment_type : str
        "short_term" or "long_term"
    data_model : str
        "investment" or "production"
    model_name : str
        Model name (e.g., "tft", "patchtst", "itf")
    horizon : str, optional
        Horizon for long_term experiments (e.g., "4w", "12w")
    
    Returns
    -------
    dict or None
        Metrics dictionary with 'smse' and 'smae' keys, or None if not found
    """
    if horizon:
        metrics_file = output_base / experiment_type / data_model / model_name / f"horizon_{horizon}" / "metrics.json"
    else:
        metrics_file = output_base / experiment_type / data_model / model_name / "metrics.json"
    
    if not metrics_file.exists():
        return None
    
    try:
        with open(metrics_file) as f:
            metrics = json.load(f)
            # Ensure both smse and smae exist
            if 'smse' in metrics and 'smae' in metrics:
                return {
                    'smse': metrics['smse'],
                    'smae': metrics['smae']
                }
            else:
                logger.warning(f"Metrics file {metrics_file} missing smse or smae")
                return None
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Error loading metrics from {metrics_file}: {e}")
        return None


def aggregate_all_results(
    output_base: Optional[Path] = None,
    models: Optional[List[str]] = None,
    data_models: Optional[List[str]] = None
) -> pd.DataFrame:
    """Aggregate all experiment results into a DataFrame.
    
    Parameters
    ----------
    output_base : Path, optional
        Base output directory. If None, uses outputs/ from project root.
    models : list of str, optional
        List of models to include. If None, discovers from directories.
    data_models : list of str, optional
        List of datasets to include. If None, uses ['investment', 'production'].
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: data_model, experiment_type, model, horizon, smse, smae
    """
    if output_base is None:
        output_base = get_project_root() / "outputs"
    
    if data_models is None:
        data_models = ['investment', 'production']
    
    if models is None:
        # Discover models from short_term directories
        models = []
        for data_model in data_models:
            short_term_dir = output_base / "short_term" / data_model
            if short_term_dir.exists():
                for item in short_term_dir.iterdir():
                    if item.is_dir() and (item / "metrics.json").exists():
                        if item.name not in models:
                            models.append(item.name)
        if not models:
            models = ['tft', 'patchtst', 'itf', 'dfm', 'ddfm']  # Default fallback
    
    results = []
    
    # Process short-term experiments
    logger.info("Processing short-term experiments...")
    for data_model in data_models:
        for model in models:
            metrics = load_metrics_from_experiment(
                output_base, "short_term", data_model, model, horizon=None
            )
            if metrics:
                results.append({
                    'data_model': data_model,
                    'experiment_type': 'short_term',
                    'model': model,
                    'horizon': '1-step',
                    'smse': metrics['smse'],
                    'smae': metrics['smae']
                })
                logger.debug(f"Loaded short_term/{data_model}/{model}: sMSE={metrics['smse']:.4f}, sMAE={metrics['smae']:.4f}")
    
    # Process long-term experiments
    logger.info("Processing long-term experiments...")
    for data_model in data_models:
        for model in models:
            long_term_dir = output_base / "long_term" / data_model / model
            if not long_term_dir.exists():
                continue
            
            # Find all horizon directories
            for item in long_term_dir.iterdir():
                if item.is_dir() and item.name.startswith('horizon_'):
                    horizon = extract_horizon_from_path(item)
                    if horizon:
                        metrics = load_metrics_from_experiment(
                            output_base, "long_term", data_model, model, horizon=horizon
                        )
                        if metrics:
                            results.append({
                                'data_model': data_model,
                                'experiment_type': 'long_term',
                                'model': model,
                                'horizon': horizon,
                                'smse': metrics['smse'],
                                'smae': metrics['smae']
                            })
                            logger.debug(f"Loaded long_term/{data_model}/{model}/{horizon}: sMSE={metrics['smse']:.4f}, sMAE={metrics['smae']:.4f}")
    
    if not results:
        logger.warning("No results found!")
        return pd.DataFrame(columns=['data_model', 'experiment_type', 'model', 'horizon', 'smse', 'smae'])
    
    df = pd.DataFrame(results)
    
    # Sort by data_model, experiment_type, model, then horizon
    # For horizon, sort numerically (4w, 8w, 12w, ...) not alphabetically
    df['horizon_num'] = df['horizon'].apply(lambda x: int(x.rstrip('w')) if isinstance(x, str) and x.endswith('w') else (0 if x == '1-step' else 999))
    df = df.sort_values(['data_model', 'experiment_type', 'model', 'horizon_num'])
    df = df.drop(columns=['horizon_num'])
    
    logger.info(f"Aggregated {len(df)} results")
    
    return df


def create_results_table(
    output_path: Optional[Path] = None,
    output_base: Optional[Path] = None
) -> pd.DataFrame:
    """Create and save results table.
    
    Parameters
    ----------
    output_path : Path, optional
        Path to save CSV file. If None, saves to outputs/results_table.csv
    output_base : Path, optional
        Base output directory. If None, uses outputs/ from project root.
    
    Returns
    -------
    pd.DataFrame
        Aggregated results DataFrame
    """
    df = aggregate_all_results(output_base=output_base)
    
    if output_path is None:
        output_path = get_project_root() / "outputs" / "results_table.csv"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved results table to: {output_path}")
    
    return df


def main():
    """Main entry point for creating results table."""
    parser = argparse.ArgumentParser(
        description="Aggregate experiment results into CSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create default results table
  python -m src.paper.table_results
  
  # Save to specific location
  python -m src.paper.table_results --output results/summary.csv
        """
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path. Default: outputs/results_table.csv'
    )
    
    parser.add_argument(
        '--output-base',
        type=str,
        default=None,
        help='Base output directory. Default: outputs/ from project root'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Convert paths
    output_path = Path(args.output) if args.output else None
    output_base = Path(args.output_base) if args.output_base else None
    
    # Create table
    df = create_results_table(output_path=output_path, output_base=output_base)
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nTotal results: {len(df)}")
    print(f"\nBy experiment type:")
    print(df.groupby('experiment_type').size())
    print(f"\nBy data model:")
    print(df.groupby('data_model').size())
    print(f"\nBy model:")
    print(df.groupby('model').size())
    
    if len(df) > 0:
        print(f"\nFirst few rows:")
        print(df.head(10).to_string(index=False))
        print(f"\nResults table saved to: {output_path or (get_project_root() / 'outputs' / 'results_table.csv')}")


if __name__ == '__main__':
    main()
