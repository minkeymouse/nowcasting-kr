"""Aggregate comparison results from all targets into a single CSV file.

This script reads comparison_results.json files from outputs/comparisons/{target}/
and aggregates them into a single CSV file at outputs/experiments/aggregated_results.csv.

The aggregated CSV contains metrics for each target, model, and horizon combination.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional

from src.utils import get_project_root, setup_paths

# Setup paths
setup_paths(include_dfm_python=True, include_src=True)
project_root = get_project_root()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Paths
OUTPUTS_DIR = project_root / "outputs"
COMPARISONS_DIR = OUTPUTS_DIR / "comparisons"
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"
AGGREGATED_FILE = EXPERIMENTS_DIR / "aggregated_results.csv"


def load_comparison_results(target_dir: Path) -> Optional[Dict[str, Any]]:
    """Load comparison_results.json from a target directory.
    
    Parameters
    ----------
    target_dir : Path
        Directory containing comparison_results.json
        
    Returns
    -------
    Optional[Dict[str, Any]]
        Comparison results data, or None if file not found or invalid
    """
    results_file = target_dir / "comparison_results.json"
    
    if not results_file.exists():
        logger.warning(f"Comparison results not found: {results_file}")
        return None
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Failed to load {results_file}: {e}")
        return None


def extract_metrics_from_results(
    comparison_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Extract metrics from comparison results into flat rows.
    
    Parameters
    ----------
    comparison_data : Dict[str, Any]
        Comparison results data loaded from JSON
        
    Returns
    -------
    List[Dict[str, Any]]
        List of metric rows, each with target, model, horizon, and metric values
    """
    target_series = comparison_data.get('target_series', 'unknown')
    horizons = comparison_data.get('horizons', [])
    results = comparison_data.get('results', {})
    
    rows = []
    
    for model_name, model_result in results.items():
        # Skip failed or skipped models
        status = model_result.get('status', 'unknown')
        if status not in ['completed', 'success']:
            continue
        
        # Extract metrics
        metrics = model_result.get('metrics', {})
        forecast_metrics = metrics.get('forecast_metrics', {})
        
        # Process each horizon
        for horizon in horizons:
            horizon_str = str(horizon)
            horizon_metrics = forecast_metrics.get(horizon_str, {})
            
            # Extract all available metrics
            row = {
                'target': target_series,
                'model': model_name.upper(),  # Standardize to uppercase
                'horizon': horizon
            }
            
            # Add metric values
            metric_names = ['sMSE', 'sMAE', 'sRMSE', 'MSE', 'MAE', 'RMSE']
            for metric in metric_names:
                value = horizon_metrics.get(metric, np.nan)
                row[metric] = value if not np.isnan(value) else None
            
            # Only add row if we have at least one valid metric
            if any(row.get(m) is not None for m in metric_names):
                rows.append(row)
    
    return rows


def aggregate_all_results() -> pd.DataFrame:
    """Aggregate results from all target directories.
    
    Returns
    -------
    pd.DataFrame
        Aggregated results DataFrame with columns: target, model, horizon, sMSE, sMAE, sRMSE, MSE, MAE, RMSE
    """
    if not COMPARISONS_DIR.exists():
        logger.error(f"Comparisons directory not found: {COMPARISONS_DIR}")
        return pd.DataFrame()
    
    all_rows = []
    target_dirs = [d for d in COMPARISONS_DIR.iterdir() if d.is_dir()]
    
    if not target_dirs:
        logger.warning(f"No target directories found in {COMPARISONS_DIR}")
        return pd.DataFrame()
    
    logger.info(f"Found {len(target_dirs)} target directories")
    
    for target_dir in target_dirs:
        target_name = target_dir.name
        logger.info(f"Processing {target_name}...")
        
        comparison_data = load_comparison_results(target_dir)
        if comparison_data is None:
            continue
        
        rows = extract_metrics_from_results(comparison_data)
        all_rows.extend(rows)
        logger.info(f"  Extracted {len(rows)} metric rows from {target_name}")
    
    if not all_rows:
        logger.warning("No metrics extracted from any comparison results")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(all_rows)
    
    # Ensure all metric columns exist
    metric_columns = ['target', 'model', 'horizon', 'sMSE', 'sMAE', 'sRMSE', 'MSE', 'MAE', 'RMSE']
    for col in metric_columns:
        if col not in df.columns:
            df[col] = None
    
    # Reorder columns
    df = df[metric_columns]
    
    # Sort by target, model, horizon
    df = df.sort_values(['target', 'model', 'horizon']).reset_index(drop=True)
    
    logger.info(f"Aggregated {len(df)} metric rows from {len(target_dirs)} targets")
    logger.info(f"Models: {sorted(df['model'].unique())}")
    logger.info(f"Targets: {sorted(df['target'].unique())}")
    
    return df


def main():
    """Main function to aggregate results and save to CSV."""
    logger.info("=" * 70)
    logger.info("Aggregating Comparison Results")
    logger.info("=" * 70)
    
    # Create experiments directory
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Aggregate results
    df = aggregate_all_results()
    
    if df.empty:
        logger.error("No results to aggregate. Make sure comparison results exist.")
        return 1
    
    # Save to CSV
    df.to_csv(AGGREGATED_FILE, index=False)
    logger.info(f"Saved aggregated results to: {AGGREGATED_FILE}")
    logger.info(f"Total rows: {len(df)}")
    
    # Print summary
    logger.info("\nSummary by model:")
    summary = df.groupby('model').agg({
        'sMSE': 'mean',
        'sMAE': 'mean',
        'sRMSE': 'mean',
        'MSE': 'mean',
        'MAE': 'mean',
        'RMSE': 'mean'
    }).round(4)
    logger.info(f"\n{summary}")
    
    logger.info("=" * 70)
    logger.info("Aggregation complete!")
    logger.info("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
