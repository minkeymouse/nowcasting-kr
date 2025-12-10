"""Aggregate forecast results from JSON files into CSV format for report generation.

This script collects all comparison_results.json files from outputs/ directories
and aggregates them into a single CSV file for table generation.
"""

import json
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_metrics_from_result(result: Dict[str, Any], model: str, target: str) -> List[Dict[str, Any]]:
    """Extract forecast metrics from a single result dictionary.
    
    Returns a list of dictionaries, one per horizon.
    """
    rows = []
    
    metrics = result.get('metrics', {})
    forecast_metrics = metrics.get('forecast_metrics', {})
    
    if not forecast_metrics:
        return rows
    
    for horizon_str, horizon_metrics in forecast_metrics.items():
        try:
            horizon = int(horizon_str)
            smae = horizon_metrics.get('sMAE', float('nan'))
            smse = horizon_metrics.get('sMSE', float('nan'))
            mae = horizon_metrics.get('MAE', float('nan'))
            mse = horizon_metrics.get('MSE', float('nan'))
            
            rows.append({
                'model': model.upper(),
                'target': target,
                'horizon': horizon,
                'sMAE': smae,
                'sMSE': smse,
                'MAE': mae,
                'MSE': mse
            })
        except (ValueError, TypeError):
            continue
    
    return rows


def aggregate_results() -> pd.DataFrame:
    """Aggregate all forecast results from outputs directories."""
    all_rows = []
    
    # Find all comparison_results.json files
    json_files = list(OUTPUTS_DIR.rglob("comparison_results.json"))
    
    if not json_files:
        logger.warning(f"No comparison_results.json files found in {OUTPUTS_DIR}")
        return pd.DataFrame()
    
    logger.info(f"Found {len(json_files)} result files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            target = data.get('target_series', 'UNKNOWN')
            results = data.get('results', {})
            
            for model, result in results.items():
                if result.get('status') == 'completed':
                    rows = extract_metrics_from_result(result, model, target)
                    all_rows.extend(rows)
                    logger.debug(f"Extracted {len(rows)} horizons from {target}/{model}")
        
        except Exception as e:
            logger.warning(f"Failed to process {json_file}: {e}")
            continue
    
    if not all_rows:
        logger.warning("No forecast metrics found in any result files")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_rows)
    logger.info(f"Aggregated {len(df)} rows from {len(json_files)} files")
    
    return df


def main():
    """Main function to aggregate results and save to CSV."""
    logger.info("=" * 70)
    logger.info("Aggregating Forecast Results")
    logger.info("=" * 70)
    
    df = aggregate_results()
    
    if df.empty:
        logger.error("No data to aggregate. Make sure training has completed and generated results.")
        return
    
    # Save aggregated results
    output_file = EXPERIMENTS_DIR / "aggregated_results.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Saved aggregated results to: {output_file}")
    logger.info(f"  Total rows: {len(df)}")
    logger.info(f"  Models: {df['model'].unique().tolist()}")
    logger.info(f"  Targets: {df['target'].unique().tolist()}")
    logger.info(f"  Horizons: {df['horizon'].min()} to {df['horizon'].max()}")
    
    # Summary statistics
    summary = df.groupby(['model', 'target']).agg({
        'sMAE': 'mean',
        'sMSE': 'mean',
        'MAE': 'mean',
        'MSE': 'mean'
    }).reset_index()
    logger.info("\nSummary (average across horizons):")
    logger.info(summary.to_string(index=False))
    
    logger.info("=" * 70)
    logger.info("Aggregation complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

