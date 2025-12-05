"""Aggregate and organize experiment results according to RESULTS_NEEDED.md structure."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import pandas as pd
import numpy as np
from datetime import datetime


def collect_all_comparison_results(outputs_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Collect all comparison results from outputs/comparisons/."""
    if outputs_dir is None:
        outputs_dir = Path(__file__).parent.parent.parent / "outputs"
    
    comparisons_dir = outputs_dir / "comparisons"
    if not comparisons_dir.exists():
        return {}
    
    all_results = {}
    
    for comparison_dir in comparisons_dir.iterdir():
        if not comparison_dir.is_dir():
            continue
        
        results_file = comparison_dir / "comparison_results.json"
        if not results_file.exists():
            continue
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            target_series = data.get('target_series')
            if target_series:
                if target_series not in all_results:
                    all_results[target_series] = []
                all_results[target_series].append(data)
        except Exception as e:
            print(f"Warning: Could not load {results_file}: {e}")
            continue
    
    return all_results


def aggregate_overall_performance(all_results: Dict[str, Any]) -> pd.DataFrame:
    """Aggregate overall performance metrics across all models, targets, and horizons."""
    rows = []
    
    for target_series, results_list in all_results.items():
        for result_data in results_list:
            comparison = result_data.get('comparison', {})
            metrics_table = comparison.get('metrics_table')
            
            if metrics_table is None:
                continue
            
            # Convert to DataFrame if it's a dict
            if isinstance(metrics_table, dict):
                df = pd.DataFrame(metrics_table)
            elif isinstance(metrics_table, pd.DataFrame):
                df = metrics_table
            else:
                continue
            
            # Add target_series column
            df['target_series'] = target_series
            
            rows.append(df)
    
    if not rows:
        return pd.DataFrame()
    
    # Concatenate all DataFrames
    aggregated = pd.concat(rows, ignore_index=True)
    return aggregated


def main():
    """Main entry point for aggregator module."""
    print("=" * 70)
    print("Aggregating Experiment Results")
    print("=" * 70)
    
    # Collect all results
    all_results = collect_all_comparison_results()
    
    if not all_results:
        print("No comparison results found in outputs/comparisons/")
        return
    
    print(f"\nFound results for {len(all_results)} target series:")
    for target, results in all_results.items():
        print(f"  - {target}: {len(results)} comparison(s)")
    
    # Aggregate performance
    aggregated = aggregate_overall_performance(all_results)
    
    if aggregated.empty:
        print("\nNo metrics to aggregate.")
        return
    
    # Save aggregated results
    outputs_dir = Path(__file__).parent.parent.parent / "outputs"
    experiments_dir = outputs_dir / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = experiments_dir / "aggregated_results.csv"
    aggregated.to_csv(output_file, index=False)
    
    print(f"\n✓ Aggregated results saved to: {output_file}")
    print(f"  Total rows: {len(aggregated)}")
    print(f"  Columns: {', '.join(aggregated.columns)}")


if __name__ == "__main__":
    main()
