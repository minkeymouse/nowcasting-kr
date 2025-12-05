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
        outputs_dir = Path(__file__).parent.parent / "outputs"
    
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
            comparison = result_data.get('comparison')
            if not comparison:
                continue
            
            metrics_table = comparison.get('metrics_table')
            if metrics_table is None:
                continue
            
            # Convert to DataFrame if needed
            if isinstance(metrics_table, dict):
                metrics_df = pd.DataFrame([metrics_table])
            else:
                metrics_df = metrics_table
            
            for _, row in metrics_df.iterrows():
                model = row.get('model', 'unknown')
                
                # Extract metrics for each horizon
                # Check for horizon-specific columns (e.g., sMSE_h1, sMAE_h1, etc.)
                for horizon in [1, 7, 28]:
                    for metric in ['sMSE', 'sMAE', 'sRMSE']:
                        # Try different column name formats
                        col_name = f"{metric}_h{horizon}"
                        if col_name in row and pd.notna(row[col_name]):
                            rows.append({
                                'model': model,
                                'target': target_series,
                                'horizon': horizon,
                                'metric': metric,
                                'value': row[col_name]
                            })
                
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    return df


def save_overall_performance_tables(aggregated_df: pd.DataFrame, output_dir: Path):
    """Save overall performance tables as per RESULTS_NEEDED.md section 1.1."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if aggregated_df.empty:
        print("Warning: No data for overall performance tables")
        return
    
    # Table 1: Overall average performance by model
    overall_avg = aggregated_df.groupby(['model', 'metric'])['value'].mean().reset_index()
    overall_avg_pivot = overall_avg.pivot(index='model', columns='metric', values='value')
    overall_avg_pivot.to_csv(output_dir / "tab_overall_metrics.csv")
    
    # Table 2: Performance by target series
    by_target = aggregated_df.groupby(['model', 'target', 'metric'])['value'].mean().reset_index()
    by_target_pivot = by_target.pivot_table(
        index=['model', 'target'], columns='metric', values='value'
    )
    by_target_pivot.to_csv(output_dir / "tab_overall_metrics_by_target.csv")
    
    # Table 3: Performance by horizon
    by_horizon = aggregated_df.groupby(['model', 'horizon', 'metric'])['value'].mean().reset_index()
    by_horizon_pivot = by_horizon.pivot_table(
        index=['model', 'horizon'], columns='metric', values='value'
    )
    by_horizon_pivot.to_csv(output_dir / "tab_overall_metrics_by_horizon.csv")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'n_models': aggregated_df['model'].nunique(),
        'n_targets': aggregated_df['target'].nunique(),
        'n_horizons': aggregated_df['horizon'].nunique(),
        'metrics': aggregated_df['metric'].unique().tolist()
    }
    with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def save_target_analysis(all_results: Dict[str, Any], output_dir: Path):
    """Save target-specific analysis as per RESULTS_NEEDED.md section 2."""
    target_map = {
        "KOGDP...D": "gdp",
        "KOCNPER.D": "consumption",
        "KOGFCF..D": "investment"
    }
    
    for target_series, results_list in all_results.items():
        target_name = target_map.get(target_series, target_series.lower())
        target_dir = output_dir / target_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Aggregate metrics for this target
        rows = []
        for result_data in results_list:
            comparison = result_data.get('comparison')
            if not comparison:
                continue
            
            metrics_table = comparison.get('metrics_table')
            if metrics_table is None:
                continue
            
            if isinstance(metrics_table, dict):
                metrics_df = pd.DataFrame([metrics_table])
            else:
                metrics_df = metrics_table
            
            for _, row in metrics_df.iterrows():
                model = row.get('model', 'unknown')
                for horizon in [1, 7, 28]:
                    for metric in ['sMSE', 'sMAE', 'sRMSE']:
                        col_name = f"{metric}_h{horizon}"
                        if col_name in row and pd.notna(row[col_name]):
                            rows.append({
                                'model': model,
                                'horizon': horizon,
                                'metric': metric,
                                'value': row[col_name]
                            })
        
        if rows:
            df = pd.DataFrame(rows)
            pivot = df.pivot_table(
                index=['model', 'horizon'], columns='metric', values='value'
            )
            pivot.to_csv(target_dir / f"tab_{target_name}_performance.csv")
            
            # Save metadata
            metadata = {
                'target_series': target_series,
                'target_name': target_name,
                'timestamp': datetime.now().isoformat(),
                'n_models': df['model'].nunique(),
                'n_horizons': df['horizon'].nunique()
            }
            with open(target_dir / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)


def save_horizon_analysis(all_results: Dict[str, Any], output_dir: Path):
    """Save horizon-specific analysis as per RESULTS_NEEDED.md section 3."""
    for horizon in [1, 7, 28]:
        horizon_dir = output_dir / f"horizon_{horizon}day"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        
        rows = []
        for target_series, results_list in all_results.items():
            for result_data in results_list:
                comparison = result_data.get('comparison')
                if not comparison:
                    continue
                
                metrics_table = comparison.get('metrics_table')
                if metrics_table is None:
                    continue
                
                if isinstance(metrics_table, dict):
                    metrics_df = pd.DataFrame([metrics_table])
                else:
                    metrics_df = metrics_table
                
                for _, row in metrics_df.iterrows():
                    model = row.get('model', 'unknown')
                    for metric in ['sMSE', 'sMAE', 'sRMSE']:
                        col_name = f"{metric}_h{horizon}"
                        if col_name in row and pd.notna(row[col_name]):
                            rows.append({
                                'model': model,
                                'target': target_series,
                                'metric': metric,
                                'value': row[col_name]
                            })
        
        if rows:
            df = pd.DataFrame(rows)
            pivot = df.pivot_table(
                index=['model', 'target'], columns='metric', values='value'
            )
            pivot.to_csv(horizon_dir / f"tab_horizon_{horizon}day.csv")
            
            # Save metadata
            metadata = {
                'horizon': horizon,
                'timestamp': datetime.now().isoformat(),
                'n_models': df['model'].nunique(),
                'n_targets': df['target'].nunique()
            }
            with open(horizon_dir / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)


def aggregate_all_results(outputs_dir: Optional[Path] = None):
    """Main function to aggregate all results according to RESULTS_NEEDED.md structure."""
    if outputs_dir is None:
        outputs_dir = Path(__file__).parent.parent / "outputs"
    
    print("=" * 70)
    print("Aggregating Results for RESULTS_NEEDED.md")
    print("=" * 70)
    
    # Collect all comparison results
    print("\n1. Collecting comparison results...")
    all_results = collect_all_comparison_results(outputs_dir)
    
    if not all_results:
        print("  No comparison results found. Run experiments first.")
        return
    
    print(f"  Found results for {len(all_results)} target series")
    
    # 1. Overall performance (section 1.1)
    print("\n2. Generating overall performance tables...")
    aggregated_df = aggregate_overall_performance(all_results)
    overall_dir = outputs_dir / "experiments" / "overall_performance"
    save_overall_performance_tables(aggregated_df, overall_dir)
    print(f"  Saved to: {overall_dir}")
    
    # 2. Target-specific analysis (section 2)
    print("\n3. Generating target-specific analysis...")
    target_dir = outputs_dir / "experiments" / "target_analysis"
    save_target_analysis(all_results, target_dir)
    print(f"  Saved to: {target_dir}")
    
    # 3. Horizon-specific analysis (section 3)
    print("\n4. Generating horizon-specific analysis...")
    horizon_dir = outputs_dir / "experiments" / "horizon_analysis"
    save_horizon_analysis(all_results, horizon_dir)
    print(f"  Saved to: {horizon_dir}")
    
    print("\n" + "=" * 70)
    print("Aggregation complete!")
    print("=" * 70)
    print(f"\nResults saved in: {outputs_dir / 'experiments'}")


if __name__ == "__main__":
    aggregate_all_results()

