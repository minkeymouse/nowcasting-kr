"""Generate LaTeX tables from results data."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import pandas as pd
import numpy as np
from datetime import datetime


def load_comparison_results(outputs_dir: Path) -> Dict[str, List[Dict]]:
    """Load all comparison results from outputs/comparisons/."""
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
            continue
    
    return all_results


def extract_metrics_from_results(all_results: Dict[str, List[Dict]]) -> pd.DataFrame:
    """Extract metrics from comparison results into a DataFrame."""
    rows = []
    
    models = ['ARIMA', 'VAR', 'VECM', 'DeepAR', 'TFT', 'XGBoost', 'LightGBM', 'DFM', 'DDFM']
    targets = ['KOGDP...D', 'KOCNPER.D', 'KOGFCF..D']
    horizons = [1, 7, 28]
    metrics = ['sMSE', 'sMAE', 'sRMSE']
    
    for target in targets:
        for model in models:
            for horizon in horizons:
                for metric in metrics:
                    # Try to find actual result
                    value = None
                    if target in all_results:
                        for result_data in all_results[target]:
                            comparison = result_data.get('comparison')
                            if comparison:
                                metrics_table = comparison.get('metrics_table')
                                if metrics_table:
                                    if isinstance(metrics_table, dict):
                                        metrics_table = pd.DataFrame([metrics_table])
                                    elif not isinstance(metrics_table, pd.DataFrame):
                                        continue
                                    
                                    for _, row in metrics_table.iterrows():
                                        if row.get('model', '').lower() == model.lower():
                                            col_name = f"{metric}_h{horizon}"
                                            if col_name in row and pd.notna(row[col_name]):
                                                value = row[col_name]
                                                break
                    
                    # Use None for missing values (will be displayed as "-")
                    if value is None or (isinstance(value, float) and np.isnan(value)):
                        value = None
                    
                    rows.append({
                        'model': model,
                        'target': target,
                        'horizon': horizon,
                        'metric': metric,
                        'value': value
                    })
    
    return pd.DataFrame(rows)


def format_value(value: float, decimals: int = 3) -> str:
    """Format a float value for LaTeX table."""
    if value is None or pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"{value:.{decimals}f}"


def generate_latex_table_overall_metrics(df: pd.DataFrame) -> str:
    """Generate LaTeX table for overall metrics."""
    # Aggregate by model (average across all targets and horizons)
    # Exclude None values from mean calculation
    df_valid = df[df['value'].notna()].copy()
    if len(df_valid) == 0:
        # No valid data, return empty table
        return "\\begin{table}[h]\n\\centering\n\\caption{전체 모형 성능 비교 (표준화된 지표, 전체 평균)}\n\\label{tab:overall_metrics}\n\\begin{tabular}{lccc}\n\\toprule\n모형 & sMSE & sMAE & sRMSE \\\\\n\\midrule\n\\bottomrule\n\\end{tabular}\n\\end{table}"
    
    model_avg = df_valid.groupby(['model', 'metric'])['value'].mean().reset_index()
    model_avg_pivot = model_avg.pivot(index='model', columns='metric', values='value')
    
    # Sort by sRMSE (handle None values)
    if 'sRMSE' in model_avg_pivot.columns:
        model_avg_pivot = model_avg_pivot.sort_values('sRMSE', na_position='last')
    
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{전체 모형 성능 비교 (표준화된 지표, 전체 평균)}",
        "\\label{tab:overall_metrics}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "모형 & sMSE & sMAE & sRMSE \\\\",
        "\\midrule"
    ]
    
    for model in model_avg_pivot.index:
        sMSE = format_value(model_avg_pivot.loc[model, 'sMSE'])
        sMAE = format_value(model_avg_pivot.loc[model, 'sMAE'])
        sRMSE = format_value(model_avg_pivot.loc[model, 'sRMSE'])
        lines.append(f"{model} & {sMSE} & {sMAE} & {sRMSE} \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)


def generate_latex_table_by_target(df: pd.DataFrame) -> str:
    """Generate LaTeX table for performance by target."""
    # Aggregate by model and target (average across horizons)
    # Exclude None values from mean calculation
    df_valid = df[df['value'].notna()].copy()
    if len(df_valid) == 0:
        # No valid data, return empty table
        return "\\begin{table}[h]\n\\centering\n\\caption{목표 변수별 모형 성능 비교 (표준화된 RMSE)}\n\\label{tab:overall_metrics_by_target}\n\\begin{tabular}{lccc}\n\\toprule\n모형 & GDP & 민간 소비 & 총고정자본형성 \\\\\n\\midrule\n\\bottomrule\n\\end{tabular}\n\\end{table}"
    
    target_avg = df_valid.groupby(['model', 'target', 'metric'])['value'].mean().reset_index()
    target_avg_pivot = target_avg[target_avg['metric'] == 'sRMSE'].pivot(
        index='model', columns='target', values='value'
    )
    
    # Rename targets
    target_avg_pivot.columns = ['GDP', '민간 소비', '총고정자본형성']
    
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{목표 변수별 모형 성능 비교 (표준화된 RMSE)}",
        "\\label{tab:overall_metrics_by_target}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "모형 & GDP & 민간 소비 & 총고정자본형성 \\\\",
        "\\midrule"
    ]
    
    for model in target_avg_pivot.index:
        gdp = format_value(target_avg_pivot.loc[model, 'GDP'])
        cons = format_value(target_avg_pivot.loc[model, '민간 소비'])
        inv = format_value(target_avg_pivot.loc[model, '총고정자본형성'])
        lines.append(f"{model} & {gdp} & {cons} & {inv} \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)


def generate_latex_table_by_horizon(df: pd.DataFrame) -> str:
    """Generate LaTeX table for performance by horizon."""
    # Aggregate by model and horizon (average across targets)
    # Exclude None values from mean calculation
    df_valid = df[df['value'].notna()].copy()
    if len(df_valid) == 0:
        # No valid data, return empty table
        return "\\begin{table}[h]\n\\centering\n\\caption{예측 기간별 모형 성능 비교 (표준화된 RMSE)}\n\\label{tab:overall_metrics_by_horizon}\n\\begin{tabular}{lccc}\n\\toprule\n모형 & 1일 & 7일 & 28일 \\\\\n\\midrule\n\\bottomrule\n\\end{tabular}\n\\end{table}"
    
    horizon_avg = df_valid.groupby(['model', 'horizon', 'metric'])['value'].mean().reset_index()
    horizon_avg_pivot = horizon_avg[horizon_avg['metric'] == 'sRMSE'].pivot(
        index='model', columns='horizon', values='value'
    )
    
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{예측 기간별 모형 성능 비교 (표준화된 RMSE)}",
        "\\label{tab:overall_metrics_by_horizon}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "모형 & 1일 & 7일 & 28일 \\\\",
        "\\midrule"
    ]
    
    for model in horizon_avg_pivot.index:
        h1 = format_value(horizon_avg_pivot.loc[model, 1])
        h7 = format_value(horizon_avg_pivot.loc[model, 7])
        h28 = format_value(horizon_avg_pivot.loc[model, 28])
        lines.append(f"{model} & {h1} & {h7} & {h28} \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)


def generate_latex_table_nowcasting(df: pd.DataFrame) -> str:
    """Generate LaTeX table for nowcasting comparison."""
    # Filter for DFM and DDFM only
    dfm_ddfm = df[df['model'].isin(['DFM', 'DDFM'])].copy()
    
    # Calculate averages, excluding None values
    dfm_values = dfm_ddfm[dfm_ddfm['model'] == 'DFM']['value']
    ddfm_values = dfm_ddfm[dfm_ddfm['model'] == 'DDFM']['value']
    
    dfm_avg = dfm_values.dropna().mean() if len(dfm_values.dropna()) > 0 else None
    ddfm_avg = ddfm_values.dropna().mean() if len(ddfm_values.dropna()) > 0 else None
    
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{DFM vs DDFM 나우캐스팅 성능 비교 (전체 평균)}",
        "\\label{tab:nowcasting_metrics}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "모형 & sMSE & sMAE & sRMSE \\\\",
        "\\midrule",
        f"DFM & {format_value(dfm_avg)} & {format_value(dfm_avg)} & {format_value(dfm_avg)} \\\\",
        f"DDFM & {format_value(ddfm_avg)} & {format_value(ddfm_avg)} & {format_value(ddfm_avg)} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ]
    
    return "\n".join(lines)


def generate_all_tables(outputs_dir: Optional[Path] = None,
                        tables_dir: Optional[Path] = None):
    """Generate all LaTeX tables."""
    if outputs_dir is None:
        outputs_dir = Path(__file__).parent.parent.parent / "outputs"
    if tables_dir is None:
        tables_dir = Path(__file__).parent.parent.parent / "nowcasting-report" / "tables"
    
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Generating LaTeX Tables")
    print("=" * 70)
    
    # Load results
    print("\n1. Loading comparison results...")
    all_results = load_comparison_results(outputs_dir)
    print(f"   Found results for {len(all_results)} target series")
    
    # Extract metrics
    print("\n2. Extracting metrics...")
    df = extract_metrics_from_results(all_results)
    print(f"   Extracted {len(df)} metric values")
    
    # Generate tables
    print("\n3. Generating LaTeX tables...")
    
    # Overall metrics
    table1 = generate_latex_table_overall_metrics(df)
    with open(tables_dir / "tab_overall_metrics.tex", 'w', encoding='utf-8') as f:
        f.write(table1)
    print("   Generated: tab_overall_metrics.tex")
    
    # By target
    table2 = generate_latex_table_by_target(df)
    with open(tables_dir / "tab_overall_metrics_by_target.tex", 'w', encoding='utf-8') as f:
        f.write(table2)
    print("   Generated: tab_overall_metrics_by_target.tex")
    
    # By horizon
    table3 = generate_latex_table_by_horizon(df)
    with open(tables_dir / "tab_overall_metrics_by_horizon.tex", 'w', encoding='utf-8') as f:
        f.write(table3)
    print("   Generated: tab_overall_metrics_by_horizon.tex")
    
    # Nowcasting
    table4 = generate_latex_table_nowcasting(df)
    with open(tables_dir / "tab_nowcasting_metrics.tex", 'w', encoding='utf-8') as f:
        f.write(table4)
    print("   Generated: tab_nowcasting_metrics.tex")
    
    print("\n" + "=" * 70)
    print("Table generation complete!")
    print("=" * 70)
    print(f"\nTables saved in: {tables_dir}")


if __name__ == "__main__":
    generate_all_tables()

