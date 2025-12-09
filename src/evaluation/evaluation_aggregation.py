"""Result aggregation and comparison functions."""

from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import numpy as np
import pandas as pd
import logging
from src.utils import setup_logging

setup_logging()
_module_logger = logging.getLogger(__name__)
EXTREME_VALUE_THRESHOLD = 1e10

logger = _module_logger


def _json_safe(obj):
    """Convert common numpy/pandas scalars to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return obj.isoformat()
    return obj

# ============================================================================
# Model Comparison Functions
# ============================================================================

def compare_multiple_models(
    model_results: Dict[str, Dict[str, Any]],
    horizons: List[int],
    target_series: Optional[str] = None
) -> Dict[str, Any]:
    """Compare results from multiple forecasting models.
    
    This function takes results from multiple model training runs and
    generates a comparison table with standardized metrics for each horizon.
    
    Parameters
    ----------
    model_results : dict
        Dictionary mapping model name to experiment results.
        Each result should have:
        - 'status': 'completed' or 'failed'
        - 'metrics': Dictionary with training metrics (converged, num_iter, loglik, etc.)
        - 'result': Model result object (optional, for extracting forecasts)
        - 'metadata': Model metadata (optional)
    horizons : List[int]
        List of forecast horizons to compare (e.g., [1, 11, 22])
    target_series : str, optional
        Target series name (for context and filtering)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'metrics_table': pd.DataFrame with metrics per model and horizon
        - 'summary': Summary statistics dictionary
        - 'best_model_per_horizon': Dictionary mapping horizon to best model name
        - 'best_model_overall': Best model across all horizons
        
    Notes
    -----
    - Only models with status='completed' and valid metrics are included
    - Metrics are extracted from training results (converged, num_iter, loglik)
    - For forecast metrics, models need to have prediction results available
    - Standardized metrics (sMSE, sMAE, sRMSE) are preferred for comparison
    """
    # Filter successful models
    successful_models = {
        name: result for name, result in model_results.items()
        if result.get('status') == 'completed' and result.get('metrics') is not None
    }
    
    if len(successful_models) == 0:
        return {
            'metrics_table': None,
            'summary': 'No successful models to compare',
            'best_model_per_horizon': {},
            'best_model_overall': None
        }
    
    # Extract metrics for each model
    comparison_data = []
    
    for model_name, result in successful_models.items():
        metrics = result.get('metrics', {})
        forecast_metrics = metrics.get('forecast_metrics', {})
        
        # Extract training metrics
        row = {
            'model': model_name,
            'converged': metrics.get('converged', False),
            'num_iter': metrics.get('num_iter', 0),
            'loglik': metrics.get('loglik', np.nan),
            'model_type': metrics.get('model_type', 'unknown')
        }
        
        # Extract forecast metrics for each horizon
        for horizon in horizons:
            horizon_metrics = forecast_metrics.get(horizon, {})
            if horizon_metrics:
                row[f'sMSE_h{horizon}'] = horizon_metrics.get('sMSE', np.nan)
                row[f'sMAE_h{horizon}'] = horizon_metrics.get('sMAE', np.nan)
                row[f'sRMSE_h{horizon}'] = horizon_metrics.get('sRMSE', np.nan)
            else:
                row[f'sMSE_h{horizon}'] = np.nan
                row[f'sMAE_h{horizon}'] = np.nan
                row[f'sRMSE_h{horizon}'] = np.nan
        
        comparison_data.append(row)
    
    # Create DataFrame
    metrics_df = pd.DataFrame(comparison_data)
    
    # Generate summary statistics
    summary = {
        'total_models': len(successful_models),
        'converged_models': int(metrics_df['converged'].sum()),
        'avg_loglik': float(metrics_df['loglik'].mean()) if not metrics_df['loglik'].isna().all() else np.nan,
        'best_loglik': float(metrics_df['loglik'].max()) if not metrics_df['loglik'].isna().all() else np.nan,
        'best_model_by_loglik': metrics_df.loc[metrics_df['loglik'].idxmax(), 'model'] if not metrics_df['loglik'].isna().all() else None
    }
    
    # Determine best model per horizon based on sMSE (lower is better)
    best_model_per_horizon = {}
    for horizon in horizons:
        sMSE_col = f'sMSE_h{horizon}'
        if sMSE_col in metrics_df.columns:
            valid_models = metrics_df[metrics_df[sMSE_col].notna()]
            if len(valid_models) > 0:
                best_idx = valid_models[sMSE_col].idxmin()
                best_model_per_horizon[horizon] = metrics_df.loc[best_idx, 'model']
    
    # Best model overall (lowest average sMSE across all horizons)
    sMSE_cols = [f'sMSE_h{h}' for h in horizons if f'sMSE_h{h}' in metrics_df.columns]
    if sMSE_cols:
        metrics_df['avg_sMSE'] = metrics_df[sMSE_cols].mean(axis=1)
        valid_models = metrics_df[metrics_df['avg_sMSE'].notna()]
        if len(valid_models) > 0:
            best_model_overall = valid_models.loc[valid_models['avg_sMSE'].idxmin(), 'model']
        else:
            best_model_overall = summary.get('best_model_by_loglik')
    else:
        best_model_overall = summary.get('best_model_by_loglik')
    
    return {
        'metrics_table': metrics_df,
        'summary': summary,
        'best_model_per_horizon': best_model_per_horizon,
        'best_model_overall': best_model_overall,
        'target_series': target_series,
        'horizons': horizons
    }


def generate_comparison_table(
    comparison_results: Dict[str, Any],
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """Generate a formatted comparison table from comparison results.
    
    Parameters
    ----------
    comparison_results : dict
        Results from compare_multiple_models() function
    output_path : str, optional
        Path to save the comparison table (CSV format)
        
    Returns
    -------
    pd.DataFrame
        Formatted comparison table
    """
    metrics_table = comparison_results.get('metrics_table')
    
    if metrics_table is None:
        return pd.DataFrame()
    
    # Format table for better readability
    formatted_table = metrics_table.copy()
    
    # Round numeric columns
    numeric_cols = formatted_table.select_dtypes(include=[np.number]).columns
    formatted_table[numeric_cols] = formatted_table[numeric_cols].round(4)
    
    # Save if output path provided
    if output_path:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        formatted_table.to_csv(output_path_obj, index=False, encoding='utf-8')
    
    return formatted_table

# ============================================================================
# Result Aggregation Functions
# ============================================================================

def collect_all_comparison_results(outputs_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Collect all comparison results from outputs/comparisons/."""
    logger = _module_logger
    
    if outputs_dir is None:
        from src.utils import get_project_root
        outputs_dir = get_project_root() / "outputs"
    
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
            logger.warning(f"Could not load {results_file}: {e}")
            continue
    
    return all_results


def aggregate_overall_performance(all_results: Dict[str, Any]) -> pd.DataFrame:
    """Aggregate overall performance metrics across all models, targets, and horizons."""
    rows = []
    
    for target_series, results_list in all_results.items():
        # Sort by timestamp and use the latest result for each target
        if results_list:
            # Sort by timestamp (newest first)
            results_list_sorted = sorted(
                results_list,
                key=lambda x: x.get('timestamp', ''),
                reverse=True
            )
            result_data = results_list_sorted[0] if results_list_sorted else None
        else:
            result_data = None
        
        if not result_data:
            continue
        
        results = result_data.get('results', {})
        horizons = result_data.get('horizons', list(range(1, 23)))  # Default: horizons 1-22 (monthly: 2024-01 to 2025-10)
        
        # Extract metrics from each model
        for model_name, model_data in results.items():
            if not isinstance(model_data, dict):
                continue
            
            metrics = model_data.get('metrics', {})
            if not isinstance(metrics, dict):
                continue
            
            forecast_metrics = metrics.get('forecast_metrics', {})
            if not isinstance(forecast_metrics, dict):
                continue
            
            # Extract metrics for each horizon
            for horizon in horizons:
                horizon_str = str(horizon)
                if horizon_str not in forecast_metrics:
                    continue
                
                horizon_metrics = forecast_metrics[horizon_str]
                if not isinstance(horizon_metrics, dict):
                    continue
                
                n_valid = horizon_metrics.get('n_valid', 0)
                
                # Apply validation to filter extreme values (numerical instability)
                # This ensures extreme VAR values (e.g., > 1e10) are marked as NaN
                def validate_metric(val):
                    """Validate metric value and return NaN if extreme.
                    
                    This function filters out extreme values that indicate numerical instability,
                    such as VAR forecasts that become unstable for long horizons.
                    """
                    if val is None:
                        return np.nan
                    if isinstance(val, (int, float)):
                        if np.isnan(val) or np.isinf(val):
                            return np.nan
                        if abs(val) > EXTREME_VALUE_THRESHOLD:
                            # Log warning for extreme values
                            _module_logger.warning(
                                f"aggregate_overall_performance: Extreme value detected for "
                                f"{model_name.upper()} {target_series} horizon {horizon}: {val:.2e}. "
                                f"Marking as NaN due to numerical instability."
                            )
                            return np.nan
                    # Handle string representations of numbers
                    if isinstance(val, str):
                        try:
                            val_float = float(val)
                            return validate_metric(val_float)
                        except (ValueError, TypeError):
                            return np.nan
                    return val
                
                # Validate all metrics (standardized and raw) to filter extreme values
                smse = validate_metric(horizon_metrics.get('sMSE'))
                smae = validate_metric(horizon_metrics.get('sMAE'))
                srmse = validate_metric(horizon_metrics.get('sRMSE'))
                mse = validate_metric(horizon_metrics.get('MSE'))
                mae = validate_metric(horizon_metrics.get('MAE'))
                rmse = validate_metric(horizon_metrics.get('RMSE'))
                
                # Check for suspiciously good results (potential data leakage, numerical issues, or single-point luck)
                # Threshold: sMSE < 1e-4 or sMAE < 1e-3 is suspiciously good for any model/horizon
                # This is especially important when n_valid=1 (single test point)
                # Note: Zero values (perfect predictions) are also considered suspicious
                SUSPICIOUSLY_GOOD_SMSE_THRESHOLD = 1e-4
                SUSPICIOUSLY_GOOD_SMAE_THRESHOLD = 1e-3
                if isinstance(smse, (int, float)) and not np.isnan(smse) and (0 <= abs(smse) < SUSPICIOUSLY_GOOD_SMSE_THRESHOLD):
                    _module_logger.warning(
                        f"aggregate_overall_performance: Suspiciously good sMSE detected for "
                        f"{model_name.upper()} {target_series} horizon {horizon}: {smse:.2e}. "
                        f"This may indicate data leakage, numerical precision issues, or single-point luck (n_valid={n_valid}). "
                        f"Marking as NaN for reliability."
                    )
                    smse = np.nan
                    smae = np.nan
                    srmse = np.nan
                elif isinstance(smae, (int, float)) and not np.isnan(smae) and (0 <= abs(smae) < SUSPICIOUSLY_GOOD_SMAE_THRESHOLD):
                    _module_logger.warning(
                        f"aggregate_overall_performance: Suspiciously good sMAE detected for "
                        f"{model_name.upper()} {target_series} horizon {horizon}: {smae:.2e}. "
                        f"This may indicate data leakage, numerical precision issues, or single-point luck (n_valid={n_valid}). "
                        f"Marking as NaN for reliability."
                    )
                    smse = np.nan
                    smae = np.nan
                    srmse = np.nan
                
                # Extract enhanced diagnostic metrics if available
                # These metrics help diagnose DDFM performance issues (linear collapse, error patterns, etc.)
                error_skewness = validate_metric(horizon_metrics.get('error_skewness'))
                error_kurtosis = validate_metric(horizon_metrics.get('error_kurtosis'))
                error_bias_squared = validate_metric(horizon_metrics.get('error_bias_squared'))
                error_variance = validate_metric(horizon_metrics.get('error_variance'))
                error_concentration = validate_metric(horizon_metrics.get('error_concentration'))
                prediction_bias = validate_metric(horizon_metrics.get('prediction_bias'))
                directional_accuracy = validate_metric(horizon_metrics.get('directional_accuracy'))
                theil_u = validate_metric(horizon_metrics.get('theil_u'))
                mape = validate_metric(horizon_metrics.get('mape'))
                
                # Include all horizons, even if n_valid=0 (for complete 36-row dataset)
                # This allows the report to show NaN/N/A for unavailable combinations
                row = {
                    'target': target_series,
                    'model': model_name.upper(),
                    'horizon': horizon,
                    'sMSE': smse,
                    'sMAE': smae,
                    'sRMSE': srmse,
                    'MSE': mse,
                    'MAE': mae,
                    'RMSE': rmse,
                    'sigma': horizon_metrics.get('sigma'),
                    'n_valid': n_valid,
                    # Enhanced diagnostic metrics for DDFM analysis
                    'error_skewness': error_skewness,
                    'error_kurtosis': error_kurtosis,
                    'error_bias_squared': error_bias_squared,
                    'error_variance': error_variance,
                    'error_concentration': error_concentration,
                    'prediction_bias': prediction_bias,
                    'directional_accuracy': directional_accuracy,
                    'theil_u': theil_u,
                    'mape': mape
                }
                rows.append(row)
    
    if not rows:
        return pd.DataFrame()
    
    # Create DataFrame from rows
    aggregated = pd.DataFrame(rows)
    return aggregated


def detect_ddfm_linearity(
    aggregated_results: pd.DataFrame,
    similarity_threshold: float = 0.95,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """Detect when DDFM is collapsing to linear behavior (identical to DFM).
    
    This function compares DDFM and DFM metrics to identify when DDFM encoder
    is learning only linear relationships, which indicates the encoder is not
    capturing nonlinear features.
    
    Enhanced with additional diagnostic metrics:
    - Performance improvement ratio (DDFM vs DFM)
    - Horizon-specific improvement patterns
    - Volatile horizon analysis
    
    Parameters
    ----------
    aggregated_results : pd.DataFrame
        Aggregated results DataFrame from aggregate_overall_performance()
        Must have columns: target, model, horizon, sMSE, sMAE, sRMSE
    similarity_threshold : float, default 0.95
        Threshold for considering DDFM and DFM metrics as "similar" (0-1).
        Metrics are considered similar if relative difference < (1 - threshold).
    output_path : str, optional
        Path to save linearity detection results (JSON format)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'linearity_scores': Per-target linearity scores (0-1, higher = more linear)
        - 'similarity_by_horizon': Detailed similarity metrics per target and horizon
        - 'performance_improvement': DDFM improvement over DFM (negative = worse, positive = better)
        - 'summary': Summary statistics and warnings
        - 'recommendations': Recommendations for improving DDFM performance
    """
    logger.info("Detecting DDFM linearity (comparing DDFM vs DFM metrics)")
    
    if aggregated_results.empty:
        logger.warning("Empty aggregated_results, cannot detect linearity")
        return {'error': 'Empty results'}
    
    # Filter for DFM and DDFM models only
    dfm_ddfm = aggregated_results[
        aggregated_results['model'].isin(['DFM', 'DDFM'])
    ].copy()
    
    if dfm_ddfm.empty:
        logger.warning("No DFM/DDFM results found")
        return {'error': 'No DFM/DDFM results'}
    
    # Group by target and horizon
    linearity_scores = {}
    similarity_by_horizon = {}
    performance_improvement = {}
    
    for target in dfm_ddfm['target'].unique():
        target_data = dfm_ddfm[dfm_ddfm['target'] == target]
        
        dfm_data = target_data[target_data['model'] == 'DFM'].set_index('horizon')
        ddfm_data = target_data[target_data['model'] == 'DDFM'].set_index('horizon')
        
        # Find common horizons
        common_horizons = dfm_data.index.intersection(ddfm_data.index)
        
        if len(common_horizons) == 0:
            logger.warning(f"No common horizons for {target}")
            continue
        
        # Calculate similarity for each horizon
        horizon_similarities = []
        horizon_details = []
        improvement_ratios = []
        
        for horizon in common_horizons:
            dfm_row = dfm_data.loc[horizon]
            ddfm_row = ddfm_data.loc[horizon]
            
            # Skip if either has NaN metrics
            if (pd.isna(dfm_row.get('sMAE')) or pd.isna(ddfm_row.get('sMAE')) or
                pd.isna(dfm_row.get('sMSE')) or pd.isna(ddfm_row.get('sMSE'))):
                continue
            
            # Calculate relative differences
            smae_dfm = dfm_row['sMAE']
            smae_ddfm = ddfm_row['sMAE']
            smse_dfm = dfm_row['sMSE']
            smse_ddfm = ddfm_row['sMSE']
            
            # Relative difference: |DDFM - DFM| / max(DFM, DDFM)
            # Lower difference = more similar = more linear
            smae_diff = abs(smae_ddfm - smae_dfm) / max(abs(smae_dfm), abs(smae_ddfm), 1e-10)
            smse_diff = abs(smse_ddfm - smse_dfm) / max(abs(smse_dfm), abs(smse_ddfm), 1e-10)
            
            # Similarity score: 1 - relative_difference (higher = more similar)
            smae_similarity = 1.0 - min(smae_diff, 1.0)
            smse_similarity = 1.0 - min(smse_diff, 1.0)
            
            # Average similarity
            avg_similarity = (smae_similarity + smse_similarity) / 2.0
            
            # Performance improvement ratio: (DFM - DDFM) / DFM
            # Positive = DDFM better (lower error), Negative = DDFM worse
            smae_improvement = (smae_dfm - smae_ddfm) / max(abs(smae_dfm), 1e-10)
            smse_improvement = (smse_dfm - smse_ddfm) / max(abs(smse_dfm), 1e-10)
            avg_improvement = (smae_improvement + smse_improvement) / 2.0
            
            horizon_similarities.append(avg_similarity)
            improvement_ratios.append(avg_improvement)
            horizon_details.append({
                'horizon': int(horizon),
                'sMAE_similarity': float(smae_similarity),
                'sMSE_similarity': float(smse_similarity),
                'avg_similarity': float(avg_similarity),
                'sMAE_improvement': float(smae_improvement),
                'sMSE_improvement': float(smse_improvement),
                'avg_improvement': float(avg_improvement),
                'dfm_sMAE': float(smae_dfm),
                'ddfm_sMAE': float(smae_ddfm),
                'dfm_sMSE': float(smse_dfm),
                'ddfm_sMSE': float(smse_ddfm),
                'is_linear': avg_similarity >= similarity_threshold
            })
        
        if len(horizon_similarities) == 0:
            continue
        
        # Overall linearity score: average similarity across all horizons
        overall_linearity = np.mean(horizon_similarities)
        overall_improvement = np.mean(improvement_ratios) if improvement_ratios else 0.0
        
        linearity_scores[target] = {
            'overall_linearity': float(overall_linearity),
            'n_horizons': len(horizon_similarities),
            'linear_horizons': sum(1 for d in horizon_details if d['is_linear']),
            'linear_fraction': sum(1 for d in horizon_details if d['is_linear']) / len(horizon_details) if horizon_details else 0.0
        }
        
        performance_improvement[target] = {
            'overall_improvement': float(overall_improvement),
            'avg_sMAE_improvement': float(np.mean([d['sMAE_improvement'] for d in horizon_details])),
            'avg_sMSE_improvement': float(np.mean([d['sMSE_improvement'] for d in horizon_details])),
            'improvement_std': float(np.std(improvement_ratios)) if len(improvement_ratios) > 1 else 0.0,
            'best_horizon_improvement': float(max([d['avg_improvement'] for d in horizon_details])) if horizon_details else 0.0,
            'worst_horizon_improvement': float(min([d['avg_improvement'] for d in horizon_details])) if horizon_details else 0.0
        }
        
        similarity_by_horizon[target] = horizon_details
    
    # Generate summary and recommendations
    summary = {
        'total_targets': len(linearity_scores),
        'high_linearity_targets': [],
        'moderate_linearity_targets': [],
        'low_linearity_targets': []
    }
    
    recommendations = {}
    
    for target, score_info in linearity_scores.items():
        linearity = score_info['overall_linearity']
        linear_fraction = score_info['linear_fraction']
        
        if linearity >= similarity_threshold:
            summary['high_linearity_targets'].append({
                'target': target,
                'linearity': linearity,
                'linear_fraction': linear_fraction,
                'status': 'CRITICAL: DDFM is learning linear relationships only'
            })
            recommendations[target] = [
                'DDFM encoder is collapsing to linear behavior (identical to DFM)',
                'Consider: deeper encoder architecture, tanh activation (instead of ReLU), weight decay regularization',
                'Consider: increased pre-training epochs, smaller batch size for gradient diversity',
                'Consider: analyzing correlation structure to identify negative correlations that ReLU cannot capture'
            ]
        elif linearity >= 0.8:
            summary['moderate_linearity_targets'].append({
                'target': target,
                'linearity': linearity,
                'linear_fraction': linear_fraction,
                'status': 'WARNING: DDFM shows high similarity to DFM'
            })
            recommendations[target] = [
                'DDFM shows moderate similarity to DFM',
                'Consider: tuning encoder architecture or activation function',
                'Monitor: Check if improvements (deeper encoder, tanh) are being applied'
            ]
        else:
            summary['low_linearity_targets'].append({
                'target': target,
                'linearity': linearity,
                'linear_fraction': linear_fraction,
                'status': 'OK: DDFM is learning nonlinear relationships'
            })
            recommendations[target] = [
                'DDFM is successfully learning nonlinear features',
                'Current configuration appears effective'
            ]
    
    results = _json_safe({
        'linearity_scores': linearity_scores,
        'similarity_by_horizon': similarity_by_horizon,
        'performance_improvement': performance_improvement,
        'summary': summary,
        'recommendations': recommendations,
        'similarity_threshold': similarity_threshold
    })
    
    # Save if output path provided
    if output_path:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path_obj, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Linearity detection results saved to: {output_path}")
    
    # Log summary
    logger.info("DDFM Linearity Detection Summary:")
    for target, score_info in linearity_scores.items():
        linearity = score_info['overall_linearity']
        linear_fraction = score_info['linear_fraction']
        improvement_info = performance_improvement.get(target, {})
        overall_improvement = improvement_info.get('overall_improvement', 0.0)
        
        logger.info(f"  {target}: Linearity={linearity:.3f}, Linear horizons={linear_fraction:.1%}, Improvement={overall_improvement:.1%}")
        if linearity >= similarity_threshold:
            logger.warning(f"    ⚠️  CRITICAL: {target} DDFM is learning linear relationships only (identical to DFM)")
        elif overall_improvement < 0:
            logger.warning(f"    ⚠️  WARNING: {target} DDFM performs worse than DFM (improvement={overall_improvement:.1%})")
        elif overall_improvement > 0.1:
            logger.info(f"    ✅ GOOD: {target} DDFM shows significant improvement over DFM ({overall_improvement:.1%})")
    
    return results


def analyze_ddfm_prediction_quality(
    aggregated_results: pd.DataFrame,
    target: Optional[str] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze DDFM prediction quality across horizons and compare with DFM.
    
    Enhanced with additional diagnostic metrics:
    - Prediction stability metrics (coefficient of variation, robust stability using median/IQR)
    - Horizon-specific improvement patterns with enhanced edge case handling
    - Short-term vs long-term performance analysis
    - Consistency metrics (how similar DDFM is to DFM across horizons)
    - Best/worst horizon identification
    - Linear collapse risk assessment (early warning system with adaptive weighting)
    - Near-linear collapse detection (detects when DDFM and DFM errors are within numerical precision)
    - Horizon-specific degradation detection
    - Robust statistics (median-based metrics for outlier resistance)
    - Horizon-weighted metrics (prioritizing short-term horizons)
    - Relative error stability metrics
    - Improvement persistence metrics
    - Factor dynamics stability inference (from both sMAE and sMSE error patterns)
    - Error distribution analysis (skewness, kurtosis, bias-variance decomposition)
    - Enhanced improvement ratio calculation with better edge case handling
    - Cross-horizon error pattern analysis with absolute and relative differences
    - Quantile-based error metrics (tail ratio, IQR, quantile sMAE/sMSE for robust evaluation)
    - Factor loading comparison (if available from models - requires model internals access)
    - Relative skill assessment (skill-like metric using error patterns when predictions unavailable)
    - Volatile horizon performance (measures how well DDFM handles challenging horizons)
    - Error autocorrelation analysis (detects systematic bias patterns across consecutive horizons)
    - Improvement stability metrics (measures consistency of improvements across horizons)
    
    This function provides detailed analysis of DDFM performance including:
    - Horizon-specific performance patterns
    - Volatile horizon identification
    - Prediction stability metrics
    - Comparison with DFM baseline
    - Early warning for linear collapse risk
    
    Notes
    -----
    - Metrics are calculated from aggregated_results.csv which contains error metrics
      (sMSE, sMAE, sRMSE) but not actual predictions. Some metrics (e.g., factor dynamics
      stability) use error patterns as proxies for prediction patterns.
    - Forecast skill score and information gain metrics require actual predictions (y_true, y_pred)
      and are not calculated here. These would need to be computed during evaluation or from
      comparison results in outputs/comparisons/ if predictions are stored there.
    - Robust metrics (median-based) are automatically used when coefficient of variation > 0.5
      (indicating outliers), providing more reliable evaluation for targets with volatile horizons.
    
    Parameters
    ----------
    aggregated_results : pd.DataFrame
        Aggregated results DataFrame from aggregate_overall_performance()
        Must have columns: target, model, horizon, sMSE, sMAE, sRMSE, n_valid
        Optional columns (for enhanced analysis): error_skewness, error_kurtosis, error_bias_squared,
        error_variance, error_concentration, prediction_bias, directional_accuracy, theil_u, mape
    target : str, optional
        Target series to analyze. If None, analyzes all targets.
    output_path : str, optional
        Path to save analysis results (JSON format)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'target_analysis': Per-target detailed analysis with all metrics
        - 'volatile_horizons': Horizons with high error variance
        - 'performance_trends': Performance trends across horizons
        - 'recommendations': Specific recommendations for improvement
        - 'linear_collapse_risk': Risk assessment for linear collapse (0-1, higher = more risk)
        - 'horizon_degradation': Detection of horizon-specific performance degradation
        - 'horizon_weighted_metrics': Horizon-weighted performance summary
    """
    logger.info(f"Analyzing DDFM prediction quality{' for ' + target if target else ''}")
    
    if aggregated_results.empty:
        logger.warning("Empty aggregated_results, cannot analyze prediction quality")
        return {'error': 'Empty results'}
    
    # Filter for DDFM and DFM models
    dfm_ddfm = aggregated_results[
        aggregated_results['model'].isin(['DFM', 'DDFM'])
    ].copy()
    
    if target:
        dfm_ddfm = dfm_ddfm[dfm_ddfm['target'] == target]
    
    if dfm_ddfm.empty:
        logger.warning(f"No DFM/DDFM results found{' for ' + target if target else ''}")
        return {'error': 'No DFM/DDFM results'}
    
    target_analysis = {}
    volatile_horizons = {}
    
    for tgt in dfm_ddfm['target'].unique():
        target_data = dfm_ddfm[dfm_ddfm['target'] == tgt]
        
        ddfm_data = target_data[target_data['model'] == 'DDFM'].sort_values('horizon')
        dfm_data = target_data[target_data['model'] == 'DFM'].sort_values('horizon')
        
        # Extract metrics
        ddfm_smae = ddfm_data['sMAE'].values
        ddfm_smse = ddfm_data['sMSE'].values
        ddfm_horizons = ddfm_data['horizon'].values
        
        dfm_smae = dfm_data['sMAE'].values
        dfm_smse = dfm_data['sMSE'].values
        dfm_horizons = dfm_data['horizon'].values
        
        # Find common horizons
        common_horizons = np.intersect1d(ddfm_horizons, dfm_horizons)
        
        if len(common_horizons) == 0:
            continue
        
        # Calculate performance metrics
        valid_ddfm = ~np.isnan(ddfm_smae)
        valid_dfm = ~np.isnan(dfm_smae)
        
        if np.sum(valid_ddfm) == 0:
            continue
        
        # Overall performance (using mean for primary metrics)
        avg_ddfm_smae = np.nanmean(ddfm_smae)
        avg_ddfm_smse = np.nanmean(ddfm_smse)
        avg_dfm_smae = np.nanmean(dfm_smae[valid_dfm]) if np.sum(valid_dfm) > 0 else np.nan
        avg_dfm_smse = np.nanmean(dfm_smse[valid_dfm]) if np.sum(valid_dfm) > 0 else np.nan
        
        # Robust performance metrics (median-based, more resistant to outliers)
        # Use robust metrics when coefficient of variation is high (indicating outliers)
        robust_ddfm_smae = np.nanmedian(ddfm_smae)
        robust_ddfm_smse = np.nanmedian(ddfm_smse)
        robust_dfm_smae = np.nanmedian(dfm_smae[valid_dfm]) if np.sum(valid_dfm) > 0 else np.nan
        robust_dfm_smse = np.nanmedian(dfm_smse[valid_dfm]) if np.sum(valid_dfm) > 0 else np.nan
        
        # Performance improvement (enhanced with better edge case handling)
        # Improvement ratio: (DFM - DDFM) / DFM, where positive = DDFM better
        # Handle edge cases: zero DFM error, very small differences, NaN values
        if not np.isnan(avg_dfm_smae) and avg_dfm_smae > 1e-10:
            improvement_smae = (avg_dfm_smae - avg_ddfm_smae) / avg_dfm_smae
            # Clip to reasonable range to avoid extreme values from numerical issues
            improvement_smae = float(np.clip(improvement_smae, -10.0, 10.0))
        elif not np.isnan(avg_dfm_smae) and avg_dfm_smae <= 1e-10:
            # DFM error is essentially zero - DDFM can only be worse or equal
            if not np.isnan(avg_ddfm_smae):
                improvement_smae = -1.0 if avg_ddfm_smae > 1e-10 else 0.0
            else:
                improvement_smae = np.nan
        else:
            improvement_smae = np.nan
        
        if not np.isnan(avg_dfm_smse) and avg_dfm_smse > 1e-10:
            improvement_smse = (avg_dfm_smse - avg_ddfm_smse) / avg_dfm_smse
            improvement_smse = float(np.clip(improvement_smse, -10.0, 10.0))
        elif not np.isnan(avg_dfm_smse) and avg_dfm_smse <= 1e-10:
            if not np.isnan(avg_ddfm_smse):
                improvement_smse = -1.0 if avg_ddfm_smse > 1e-10 else 0.0
            else:
                improvement_smse = np.nan
        else:
            improvement_smse = np.nan
        
        # Robust improvement (using median-based metrics, with same edge case handling)
        if not np.isnan(robust_dfm_smae) and robust_dfm_smae > 1e-10:
            robust_improvement_smae = (robust_dfm_smae - robust_ddfm_smae) / robust_dfm_smae
            robust_improvement_smae = float(np.clip(robust_improvement_smae, -10.0, 10.0))
        elif not np.isnan(robust_dfm_smae) and robust_dfm_smae <= 1e-10:
            if not np.isnan(robust_ddfm_smae):
                robust_improvement_smae = -1.0 if robust_ddfm_smae > 1e-10 else 0.0
            else:
                robust_improvement_smae = np.nan
        else:
            robust_improvement_smae = np.nan
        
        if not np.isnan(robust_dfm_smse) and robust_dfm_smse > 1e-10:
            robust_improvement_smse = (robust_dfm_smse - robust_ddfm_smse) / robust_dfm_smse
            robust_improvement_smse = float(np.clip(robust_improvement_smse, -10.0, 10.0))
        elif not np.isnan(robust_dfm_smse) and robust_dfm_smse <= 1e-10:
            if not np.isnan(robust_ddfm_smse):
                robust_improvement_smse = -1.0 if robust_ddfm_smse > 1e-10 else 0.0
            else:
                robust_improvement_smse = np.nan
        else:
            robust_improvement_smse = np.nan
        
        # Identify volatile horizons (high error variance or spikes)
        horizon_errors = []
        horizon_improvements = []
        for h in common_horizons:
            ddfm_idx = np.where(ddfm_horizons == h)[0]
            dfm_idx = np.where(dfm_horizons == h)[0]
            
            if len(ddfm_idx) > 0 and len(dfm_idx) > 0:
                ddfm_err = ddfm_smae[ddfm_idx[0]]
                dfm_err = dfm_smae[dfm_idx[0]]
                
                if not (np.isnan(ddfm_err) or np.isnan(dfm_err)):
                    # Enhanced improvement calculation with better edge case handling
                    # Improved for near-linear collapse detection (when errors are nearly identical):
                    # 1. For nearly identical errors (< 0.01 absolute diff or < 0.1% relative diff):
                    #    Use absolute difference as primary signal (more sensitive to linear collapse)
                    # 2. For different errors: Use geometric mean for balanced scaling
                    # 3. Fall back to max() for very small errors
                    
                    abs_diff = abs(ddfm_err - dfm_err)
                    rel_diff = abs_diff / max(abs(dfm_err), abs(ddfm_err), 1e-10)
                    
                    # Detect near-linear collapse: errors within numerical precision
                    is_near_linear = (abs_diff < 0.01) or (rel_diff < 0.001)
                    
                    if is_near_linear:
                        # Near-linear collapse: use absolute difference as primary metric
                        # For nearly identical errors, improvement ratio is misleading
                        # Instead, use absolute difference normalized by average error magnitude
                        avg_error = (dfm_err + ddfm_err) / 2.0
                        if avg_error > 1e-10:
                            # Normalized improvement: negative if DDFM worse, positive if better
                            h_improvement = (dfm_err - ddfm_err) / avg_error
                        else:
                            h_improvement = 0.0
                        # For near-linear cases, clip more aggressively to avoid false signals
                        h_improvement = float(np.clip(h_improvement, -1.0, 1.0))
                    elif dfm_err > 1e-6 and ddfm_err > 1e-6:
                        # Both errors are substantial: use geometric mean for balanced scaling
                        # Geometric mean: sqrt(a * b) - more balanced than max when errors differ
                        denominator = np.sqrt(dfm_err * ddfm_err)
                        h_improvement = (dfm_err - ddfm_err) / denominator
                        h_improvement = float(np.clip(h_improvement, -10.0, 10.0))
                    elif dfm_err > 1e-10 or ddfm_err > 1e-10:
                        # One error is very small: use max to avoid division issues
                        denominator = max(abs(dfm_err), abs(ddfm_err), 1e-10)
                        h_improvement = (dfm_err - ddfm_err) / denominator
                        h_improvement = float(np.clip(h_improvement, -10.0, 10.0))
                    else:
                        # Both errors are very small: use standard denominator
                        denominator = max(abs(dfm_err), abs(ddfm_err), 1e-10)
                        h_improvement = (dfm_err - ddfm_err) / denominator
                        h_improvement = float(np.clip(h_improvement, -10.0, 10.0))
                    
                    # Calculate relative error ratio (more robust than simple division)
                    error_ratio = ddfm_err / max(dfm_err, 1e-10) if dfm_err > 1e-10 else (ddfm_err / max(ddfm_err, 1e-10) if ddfm_err > 1e-10 else 1.0)
                    error_ratio = float(np.clip(error_ratio, 0.0, 100.0))  # Clip to reasonable range
                    
                    horizon_errors.append({
                        'horizon': int(h),
                        'ddfm_sMAE': float(ddfm_err),
                        'dfm_sMAE': float(dfm_err),
                        'error_ratio': error_ratio,
                        'improvement': h_improvement,
                        'absolute_diff': float(abs(ddfm_err - dfm_err)),  # NEW: Absolute difference for analysis
                        'relative_diff': float(abs(ddfm_err - dfm_err) / max(abs(dfm_err), abs(ddfm_err), 1e-10)),  # NEW: Relative difference
                        'is_near_linear': is_near_linear,  # NEW: Flag for near-linear collapse detection
                        'ddfm_worse': ddfm_err > dfm_err  # NEW: Flag when DDFM is worse than DFM
                    })
                    horizon_improvements.append(h_improvement)
        
        # Calculate error variance across horizons
        if len(horizon_errors) > 1:
            errors = [e['ddfm_sMAE'] for e in horizon_errors]
            error_variance = float(np.var(errors))
            error_std = float(np.std(errors))
            error_mean = float(np.mean(errors))
            # Coefficient of variation (stability metric: lower = more stable)
            error_cv = float(error_std / max(error_mean, 1e-10)) if error_mean > 0 else 0.0
            
            # Enhanced stability metric: sMSE/sMAE ratio stability across horizons
            # High variance in this ratio suggests unstable predictions
            smse_smae_ratios = []
            for e in horizon_errors:
                ddfm_horizon_data = target_data[(target_data['model'] == 'DDFM') & (target_data['horizon'] == e['horizon'])]
                if len(ddfm_horizon_data) > 0:
                    smse = ddfm_horizon_data['sMSE'].iloc[0]
                    smae = ddfm_horizon_data['sMAE'].iloc[0]
                    if not (pd.isna(smse) or pd.isna(smae)) and smae > 1e-10:
                        ratio = smse / smae
                        smse_smae_ratios.append(ratio)
            
            if len(smse_smae_ratios) > 1:
                ratio_variance = float(np.var(smse_smae_ratios))
                ratio_cv = float(np.std(smse_smae_ratios) / max(np.mean(smse_smae_ratios), 1e-10))
            else:
                ratio_variance = 0.0
                ratio_cv = 0.0
        else:
            error_variance = 0.0
            error_std = 0.0
            error_mean = 0.0
            error_cv = 0.0
            ratio_variance = 0.0
            ratio_cv = 0.0
        
        # Identify volatile horizons (error > mean + 1.5 * std)
        if error_std > 0:
            volatile_threshold = error_mean + 1.5 * error_std
            volatile = [e for e in horizon_errors if e['ddfm_sMAE'] > volatile_threshold]
        else:
            volatile = []
        
        # Short-term vs long-term performance (horizons 1-6 vs 13-22)
        short_term_horizons = [e for e in horizon_errors if 1 <= e['horizon'] <= 6]
        long_term_horizons = [e for e in horizon_errors if 13 <= e['horizon'] <= 22]
        
        short_term_avg = float(np.mean([e['ddfm_sMAE'] for e in short_term_horizons])) if short_term_horizons else None
        long_term_avg = float(np.mean([e['ddfm_sMAE'] for e in long_term_horizons])) if long_term_horizons else None
        
        # Consistency metric: how similar are DDFM and DFM across all horizons?
        # Lower consistency = more variation in improvement across horizons
        if len(horizon_improvements) > 1:
            improvement_std = float(np.std(horizon_improvements))
            improvement_mean = float(np.mean(horizon_improvements))
            consistency = 1.0 - min(improvement_std / max(abs(improvement_mean), 1e-10), 1.0) if improvement_mean != 0 else 0.0
        else:
            improvement_std = 0.0
            consistency = 1.0
        
        # Best and worst horizon improvements
        if horizon_improvements:
            best_horizon_idx = np.argmax(horizon_improvements)
            worst_horizon_idx = np.argmin(horizon_improvements)
            best_horizon = horizon_errors[best_horizon_idx]['horizon']
            worst_horizon = horizon_errors[worst_horizon_idx]['horizon']
            best_improvement = float(horizon_improvements[best_horizon_idx])
            worst_improvement = float(horizon_improvements[worst_horizon_idx])
        else:
            best_horizon = None
            worst_horizon = None
            best_improvement = None
            worst_improvement = None
        
        # Calculate relative improvement consistency: how consistent is improvement across horizons
        # This helps identify if DDFM improvement is systematic or just luck at specific horizons
        # Calculate this BEFORE linear collapse risk assessment so it can be used there
        relative_improvement_consistency = None
        if len(horizon_improvements) > 1:
            positive_improvements = [imp for imp in horizon_improvements if imp > 0]
            if len(positive_improvements) > 0:
                # Fraction of horizons where DDFM improves over DFM
                relative_improvement_consistency = float(len(positive_improvements) / len(horizon_improvements))
            else:
                relative_improvement_consistency = 0.0
        
        # Enhanced linear collapse risk assessment (0-1, higher = more risk)
        # Risk factors:
        # 1. Very small improvement over DFM (< 5%)
        # 2. High similarity to DFM across all horizons
        # 3. Low consistency (improvement varies a lot)
        # 4. Error pattern similarity (sMSE/sMAE ratio similarity)
        # 5. Horizon error correlation (high correlation suggests linear behavior)
        # 6. Low relative improvement consistency (few horizons with positive improvement)
        linear_collapse_risk = 0.0
        error_pattern_similarity = 0.0
        horizon_error_correlation = 0.0
        
        if improvement_smae is not None and not np.isnan(improvement_smae):
            # Risk factor 1: Small or negative improvement
            if improvement_smae < 0.05:
                risk_factor_1 = 1.0 - max(0.0, improvement_smae / 0.05)  # 1.0 if improvement <= 0, 0.0 if improvement >= 5%
            else:
                risk_factor_1 = 0.0
            
            # Risk factor 2: High similarity to DFM (calculate average similarity across horizons)
            horizon_similarities = []
            for e in horizon_errors:
                ddfm_err = e['ddfm_sMAE']
                dfm_err = e['dfm_sMAE']
                if dfm_err > 0:
                    rel_diff = abs(ddfm_err - dfm_err) / max(abs(dfm_err), 1e-10)
                    similarity = 1.0 - min(rel_diff, 1.0)  # 1.0 = identical, 0.0 = very different
                    horizon_similarities.append(similarity)
            avg_similarity = float(np.mean(horizon_similarities)) if horizon_similarities else 0.0
            risk_factor_2 = avg_similarity  # Higher similarity = higher risk
            
            # Risk factor 3: Low consistency (inconsistent improvement)
            risk_factor_3 = 1.0 - consistency  # Lower consistency = higher risk
            
            # Risk factor 4: Error pattern similarity (sMSE/sMAE ratio similarity)
            # If DDFM and DFM have similar error patterns (sMSE/sMAE ratios), suggests linear behavior
            ddfm_smse_values = []
            ddfm_smae_values = []
            dfm_smse_values = []
            dfm_smae_values = []
            pattern_similarities = []  # Separate list for pattern similarities
            for e in horizon_errors:
                # Get sMSE and sMAE from aggregated results
                ddfm_horizon_data = target_data[(target_data['model'] == 'DDFM') & (target_data['horizon'] == e['horizon'])]
                dfm_horizon_data = target_data[(target_data['model'] == 'DFM') & (target_data['horizon'] == e['horizon'])]
                if len(ddfm_horizon_data) > 0 and len(dfm_horizon_data) > 0:
                    ddfm_smse = ddfm_horizon_data['sMSE'].iloc[0]
                    ddfm_smae = ddfm_horizon_data['sMAE'].iloc[0]
                    dfm_smse = dfm_horizon_data['sMSE'].iloc[0]
                    dfm_smae = dfm_horizon_data['sMAE'].iloc[0]
                    if not (pd.isna(ddfm_smse) or pd.isna(ddfm_smae) or pd.isna(dfm_smse) or pd.isna(dfm_smae)):
                        if ddfm_smae > 1e-10 and dfm_smae > 1e-10:
                            ddfm_ratio = ddfm_smse / ddfm_smae if ddfm_smae > 0 else 0.0
                            dfm_ratio = dfm_smse / dfm_smae if dfm_smae > 0 else 0.0
                            if ddfm_ratio > 0 and dfm_ratio > 0:
                                ratio_similarity = 1.0 - min(abs(ddfm_ratio - dfm_ratio) / max(abs(dfm_ratio), 1e-10), 1.0)
                                ddfm_smse_values.append(ddfm_smse)
                                ddfm_smae_values.append(ddfm_smae)
                                dfm_smse_values.append(dfm_smse)
                                dfm_smae_values.append(dfm_smae)
                                pattern_similarities.append(ratio_similarity)
            
            if len(pattern_similarities) > 0:
                error_pattern_similarity = float(np.mean(pattern_similarities))
            else:
                error_pattern_similarity = 0.0
            risk_factor_4 = error_pattern_similarity  # Higher pattern similarity = higher risk
            
            # Risk factor 5: Horizon error correlation (high correlation suggests systematic linear behavior)
            # Calculate correlation of errors across horizons
            if len(ddfm_smae_values) > 2:
                try:
                    # Calculate correlation between DDFM and DFM error patterns across horizons
                    horizon_error_correlation = float(np.corrcoef(ddfm_smae_values, dfm_smae_values)[0, 1])
                    if np.isnan(horizon_error_correlation):
                        horizon_error_correlation = 0.0
                    # High positive correlation suggests similar error patterns (linear behavior)
                    risk_factor_5 = max(0.0, horizon_error_correlation)  # Only positive correlation increases risk
                except Exception:
                    risk_factor_5 = 0.0
                    horizon_error_correlation = 0.0
            else:
                risk_factor_5 = 0.0
            
            # Additional risk factors based on new metrics
            # Risk factor 6: Low relative improvement consistency (if DDFM only improves at few horizons, may be linear)
            rel_consistency_val = relative_improvement_consistency if relative_improvement_consistency is not None else 0.5
            risk_factor_6 = 1.0 - rel_consistency_val  # Lower consistency = higher risk
            
            # Risk factor 7: Error distribution similarity (if DDFM and DFM have similar error distributions, suggests linear behavior)
            # Check if error distribution metrics are available in aggregated results
            risk_factor_7 = 0.0
            error_dist_similarity = 0.0
            try:
                target_ddfm_data = aggregated_results[
                    (aggregated_results['target'] == tgt) & 
                    (aggregated_results['model'] == 'DDFM')
                ]
                target_dfm_data = aggregated_results[
                    (aggregated_results['target'] == tgt) & 
                    (aggregated_results['model'] == 'DFM')
                ]
                
                # Check error skewness similarity
                if 'error_skewness' in target_ddfm_data.columns and 'error_skewness' in target_dfm_data.columns:
                    ddfm_skew = target_ddfm_data['error_skewness'].dropna()
                    dfm_skew = target_dfm_data['error_skewness'].dropna()
                    if len(ddfm_skew) > 0 and len(dfm_skew) > 0:
                        avg_ddfm_skew = ddfm_skew.mean()
                        avg_dfm_skew = dfm_skew.mean()
                        skew_diff = abs(avg_ddfm_skew - avg_dfm_skew)
                        # Similar skewness (< 0.2 difference) suggests similar error distributions
                        skew_similarity = 1.0 - min(skew_diff / 0.2, 1.0) if skew_diff < 0.2 else 0.0
                        error_dist_similarity += 0.5 * skew_similarity
                
                # Check error kurtosis similarity
                if 'error_kurtosis' in target_ddfm_data.columns and 'error_kurtosis' in target_dfm_data.columns:
                    ddfm_kurt = target_ddfm_data['error_kurtosis'].dropna()
                    dfm_kurt = target_dfm_data['error_kurtosis'].dropna()
                    if len(ddfm_kurt) > 0 and len(dfm_kurt) > 0:
                        avg_ddfm_kurt = ddfm_kurt.mean()
                        avg_dfm_kurt = dfm_kurt.mean()
                        kurt_diff = abs(avg_ddfm_kurt - avg_dfm_kurt)
                        # Similar kurtosis (< 1.0 difference) suggests similar error tail behavior
                        kurt_similarity = 1.0 - min(kurt_diff / 1.0, 1.0) if kurt_diff < 1.0 else 0.0
                        error_dist_similarity += 0.5 * kurt_similarity
                
                # Normalize to 0-1 range
                error_dist_similarity = min(error_dist_similarity, 1.0)
                risk_factor_7 = error_dist_similarity  # Higher similarity = higher risk
            except Exception as e:
                logger.debug(f"Could not calculate error distribution similarity: {e}")
                risk_factor_7 = 0.0
            
            # Enhanced weighted combination with adaptive weighting based on target characteristics
            # If improvement is very small (< 1%), give more weight to similarity metrics
            # If improvement is moderate (1-5%), balance all factors
            # If improvement is larger (> 5%), reduce weight on improvement factor
            if improvement_smae is not None and not np.isnan(improvement_smae):
                if improvement_smae < 0.01:
                    # Very small improvement: emphasize similarity and pattern metrics
                    weights = [0.20, 0.20, 0.15, 0.15, 0.12, 0.10, 0.08]  # More weight on similarity
                elif improvement_smae < 0.05:
                    # Moderate improvement: balanced weighting
                    weights = [0.25, 0.18, 0.13, 0.13, 0.10, 0.10, 0.11]  # Standard weights
                else:
                    # Larger improvement: reduce weight on improvement factor, emphasize consistency
                    weights = [0.15, 0.15, 0.18, 0.15, 0.12, 0.15, 0.10]  # More weight on consistency
            else:
                # Default weights if improvement cannot be calculated
                weights = [0.25, 0.18, 0.13, 0.13, 0.10, 0.10, 0.11]
            
            # Weighted combination with adaptive weights
            linear_collapse_risk = (weights[0] * risk_factor_1 + weights[1] * risk_factor_2 + 
                                   weights[2] * risk_factor_3 + weights[3] * risk_factor_4 + 
                                   weights[4] * risk_factor_5 + weights[5] * risk_factor_6 + 
                                   weights[6] * risk_factor_7)
            linear_collapse_risk = float(np.clip(linear_collapse_risk, 0.0, 1.0))
        
        # Horizon-specific degradation detection
        # Identify horizons where DDFM performs significantly worse than DFM
        degraded_horizons = []
        for e in horizon_errors:
            ddfm_err = e['ddfm_sMAE']
            dfm_err = e['dfm_sMAE']
            improvement = e['improvement']
            # Degradation: DDFM error is > 10% worse than DFM
            if improvement < -0.10:
                degraded_horizons.append({
                    'horizon': e['horizon'],
                    'ddfm_sMAE': ddfm_err,
                    'dfm_sMAE': dfm_err,
                    'degradation_pct': float(-improvement * 100)  # Positive percentage
                })
        
        # NEW: Systematic bias detection - detect when DDFM is consistently worse than DFM
        # This helps identify cases where DDFM not only fails to improve but is systematically worse
        n_near_linear_horizons = sum(1 for e in horizon_errors if e.get('is_near_linear', False))
        n_ddfm_worse_horizons = sum(1 for e in horizon_errors if e.get('ddfm_worse', False))
        n_total_horizons = len(horizon_errors)
        
        near_linear_fraction = float(n_near_linear_horizons / n_total_horizons) if n_total_horizons > 0 else 0.0
        ddfm_worse_fraction = float(n_ddfm_worse_horizons / n_total_horizons) if n_total_horizons > 0 else 0.0
        
        # Systematic bias score: 0-1, higher = DDFM is more consistently worse than DFM
        # Combines both near-linear collapse and worse performance
        systematic_bias_score = 0.0
        if n_total_horizons > 0:
            # If DDFM is worse at >50% of horizons, there's systematic bias
            if ddfm_worse_fraction > 0.5:
                systematic_bias_score = ddfm_worse_fraction
            # If near-linear collapse at >80% of horizons, also indicates systematic issue
            elif near_linear_fraction > 0.8:
                systematic_bias_score = near_linear_fraction * 0.8  # Slightly lower weight for near-linear vs worse
            # If both conditions: high systematic bias
            if ddfm_worse_fraction > 0.3 and near_linear_fraction > 0.5:
                systematic_bias_score = min(1.0, (ddfm_worse_fraction + near_linear_fraction) / 2.0)
        
        systematic_bias_score = float(np.clip(systematic_bias_score, 0.0, 1.0))
        
        # Calculate horizon-weighted metrics for this target (calculate here, not from pre-computed dict)
        from src.evaluation.evaluation_metrics import calculate_horizon_weighted_metrics
        ddfm_weighted = calculate_horizon_weighted_metrics(aggregated_results, target=tgt, model='DDFM')
        dfm_weighted = calculate_horizon_weighted_metrics(aggregated_results, target=tgt, model='DFM')
        
        weighted_metrics = {}
        if ddfm_weighted and ddfm_weighted.get('weighted_sMAE') is not None:
            weighted_metrics['ddfm_weighted_sMAE'] = ddfm_weighted.get('weighted_sMAE')
            if dfm_weighted and dfm_weighted.get('weighted_sMAE') is not None:
                dfm_w = dfm_weighted['weighted_sMAE']
                ddfm_w = ddfm_weighted['weighted_sMAE']
                weighted_metrics['dfm_weighted_sMAE'] = dfm_w
                if dfm_w > 0:
                    weighted_metrics['weighted_improvement'] = float((dfm_w - ddfm_w) / dfm_w * 100)
        
        # Calculate relative error stability metrics (NEW)
        # Calculate nonlinearity score (NEW) - quantifies how nonlinear DDFM predictions are
        nonlinearity_analysis = None
        nonlinearity_score = np.nan
        try:
            from src.evaluation.evaluation_metrics import (
                calculate_relative_error_stability, calculate_improvement_persistence, 
                calculate_factor_dynamics_stability, calculate_quantile_based_metrics,
                calculate_nonlinearity_score
            )
            # Calculate nonlinearity score for this target
            nonlinearity_analysis = calculate_nonlinearity_score(
                aggregated_results, target=tgt
            )
            if nonlinearity_analysis and tgt in nonlinearity_analysis.get('target_scores', {}):
                nonlinearity_score = nonlinearity_analysis['target_scores'][tgt].get('nonlinearity_score', np.nan)
            else:
                nonlinearity_score = np.nan
        except Exception as e:
            logger.debug(f"Could not calculate nonlinearity score: {e}")
            nonlinearity_score = np.nan
            nonlinearity_analysis = None
        
        # Calculate relative skill assessment (NEW) - works with error metrics when predictions aren't available
        relative_skill_analysis = None
        try:
            from src.evaluation.evaluation_metrics import calculate_relative_skill_assessment
            relative_skill_analysis = calculate_relative_skill_assessment(
                aggregated_results, target=tgt, baseline_model='VAR'
            )
        except Exception as e:
            logger.debug(f"Could not calculate relative skill assessment: {e}")
            relative_skill_analysis = None
        
        # Calculate volatile horizon performance (NEW) - measures how well DDFM handles challenging horizons
        volatile_horizon_analysis = None
        try:
            from src.evaluation.evaluation_metrics import calculate_volatile_horizon_performance
            volatile_horizon_analysis = calculate_volatile_horizon_performance(
                aggregated_results, target=tgt
            )
            if volatile_horizon_analysis and tgt in volatile_horizon_analysis.get('target_analysis', {}):
                volatile_horizon_info = volatile_horizon_analysis['target_analysis'][tgt]
                # Add volatile horizon metrics to analysis
                analysis['volatile_horizon_handling_score'] = volatile_horizon_info.get('volatile_horizon_handling_score', np.nan)
                analysis['n_volatile_horizons'] = volatile_horizon_info.get('n_volatile_horizons', 0)
                analysis['avg_volatile_improvement'] = volatile_horizon_info.get('avg_volatile_improvement', np.nan)
                analysis['avg_stable_improvement'] = volatile_horizon_info.get('avg_stable_improvement', np.nan)
                analysis['relative_advantage_volatile'] = volatile_horizon_info.get('relative_advantage', np.nan)
        except Exception as e:
            logger.debug(f"Could not calculate volatile horizon performance: {e}")
            volatile_horizon_analysis = None
        
        # Calculate near-linear collapse detection (NEW) - detects when DDFM and DFM are within numerical precision
        near_collapse_analysis = None
        try:
            from src.evaluation.evaluation_metrics import calculate_near_linear_collapse_detection
            near_collapse_analysis = calculate_near_linear_collapse_detection(
                aggregated_results, target=tgt
            )
            if near_collapse_analysis and tgt in near_collapse_analysis.get('target_analysis', {}):
                near_collapse_info = near_collapse_analysis['target_analysis'][tgt]
                # Add near-collapse metrics to analysis
                analysis['near_collapse_score'] = near_collapse_info.get('near_collapse_score', np.nan)
                analysis['n_near_collapse_horizons'] = near_collapse_info.get('n_near_collapse', 0)
                analysis['near_collapse_fraction'] = near_collapse_info.get('near_collapse_fraction', 0.0)
                analysis['avg_abs_diff_sMAE'] = near_collapse_info.get('avg_abs_diff_sMAE', np.nan)
                analysis['avg_rel_diff_sMAE'] = near_collapse_info.get('avg_rel_diff_sMAE', np.nan)
                analysis['max_abs_diff_sMAE'] = near_collapse_info.get('max_abs_diff_sMAE', np.nan)
                analysis['max_rel_diff_sMAE'] = near_collapse_info.get('max_rel_diff_sMAE', np.nan)
                analysis['near_collapse_interpretation'] = near_collapse_info.get('interpretation', '')
        except Exception as e:
            logger.debug(f"Could not calculate near-linear collapse detection: {e}")
            near_collapse_analysis = None
        
        # Calculate error pattern smoothness (NEW) - measures how smooth error patterns are across horizons
        error_smoothness_analysis = None
        try:
            from src.evaluation.evaluation_metrics import calculate_error_pattern_smoothness
            ddfm_errors_dict = {int(e['horizon']): float(e['ddfm_sMAE']) for e in horizon_errors if not np.isnan(e['ddfm_sMAE'])}
            if len(ddfm_errors_dict) >= 3:
                error_smoothness_analysis = calculate_error_pattern_smoothness(ddfm_errors_dict)
                if error_smoothness_analysis:
                    analysis['error_smoothness_score'] = error_smoothness_analysis.get('smoothness_score', np.nan)
                    analysis['error_cv'] = error_smoothness_analysis.get('cv', np.nan)
                    analysis['error_autocorr'] = error_smoothness_analysis.get('autocorr', np.nan)
                    analysis['error_smoothness_interpretation'] = error_smoothness_analysis.get('interpretation', '')
        except Exception as e:
            logger.debug(f"Could not calculate error pattern smoothness: {e}")
            error_smoothness_analysis = None
        
        # Calculate improvement significance (NEW) - statistical significance testing for improvements
        improvement_significance_analysis = None
        try:
            from src.evaluation.evaluation_metrics import calculate_improvement_significance
            ddfm_errors_dict = {int(e['horizon']): float(e['ddfm_sMAE']) for e in horizon_errors if not np.isnan(e['ddfm_sMAE'])}
            dfm_errors_dict = {int(e['horizon']): float(e['dfm_sMAE']) for e in horizon_errors if not np.isnan(e['dfm_sMAE'])}
            if len(ddfm_errors_dict) >= 3 and len(dfm_errors_dict) >= 3:
                improvement_significance_analysis = calculate_improvement_significance(
                    ddfm_errors_dict, dfm_errors_dict
                )
                if improvement_significance_analysis:
                    analysis['improvement_is_significant'] = improvement_significance_analysis.get('is_significant', False)
                    analysis['improvement_p_value'] = improvement_significance_analysis.get('p_value', np.nan)
                    analysis['improvement_ci_lower'] = improvement_significance_analysis.get('improvement_ci_lower', np.nan)
                    analysis['improvement_ci_upper'] = improvement_significance_analysis.get('improvement_ci_upper', np.nan)
                    analysis['n_significant_horizons'] = len(improvement_significance_analysis.get('significant_horizons', []))
                    analysis['significant_horizons'] = improvement_significance_analysis.get('significant_horizons', [])
        except Exception as e:
            logger.debug(f"Could not calculate improvement significance: {e}")
            improvement_significance_analysis = None
        
        # Calculate error autocorrelation analysis (NEW) - detects systematic bias patterns
        error_autocorr_analysis = None
        try:
            from src.evaluation.evaluation_metrics import calculate_error_autocorrelation_analysis
            ddfm_errors_dict = {int(e['horizon']): float(e['ddfm_sMAE']) for e in horizon_errors if not np.isnan(e['ddfm_sMAE'])}
            dfm_errors_dict = {int(e['horizon']): float(e['dfm_sMAE']) for e in horizon_errors if not np.isnan(e['dfm_sMAE'])}
            if len(ddfm_errors_dict) >= 4 and len(dfm_errors_dict) >= 4:
                error_autocorr_analysis = calculate_error_autocorrelation_analysis(
                    ddfm_errors_dict, dfm_errors_dict, max_lag=3
                )
                if error_autocorr_analysis:
                    analysis['ddfm_autocorr_lag1'] = error_autocorr_analysis.get('ddfm_autocorr_lag1', np.nan)
                    analysis['dfm_autocorr_lag1'] = error_autocorr_analysis.get('dfm_autocorr_lag1', np.nan)
                    analysis['autocorr_diff'] = error_autocorr_analysis.get('autocorr_diff', np.nan)
                    analysis['error_systematic_bias_score'] = error_autocorr_analysis.get('systematic_bias_score', np.nan)
                    analysis['error_autocorr_interpretation'] = error_autocorr_analysis.get('interpretation', '')
        except Exception as e:
            logger.debug(f"Could not calculate error autocorrelation analysis: {e}")
            error_autocorr_analysis = None
        
        # Calculate improvement stability (NEW) - measures consistency of improvements across horizons
        improvement_stability_analysis = None
        try:
            from src.evaluation.evaluation_metrics import calculate_improvement_stability
            improvement_dict = {int(e['horizon']): float(e['improvement']) for e in horizon_errors if not np.isnan(e['improvement'])}
            if len(improvement_dict) >= 3:
                improvement_stability_analysis = calculate_improvement_stability(improvement_dict, method='combined')
                if improvement_stability_analysis:
                    analysis['improvement_stability_score'] = improvement_stability_analysis.get('stability_score', np.nan)
                    analysis['improvement_variance'] = improvement_stability_analysis.get('improvement_variance', np.nan)
                    analysis['improvement_cv'] = improvement_stability_analysis.get('improvement_cv', np.nan)
                    analysis['improvement_range'] = improvement_stability_analysis.get('improvement_range', np.nan)
                    analysis['n_positive_improvements'] = improvement_stability_analysis.get('n_positive_improvements', 0)
                    analysis['n_negative_improvements'] = improvement_stability_analysis.get('n_negative_improvements', 0)
                    analysis['improvement_stability_interpretation'] = improvement_stability_analysis.get('interpretation', '')
        except Exception as e:
            logger.debug(f"Could not calculate improvement stability: {e}")
            improvement_stability_analysis = None
        
        try:
            from src.evaluation.evaluation_metrics import (
                calculate_relative_error_stability, calculate_improvement_persistence, 
                calculate_factor_dynamics_stability, calculate_quantile_based_metrics
            )
            ddfm_errors_dict = {int(e['horizon']): float(e['ddfm_sMAE']) for e in horizon_errors}
            dfm_errors_dict = {int(e['horizon']): float(e['dfm_sMAE']) for e in horizon_errors}
            relative_error_stability = calculate_relative_error_stability(ddfm_errors_dict, dfm_errors_dict)
            
            # Calculate quantile-based metrics for DDFM error distribution analysis
            # This helps identify if DDFM has different error distribution characteristics than DFM
            quantile_metrics = None
            try:
                # Extract error values across all horizons for quantile analysis
                ddfm_error_values = [float(e['ddfm_sMAE']) for e in horizon_errors if not np.isnan(e['ddfm_sMAE'])]
                dfm_error_values = [float(e['dfm_sMAE']) for e in horizon_errors if not np.isnan(e['dfm_sMAE'])]
                
                if len(ddfm_error_values) > 5 and len(dfm_error_values) > 5:
                    # Create arrays for quantile analysis
                    # Use error values as "predictions" vs mean error as "true" for quantile analysis
                    ddfm_error_arr = np.array(ddfm_error_values)
                    dfm_error_arr = np.array(dfm_error_values)
                    mean_error = np.mean(ddfm_error_arr)
                    
                    # Calculate quantile metrics for DDFM error distribution
                    # This helps identify tail behavior and error concentration
                    quantile_metrics = calculate_quantile_based_metrics(
                        np.full_like(ddfm_error_arr, mean_error),  # "true" = mean error
                        ddfm_error_arr,  # "predicted" = actual errors
                        y_train=ddfm_error_arr  # Use errors themselves for normalization
                    )
            except Exception as e:
                logger.debug(f"Could not calculate quantile-based metrics: {e}")
                quantile_metrics = None
            
            # Calculate improvement persistence metrics (NEW)
            improvement_dict = {int(e['horizon']): float(e['improvement']) for e in horizon_errors}
            improvement_persistence = calculate_improvement_persistence(improvement_dict, persistence_threshold=0.05)
            
            # Calculate factor dynamics stability (NEW) - infer VAR stability from prediction patterns
            # NOTE: This analysis uses sMAE as a proxy for prediction patterns since actual predictions
            # are not stored in aggregated_results.csv. For more accurate factor dynamics analysis,
            # actual predictions would need to be stored during evaluation or extracted from comparison results.
            # The current approach analyzes error patterns across horizons to infer VAR stability,
            # which provides useful insights but may not capture all factor dynamics nuances.
            # Enhanced factor dynamics analysis: use both sMAE and sMSE patterns
            # Extract predictions by horizon from aggregated results for factor dynamics analysis
            predictions_by_horizon = {}
            predictions_smse_by_horizon = {}  # NEW: Also track sMSE patterns
            for e in horizon_errors:
                h = int(e['horizon'])
                # Get prediction values if available (for now, use error as proxy for prediction pattern)
                # LIMITATION: Actual predictions are not stored in aggregated_results.csv, so we use
                # sMAE and sMSE as proxies. This provides approximate factor dynamics stability analysis but
                # may not capture all nuances. For full accuracy, actual predictions should be stored
                # during evaluation or extracted from outputs/comparisons/ JSON files.
                ddfm_horizon_data = target_data[(target_data['model'] == 'DDFM') & (target_data['horizon'] == h)]
                if len(ddfm_horizon_data) > 0:
                    # Use sMAE as proxy for prediction magnitude (for stability analysis)
                    # This is a simplified approach - full implementation would use actual predictions
                    pred_value_smae = ddfm_horizon_data['sMAE'].iloc[0] if not pd.isna(ddfm_horizon_data['sMAE'].iloc[0]) else None
                    pred_value_smse = ddfm_horizon_data['sMSE'].iloc[0] if not pd.isna(ddfm_horizon_data['sMSE'].iloc[0]) else None
                    if pred_value_smae is not None:
                        predictions_by_horizon[h] = np.array([pred_value_smae])
                    if pred_value_smse is not None:
                        predictions_smse_by_horizon[h] = np.array([pred_value_smse])
            
            # Enhanced factor dynamics analysis: analyze both sMAE and sMSE patterns
            # Only calculate if we have enough horizons for meaningful analysis
            factor_dynamics_stability = {}
            factor_dynamics_stability_smse = {}  # NEW: Also analyze sMSE patterns
            if len(predictions_by_horizon) >= 3:
                try:
                    factor_dynamics_stability = calculate_factor_dynamics_stability(predictions_by_horizon)
                except Exception as e:
                    logger.debug(f"Could not calculate factor dynamics stability (sMAE): {e}")
                    factor_dynamics_stability = {}
            
            # Also analyze sMSE patterns for additional insights
            if len(predictions_smse_by_horizon) >= 3:
                try:
                    factor_dynamics_stability_smse = calculate_factor_dynamics_stability(predictions_smse_by_horizon)
                except Exception as e:
                    logger.debug(f"Could not calculate factor dynamics stability (sMSE): {e}")
                    factor_dynamics_stability_smse = {}
            
            # Combine insights from both sMAE and sMSE analysis
            if factor_dynamics_stability and factor_dynamics_stability_smse:
                # Use the more conservative (worse) stability score
                stability_smae = factor_dynamics_stability.get('stability_score', 1.0)
                stability_smse = factor_dynamics_stability_smse.get('stability_score', 1.0)
                if stability_smae is not None and stability_smse is not None:
                    combined_stability = min(stability_smae, stability_smse)
                    factor_dynamics_stability['combined_stability_score'] = combined_stability
                    factor_dynamics_stability['sMSE_stability'] = factor_dynamics_stability_smse
        except Exception as e:
            logger.debug(f"Could not calculate relative error stability, improvement persistence, or factor dynamics stability: {e}")
            relative_error_stability = {}
            improvement_persistence = {}
            factor_dynamics_stability = {}
        
        # Store quantile metrics (will be added to analysis dict later)
        # Note: analysis dict is created later, so we store quantile_metrics separately for now
        
        # Enhanced cross-horizon error pattern analysis
        # Analyze how error patterns change across horizons to detect systematic issues
        horizon_error_trend = None
        horizon_improvement_trend = None
        if len(horizon_errors) > 3:
            # Calculate trend in errors across horizons (linear regression slope)
            horizons_list = [e['horizon'] for e in horizon_errors]
            errors_list = [e['ddfm_sMAE'] for e in horizon_errors]
            improvements_list = [e['improvement'] for e in horizon_errors]
            
            try:
                # Linear regression to detect trends
                from scipy import stats
                slope_err, intercept_err, r_err, p_err, std_err = stats.linregress(horizons_list, errors_list)
                slope_imp, intercept_imp, r_imp, p_imp, std_imp = stats.linregress(horizons_list, improvements_list)
                
                horizon_error_trend = {
                    'slope': float(slope_err),  # Positive = errors increase with horizon (degradation)
                    'r_squared': float(r_err ** 2),  # How well linear trend fits
                    'p_value': float(p_err),  # Statistical significance
                    'interpretation': 'degrading' if slope_err > 0.01 else 'improving' if slope_err < -0.01 else 'stable'
                }
                horizon_improvement_trend = {
                    'slope': float(slope_imp),  # Positive = improvement increases with horizon
                    'r_squared': float(r_imp ** 2),
                    'p_value': float(p_imp),
                    'interpretation': 'improving' if slope_imp > 0.001 else 'degrading' if slope_imp < -0.001 else 'stable'
                }
            except (ImportError, Exception) as e:
                logger.debug(f"Could not calculate horizon error trends: {e}")
        
        # Enhanced horizon improvement tracking: categorize horizons by improvement level
        horizon_improvement_categories = {
            'significant_improvement': [],  # > 10% improvement
            'moderate_improvement': [],      # 5-10% improvement
            'marginal_improvement': [],      # 0-5% improvement
            'no_improvement': [],            # 0% (within numerical precision)
            'degradation': []                # < 0% (worse than DFM)
        }
        
        for e in horizon_errors:
            improvement = e['improvement']
            horizon = e['horizon']
            if improvement > 0.10:
                horizon_improvement_categories['significant_improvement'].append(horizon)
            elif improvement > 0.05:
                horizon_improvement_categories['moderate_improvement'].append(horizon)
            elif improvement > 0.001:
                horizon_improvement_categories['marginal_improvement'].append(horizon)
            elif improvement > -0.001:
                horizon_improvement_categories['no_improvement'].append(horizon)
            else:
                horizon_improvement_categories['degradation'].append(horizon)
        
        # Calculate improvement distribution statistics
        improvement_distribution = {
            'n_significant': len(horizon_improvement_categories['significant_improvement']),
            'n_moderate': len(horizon_improvement_categories['moderate_improvement']),
            'n_marginal': len(horizon_improvement_categories['marginal_improvement']),
            'n_no_improvement': len(horizon_improvement_categories['no_improvement']),
            'n_degradation': len(horizon_improvement_categories['degradation']),
            'total_horizons': len(horizon_errors),
            'improvement_fraction': (len(horizon_improvement_categories['significant_improvement']) +
                                   len(horizon_improvement_categories['moderate_improvement']) +
                                   len(horizon_improvement_categories['marginal_improvement'])) / max(len(horizon_errors), 1),
            'significant_improvement_fraction': len(horizon_improvement_categories['significant_improvement']) / max(len(horizon_errors), 1),
            'categories': horizon_improvement_categories
        }
        
        # Enhanced error stability metric: how much do errors vary relative to mean
        # Lower stability = more unpredictable performance across horizons
        error_stability = None
        error_stability_robust = None  # NEW: Robust stability using median
        if error_mean > 0:
            # Stability = 1 - (CV / max_CV), where max_CV is normalized to 1.0
            # Higher stability (closer to 1.0) = more consistent performance
            error_stability = float(max(0.0, 1.0 - min(error_cv, 1.0)))
            
            # Robust stability using median and IQR instead of mean and std
            if len(errors) > 1:
                error_median = float(np.median(errors))
                error_iqr = float(np.percentile(errors, 75) - np.percentile(errors, 25))
                if error_median > 0:
                    robust_cv = error_iqr / max(error_median, 1e-10) if error_iqr > 0 else 0.0
                    error_stability_robust = float(max(0.0, 1.0 - min(robust_cv, 1.0)))
                else:
                    error_stability_robust = 0.0
            else:
                error_stability_robust = 1.0 if len(errors) == 1 else None
        
        target_analysis[tgt] = {
            'avg_sMAE': float(avg_ddfm_smae),
            'avg_sMSE': float(avg_ddfm_smse),
            'dfm_avg_sMAE': float(avg_dfm_smae) if not np.isnan(avg_dfm_smae) else None,
            'dfm_avg_sMSE': float(avg_dfm_smse) if not np.isnan(avg_dfm_smse) else None,
            'improvement_sMAE': float(improvement_smae) if not np.isnan(improvement_smae) else None,
            'improvement_sMSE': float(improvement_smse) if not np.isnan(improvement_smse) else None,
            # Robust metrics (median-based, more resistant to outliers)
            'robust_avg_sMAE': float(robust_ddfm_smae) if not np.isnan(robust_ddfm_smae) else None,
            'robust_avg_sMSE': float(robust_ddfm_smse) if not np.isnan(robust_ddfm_smse) else None,
            'robust_dfm_avg_sMAE': float(robust_dfm_smae) if not np.isnan(robust_dfm_smae) else None,
            'robust_dfm_avg_sMSE': float(robust_dfm_smse) if not np.isnan(robust_dfm_smse) else None,
            'robust_improvement_sMAE': float(robust_improvement_smae) if not np.isnan(robust_improvement_smae) else None,
            'robust_improvement_sMSE': float(robust_improvement_smse) if not np.isnan(robust_improvement_smse) else None,
            'error_variance': error_variance,
            'error_std': error_std,
            'error_cv': error_cv,  # Coefficient of variation (stability)
            'error_stability': error_stability,  # Error stability metric (0-1, higher = more stable)
            'error_stability_robust': error_stability_robust,  # NEW: Robust error stability using median/IQR (0-1, higher = more stable)
            'smse_smae_ratio_cv': float(ratio_cv),  # sMSE/sMAE ratio stability (lower = more stable)
            'smse_smae_ratio_variance': float(ratio_variance),  # Variance in sMSE/sMAE ratio
            'n_horizons': len(common_horizons),
            'n_valid_horizons': len(horizon_errors),
            'short_term_avg_sMAE': short_term_avg,
            'long_term_avg_sMAE': long_term_avg,
            'consistency': float(consistency),  # 0-1, higher = more consistent improvement
            'relative_improvement_consistency': relative_improvement_consistency,  # NEW: Fraction of horizons with positive improvement
            'improvement_std': float(improvement_std),
            'best_horizon': best_horizon,
            'worst_horizon': worst_horizon,
            'best_improvement': best_improvement,
            'worst_improvement': worst_improvement,
            'linear_collapse_risk': linear_collapse_risk,  # 0-1, higher = more risk (enhanced with pattern similarity, correlation, and error distribution)
            'error_pattern_similarity': float(error_pattern_similarity),  # 0-1, higher = more similar error patterns to DFM
            'horizon_error_correlation': float(horizon_error_correlation),  # -1 to 1, high positive = similar error patterns across horizons
            'error_distribution_similarity': float(error_dist_similarity) if 'error_dist_similarity' in locals() else 0.0,  # 0-1, higher = more similar error distributions (skewness/kurtosis)
            # NEW: Systematic bias detection metrics
            'systematic_bias_score': systematic_bias_score,  # 0-1, higher = DDFM more consistently worse than DFM
            'n_near_linear_horizons': n_near_linear_horizons,  # Number of horizons showing near-linear collapse
            'near_linear_fraction': near_linear_fraction,  # Fraction of horizons with near-linear collapse
            'n_ddfm_worse_horizons': n_ddfm_worse_horizons,  # Number of horizons where DDFM is worse than DFM
            'ddfm_worse_fraction': ddfm_worse_fraction,  # Fraction of horizons where DDFM is worse than DFM
            'horizon_error_trend': horizon_error_trend,  # NEW: Trend analysis of errors across horizons
            'horizon_improvement_trend': horizon_improvement_trend,  # NEW: Trend analysis of improvement across horizons
            'degraded_horizons': degraded_horizons,  # Horizons where DDFM performs worse than DFM
            # Horizon-weighted metrics (NEW)
            'weighted_sMAE': weighted_metrics.get('ddfm_weighted_sMAE'),
            'dfm_weighted_sMAE': weighted_metrics.get('dfm_weighted_sMAE'),
            'weighted_improvement_pct': weighted_metrics.get('weighted_improvement'),
            # Relative error stability metrics (NEW)
            'relative_error_stability': relative_error_stability.get('relative_error_stability'),
            'relative_error_cv': relative_error_stability.get('relative_error_cv'),
            'relative_error_trend': relative_error_stability.get('relative_error_trend'),
            'avg_relative_error': relative_error_stability.get('avg_relative_error'),
            'min_relative_error': relative_error_stability.get('min_relative_error'),
            'max_relative_error': relative_error_stability.get('max_relative_error'),
            # Improvement persistence metrics (NEW)
            'improvement_persistence_score': improvement_persistence.get('persistence_score'),
            'improvement_fraction': improvement_persistence.get('improvement_fraction'),
            'consecutive_improvements': improvement_persistence.get('consecutive_improvements'),
            'improvement_clusters': improvement_persistence.get('improvement_clusters'),
            'improvement_consistency': improvement_persistence.get('improvement_consistency'),
            # Factor dynamics stability metrics (NEW)
            'factor_dynamics_stability_score': factor_dynamics_stability.get('stability_score'),
            'factor_dynamics_oscillation_detected': factor_dynamics_stability.get('oscillation_detected'),
            'factor_dynamics_oscillation_magnitude': factor_dynamics_stability.get('oscillation_magnitude'),
            'factor_dynamics_growth_rate': factor_dynamics_stability.get('growth_rate'),
            'factor_dynamics_smoothness_score': factor_dynamics_stability.get('smoothness_score'),
            'factor_dynamics_divergence_detected': factor_dynamics_stability.get('divergence_detected'),
            'factor_dynamics_convergence_detected': factor_dynamics_stability.get('convergence_detected'),
            'factor_dynamics_interpretation': factor_dynamics_stability.get('stability_interpretation'),
            # Enhanced horizon improvement tracking (NEW)
            'horizon_improvement_distribution': improvement_distribution,
            'n_significant_improvement_horizons': improvement_distribution['n_significant'],
            'n_moderate_improvement_horizons': improvement_distribution['n_moderate'],
            'n_marginal_improvement_horizons': improvement_distribution['n_marginal'],
            'n_no_improvement_horizons': improvement_distribution['n_no_improvement'],
            'n_degradation_horizons': improvement_distribution['n_degradation'],
            'improvement_fraction': improvement_distribution['improvement_fraction'],
            'significant_improvement_fraction': improvement_distribution['significant_improvement_fraction'],
            # Nonlinearity score metrics (NEW)
            'nonlinearity_score': float(nonlinearity_score) if not np.isnan(nonlinearity_score) else None,
            'pattern_divergence': float(nonlinearity_analysis['target_scores'][tgt].get('pattern_divergence', np.nan)) if nonlinearity_analysis and tgt in nonlinearity_analysis.get('target_scores', {}) else None,
            'error_nonlinearity': float(nonlinearity_analysis['target_scores'][tgt].get('error_nonlinearity', np.nan)) if nonlinearity_analysis and tgt in nonlinearity_analysis.get('target_scores', {}) else None,
            'horizon_interaction': float(nonlinearity_analysis['target_scores'][tgt].get('horizon_interaction', np.nan)) if nonlinearity_analysis and tgt in nonlinearity_analysis.get('target_scores', {}) else None,
            # Relative skill assessment metrics (NEW) - skill-like assessment using error metrics
            'relative_skill_vs_dfm': float(relative_skill_analysis['target_scores'][tgt].get('skill_vs_dfm', np.nan)) if relative_skill_analysis and tgt in relative_skill_analysis.get('target_scores', {}) else None,
            'relative_skill_vs_dfm_pct': float(relative_skill_analysis['target_scores'][tgt].get('skill_vs_dfm_pct', np.nan)) if relative_skill_analysis and tgt in relative_skill_analysis.get('target_scores', {}) else None,
            'relative_skill_vs_baseline': float(relative_skill_analysis['target_scores'][tgt].get('skill_vs_baseline', np.nan)) if relative_skill_analysis and tgt in relative_skill_analysis.get('target_scores', {}) else None,
            'relative_skill_vs_baseline_pct': float(relative_skill_analysis['target_scores'][tgt].get('skill_vs_baseline_pct', np.nan)) if relative_skill_analysis and tgt in relative_skill_analysis.get('target_scores', {}) else None,
            'relative_skill_consistency': float(relative_skill_analysis['target_scores'][tgt].get('skill_consistency', np.nan)) if relative_skill_analysis and tgt in relative_skill_analysis.get('target_scores', {}) else None,
            'relative_skill_level': relative_skill_analysis['target_scores'][tgt].get('skill_level', None) if relative_skill_analysis and tgt in relative_skill_analysis.get('target_scores', {}) else None,
            # Volatile horizon performance metrics (NEW)
            'volatile_horizon_handling_score': float(volatile_horizon_info.get('volatile_horizon_handling_score', np.nan)) if 'volatile_horizon_info' in locals() and volatile_horizon_info else None,
            'n_volatile_horizons': volatile_horizon_info.get('n_volatile_horizons', 0) if 'volatile_horizon_info' in locals() and volatile_horizon_info else 0,
            'avg_volatile_improvement': float(volatile_horizon_info.get('avg_volatile_improvement', np.nan)) if 'volatile_horizon_info' in locals() and volatile_horizon_info else None,
            'avg_stable_improvement': float(volatile_horizon_info.get('avg_stable_improvement', np.nan)) if 'volatile_horizon_info' in locals() and volatile_horizon_info else None,
            'relative_advantage_volatile': float(volatile_horizon_info.get('relative_advantage', np.nan)) if 'volatile_horizon_info' in locals() and volatile_horizon_info else None,
            # Quantile-based metrics (NEW)
            'quantile_metrics': quantile_metrics if 'quantile_metrics' in locals() and quantile_metrics is not None else None,
            # Error pattern smoothness metrics (NEW)
            'error_smoothness_score': error_smoothness_analysis.get('smoothness_score', np.nan) if error_smoothness_analysis else None,
            'error_cv': error_smoothness_analysis.get('cv', np.nan) if error_smoothness_analysis else None,
            'error_autocorr': error_smoothness_analysis.get('autocorr', np.nan) if error_smoothness_analysis else None,
            'error_smoothness_interpretation': error_smoothness_analysis.get('interpretation', '') if error_smoothness_analysis else None,
            # Improvement significance metrics (NEW)
            'improvement_is_significant': improvement_significance_analysis.get('is_significant', False) if improvement_significance_analysis else None,
            'improvement_p_value': improvement_significance_analysis.get('p_value', np.nan) if improvement_significance_analysis else None,
            'improvement_ci_lower': improvement_significance_analysis.get('improvement_ci_lower', np.nan) if improvement_significance_analysis else None,
            'improvement_ci_upper': improvement_significance_analysis.get('improvement_ci_upper', np.nan) if improvement_significance_analysis else None,
            'n_significant_horizons': len(improvement_significance_analysis.get('significant_horizons', [])) if improvement_significance_analysis else 0,
            'significant_horizons': improvement_significance_analysis.get('significant_horizons', []) if improvement_significance_analysis else []
        }
        
        volatile_horizons[tgt] = {
            'volatile_count': len(volatile),
            'volatile_horizons': [e['horizon'] for e in volatile],
            'volatile_details': volatile
        }
    
    # Calculate linear collapse risk summary
    linear_collapse_risk_summary = {}
    for tgt, analysis in target_analysis.items():
        risk = analysis.get('linear_collapse_risk', 0.0)
        linear_collapse_risk_summary[tgt] = {
            'risk_score': risk,
            'risk_level': 'HIGH' if risk > 0.7 else 'MEDIUM' if risk > 0.4 else 'LOW',
            'degraded_horizons_count': len(analysis.get('degraded_horizons', []))
        }
    
    # Calculate horizon-weighted metrics for DDFM
    from src.evaluation.evaluation_metrics import calculate_horizon_weighted_metrics
    
    horizon_weighted_metrics = {}
    for tgt in dfm_ddfm['target'].unique():
        ddfm_weighted = calculate_horizon_weighted_metrics(
            aggregated_results, target=tgt, model='DDFM'
        )
        dfm_weighted = calculate_horizon_weighted_metrics(
            aggregated_results, target=tgt, model='DFM'
        )
        
        if ddfm_weighted and dfm_weighted.get('weighted_sMAE') is not None:
            horizon_weighted_metrics[tgt] = {
                'ddfm_weighted_sMAE': ddfm_weighted.get('weighted_sMAE'),
                'dfm_weighted_sMAE': dfm_weighted.get('weighted_sMAE') if dfm_weighted else None,
                'weighted_improvement': None
            }
            
            if dfm_weighted and dfm_weighted.get('weighted_sMAE') is not None:
                dfm_w = dfm_weighted['weighted_sMAE']
                ddfm_w = ddfm_weighted['weighted_sMAE']
                if dfm_w > 0:
                    horizon_weighted_metrics[tgt]['weighted_improvement'] = float(
                        (dfm_w - ddfm_w) / dfm_w * 100
                    )
        else:
            horizon_weighted_metrics[tgt] = {}
    
    # Horizon degradation summary
    horizon_degradation_summary = {}
    for tgt, analysis in target_analysis.items():
        degraded = analysis.get('degraded_horizons', [])
        if degraded:
            horizon_degradation_summary[tgt] = {
                'count': len(degraded),
                'horizons': [d['horizon'] for d in degraded],
                'max_degradation_pct': float(max([d['degradation_pct'] for d in degraded])) if degraded else 0.0,
                'details': degraded
            }
        else:
            horizon_degradation_summary[tgt] = {
                'count': 0,
                'horizons': [],
                'max_degradation_pct': 0.0,
                'details': []
            }
    
    # Generate recommendations
    recommendations = {}
    for tgt, analysis in target_analysis.items():
        recs = []
        
        # Enhanced linear collapse risk warnings with pattern analysis and error distribution
        risk = analysis.get('linear_collapse_risk', 0.0)
        pattern_similarity = analysis.get('error_pattern_similarity', 0.0)
        error_correlation = analysis.get('horizon_error_correlation', 0.0)
        error_dist_similarity = analysis.get('error_distribution_similarity', 0.0)
        
        if risk > 0.7:
            recs.append(f"CRITICAL: High linear collapse risk ({risk:.2f}) - DDFM encoder likely learning only linear features")
            if pattern_similarity > 0.8:
                recs.append(f"  - Error pattern similarity to DFM: {pattern_similarity:.2f} (very high, suggests identical error structure)")
            if error_correlation > 0.7:
                recs.append(f"  - Horizon error correlation with DFM: {error_correlation:.2f} (high, suggests systematic linear behavior)")
            if error_dist_similarity > 0.7:
                recs.append(f"  - Error distribution similarity: {error_dist_similarity:.2f} (very high, similar skewness/kurtosis suggests linear behavior)")
            recs.append("  - Actions: deeper encoder, tanh activation, weight decay, increased pre-training, smaller batch size")
        elif risk > 0.4:
            recs.append(f"WARNING: Moderate linear collapse risk ({risk:.2f}) - monitor training and consider encoder architecture improvements")
            if pattern_similarity > 0.6:
                recs.append(f"  - Error pattern similarity to DFM: {pattern_similarity:.2f} (moderate, monitor for linear collapse)")
            if error_correlation > 0.5:
                recs.append(f"  - Horizon error correlation with DFM: {error_correlation:.2f} (moderate, suggests some linear behavior)")
            if error_dist_similarity > 0.5:
                recs.append(f"  - Error distribution similarity: {error_dist_similarity:.2f} (moderate, similar error distributions suggest some linear behavior)")
        
        # Degraded horizons
        degraded = analysis.get('degraded_horizons', [])
        if degraded:
            degraded_horizons_list = [d['horizon'] for d in degraded]
            max_degradation = max([d['degradation_pct'] for d in degraded])
            recs.append(f"DDFM performs worse than DFM at horizons {degraded_horizons_list} (max degradation: {max_degradation:.1f}%) - investigate horizon-specific issues")
        
        # Use robust metrics when CV is high (indicating outliers)
        error_cv = analysis.get('error_cv') or 0
        use_robust = error_cv > 0.5
        improvement_metric = analysis.get('robust_improvement_sMAE') if use_robust else analysis.get('improvement_sMAE')
        avg_metric = analysis.get('robust_avg_sMAE') if use_robust else analysis.get('avg_sMAE')
        
        if improvement_metric is not None and improvement_metric < -0.05:
            metric_type = "robust (median-based)" if use_robust else "mean-based"
            recs.append(f"DDFM performs worse than DFM overall ({metric_type} improvement: {improvement_metric:.1%}) - investigate encoder architecture and training")
        
        error_variance = analysis.get('error_variance')
        if error_variance is not None and error_variance > 1.0:
            recs.append(f"High error variance ({error_variance:.2f}) - consider regularization or ensemble methods")
            # When variance is high, recommend using robust metrics
            if not use_robust:
                recs.append("  - Note: High variance detected - robust (median-based) metrics may provide more reliable evaluation")
        
        if error_cv > 0.5:
            recs.append(f"High coefficient of variation ({error_cv:.2f}) - predictions are unstable across horizons")
            recs.append("  - Using robust (median-based) metrics for more reliable evaluation due to outliers")
        
        # Enhanced horizon improvement tracking recommendations
        improvement_dist = analysis.get('horizon_improvement_distribution', {})
        if improvement_dist:
            n_significant = improvement_dist.get('n_significant', 0)
            n_moderate = improvement_dist.get('n_moderate', 0)
            n_degradation = improvement_dist.get('n_degradation', 0)
            total = improvement_dist.get('total_horizons', 0)
            
            if n_significant > 0:
                significant_horizons = improvement_dist.get('categories', {}).get('significant_improvement', [])
                recs.append(f"DDFM shows significant improvement (>10%) at {n_significant}/{total} horizons: {significant_horizons}")
            
            if n_degradation > 0:
                degradation_horizons = improvement_dist.get('categories', {}).get('degradation', [])
                recs.append(f"DDFM performs worse than DFM at {n_degradation}/{total} horizons: {degradation_horizons} - investigate horizon-specific issues")
            
            improvement_fraction = improvement_dist.get('improvement_fraction', 0)
            if improvement_fraction < 0.5:
                recs.append(f"Low improvement fraction ({improvement_fraction:.1%}) - DDFM improves at <50% of horizons, consider encoder architecture improvements")
            elif improvement_fraction > 0.8:
                recs.append(f"High improvement fraction ({improvement_fraction:.1%}) - DDFM improves at >80% of horizons, encoder learning nonlinear features effectively")
        
        # Enhanced error stability recommendations
        error_stability = analysis.get('error_stability')
        if error_stability is not None and error_stability < 0.5:
            recs.append(f"Low error stability ({error_stability:.2f}) - performance is highly variable across horizons, consider regularization or ensemble methods")
        
        # Horizon trend analysis recommendations
        horizon_error_trend = analysis.get('horizon_error_trend')
        if horizon_error_trend and horizon_error_trend.get('interpretation') == 'degrading':
            slope = horizon_error_trend.get('slope', 0)
            r_sq = horizon_error_trend.get('r_squared', 0)
            if r_sq > 0.3:  # Significant trend
                recs.append(f"Error degradation trend detected (slope={slope:.4f}, R²={r_sq:.2f}) - DDFM performance degrades with longer horizons, consider horizon-specific tuning")
        
        horizon_improvement_trend = analysis.get('horizon_improvement_trend')
        if horizon_improvement_trend and horizon_improvement_trend.get('interpretation') == 'degrading':
            slope = horizon_improvement_trend.get('slope', 0)
            r_sq = horizon_improvement_trend.get('r_squared', 0)
            if r_sq > 0.3:
                recs.append(f"Improvement degradation trend detected (slope={slope:.4f}, R²={r_sq:.2f}) - DDFM advantage over DFM decreases with longer horizons")
        
        # Relative improvement consistency recommendations
        rel_consistency = analysis.get('relative_improvement_consistency')
        if rel_consistency is not None:
            if rel_consistency < 0.5:
                recs.append(f"Low relative improvement consistency ({rel_consistency:.1%}) - DDFM improves over DFM at fewer than half of horizons, encoder may need improvement")
            elif rel_consistency > 0.8:
                recs.append(f"High relative improvement consistency ({rel_consistency:.1%}) - DDFM consistently outperforms DFM across most horizons")
        
        # Relative error stability recommendations (NEW)
        rel_error_stability = analysis.get('relative_error_stability')
        rel_error_trend = analysis.get('relative_error_trend')
        if rel_error_stability is not None:
            if rel_error_stability < 0.5:
                recs.append(f"Low relative error stability ({rel_error_stability:.2f}) - DDFM vs DFM performance varies significantly across horizons, consider horizon-specific tuning")
            if rel_error_trend == 'degrading':
                recs.append(f"Relative error degrading trend - DDFM advantage over DFM decreases with longer horizons, investigate factor dynamics or encoder capacity")
            elif rel_error_trend == 'improving':
                recs.append(f"Relative error improving trend - DDFM advantage over DFM increases with longer horizons, encoder captures long-term patterns well")
        
        # Improvement persistence recommendations (NEW)
        persistence_score = analysis.get('improvement_persistence_score')
        improvement_fraction = analysis.get('improvement_fraction')
        consecutive_improvements = analysis.get('consecutive_improvements')
        if persistence_score is not None:
            if persistence_score < 0.3:
                recs.append(f"Low improvement persistence ({persistence_score:.2f}) - DDFM improvements are transient, not consistent across horizons, encoder may need improvement")
            elif persistence_score > 0.7:
                recs.append(f"High improvement persistence ({persistence_score:.2f}) - DDFM improvements are consistent and persistent across horizons")
            if improvement_fraction is not None and improvement_fraction < 0.3:
                recs.append(f"Low improvement fraction ({improvement_fraction:.1%}) - DDFM improves over DFM at fewer than 30% of horizons, improvements may be noise")
            if consecutive_improvements is not None and consecutive_improvements >= 5:
                recs.append(f"Strong consecutive improvement streak ({consecutive_improvements} horizons) - DDFM shows consistent improvement over multiple horizons")
        
        # Quantile-based metrics recommendations (NEW)
        quantile_metrics = analysis.get('quantile_metrics')
        if quantile_metrics is not None:
            tail_ratio = quantile_metrics.get('tail_ratio')
            iqr_smae = quantile_metrics.get('iqr_sMAE')
            if tail_ratio is not None and not np.isnan(tail_ratio):
                if tail_ratio > 5.0:
                    recs.append(f"High tail ratio ({tail_ratio:.2f}) - error distribution has heavy tails, some horizons have extreme errors, consider robust metrics or outlier handling")
                elif tail_ratio < 2.0:
                    recs.append(f"Low tail ratio ({tail_ratio:.2f}) - error distribution is concentrated, errors are relatively uniform across horizons")
            if iqr_smae is not None and not np.isnan(iqr_smae):
                if iqr_smae > 1.0:
                    recs.append(f"High IQR sMAE ({iqr_smae:.2f}) - large spread in errors across horizons, performance is inconsistent")
        
        # Nonlinearity score recommendations (NEW)
        nonlinearity_score = analysis.get('nonlinearity_score')
        pattern_divergence = analysis.get('pattern_divergence')
        error_nonlinearity = analysis.get('error_nonlinearity')
        horizon_interaction = analysis.get('horizon_interaction')
        if nonlinearity_score is not None and not np.isnan(nonlinearity_score):
            if nonlinearity_score < 0.2:
                recs.append(f"Very low nonlinearity score ({nonlinearity_score:.2f}) - DDFM patterns are nearly identical to DFM, encoder likely learning only linear features (linear collapse)")
                recs.append("  - Action: Apply deeper encoder, tanh activation, weight decay, increased pre-training")
            elif nonlinearity_score < 0.4:
                recs.append(f"Low nonlinearity score ({nonlinearity_score:.2f}) - DDFM shows minimal nonlinear behavior, patterns are similar to DFM")
                recs.append("  - Action: Consider encoder architecture improvements or activation function tuning")
            elif nonlinearity_score > 0.7:
                recs.append(f"High nonlinearity score ({nonlinearity_score:.2f}) - DDFM shows strong nonlinear behavior, learning different patterns from DFM")
            else:
                recs.append(f"Moderate nonlinearity score ({nonlinearity_score:.2f}) - DDFM shows some nonlinear behavior")
            
            if pattern_divergence is not None and not np.isnan(pattern_divergence):
                if pattern_divergence < 0.2:
                    recs.append(f"  - Low pattern divergence ({pattern_divergence:.2f}) - DDFM and DFM error patterns are highly correlated, suggesting similar learned features")
                elif pattern_divergence > 0.6:
                    recs.append(f"  - High pattern divergence ({pattern_divergence:.2f}) - DDFM and DFM error patterns differ significantly, encoder learning distinct features")
            
            if error_nonlinearity is not None and not np.isnan(error_nonlinearity):
                if error_nonlinearity > 0.5:
                    recs.append(f"  - High error nonlinearity ({error_nonlinearity:.2f}) - DDFM improvement varies significantly across horizons, indicating horizon-specific nonlinear effects")
            
            if horizon_interaction is not None and not np.isnan(horizon_interaction):
                if horizon_interaction > 0.5:
                    recs.append(f"  - Strong horizon interaction ({horizon_interaction:.2f}) - DDFM shows systematic horizon-specific effects, encoder captures different patterns at different horizons")
        
        # Factor dynamics stability recommendations (NEW)
        factor_stability = analysis.get('factor_dynamics_stability_score')
        factor_oscillation = analysis.get('factor_dynamics_oscillation_detected')
        factor_interpretation = analysis.get('factor_dynamics_interpretation')
        if factor_stability is not None:
            if factor_stability < 0.5:
                recs.append(f"Low factor dynamics stability ({factor_stability:.2f}) - VAR factor dynamics may be unstable, check transition matrix eigenvalues")
                if factor_oscillation:
                    osc_magnitude = analysis.get('factor_dynamics_oscillation_magnitude', 0.0)
                    recs.append(f"  - Oscillatory behavior detected (magnitude={osc_magnitude:.2f}) - suggests complex eigenvalues in VAR transition matrix")
                if factor_interpretation in ['diverging', 'unstable']:
                    recs.append(f"  - Factor dynamics interpretation: {factor_interpretation} - VAR transition matrix may have eigenvalues outside unit circle")
            elif factor_stability > 0.8:
                recs.append(f"High factor dynamics stability ({factor_stability:.2f}) - VAR factor dynamics are stable, encoder is learning well-behaved factors")
        
        growth_rate = analysis.get('factor_dynamics_growth_rate')
        if growth_rate is not None and abs(growth_rate) > 0.05:
            if growth_rate > 0:
                recs.append(f"Factor dynamics growth rate detected ({growth_rate:.4f}) - predictions may diverge at long horizons, consider regularization or factor order reduction")
            else:
                recs.append(f"Factor dynamics decay rate detected ({growth_rate:.4f}) - predictions may converge too quickly, consider increasing factor order or encoder capacity")
        
        # Volatile horizon performance recommendations (NEW)
        volatile_score = analysis.get('volatile_horizon_handling_score')
        n_volatile = analysis.get('n_volatile_horizons', 0)
        avg_volatile_improvement = analysis.get('avg_volatile_improvement')
        relative_advantage = analysis.get('relative_advantage_volatile')
        if volatile_score is not None and not np.isnan(volatile_score):
            if n_volatile > 0:
                if volatile_score < 0.5:
                    recs.append(f"POOR volatile horizon handling (score={volatile_score:.2f}, {n_volatile} volatile horizons) - DDFM struggles with challenging horizons, may indicate linear collapse")
                    recs.append("  - Action: Deeper encoder, tanh activation, weight decay, increased pre-training to learn nonlinear features")
                elif volatile_score < 0.6:
                    recs.append(f"MODERATE volatile horizon handling (score={volatile_score:.2f}, {n_volatile} volatile horizons) - DDFM performance on volatile horizons needs improvement")
                elif volatile_score > 0.7:
                    recs.append(f"EXCELLENT volatile horizon handling (score={volatile_score:.2f}, {n_volatile} volatile horizons) - DDFM excels at challenging horizons, suggesting strong nonlinear learning")
                
                if avg_volatile_improvement is not None and not np.isnan(avg_volatile_improvement):
                    if avg_volatile_improvement < 0:
                        recs.append(f"  - DDFM performs worse than DFM on volatile horizons (improvement={avg_volatile_improvement:.1%}) - encoder may be collapsing to linear behavior")
                    elif avg_volatile_improvement > 0.1:
                        recs.append(f"  - DDFM shows strong improvement on volatile horizons (improvement={avg_volatile_improvement:.1%}) - encoder learning nonlinear features effectively")
                
                if relative_advantage is not None and not np.isnan(relative_advantage):
                    if relative_advantage > 0.05:
                        recs.append(f"  - DDFM improves more on volatile horizons than stable (advantage={relative_advantage:.1%}) - encoder excels at complex nonlinear dynamics")
                    elif relative_advantage < -0.05:
                        recs.append(f"  - DDFM improves less on volatile horizons than stable (disadvantage={relative_advantage:.1%}) - encoder may struggle with complex patterns")
            else:
                recs.append(f"No volatile horizons detected - all horizons have relatively stable error patterns")
        
        # Enhanced stability warnings
        ratio_cv = analysis.get('smse_smae_ratio_cv', 0.0)
        if ratio_cv > 0.3:
            recs.append(f"High sMSE/sMAE ratio variation ({ratio_cv:.2f}) - prediction error structure is unstable across horizons, consider regularization")
        
        volatile = volatile_horizons.get(tgt, {})
        if volatile.get('volatile_count', 0) > 0:
            recs.append(f"Volatile horizons detected: {volatile['volatile_horizons']} - consider horizon-specific tuning")
        
        # Check improvement using appropriate metric (robust if outliers present)
        improvement_to_check = analysis.get('robust_improvement_sMAE') if use_robust else analysis.get('improvement_sMAE')
        if improvement_to_check is not None and improvement_to_check < 0.1:
            metric_type = "robust (median-based)" if use_robust else "mean-based"
            recs.append(f"Minimal improvement over DFM ({metric_type} improvement: {improvement_to_check:.1%}) - encoder may be learning linear features only")
        
        if analysis.get('consistency', 1.0) < 0.5:
            recs.append(f"Low consistency ({analysis['consistency']:.2f}) - improvement varies significantly across horizons")
        
        if analysis.get('short_term_avg_sMAE') is not None and analysis.get('long_term_avg_sMAE') is not None:
            if analysis['long_term_avg_sMAE'] > analysis['short_term_avg_sMAE'] * 1.5:
                recs.append("Long-term performance significantly worse than short-term - consider horizon-specific models")
        
        # Horizon-weighted metrics recommendations
        weighted_improvement = analysis.get('weighted_improvement_pct')
        if weighted_improvement is not None:
            if weighted_improvement < 0:
                recs.append(f"Horizon-weighted improvement is negative ({weighted_improvement:.1f}%) - DDFM performs worse than DFM when prioritizing short-term horizons")
            elif weighted_improvement < 5:
                recs.append(f"Horizon-weighted improvement is minimal ({weighted_improvement:.1f}%) - consider improvements to short-term forecasting")
        
        # Analyze error distribution patterns if diagnostic metrics are available
        # These metrics help identify systematic error patterns that indicate DDFM issues
        # Check if error distribution metrics are available in aggregated results
        target_ddfm_data = aggregated_results[
            (aggregated_results['target'] == tgt) & 
            (aggregated_results['model'] == 'DDFM')
        ]
        target_dfm_data = aggregated_results[
            (aggregated_results['target'] == tgt) & 
            (aggregated_results['model'] == 'DFM')
        ]
        
        # Analyze error distribution patterns if available
        if 'error_skewness' in target_ddfm_data.columns and 'error_skewness' in target_dfm_data.columns:
            ddfm_skewness = target_ddfm_data['error_skewness'].dropna()
            dfm_skewness = target_dfm_data['error_skewness'].dropna()
            
            if len(ddfm_skewness) > 0 and len(dfm_skewness) > 0:
                avg_ddfm_skew = ddfm_skewness.mean()
                avg_dfm_skew = dfm_skewness.mean()
                
                # Similar skewness suggests similar error distributions (potential linear collapse)
                if abs(avg_ddfm_skew - avg_dfm_skew) < 0.2:
                    recs.append(f"Error skewness similarity (DDFM={avg_ddfm_skew:.2f}, DFM={avg_dfm_skew:.2f}) - similar error distributions suggest linear behavior")
                
                # High positive skewness indicates right-skewed errors (underprediction bias)
                if avg_ddfm_skew > 1.0:
                    recs.append(f"High positive error skewness ({avg_ddfm_skew:.2f}) - systematic underprediction bias, consider adjusting loss function or regularization")
                # High negative skewness indicates left-skewed errors (overprediction bias)
                elif avg_ddfm_skew < -1.0:
                    recs.append(f"High negative error skewness ({avg_ddfm_skew:.2f}) - systematic overprediction bias, consider adjusting loss function or regularization")
        
        if 'error_kurtosis' in target_ddfm_data.columns and 'error_kurtosis' in target_dfm_data.columns:
            ddfm_kurtosis = target_ddfm_data['error_kurtosis'].dropna()
            dfm_kurtosis = target_dfm_data['error_kurtosis'].dropna()
            
            if len(ddfm_kurtosis) > 0 and len(dfm_kurtosis) > 0:
                avg_ddfm_kurt = ddfm_kurtosis.mean()
                avg_dfm_kurt = dfm_kurtosis.mean()
                
                # Similar kurtosis suggests similar error tail behavior
                if abs(avg_ddfm_kurt - avg_dfm_kurt) < 1.0:
                    recs.append(f"Error kurtosis similarity (DDFM={avg_ddfm_kurt:.2f}, DFM={avg_dfm_kurt:.2f}) - similar error tail behavior suggests linear behavior")
                
                # High kurtosis indicates heavy-tailed errors (outlier-prone predictions)
                if avg_ddfm_kurt > 3.0:
                    recs.append(f"High error kurtosis ({avg_ddfm_kurt:.2f}) - heavy-tailed error distribution, predictions are outlier-prone, consider Huber loss or robust regularization")
        
        if 'error_bias_squared' in target_ddfm_data.columns and 'error_variance' in target_ddfm_data.columns:
            ddfm_bias_sq = target_ddfm_data['error_bias_squared'].dropna()
            ddfm_var = target_ddfm_data['error_variance'].dropna()
            
            if len(ddfm_bias_sq) > 0 and len(ddfm_var) > 0:
                avg_bias_sq = ddfm_bias_sq.mean()
                avg_var = ddfm_var.mean()
                
                # High bias relative to variance indicates systematic prediction errors
                if avg_bias_sq > 0 and avg_var > 0:
                    bias_var_ratio = avg_bias_sq / (avg_bias_sq + avg_var)
                    if bias_var_ratio > 0.5:
                        recs.append(f"High bias component ({bias_var_ratio:.1%}) - systematic prediction errors dominate, consider adjusting model architecture or loss function")
                    elif bias_var_ratio < 0.2:
                        recs.append(f"Low bias component ({bias_var_ratio:.1%}) - errors are mostly variance, consider regularization or ensemble methods")
        
        if 'error_concentration' in target_ddfm_data.columns:
            ddfm_conc = target_ddfm_data['error_concentration'].dropna()
            if len(ddfm_conc) > 0:
                avg_conc = ddfm_conc.mean()
                # High concentration indicates errors are concentrated at specific points
                if avg_conc > 0.7:
                    recs.append(f"High error concentration ({avg_conc:.2f}) - errors are concentrated at specific points, investigate data quality or model assumptions")
        
        if 'prediction_bias' in target_ddfm_data.columns:
            ddfm_bias = target_ddfm_data['prediction_bias'].dropna()
            if len(ddfm_bias) > 0:
                avg_bias = ddfm_bias.mean()
                # Significant bias indicates systematic over/underprediction
                if abs(avg_bias) > 0.1:
                    bias_direction = "overprediction" if avg_bias > 0 else "underprediction"
                    recs.append(f"Systematic {bias_direction} bias ({avg_bias:.3f}) - model consistently {'over' if avg_bias > 0 else 'under'}predicts, consider adjusting loss function or regularization")
        
        # Error pattern smoothness recommendations (NEW)
        error_smoothness = analysis.get('error_smoothness_score')
        if error_smoothness is not None and not np.isnan(error_smoothness):
            if error_smoothness < 0.3:
                recs.append(f"Very rough error patterns (smoothness={error_smoothness:.2f}) - errors vary significantly across horizons, may indicate training instability or encoder issues")
            elif error_smoothness > 0.7:
                recs.append(f"Smooth error patterns (smoothness={error_smoothness:.2f}) - errors are consistent across horizons, encoder learning stable features")
        
        # Improvement significance recommendations (NEW)
        improvement_is_sig = analysis.get('improvement_is_significant')
        improvement_p_val = analysis.get('improvement_p_value')
        if improvement_is_sig is not None:
            if improvement_is_sig:
                n_sig_horizons = analysis.get('n_significant_horizons', 0)
                if n_sig_horizons > 0:
                    recs.append(f"Statistically significant improvement detected (p={improvement_p_val:.3f}) - DDFM improvement is statistically significant at {n_sig_horizons} horizon(s)")
                else:
                    recs.append(f"Statistically significant overall improvement (p={improvement_p_val:.3f}) - DDFM improvement is statistically significant")
            else:
                if improvement_p_val is not None and improvement_p_val > 0.1:
                    recs.append(f"Improvement not statistically significant (p={improvement_p_val:.3f}) - DDFM improvement may be due to noise, consider more training or architecture improvements")
        
        # Note about additional metrics requiring actual predictions
        # Forecast skill score and information gain metrics are available in evaluation_metrics.py
        # but require actual predictions (y_true, y_pred) which are not stored in aggregated_results.csv.
        # These metrics can be calculated during evaluation or from comparison results if predictions
        # are stored in outputs/comparisons/ JSON files. They provide additional insights:
        # - Forecast skill score: Compares model to naive baseline (persistence/mean forecast)
        # - Information gain: Quantifies value of nonlinear features learned by DDFM encoder
        # For now, the analysis uses available metrics from aggregated_results.csv.
        
        recommendations[tgt] = recs if recs else ["Performance is acceptable"]
    
    # Calculate cross-target pattern comparison (NEW)
    cross_target_analysis = None
    try:
        from src.evaluation.evaluation_metrics import calculate_cross_target_pattern_comparison
        cross_target_analysis = calculate_cross_target_pattern_comparison(
            aggregated_results, metric='sMAE'
        )
    except Exception as e:
        logger.debug(f"Could not calculate cross-target pattern comparison: {e}")
        cross_target_analysis = None
    
    results = _json_safe({
        'target_analysis': target_analysis,
        'volatile_horizons': volatile_horizons,
        'recommendations': recommendations,
        'linear_collapse_risk': linear_collapse_risk_summary,
        'horizon_degradation': horizon_degradation_summary,
        'horizon_weighted_metrics': horizon_weighted_metrics,  # NEW: Horizon-weighted metrics summary
        'cross_target_comparison': cross_target_analysis,  # NEW: Cross-target pattern comparison
        'metrics_limitations': {
            'note': 'Analysis uses metrics from aggregated_results.csv. Forecast skill score and information gain require actual predictions and are not calculated here.',
            'forecast_skill_score': 'Available in evaluation_metrics.calculate_forecast_skill_score() but requires y_true, y_pred',
            'information_gain': 'Available in evaluation_metrics.calculate_information_gain() but requires y_true, y_pred_ddfm, y_pred_dfm',
            'factor_dynamics_stability': 'Uses error patterns (sMAE) as proxy for predictions. For full accuracy, actual predictions needed.'
        }
    })
    
    # Save if output path provided
    if output_path:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path_obj, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Prediction quality analysis saved to: {output_path}")
    
    # Log summary with enhanced diagnostics
    logger.info("DDFM Prediction Quality Analysis Summary (Enhanced):")
    for tgt, analysis in target_analysis.items():
        improvement = analysis.get('improvement_sMAE')
        consistency = analysis.get('consistency', 0)
        error_cv = analysis.get('error_cv') or 0
        risk = analysis.get('linear_collapse_risk', 0.0)
        pattern_similarity = analysis.get('error_pattern_similarity', 0.0)
        error_correlation = analysis.get('horizon_error_correlation', 0.0)
        ratio_cv = analysis.get('smse_smae_ratio_cv', 0.0)
        degraded_count = len(analysis.get('degraded_horizons', []))
        best_horizon = analysis.get('best_horizon')
        worst_horizon = analysis.get('worst_horizon')
        best_improvement = analysis.get('best_improvement')
        worst_improvement = analysis.get('worst_improvement')
        
        weighted_improvement = analysis.get('weighted_improvement_pct')
        
        error_stability = analysis.get('error_stability')
        rel_consistency = analysis.get('relative_improvement_consistency')
        horizon_error_trend = analysis.get('horizon_error_trend')
        horizon_improvement_trend = analysis.get('horizon_improvement_trend')
        
        # Use robust metrics in summary if CV is high (outliers present)
        use_robust_summary = error_cv > 0.5
        avg_metric = analysis.get('robust_avg_sMAE') if use_robust_summary else analysis.get('avg_sMAE')
        improvement_metric = analysis.get('robust_improvement_sMAE') if use_robust_summary else improvement
        
        if improvement_metric is not None:
            metric_type = "robust" if use_robust_summary else "mean"
            log_msg = (
                f"  {tgt}: {metric_type}_sMAE={avg_metric:.3f}, {metric_type}_improvement={improvement_metric:.1%}, "
                f"consistency={consistency:.2f}, CV={error_cv:.2f}, collapse_risk={risk:.2f}"
            )
            if weighted_improvement is not None:
                log_msg += f", weighted_improvement={weighted_improvement:.1f}%"
            if error_stability is not None:
                log_msg += f", stability={error_stability:.2f}"
            if rel_consistency is not None:
                log_msg += f", rel_consistency={rel_consistency:.1%}"
            if pattern_similarity > 0.6:
                log_msg += f", pattern_sim={pattern_similarity:.2f}"
            if abs(error_correlation) > 0.5:
                log_msg += f", error_corr={error_correlation:.2f}"
            if ratio_cv > 0.2:
                log_msg += f", ratio_CV={ratio_cv:.2f}"
            if degraded_count > 0:
                log_msg += f", degraded_horizons={degraded_count}"
            if horizon_error_trend and horizon_error_trend.get('r_squared', 0) > 0.3:
                trend_int = horizon_error_trend.get('interpretation', 'unknown')
                log_msg += f", error_trend={trend_int}"
            if horizon_improvement_trend and horizon_improvement_trend.get('r_squared', 0) > 0.3:
                trend_int = horizon_improvement_trend.get('interpretation', 'unknown')
                log_msg += f", improvement_trend={trend_int}"
            if best_horizon is not None and worst_horizon is not None:
                log_msg += f", best_h={best_horizon}({best_improvement:.1%}), worst_h={worst_horizon}({worst_improvement:.1%})"
            logger.info(log_msg)
        else:
            log_msg = f"  {tgt}: avg_sMAE={analysis['avg_sMAE']:.3f}, consistency={consistency:.2f}, CV={error_cv:.2f}, collapse_risk={risk:.2f}, pattern_sim={pattern_similarity:.2f}, error_corr={error_correlation:.2f}"
            if weighted_improvement is not None:
                log_msg += f", weighted_improvement={weighted_improvement:.1f}%"
            if error_stability is not None:
                log_msg += f", stability={error_stability:.2f}"
            if rel_consistency is not None:
                log_msg += f", rel_consistency={rel_consistency:.1%}"
            logger.info(log_msg)
    
    return results


def analyze_horizon_specific_ddfm_issues(
    aggregated_results: pd.DataFrame,
    target: Optional[str] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze horizon-specific issues for DDFM predictions.
    
    This function identifies problematic horizons where DDFM shows:
    - High error spikes compared to DFM
    - Inconsistent performance patterns
    - Degradation at specific horizons
    - Missing predictions (n_valid=0)
    
    This analysis helps identify which forecast horizons are problematic for DDFM,
    enabling targeted improvements (e.g., horizon-specific tuning, regularization,
    or architecture adjustments).
    
    Parameters
    ----------
    aggregated_results : pd.DataFrame
        Aggregated results DataFrame from aggregate_overall_performance()
        Must have columns: target, model, horizon, sMSE, sMAE, sRMSE, n_valid
    target : str, optional
        Target series to analyze. If None, analyzes all targets.
    output_path : str, optional
        Path to save analysis results (JSON format)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'horizon_issues': Per-target horizon-specific issues with detailed metrics
        - 'missing_horizons': Horizons with n_valid=0 (predictions failed)
        - 'error_spikes': Horizons with unusually high error (error > mean + 2*std)
        - 'recommendations': Horizon-specific recommendations for improvement
        
    Notes
    -----
    - Problematic horizons are defined as those where DDFM error is >10% worse than DFM
    - Error spikes are identified using statistical outlier detection (mean + 2*std)
    - Missing horizons indicate validation failures or numerical instability
    """
    logger.info(f"Analyzing horizon-specific DDFM issues{' for ' + target if target else ''}")
    
    if aggregated_results.empty:
        logger.warning("Empty aggregated_results, cannot analyze horizon-specific issues")
        return {'error': 'Empty results'}
    
    # Filter for DDFM and DFM models
    dfm_ddfm = aggregated_results[
        aggregated_results['model'].isin(['DFM', 'DDFM'])
    ].copy()
    
    if target:
        dfm_ddfm = dfm_ddfm[dfm_ddfm['target'] == target]
    
    if dfm_ddfm.empty:
        logger.warning(f"No DFM/DDFM results found{' for ' + target if target else ''}")
        return {'error': 'No DFM/DDFM results'}
    
    horizon_issues = {}
    missing_horizons = {}
    error_spikes = {}
    
    for tgt in dfm_ddfm['target'].unique():
        target_data = dfm_ddfm[dfm_ddfm['target'] == tgt]
        
        ddfm_data = target_data[target_data['model'] == 'DDFM'].sort_values('horizon')
        dfm_data = target_data[target_data['model'] == 'DFM'].sort_values('horizon')
        
        # Find missing horizons (n_valid=0)
        missing = ddfm_data[ddfm_data['n_valid'] == 0]['horizon'].tolist()
        missing_horizons[tgt] = missing
        
        # Find common horizons with valid data
        valid_ddfm = ddfm_data[ddfm_data['n_valid'] > 0]
        valid_dfm = dfm_data[dfm_data['n_valid'] > 0]
        
        common_horizons = set(valid_ddfm['horizon'].values) & set(valid_dfm['horizon'].values)
        
        if len(common_horizons) == 0:
            continue
        
        # Analyze each horizon
        horizon_details = []
        ddfm_errors = []
        
        for h in sorted(common_horizons):
            ddfm_row = valid_ddfm[valid_ddfm['horizon'] == h].iloc[0]
            dfm_row = valid_dfm[valid_dfm['horizon'] == h].iloc[0]
            
            ddfm_smae = ddfm_row['sMAE']
            dfm_smae = dfm_row['sMAE']
            
            if pd.isna(ddfm_smae) or pd.isna(dfm_smae):
                continue
            
            # Calculate relative error
            error_ratio = ddfm_smae / max(dfm_smae, 1e-10)
            improvement = (dfm_smae - ddfm_smae) / max(dfm_smae, 1e-10)
            
            ddfm_errors.append(ddfm_smae)
            
            horizon_details.append({
                'horizon': int(h),
                'ddfm_sMAE': float(ddfm_smae),
                'dfm_sMAE': float(dfm_smae),
                'error_ratio': float(error_ratio),
                'improvement': float(improvement),
                'is_problematic': error_ratio > 1.1 or improvement < -0.1  # DDFM >10% worse or >10% worse than DFM
            })
        
        if len(ddfm_errors) > 0:
            # Identify error spikes (error > mean + 2*std)
            error_mean = np.mean(ddfm_errors)
            error_std = np.std(ddfm_errors)
            spike_threshold = error_mean + 2 * error_std
            
            spikes = [d for d in horizon_details if d['ddfm_sMAE'] > spike_threshold]
            error_spikes[tgt] = [s['horizon'] for s in spikes]
        
        horizon_issues[tgt] = {
            'total_horizons': len(common_horizons),
            'missing_horizons': missing,
            'problematic_horizons': [d['horizon'] for d in horizon_details if d['is_problematic']],
            'error_spikes': error_spikes.get(tgt, []),
            'horizon_details': horizon_details
        }
    
    # Generate recommendations
    recommendations = {}
    for tgt, issues in horizon_issues.items():
        recs = []
        
        if issues['missing_horizons']:
            recs.append(f"Missing predictions at horizons {issues['missing_horizons']} - investigate validation logic or numerical stability")
        
        if issues['error_spikes']:
            recs.append(f"Error spikes at horizons {issues['error_spikes']} - consider horizon-specific tuning or regularization")
        
        if issues['problematic_horizons']:
            recs.append(f"Problematic horizons {issues['problematic_horizons']} where DDFM performs worse than DFM - investigate encoder learning at these horizons")
        
        recommendations[tgt] = recs if recs else ["No significant horizon-specific issues detected"]
    
    results = {
        'horizon_issues': horizon_issues,
        'missing_horizons': missing_horizons,
        'error_spikes': error_spikes,
        'recommendations': recommendations
    }
    
    # Save if output path provided
    if output_path:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path_obj, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Horizon-specific analysis saved to: {output_path}")
    
    # Log summary
    logger.info("Horizon-Specific DDFM Issues Analysis Summary:")
    for tgt, issues in horizon_issues.items():
        logger.info(f"  {tgt}: {len(issues['problematic_horizons'])} problematic, {len(issues['error_spikes'])} spikes, {len(issues['missing_horizons'])} missing")
    
    return results


def analyze_correlation_structure(
    data_file: str,
    target_series: str,
    train_start: str = '1985-01-01',
    train_end: str = '2019-12-31',
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze correlation structure for a target series (Phase 0 of DDFM improvement research).
    
    This function analyzes the correlation patterns between a target series and all input series
    to understand why DDFM might show linear behavior (identical to DFM). This analysis can be
    done before training models.
    
    Parameters
    ----------
    data_file : str
        Path to data CSV file
    target_series : str
        Target series name (e.g., 'KOEQUIPTE')
    train_start : str
        Training period start date (default: '1985-01-01')
    train_end : str
        Training period end date (default: '2019-12-31')
    output_path : str, optional
        Path to save correlation analysis results (JSON format)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'target_series': Target series name
        - 'correlation_matrix': Full correlation matrix (pd.DataFrame)
        - 'target_correlations': Correlations between target and all other series (pd.Series)
        - 'negative_correlations': Count and details of negative correlations
        - 'positive_correlations': Count and details of positive correlations
        - 'summary': Summary statistics
    """
    logger.info(f"Analyzing correlation structure for {target_series}")
    
    # Load data
    data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    # Filter to training period
    train_start_ts = pd.Timestamp(train_start)
    train_end_ts = pd.Timestamp(train_end)
    data = data[(data.index >= train_start_ts) & (data.index <= train_end_ts)]
    
    if target_series not in data.columns:
        raise ValueError(f"Target series '{target_series}' not found in data. Available: {list(data.columns)}")
    
    # Calculate correlation matrix
    # Use only numeric columns and drop rows with all NaN
    numeric_data = data.select_dtypes(include=[np.number])
    numeric_data = numeric_data.dropna(how='all')
    
    if len(numeric_data) < 2:
        raise ValueError(f"Insufficient data for correlation analysis: {len(numeric_data)} rows")
    
    correlation_matrix = numeric_data.corr()
    
    # Extract correlations with target series
    if target_series not in correlation_matrix.columns:
        raise ValueError(f"Target series '{target_series}' not in correlation matrix")
    
    target_correlations = correlation_matrix[target_series].drop(target_series)  # Exclude self-correlation
    
    # Analyze negative vs positive correlations
    negative_corrs = target_correlations[target_correlations < 0]
    positive_corrs = target_correlations[target_correlations > 0]
    
    # Count correlations by magnitude
    strong_negative = (target_correlations < -0.3).sum()
    moderate_negative = ((target_correlations >= -0.3) & (target_correlations < -0.1)).sum()
    weak_negative = ((target_correlations >= -0.1) & (target_correlations < 0)).sum()
    
    strong_positive = (target_correlations > 0.3).sum()
    moderate_positive = ((target_correlations > 0.1) & (target_correlations <= 0.3)).sum()
    weak_positive = ((target_correlations > 0) & (target_correlations <= 0.1)).sum()
    
    summary = {
        'total_series': len(target_correlations),
        'negative_count': len(negative_corrs),
        'positive_count': len(positive_corrs),
        'negative_fraction': len(negative_corrs) / len(target_correlations) if len(target_correlations) > 0 else 0.0,
        'strong_negative_count': int(strong_negative),
        'moderate_negative_count': int(moderate_negative),
        'weak_negative_count': int(weak_negative),
        'strong_positive_count': int(strong_positive),
        'moderate_positive_count': int(moderate_positive),
        'weak_positive_count': int(weak_positive),
        'mean_correlation': float(target_correlations.mean()),
        'median_correlation': float(target_correlations.median()),
        'min_correlation': float(target_correlations.min()),
        'max_correlation': float(target_correlations.max()),
        'std_correlation': float(target_correlations.std())
    }
    
    # Get top negative and positive correlations
    top_negative = target_correlations.nsmallest(10).to_dict()
    top_positive = target_correlations.nlargest(10).to_dict()
    
    results = {
        'target_series': target_series,
        'correlation_matrix': correlation_matrix,
        'target_correlations': target_correlations,
        'negative_correlations': {
            'count': len(negative_corrs),
            'series': negative_corrs.to_dict(),
            'top_10': top_negative
        },
        'positive_correlations': {
            'count': len(positive_corrs),
            'series': positive_corrs.to_dict(),
            'top_10': top_positive
        },
        'summary': summary
    }
    
    # Save if output path provided
    if output_path:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Save summary as JSON (exclude DataFrame objects)
        json_results = {
            'target_series': target_series,
            'negative_correlations': results['negative_correlations'],
            'positive_correlations': results['positive_correlations'],
            'summary': summary
        }
        
        with open(output_path_obj, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Correlation analysis saved to: {output_path}")
    
    logger.info(f"Correlation analysis complete for {target_series}:")
    logger.info(f"  Total series: {summary['total_series']}")
    logger.info(f"  Negative correlations: {summary['negative_count']} ({summary['negative_fraction']*100:.1f}%)")
    logger.info(f"  Strong negative (|r| > 0.3): {summary['strong_negative_count']}")
    logger.info(f"  Mean correlation: {summary['mean_correlation']:.3f}")
    logger.info(f"  Min correlation: {summary['min_correlation']:.3f}")
    logger.info(f"  Max correlation: {summary['max_correlation']:.3f}")
    
    return results


def analyze_horizon_error_correlation(
    aggregated_results: pd.DataFrame,
    target: Optional[str] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze correlation of errors across forecast horizons for DDFM.
    
    This function analyzes how errors correlate across different forecast horizons
    to identify systematic patterns. High correlation suggests systematic issues
    (e.g., encoder learning linear features), while low correlation suggests
    horizon-specific issues.
    
    Parameters
    ----------
    aggregated_results : pd.DataFrame
        Aggregated results DataFrame from aggregate_overall_performance()
        Must have columns: target, model, horizon, sMAE, sMSE, sRMSE
    target : str, optional
        Target series to analyze. If None, analyzes all targets.
    output_path : str, optional
        Path to save analysis results (JSON format)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'correlation_matrix': Error correlation matrix across horizons
        - 'avg_correlation': Average correlation across horizons
        - 'max_correlation': Maximum correlation (identifies most similar horizon pairs)
        - 'min_correlation': Minimum correlation (identifies most different horizon pairs)
        - 'systematic_pattern_score': Score indicating systematic vs horizon-specific errors (0-1)
        - 'recommendations': Recommendations based on correlation patterns
    """
    logger.info(f"Analyzing horizon error correlation{' for ' + target if target else ''}")
    
    if aggregated_results.empty:
        logger.warning("Empty aggregated_results, cannot analyze horizon correlation")
        return {'error': 'Empty results'}
    
    # Filter for DDFM only
    ddfm_data = aggregated_results[
        aggregated_results['model'] == 'DDFM'
    ].copy()
    
    if target:
        ddfm_data = ddfm_data[ddfm_data['target'] == target]
    
    if ddfm_data.empty:
        logger.warning(f"No DDFM results found{' for ' + target if target else ''}")
        return {'error': 'No DDFM results'}
    
    results = {}
    
    for tgt in ddfm_data['target'].unique():
        target_data = ddfm_data[ddfm_data['target'] == tgt].sort_values('horizon')
        
        # Extract errors by horizon
        horizons = target_data['horizon'].values
        smae_by_horizon = target_data['sMAE'].values
        smse_by_horizon = target_data['sMSE'].values
        
        # Create error vectors (using sMAE as primary metric)
        valid_mask = ~np.isnan(smae_by_horizon)
        valid_horizons = horizons[valid_mask]
        valid_errors = smae_by_horizon[valid_mask]
        
        if len(valid_horizons) < 2:
            logger.warning(f"Insufficient horizons for correlation analysis for {tgt}")
            continue
        
        # Calculate correlation matrix of errors across horizons
        # For this, we need error values at each horizon
        # Since we only have aggregated metrics, we'll use the error values directly
        # and calculate correlation assuming errors are independent samples
        
        # Create a simple correlation measure: how similar are errors at different horizons?
        # Higher correlation = errors are similar across horizons (systematic issue)
        # Lower correlation = errors vary by horizon (horizon-specific issue)
        
        # For aggregated results, we can't calculate true correlation without raw predictions
        # Instead, we'll calculate a similarity metric based on error patterns
        
        # Calculate pairwise similarity of errors (normalized)
        n_horizons = len(valid_horizons)
        similarity_matrix = np.zeros((n_horizons, n_horizons))
        
        for i in range(n_horizons):
            for j in range(n_horizons):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Similarity based on relative error magnitude
                    # If errors are similar in magnitude, similarity is high
                    err_i = valid_errors[i]
                    err_j = valid_errors[j]
                    if err_i > 0 and err_j > 0:
                        # Normalized difference: 1 - |err_i - err_j| / max(err_i, err_j)
                        max_err = max(err_i, err_j)
                        similarity = 1.0 - min(abs(err_i - err_j) / max_err, 1.0)
                        similarity_matrix[i, j] = similarity
                    else:
                        similarity_matrix[i, j] = 0.0
        
        # Calculate average similarity (excluding diagonal)
        mask = ~np.eye(n_horizons, dtype=bool)
        avg_similarity = float(np.mean(similarity_matrix[mask]))
        
        # Systematic pattern score: higher = more systematic (errors similar across horizons)
        systematic_pattern_score = avg_similarity
        
        # Find most and least similar horizon pairs
        max_similarity = float(np.max(similarity_matrix[mask]))
        min_similarity = float(np.min(similarity_matrix[mask]))
        
        # Find horizon pairs with max/min similarity
        max_idx = np.unravel_index(np.argmax(similarity_matrix[mask]), similarity_matrix.shape)
        min_idx = np.unravel_index(np.argmin(similarity_matrix[mask]), similarity_matrix.shape)
        
        max_pair = (int(valid_horizons[max_idx[0]]), int(valid_horizons[max_idx[1]]))
        min_pair = (int(valid_horizons[min_idx[0]]), int(valid_horizons[min_idx[1]]))
        
        # Generate recommendations
        recs = []
        if systematic_pattern_score > 0.7:
            recs.append("High error similarity across horizons - suggests systematic issue (e.g., linear collapse). Consider encoder architecture improvements.")
        elif systematic_pattern_score < 0.3:
            recs.append("Low error similarity across horizons - suggests horizon-specific issues. Consider horizon-specific tuning or regularization.")
        else:
            recs.append("Moderate error similarity - errors show some systematic patterns but also horizon-specific variation.")
        
        results[tgt] = {
            'n_horizons': int(n_horizons),
            'valid_horizons': [int(h) for h in valid_horizons],
            'similarity_matrix': similarity_matrix.tolist(),
            'avg_similarity': avg_similarity,
            'max_similarity': max_similarity,
            'min_similarity': min_similarity,
            'most_similar_pair': max_pair,
            'least_similar_pair': min_pair,
            'systematic_pattern_score': systematic_pattern_score,
            'recommendations': recs
        }
    
    # Save if output path provided
    if output_path:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path_obj, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Horizon error correlation analysis saved to: {output_path}")
    
    # Log summary
    logger.info("Horizon Error Correlation Analysis Summary:")
    for tgt, analysis in results.items():
        logger.info(
            f"  {tgt}: avg_similarity={analysis['avg_similarity']:.3f}, "
            f"systematic_score={analysis['systematic_pattern_score']:.3f}, "
            f"n_horizons={analysis['n_horizons']}"
        )
    
    return results


def analyze_missing_horizons(
    aggregated_results: pd.DataFrame,
    target: Optional[str] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze missing horizons (n_valid=0) to identify validation issues.
    
    This function identifies horizons where predictions fail validation (n_valid=0)
    and provides insights into potential causes:
    - Data availability issues (test data not available for horizon)
    - Numerical instability at long horizons
    - Validation threshold too strict
    - Model-specific prediction failures
    
    Parameters
    ----------
    aggregated_results : pd.DataFrame
        Aggregated results DataFrame from aggregate_overall_performance()
        Must have columns: target, model, horizon, n_valid
    target : str, optional
        Target series to analyze. If None, analyzes all targets.
    output_path : str, optional
        Path to save analysis results (JSON format)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'missing_by_target': Missing horizons per target
        - 'missing_by_model': Missing horizons per model
        - 'missing_by_horizon': Which targets/models fail at each horizon
        - 'patterns': Identified patterns (e.g., all models fail at horizon 22)
        - 'recommendations': Recommendations for fixing missing horizons
    """
    logger.info(f"Analyzing missing horizons{' for ' + target if target else ''}")
    
    if aggregated_results.empty:
        logger.warning("Empty aggregated_results, cannot analyze missing horizons")
        return {'error': 'Empty results'}
    
    # Filter by target if specified
    data = aggregated_results.copy()
    if target:
        data = data[data['target'] == target]
    
    if data.empty:
        logger.warning(f"No results found{' for ' + target if target else ''}")
        return {'error': 'No results'}
    
    # Identify missing horizons (n_valid=0)
    missing = data[data['n_valid'] == 0].copy()
    valid = data[data['n_valid'] > 0].copy()
    
    # Missing by target
    missing_by_target = {}
    for tgt in data['target'].unique():
        target_data = data[data['target'] == tgt]
        missing_horizons = target_data[target_data['n_valid'] == 0]['horizon'].tolist()
        valid_horizons = target_data[target_data['n_valid'] > 0]['horizon'].tolist()
        missing_by_target[tgt] = {
            'missing_horizons': sorted(missing_horizons),
            'valid_horizons': sorted(valid_horizons),
            'missing_count': len(missing_horizons),
            'valid_count': len(valid_horizons),
            'completion_rate': len(valid_horizons) / len(target_data) if len(target_data) > 0 else 0.0
        }
    
    # Missing by model
    missing_by_model = {}
    for model in data['model'].unique():
        model_data = data[data['model'] == model]
        missing_horizons = model_data[model_data['n_valid'] == 0]['horizon'].tolist()
        valid_horizons = model_data[model_data['n_valid'] > 0]['horizon'].tolist()
        missing_by_model[model] = {
            'missing_horizons': sorted(set(missing_horizons)),
            'valid_horizons': sorted(set(valid_horizons)),
            'missing_count': len(set(missing_horizons)),
            'valid_count': len(set(valid_horizons)),
            'completion_rate': len(set(valid_horizons)) / len(model_data['horizon'].unique()) if len(model_data['horizon'].unique()) > 0 else 0.0
        }
    
    # Missing by horizon (which targets/models fail at each horizon)
    missing_by_horizon = {}
    for horizon in sorted(data['horizon'].unique()):
        horizon_data = data[data['horizon'] == horizon]
        missing_targets = horizon_data[horizon_data['n_valid'] == 0]['target'].unique().tolist()
        missing_models = horizon_data[horizon_data['n_valid'] == 0]['model'].unique().tolist()
        valid_targets = horizon_data[horizon_data['n_valid'] > 0]['target'].unique().tolist()
        valid_models = horizon_data[horizon_data['n_valid'] > 0]['model'].unique().tolist()
        
        missing_by_horizon[int(horizon)] = {
            'missing_targets': missing_targets,
            'missing_models': missing_models,
            'valid_targets': valid_targets,
            'valid_models': valid_models,
            'missing_count': len(missing_targets) * len(missing_models) if missing_targets and missing_models else 0,
            'is_fully_missing': len(valid_targets) == 0 and len(valid_models) == 0  # All targets/models fail
        }
    
    # Identify patterns
    patterns = {
        'horizons_all_models_fail': [],  # Horizons where all models fail
        'horizons_specific_model_fails': {},  # Horizons where specific models fail
        'long_horizon_failures': [],  # Failures at longest horizons (likely data/validation issue)
        'consistent_failures': {}  # Targets/models that consistently fail at same horizons
    }
    
    # Find horizons where all models fail
    for horizon, info in missing_by_horizon.items():
        if info['is_fully_missing']:
            patterns['horizons_all_models_fail'].append(horizon)
        # Check if it's a long horizon (>= 20)
        if horizon >= 20 and (info['missing_targets'] or info['missing_models']):
            patterns['long_horizon_failures'].append({
                'horizon': horizon,
                'missing_targets': info['missing_targets'],
                'missing_models': info['missing_models']
            })
    
    # Find model-specific failure patterns
    for model in data['model'].unique():
        model_missing = data[(data['model'] == model) & (data['n_valid'] == 0)]
        if len(model_missing) > 0:
            missing_horizons = sorted(model_missing['horizon'].unique().tolist())
            patterns['horizons_specific_model_fails'][model] = missing_horizons
    
    # Generate recommendations
    recommendations = []
    
    # Pattern 1: All models fail at same horizon (likely data/validation issue)
    if patterns['horizons_all_models_fail']:
        horizons_str = ', '.join(map(str, patterns['horizons_all_models_fail']))
        recommendations.append(
            f"All models fail at horizons {horizons_str} - likely data availability or validation threshold issue. "
            f"Check: (1) Test data availability for these horizons, (2) Validation threshold too strict, "
            f"(3) Index alignment issues in prediction generation"
        )
    
    # Pattern 2: Long horizon failures (likely numerical instability or data limit)
    long_failures = [h for h in patterns['long_horizon_failures'] if h['horizon'] >= 20]
    if long_failures:
        horizons_str = ', '.join([str(h['horizon']) for h in long_failures])
        recommendations.append(
            f"Failures at long horizons ({horizons_str}) - likely numerical instability or data limit. "
            f"Check: (1) Numerical stability at long horizons, (2) Test data extends to these horizons, "
            f"(3) Prediction generation handles long horizons correctly"
        )
    
    # Pattern 3: Model-specific failures
    for model, horizons in patterns['horizons_specific_model_fails'].items():
        if len(horizons) > 0:
            horizons_str = ', '.join(map(str, horizons))
            recommendations.append(
                f"{model} fails at horizons {horizons_str} - investigate {model}-specific prediction or validation issues"
            )
    
    # Pattern 4: Target-specific failures
    for tgt, info in missing_by_target.items():
        if info['missing_count'] > 0:
            missing_str = ', '.join(map(str, info['missing_horizons']))
            if info['completion_rate'] < 0.8:  # Less than 80% completion
                recommendations.append(
                    f"{tgt} has low completion rate ({info['completion_rate']:.1%}) - missing horizons: {missing_str}. "
                    f"Investigate target-specific data or validation issues"
                )
    
    if not recommendations:
        recommendations.append("No missing horizons detected - all predictions validated successfully")
    
    results = {
        'missing_by_target': missing_by_target,
        'missing_by_model': missing_by_model,
        'missing_by_horizon': missing_by_horizon,
        'patterns': patterns,
        'recommendations': recommendations,
        'summary': {
            'total_missing': len(missing),
            'total_valid': len(valid),
            'missing_rate': len(missing) / len(data) if len(data) > 0 else 0.0
        }
    }
    
    # Save if output path provided
    if output_path:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path_obj, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Missing horizons analysis saved to: {output_path}")
    
    # Log summary
    logger.info("Missing Horizons Analysis Summary:")
    logger.info(f"  Total missing: {len(missing)}, Total valid: {len(valid)}")
    if patterns['horizons_all_models_fail']:
        logger.warning(f"  ⚠️  All models fail at horizons: {patterns['horizons_all_models_fail']}")
    for model, horizons in patterns['horizons_specific_model_fails'].items():
        if horizons:
            logger.info(f"  {model} fails at: {horizons}")
    
    return results


def main_aggregator():
    """Main entry point for aggregator module."""
    logger = _module_logger
    logger.info("=" * 70)
    logger.info("Aggregating Experiment Results")
    logger.info("=" * 70)
    
    # Collect all results
    all_results = collect_all_comparison_results()
    
    if not all_results:
        logger.warning("No comparison results found in outputs/comparisons/")
        return
    
    logger.info(f"Found results for {len(all_results)} target series:")
    for target, results in all_results.items():
        logger.info(f"  - {target}: {len(results)} comparison(s)")
    
    # Aggregate performance
    aggregated = aggregate_overall_performance(all_results)
    
    if aggregated.empty:
        logger.warning("No metrics to aggregate.")
        return
    
    # Save aggregated results
    from src.utils import get_project_root
    outputs_dir = get_project_root() / "outputs"
    experiments_dir = outputs_dir / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = experiments_dir / "aggregated_results.csv"
    aggregated.to_csv(output_file, index=False)
    
    logger.info(f"Aggregated results saved to: {output_file}")
    logger.info(f"  Total rows: {len(aggregated)}")
    logger.info(f"  Columns: {', '.join(aggregated.columns)}")
    
    # Detect DDFM linearity (compare DDFM vs DDFM metrics)
    logger.info("")
    logger.info("Analyzing DDFM linearity (comparing DDFM vs DFM performance)...")
    linearity_results = detect_ddfm_linearity(
        aggregated,
        similarity_threshold=0.95,
        output_path=str(experiments_dir / "ddfm_linearity_analysis.json")
    )
    
    if 'error' not in linearity_results:
        logger.info("DDFM linearity analysis complete")
        # Log critical findings
        for target in linearity_results.get('summary', {}).get('high_linearity_targets', []):
            logger.warning(
                f"⚠️  {target['target']}: {target['status']} "
                f"(linearity={target['linearity']:.3f}, {target['linear_fraction']:.1%} horizons linear)"
            )
    
    # Analyze DDFM prediction quality
    logger.info("")
    logger.info("Analyzing DDFM prediction quality across horizons...")
    quality_results = analyze_ddfm_prediction_quality(
        aggregated,
        output_path=str(experiments_dir / "ddfm_prediction_quality.json")
    )
    
    if 'error' not in quality_results:
        logger.info("DDFM prediction quality analysis complete")
        # Log recommendations
        for target, recs in quality_results.get('recommendations', {}).items():
            if len(recs) > 0 and recs[0] != "Performance is acceptable":
                logger.info(f"  {target} recommendations: {', '.join(recs[:2])}")  # Show first 2 recommendations
        # Log linear collapse risk
        risk_summary = quality_results.get('linear_collapse_risk', {})
        for target, risk_info in risk_summary.items():
            if risk_info.get('risk_score', 0) > 0.4:
                logger.warning(
                    f"  ⚠️  {target}: Linear collapse risk = {risk_info['risk_score']:.2f} "
                    f"({risk_info['risk_level']})"
                )
    
    # Analyze missing horizons
    logger.info("")
    logger.info("Analyzing missing horizons (n_valid=0)...")
    missing_results = analyze_missing_horizons(
        aggregated,
        output_path=str(experiments_dir / "missing_horizons_analysis.json")
    )
    
    if 'error' not in missing_results:
        logger.info("Missing horizons analysis complete")
        # Log critical findings
        if missing_results.get('patterns', {}).get('horizons_all_models_fail'):
            horizons = missing_results['patterns']['horizons_all_models_fail']
            logger.warning(f"  ⚠️  All models fail at horizons: {horizons}")
        for rec in missing_results.get('recommendations', [])[:3]:  # Show first 3 recommendations
            if "No missing horizons" not in rec:
                logger.info(f"  Recommendation: {rec}")
