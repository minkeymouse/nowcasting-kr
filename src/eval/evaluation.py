"""Evaluation helper functions for forecasting experiments.

This module provides standardized metric calculation functions for evaluating
forecasting model performance, including standardized MSE, MAE, and RMSE.
"""

from typing import Union, Dict, Optional, Tuple, Any, List
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from sktime.performance_metrics.forecasting import (
        MeanSquaredError,
        MeanAbsoluteError,
    )
    HAS_SKTIME = True
except ImportError:
    HAS_SKTIME = False
    MeanSquaredError = None
    MeanAbsoluteError = None


def calculate_standardized_metrics(
    y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_pred: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_train: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
    target_series: Optional[Union[str, int]] = None
) -> Dict[str, float]:
    """Calculate standardized MSE, MAE, and RMSE metrics.
    
    Standardized metrics are normalized by the standard deviation of the
    training data to enable fair comparison across different series and scales.
    
    Parameters
    ----------
    y_true : pd.DataFrame, pd.Series, or np.ndarray
        Actual/true values (T × N) or (T,)
    y_pred : pd.DataFrame, pd.Series, or np.ndarray
        Predicted/forecasted values (T × N) or (T,), same shape as y_true
    y_train : pd.DataFrame, pd.Series, or np.ndarray, optional
        Training data used to calculate standardization factor (σ).
        If None, uses y_true to calculate σ (not recommended for out-of-sample evaluation)
    target_series : str or int, optional
        If multivariate, specify which series to evaluate (column name or index).
        If None, evaluates all series and returns average.
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'sMSE': Standardized Mean Squared Error (MSE / σ²)
        - 'sMAE': Standardized Mean Absolute Error (MAE / σ)
        - 'sRMSE': Standardized Root Mean Squared Error (RMSE / σ)
        - 'MSE': Raw Mean Squared Error
        - 'MAE': Raw Mean Absolute Error
        - 'RMSE': Raw Root Mean Squared Error
        - 'sigma': Standard deviation used for standardization
        
    Notes
    -----
    - Standardization uses training data standard deviation (σ) to normalize metrics
    - For multivariate data, metrics are calculated per series and averaged
    - Missing values (NaN) are automatically excluded from calculations
    - If all values are NaN, returns NaN for all metrics
    """
    # Convert to numpy arrays
    if isinstance(y_true, pd.DataFrame):
        y_true_arr = y_true.values
        columns = y_true.columns
    elif isinstance(y_true, pd.Series):
        y_true_arr = y_true.values.reshape(-1, 1)
        columns = [y_true.name] if y_true.name else [0]
    else:
        y_true_arr = np.asarray(y_true)
        if y_true_arr.ndim == 1:
            y_true_arr = y_true_arr.reshape(-1, 1)
        columns = list(range(y_true_arr.shape[1]))
    
    if isinstance(y_pred, pd.DataFrame):
        y_pred_arr = y_pred.values
    elif isinstance(y_pred, pd.Series):
        y_pred_arr = y_pred.values.reshape(-1, 1)
    else:
        y_pred_arr = np.asarray(y_pred)
        if y_pred_arr.ndim == 1:
            y_pred_arr = y_pred_arr.reshape(-1, 1)
    
    # Ensure same shape
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"y_true and y_pred must have same shape, "
            f"got {y_true_arr.shape} and {y_pred_arr.shape}"
        )
    
    # Select target series if specified
    if target_series is not None:
        if isinstance(target_series, str):
            if isinstance(y_true, pd.DataFrame):
                col_idx = y_true.columns.get_loc(target_series)
            else:
                raise ValueError("target_series must be int if y_true is not DataFrame")
        else:
            col_idx = target_series
        
        y_true_arr = y_true_arr[:, col_idx:col_idx+1]
        y_pred_arr = y_pred_arr[:, col_idx:col_idx+1]
        columns = [columns[col_idx]]
    
    # Calculate standardization factor from training data
    if y_train is not None:
        if isinstance(y_train, pd.DataFrame):
            train_arr = y_train.values
            if target_series is not None:
                if isinstance(target_series, str):
                    col_idx = y_train.columns.get_loc(target_series)
                else:
                    col_idx = target_series
                train_arr = train_arr[:, col_idx:col_idx+1]
        elif isinstance(y_train, pd.Series):
            train_arr = y_train.values.reshape(-1, 1)
        else:
            train_arr = np.asarray(y_train)
            if train_arr.ndim == 1:
                train_arr = train_arr.reshape(-1, 1)
    else:
        # Use y_true as fallback (not ideal for out-of-sample)
        train_arr = y_true_arr
    
    # Calculate standard deviation per series
    sigma = np.nanstd(train_arr, axis=0, ddof=1)
    # Avoid division by zero
    sigma = np.where(sigma == 0, 1.0, sigma)
    
    # Create mask for valid (non-NaN) values
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    
    # Calculate raw metrics per series
    n_series = y_true_arr.shape[1]
    mse_per_series = np.zeros(n_series)
    mae_per_series = np.zeros(n_series)
    rmse_per_series = np.zeros(n_series)
    
    for i in range(n_series):
        series_mask = mask[:, i]
        if np.sum(series_mask) > 0:
            y_true_series = y_true_arr[series_mask, i]
            y_pred_series = y_pred_arr[series_mask, i]
            
            # Calculate raw metrics
            mse_per_series[i] = np.mean((y_true_series - y_pred_series) ** 2)
            mae_per_series[i] = np.mean(np.abs(y_true_series - y_pred_series))
            rmse_per_series[i] = np.sqrt(mse_per_series[i])
        else:
            mse_per_series[i] = np.nan
            mae_per_series[i] = np.nan
            rmse_per_series[i] = np.nan
    
    # Average across series
    mse = np.nanmean(mse_per_series)
    mae = np.nanmean(mae_per_series)
    rmse = np.nanmean(rmse_per_series)
    sigma_mean = np.nanmean(sigma)
    
    # Calculate standardized metrics
    sMSE = mse / (sigma_mean ** 2) if sigma_mean > 0 else np.nan
    sMAE = mae / sigma_mean if sigma_mean > 0 else np.nan
    sRMSE = rmse / sigma_mean if sigma_mean > 0 else np.nan
    
    return {
        'sMSE': sMSE,
        'sMAE': sMAE,
        'sRMSE': sRMSE,
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'sigma': sigma_mean,
        'n_valid': int(np.sum(mask))
    }


def calculate_metrics_per_horizon(
    y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_pred: Union[pd.DataFrame, pd.Series, np.ndarray],
    horizons: Union[list, np.ndarray],
    y_train: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
    target_series: Optional[Union[str, int]] = None
) -> Dict[int, Dict[str, float]]:
    """Calculate standardized metrics for each forecast horizon.
    
    Parameters
    ----------
    y_true : pd.DataFrame, pd.Series, or np.ndarray
        Actual values (T × N) or (T,)
    y_pred : pd.DataFrame, pd.Series, or np.ndarray
        Predicted values (T × N) or (T,)
    horizons : list or np.ndarray
        List of forecast horizons (e.g., [1, 7, 28] for 1, 7, 28 days ahead)
    y_train : pd.DataFrame, pd.Series, or np.ndarray, optional
        Training data for standardization
    target_series : str or int, optional
        Target series to evaluate (for multivariate data)
        
    Returns
    -------
    dict
        Dictionary with horizon as key and metrics dict as value
        Example: {1: {'sMSE': 0.5, 'sMAE': 0.3, ...}, 7: {...}, ...}
    """
    horizons = np.asarray(horizons)
    results = {}
    
    for h in horizons:
        h = int(h)
        if h <= 0:
            continue
        
        # Extract predictions for horizon h (h-1 index since 0-indexed)
        if isinstance(y_pred, pd.DataFrame):
            y_pred_h = y_pred.iloc[h-1:h] if h <= len(y_pred) else pd.DataFrame()
        elif isinstance(y_pred, pd.Series):
            y_pred_h = y_pred.iloc[h-1:h] if h <= len(y_pred) else pd.Series()
        else:
            y_pred_h = y_pred[h-1:h] if h <= len(y_pred) else np.array([])
        
        if isinstance(y_true, pd.DataFrame):
            y_true_h = y_true.iloc[h-1:h] if h <= len(y_true) else pd.DataFrame()
        elif isinstance(y_true, pd.Series):
            y_true_h = y_true.iloc[h-1:h] if h <= len(y_true) else pd.Series()
        else:
            y_true_h = y_true[h-1:h] if h <= len(y_true) else np.array([])
        
        # Check if we have valid data
        has_pred = len(y_pred_h) > 0 if hasattr(y_pred_h, '__len__') else y_pred_h.size > 0
        has_true = len(y_true_h) > 0 if hasattr(y_true_h, '__len__') else y_true_h.size > 0
        
        # Calculate metrics for this horizon
        if has_pred and has_true:
            try:
                metrics = calculate_standardized_metrics(
                    y_true_h, y_pred_h, y_train=y_train, target_series=target_series
                )
                results[h] = metrics
            except Exception as e:
                # If calculation fails, return NaN metrics
                results[h] = {
                    'sMSE': np.nan, 'sMAE': np.nan, 'sRMSE': np.nan,
                    'MSE': np.nan, 'MAE': np.nan, 'RMSE': np.nan,
                    'sigma': np.nan, 'n_valid': 0
                }
        else:
            results[h] = {
                'sMSE': np.nan, 'sMAE': np.nan, 'sRMSE': np.nan,
                'MSE': np.nan, 'MAE': np.nan, 'RMSE': np.nan,
                'sigma': np.nan, 'n_valid': 0
            }
    
    return results


def evaluate_forecaster(
    forecaster,
    y_train: Union[pd.DataFrame, pd.Series],
    y_test: Union[pd.DataFrame, pd.Series],
    horizons: Union[list, np.ndarray],
    target_series: Optional[Union[str, int]] = None
) -> Dict[int, Dict[str, float]]:
    """Evaluate a sktime-compatible forecaster on test data.
    
    Parameters
    ----------
    forecaster : sktime BaseForecaster
        Fitted forecaster (must have fit() and predict() methods)
    y_train : pd.DataFrame or pd.Series
        Training data
    y_test : pd.DataFrame or pd.Series
        Test data
    horizons : list or np.ndarray
        Forecast horizons to evaluate
    target_series : str or int, optional
        Target series to evaluate
        
    Returns
    -------
    dict
        Dictionary with horizon as key and metrics dict as value
    """
    # Fit forecaster
    forecaster.fit(y_train)
    
    # Generate predictions for all horizons
    horizons_arr = np.asarray(horizons)
    max_horizon = int(np.max(horizons_arr))
    
    # Predict up to max horizon
    fh = np.arange(1, max_horizon + 1)
    y_pred = forecaster.predict(fh=fh)
    
    # Calculate metrics per horizon
    results = calculate_metrics_per_horizon(
        y_test, y_pred, horizons, y_train=y_train, target_series=target_series
    )
    
    return results


# ========================================================================
# Model Comparison Functions
# ========================================================================

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
        List of forecast horizons to compare (e.g., [1, 7, 28])
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


def save_comparison_plots(
    comparison_results: Dict[str, Any],
    output_dir: str,
    target_series: str
) -> List[Path]:
    """Save comparison plots for model results.
    
    This function generates and saves visualization plots comparing
    multiple models' performance across different horizons.
    
    Parameters
    ----------
    comparison_results : dict
        Results from compare_multiple_models() function
    output_dir : str
        Output directory for saving plots
    target_series : str
        Target series name (for plot titles)
        
    Returns
    -------
    List[Path]
        List of paths to saved plot files
        
    Note
    ----
    This function requires matplotlib. Plots are saved to nowcasting-report/images/
    if output_dir is set appropriately, otherwise to the specified output_dir.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        print("Warning: matplotlib not available, skipping plot generation")
        return []
    
    metrics_table = comparison_results.get('metrics_table')
    if metrics_table is None or len(metrics_table) == 0:
        print("Warning: No metrics table available for plotting")
        return []
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_plots = []
    
    # Plot 1: Log-likelihood comparison
    if 'loglik' in metrics_table.columns and not metrics_table['loglik'].isna().all():
        fig, ax = plt.subplots(figsize=(10, 6))
        models = metrics_table['model']
        logliks = metrics_table['loglik']
        
        ax.bar(models, logliks)
        ax.set_xlabel('Model')
        ax.set_ylabel('Log-Likelihood')
        ax.set_title(f'Model Comparison: Log-Likelihood ({target_series})')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        
        plot_path = output_path / f"comparison_loglik_{target_series}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(plot_path)
    
    # Plot 2: Convergence status
    if 'converged' in metrics_table.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        models = metrics_table['model']
        converged = metrics_table['converged'].astype(int)
        
        ax.bar(models, converged, color=['red' if not c else 'green' for c in metrics_table['converged']])
        ax.set_xlabel('Model')
        ax.set_ylabel('Converged (1=Yes, 0=No)')
        ax.set_title(f'Model Comparison: Convergence Status ({target_series})')
        ax.set_ylim([-0.1, 1.1])
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        
        plot_path = output_path / f"comparison_convergence_{target_series}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(plot_path)
    
    # Also save to nowcasting-report/images/ if possible
    try:
        report_images_dir = Path(__file__).parent.parent / "nowcasting-report" / "images"
        if report_images_dir.exists():
            for plot_path in saved_plots:
                import shutil
                shutil.copy(plot_path, report_images_dir / plot_path.name)
    except Exception as e:
        print(f"Warning: Could not copy plots to nowcasting-report/images/: {e}")
    
    return saved_plots
