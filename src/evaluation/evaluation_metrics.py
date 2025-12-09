"""Metric calculation functions for evaluation."""

from typing import Union, Dict, Optional, Any, List, Tuple
import numpy as np
import pandas as pd
from .evaluation import _module_logger, EXTREME_VALUE_THRESHOLD

try:
    from scipy import stats
except ImportError:
    stats = None


def calculate_standardized_metrics(
    y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_pred: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_train: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
    target_series: Optional[Union[str, int]] = None
) -> Dict[str, float]:
    """Calculate standardized MSE, MAE, and RMSE metrics with additional diagnostic metrics.
    
    This function calculates standard forecasting metrics (sMSE, sMAE, sRMSE) and
    additional diagnostic metrics useful for DDFM analysis:
    - prediction_bias: Mean prediction error (positive = overprediction, negative = underprediction)
    - directional_accuracy: Fraction of predictions with correct direction (0-1)
    - theil_u: Theil's U statistic (U < 1 = better than naive forecast)
    - mape: Mean absolute percentage error (%)
    - error_skewness: Error distribution skewness (positive = right-skewed, negative = left-skewed)
    - error_kurtosis: Error distribution kurtosis (positive = heavy-tailed, negative = light-tailed)
    - error_bias_squared: Systematic bias component (Bias^2 in bias-variance decomposition)
    - error_variance: Variance component in bias-variance decomposition
    - error_concentration: Error concentration metric (0 = uniform errors, 1 = highly concentrated)
    
    The standardized metrics (sMSE, sMAE, sRMSE) are normalized by the training data
    standard deviation, making them comparable across different series scales.
    
    The enhanced error distribution metrics help identify:
    - Systematic error patterns (skewness, kurtosis)
    - Error sources (bias vs variance decomposition)
    - Error concentration (whether errors are uniform or concentrated at specific points)
    
    Parameters
    ----------
    y_true : pd.DataFrame, pd.Series, or np.ndarray
        True/actual values
    y_pred : pd.DataFrame, pd.Series, or np.ndarray
        Predicted/forecasted values
    y_train : pd.DataFrame, pd.Series, or np.ndarray, optional
        Training data used for standardization (if None, uses y_true)
    target_series : str or int, optional
        Target series name or index (for multivariate data)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'sMSE', 'sMAE', 'sRMSE': Standardized metrics (normalized by training std)
        - 'MSE', 'MAE', 'RMSE': Raw metrics (not normalized)
        - 'sigma': Training data standard deviation used for normalization
        - 'n_valid': Number of valid (finite) observations
        - 'prediction_bias': Mean prediction error (optional, None if cannot calculate)
        - 'directional_accuracy': Fraction with correct direction (optional, None if cannot calculate)
        - 'theil_u': Theil's U statistic (optional, None if cannot calculate)
        - 'mape': Mean absolute percentage error % (optional, None if cannot calculate)
        
    Notes
    -----
    - Handles CUDA tensors by converting to CPU before numpy conversion
    - Filters extreme values (> 1e10) that indicate numerical instability
    - Validates for suspiciously good results (< 1e-4) that may indicate data leakage
    - Additional diagnostic metrics help identify specific DDFM issues:
      * prediction_bias: Systematic over/underprediction
      * directional_accuracy: Whether model captures trend direction
      * theil_u: Relative performance vs naive forecast
      * mape: Percentage-based error metric
    """
    logger = _module_logger
    
    is_y_true_series = isinstance(y_true, pd.Series)
    is_y_true_dataframe = isinstance(y_true, pd.DataFrame)
    
    if is_y_true_dataframe:
        y_true_arr = y_true.values
        columns = y_true.columns
    elif is_y_true_series:
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
        # Convert torch tensors to numpy array (handle CUDA tensors)
        try:
            import torch
            if isinstance(y_pred, torch.Tensor):
                y_pred_arr = y_pred.cpu().numpy()
            else:
                y_pred_arr = np.asarray(y_pred)
        except ImportError:
            y_pred_arr = np.asarray(y_pred)
        if y_pred_arr.ndim == 1:
            y_pred_arr = y_pred_arr.reshape(-1, 1)
    
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(f"y_true and y_pred must have same shape, got {y_true_arr.shape} and {y_pred_arr.shape}")
    
    if target_series is not None:
        if isinstance(target_series, str):
            if is_y_true_dataframe:
                try:
                    col_idx = y_true.columns.get_loc(target_series)
                except KeyError:
                    if len(y_true.columns) == 1:
                        col_idx = 0
                        columns = [y_true.columns[0]]
                    else:
                        if isinstance(y_pred, pd.DataFrame) and target_series in y_pred.columns:
                            pred_col_idx = y_pred.columns.get_loc(target_series)
                            if pred_col_idx < len(y_true.columns):
                                col_idx = pred_col_idx
                                logger.warning(f"target_series '{target_series}' not found in y_true.columns, using column index {col_idx} from y_pred")
                            else:
                                col_idx = 0
                                logger.warning(f"target_series '{target_series}' not found in y_true.columns={list(y_true.columns)}, using first column as fallback")
                        else:
                            col_idx = 0
                            logger.warning(f"target_series '{target_series}' not found in y_true.columns={list(y_true.columns)}, using first column as fallback")
            elif is_y_true_series:
                col_idx = 0
            else:
                raise ValueError("target_series must be int if y_true is not DataFrame or Series")
        else:
            col_idx = target_series
        
        y_true_arr = y_true_arr[:, col_idx:col_idx+1]
        y_pred_arr = y_pred_arr[:, col_idx:col_idx+1]
        columns = [columns[col_idx]]
    
    if y_train is not None:
        if isinstance(y_train, pd.DataFrame):
            train_arr = y_train.values
            if target_series is not None:
                if isinstance(target_series, str):
                    try:
                        col_idx = y_train.columns.get_loc(target_series)
                    except KeyError:
                        if is_y_true_dataframe:
                            try:
                                col_idx = y_true.columns.get_loc(target_series)
                            except KeyError:
                                col_idx = 0
                                logger.warning(f"target_series '{target_series}' not found in y_train.columns={list(y_train.columns)} or y_true.columns={list(y_true.columns)}, using column 0")
                        else:
                            col_idx = 0
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
        train_arr = y_true_arr
    
    # Calculate sigma (standard deviation) for standardization
    # Use ddof=1 for sample standard deviation (unbiased estimator)
    sigma = np.nanstd(train_arr, axis=0, ddof=1)
    # Handle zero standard deviation (constant series) by setting to 1.0
    # This prevents division by zero in standardization
    sigma = np.where(sigma == 0, 1.0, sigma)
    
    # Create mask for valid (finite) values
    # Only calculate metrics for finite values to avoid NaN/Inf contamination
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    
    # Additional robustness: Check for constant predictions (all same value)
    # This can indicate model collapse or numerical issues
    if np.sum(mask) > 0:
        pred_values = y_pred_arr[mask]
        if len(pred_values) > 1:
            pred_std = np.nanstd(pred_values)
            if pred_std < 1e-10:
                logger.debug("calculate_standardized_metrics: Predictions are nearly constant (std < 1e-10), may indicate model collapse")
    
    logger.debug(f"calculate_standardized_metrics: y_true_arr shape={y_true_arr.shape}, y_pred_arr shape={y_pred_arr.shape}")
    logger.debug(f"calculate_standardized_metrics: mask sum={np.sum(mask)}, mask shape={mask.shape}, n_valid={int(np.sum(mask))}")
    if np.sum(mask) == 0:
        logger.warning(f"calculate_standardized_metrics: mask is all False! y_true_arr finite={np.sum(np.isfinite(y_true_arr))}, y_pred_arr finite={np.sum(np.isfinite(y_pred_arr))}")
        logger.warning(f"calculate_standardized_metrics: y_true_arr has NaN={np.sum(np.isnan(y_true_arr))}, y_pred_arr has NaN={np.sum(np.isnan(y_pred_arr))}")
        logger.warning(f"calculate_standardized_metrics: y_true_arr has Inf={np.sum(np.isinf(y_true_arr))}, y_pred_arr has Inf={np.sum(np.isinf(y_pred_arr))}")
    
    n_series = y_true_arr.shape[1]
    mse_per_series = np.zeros(n_series)
    mae_per_series = np.zeros(n_series)
    rmse_per_series = np.zeros(n_series)
    
    for i in range(n_series):
        series_mask = mask[:, i]
        if np.sum(series_mask) > 0:
            y_true_series = y_true_arr[series_mask, i]
            y_pred_series = y_pred_arr[series_mask, i]
            mse_per_series[i] = np.mean((y_true_series - y_pred_series) ** 2)
            mae_per_series[i] = np.mean(np.abs(y_true_series - y_pred_series))
            rmse_per_series[i] = np.sqrt(mse_per_series[i])
        else:
            mse_per_series[i] = np.nan
            mae_per_series[i] = np.nan
            rmse_per_series[i] = np.nan
    
    mse = np.nanmean(mse_per_series)
    mae = np.nanmean(mae_per_series)
    rmse = np.nanmean(rmse_per_series)
    sigma_mean = np.nanmean(sigma)
    
    sMSE = mse / (sigma_mean ** 2) if sigma_mean > 0 else np.nan
    sMAE = mae / sigma_mean if sigma_mean > 0 else np.nan
    sRMSE = rmse / sigma_mean if sigma_mean > 0 else np.nan
    
    if isinstance(sMSE, (int, float)) and (abs(sMSE) > EXTREME_VALUE_THRESHOLD or np.isinf(sMSE)):
        logger.warning(f"calculate_standardized_metrics: Extreme sMSE detected: {sMSE}. This indicates numerical instability.")
        sMSE = np.nan
    if isinstance(sMAE, (int, float)) and (abs(sMAE) > EXTREME_VALUE_THRESHOLD or np.isinf(sMAE)):
        logger.warning(f"calculate_standardized_metrics: Extreme sMAE detected: {sMAE}. This indicates numerical instability.")
        sMAE = np.nan
    if isinstance(sRMSE, (int, float)) and (abs(sRMSE) > EXTREME_VALUE_THRESHOLD or np.isinf(sRMSE)):
        logger.warning(f"calculate_standardized_metrics: Extreme sRMSE detected: {sRMSE}. This indicates numerical instability.")
        sRMSE = np.nan
    
    SUSPICIOUSLY_GOOD_THRESHOLD = 1e-4
    if isinstance(sRMSE, (int, float)) and (0 < sRMSE < SUSPICIOUSLY_GOOD_THRESHOLD):
        logger.warning(f"calculate_standardized_metrics: Suspiciously good sRMSE detected: {sRMSE}. This may indicate data leakage, numerical precision issues, persistence predictions (VAR), or overfitting. Verify train-test split and model implementation.")
    
    # Improved numerical stability: Use higher precision for small values
    # Round very small values to prevent numerical precision issues
    EPSILON = 1e-10
    if isinstance(sMSE, (int, float)) and abs(sMSE) < EPSILON:
        sMSE = 0.0
    if isinstance(sMAE, (int, float)) and abs(sMAE) < EPSILON:
        sMAE = 0.0
    if isinstance(sRMSE, (int, float)) and abs(sRMSE) < EPSILON:
        sRMSE = 0.0
    
    # Additional validation: Check for division by zero in standardization
    if sigma_mean == 0:
        logger.warning(f"calculate_standardized_metrics: sigma_mean is zero, standardized metrics may be invalid. Using raw metrics instead.")
    
    # Additional diagnostic metrics for DDFM analysis
    # These help identify specific issues with DDFM predictions
    
    # Prediction bias: mean prediction error (positive = overprediction, negative = underprediction)
    prediction_bias = None
    # Directional accuracy: fraction of predictions with correct direction (for time series with trends)
    directional_accuracy = None
    # Theil's U statistic: relative forecast accuracy (compares to naive forecast)
    theil_u = None
    # Mean absolute percentage error (MAPE): percentage-based error metric
    mape = None
    # Error distribution metrics: skewness and kurtosis (identify systematic error patterns)
    error_skewness = None
    error_kurtosis = None
    # Bias-variance decomposition: separate systematic bias from variance
    error_bias_squared = None
    error_variance = None
    # Error concentration: measure of how concentrated errors are (lower = more uniform, higher = more concentrated)
    error_concentration = None
    
    if np.sum(mask) > 0:
        # Calculate prediction bias (mean error)
        errors = y_true_arr[mask] - y_pred_arr[mask]
        prediction_bias = float(np.nanmean(errors))
        
        # Calculate directional accuracy (for time series with trends)
        # Directional accuracy: fraction of predictions where direction matches actual
        # Only meaningful if we have multiple time points or can compare with previous values
        if len(errors) > 1 and y_train is not None:
            try:
                # Compare direction of change: actual vs predicted
                # For single-point evaluation, compare with last training value
                if isinstance(y_train, (pd.DataFrame, pd.Series)):
                    last_train = y_train.iloc[-1] if hasattr(y_train, 'iloc') else y_train[-1]
                    if isinstance(last_train, pd.Series) and target_series is not None:
                        if isinstance(target_series, str) and target_series in last_train.index:
                            last_train_val = last_train[target_series]
                        else:
                            last_train_val = last_train.iloc[0] if len(last_train) > 0 else None
                    else:
                        last_train_val = last_train.iloc[0] if hasattr(last_train, 'iloc') and len(last_train) > 0 else (last_train if not isinstance(last_train, pd.Series) else None)
                    
                    if last_train_val is not None and not np.isnan(last_train_val):
                        # For each prediction, check if direction matches actual
                        y_true_vals = y_true_arr[mask]
                        y_pred_vals = y_pred_arr[mask]
                        if len(y_true_vals) > 0 and len(y_pred_vals) > 0:
                            actual_direction = np.sign(y_true_vals - last_train_val)
                            pred_direction = np.sign(y_pred_vals - last_train_val)
                            # Directional accuracy: fraction where directions match
                            matches = (actual_direction == pred_direction) | (actual_direction == 0) | (pred_direction == 0)
                            directional_accuracy = float(np.nanmean(matches)) if len(matches) > 0 else None
            except Exception as e:
                logger.debug(f"Could not calculate directional accuracy: {e}")
        
        # Calculate Theil's U statistic (relative forecast accuracy)
        # Theil's U = RMSE(predicted) / RMSE(naive_forecast)
        # U < 1: better than naive, U > 1: worse than naive, U = 1: same as naive
        try:
            if y_train is not None:
                # Naive forecast: last value (persistence forecast)
                if isinstance(y_train, (pd.DataFrame, pd.Series)):
                    naive_forecast = y_train.iloc[-1] if hasattr(y_train, 'iloc') else y_train[-1]
                    if isinstance(naive_forecast, pd.Series) and target_series is not None:
                        if isinstance(target_series, str) and target_series in naive_forecast.index:
                            naive_val = naive_forecast[target_series]
                        else:
                            naive_val = naive_forecast.iloc[0] if len(naive_forecast) > 0 else None
                    else:
                        naive_val = naive_forecast.iloc[0] if hasattr(naive_forecast, 'iloc') and len(naive_forecast) > 0 else (naive_forecast if not isinstance(naive_forecast, pd.Series) else None)
                    
                    if naive_val is not None and not np.isnan(naive_val):
                        # Naive forecast RMSE
                        naive_errors = y_true_arr[mask] - naive_val
                        naive_rmse = np.sqrt(np.nanmean(naive_errors ** 2))
                        # Theil's U: predicted RMSE / naive RMSE
                        if naive_rmse > 0:
                            theil_u = float(rmse / naive_rmse) if rmse is not None and not np.isnan(rmse) else None
        except Exception as e:
            logger.debug(f"Could not calculate Theil's U: {e}")
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        # MAPE = mean(|actual - predicted| / |actual|) * 100
        # Only meaningful if actual values are non-zero
        try:
            y_true_vals = y_true_arr[mask]
            y_pred_vals = y_pred_arr[mask]
            # Avoid division by zero: only calculate MAPE for non-zero actual values
            non_zero_mask = np.abs(y_true_vals) > 1e-10
            if np.sum(non_zero_mask) > 0:
                pct_errors = np.abs((y_true_vals[non_zero_mask] - y_pred_vals[non_zero_mask]) / y_true_vals[non_zero_mask])
                mape = float(np.nanmean(pct_errors) * 100)  # Convert to percentage
        except Exception as e:
            logger.debug(f"Could not calculate MAPE: {e}")
        
        # Calculate error distribution metrics (skewness and kurtosis)
        # These help identify systematic error patterns in DDFM predictions
        try:
            errors = y_true_arr[mask] - y_pred_arr[mask]
            if len(errors) > 2:  # Need at least 3 points for meaningful skewness/kurtosis
                try:
                    from scipy import stats
                    error_skewness = float(stats.skew(errors))
                    error_kurtosis = float(stats.kurtosis(errors))
                except ImportError:
                    # Fallback: manual calculation if scipy not available
                    mean_err = np.nanmean(errors)
                    std_err = np.nanstd(errors)
                    if std_err > 1e-10:
                        # Skewness: E[(X - mean)^3] / std^3
                        skew = np.nanmean(((errors - mean_err) / std_err) ** 3)
                        # Kurtosis: E[(X - mean)^4] / std^4 - 3 (excess kurtosis)
                        kurt = np.nanmean(((errors - mean_err) / std_err) ** 4) - 3.0
                        error_skewness = float(skew)
                        error_kurtosis = float(kurt)
                    else:
                        error_skewness = None
                        error_kurtosis = None
        except Exception as e:
            logger.debug(f"Could not calculate error distribution metrics: {e}")
        
        # Bias-variance decomposition of errors
        # MSE = Bias^2 + Variance + Irreducible Error
        # This helps understand if errors come from systematic bias or high variance
        try:
            errors = y_true_arr[mask] - y_pred_arr[mask]
            if len(errors) > 0:
                # Bias squared: (mean error)^2 (systematic bias component)
                mean_error = np.nanmean(errors)
                error_bias_squared = float(mean_error ** 2)
                # Variance: variance of errors (variance component)
                error_variance = float(np.nanvar(errors))
        except Exception as e:
            logger.debug(f"Could not calculate bias-variance decomposition: {e}")
        
        # Error concentration: measure of how concentrated errors are
        # Uses Gini coefficient-like measure: 0 = uniform errors, 1 = all errors in one point
        # Higher concentration suggests systematic issues (e.g., all errors at specific horizons)
        try:
            errors = np.abs(y_true_arr[mask] - y_pred_arr[mask])
            if len(errors) > 1:
                # Sort errors in ascending order
                sorted_errors = np.sort(errors)
                # Calculate cumulative sum
                cumsum = np.cumsum(sorted_errors)
                # Normalize
                if cumsum[-1] > 0:
                    cumsum_norm = cumsum / cumsum[-1]
                    # Gini-like coefficient: area under curve
                    n = len(sorted_errors)
                    # Perfect equality (uniform): area = 0.5
                    # Perfect inequality (all in one): area = 1.0
                    # Concentration = 2 * (area - 0.5)
                    area = np.trapz(cumsum_norm) / n
                    error_concentration = float(2 * (area - 0.5))  # 0 = uniform, 1 = concentrated
                    error_concentration = np.clip(error_concentration, 0.0, 1.0)
                else:
                    error_concentration = 0.0
        except Exception as e:
            logger.debug(f"Could not calculate error concentration: {e}")
    
    # Convert to float or None for JSON serialization
    prediction_bias = float(prediction_bias) if prediction_bias is not None and not np.isnan(prediction_bias) else None
    directional_accuracy = float(directional_accuracy) if directional_accuracy is not None and not np.isnan(directional_accuracy) else None
    theil_u = float(theil_u) if theil_u is not None and not np.isnan(theil_u) else None
    mape = float(mape) if mape is not None and not np.isnan(mape) else None
    error_skewness = float(error_skewness) if error_skewness is not None and not np.isnan(error_skewness) else None
    error_kurtosis = float(error_kurtosis) if error_kurtosis is not None and not np.isnan(error_kurtosis) else None
    error_bias_squared = float(error_bias_squared) if error_bias_squared is not None and not np.isnan(error_bias_squared) else None
    error_variance = float(error_variance) if error_variance is not None and not np.isnan(error_variance) else None
    error_concentration = float(error_concentration) if error_concentration is not None and not np.isnan(error_concentration) else None
    
    return {
        'sMSE': sMSE, 'sMAE': sMAE, 'sRMSE': sRMSE,
        'MSE': mse, 'MAE': mae, 'RMSE': rmse,
        'sigma': sigma_mean, 'n_valid': int(np.sum(mask)),
        # Additional diagnostic metrics for DDFM analysis
        'prediction_bias': prediction_bias,  # Mean prediction error (positive = overprediction)
        'directional_accuracy': directional_accuracy,  # Fraction with correct direction (0-1)
        'theil_u': theil_u,  # Theil's U statistic (U < 1 = better than naive)
        'mape': mape,  # Mean absolute percentage error (%)
        # Enhanced error distribution metrics for DDFM analysis
        'error_skewness': error_skewness,  # Error distribution skewness (positive = right-skewed, negative = left-skewed)
        'error_kurtosis': error_kurtosis,  # Error distribution kurtosis (positive = heavy-tailed, negative = light-tailed)
        'error_bias_squared': error_bias_squared,  # Systematic bias component (Bias^2 in bias-variance decomposition)
        'error_variance': error_variance,  # Variance component in bias-variance decomposition
        'error_concentration': error_concentration  # Error concentration (0 = uniform, 1 = highly concentrated)
    }


def calculate_horizon_weighted_metrics(
    aggregated_results: pd.DataFrame,
    target: Optional[str] = None,
    model: Optional[str] = None,
    short_term_weight: float = 2.0,
    mid_term_weight: float = 1.0,
    long_term_weight: float = 0.5
) -> Dict[str, float]:
    """Calculate horizon-weighted metrics for DDFM evaluation.
    
    This function calculates weighted averages of metrics across forecast horizons,
    giving more weight to short-term horizons (which are typically more important
    for practical forecasting) and less weight to long-term horizons.
    
    Horizon classification:
    - Short-term: horizons 1-6 (higher weight)
    - Mid-term: horizons 7-12 (standard weight)
    - Long-term: horizons 13-22 (lower weight)
    
    Parameters
    ----------
    aggregated_results : pd.DataFrame
        Aggregated results DataFrame with columns: target, model, horizon, sMSE, sMAE, sRMSE
    target : str, optional
        Target series to filter (if None, aggregates across all targets)
    model : str, optional
        Model to filter (if None, aggregates across all models)
    short_term_weight : float, default 2.0
        Weight for short-term horizons (1-6)
    mid_term_weight : float, default 1.0
        Weight for mid-term horizons (7-12)
    long_term_weight : float, default 0.5
        Weight for long-term horizons (13-22)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'weighted_sMAE': Horizon-weighted average sMAE
        - 'weighted_sMSE': Horizon-weighted average sMSE
        - 'weighted_sRMSE': Horizon-weighted average sRMSE
        - 'short_term_avg_sMAE': Average sMAE for horizons 1-6
        - 'mid_term_avg_sMAE': Average sMAE for horizons 7-12
        - 'long_term_avg_sMAE': Average sMAE for horizons 13-22
        - 'n_short_term': Number of valid short-term horizons
        - 'n_mid_term': Number of valid mid-term horizons
        - 'n_long_term': Number of valid long-term horizons
    """
    logger = _module_logger
    
    # Filter results
    filtered = aggregated_results.copy()
    if target:
        filtered = filtered[filtered['target'] == target]
    if model:
        filtered = filtered[filtered['model'] == model]
    
    if filtered.empty:
        logger.warning(f"No results found for target={target}, model={model}")
        return {}
    
    # Classify horizons
    short_term = filtered[(filtered['horizon'] >= 1) & (filtered['horizon'] <= 6)]
    mid_term = filtered[(filtered['horizon'] >= 7) & (filtered['horizon'] <= 12)]
    long_term = filtered[(filtered['horizon'] >= 13) & (filtered['horizon'] <= 22)]
    
    # Calculate weighted averages
    weighted_smae = 0.0
    weighted_smse = 0.0
    weighted_srmse = 0.0
    total_weight = 0.0
    
    short_term_smae = []
    mid_term_smae = []
    long_term_smae = []
    
    # Short-term horizons
    for _, row in short_term.iterrows():
        if pd.notna(row.get('sMAE')) and row.get('n_valid', 0) > 0:
            weight = short_term_weight
            weighted_smae += row['sMAE'] * weight
            weighted_smse += row.get('sMSE', 0) * weight if pd.notna(row.get('sMSE')) else 0
            weighted_srmse += row.get('sRMSE', 0) * weight if pd.notna(row.get('sRMSE')) else 0
            total_weight += weight
            short_term_smae.append(row['sMAE'])
    
    # Mid-term horizons
    for _, row in mid_term.iterrows():
        if pd.notna(row.get('sMAE')) and row.get('n_valid', 0) > 0:
            weight = mid_term_weight
            weighted_smae += row['sMAE'] * weight
            weighted_smse += row.get('sMSE', 0) * weight if pd.notna(row.get('sMSE')) else 0
            weighted_srmse += row.get('sRMSE', 0) * weight if pd.notna(row.get('sRMSE')) else 0
            total_weight += weight
            mid_term_smae.append(row['sMAE'])
    
    # Long-term horizons
    for _, row in long_term.iterrows():
        if pd.notna(row.get('sMAE')) and row.get('n_valid', 0) > 0:
            weight = long_term_weight
            weighted_smae += row['sMAE'] * weight
            weighted_smse += row.get('sMSE', 0) * weight if pd.notna(row.get('sMSE')) else 0
            weighted_srmse += row.get('sRMSE', 0) * weight if pd.notna(row.get('sRMSE')) else 0
            total_weight += weight
            long_term_smae.append(row['sMAE'])
    
    # Normalize by total weight
    if total_weight > 0:
        weighted_smae /= total_weight
        weighted_smse /= total_weight
        weighted_srmse /= total_weight
    else:
        weighted_smae = None
        weighted_smse = None
        weighted_srmse = None
    
    return {
        'weighted_sMAE': float(weighted_smae) if weighted_smae is not None else None,
        'weighted_sMSE': float(weighted_smse) if weighted_smse is not None else None,
        'weighted_sRMSE': float(weighted_srmse) if weighted_srmse is not None else None,
        'short_term_avg_sMAE': float(np.mean(short_term_smae)) if short_term_smae else None,
        'mid_term_avg_sMAE': float(np.mean(mid_term_smae)) if mid_term_smae else None,
        'long_term_avg_sMAE': float(np.mean(long_term_smae)) if long_term_smae else None,
        'n_short_term': len(short_term_smae),
        'n_mid_term': len(mid_term_smae),
        'n_long_term': len(long_term_smae)
    }


def calculate_training_aligned_metrics(
    y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_pred: Union[pd.DataFrame, pd.Series, np.ndarray],
    loss_function: str = 'mse',
    huber_delta: float = 1.0,
    y_train: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None
) -> Dict[str, float]:
    """Calculate metrics aligned with training loss function.
    
    This function calculates evaluation metrics that match what the model was
    optimized for during training. If the model was trained with Huber loss,
    this function calculates Huber-based metrics; if trained with MSE, it
    calculates MSE-based metrics.
    
    Parameters
    ----------
    y_true : pd.DataFrame, pd.Series, or np.ndarray
        True/actual values
    y_pred : pd.DataFrame, pd.Series, or np.ndarray
        Predicted/forecasted values
    loss_function : str, default 'mse'
        Loss function used during training ('mse' or 'huber')
    huber_delta : float, default 1.0
        Delta parameter for Huber loss (only used if loss_function='huber')
    y_train : pd.DataFrame, pd.Series, or np.ndarray, optional
        Training data used for standardization (if None, uses y_true)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'training_loss': Training-aligned loss value
        - 'training_loss_std': Standardized training loss (normalized by training std)
        - 'mse_loss': MSE loss (for comparison)
        - 'huber_loss': Huber loss (if loss_function='huber')
    """
    logger = _module_logger
    
    # Convert to numpy arrays
    if isinstance(y_true, pd.DataFrame):
        y_true_arr = y_true.values
    elif isinstance(y_true, pd.Series):
        y_true_arr = y_true.values.reshape(-1, 1)
    else:
        try:
            import torch
            if isinstance(y_true, torch.Tensor):
                y_true_arr = y_true.cpu().numpy()
            else:
                y_true_arr = np.asarray(y_true)
        except ImportError:
            y_true_arr = np.asarray(y_true)
    
    if isinstance(y_pred, pd.DataFrame):
        y_pred_arr = y_pred.values
    elif isinstance(y_pred, pd.Series):
        y_pred_arr = y_pred.values.reshape(-1, 1)
    else:
        try:
            import torch
            if isinstance(y_pred, torch.Tensor):
                y_pred_arr = y_pred.cpu().numpy()
            else:
                y_pred_arr = np.asarray(y_pred)
        except ImportError:
            y_pred_arr = np.asarray(y_pred)
    
    # Ensure same shape
    if y_true_arr.ndim == 1:
        y_true_arr = y_true_arr.reshape(-1, 1)
    if y_pred_arr.ndim == 1:
        y_pred_arr = y_pred_arr.reshape(-1, 1)
    
    # Flatten for calculation
    y_true_flat = y_true_arr.flatten()
    y_pred_flat = y_pred_arr.flatten()
    
    # Filter valid values
    mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
    if np.sum(mask) == 0:
        logger.warning("No valid values for training-aligned metrics")
        return {
            'training_loss': None,
            'training_loss_std': None,
            'mse_loss': None,
            'huber_loss': None
        }
    
    y_true_valid = y_true_flat[mask]
    y_pred_valid = y_pred_flat[mask]
    
    # Calculate MSE loss (always calculated for comparison)
    mse_loss = float(np.mean((y_true_valid - y_pred_valid) ** 2))
    
    # Calculate training-aligned loss
    if loss_function == 'huber':
        diff = y_true_valid - y_pred_valid
        abs_diff = np.abs(diff)
        huber_loss = np.where(
            abs_diff <= huber_delta,
            0.5 * diff ** 2,
            huber_delta * (abs_diff - 0.5 * huber_delta)
        )
        training_loss = float(np.mean(huber_loss))
    else:
        # MSE loss
        training_loss = mse_loss
        huber_loss = None
    
    # Standardize by training data std
    if y_train is not None:
        if isinstance(y_train, pd.DataFrame):
            y_train_arr = y_train.values
        elif isinstance(y_train, pd.Series):
            y_train_arr = y_train.values.reshape(-1, 1)
        else:
            y_train_arr = np.asarray(y_train)
        
        if y_train_arr.ndim == 1:
            y_train_arr = y_train_arr.reshape(-1, 1)
        
        sigma = float(np.std(y_train_arr))
        if sigma > 1e-10:
            training_loss_std = training_loss / (sigma ** 2)
        else:
            training_loss_std = None
    else:
        sigma = float(np.std(y_true_valid))
        if sigma > 1e-10:
            training_loss_std = training_loss / (sigma ** 2)
        else:
            training_loss_std = None
    
    return {
        'training_loss': training_loss,
        'training_loss_std': training_loss_std,
        'mse_loss': mse_loss,
        'huber_loss': huber_loss
    }


def calculate_metrics_per_horizon(
    y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_pred: Union[pd.DataFrame, pd.Series, np.ndarray],
    horizons: Union[list, np.ndarray],
    y_train: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
    target_series: Optional[Union[str, int]] = None
) -> Dict[int, Dict[str, float]]:
    """Calculate standardized metrics for each forecast horizon."""
    logger = _module_logger
    horizons = np.asarray(horizons)
    results = {}
    
    for h in horizons:
        h = int(h)
        if h <= 0:
            continue
        
        if isinstance(y_pred, (pd.DataFrame, pd.Series)) and isinstance(y_true, (pd.DataFrame, pd.Series)):
            pred_idx = h - 1
            if pred_idx < len(y_pred):
                pred_time_idx = y_pred.index[pred_idx] if hasattr(y_pred.index, '__getitem__') else None
                if pred_time_idx is not None and pred_time_idx in y_true.index:
                    y_pred_h = y_pred.loc[[pred_time_idx]]
                    y_true_h = y_true.loc[[pred_time_idx]]
                else:
                    y_pred_h = y_pred.iloc[pred_idx:pred_idx+1] if pred_idx < len(y_pred) else (pd.DataFrame() if isinstance(y_pred, pd.DataFrame) else pd.Series())
                    y_true_h = y_true.iloc[pred_idx:pred_idx+1] if pred_idx < len(y_true) else (pd.DataFrame() if isinstance(y_true, pd.DataFrame) else pd.Series())
            else:
                y_pred_h = pd.DataFrame() if isinstance(y_pred, pd.DataFrame) else pd.Series()
                y_true_h = pd.DataFrame() if isinstance(y_true, pd.DataFrame) else pd.Series()
        else:
            pred_idx = h - 1
            if isinstance(y_pred, pd.DataFrame):
                y_pred_h = y_pred.iloc[pred_idx:pred_idx+1] if pred_idx < len(y_pred) else pd.DataFrame()
            elif isinstance(y_pred, pd.Series):
                y_pred_h = y_pred.iloc[pred_idx:pred_idx+1] if pred_idx < len(y_pred) else pd.Series()
            else:
                y_pred_h = y_pred[pred_idx:pred_idx+1] if pred_idx < len(y_pred) else np.array([])
            
            if isinstance(y_true, pd.DataFrame):
                y_true_h = y_true.iloc[pred_idx:pred_idx+1] if pred_idx < len(y_true) else pd.DataFrame()
            elif isinstance(y_true, pd.Series):
                y_true_h = y_true.iloc[pred_idx:pred_idx+1] if pred_idx < len(y_true) else pd.Series()
            else:
                y_true_h = y_true[pred_idx:pred_idx+1] if pred_idx < len(y_true) else np.array([])
        
        has_pred = len(y_pred_h) > 0 if hasattr(y_pred_h, '__len__') else (y_pred_h.size > 0 if hasattr(y_pred_h, 'size') else False)
        has_true = len(y_true_h) > 0 if hasattr(y_true_h, '__len__') else (y_true_h.size > 0 if hasattr(y_true_h, 'size') else False)
        
        if has_pred and has_true:
            try:
                target_series_for_metrics = target_series
                if isinstance(y_true_h, pd.Series) and isinstance(target_series, str):
                    target_series_for_metrics = None
                elif isinstance(y_true_h, pd.DataFrame) and len(y_true_h.columns) == 1:
                    if isinstance(target_series, str) and target_series not in y_true_h.columns:
                        target_series_for_metrics = None
                
                if h == 1 and y_train is not None:
                    try:
                        if isinstance(y_train, (pd.DataFrame, pd.Series)):
                            last_train_val = y_train.iloc[-1]
                            if isinstance(y_pred_h, (pd.DataFrame, pd.Series)):
                                pred_val = y_pred_h.iloc[0] if len(y_pred_h) > 0 else None
                                if pred_val is not None and isinstance(last_train_val, (pd.Series, pd.DataFrame)):
                                    if isinstance(pred_val, pd.Series) and isinstance(last_train_val, pd.Series):
                                        diff = (pred_val - last_train_val).abs()
                                        if target_series_for_metrics is not None:
                                            if isinstance(target_series_for_metrics, str) and target_series_for_metrics in diff.index:
                                                diff_val = diff[target_series_for_metrics]
                                            elif isinstance(target_series_for_metrics, int) and target_series_for_metrics < len(diff):
                                                diff_val = diff.iloc[target_series_for_metrics]
                                            else:
                                                diff_val = diff.iloc[0] if len(diff) > 0 else None
                                        else:
                                            diff_val = diff.iloc[0] if len(diff) > 0 else None
                                        
                                        if diff_val is not None and diff_val < 1e-6:
                                            logger.warning(f"Horizon {h}: Prediction is extremely close to last training value (diff={diff_val}). This may indicate VAR is essentially predicting persistence (last value), which could explain suspiciously good results. For VAR models, this is often a sign of numerical underflow or model instability rather than true forecasting ability.")
                    except Exception as e:
                        logger.debug(f"Horizon {h}: Could not check prediction vs last training value: {e}")
                
                metrics = calculate_standardized_metrics(y_true_h, y_pred_h, y_train=y_train, target_series=target_series_for_metrics)
                results[h] = metrics
            except Exception as e:
                results[h] = {'sMSE': np.nan, 'sMAE': np.nan, 'sRMSE': np.nan, 'MSE': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'sigma': np.nan, 'n_valid': 0}
        else:
            results[h] = {'sMSE': np.nan, 'sMAE': np.nan, 'sRMSE': np.nan, 'MSE': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'sigma': np.nan, 'n_valid': 0}
    
    return results


def calculate_temporal_consistency_metrics(
    predictions_by_horizon: Dict[int, Union[pd.DataFrame, pd.Series, np.ndarray]],
    threshold_std: float = 2.0
) -> Dict[str, float]:
    """Calculate temporal consistency metrics for predictions across horizons.
    
    This function analyzes how predictions change across consecutive horizons to
    detect sudden jumps or inconsistencies that may indicate model instability
    or factor dynamics issues.
    
    Parameters
    ----------
    predictions_by_horizon : dict
        Dictionary mapping horizon (int) to predictions (DataFrame, Series, or array)
        Predictions should be for the same target time point across different horizons
    threshold_std : float, default 2.0
        Number of standard deviations to use as threshold for detecting jumps
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'temporal_consistency': Overall consistency score (0-1, higher = more consistent)
        - 'jump_count': Number of detected jumps between consecutive horizons
        - 'jump_fraction': Fraction of horizon transitions with jumps
        - 'max_jump_magnitude': Maximum jump magnitude (in standard deviations)
        - 'avg_jump_magnitude': Average jump magnitude for detected jumps
        - 'consistency_by_horizon': Per-horizon consistency scores
    """
    logger = _module_logger
    
    if len(predictions_by_horizon) < 2:
        logger.warning("Need at least 2 horizons for temporal consistency analysis")
        return {
            'temporal_consistency': None,
            'jump_count': 0,
            'jump_fraction': 0.0,
            'max_jump_magnitude': 0.0,
            'avg_jump_magnitude': 0.0,
            'consistency_by_horizon': {}
        }
    
    # Convert all predictions to numpy arrays
    pred_arrays = {}
    for h, pred in predictions_by_horizon.items():
        if isinstance(pred, pd.DataFrame):
            arr = pred.values.flatten()
        elif isinstance(pred, pd.Series):
            arr = pred.values
        else:
            try:
                import torch
                if isinstance(pred, torch.Tensor):
                    arr = pred.cpu().numpy().flatten()
                else:
                    arr = np.asarray(pred).flatten()
            except ImportError:
                arr = np.asarray(pred).flatten()
        pred_arrays[h] = arr
    
    # Sort horizons
    sorted_horizons = sorted(pred_arrays.keys())
    
    # Calculate differences between consecutive horizons
    jumps = []
    consistency_scores = []
    
    for i in range(len(sorted_horizons) - 1):
        h1 = sorted_horizons[i]
        h2 = sorted_horizons[i + 1]
        
        pred1 = pred_arrays[h1]
        pred2 = pred_arrays[h2]
        
        # Ensure same length (take minimum)
        min_len = min(len(pred1), len(pred2))
        if min_len == 0:
            continue
        
        pred1 = pred1[:min_len]
        pred2 = pred2[:min_len]
        
        # Filter valid values
        mask = np.isfinite(pred1) & np.isfinite(pred2)
        if np.sum(mask) == 0:
            continue
        
        pred1_valid = pred1[mask]
        pred2_valid = pred2[mask]
        
        # Calculate differences
        diff = pred2_valid - pred1_valid
        
        # Calculate jump statistics
        diff_mean = np.mean(diff)
        diff_std = np.std(diff)
        
        if diff_std > 0:
            # Normalize differences by standard deviation
            normalized_diff = np.abs(diff - diff_mean) / diff_std
            
            # Count jumps (values exceeding threshold)
            jump_mask = normalized_diff > threshold_std
            jump_count = np.sum(jump_mask)
            jump_fraction = jump_count / len(normalized_diff)
            
            # Calculate consistency score for this transition (1 - jump_fraction)
            consistency = 1.0 - min(jump_fraction, 1.0)
            consistency_scores.append(consistency)
            
            if jump_count > 0:
                jump_magnitudes = normalized_diff[jump_mask]
                jumps.append({
                    'horizon_from': h1,
                    'horizon_to': h2,
                    'jump_count': int(jump_count),
                    'jump_fraction': float(jump_fraction),
                    'max_magnitude': float(np.max(jump_magnitudes)),
                    'avg_magnitude': float(np.mean(jump_magnitudes))
                })
        else:
            # No variation, perfect consistency
            consistency_scores.append(1.0)
    
    # Overall consistency
    if consistency_scores:
        overall_consistency = float(np.mean(consistency_scores))
    else:
        overall_consistency = None
    
    # Jump statistics
    if jumps:
        jump_count_total = sum(j['jump_count'] for j in jumps)
        jump_fraction_total = sum(j['jump_fraction'] for j in jumps) / len(jumps) if jumps else 0.0
        max_jump = max(j['max_magnitude'] for j in jumps)
        avg_jump = np.mean([j['avg_magnitude'] for j in jumps])
    else:
        jump_count_total = 0
        jump_fraction_total = 0.0
        max_jump = 0.0
        avg_jump = 0.0
    
    return {
        'temporal_consistency': overall_consistency,
        'jump_count': jump_count_total,
        'jump_fraction': float(jump_fraction_total),
        'max_jump_magnitude': float(max_jump),
        'avg_jump_magnitude': float(avg_jump),
        'consistency_by_horizon': {f"{j['horizon_from']}-{j['horizon_to']}": 1.0 - j['jump_fraction'] for j in jumps}
    }


def calculate_relative_error_stability(
    ddfm_errors_by_horizon: Dict[int, float],
    dfm_errors_by_horizon: Dict[int, float]
) -> Dict[str, float]:
    """Calculate relative error stability metrics between DDFM and DFM.
    
    This function analyzes how the relative performance (DDFM vs DFM) changes
    across horizons to detect systematic patterns or instability.
    
    Parameters
    ----------
    ddfm_errors_by_horizon : dict
        Dictionary mapping horizon (int) to DDFM error (float)
    dfm_errors_by_horizon : dict
        Dictionary mapping horizon (int) to DFM error (float)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'relative_error_stability': Stability score (0-1, higher = more stable)
        - 'relative_error_cv': Coefficient of variation of relative errors
        - 'relative_error_trend': Trend in relative errors across horizons ('improving', 'degrading', 'stable')
        - 'avg_relative_error': Average relative error (DDFM/DFM ratio)
        - 'min_relative_error': Minimum relative error (best DDFM performance)
        - 'max_relative_error': Maximum relative error (worst DDFM performance)
    """
    logger = _module_logger
    
    # Find common horizons
    common_horizons = sorted(set(ddfm_errors_by_horizon.keys()) & set(dfm_errors_by_horizon.keys()))
    
    if len(common_horizons) < 2:
        logger.warning("Need at least 2 common horizons for relative error stability analysis")
        return {
            'relative_error_stability': None,
            'relative_error_cv': None,
            'relative_error_trend': None,
            'avg_relative_error': None,
            'min_relative_error': None,
            'max_relative_error': None
        }
    
    # Calculate relative errors (DDFM/DFM ratio)
    relative_errors = []
    for h in common_horizons:
        ddfm_err = ddfm_errors_by_horizon[h]
        dfm_err = dfm_errors_by_horizon[h]
        
        if dfm_err > 0 and np.isfinite(ddfm_err) and np.isfinite(dfm_err):
            rel_err = ddfm_err / dfm_err
            relative_errors.append(rel_err)
        else:
            relative_errors.append(np.nan)
    
    # Filter valid relative errors
    valid_relative_errors = [r for r in relative_errors if not np.isnan(r)]
    
    if len(valid_relative_errors) < 2:
        return {
            'relative_error_stability': None,
            'relative_error_cv': None,
            'relative_error_trend': None,
            'avg_relative_error': None,
            'min_relative_error': None,
            'max_relative_error': None
        }
    
    rel_err_array = np.array(valid_relative_errors)
    
    # Calculate statistics
    avg_rel_err = float(np.mean(rel_err_array))
    min_rel_err = float(np.min(rel_err_array))
    max_rel_err = float(np.max(rel_err_array))
    std_rel_err = float(np.std(rel_err_array))
    
    # Coefficient of variation
    if avg_rel_err > 0:
        cv = std_rel_err / avg_rel_err
    else:
        cv = None
    
    # Stability score (inverse of CV, normalized to 0-1)
    if cv is not None:
        stability = float(max(0.0, 1.0 - min(cv, 1.0)))
    else:
        stability = None
    
    # Trend analysis
    if len(valid_relative_errors) >= 3:
        try:
            from scipy import stats
            horizons_array = np.array(common_horizons[:len(valid_relative_errors)])
            valid_horizons = horizons_array[~np.isnan(relative_errors[:len(horizons_array)])]
            valid_errors = rel_err_array
            
            if len(valid_horizons) >= 3 and len(valid_errors) >= 3:
                slope, intercept, r, p, std_err = stats.linregress(valid_horizons, valid_errors)
                r_squared = r ** 2
                
                if r_squared > 0.3:  # Significant trend
                    if slope < -0.001:
                        trend = 'improving'  # Relative error decreasing (DDFM getting better)
                    elif slope > 0.001:
                        trend = 'degrading'  # Relative error increasing (DDFM getting worse)
                    else:
                        trend = 'stable'
                else:
                    trend = 'stable'
            else:
                trend = 'stable'
        except (ImportError, Exception) as e:
            logger.debug(f"Could not calculate relative error trend: {e}")
            trend = 'stable'
    else:
        trend = 'stable'
    
    return {
        'relative_error_stability': stability,
        'relative_error_cv': float(cv) if cv is not None else None,
        'relative_error_trend': trend,
        'avg_relative_error': avg_rel_err,
        'min_relative_error': min_rel_err,
        'max_relative_error': max_rel_err
    }


def calculate_improvement_persistence(
    improvement_by_horizon: Dict[int, float],
    persistence_threshold: float = 0.05
) -> Dict[str, Any]:
    """Calculate improvement persistence metrics across horizons.
    
    This function analyzes whether DDFM improvements over DFM are persistent
    (consistent across horizons) or transient (just noise at specific horizons).
    
    Parameters
    ----------
    improvement_by_horizon : dict
        Dictionary mapping horizon (int) to improvement ratio (float)
        Improvement ratio = (DFM_error - DDFM_error) / DFM_error
    persistence_threshold : float, default 0.05
        Minimum improvement ratio to consider as "improvement" (5%)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'persistence_score': Persistence score (0-1, higher = more persistent)
        - 'improvement_fraction': Fraction of horizons with improvement > threshold
        - 'consecutive_improvements': Longest streak of consecutive improvements
        - 'improvement_clusters': Number of clusters of improved horizons
        - 'avg_improvement': Average improvement across all horizons
        - 'improvement_consistency': Consistency of improvement (std of improvements)
    """
    logger = _module_logger
    
    if len(improvement_by_horizon) == 0:
        return {
            'persistence_score': None,
            'improvement_fraction': 0.0,
            'consecutive_improvements': 0,
            'improvement_clusters': 0,
            'avg_improvement': None,
            'improvement_consistency': None
        }
    
    # Sort horizons
    sorted_horizons = sorted(improvement_by_horizon.keys())
    improvements = [improvement_by_horizon[h] for h in sorted_horizons]
    
    # Filter valid improvements
    valid_mask = [not np.isnan(imp) for imp in improvements]
    valid_improvements = [imp for imp, mask in zip(improvements, valid_mask) if mask]
    valid_horizons = [h for h, mask in zip(sorted_horizons, valid_mask) if mask]
    
    if len(valid_improvements) == 0:
        return {
            'persistence_score': None,
            'improvement_fraction': 0.0,
            'consecutive_improvements': 0,
            'improvement_clusters': 0,
            'avg_improvement': None,
            'improvement_consistency': None
        }
    
    # Calculate improvement fraction
    improvements_above_threshold = [imp for imp in valid_improvements if imp > persistence_threshold]
    improvement_fraction = len(improvements_above_threshold) / len(valid_improvements)
    
    # Find consecutive improvements
    consecutive_streaks = []
    current_streak = 0
    max_streak = 0
    
    for i, imp in enumerate(valid_improvements):
        if imp > persistence_threshold:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            if current_streak > 0:
                consecutive_streaks.append(current_streak)
            current_streak = 0
    
    if current_streak > 0:
        consecutive_streaks.append(current_streak)
        max_streak = max(max_streak, current_streak)
    
    # Count improvement clusters (groups of consecutive improvements)
    improvement_clusters = 0
    in_cluster = False
    for imp in valid_improvements:
        if imp > persistence_threshold:
            if not in_cluster:
                improvement_clusters += 1
                in_cluster = True
        else:
            in_cluster = False
    
    # Average improvement
    avg_improvement = float(np.mean(valid_improvements))
    
    # Improvement consistency (inverse of std, normalized)
    if len(valid_improvements) > 1:
        std_improvement = float(np.std(valid_improvements))
        mean_abs_improvement = float(np.mean(np.abs(valid_improvements)))
        if mean_abs_improvement > 0:
            consistency = 1.0 - min(std_improvement / mean_abs_improvement, 1.0)
        else:
            consistency = 0.0
    else:
        consistency = 1.0 if len(valid_improvements) > 0 else None
    
    # Persistence score: combination of fraction, streak, and consistency
    if improvement_fraction > 0:
        persistence_score = (
            0.4 * improvement_fraction +
            0.3 * min(max_streak / max(len(valid_horizons), 1), 1.0) +
            0.3 * (consistency if consistency is not None else 0.0)
        )
    else:
        persistence_score = 0.0
    
    return {
        'persistence_score': float(persistence_score),
        'improvement_fraction': float(improvement_fraction),
        'consecutive_improvements': int(max_streak),
        'improvement_clusters': int(improvement_clusters),
        'avg_improvement': avg_improvement,
        'improvement_consistency': float(consistency) if consistency is not None else None
    }


def calculate_robust_metrics(
    y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_pred: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_train: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
    target_series: Optional[Union[str, int]] = None
) -> Dict[str, float]:
    """Calculate robust (median-based) metrics for DDFM evaluation.
    
    This function calculates robust alternatives to mean-based metrics using
    median and IQR (Interquartile Range) statistics, which are more resistant
    to outliers. This is particularly useful for DDFM evaluation when some
    horizons may have extreme errors due to numerical instability or model issues.
    
    Parameters
    ----------
    y_true : pd.DataFrame, pd.Series, or np.ndarray
        True/actual values
    y_pred : pd.DataFrame, pd.Series, or np.ndarray
        Predicted/forecasted values
    y_train : pd.DataFrame, pd.Series, or np.ndarray, optional
        Training data used for standardization (if None, uses y_true)
    target_series : str or int, optional
        Target series name or index (for multivariate data)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'robust_sMAE': Median absolute error normalized by training MAD (Median Absolute Deviation)
        - 'robust_sMSE': Median squared error normalized by training variance
        - 'robust_sRMSE': Square root of robust_sMSE
        - 'iqr_sMAE': IQR of absolute errors normalized by training IQR
        - 'mad_train': Median Absolute Deviation of training data
        - 'iqr_train': Interquartile Range of training data
        - 'outlier_fraction': Fraction of predictions that are outliers (using IQR method)
        - 'n_valid': Number of valid (finite) observations
    """
    logger = _module_logger
    
    # Convert to numpy arrays (same as calculate_standardized_metrics)
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
        try:
            import torch
            if isinstance(y_pred, torch.Tensor):
                y_pred_arr = y_pred.cpu().numpy()
            else:
                y_pred_arr = np.asarray(y_pred)
        except ImportError:
            y_pred_arr = np.asarray(y_pred)
    
    if y_pred_arr.ndim == 1:
        y_pred_arr = y_pred_arr.reshape(-1, 1)
    
    # Ensure same shape
    min_rows = min(y_true_arr.shape[0], y_pred_arr.shape[0])
    min_cols = min(y_true_arr.shape[1], y_pred_arr.shape[1])
    y_true_arr = y_true_arr[:min_rows, :min_cols]
    y_pred_arr = y_pred_arr[:min_rows, :min_cols]
    
    # Get target series index
    target_idx = None
    if target_series is not None:
        if isinstance(target_series, str) and target_series in columns:
            target_idx = columns.index(target_series)
        elif isinstance(target_series, int) and 0 <= target_series < len(columns):
            target_idx = target_series
    
    # Calculate errors
    errors = y_pred_arr - y_true_arr
    
    # Filter valid values
    if target_idx is not None:
        errors_series = errors[:, target_idx]
        y_train_series = None
        if y_train is not None:
            if isinstance(y_train, pd.DataFrame):
                if target_series in y_train.columns:
                    y_train_series = y_train[target_series].values
            elif isinstance(y_train, pd.Series):
                y_train_series = y_train.values
            else:
                y_train_arr = np.asarray(y_train)
                if y_train_arr.ndim == 1:
                    y_train_series = y_train_arr
                else:
                    y_train_series = y_train_arr[:, target_idx] if target_idx < y_train_arr.shape[1] else y_train_arr[:, 0]
    else:
        errors_series = errors.flatten()
        if y_train is not None:
            if isinstance(y_train, pd.DataFrame):
                y_train_series = y_train.values.flatten()
            elif isinstance(y_train, pd.Series):
                y_train_series = y_train.values
            else:
                y_train_arr = np.asarray(y_train)
                y_train_series = y_train_arr.flatten()
    
    # Filter finite values
    mask = np.isfinite(errors_series)
    errors_valid = errors_series[mask]
    
    if len(errors_valid) == 0:
        return {
            'robust_sMAE': np.nan,
            'robust_sMSE': np.nan,
            'robust_sRMSE': np.nan,
            'iqr_sMAE': np.nan,
            'mad_train': np.nan,
            'iqr_train': np.nan,
            'outlier_fraction': np.nan,
            'n_valid': 0
        }
    
    # Calculate training statistics for normalization
    if y_train_series is not None:
        train_mask = np.isfinite(y_train_series)
        train_valid = y_train_series[train_mask]
        if len(train_valid) > 0:
            mad_train = np.median(np.abs(train_valid - np.median(train_valid)))
            q75_train, q25_train = np.percentile(train_valid, [75, 25])
            iqr_train = q75_train - q25_train
        else:
            mad_train = np.nan
            iqr_train = np.nan
    else:
        # Use errors for normalization if no training data
        mad_train = np.median(np.abs(errors_valid - np.median(errors_valid))) if len(errors_valid) > 0 else np.nan
        q75, q25 = np.percentile(errors_valid, [75, 25]) if len(errors_valid) > 0 else (np.nan, np.nan)
        iqr_train = q75 - q25
    
    # Calculate robust metrics
    abs_errors = np.abs(errors_valid)
    squared_errors = errors_valid ** 2
    
    # Median-based metrics
    median_ae = np.median(abs_errors)
    median_se = np.median(squared_errors)
    
    # IQR-based metrics
    q75_ae, q25_ae = np.percentile(abs_errors, [75, 25])
    iqr_ae = q75_ae - q25_ae
    
    # Normalize by training statistics
    if mad_train is not None and not np.isnan(mad_train) and mad_train > 0:
        robust_smae = median_ae / mad_train
    else:
        robust_smae = np.nan
    
    if iqr_train is not None and not np.isnan(iqr_train) and iqr_train > 0:
        iqr_smae = iqr_ae / iqr_train
    else:
        iqr_smae = np.nan
    
    # For MSE, use median of squared errors normalized by training variance
    if y_train_series is not None and len(train_valid) > 0:
        train_var = np.var(train_valid)
        if train_var > 0:
            robust_smse = median_se / train_var
            robust_srmse = np.sqrt(robust_smse) if not np.isnan(robust_smse) else np.nan
        else:
            robust_smse = np.nan
            robust_srmse = np.nan
    else:
        robust_smse = np.nan
        robust_srmse = np.nan
    
    # Outlier detection using IQR method
    if len(abs_errors) > 0:
        q1_ae, q3_ae = np.percentile(abs_errors, [25, 75])
        iqr_ae_val = q3_ae - q1_ae
        if iqr_ae_val > 0:
            lower_bound = q1_ae - 1.5 * iqr_ae_val
            upper_bound = q3_ae + 1.5 * iqr_ae_val
            outliers = np.sum((abs_errors < lower_bound) | (abs_errors > upper_bound))
            outlier_fraction = outliers / len(abs_errors)
        else:
            outlier_fraction = 0.0
    else:
        outlier_fraction = np.nan
    
    return {
        'robust_sMAE': float(robust_smae) if not np.isnan(robust_smae) else np.nan,
        'robust_sMSE': float(robust_smse) if not np.isnan(robust_smse) else np.nan,
        'robust_sRMSE': float(robust_srmse) if not np.isnan(robust_srmse) else np.nan,
        'iqr_sMAE': float(iqr_smae) if not np.isnan(iqr_smae) else np.nan,
        'mad_train': float(mad_train) if not np.isnan(mad_train) else np.nan,
        'iqr_train': float(iqr_train) if not np.isnan(iqr_train) else np.nan,
        'outlier_fraction': float(outlier_fraction) if not np.isnan(outlier_fraction) else np.nan,
        'n_valid': int(len(errors_valid))
    }


def calculate_bootstrap_confidence_intervals(
    y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_pred: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_train: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
    target_series: Optional[Union[str, int]] = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Dict[str, Dict[str, float]]:
    """Calculate bootstrap confidence intervals for metrics.
    
    This function uses bootstrap resampling to estimate confidence intervals
    for forecasting metrics, providing uncertainty quantification for DDFM
    performance evaluation.
    
    Parameters
    ----------
    y_true : pd.DataFrame, pd.Series, or np.ndarray
        True/actual values
    y_pred : pd.DataFrame, pd.Series, or np.ndarray
        Predicted/forecasted values
    y_train : pd.DataFrame, pd.Series, or np.ndarray, optional
        Training data used for standardization
    target_series : str or int, optional
        Target series name or index (for multivariate data)
    n_bootstrap : int, default 1000
        Number of bootstrap samples
    confidence_level : float, default 0.95
        Confidence level (e.g., 0.95 for 95% confidence interval)
        
    Returns
    -------
    dict
        Dictionary containing confidence intervals for each metric:
        - 'sMAE': {'lower': float, 'upper': float, 'mean': float}
        - 'sMSE': {'lower': float, 'upper': float, 'mean': float}
        - 'sRMSE': {'lower': float, 'upper': float, 'mean': float}
    """
    logger = _module_logger
    
    # Calculate base metrics
    base_metrics = calculate_standardized_metrics(y_true, y_pred, y_train=y_train, target_series=target_series)
    
    if base_metrics['n_valid'] == 0:
        return {
            'sMAE': {'lower': np.nan, 'upper': np.nan, 'mean': np.nan},
            'sMSE': {'lower': np.nan, 'upper': np.nan, 'mean': np.nan},
            'sRMSE': {'lower': np.nan, 'upper': np.nan, 'mean': np.nan}
        }
    
    # Convert to numpy arrays for bootstrap
    if isinstance(y_true, pd.DataFrame):
        y_true_arr = y_true.values
    elif isinstance(y_true, pd.Series):
        y_true_arr = y_true.values.reshape(-1, 1)
    else:
        y_true_arr = np.asarray(y_true)
        if y_true_arr.ndim == 1:
            y_true_arr = y_true_arr.reshape(-1, 1)
    
    if isinstance(y_pred, pd.DataFrame):
        y_pred_arr = y_pred.values
    elif isinstance(y_pred, pd.Series):
        y_pred_arr = y_pred.values.reshape(-1, 1)
    else:
        try:
            import torch
            if isinstance(y_pred, torch.Tensor):
                y_pred_arr = y_pred.cpu().numpy()
            else:
                y_pred_arr = np.asarray(y_pred)
        except ImportError:
            y_pred_arr = np.asarray(y_pred)
    
    if y_pred_arr.ndim == 1:
        y_pred_arr = y_pred_arr.reshape(-1, 1)
    
    # Get target series index
    target_idx = None
    if target_series is not None:
        if isinstance(y_true, pd.DataFrame) and isinstance(target_series, str):
            if target_series in y_true.columns:
                target_idx = list(y_true.columns).index(target_series)
        elif isinstance(target_series, int):
            target_idx = target_series
    
    # Extract errors for bootstrap
    min_rows = min(y_true_arr.shape[0], y_pred_arr.shape[0])
    min_cols = min(y_true_arr.shape[1], y_pred_arr.shape[1])
    y_true_arr = y_true_arr[:min_rows, :min_cols]
    y_pred_arr = y_pred_arr[:min_rows, :min_cols]
    
    if target_idx is not None and target_idx < min_cols:
        errors = (y_pred_arr[:, target_idx] - y_true_arr[:, target_idx])
    else:
        errors = (y_pred_arr - y_true_arr).flatten()
    
    # Filter valid values
    mask = np.isfinite(errors)
    errors_valid = errors[mask]
    n_valid = len(errors_valid)
    
    if n_valid < 10:  # Need at least 10 samples for bootstrap
        logger.warning(f"Too few valid samples ({n_valid}) for bootstrap confidence intervals")
        return {
            'sMAE': {'lower': base_metrics.get('sMAE'), 'upper': base_metrics.get('sMAE'), 'mean': base_metrics.get('sMAE')},
            'sMSE': {'lower': base_metrics.get('sMSE'), 'upper': base_metrics.get('sMSE'), 'mean': base_metrics.get('sMSE')},
            'sRMSE': {'lower': base_metrics.get('sRMSE'), 'upper': base_metrics.get('sRMSE'), 'mean': base_metrics.get('sRMSE')}
        }
    
    # Get training std for normalization
    sigma = base_metrics.get('sigma')
    if sigma is None or np.isnan(sigma) or sigma <= 0:
        sigma = np.std(errors_valid) if len(errors_valid) > 0 else 1.0
    
    # Bootstrap resampling
    bootstrap_smae = []
    bootstrap_smse = []
    bootstrap_srmse = []
    
    np.random.seed(42)  # For reproducibility
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_valid, size=n_valid, replace=True)
        errors_bootstrap = errors_valid[indices]
        
        # Calculate metrics for this bootstrap sample
        mae_boot = np.mean(np.abs(errors_bootstrap))
        mse_boot = np.mean(errors_bootstrap ** 2)
        rmse_boot = np.sqrt(mse_boot)
        
        # Normalize
        smae_boot = mae_boot / sigma if sigma > 0 else np.nan
        smse_boot = mse_boot / (sigma ** 2) if sigma > 0 else np.nan
        srmse_boot = rmse_boot / sigma if sigma > 0 else np.nan
        
        if not np.isnan(smae_boot):
            bootstrap_smae.append(smae_boot)
        if not np.isnan(smse_boot):
            bootstrap_smse.append(smse_boot)
        if not np.isnan(srmse_boot):
            bootstrap_srmse.append(srmse_boot)
    
    # Calculate confidence intervals
    alpha = 1.0 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)
    
    def calc_ci(values):
        if len(values) == 0:
            return {'lower': np.nan, 'upper': np.nan, 'mean': np.nan}
        return {
            'lower': float(np.percentile(values, lower_percentile)),
            'upper': float(np.percentile(values, upper_percentile)),
            'mean': float(np.mean(values))
        }
    
    return {
        'sMAE': calc_ci(bootstrap_smae),
        'sMSE': calc_ci(bootstrap_smse),
        'sRMSE': calc_ci(bootstrap_srmse)
    }


def aggregate_robust_metrics_across_horizons(
    metrics_by_horizon: Dict[int, Dict[str, float]],
    use_median: bool = True
) -> Dict[str, float]:
    """Aggregate metrics across horizons using robust statistics.
    
    This function aggregates metrics across multiple forecast horizons using
    robust statistics (median, IQR) instead of mean, making the aggregation
    more resistant to outliers from specific problematic horizons.
    
    Parameters
    ----------
    metrics_by_horizon : dict
        Dictionary mapping horizon (int) to metrics dict
    use_median : bool, default True
        If True, use median for aggregation; if False, use mean
        
    Returns
    -------
    dict
        Dictionary containing aggregated metrics:
        - 'aggregated_sMAE': Aggregated sMAE (median or mean)
        - 'aggregated_sMSE': Aggregated sMSE (median or mean)
        - 'aggregated_sRMSE': Aggregated sRMSE (median or mean)
        - 'iqr_sMAE': IQR of sMAE across horizons
        - 'iqr_sMSE': IQR of sMSE across horizons
        - 'iqr_sRMSE': IQR of sRMSE across horizons
        - 'n_horizons': Number of valid horizons
    """
    logger = _module_logger
    
    if len(metrics_by_horizon) == 0:
        return {
            'aggregated_sMAE': np.nan,
            'aggregated_sMSE': np.nan,
            'aggregated_sRMSE': np.nan,
            'iqr_sMAE': np.nan,
            'iqr_sMSE': np.nan,
            'iqr_sRMSE': np.nan,
            'n_horizons': 0
        }
    
    # Extract metrics
    smae_values = []
    smse_values = []
    srmse_values = []
    
    for h, metrics in metrics_by_horizon.items():
        smae = metrics.get('sMAE')
        smse = metrics.get('sMSE')
        srmse = metrics.get('sRMSE')
        
        if smae is not None and not np.isnan(smae):
            smae_values.append(smae)
        if smse is not None and not np.isnan(smse):
            smse_values.append(smse)
        if srmse is not None and not np.isnan(srmse):
            srmse_values.append(srmse)
    
    # Aggregate using robust statistics
    if use_median:
        agg_func = np.median
    else:
        agg_func = np.nanmean
    
    def calc_iqr(values):
        if len(values) < 2:
            return np.nan
        q75, q25 = np.percentile(values, [75, 25])
        return float(q75 - q25)
    
    return {
        'aggregated_sMAE': float(agg_func(smae_values)) if smae_values else np.nan,
        'aggregated_sMSE': float(agg_func(smse_values)) if smse_values else np.nan,
        'aggregated_sRMSE': float(agg_func(srmse_values)) if srmse_values else np.nan,
        'iqr_sMAE': calc_iqr(smae_values),
        'iqr_sMSE': calc_iqr(smse_values),
        'iqr_sRMSE': calc_iqr(srmse_values),
        'n_horizons': len(metrics_by_horizon)
    }


def calculate_factor_dynamics_stability(
    predictions_by_horizon: Dict[int, Union[pd.DataFrame, pd.Series, np.ndarray]],
    threshold_oscillation: float = 0.1,
    threshold_growth: float = 0.05
) -> Dict[str, float]:
    """Infer factor dynamics stability from prediction patterns.
    
    This function analyzes prediction patterns across horizons to infer whether
    the VAR factor dynamics are stable. Unstable factor dynamics can cause:
    - Oscillations (predictions alternate between high and low)
    - Exponential growth/decay (predictions diverge or converge rapidly)
    - Numerical instability (sudden jumps or NaN values)
    
    This analysis helps identify if the VAR transition matrix has eigenvalues
    outside the unit circle (unstable) or complex eigenvalues with large
    imaginary parts (oscillatory behavior).
    
    Parameters
    ----------
    predictions_by_horizon : dict
        Dictionary mapping horizon (int) to predictions (DataFrame, Series, or array)
        Predictions should be for the same target time point across different horizons
    threshold_oscillation : float, default 0.1
        Threshold for detecting oscillations (relative change in direction)
    threshold_growth : float, default 0.05
        Threshold for detecting exponential growth/decay (relative change per horizon)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'stability_score': Overall stability score (0-1, higher = more stable)
        - 'oscillation_detected': Whether oscillations are detected (bool)
        - 'oscillation_magnitude': Magnitude of oscillations (0-1, higher = more oscillatory)
        - 'growth_rate': Estimated growth/decay rate per horizon (positive = growth, negative = decay)
        - 'smoothness_score': Prediction smoothness (0-1, higher = smoother)
        - 'divergence_detected': Whether predictions are diverging (bool)
        - 'convergence_detected': Whether predictions are converging (bool)
        - 'second_derivative_variance': Variance of second derivatives (higher = less smooth)
        - 'stability_interpretation': Text interpretation of stability ('stable', 'oscillatory', 'unstable', 'diverging')
    """
    logger = _module_logger
    
    if len(predictions_by_horizon) < 3:
        logger.warning("Need at least 3 horizons for factor dynamics stability analysis")
        return {
            'stability_score': None,
            'oscillation_detected': False,
            'oscillation_magnitude': 0.0,
            'growth_rate': 0.0,
            'smoothness_score': None,
            'divergence_detected': False,
            'convergence_detected': False,
            'second_derivative_variance': 0.0,
            'stability_interpretation': 'insufficient_data'
        }
    
    # Convert all predictions to numpy arrays
    pred_arrays = {}
    for h, pred in predictions_by_horizon.items():
        if isinstance(pred, pd.DataFrame):
            arr = pred.values.flatten()
        elif isinstance(pred, pd.Series):
            arr = pred.values
        else:
            try:
                import torch
                if isinstance(pred, torch.Tensor):
                    arr = pred.cpu().numpy().flatten()
                else:
                    arr = np.asarray(pred).flatten()
            except ImportError:
                arr = np.asarray(pred).flatten()
        pred_arrays[h] = arr
    
    # Sort horizons
    sorted_horizons = sorted(pred_arrays.keys())
    
    # Extract prediction values (use first element if array, or mean if multiple)
    pred_values = []
    valid_horizons = []
    for h in sorted_horizons:
        arr = pred_arrays[h]
        if len(arr) > 0:
            # Use first value or mean if multiple values
            val = arr[0] if len(arr) == 1 else np.mean(arr)
            if np.isfinite(val):
                pred_values.append(val)
                valid_horizons.append(h)
    
    if len(pred_values) < 3:
        logger.warning("Need at least 3 valid predictions for factor dynamics stability analysis")
        return {
            'stability_score': None,
            'oscillation_detected': False,
            'oscillation_magnitude': 0.0,
            'growth_rate': 0.0,
            'smoothness_score': None,
            'divergence_detected': False,
            'convergence_detected': False,
            'second_derivative_variance': 0.0,
            'stability_interpretation': 'insufficient_data'
        }
    
    pred_values = np.array(pred_values)
    horizons = np.array(valid_horizons)
    
    # Calculate first derivatives (rate of change)
    first_derivatives = np.diff(pred_values) / np.diff(horizons)
    
    # Calculate second derivatives (acceleration/curvature)
    if len(first_derivatives) > 1:
        second_derivatives = np.diff(first_derivatives) / np.diff(horizons[:-1])
        second_derivative_variance = float(np.var(second_derivatives)) if len(second_derivatives) > 0 else 0.0
    else:
        second_derivatives = np.array([])
        second_derivative_variance = 0.0
    
    # Smoothness score: inverse of second derivative variance (normalized)
    # Higher smoothness = lower variance in second derivatives
    if len(second_derivatives) > 0 and np.std(pred_values) > 1e-10:
        # Normalize by prediction scale
        normalized_variance = second_derivative_variance / (np.std(pred_values) ** 2 + 1e-10)
        smoothness_score = float(max(0.0, 1.0 - min(normalized_variance, 1.0)))
    else:
        smoothness_score = 1.0 if len(second_derivatives) == 0 else 0.0
    
    # Detect oscillations: count sign changes in first derivatives
    if len(first_derivatives) > 1:
        sign_changes = np.sum(np.diff(np.sign(first_derivatives)) != 0)
        oscillation_fraction = sign_changes / (len(first_derivatives) - 1)
        oscillation_detected = oscillation_fraction > threshold_oscillation
        
        # Oscillation magnitude: how much do predictions alternate?
        if oscillation_detected:
            # Calculate average magnitude of oscillations
            oscillation_magnitudes = []
            for i in range(len(first_derivatives) - 1):
                if np.sign(first_derivatives[i]) != np.sign(first_derivatives[i + 1]):
                    # Oscillation detected: calculate relative change
                    rel_change = abs(pred_values[i + 1] - pred_values[i]) / (abs(pred_values[i]) + 1e-10)
                    oscillation_magnitudes.append(rel_change)
            oscillation_magnitude = float(np.mean(oscillation_magnitudes)) if oscillation_magnitudes else 0.0
        else:
            oscillation_magnitude = 0.0
    else:
        oscillation_detected = False
        oscillation_magnitude = 0.0
        oscillation_fraction = 0.0
    
    # Detect exponential growth/decay: fit exponential trend
    # Use linear regression on log(abs(pred_values)) vs horizons
    try:
        if stats is None:
            raise ImportError("scipy.stats not available")
        
        # Filter out zeros and negative values for log
        abs_pred = np.abs(pred_values)
        log_pred = np.log(abs_pred + 1e-10)
        
        # Linear regression: log(pred) = a * horizon + b
        # Growth rate = a (slope)
        if len(horizons) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(horizons, log_pred)
            growth_rate = float(slope)
            
            # Detect divergence (growth rate > threshold) or convergence (growth rate < -threshold)
            divergence_detected = growth_rate > threshold_growth
            convergence_detected = growth_rate < -threshold_growth
        else:
            growth_rate = 0.0
            divergence_detected = False
            convergence_detected = False
    except Exception as e:
        logger.debug(f"Could not calculate growth rate: {e}")
        growth_rate = 0.0
        divergence_detected = False
        convergence_detected = False
    
    # Overall stability score: combine all factors
    # Penalize oscillations, divergence, and low smoothness
    stability_components = []
    
    # Smoothness component (0-1, higher = better)
    stability_components.append(smoothness_score)
    
    # Oscillation penalty (0-1, lower oscillation = better)
    oscillation_penalty = 1.0 - min(oscillation_magnitude, 1.0)
    stability_components.append(oscillation_penalty)
    
    # Growth/decay penalty (stable if growth_rate is near 0)
    growth_penalty = 1.0 - min(abs(growth_rate) / threshold_growth, 1.0) if threshold_growth > 0 else 1.0
    stability_components.append(growth_penalty)
    
    # Overall stability score (weighted average)
    stability_score = float(np.mean(stability_components)) if stability_components else 0.0
    
    # Interpretation
    if divergence_detected:
        stability_interpretation = 'diverging'
    elif convergence_detected:
        stability_interpretation = 'converging'
    elif oscillation_detected:
        stability_interpretation = 'oscillatory'
    elif stability_score > 0.7:
        stability_interpretation = 'stable'
    else:
        stability_interpretation = 'unstable'
    
    return {
        'stability_score': stability_score,
        'oscillation_detected': bool(oscillation_detected),
        'oscillation_magnitude': oscillation_magnitude,
        'oscillation_fraction': float(oscillation_fraction) if 'oscillation_fraction' in locals() else 0.0,
        'growth_rate': growth_rate,
        'smoothness_score': smoothness_score,
        'divergence_detected': bool(divergence_detected),
        'convergence_detected': bool(convergence_detected),
        'second_derivative_variance': second_derivative_variance,
        'stability_interpretation': stability_interpretation
    }


def calculate_forecast_skill_score(
    y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_pred: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_train: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
    baseline_method: str = 'persistence',
    target_series: Optional[Union[str, int]] = None
) -> Dict[str, float]:
    """Calculate forecast skill score comparing model to naive baseline.
    
    Forecast skill score measures how much better a model performs compared
    to a naive baseline (random walk/persistence or mean forecast). Skill score
    ranges from -inf to 1, where:
    - 1.0 = perfect forecast (zero error)
    - 0.0 = same performance as baseline
    - < 0.0 = worse than baseline
    - > 0.0 = better than baseline
    
    This metric is particularly useful for DDFM evaluation as it provides
    a standardized measure of forecast improvement relative to simple baselines.
    
    Parameters
    ----------
    y_true : pd.DataFrame, pd.Series, or np.ndarray
        True/actual values
    y_pred : pd.DataFrame, pd.Series, or np.ndarray
        Predicted/forecasted values
    y_train : pd.DataFrame, pd.Series, or np.ndarray, optional
        Training data used for baseline calculation
    baseline_method : str, default 'persistence'
        Baseline method: 'persistence' (random walk) or 'mean' (mean forecast)
    target_series : str or int, optional
        Target series name or index (for multivariate data)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'skill_score_mse': Skill score based on MSE (1 - MSE_model/MSE_baseline)
        - 'skill_score_mae': Skill score based on MAE (1 - MAE_model/MAE_baseline)
        - 'skill_score_rmse': Skill score based on RMSE (1 - RMSE_model/RMSE_baseline)
        - 'baseline_mse': Baseline MSE
        - 'baseline_mae': Baseline MAE
        - 'baseline_rmse': Baseline RMSE
        - 'model_mse': Model MSE
        - 'model_mae': Model MAE
        - 'model_rmse': Model RMSE
        - 'improvement_pct_mse': Percentage improvement over baseline (MSE)
        - 'improvement_pct_mae': Percentage improvement over baseline (MAE)
        - 'improvement_pct_rmse': Percentage improvement over baseline (RMSE)
    """
    logger = _module_logger
    
    # Calculate model metrics
    model_metrics = calculate_standardized_metrics(y_true, y_pred, y_train=y_train, target_series=target_series)
    
    if model_metrics['n_valid'] == 0:
        return {
            'skill_score_mse': np.nan,
            'skill_score_mae': np.nan,
            'skill_score_rmse': np.nan,
            'baseline_mse': np.nan,
            'baseline_mae': np.nan,
            'baseline_rmse': np.nan,
            'model_mse': np.nan,
            'model_mae': np.nan,
            'model_rmse': np.nan,
            'improvement_pct_mse': np.nan,
            'improvement_pct_mae': np.nan,
            'improvement_pct_rmse': np.nan
        }
    
    # Convert to numpy arrays for baseline calculation
    if isinstance(y_true, pd.DataFrame):
        y_true_arr = y_true.values
        if target_series is not None and isinstance(target_series, str):
            try:
                col_idx = y_true.columns.get_loc(target_series)
                y_true_arr = y_true_arr[:, col_idx:col_idx+1]
            except KeyError:
                y_true_arr = y_true_arr[:, 0:1]
    elif isinstance(y_true, pd.Series):
        y_true_arr = y_true.values.reshape(-1, 1)
    else:
        y_true_arr = np.asarray(y_true)
        if y_true_arr.ndim == 1:
            y_true_arr = y_true_arr.reshape(-1, 1)
    
    if isinstance(y_pred, pd.DataFrame):
        y_pred_arr = y_pred.values
        if target_series is not None and isinstance(target_series, str):
            try:
                col_idx = y_pred.columns.get_loc(target_series)
                y_pred_arr = y_pred_arr[:, col_idx:col_idx+1]
            except KeyError:
                y_pred_arr = y_pred_arr[:, 0:1]
    elif isinstance(y_pred, pd.Series):
        y_pred_arr = y_pred.values.reshape(-1, 1)
    else:
        try:
            import torch
            if isinstance(y_pred, torch.Tensor):
                y_pred_arr = y_pred.cpu().numpy()
            else:
                y_pred_arr = np.asarray(y_pred)
        except ImportError:
            y_pred_arr = np.asarray(y_pred)
        if y_pred_arr.ndim == 1:
            y_pred_arr = y_pred_arr.reshape(-1, 1)
    
    # Prepare training data for baseline
    if y_train is not None:
        if isinstance(y_train, pd.DataFrame):
            train_arr = y_train.values
            if target_series is not None and isinstance(target_series, str):
                try:
                    col_idx = y_train.columns.get_loc(target_series)
                    train_arr = train_arr[:, col_idx:col_idx+1]
                except KeyError:
                    train_arr = train_arr[:, 0:1]
        elif isinstance(y_train, pd.Series):
            train_arr = y_train.values.reshape(-1, 1)
        else:
            train_arr = np.asarray(y_train)
            if train_arr.ndim == 1:
                train_arr = train_arr.reshape(-1, 1)
    else:
        train_arr = y_true_arr
    
    # Filter valid values
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    if np.sum(mask) == 0:
        return {
            'skill_score_mse': np.nan,
            'skill_score_mae': np.nan,
            'skill_score_rmse': np.nan,
            'baseline_mse': np.nan,
            'baseline_mae': np.nan,
            'baseline_rmse': np.nan,
            'model_mse': np.nan,
            'model_mae': np.nan,
            'model_rmse': np.nan,
            'improvement_pct_mse': np.nan,
            'improvement_pct_mae': np.nan,
            'improvement_pct_rmse': np.nan
        }
    
    y_true_valid = y_true_arr[mask].flatten()
    y_pred_valid = y_pred_arr[mask].flatten()
    
    # Calculate baseline predictions
    if baseline_method == 'persistence':
        # Random walk: predict last training value
        if len(train_arr) > 0:
            train_valid = train_arr[np.isfinite(train_arr)].flatten()
            if len(train_valid) > 0:
                baseline_pred = np.full_like(y_true_valid, train_valid[-1])
            else:
                baseline_pred = np.full_like(y_true_valid, np.nanmean(y_true_valid))
        else:
            baseline_pred = np.full_like(y_true_valid, np.nanmean(y_true_valid))
    elif baseline_method == 'mean':
        # Mean forecast: predict mean of training data
        train_valid = train_arr[np.isfinite(train_arr)].flatten()
        if len(train_valid) > 0:
            baseline_pred = np.full_like(y_true_valid, np.nanmean(train_valid))
        else:
            baseline_pred = np.full_like(y_true_valid, np.nanmean(y_true_valid))
    else:
        raise ValueError(f"Unknown baseline_method: {baseline_method}. Use 'persistence' or 'mean'")
    
    # Calculate baseline metrics
    baseline_mse = float(np.mean((y_true_valid - baseline_pred) ** 2))
    baseline_mae = float(np.mean(np.abs(y_true_valid - baseline_pred)))
    baseline_rmse = float(np.sqrt(baseline_mse))
    
    # Model metrics (already calculated)
    model_mse = model_metrics['MSE']
    model_mae = model_metrics['MAE']
    model_rmse = model_metrics['RMSE']
    
    # Calculate skill scores: 1 - (model_error / baseline_error)
    # Higher skill score = better performance
    if baseline_mse > 1e-10:
        skill_score_mse = 1.0 - (model_mse / baseline_mse)
        improvement_pct_mse = (baseline_mse - model_mse) / baseline_mse * 100
    else:
        skill_score_mse = np.nan
        improvement_pct_mse = np.nan
    
    if baseline_mae > 1e-10:
        skill_score_mae = 1.0 - (model_mae / baseline_mae)
        improvement_pct_mae = (baseline_mae - model_mae) / baseline_mae * 100
    else:
        skill_score_mae = np.nan
        improvement_pct_mae = np.nan
    
    if baseline_rmse > 1e-10:
        skill_score_rmse = 1.0 - (model_rmse / baseline_rmse)
        improvement_pct_rmse = (baseline_rmse - model_rmse) / baseline_rmse * 100
    else:
        skill_score_rmse = np.nan
        improvement_pct_rmse = np.nan
    
    return {
        'skill_score_mse': float(skill_score_mse) if not np.isnan(skill_score_mse) else np.nan,
        'skill_score_mae': float(skill_score_mae) if not np.isnan(skill_score_mae) else np.nan,
        'skill_score_rmse': float(skill_score_rmse) if not np.isnan(skill_score_rmse) else np.nan,
        'baseline_mse': baseline_mse,
        'baseline_mae': baseline_mae,
        'baseline_rmse': baseline_rmse,
        'model_mse': model_mse,
        'model_mae': model_mae,
        'model_rmse': model_rmse,
        'improvement_pct_mse': float(improvement_pct_mse) if not np.isnan(improvement_pct_mse) else np.nan,
        'improvement_pct_mae': float(improvement_pct_mae) if not np.isnan(improvement_pct_mae) else np.nan,
        'improvement_pct_rmse': float(improvement_pct_rmse) if not np.isnan(improvement_pct_rmse) else np.nan
    }


def calculate_quantile_based_metrics(
    y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_pred: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_train: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
    target_series: Optional[Union[str, int]] = None,
    quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]
) -> Dict[str, float]:
    """Calculate quantile-based error metrics for DDFM evaluation.
    
    Quantile-based metrics provide more robust evaluation than mean-based metrics,
    especially for volatile horizons or when error distributions are skewed.
    This is particularly useful for DDFM evaluation where some horizons may have
    extreme errors due to numerical instability or model issues.
    
    Parameters
    ----------
    y_true : pd.DataFrame, pd.Series, or np.ndarray
        True/actual values
    y_pred : pd.DataFrame, pd.Series, or np.ndarray
        Predicted/forecasted values
    y_train : pd.DataFrame, pd.Series, or np.ndarray, optional
        Training data used for standardization
    target_series : str or int, optional
        Target series name or index (for multivariate data)
    quantiles : list of float, default [0.1, 0.25, 0.5, 0.75, 0.9]
        Quantiles to calculate (0.5 = median, 0.25/0.75 = quartiles)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'quantile_sMAE_{q}': Quantile-based sMAE for each quantile q
        - 'quantile_sMSE_{q}': Quantile-based sMSE for each quantile q
        - 'iqr_sMAE': Interquartile range of absolute errors (normalized)
        - 'tail_ratio': Ratio of 90th to 10th percentile errors (measures tail heaviness)
        - 'n_valid': Number of valid observations
    """
    logger = _module_logger
    
    # Calculate base metrics to get standardization factor
    base_metrics = calculate_standardized_metrics(y_true, y_pred, y_train=y_train, target_series=target_series)
    
    if base_metrics['n_valid'] == 0:
        result = {'n_valid': 0}
        for q in quantiles:
            result[f'quantile_sMAE_{q:.2f}'] = np.nan
            result[f'quantile_sMSE_{q:.2f}'] = np.nan
        result['iqr_sMAE'] = np.nan
        result['tail_ratio'] = np.nan
        return result
    
    # Convert to numpy arrays (same pattern as calculate_standardized_metrics)
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
        try:
            import torch
            if isinstance(y_pred, torch.Tensor):
                y_pred_arr = y_pred.cpu().numpy()
            else:
                y_pred_arr = np.asarray(y_pred)
        except ImportError:
            y_pred_arr = np.asarray(y_pred)
    
    if y_pred_arr.ndim == 1:
        y_pred_arr = y_pred_arr.reshape(-1, 1)
    
    # Get target series index
    target_idx = None
    if target_series is not None:
        if isinstance(target_series, str) and target_series in columns:
            target_idx = columns.index(target_series)
        elif isinstance(target_series, int) and 0 <= target_series < len(columns):
            target_idx = target_series
    
    # Calculate errors
    errors = y_pred_arr - y_true_arr
    
    # Filter valid values
    if target_idx is not None:
        errors_series = errors[:, target_idx]
    else:
        errors_series = errors.flatten()
    
    # Filter finite values
    mask = np.isfinite(errors_series)
    errors_valid = errors_series[mask]
    
    if len(errors_valid) == 0:
        result = {'n_valid': 0}
        for q in quantiles:
            result[f'quantile_sMAE_{q:.2f}'] = np.nan
            result[f'quantile_sMSE_{q:.2f}'] = np.nan
        result['iqr_sMAE'] = np.nan
        result['tail_ratio'] = np.nan
        return result
    
    # Get standardization factor (sigma from training data)
    sigma = base_metrics.get('sigma', 1.0)
    if sigma is None or np.isnan(sigma) or sigma <= 0:
        sigma = 1.0
    
    # Calculate absolute and squared errors
    abs_errors = np.abs(errors_valid)
    squared_errors = errors_valid ** 2
    
    # Calculate quantile-based metrics
    result = {'n_valid': int(len(errors_valid))}
    
    for q in quantiles:
        # Quantile of absolute errors
        q_ae = np.percentile(abs_errors, q * 100)
        result[f'quantile_sMAE_{q:.2f}'] = float(q_ae / sigma) if sigma > 0 else np.nan
        
        # Quantile of squared errors
        q_se = np.percentile(squared_errors, q * 100)
        result[f'quantile_sMSE_{q:.2f}'] = float(q_se / (sigma ** 2)) if sigma > 0 else np.nan
    
    # IQR of absolute errors (normalized)
    q75_ae = np.percentile(abs_errors, 75)
    q25_ae = np.percentile(abs_errors, 25)
    iqr_ae = q75_ae - q25_ae
    result['iqr_sMAE'] = float(iqr_ae / sigma) if sigma > 0 else np.nan
    
    # Tail ratio (90th / 10th percentile) - measures tail heaviness
    q90_ae = np.percentile(abs_errors, 90)
    q10_ae = np.percentile(abs_errors, 10)
    if q10_ae > 0:
        result['tail_ratio'] = float(q90_ae / q10_ae)
    else:
        result['tail_ratio'] = np.nan
    
    return result


def compare_factor_loadings(
    dfm_loadings: Optional[np.ndarray],
    ddfm_loadings: Optional[np.ndarray],
    series_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Compare factor loadings between DFM and DDFM to detect linear collapse.
    
    If DDFM encoder is learning only linear features (linear collapse), the
    factor loadings should be highly similar to DFM loadings. This function
    quantifies the similarity to help detect linear collapse.
    
    Parameters
    ----------
    dfm_loadings : np.ndarray, optional
        DFM factor loadings matrix (N × k) where N = number of series, k = number of factors
    ddfm_loadings : np.ndarray, optional
        DDFM factor loadings matrix (N × k) - extracted from encoder output
    series_names : list of str, optional
        Names of series (for reporting which series have similar loadings)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'loading_similarity': Overall similarity score (0-1, higher = more similar)
        - 'factor_correlations': Correlation between DFM and DDFM factors (per factor)
        - 'loading_cosine_similarity': Cosine similarity of loading vectors
        - 'linear_collapse_risk': Risk score based on loading similarity (0-1, higher = more risk)
        - 'series_similarity': Per-series loading similarity (if series_names provided)
    """
    logger = _module_logger
    
    if dfm_loadings is None or ddfm_loadings is None:
        logger.warning("Cannot compare factor loadings: one or both are None")
        return {
            'loading_similarity': np.nan,
            'factor_correlations': [],
            'loading_cosine_similarity': np.nan,
            'linear_collapse_risk': np.nan,
            'series_similarity': {}
        }
    
    dfm_loadings = np.asarray(dfm_loadings)
    ddfm_loadings = np.asarray(ddfm_loadings)
    
    # Ensure same shape
    if dfm_loadings.shape != ddfm_loadings.shape:
        logger.warning(
            f"Factor loading shapes don't match: DFM {dfm_loadings.shape} vs DDFM {ddfm_loadings.shape}"
        )
        # Try to align by taking minimum dimensions
        min_rows = min(dfm_loadings.shape[0], ddfm_loadings.shape[0])
        min_cols = min(dfm_loadings.shape[1], ddfm_loadings.shape[1])
        dfm_loadings = dfm_loadings[:min_rows, :min_cols]
        ddfm_loadings = ddfm_loadings[:min_rows, :min_cols]
    
    # Filter finite values
    mask = np.isfinite(dfm_loadings) & np.isfinite(ddfm_loadings)
    if np.sum(mask) == 0:
        logger.warning("No valid factor loadings to compare")
        return {
            'loading_similarity': np.nan,
            'factor_correlations': [],
            'loading_cosine_similarity': np.nan,
            'linear_collapse_risk': np.nan,
            'series_similarity': {}
        }
    
    # Calculate per-factor correlations
    factor_correlations = []
    num_factors = dfm_loadings.shape[1]
    
    for k in range(num_factors):
        dfm_factor = dfm_loadings[:, k]
        ddfm_factor = ddfm_loadings[:, k]
        
        # Filter finite values for this factor
        factor_mask = np.isfinite(dfm_factor) & np.isfinite(ddfm_factor)
        if np.sum(factor_mask) > 1:
            corr = np.corrcoef(dfm_factor[factor_mask], ddfm_factor[factor_mask])[0, 1]
            if not np.isnan(corr):
                factor_correlations.append(float(corr))
            else:
                factor_correlations.append(0.0)
        else:
            factor_correlations.append(0.0)
    
    # Overall loading similarity (mean of factor correlations)
    if len(factor_correlations) > 0:
        loading_similarity = float(np.mean(factor_correlations))
    else:
        loading_similarity = 0.0
    
    # Cosine similarity of flattened loading matrices
    dfm_flat = dfm_loadings[mask].flatten()
    ddfm_flat = ddfm_loadings[mask].flatten()
    
    if len(dfm_flat) > 0 and len(ddfm_flat) > 0:
        # Normalize vectors
        dfm_norm = np.linalg.norm(dfm_flat)
        ddfm_norm = np.linalg.norm(ddfm_flat)
        
        if dfm_norm > 0 and ddfm_norm > 0:
            cosine_sim = float(np.dot(dfm_flat, ddfm_flat) / (dfm_norm * ddfm_norm))
        else:
            cosine_sim = 0.0
    else:
        cosine_sim = 0.0
    
    # Linear collapse risk: high similarity = high risk
    # Combine correlation and cosine similarity
    linear_collapse_risk = float((loading_similarity * 0.7 + abs(cosine_sim) * 0.3))
    
    # Per-series similarity (if series names provided)
    series_similarity = {}
    if series_names is not None and len(series_names) == dfm_loadings.shape[0]:
        for i, series_name in enumerate(series_names):
            dfm_series = dfm_loadings[i, :]
            ddfm_series = ddfm_loadings[i, :]
            
            series_mask = np.isfinite(dfm_series) & np.isfinite(ddfm_series)
            if np.sum(series_mask) > 1:
                series_corr = np.corrcoef(dfm_series[series_mask], ddfm_series[series_mask])[0, 1]
                if not np.isnan(series_corr):
                    series_similarity[series_name] = float(series_corr)
                else:
                    series_similarity[series_name] = 0.0
            else:
                series_similarity[series_name] = 0.0
    
    return {
        'loading_similarity': loading_similarity,
        'factor_correlations': factor_correlations,
        'loading_cosine_similarity': float(cosine_sim),
        'linear_collapse_risk': linear_collapse_risk,
        'series_similarity': series_similarity
    }


def calculate_information_gain(
    y_true: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_pred_ddfm: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_pred_dfm: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_train: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
    target_series: Optional[Union[str, int]] = None,
    method: str = 'kl_divergence'
) -> Dict[str, float]:
    """Calculate information gain of DDFM over DFM.
    
    Information gain measures how much additional information DDFM provides
    compared to DFM. This metric helps quantify the value of nonlinear
    features learned by DDFM encoder.
    
    Two methods are available:
    1. 'kl_divergence': KL divergence between error distributions
    2. 'mutual_information': Mutual information between predictions and true values
    
    Parameters
    ----------
    y_true : pd.DataFrame, pd.Series, or np.ndarray
        True/actual values
    y_pred_ddfm : pd.DataFrame, pd.Series, or np.ndarray
        DDFM predicted values
    y_pred_dfm : pd.DataFrame, pd.Series, or np.ndarray
        DFM predicted values
    y_train : pd.DataFrame, pd.Series, or np.ndarray, optional
        Training data (for standardization)
    target_series : str or int, optional
        Target series name or index (for multivariate data)
    method : str, default 'kl_divergence'
        Method: 'kl_divergence' or 'mutual_information'
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'information_gain': Information gain value (higher = more information)
        - 'ddfm_error_entropy': Entropy of DDFM error distribution
        - 'dfm_error_entropy': Entropy of DFM error distribution
        - 'kl_divergence': KL divergence (if method='kl_divergence')
        - 'mutual_information_ddfm': Mutual information for DDFM (if method='mutual_information')
        - 'mutual_information_dfm': Mutual information for DFM (if method='mutual_information')
    """
    logger = _module_logger
    
    # Calculate errors for both models
    ddfm_metrics = calculate_standardized_metrics(y_true, y_pred_ddfm, y_train=y_train, target_series=target_series)
    dfm_metrics = calculate_standardized_metrics(y_true, y_pred_dfm, y_train=y_train, target_series=target_series)
    
    if ddfm_metrics['n_valid'] == 0 or dfm_metrics['n_valid'] == 0:
        return {
            'information_gain': np.nan,
            'ddfm_error_entropy': np.nan,
            'dfm_error_entropy': np.nan,
            'kl_divergence': np.nan,
            'mutual_information_ddfm': np.nan,
            'mutual_information_dfm': np.nan
        }
    
    # Convert to numpy arrays
    if isinstance(y_true, pd.DataFrame):
        y_true_arr = y_true.values
        if target_series is not None and isinstance(target_series, str):
            try:
                col_idx = y_true.columns.get_loc(target_series)
                y_true_arr = y_true_arr[:, col_idx:col_idx+1]
            except KeyError:
                y_true_arr = y_true_arr[:, 0:1]
    elif isinstance(y_true, pd.Series):
        y_true_arr = y_true.values.reshape(-1, 1)
    else:
        y_true_arr = np.asarray(y_true)
        if y_true_arr.ndim == 1:
            y_true_arr = y_true_arr.reshape(-1, 1)
    
    # Convert predictions
    def _convert_pred(pred):
        if isinstance(pred, pd.DataFrame):
            arr = pred.values
            if target_series is not None and isinstance(target_series, str):
                try:
                    col_idx = pred.columns.get_loc(target_series)
                    arr = arr[:, col_idx:col_idx+1]
                except KeyError:
                    arr = arr[:, 0:1]
            return arr
        elif isinstance(pred, pd.Series):
            return pred.values.reshape(-1, 1)
        else:
            try:
                import torch
                if isinstance(pred, torch.Tensor):
                    return pred.cpu().numpy()
            except ImportError:
                pass
            arr = np.asarray(pred)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            return arr
    
    y_pred_ddfm_arr = _convert_pred(y_pred_ddfm)
    y_pred_dfm_arr = _convert_pred(y_pred_dfm)
    
    # Filter valid values
    mask = (np.isfinite(y_true_arr) & np.isfinite(y_pred_ddfm_arr) & 
            np.isfinite(y_pred_dfm_arr))
    
    if np.sum(mask) == 0:
        return {
            'information_gain': np.nan,
            'ddfm_error_entropy': np.nan,
            'dfm_error_entropy': np.nan,
            'kl_divergence': np.nan,
            'mutual_information_ddfm': np.nan,
            'mutual_information_dfm': np.nan
        }
    
    y_true_valid = y_true_arr[mask].flatten()
    y_pred_ddfm_valid = y_pred_ddfm_arr[mask].flatten()
    y_pred_dfm_valid = y_pred_dfm_arr[mask].flatten()
    
    # Calculate errors
    errors_ddfm = y_true_valid - y_pred_ddfm_valid
    errors_dfm = y_true_valid - y_pred_dfm_valid
    
    if method == 'kl_divergence':
        # Calculate KL divergence between error distributions
        # Use histogram-based approach for continuous distributions
        try:
            # Create histograms for error distributions
            min_err = min(np.min(errors_ddfm), np.min(errors_dfm))
            max_err = max(np.max(errors_ddfm), np.max(errors_dfm))
            bins = np.linspace(min_err, max_err, 50)
            
            hist_ddfm, _ = np.histogram(errors_ddfm, bins=bins, density=True)
            hist_dfm, _ = np.histogram(errors_dfm, bins=bins, density=True)
            
            # Add small epsilon to avoid log(0)
            eps = 1e-10
            hist_ddfm = hist_ddfm + eps
            hist_dfm = hist_dfm + eps
            
            # Normalize
            hist_ddfm = hist_ddfm / np.sum(hist_ddfm)
            hist_dfm = hist_dfm / np.sum(hist_dfm)
            
            # Calculate KL divergence: D_KL(P||Q) = sum(P * log(P/Q))
            # P = DDFM, Q = DFM
            kl_div = np.sum(hist_ddfm * np.log(hist_ddfm / hist_dfm))
            
            # Information gain: higher KL divergence = more different distributions = more information
            information_gain = float(kl_div)
            
            # Calculate entropy of error distributions
            entropy_ddfm = -np.sum(hist_ddfm * np.log(hist_ddfm + eps))
            entropy_dfm = -np.sum(hist_dfm * np.log(hist_dfm + eps))
            
            return {
                'information_gain': information_gain,
                'ddfm_error_entropy': float(entropy_ddfm),
                'dfm_error_entropy': float(entropy_dfm),
                'kl_divergence': information_gain,
                'mutual_information_ddfm': np.nan,
                'mutual_information_dfm': np.nan
            }
        except Exception as e:
            logger.debug(f"Could not calculate KL divergence: {e}")
            return {
                'information_gain': np.nan,
                'ddfm_error_entropy': np.nan,
                'dfm_error_entropy': np.nan,
                'kl_divergence': np.nan,
                'mutual_information_ddfm': np.nan,
                'mutual_information_dfm': np.nan
            }
    
    elif method == 'mutual_information':
        # Calculate mutual information between predictions and true values
        # This measures how much information predictions contain about true values
        try:
            from scipy.stats import entropy
            from sklearn.metrics import mutual_info_score
            
            # Discretize for mutual information calculation
            n_bins = 20
            true_discrete = np.digitize(y_true_valid, np.linspace(np.min(y_true_valid), np.max(y_true_valid), n_bins))
            ddfm_discrete = np.digitize(y_pred_ddfm_valid, np.linspace(np.min(y_pred_ddfm_valid), np.max(y_pred_ddfm_valid), n_bins))
            dfm_discrete = np.digitize(y_pred_dfm_valid, np.linspace(np.min(y_pred_dfm_valid), np.max(y_pred_dfm_valid), n_bins))
            
            mi_ddfm = mutual_info_score(true_discrete, ddfm_discrete)
            mi_dfm = mutual_info_score(true_discrete, dfm_discrete)
            
            # Information gain: difference in mutual information
            information_gain = float(mi_ddfm - mi_dfm)
            
            # Calculate entropy of error distributions
            hist_ddfm, _ = np.histogram(errors_ddfm, bins=50, density=True)
            hist_dfm, _ = np.histogram(errors_dfm, bins=50, density=True)
            eps = 1e-10
            hist_ddfm = (hist_ddfm + eps) / (np.sum(hist_ddfm) + eps * len(hist_ddfm))
            hist_dfm = (hist_dfm + eps) / (np.sum(hist_dfm) + eps * len(hist_dfm))
            
            entropy_ddfm = entropy(hist_ddfm)
            entropy_dfm = entropy(hist_dfm)
            
            return {
                'information_gain': information_gain,
                'ddfm_error_entropy': float(entropy_ddfm),
                'dfm_error_entropy': float(entropy_dfm),
                'kl_divergence': np.nan,
                'mutual_information_ddfm': float(mi_ddfm),
                'mutual_information_dfm': float(mi_dfm)
            }
        except (ImportError, Exception) as e:
            logger.debug(f"Could not calculate mutual information: {e}")
            return {
                'information_gain': np.nan,
                'ddfm_error_entropy': np.nan,
                'dfm_error_entropy': np.nan,
                'kl_divergence': np.nan,
                'mutual_information_ddfm': np.nan,
                'mutual_information_dfm': np.nan
            }
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'kl_divergence' or 'mutual_information'")


def calculate_nonlinearity_score(
    aggregated_results: pd.DataFrame,
    target: Optional[str] = None,
    dfm_model: Optional[str] = 'DFM',
    ddfm_model: Optional[str] = 'DDFM'
) -> Dict[str, Any]:
    """Calculate nonlinearity score for DDFM predictions compared to DFM baseline.
    
    This function quantifies how nonlinear DDFM predictions are by comparing
    prediction patterns across horizons. A high nonlinearity score indicates
    that DDFM is learning different (nonlinear) patterns from DFM, even if
    overall performance metrics are similar.
    
    The nonlinearity score is calculated using:
    1. Prediction pattern divergence: How different are DDFM vs DFM prediction patterns across horizons
    2. Error pattern nonlinearity: Whether DDFM errors show nonlinear patterns (non-constant variance, skewness)
    3. Horizon interaction effects: Whether DDFM shows different improvement patterns across horizons
    
    This complements the linearity detection by providing a positive measure
    of nonlinearity (rather than just detecting linear collapse).
    
    Parameters
    ----------
    aggregated_results : pd.DataFrame
        Aggregated results DataFrame with columns: target, model, horizon, sMSE, sMAE, sRMSE
    target : str, optional
        Target series to analyze. If None, analyzes all targets.
    dfm_model : str, default 'DFM'
        Name of DFM model in results
    ddfm_model : str, default 'DDFM'
        Name of DDFM model in results
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'nonlinearity_score': Overall nonlinearity score (0-1, higher = more nonlinear)
        - 'pattern_divergence': How different prediction patterns are (0-1)
        - 'error_nonlinearity': Nonlinearity in error patterns (0-1)
        - 'horizon_interaction': Evidence of horizon-specific nonlinear effects (0-1)
        - 'target_scores': Per-target detailed scores
        - 'interpretation': Interpretation of scores
    """
    logger = _module_logger
    
    if aggregated_results.empty:
        logger.warning("Empty aggregated_results, cannot calculate nonlinearity score")
        return {'error': 'Empty results'}
    
    # Filter for DFM and DDFM models
    dfm_ddfm = aggregated_results[
        aggregated_results['model'].isin([dfm_model, ddfm_model])
    ].copy()
    
    if target:
        dfm_ddfm = dfm_ddfm[dfm_ddfm['target'] == target]
    
    if dfm_ddfm.empty:
        logger.warning(f"No DFM/DDFM results found{' for ' + target if target else ''}")
        return {'error': 'No DFM/DDFM results'}
    
    target_scores = {}
    
    for tgt in dfm_ddfm['target'].unique():
        target_data = dfm_ddfm[dfm_ddfm['target'] == tgt]
        
        ddfm_data = target_data[target_data['model'] == ddfm_model].sort_values('horizon')
        dfm_data = target_data[target_data['model'] == dfm_model].sort_values('horizon')
        
        # Extract metrics
        ddfm_smae = ddfm_data['sMAE'].values
        ddfm_smse = ddfm_data['sMSE'].values
        ddfm_horizons = ddfm_data['horizon'].values
        
        dfm_smae = dfm_data['sMAE'].values
        dfm_smse = dfm_data['sMSE'].values
        dfm_horizons = dfm_data['horizon'].values
        
        # Find common horizons
        common_horizons = np.intersect1d(ddfm_horizons, dfm_horizons)
        
        if len(common_horizons) < 3:
            target_scores[tgt] = {
                'nonlinearity_score': np.nan,
                'pattern_divergence': np.nan,
                'error_nonlinearity': np.nan,
                'horizon_interaction': np.nan,
                'n_horizons': len(common_horizons),
                'error': 'Insufficient horizons for analysis'
            }
            continue
        
        # Extract valid metrics for common horizons
        ddfm_smae_valid = []
        ddfm_smse_valid = []
        dfm_smae_valid = []
        dfm_smse_valid = []
        horizons_valid = []
        
        for h in common_horizons:
            ddfm_idx = np.where(ddfm_horizons == h)[0]
            dfm_idx = np.where(dfm_horizons == h)[0]
            
            if len(ddfm_idx) > 0 and len(dfm_idx) > 0:
                ddfm_smae_val = ddfm_smae[ddfm_idx[0]]
                ddfm_smse_val = ddfm_smse[ddfm_idx[0]]
                dfm_smae_val = dfm_smae[dfm_idx[0]]
                dfm_smse_val = dfm_smse[dfm_idx[0]]
                
                if not (np.isnan(ddfm_smae_val) or np.isnan(dfm_smae_val) or
                       np.isnan(ddfm_smse_val) or np.isnan(dfm_smse_val)):
                    ddfm_smae_valid.append(ddfm_smae_val)
                    ddfm_smse_valid.append(ddfm_smse_val)
                    dfm_smae_valid.append(dfm_smae_val)
                    dfm_smse_valid.append(dfm_smse_val)
                    horizons_valid.append(h)
        
        if len(ddfm_smae_valid) < 3:
            target_scores[tgt] = {
                'nonlinearity_score': np.nan,
                'pattern_divergence': np.nan,
                'error_nonlinearity': np.nan,
                'horizon_interaction': np.nan,
                'n_horizons': len(ddfm_smae_valid),
                'error': 'Insufficient valid horizons'
            }
            continue
        
        ddfm_smae_arr = np.array(ddfm_smae_valid)
        ddfm_smse_arr = np.array(ddfm_smse_valid)
        dfm_smae_arr = np.array(dfm_smae_valid)
        dfm_smse_arr = np.array(dfm_smse_valid)
        horizons_arr = np.array(horizons_valid)
        
        # 1. Pattern Divergence: How different are prediction patterns?
        # Calculate correlation between DDFM and DFM error patterns
        # Low correlation = high divergence = high nonlinearity
        try:
            pattern_corr_smae = np.corrcoef(ddfm_smae_arr, dfm_smae_arr)[0, 1]
            pattern_corr_smse = np.corrcoef(ddfm_smse_arr, dfm_smse_arr)[0, 1]
            
            if np.isnan(pattern_corr_smae):
                pattern_corr_smae = 1.0  # If cannot calculate, assume high correlation (low nonlinearity)
            if np.isnan(pattern_corr_smse):
                pattern_corr_smse = 1.0
            
            # Pattern divergence = 1 - correlation (higher = more divergent = more nonlinear)
            pattern_divergence = 1.0 - (abs(pattern_corr_smae) + abs(pattern_corr_smse)) / 2.0
            pattern_divergence = max(0.0, min(1.0, pattern_divergence))
        except Exception:
            pattern_divergence = 0.0
        
        # 2. Error Nonlinearity: Do DDFM errors show nonlinear patterns?
        # Calculate variance in improvement ratio across horizons
        # High variance = different improvement at different horizons = nonlinear behavior
        improvements = []
        for i in range(len(ddfm_smae_arr)):
            if dfm_smae_arr[i] > 1e-10:
                imp = (dfm_smae_arr[i] - ddfm_smae_arr[i]) / dfm_smae_arr[i]
                improvements.append(imp)
        
        if len(improvements) > 1:
            improvement_std = np.std(improvements)
            improvement_mean = np.abs(np.mean(improvements))
            
            # Error nonlinearity = normalized std of improvements
            # Higher std relative to mean = more nonlinear (different behavior at different horizons)
            if improvement_mean > 1e-10:
                error_nonlinearity = min(1.0, improvement_std / (improvement_mean + 1e-10))
            else:
                # If mean improvement is near zero, check if there's variation anyway
                error_nonlinearity = min(1.0, improvement_std * 10.0)  # Scale up small std
        else:
            error_nonlinearity = 0.0
        
        # 3. Horizon Interaction: Evidence of horizon-specific nonlinear effects
        # Calculate whether improvement varies systematically with horizon
        # (e.g., better at short horizons, worse at long horizons)
        if len(improvements) >= 3:
            try:
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(horizons_arr, improvements)
                
                # If improvement varies significantly with horizon (high |slope| or high |r|),
                # this indicates horizon-specific nonlinear effects
                horizon_interaction = min(1.0, abs(slope) * 10.0 + abs(r_value))
                horizon_interaction = max(0.0, min(1.0, horizon_interaction))
            except (ImportError, Exception):
                # Fallback: use variance of improvements as proxy
                horizon_interaction = min(1.0, np.std(improvements) * 5.0) if len(improvements) > 1 else 0.0
        else:
            horizon_interaction = 0.0
        
        # Combine scores (weighted average)
        # Pattern divergence is most important (direct measure of difference)
        # Error nonlinearity and horizon interaction provide additional evidence
        nonlinearity_score = (
            0.5 * pattern_divergence +
            0.3 * error_nonlinearity +
            0.2 * horizon_interaction
        )
        
        target_scores[tgt] = {
            'nonlinearity_score': float(nonlinearity_score),
            'pattern_divergence': float(pattern_divergence),
            'error_nonlinearity': float(error_nonlinearity),
            'horizon_interaction': float(horizon_interaction),
            'pattern_correlation_smae': float(pattern_corr_smae) if 'pattern_corr_smae' in locals() else np.nan,
            'pattern_correlation_smse': float(pattern_corr_smse) if 'pattern_corr_smse' in locals() else np.nan,
            'improvement_std': float(improvement_std) if 'improvement_std' in locals() else np.nan,
            'improvement_mean': float(improvement_mean) if 'improvement_mean' in locals() else np.nan,
            'n_horizons': len(ddfm_smae_valid)
        }
    
    # Calculate overall summary
    valid_scores = [s['nonlinearity_score'] for s in target_scores.values() 
                   if not np.isnan(s.get('nonlinearity_score', np.nan))]
    
    if len(valid_scores) > 0:
        overall_nonlinearity = np.mean(valid_scores)
        overall_pattern_divergence = np.mean([s.get('pattern_divergence', 0.0) 
                                              for s in target_scores.values() 
                                              if not np.isnan(s.get('pattern_divergence', np.nan))])
        overall_error_nonlinearity = np.mean([s.get('error_nonlinearity', 0.0) 
                                             for s in target_scores.values() 
                                             if not np.isnan(s.get('error_nonlinearity', np.nan))])
    else:
        overall_nonlinearity = np.nan
        overall_pattern_divergence = np.nan
        overall_error_nonlinearity = np.nan
    
    # Interpretation
    if not np.isnan(overall_nonlinearity):
        if overall_nonlinearity > 0.7:
            interpretation = "HIGH: DDFM shows strong nonlinear behavior, learning different patterns from DFM"
        elif overall_nonlinearity > 0.4:
            interpretation = "MODERATE: DDFM shows some nonlinear behavior, but patterns are somewhat similar to DFM"
        elif overall_nonlinearity > 0.2:
            interpretation = "LOW: DDFM shows minimal nonlinear behavior, patterns are similar to DFM (possible linear collapse)"
        else:
            interpretation = "VERY LOW: DDFM patterns are nearly identical to DFM (likely linear collapse)"
    else:
        interpretation = "Cannot determine: insufficient data"
    
    return {
        'nonlinearity_score': float(overall_nonlinearity) if not np.isnan(overall_nonlinearity) else np.nan,
        'pattern_divergence': float(overall_pattern_divergence) if not np.isnan(overall_pattern_divergence) else np.nan,
        'error_nonlinearity': float(overall_error_nonlinearity) if not np.isnan(overall_error_nonlinearity) else np.nan,
        'target_scores': target_scores,
        'interpretation': interpretation
    }


def calculate_relative_skill_assessment(
    aggregated_results: pd.DataFrame,
    target: Optional[str] = None,
    dfm_model: Optional[str] = 'DFM',
    ddfm_model: Optional[str] = 'DDFM',
    baseline_model: Optional[str] = 'VAR'
) -> Dict[str, Any]:
    """Calculate relative skill assessment for DDFM using error metrics.
    
    This function provides a skill-like assessment when actual predictions
    are not available. It compares DDFM performance relative to DFM and a
    baseline model (typically VAR) using error metrics, providing insights
    similar to forecast skill score but working with aggregated error data.
    
    The relative skill metric assesses:
    1. DDFM vs DFM: How much better/worse is DDFM compared to DFM baseline
    2. DDFM vs VAR: How much better/worse is DDFM compared to VAR baseline
    3. Skill consistency: How consistent is skill across horizons
    4. Horizon-specific skill: Skill assessment at each forecast horizon
    
    This complements forecast_skill_score() by working with error metrics
    when predictions are not available in aggregated_results.csv.
    
    Parameters
    ----------
    aggregated_results : pd.DataFrame
        Aggregated results DataFrame with columns: target, model, horizon, sMSE, sMAE, sRMSE
    target : str, optional
        Target series to analyze. If None, analyzes all targets.
    dfm_model : str, default 'DFM'
        Name of DFM model in results (used as intermediate baseline)
    ddfm_model : str, default 'DDFM'
        Name of DDFM model in results
    baseline_model : str, default 'VAR'
        Name of baseline model for skill assessment (typically VAR or ARIMA)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'target_scores': Per-target detailed skill assessments
        - 'overall_skill': Overall skill summary across all targets
        - 'skill_interpretation': Interpretation of skill scores
    """
    logger = _module_logger
    
    if aggregated_results.empty:
        logger.warning("Empty aggregated_results, cannot calculate relative skill assessment")
        return {'error': 'Empty results'}
    
    # Filter for relevant models
    relevant_models = [dfm_model, ddfm_model]
    if baseline_model:
        relevant_models.append(baseline_model)
    
    model_data = aggregated_results[
        aggregated_results['model'].isin(relevant_models)
    ].copy()
    
    if target:
        model_data = model_data[model_data['target'] == target]
    
    if model_data.empty:
        logger.warning(f"No relevant model results found{' for ' + target if target else ''}")
        return {'error': 'No relevant model results'}
    
    target_scores = {}
    
    for tgt in model_data['target'].unique():
        target_data = model_data[model_data['target'] == tgt]
        
        ddfm_data = target_data[target_data['model'] == ddfm_model].sort_values('horizon')
        dfm_data = target_data[target_data['model'] == dfm_model].sort_values('horizon')
        baseline_data = target_data[target_data['model'] == baseline_model].sort_values('horizon') if baseline_model else None
        
        # Extract metrics
        ddfm_smae = ddfm_data['sMAE'].values
        ddfm_smse = ddfm_data['sMSE'].values
        ddfm_horizons = ddfm_data['horizon'].values
        
        dfm_smae = dfm_data['sMAE'].values
        dfm_smse = dfm_data['sMSE'].values
        dfm_horizons = dfm_data['horizon'].values
        
        baseline_smae = None
        baseline_smse = None
        baseline_horizons = None
        if baseline_data is not None and len(baseline_data) > 0:
            baseline_smae = baseline_data['sMAE'].values
            baseline_smse = baseline_data['sMSE'].values
            baseline_horizons = baseline_data['horizon'].values
        
        # Find common horizons
        common_horizons = np.intersect1d(ddfm_horizons, dfm_horizons)
        if baseline_horizons is not None:
            common_horizons = np.intersect1d(common_horizons, baseline_horizons)
        
        if len(common_horizons) == 0:
            target_scores[tgt] = {
                'skill_vs_dfm': np.nan,
                'skill_vs_baseline': np.nan,
                'skill_consistency': np.nan,
                'n_horizons': 0,
                'error': 'No common horizons'
            }
            continue
        
        # Calculate skill metrics for each horizon
        horizon_skills = []
        skill_vs_dfm_list = []
        skill_vs_baseline_list = []
        
        for h in common_horizons:
            ddfm_idx = np.where(ddfm_horizons == h)[0]
            dfm_idx = np.where(dfm_horizons == h)[0]
            
            if len(ddfm_idx) == 0 or len(dfm_idx) == 0:
                continue
            
            ddfm_err = ddfm_smae[ddfm_idx[0]]
            dfm_err = dfm_smae[dfm_idx[0]]
            
            if np.isnan(ddfm_err) or np.isnan(dfm_err) or dfm_err <= 0:
                continue
            
            # Skill vs DFM: (DFM - DDFM) / DFM, normalized to 0-1 range
            # Positive = DDFM better, Negative = DDFM worse
            skill_vs_dfm = (dfm_err - ddfm_err) / dfm_err
            skill_vs_dfm_list.append(skill_vs_dfm)
            
            # Skill vs baseline (if available)
            skill_vs_baseline = None
            if baseline_smae is not None:
                baseline_idx = np.where(baseline_horizons == h)[0]
                if len(baseline_idx) > 0:
                    baseline_err = baseline_smae[baseline_idx[0]]
                    if not np.isnan(baseline_err) and baseline_err > 0:
                        skill_vs_baseline = (baseline_err - ddfm_err) / baseline_err
                        skill_vs_baseline_list.append(skill_vs_baseline)
            
            horizon_skills.append({
                'horizon': int(h),
                'skill_vs_dfm': float(skill_vs_dfm),
                'skill_vs_baseline': float(skill_vs_baseline) if skill_vs_baseline is not None else None,
                'ddfm_sMAE': float(ddfm_err),
                'dfm_sMAE': float(dfm_err),
                'baseline_sMAE': float(baseline_smae[baseline_idx[0]]) if baseline_smae is not None and len(baseline_idx) > 0 else None
            })
        
        if len(skill_vs_dfm_list) == 0:
            target_scores[tgt] = {
                'skill_vs_dfm': np.nan,
                'skill_vs_baseline': np.nan,
                'skill_consistency': np.nan,
                'n_horizons': 0,
                'error': 'No valid horizons'
            }
            continue
        
        # Overall skill metrics
        avg_skill_vs_dfm = float(np.mean(skill_vs_dfm_list))
        skill_std_vs_dfm = float(np.std(skill_vs_dfm_list))
        skill_consistency = 1.0 - min(1.0, skill_std_vs_dfm / max(abs(avg_skill_vs_dfm), 0.1)) if avg_skill_vs_dfm != 0 else 0.0
        
        avg_skill_vs_baseline = None
        if len(skill_vs_baseline_list) > 0:
            avg_skill_vs_baseline = float(np.mean(skill_vs_baseline_list))
        
        # Skill interpretation
        if avg_skill_vs_dfm > 0.10:
            skill_level = 'HIGH'  # DDFM significantly better than DFM
        elif avg_skill_vs_dfm > 0.05:
            skill_level = 'MODERATE'  # DDFM moderately better than DFM
        elif avg_skill_vs_dfm > -0.05:
            skill_level = 'LOW'  # DDFM similar to DFM
        else:
            skill_level = 'NEGATIVE'  # DDFM worse than DFM
        
        target_scores[tgt] = {
            'skill_vs_dfm': avg_skill_vs_dfm,
            'skill_vs_dfm_pct': float(avg_skill_vs_dfm * 100),  # Percentage improvement
            'skill_vs_baseline': avg_skill_vs_baseline,
            'skill_vs_baseline_pct': float(avg_skill_vs_baseline * 100) if avg_skill_vs_baseline is not None else None,
            'skill_consistency': float(skill_consistency),  # 0-1, higher = more consistent
            'skill_level': skill_level,
            'skill_std': float(skill_std_vs_dfm),
            'n_horizons': len(common_horizons),
            'n_valid_horizons': len(horizon_skills),
            'horizon_skills': horizon_skills,
            'positive_skill_horizons': len([s for s in skill_vs_dfm_list if s > 0]),
            'negative_skill_horizons': len([s for s in skill_vs_dfm_list if s < 0])
        }
    
    # Overall summary
    if target_scores:
        valid_skills = [s['skill_vs_dfm'] for s in target_scores.values() if not np.isnan(s.get('skill_vs_dfm', np.nan))]
        valid_consistencies = [s['skill_consistency'] for s in target_scores.values() if not np.isnan(s.get('skill_consistency', np.nan))]
        
        if len(valid_skills) > 0:
            overall_skill_vs_dfm = float(np.mean(valid_skills))
        else:
            overall_skill_vs_dfm = np.nan
            
        if len(valid_consistencies) > 0:
            overall_skill_consistency = float(np.mean(valid_consistencies))
        else:
            overall_skill_consistency = np.nan
        
        interpretation = f"Overall DDFM skill vs DFM: {overall_skill_vs_dfm*100:.1f}% improvement. "
        interpretation += f"Skill consistency: {overall_skill_consistency:.2f} (0-1, higher = more consistent). "
        if overall_skill_vs_dfm > 0.10:
            interpretation += "DDFM shows high skill improvement over DFM."
        elif overall_skill_vs_dfm > 0.05:
            interpretation += "DDFM shows moderate skill improvement over DFM."
        elif overall_skill_vs_dfm > -0.05:
            interpretation += "DDFM shows similar skill to DFM (potential linear collapse)."
        else:
            interpretation += "DDFM shows negative skill (worse than DFM)."
    else:
        overall_skill_vs_dfm = np.nan
        overall_skill_consistency = np.nan
        interpretation = "No valid skill assessment available."
    
    return {
        'target_scores': target_scores,
        'overall_skill_vs_dfm': float(overall_skill_vs_dfm) if not np.isnan(overall_skill_vs_dfm) else None,
        'overall_skill_consistency': float(overall_skill_consistency) if not np.isnan(overall_skill_consistency) else None,
        'skill_interpretation': interpretation
    }


def calculate_volatile_horizon_performance(
    aggregated_results: pd.DataFrame,
    target: Optional[str] = None,
    dfm_model: Optional[str] = 'DFM',
    ddfm_model: Optional[str] = 'DDFM',
    volatility_threshold: float = 1.5
) -> Dict[str, Any]:
    """Calculate DDFM performance specifically on volatile horizons compared to DFM.
    
    This function identifies horizons with high error variance (volatile horizons)
    and measures how well DDFM handles these challenging horizons compared to DFM.
    This is particularly important for DDFM evaluation because:
    1. Volatile horizons often indicate complex nonlinear dynamics
    2. DDFM should excel at these horizons if it's learning nonlinear features
    3. Poor performance on volatile horizons suggests the model may be collapsing to linear behavior
    
    The function calculates:
    - Volatile horizon identification (error > mean + threshold * std)
    - DDFM vs DFM improvement on volatile horizons
    - DDFM vs DFM improvement on stable horizons (for comparison)
    - Volatile horizon handling score (0-1, higher = better handling of volatile horizons)
    
    Parameters
    ----------
    aggregated_results : pd.DataFrame
        Aggregated results DataFrame with columns: target, model, horizon, sMSE, sMAE, sRMSE
    target : str, optional
        Target series to analyze. If None, analyzes all targets.
    dfm_model : str, default 'DFM'
        Name of DFM model in results
    ddfm_model : str, default 'DDFM'
        Name of DDFM model in results
    volatility_threshold : float, default 1.5
        Threshold for identifying volatile horizons (error > mean + threshold * std)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'target_analysis': Per-target detailed analysis
        - 'overall_summary': Overall summary across all targets
        - 'volatile_horizon_handling_score': Overall score (0-1, higher = better)
    """
    logger = _module_logger
    
    if aggregated_results.empty:
        logger.warning("Empty aggregated_results, cannot calculate volatile horizon performance")
        return {'error': 'Empty results'}
    
    # Filter for DDFM and DFM models
    model_data = aggregated_results[
        aggregated_results['model'].isin([dfm_model, ddfm_model])
    ].copy()
    
    if target:
        model_data = model_data[model_data['target'] == target]
    
    if model_data.empty:
        logger.warning(f"No DFM/DDFM results found{' for ' + target if target else ''}")
        return {'error': 'No DFM/DDFM results'}
    
    target_analysis = {}
    
    for tgt in model_data['target'].unique():
        target_data = model_data[model_data['target'] == tgt]
        
        ddfm_data = target_data[target_data['model'] == ddfm_model].sort_values('horizon')
        dfm_data = target_data[target_data['model'] == dfm_model].sort_values('horizon')
        
        # Extract metrics
        ddfm_smae = ddfm_data['sMAE'].values
        ddfm_horizons = ddfm_data['horizon'].values
        dfm_smae = dfm_data['sMAE'].values
        dfm_horizons = dfm_data['horizon'].values
        
        # Find common horizons
        common_horizons = np.intersect1d(ddfm_horizons, dfm_horizons)
        
        if len(common_horizons) == 0:
            continue
        
        # Identify volatile horizons based on DFM error distribution
        # Volatile = error > mean + threshold * std
        valid_dfm_errors = dfm_smae[~np.isnan(dfm_smae)]
        if len(valid_dfm_errors) < 3:
            continue
        
        dfm_error_mean = float(np.mean(valid_dfm_errors))
        dfm_error_std = float(np.std(valid_dfm_errors))
        volatile_threshold = dfm_error_mean + volatility_threshold * dfm_error_std
        
        # Classify horizons
        volatile_horizons = []
        stable_horizons = []
        
        for h in common_horizons:
            ddfm_idx = np.where(ddfm_horizons == h)[0]
            dfm_idx = np.where(dfm_horizons == h)[0]
            
            if len(ddfm_idx) == 0 or len(dfm_idx) == 0:
                continue
            
            ddfm_err = ddfm_smae[ddfm_idx[0]]
            dfm_err = dfm_smae[dfm_idx[0]]
            
            if np.isnan(ddfm_err) or np.isnan(dfm_err):
                continue
            
            horizon_info = {
                'horizon': int(h),
                'ddfm_sMAE': float(ddfm_err),
                'dfm_sMAE': float(dfm_err),
                'improvement': float((dfm_err - ddfm_err) / max(dfm_err, 1e-10))
            }
            
            if dfm_err > volatile_threshold:
                volatile_horizons.append(horizon_info)
            else:
                stable_horizons.append(horizon_info)
        
        # Calculate improvement on volatile vs stable horizons
        volatile_improvements = [h['improvement'] for h in volatile_horizons]
        stable_improvements = [h['improvement'] for h in stable_horizons]
        
        avg_volatile_improvement = float(np.mean(volatile_improvements)) if volatile_improvements else np.nan
        avg_stable_improvement = float(np.mean(stable_improvements)) if stable_improvements else np.nan
        
        # Volatile horizon handling score (0-1, higher = better)
        # Positive improvement on volatile horizons = good handling
        # If DDFM improves more on volatile horizons than stable, score is higher
        if not np.isnan(avg_volatile_improvement):
            # Normalize improvement to 0-1 range (assuming max improvement is 1.0 = 100%)
            volatile_score = float(np.clip((avg_volatile_improvement + 1.0) / 2.0, 0.0, 1.0))
        else:
            volatile_score = 0.5  # Neutral if no volatile horizons
        
        # Relative performance: how much better is DDFM on volatile vs stable horizons
        if not np.isnan(avg_volatile_improvement) and not np.isnan(avg_stable_improvement):
            relative_advantage = avg_volatile_improvement - avg_stable_improvement
        else:
            relative_advantage = np.nan
        
        target_analysis[tgt] = {
            'n_volatile_horizons': len(volatile_horizons),
            'n_stable_horizons': len(stable_horizons),
            'volatile_threshold': float(volatile_threshold),
            'avg_volatile_improvement': avg_volatile_improvement,
            'avg_stable_improvement': avg_stable_improvement,
            'relative_advantage': float(relative_advantage) if not np.isnan(relative_advantage) else np.nan,
            'volatile_horizon_handling_score': volatile_score,
            'volatile_horizons': volatile_horizons,
            'stable_horizons': stable_horizons[:5]  # Limit to first 5 for brevity
        }
    
    # Overall summary
    if target_analysis:
        valid_scores = [a['volatile_horizon_handling_score'] for a in target_analysis.values() 
                       if not np.isnan(a.get('volatile_horizon_handling_score', np.nan))]
        overall_score = float(np.mean(valid_scores)) if valid_scores else np.nan
        
        # Interpretation
        if not np.isnan(overall_score):
            if overall_score > 0.7:
                interpretation = "EXCELLENT: DDFM handles volatile horizons very well, suggesting strong nonlinear learning"
            elif overall_score > 0.6:
                interpretation = "GOOD: DDFM handles volatile horizons reasonably well"
            elif overall_score > 0.5:
                interpretation = "MODERATE: DDFM performance on volatile horizons is similar to stable horizons"
            else:
                interpretation = "POOR: DDFM struggles with volatile horizons, may indicate linear collapse"
        else:
            interpretation = "Cannot determine: insufficient data"
    else:
        overall_score = np.nan
        interpretation = "No analysis available"
    
    return {
        'target_analysis': target_analysis,
        'overall_summary': {
            'volatile_horizon_handling_score': float(overall_score) if not np.isnan(overall_score) else None,
            'interpretation': interpretation,
            'n_targets': len(target_analysis)
        }
    }


def calculate_near_linear_collapse_detection(
    aggregated_results: pd.DataFrame,
    target: Optional[str] = None,
    dfm_model: Optional[str] = 'DFM',
    ddfm_model: Optional[str] = 'DDFM',
    numerical_precision_threshold: float = 0.01,
    relative_precision_threshold: float = 0.001
) -> Dict[str, Any]:
    """Detect near-linear collapse by analyzing numerical precision of differences between DDFM and DFM.
    
    This function specifically identifies cases where DDFM and DFM errors are within
    numerical precision, which is a stronger signal of linear collapse than just similarity.
    This is particularly important for cases like KOEQUIPTE where differences are < 0.01
    across all horizons, indicating the encoder is learning essentially linear features.
    
    The function analyzes:
    1. Absolute differences: Are DDFM and DFM errors within numerical precision (< threshold)?
    2. Relative differences: Are relative differences < 0.1% (indicating near-identical performance)?
    3. Consistency of near-collapse: How many horizons show near-collapse behavior?
    4. Numerical precision score: Quantifies how close DDFM is to DFM (0-1, higher = more linear)
    
    Parameters
    ----------
    aggregated_results : pd.DataFrame
        Aggregated results DataFrame with columns: target, model, horizon, sMSE, sMAE, sRMSE
    target : str, optional
        Target series to analyze. If None, analyzes all targets.
    dfm_model : str, default 'DFM'
        Name of DFM model in results
    ddfm_model : str, default 'DDFM'
        Name of DDFM model in results
    numerical_precision_threshold : float, default 0.01
        Threshold for absolute difference (errors within this are considered near-identical)
    relative_precision_threshold : float, default 0.001
        Threshold for relative difference (0.1% = 0.001)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'target_analysis': Per-target detailed analysis with near-collapse detection
        - 'overall_summary': Overall summary across all targets
        - 'near_collapse_score': Overall score (0-1, higher = more likely near-collapse)
    """
    logger = _module_logger
    
    if aggregated_results.empty:
        logger.warning("Empty aggregated_results, cannot detect near-linear collapse")
        return {'error': 'Empty results'}
    
    # Filter for DDFM and DFM models
    model_data = aggregated_results[
        aggregated_results['model'].isin([dfm_model, ddfm_model])
    ].copy()
    
    if target:
        model_data = model_data[model_data['target'] == target]
    
    if model_data.empty:
        logger.warning(f"No DFM/DDFM results found{' for ' + target if target else ''}")
        return {'error': 'No DFM/DDFM results'}
    
    target_analysis = {}
    
    for tgt in model_data['target'].unique():
        target_data = model_data[model_data['target'] == tgt]
        
        ddfm_data = target_data[target_data['model'] == ddfm_model].sort_values('horizon')
        dfm_data = target_data[target_data['model'] == dfm_model].sort_values('horizon')
        
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
        
        # Analyze differences for each horizon
        near_collapse_horizons = []
        horizon_differences = []
        
        for h in common_horizons:
            ddfm_idx = np.where(ddfm_horizons == h)[0]
            dfm_idx = np.where(dfm_horizons == h)[0]
            
            if len(ddfm_idx) == 0 or len(dfm_idx) == 0:
                continue
            
            ddfm_err_smae = ddfm_smae[ddfm_idx[0]]
            dfm_err_smae = dfm_smae[dfm_idx[0]]
            ddfm_err_smse = ddfm_smse[ddfm_idx[0]] if len(ddfm_smse) > ddfm_idx[0] else np.nan
            dfm_err_smse = dfm_smse[dfm_idx[0]] if len(dfm_smse) > dfm_idx[0] else np.nan
            
            if np.isnan(ddfm_err_smae) or np.isnan(dfm_err_smae):
                continue
            
            # Calculate absolute and relative differences
            abs_diff_smae = abs(ddfm_err_smae - dfm_err_smae)
            rel_diff_smae = abs_diff_smae / max(abs(dfm_err_smae), abs(ddfm_err_smae), 1e-10)
            
            abs_diff_smse = abs(ddfm_err_smse - dfm_err_smse) if not (np.isnan(ddfm_err_smse) or np.isnan(dfm_err_smse)) else np.nan
            rel_diff_smse = abs_diff_smse / max(abs(dfm_err_smse), abs(ddfm_err_smse), 1e-10) if not np.isnan(abs_diff_smse) else np.nan
            
            # Check if within numerical precision (near-collapse)
            is_near_collapse = (
                abs_diff_smae < numerical_precision_threshold and
                rel_diff_smae < relative_precision_threshold
            )
            
            # Also check sMSE if available
            if not np.isnan(abs_diff_smse) and not np.isnan(rel_diff_smse):
                is_near_collapse = is_near_collapse and (
                    abs_diff_smse < numerical_precision_threshold and
                    rel_diff_smse < relative_precision_threshold
                )
            
            horizon_info = {
                'horizon': int(h),
                'ddfm_sMAE': float(ddfm_err_smae),
                'dfm_sMAE': float(dfm_err_smae),
                'abs_diff_sMAE': float(abs_diff_smae),
                'rel_diff_sMAE': float(rel_diff_smae),
                'is_near_collapse': is_near_collapse
            }
            
            if not np.isnan(abs_diff_smse):
                horizon_info['ddfm_sMSE'] = float(ddfm_err_smse)
                horizon_info['dfm_sMSE'] = float(dfm_err_smse)
                horizon_info['abs_diff_sMSE'] = float(abs_diff_smse)
                horizon_info['rel_diff_sMSE'] = float(rel_diff_smse)
            
            horizon_differences.append(horizon_info)
            
            if is_near_collapse:
                near_collapse_horizons.append(horizon_info)
        
        # Calculate near-collapse metrics
        n_horizons = len(horizon_differences)
        n_near_collapse = len(near_collapse_horizons)
        near_collapse_fraction = float(n_near_collapse / n_horizons) if n_horizons > 0 else 0.0
        
        # Calculate average absolute and relative differences
        avg_abs_diff = float(np.mean([h['abs_diff_sMAE'] for h in horizon_differences])) if horizon_differences else np.nan
        avg_rel_diff = float(np.mean([h['rel_diff_sMAE'] for h in horizon_differences])) if horizon_differences else np.nan
        max_abs_diff = float(np.max([h['abs_diff_sMAE'] for h in horizon_differences])) if horizon_differences else np.nan
        max_rel_diff = float(np.max([h['rel_diff_sMAE'] for h in horizon_differences])) if horizon_differences else np.nan
        
        # Near-collapse score (0-1, higher = more likely near-collapse)
        # Based on: fraction of near-collapse horizons, average relative difference, max relative difference
        if n_horizons > 0:
            # Score component 1: Fraction of horizons with near-collapse (0-1)
            fraction_score = near_collapse_fraction
            
            # Score component 2: Average relative difference (lower = higher score)
            # Normalize: 0.001 (0.1%) = score 1.0, 0.01 (1%) = score 0.0
            if not np.isnan(avg_rel_diff):
                avg_diff_score = max(0.0, 1.0 - (avg_rel_diff / relative_precision_threshold))
                avg_diff_score = min(1.0, avg_diff_score)
            else:
                avg_diff_score = 0.0
            
            # Score component 3: Max relative difference (lower = higher score)
            if not np.isnan(max_rel_diff):
                max_diff_score = max(0.0, 1.0 - (max_rel_diff / (relative_precision_threshold * 10)))
                max_diff_score = min(1.0, max_diff_score)
            else:
                max_diff_score = 0.0
            
            # Weighted combination (fraction is most important)
            near_collapse_score = (
                0.5 * fraction_score +
                0.3 * avg_diff_score +
                0.2 * max_diff_score
            )
        else:
            near_collapse_score = np.nan
        
        # Interpretation
        if not np.isnan(near_collapse_score):
            if near_collapse_score > 0.9:
                interpretation = "CRITICAL: Near-linear collapse detected - DDFM and DFM errors are within numerical precision across most horizons. Encoder is likely learning only linear features."
            elif near_collapse_score > 0.7:
                interpretation = "HIGH RISK: Strong evidence of near-linear collapse - DDFM and DFM errors are very similar across many horizons."
            elif near_collapse_score > 0.5:
                interpretation = "MODERATE RISK: Some evidence of near-linear collapse - DDFM and DFM errors are similar at several horizons."
            elif near_collapse_score > 0.3:
                interpretation = "LOW RISK: Minimal evidence of near-linear collapse - DDFM shows some differences from DFM."
            else:
                interpretation = "LOW RISK: DDFM shows meaningful differences from DFM, suggesting nonlinear learning."
        else:
            interpretation = "Cannot determine: insufficient data"
        
        target_analysis[tgt] = {
            'n_horizons': n_horizons,
            'n_near_collapse': n_near_collapse,
            'near_collapse_fraction': near_collapse_fraction,
            'avg_abs_diff_sMAE': avg_abs_diff,
            'avg_rel_diff_sMAE': avg_rel_diff,
            'max_abs_diff_sMAE': max_abs_diff,
            'max_rel_diff_sMAE': max_rel_diff,
            'near_collapse_score': float(near_collapse_score) if not np.isnan(near_collapse_score) else np.nan,
            'interpretation': interpretation,
            'near_collapse_horizons': near_collapse_horizons[:10],  # Limit to first 10 for brevity
            'all_horizon_differences': horizon_differences[:10]  # Limit to first 10 for brevity
        }
    
    # Overall summary
    if target_analysis:
        valid_scores = [a['near_collapse_score'] for a in target_analysis.values() 
                       if not np.isnan(a.get('near_collapse_score', np.nan))]
        overall_score = float(np.mean(valid_scores)) if valid_scores else np.nan
        
        # Overall interpretation
        if not np.isnan(overall_score):
            if overall_score > 0.7:
                overall_interpretation = "CRITICAL: Strong evidence of near-linear collapse across targets. DDFM encoders may be learning only linear features."
            elif overall_score > 0.5:
                overall_interpretation = "MODERATE: Some evidence of near-linear collapse. Review encoder architecture and training procedures."
            else:
                overall_interpretation = "LOW: DDFM shows meaningful differences from DFM, suggesting successful nonlinear learning."
        else:
            overall_interpretation = "Cannot determine: insufficient data"
    else:
        overall_score = np.nan
        overall_interpretation = "No analysis available"
    
    return {
        'target_analysis': target_analysis,
        'overall_summary': {
            'near_collapse_score': float(overall_score) if not np.isnan(overall_score) else None,
            'interpretation': overall_interpretation,
            'n_targets': len(target_analysis),
            'numerical_precision_threshold': numerical_precision_threshold,
            'relative_precision_threshold': relative_precision_threshold
        }
    }


def calculate_error_pattern_smoothness(
    errors_by_horizon: Dict[int, float],
    method: str = 'variation'
) -> Dict[str, Any]:
    """Calculate error pattern smoothness across horizons.
    
    This metric measures how smooth/consistent error patterns are across forecast horizons.
    DDFM should ideally have smoother error patterns if the encoder is learning well,
    as nonlinear features should provide more consistent performance across horizons.
    
    Smoothness is measured using:
    1. Coefficient of variation (CV) of errors across horizons (lower = smoother)
    2. First-order differences (how much errors change between consecutive horizons)
    3. Second-order differences (acceleration/deceleration in error changes)
    4. Autocorrelation of errors (higher = smoother pattern)
    
    Parameters
    ----------
    errors_by_horizon : dict
        Dictionary mapping horizon (int) to error value (float)
    method : str, default 'variation'
        Method for calculating smoothness:
        - 'variation': Uses coefficient of variation (lower = smoother)
        - 'differences': Uses first-order differences (lower = smoother)
        - 'autocorr': Uses autocorrelation (higher = smoother)
        - 'combined': Combines all methods
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'smoothness_score': Overall smoothness score (0-1, higher = smoother)
        - 'cv': Coefficient of variation of errors
        - 'avg_first_diff': Average absolute first-order difference
        - 'avg_second_diff': Average absolute second-order difference
        - 'autocorr': First-order autocorrelation of errors
        - 'interpretation': Interpretation of smoothness level
    """
    logger = _module_logger
    
    if len(errors_by_horizon) < 3:
        logger.warning("Need at least 3 horizons for smoothness analysis")
        return {
            'smoothness_score': None,
            'cv': None,
            'avg_first_diff': None,
            'avg_second_diff': None,
            'autocorr': None,
            'interpretation': 'Insufficient data'
        }
    
    # Sort by horizon
    sorted_horizons = sorted(errors_by_horizon.keys())
    error_values = [errors_by_horizon[h] for h in sorted_horizons]
    error_array = np.array(error_values)
    
    # Filter valid values
    valid_mask = np.isfinite(error_array)
    if np.sum(valid_mask) < 3:
        return {
            'smoothness_score': None,
            'cv': None,
            'avg_first_diff': None,
            'avg_second_diff': None,
            'autocorr': None,
            'interpretation': 'Insufficient valid data'
        }
    
    error_valid = error_array[valid_mask]
    
    # 1. Coefficient of variation (lower = smoother)
    if np.mean(error_valid) > 1e-10:
        cv = float(np.std(error_valid) / np.mean(error_valid))
    else:
        cv = 0.0
    
    # 2. First-order differences (how much errors change between consecutive horizons)
    first_diffs = np.diff(error_valid)
    avg_first_diff = float(np.mean(np.abs(first_diffs))) if len(first_diffs) > 0 else 0.0
    
    # 3. Second-order differences (acceleration/deceleration)
    if len(first_diffs) > 1:
        second_diffs = np.diff(first_diffs)
        avg_second_diff = float(np.mean(np.abs(second_diffs)))
    else:
        avg_second_diff = 0.0
    
    # 4. Autocorrelation (higher = smoother pattern)
    if len(error_valid) > 1 and np.std(error_valid) > 1e-10:
        autocorr = float(np.corrcoef(error_valid[:-1], error_valid[1:])[0, 1])
        if np.isnan(autocorr):
            autocorr = 0.0
    else:
        autocorr = 0.0
    
    # Normalize metrics to 0-1 scale for smoothness score
    # Lower CV = smoother, lower differences = smoother, higher autocorr = smoother
    # Normalize CV: assume CV > 1.0 is very rough, CV < 0.1 is very smooth
    cv_score = max(0.0, min(1.0, 1.0 - (cv / 1.0)))  # Invert: lower CV = higher score
    
    # Normalize first diff: assume avg_diff > mean(error) is rough, < 0.1*mean is smooth
    if np.mean(error_valid) > 1e-10:
        first_diff_normalized = avg_first_diff / np.mean(error_valid)
        first_diff_score = max(0.0, min(1.0, 1.0 - min(first_diff_normalized / 0.5, 1.0)))
    else:
        first_diff_score = 1.0
    
    # Normalize second diff similarly
    if np.mean(np.abs(first_diffs)) > 1e-10 and len(first_diffs) > 1:
        second_diff_normalized = avg_second_diff / np.mean(np.abs(first_diffs))
        second_diff_score = max(0.0, min(1.0, 1.0 - min(second_diff_normalized / 0.5, 1.0)))
    else:
        second_diff_score = 1.0
    
    # Autocorrelation: already 0-1 scale (or -1 to 1, normalize to 0-1)
    autocorr_score = max(0.0, (autocorr + 1.0) / 2.0)  # Map [-1, 1] to [0, 1]
    
    # Combined smoothness score (weighted average)
    smoothness_score = float(0.3 * cv_score + 0.3 * first_diff_score + 
                            0.2 * second_diff_score + 0.2 * autocorr_score)
    
    # Interpretation
    if smoothness_score > 0.7:
        interpretation = "Very smooth: Error patterns are highly consistent across horizons"
    elif smoothness_score > 0.5:
        interpretation = "Moderately smooth: Error patterns show reasonable consistency"
    elif smoothness_score > 0.3:
        interpretation = "Rough: Error patterns show significant variation across horizons"
    else:
        interpretation = "Very rough: Error patterns are highly inconsistent, may indicate training instability"
    
    return {
        'smoothness_score': float(smoothness_score),
        'cv': float(cv),
        'avg_first_diff': float(avg_first_diff),
        'avg_second_diff': float(avg_second_diff),
        'autocorr': float(autocorr),
        'interpretation': interpretation
    }


def calculate_improvement_significance(
    ddfm_errors_by_horizon: Dict[int, float],
    dfm_errors_by_horizon: Dict[int, float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """Calculate statistical significance of DDFM improvements over DFM.
    
    Uses bootstrap resampling to estimate confidence intervals for improvement ratios
    and determine if improvements are statistically significant.
    
    Parameters
    ----------
    ddfm_errors_by_horizon : dict
        Dictionary mapping horizon (int) to DDFM error (float)
    dfm_errors_by_horizon : dict
        Dictionary mapping horizon (int) to DFM error (float)
    n_bootstrap : int, default 1000
        Number of bootstrap resamples
    confidence_level : float, default 0.95
        Confidence level for intervals (0.95 = 95% confidence)
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'mean_improvement': Mean improvement ratio (positive = DDFM better)
        - 'improvement_ci_lower': Lower bound of confidence interval
        - 'improvement_ci_upper': Upper bound of confidence interval
        - 'is_significant': Whether improvement is statistically significant (CI doesn't include 0)
        - 'p_value': Approximate p-value (fraction of bootstrap samples with improvement <= 0)
        - 'significant_horizons': List of horizons with statistically significant improvements
        - 'horizon_significance': Per-horizon significance analysis
    """
    logger = _module_logger
    
    # Find common horizons
    common_horizons = sorted(set(ddfm_errors_by_horizon.keys()) & 
                            set(dfm_errors_by_horizon.keys()))
    
    if len(common_horizons) < 3:
        logger.warning("Need at least 3 common horizons for significance testing")
        return {
            'mean_improvement': None,
            'improvement_ci_lower': None,
            'improvement_ci_upper': None,
            'is_significant': False,
            'p_value': None,
            'significant_horizons': [],
            'horizon_significance': {}
        }
    
    # Calculate improvement ratios for each horizon
    improvements = []
    horizon_improvements = {}
    
    for h in common_horizons:
        ddfm_err = ddfm_errors_by_horizon[h]
        dfm_err = dfm_errors_by_horizon[h]
        
        if np.isfinite(ddfm_err) and np.isfinite(dfm_err) and dfm_err > 1e-10:
            improvement = (dfm_err - ddfm_err) / dfm_err  # Positive = DDFM better
            improvements.append(improvement)
            horizon_improvements[h] = improvement
        else:
            horizon_improvements[h] = np.nan
    
    if len(improvements) < 3:
        return {
            'mean_improvement': None,
            'improvement_ci_lower': None,
            'improvement_ci_upper': None,
            'is_significant': False,
            'p_value': None,
            'significant_horizons': [],
            'horizon_significance': {}
        }
    
    improvements_array = np.array(improvements)
    mean_improvement = float(np.mean(improvements_array))
    
    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(improvements_array, size=len(improvements_array), replace=True)
        bootstrap_means.append(float(np.mean(bootstrap_sample)))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Calculate confidence interval
    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
    
    # Statistical significance: CI doesn't include 0
    is_significant = ci_lower > 0.0 or ci_upper < 0.0
    
    # Approximate p-value: fraction of bootstrap samples with improvement <= 0
    p_value = float(np.mean(bootstrap_means <= 0.0))
    
    # Per-horizon significance (simpler: just check if improvement is > 0 and consistent)
    horizon_significance = {}
    significant_horizons = []
    
    for h, imp in horizon_improvements.items():
        if not np.isnan(imp):
            # Simple significance: improvement > 5% and positive
            is_horizon_significant = imp > 0.05
            horizon_significance[h] = {
                'improvement': float(imp),
                'is_significant': is_horizon_significant,
                'interpretation': 'Significant improvement' if is_horizon_significant else 
                                ('Degradation' if imp < -0.05 else 'No significant difference')
            }
            if is_horizon_significant:
                significant_horizons.append(int(h))
    
    return {
        'mean_improvement': mean_improvement,
        'improvement_ci_lower': ci_lower,
        'improvement_ci_upper': ci_upper,
        'is_significant': is_significant,
        'p_value': p_value,
        'significant_horizons': sorted(significant_horizons),
        'horizon_significance': horizon_significance,
        'n_horizons': len(common_horizons),
        'n_valid_horizons': len(improvements)
    }


def calculate_cross_target_pattern_comparison(
    aggregated_results: pd.DataFrame,
    metric: str = 'sMAE'
) -> Dict[str, Any]:
    """Compare DDFM performance patterns across different targets.
    
    This function identifies common patterns, outliers, and target-specific issues
    in DDFM performance across multiple targets. Helps identify if issues are
    target-specific or systematic.
    
    Parameters
    ----------
    aggregated_results : pd.DataFrame
        Aggregated results DataFrame with columns: target, model, horizon, sMSE, sMAE, sRMSE
    metric : str, default 'sMAE'
        Metric to analyze ('sMAE', 'sMSE', 'sRMSE')
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'target_comparison': Per-target summary statistics
        - 'common_patterns': Identified common patterns across targets
        - 'outliers': Target-specific outliers or issues
        - 'improvement_rankings': Ranking of targets by DDFM improvement
        - 'pattern_clusters': Clustering of targets by performance patterns
    """
    logger = _module_logger
    
    if aggregated_results.empty:
        logger.warning("Empty aggregated_results, cannot compare patterns")
        return {'error': 'Empty results'}
    
    # Filter for DDFM and DFM
    dfm_ddfm = aggregated_results[
        aggregated_results['model'].isin(['DFM', 'DDFM'])
    ].copy()
    
    if dfm_ddfm.empty:
        return {'error': 'No DFM/DDFM results'}
    
    # Calculate per-target statistics
    target_comparison = {}
    target_improvements = {}
    
    for target in dfm_ddfm['target'].unique():
        target_data = dfm_ddfm[dfm_ddfm['target'] == target]
        
        ddfm_data = target_data[target_data['model'] == 'DDFM']
        dfm_data = target_data[target_data['model'] == 'DFM']
        
        ddfm_metric = ddfm_data[metric].values
        dfm_metric = dfm_data[metric].values
        
        # Filter valid values
        valid_ddfm = ddfm_metric[np.isfinite(ddfm_metric)]
        valid_dfm = dfm_metric[np.isfinite(dfm_metric)]
        
        if len(valid_ddfm) == 0 or len(valid_dfm) == 0:
            continue
        
        avg_ddfm = float(np.mean(valid_ddfm))
        avg_dfm = float(np.mean(valid_dfm))
        
        # Improvement ratio
        if avg_dfm > 1e-10:
            improvement = (avg_dfm - avg_ddfm) / avg_dfm
        else:
            improvement = 0.0
        
        target_improvements[target] = improvement
        
        # Coefficient of variation (measure of consistency)
        cv_ddfm = float(np.std(valid_ddfm) / np.mean(valid_ddfm)) if np.mean(valid_ddfm) > 1e-10 else 0.0
        cv_dfm = float(np.std(valid_dfm) / np.mean(valid_dfm)) if np.mean(valid_dfm) > 1e-10 else 0.0
        
        target_comparison[target] = {
            'avg_ddfm': avg_ddfm,
            'avg_dfm': avg_dfm,
            'improvement': float(improvement),
            'improvement_pct': float(improvement * 100),
            'cv_ddfm': cv_ddfm,
            'cv_dfm': cv_dfm,
            'n_horizons': len(valid_ddfm)
        }
    
    # Identify outliers (targets with very different improvement patterns)
    improvements_list = list(target_improvements.values())
    if len(improvements_list) > 2:
        improvements_array = np.array(improvements_list)
        q1 = float(np.percentile(improvements_array, 25))
        q3 = float(np.percentile(improvements_array, 75))
        iqr = q3 - q1
        
        outliers = []
        for target, imp in target_improvements.items():
            if imp < (q1 - 1.5 * iqr) or imp > (q3 + 1.5 * iqr):
                outliers.append({
                    'target': target,
                    'improvement': float(imp),
                    'reason': 'Low improvement' if imp < (q1 - 1.5 * iqr) else 'High improvement'
                })
    else:
        outliers = []
    
    # Ranking by improvement
    sorted_targets = sorted(target_improvements.items(), key=lambda x: x[1], reverse=True)
    improvement_rankings = [
        {'rank': i+1, 'target': target, 'improvement': float(imp)}
        for i, (target, imp) in enumerate(sorted_targets)
    ]
    
    # Common patterns
    common_patterns = []
    
    # Pattern 1: All targets show improvement
    if all(imp > 0 for imp in target_improvements.values()):
        common_patterns.append({
            'pattern': 'Universal improvement',
            'description': 'All targets show DDFM improvement over DFM',
            'targets': list(target_improvements.keys())
        })
    
    # Pattern 2: Mixed results
    positive_count = sum(1 for imp in target_improvements.values() if imp > 0)
    if 0 < positive_count < len(target_improvements):
        common_patterns.append({
            'pattern': 'Mixed results',
            'description': f'{positive_count}/{len(target_improvements)} targets show improvement',
            'targets': [t for t, imp in target_improvements.items() if imp > 0]
        })
    
    # Pattern 3: Linear collapse (improvement near 0)
    near_zero_count = sum(1 for imp in target_improvements.values() if abs(imp) < 0.01)
    if near_zero_count > 0:
        common_patterns.append({
            'pattern': 'Linear collapse',
            'description': f'{near_zero_count} target(s) show near-zero improvement (linear collapse)',
            'targets': [t for t, imp in target_improvements.items() if abs(imp) < 0.01]
        })
    
    return {
        'target_comparison': target_comparison,
        'common_patterns': common_patterns,
        'outliers': outliers,
        'improvement_rankings': improvement_rankings,
        'metric_analyzed': metric,
        'n_targets': len(target_comparison)
    }


def calculate_error_autocorrelation_analysis(
    ddfm_errors_by_horizon: Dict[int, float],
    dfm_errors_by_horizon: Dict[int, float],
    max_lag: int = 3
) -> Dict[str, Any]:
    """Calculate error autocorrelation analysis to detect systematic bias patterns.
    
    This metric analyzes how correlated errors are across consecutive horizons.
    High autocorrelation indicates systematic bias (errors persist across horizons),
    while low autocorrelation suggests more random/independent errors.
    
    This is particularly useful for DDFM analysis because:
    - High autocorrelation in DDFM but not DFM: Encoder may be learning systematic patterns
    - High autocorrelation in both: Common data structure issue
    - Low autocorrelation: Errors are more independent (better for forecasting)
    
    Parameters
    ----------
    ddfm_errors_by_horizon : dict
        Dictionary mapping horizon (int) to DDFM error value (float)
    dfm_errors_by_horizon : dict
        Dictionary mapping horizon (int) to DFM error value (float)
    max_lag : int, default 3
        Maximum lag to calculate autocorrelation for (1 = consecutive horizons)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'ddfm_autocorr_lag1': First-order autocorrelation for DDFM (consecutive horizons)
        - 'dfm_autocorr_lag1': First-order autocorrelation for DFM
        - 'autocorr_diff': Difference between DDFM and DFM autocorrelation
        - 'systematic_bias_score': Score indicating systematic bias (0-1, higher = more systematic)
        - 'autocorr_by_lag': Autocorrelation values for each lag (1 to max_lag)
        - 'interpretation': Interpretation of autocorrelation patterns
    """
    logger = _module_logger
    
    if len(ddfm_errors_by_horizon) < max_lag + 2:
        logger.warning(f"Need at least {max_lag + 2} horizons for autocorrelation analysis")
        return {
            'ddfm_autocorr_lag1': None,
            'dfm_autocorr_lag1': None,
            'autocorr_diff': None,
            'systematic_bias_score': None,
            'autocorr_by_lag': {},
            'interpretation': 'Insufficient data'
        }
    
    # Sort by horizon
    sorted_horizons = sorted(ddfm_errors_by_horizon.keys())
    ddfm_errors = np.array([ddfm_errors_by_horizon[h] for h in sorted_horizons])
    dfm_errors = np.array([dfm_errors_by_horizon.get(h, np.nan) for h in sorted_horizons])
    
    # Filter valid values
    ddfm_valid = np.isfinite(ddfm_errors)
    dfm_valid = np.isfinite(dfm_errors)
    
    if np.sum(ddfm_valid) < max_lag + 2:
        return {
            'ddfm_autocorr_lag1': None,
            'dfm_autocorr_lag1': None,
            'autocorr_diff': None,
            'systematic_bias_score': None,
            'autocorr_by_lag': {},
            'interpretation': 'Insufficient valid data'
        }
    
    ddfm_clean = ddfm_errors[ddfm_valid]
    dfm_clean = dfm_errors[dfm_valid] if np.sum(dfm_valid) >= max_lag + 2 else None
    
    # Calculate autocorrelation for each lag
    autocorr_by_lag = {}
    ddfm_autocorr_lag1 = None
    dfm_autocorr_lag1 = None
    
    for lag in range(1, min(max_lag + 1, len(ddfm_clean))):
        if len(ddfm_clean) > lag:
            # DDFM autocorrelation
            if np.std(ddfm_clean[:-lag]) > 1e-10 and np.std(ddfm_clean[lag:]) > 1e-10:
                ddfm_corr = np.corrcoef(ddfm_clean[:-lag], ddfm_clean[lag:])[0, 1]
                if np.isnan(ddfm_corr):
                    ddfm_corr = 0.0
                autocorr_by_lag[f'ddfm_lag{lag}'] = float(ddfm_corr)
                
                if lag == 1:
                    ddfm_autocorr_lag1 = float(ddfm_corr)
            
            # DFM autocorrelation (if available)
            if dfm_clean is not None and len(dfm_clean) > lag:
                if np.std(dfm_clean[:-lag]) > 1e-10 and np.std(dfm_clean[lag:]) > 1e-10:
                    dfm_corr = np.corrcoef(dfm_clean[:-lag], dfm_clean[lag:])[0, 1]
                    if np.isnan(dfm_corr):
                        dfm_corr = 0.0
                    autocorr_by_lag[f'dfm_lag{lag}'] = float(dfm_corr)
                    
                    if lag == 1:
                        dfm_autocorr_lag1 = float(dfm_corr)
    
    # Calculate difference
    if ddfm_autocorr_lag1 is not None and dfm_autocorr_lag1 is not None:
        autocorr_diff = float(ddfm_autocorr_lag1 - dfm_autocorr_lag1)
    else:
        autocorr_diff = None
    
    # Systematic bias score: 0-1, higher = more systematic bias
    # High autocorrelation (> 0.5) indicates systematic patterns
    systematic_bias_score = 0.0
    if ddfm_autocorr_lag1 is not None:
        # Normalize autocorrelation to 0-1 scale (abs value, since sign indicates direction)
        systematic_bias_score = min(1.0, abs(ddfm_autocorr_lag1))
        
        # If DDFM has higher autocorrelation than DFM, it's more systematic
        if autocorr_diff is not None and autocorr_diff > 0.2:
            systematic_bias_score = min(1.0, systematic_bias_score + 0.2)
    
    # Interpretation
    if ddfm_autocorr_lag1 is None:
        interpretation = 'Insufficient data for autocorrelation analysis'
    elif abs(ddfm_autocorr_lag1) < 0.2:
        interpretation = 'Low autocorrelation: errors are relatively independent across horizons (good for forecasting)'
    elif abs(ddfm_autocorr_lag1) < 0.5:
        interpretation = 'Moderate autocorrelation: some systematic patterns in errors across horizons'
    else:
        if ddfm_autocorr_lag1 > 0:
            interpretation = 'High positive autocorrelation: errors persist across horizons (systematic bias detected)'
        else:
            interpretation = 'High negative autocorrelation: errors alternate across horizons (oscillatory pattern)'
    
    # Add comparison with DFM if available
    if dfm_autocorr_lag1 is not None and autocorr_diff is not None:
        if abs(autocorr_diff) > 0.2:
            if autocorr_diff > 0:
                interpretation += f'. DDFM has {abs(autocorr_diff):.2f} higher autocorrelation than DFM (more systematic).'
            else:
                interpretation += f'. DDFM has {abs(autocorr_diff):.2f} lower autocorrelation than DFM (less systematic).'
        else:
            interpretation += '. DDFM and DFM have similar autocorrelation patterns.'
    
    return {
        'ddfm_autocorr_lag1': ddfm_autocorr_lag1,
        'dfm_autocorr_lag1': dfm_autocorr_lag1,
        'autocorr_diff': autocorr_diff,
        'systematic_bias_score': float(systematic_bias_score),
        'autocorr_by_lag': autocorr_by_lag,
        'interpretation': interpretation
    }


def calculate_improvement_stability(
    improvement_by_horizon: Dict[int, float],
    method: str = 'variance'
) -> Dict[str, Any]:
    """Calculate improvement stability across horizons.
    
    This metric measures how consistent DDFM improvements are across forecast horizons.
    High stability (low variance) indicates consistent improvement, while low stability
    (high variance) suggests improvements are horizon-specific or inconsistent.
    
    This is important for DDFM evaluation because:
    - Consistent improvement across horizons: Encoder learning generalizable features
    - Variable improvement: May indicate encoder overfitting to specific horizons
    - Negative improvements at some horizons: Encoder may be learning suboptimal features
    
    Parameters
    ----------
    improvement_by_horizon : dict
        Dictionary mapping horizon (int) to improvement ratio (float)
        Improvement ratio: (DFM_error - DDFM_error) / DFM_error (positive = DDFM better)
    method : str, default 'variance'
        Method for calculating stability:
        - 'variance': Uses variance of improvements (lower = more stable)
        - 'cv': Uses coefficient of variation (lower = more stable)
        - 'range': Uses range of improvements (lower = more stable)
        - 'iqr': Uses interquartile range (lower = more stable)
        - 'combined': Combines all methods
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'stability_score': Overall stability score (0-1, higher = more stable)
        - 'improvement_variance': Variance of improvements across horizons
        - 'improvement_cv': Coefficient of variation of improvements
        - 'improvement_range': Range (max - min) of improvements
        - 'improvement_iqr': Interquartile range of improvements
        - 'mean_improvement': Mean improvement across horizons
        - 'median_improvement': Median improvement across horizons
        - 'n_positive_improvements': Number of horizons with positive improvement
        - 'n_negative_improvements': Number of horizons with negative improvement
        - 'interpretation': Interpretation of stability level
    """
    logger = _module_logger
    
    if len(improvement_by_horizon) < 3:
        logger.warning("Need at least 3 horizons for improvement stability analysis")
        return {
            'stability_score': None,
            'improvement_variance': None,
            'improvement_cv': None,
            'improvement_range': None,
            'improvement_iqr': None,
            'mean_improvement': None,
            'median_improvement': None,
            'n_positive_improvements': 0,
            'n_negative_improvements': 0,
            'interpretation': 'Insufficient data'
        }
    
    # Extract improvement values
    improvements = np.array([improvement_by_horizon[h] for h in sorted(improvement_by_horizon.keys())])
    improvements = improvements[np.isfinite(improvements)]
    
    if len(improvements) < 3:
        return {
            'stability_score': None,
            'improvement_variance': None,
            'improvement_cv': None,
            'improvement_range': None,
            'improvement_iqr': None,
            'mean_improvement': None,
            'median_improvement': None,
            'n_positive_improvements': 0,
            'n_negative_improvements': 0,
            'interpretation': 'Insufficient valid data'
        }
    
    # Calculate statistics
    mean_improvement = float(np.mean(improvements))
    median_improvement = float(np.median(improvements))
    improvement_variance = float(np.var(improvements))
    improvement_std = float(np.std(improvements))
    
    # Coefficient of variation
    if abs(mean_improvement) > 1e-10:
        improvement_cv = float(improvement_std / abs(mean_improvement))
    else:
        improvement_cv = float(improvement_std) if improvement_std > 0 else 0.0
    
    # Range
    improvement_range = float(np.max(improvements) - np.min(improvements))
    
    # Interquartile range
    q75, q25 = np.percentile(improvements, [75, 25])
    improvement_iqr = float(q75 - q25)
    
    # Count positive/negative improvements
    n_positive = int(np.sum(improvements > 0))
    n_negative = int(np.sum(improvements < 0))
    n_total = len(improvements)
    
    # Calculate stability score (0-1, higher = more stable)
    # Normalize each metric to 0-1 scale, then combine
    stability_scores = []
    
    # Variance score: lower variance = higher score
    # Assume variance > 0.1 is high, < 0.01 is low
    if improvement_variance > 0:
        var_score = max(0.0, min(1.0, 1.0 - (improvement_variance / 0.1)))
        stability_scores.append(var_score)
    
    # CV score: lower CV = higher score
    # Assume CV > 2.0 is high, < 0.2 is low
    if improvement_cv > 0:
        cv_score = max(0.0, min(1.0, 1.0 - (improvement_cv / 2.0)))
        stability_scores.append(cv_score)
    
    # Range score: lower range = higher score
    # Assume range > 0.5 is high, < 0.05 is low
    if improvement_range > 0:
        range_score = max(0.0, min(1.0, 1.0 - (improvement_range / 0.5)))
        stability_scores.append(range_score)
    
    # IQR score: lower IQR = higher score
    # Assume IQR > 0.3 is high, < 0.02 is low
    if improvement_iqr > 0:
        iqr_score = max(0.0, min(1.0, 1.0 - (improvement_iqr / 0.3)))
        stability_scores.append(iqr_score)
    
    # Consistency score: fraction of horizons with same sign as mean
    if mean_improvement > 0:
        consistency_score = n_positive / n_total if n_total > 0 else 0.0
    elif mean_improvement < 0:
        consistency_score = n_negative / n_total if n_total > 0 else 0.0
    else:
        consistency_score = 0.5  # Neutral if mean is zero
    
    stability_scores.append(consistency_score)
    
    # Overall stability score: average of all component scores
    if len(stability_scores) > 0:
        stability_score = float(np.mean(stability_scores))
    else:
        stability_score = 0.0
    
    # Interpretation
    if stability_score >= 0.8:
        interpretation = f'Very stable improvement (score={stability_score:.2f}): Consistent improvement across {n_positive}/{n_total} horizons'
    elif stability_score >= 0.6:
        interpretation = f'Moderately stable improvement (score={stability_score:.2f}): Improvement at {n_positive}/{n_total} horizons'
    elif stability_score >= 0.4:
        interpretation = f'Variable improvement (score={stability_score:.2f}): Mixed results ({n_positive} positive, {n_negative} negative)'
    else:
        interpretation = f'Unstable improvement (score={stability_score:.2f}): Highly variable across horizons ({n_positive} positive, {n_negative} negative)'
    
    if mean_improvement < -0.05:
        interpretation += '. Mean improvement is negative, indicating DDFM is worse than DFM on average.'
    elif mean_improvement > 0.1:
        interpretation += f'. Strong average improvement ({mean_improvement*100:.1f}%).'
    
    return {
        'stability_score': float(stability_score),
        'improvement_variance': improvement_variance,
        'improvement_cv': improvement_cv,
        'improvement_range': improvement_range,
        'improvement_iqr': improvement_iqr,
        'mean_improvement': mean_improvement,
        'median_improvement': median_improvement,
        'n_positive_improvements': n_positive,
        'n_negative_improvements': n_negative,
        'n_total_horizons': n_total,
        'interpretation': interpretation
    }

