"""Evaluation module - model performance evaluation and comparison."""

import logging
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np

from src.utils import ValidationError

logger = logging.getLogger(__name__)


def evaluate_forecaster(
    forecaster: Any,
    y_train: Union[pd.DataFrame, pd.Series],
    y_test: Union[pd.DataFrame, pd.Series],
    horizons: Union[List[int], np.ndarray],
    target_series: Optional[Union[str, int]] = None,
    y_recent: Optional[Union[pd.DataFrame, pd.Series]] = None
) -> Dict[int, Dict[str, float]]:
    """Evaluate forecaster on test data for multiple horizons.
    
    Parameters
    ----------
    forecaster : Any
        Trained forecaster (sktime-compatible)
    y_train : pd.DataFrame or pd.Series
        Training data
    y_test : pd.DataFrame or pd.Series
        Test data
    horizons : List[int] or np.ndarray
        Forecast horizons to evaluate
    target_series : Optional[str or int]
        Target series name/index (for multivariate models)
    y_recent : Optional[pd.DataFrame or pd.Series]
        Recent data for model update (for DFM/DDFM)
        
    Returns
    -------
    Dict[int, Dict[str, float]]
        Dictionary mapping horizon to metrics (sMSE, sMAE, sRMSE, n_valid)
    """
    from sktime.forecasting.base import ForecastingHorizon
    
    # Ensure forecaster is fitted
    if not hasattr(forecaster, 'is_fitted') or not forecaster.is_fitted:
        forecaster.fit(y_train)
    
    # Update with recent data if provided (for DFM/DDFM)
    # Use history=None (window=None) to include full period for state update
    if y_recent is not None and hasattr(forecaster, 'update'):
        try:
            # Convert to numpy array if DataFrame/Series
            if isinstance(y_recent, (pd.DataFrame, pd.Series)):
                y_recent_array = y_recent.values
            else:
                y_recent_array = y_recent
            
            # For DFM/DDFM, update() expects standardized data
            # Check if forecaster has result with Mx/Wx for standardization
            if hasattr(forecaster, 'result') and hasattr(forecaster.result, 'Mx') and hasattr(forecaster.result, 'Wx'):
                Mx = np.asarray(forecaster.result.Mx)
                Wx = np.asarray(forecaster.result.Wx)
                # Handle shape mismatch
                if y_recent_array.ndim == 2:
                    Mx_use = Mx[:y_recent_array.shape[1]] if len(Mx) >= y_recent_array.shape[1] else Mx
                    Wx_use = Wx[:y_recent_array.shape[1]] if len(Wx) >= y_recent_array.shape[1] else Wx
                    Wx_use = np.where(Wx_use == 0, 1.0, Wx_use)
                    y_recent_std = (y_recent_array - Mx_use) / Wx_use
                else:
                    y_recent_std = y_recent_array
            else:
                # If no Mx/Wx, assume data is already standardized or use as-is
                y_recent_std = y_recent_array
            
            # Update with history=None to use full period (window=None)
            forecaster.update(y_recent_std, history=None)
            model_type = forecaster.__class__.__name__ if hasattr(forecaster, '__class__') else 'forecaster'
            logger.debug(f"Updated {model_type} state with {len(y_recent_array)} periods (window=None)")
        except Exception as e:
            logger.warning(f"Failed to update forecaster with recent data: {e}")
    
    # Extract target series if needed
    if isinstance(y_test, pd.DataFrame):
        if target_series is not None:
            if isinstance(target_series, str) and target_series in y_test.columns:
                y_test_actual = y_test[target_series]
            elif isinstance(target_series, int) and target_series < len(y_test.columns):
                y_test_actual = y_test.iloc[:, target_series]
            else:
                y_test_actual = y_test.iloc[:, 0]
        else:
            y_test_actual = y_test.iloc[:, 0]
    else:
        y_test_actual = y_test
    
    results = {}
    horizons = np.array(horizons) if not isinstance(horizons, np.ndarray) else horizons
    max_horizon = int(horizons.max())
    
    # Check if this is a DFM/DDFM model (from dfm-python package)
    is_dfm_model = False
    model_type_name = forecaster.__class__.__name__ if hasattr(forecaster, '__class__') else ''
    if model_type_name in ['DFM', 'DDFM']:
        is_dfm_model = True
    
    # Check if this is a direct forecasting model (TFT, LSTM, Chronos)
    # These models support direct long-horizon forecasting, not recursive
    is_direct_forecasting_model = False
    if hasattr(forecaster, 'steps') and len(forecaster.steps) > 0:
        # Check if it's a ForecastingPipeline with a direct forecasting model
        last_step = forecaster.steps[-1]
        if hasattr(last_step, '__class__'):
            last_step_name = last_step.__class__.__name__
            if any(x in last_step_name for x in ['PytorchForecastingTFT', 'NeuralForecastLSTM', 
                                                   'Chronos']):
                is_direct_forecasting_model = True
    elif any(x in model_type_name for x in ['TFT', 'LSTM', 'Chronos']):
        is_direct_forecasting_model = True
    
    # For DFM/DDFM: Use dfm-python's predict interface
    # For ARIMA/VAR: Use sktime's predict interface
    try:
        if is_dfm_model:
            # DFM/DDFM models use horizon parameter, not fh
            # 1 horizon = 1 month = 4 weeks
            # For 24 months forecast, need 24 * 4 = 96 weeks
            # But we'll predict in chunks: 6 months at a time (24 weeks per chunk)
            # After 6 months, update with latest data and predict next 6 months
            
            # Check if model uses weekly clock (clock='w')
            clock = getattr(forecaster.config, 'clock', 'm') if hasattr(forecaster, 'config') else 'm'
            weeks_per_month = 4
            months_per_update = 6  # Update every 6 months
            
            # Calculate total weeks needed for max_horizon months
            total_weeks = max_horizon * weeks_per_month
            
            # For weekly models: predict in weeks and convert to monthly by averaging
            if clock == 'w':
                # Model state was updated with y_recent data above (if provided)
                # Now predict in weeks (horizon is in model's clock frequency)
                X_forecast, _ = forecaster.predict(
                    horizon=total_weeks,
                    return_series=True,
                    return_factors=True
                )
                
                # Convert weekly forecasts to monthly using aggregation function
                # X_forecast shape: (total_weeks, n_series)
                # Convert to DataFrame first
                try:
                    series_names = [s.series_id for s in forecaster.config.series]
                except:
                    n_series = X_forecast.shape[1] if X_forecast.ndim > 1 else 1
                    series_names = [f"series_{i}" for i in range(n_series)]
                
                # Create DataFrame with weekly index
                if last_date is not None:
                    weekly_index = pd.date_range(
                        start=last_date + pd.DateOffset(weeks=1),
                        periods=total_weeks,
                        freq='W'
                    )
                else:
                    weekly_index = pd.RangeIndex(start=0, stop=total_weeks)
                
                y_pred_all_weekly = pd.DataFrame(
                    X_forecast,
                    index=weekly_index,
                    columns=series_names[:X_forecast.shape[1]]
                )
                
                # Aggregate weekly to monthly
                from src.utils import aggregate_weekly_to_monthly
                y_pred_all = aggregate_weekly_to_monthly(y_pred_all_weekly, weeks_per_month=weeks_per_month)
                
                logger.info(f"Converted {total_weeks} weekly forecasts to {len(y_pred_all)} monthly forecasts")
            else:
                # For non-weekly models, use original logic
                X_forecast, _ = forecaster.predict(
                    horizon=max_horizon,
                    return_series=True,
                    return_factors=True
                )
                
                # Convert to DataFrame if needed
                if isinstance(X_forecast, np.ndarray):
                    try:
                        series_names = [s.series_id for s in forecaster.config.series]
                    except:
                        series_names = [f"series_{i}" for i in range(X_forecast.shape[1])]
                    
                    y_pred_all = pd.DataFrame(
                        X_forecast,
                        columns=series_names[:X_forecast.shape[1]]
                    )
                else:
                    y_pred_all = pd.DataFrame(X_forecast)
        else:
            # For ARIMA/VAR: Use recursive forecasting (step-by-step)
            # For TFT/LSTM/Chronos: Use direct long-horizon forecasting
            # Horizons are in months, convert to weeks (max_horizon months = max_horizon * 4 weeks)
            weeks_per_month = 4
            max_horizon_weeks = max_horizon * weeks_per_month  # Convert months to weeks
            
            if is_direct_forecasting_model:
                # Direct forecasting models: Generate all weeks at once (long-horizon forecasting)
                # These models support direct long-horizon forecasting (88 weeks = 22 months)
                fh = np.arange(1, max_horizon_weeks + 1)
                y_pred_all_weekly = forecaster.predict(fh=fh)
                logger.info(f"Generated {max_horizon_weeks} weekly forecasts using direct long-horizon forecasting")
            else:
                # Recursive forecasting for ARIMA/VAR: step-by-step prediction
                # sktime will automatically do recursive forecasting: 
                # step 1 uses last training value, step 2 uses step 1 prediction, etc.
                fh = np.arange(1, max_horizon_weeks + 1)
                y_pred_all_weekly = forecaster.predict(fh=fh)
                logger.info(f"Generated {max_horizon_weeks} weekly forecasts using recursive forecasting")
            
            # Aggregate weekly forecasts to monthly by averaging every 4 weeks
            from src.utils import aggregate_weekly_to_monthly
            y_pred_all = aggregate_weekly_to_monthly(y_pred_all_weekly, weeks_per_month=weeks_per_month)
            
            logger.info(f"Converted {max_horizon_weeks} weekly forecasts to {len(y_pred_all)} monthly forecasts")
        
        # Extract target series from prediction if needed
        if isinstance(y_pred_all, pd.DataFrame):
            if target_series is not None:
                if isinstance(target_series, str) and target_series in y_pred_all.columns:
                    y_pred_series_all = y_pred_all[target_series]
                elif isinstance(target_series, int) and target_series < len(y_pred_all.columns):
                    y_pred_series_all = y_pred_all.iloc[:, target_series]
                else:
                    # For DFM/DDFM, try to find target series by matching column names
                    if is_dfm_model:
                        # Try to find exact match or partial match
                        matching_cols = [col for col in y_pred_all.columns if target_series in str(col)]
                        if matching_cols:
                            y_pred_series_all = y_pred_all[matching_cols[0]]
                        else:
                            y_pred_series_all = y_pred_all.iloc[:, 0]
                    else:
                        y_pred_series_all = y_pred_all.iloc[:, 0]
            else:
                y_pred_series_all = y_pred_all.iloc[:, 0]
        else:
            y_pred_series_all = y_pred_all
        
        # Convert to numpy array for easier indexing
        if isinstance(y_pred_series_all, (pd.Series, pd.DataFrame)):
            y_pred_array = y_pred_series_all.values
        else:
            y_pred_array = np.array(y_pred_series_all)
        
        # Convert test data to array for easier indexing
        if isinstance(y_test_actual, (pd.Series, pd.DataFrame)):
            y_test_array = y_test_actual.values
        else:
            y_test_array = np.array(y_test_actual)
        
        # Calculate metrics for each horizon
        if isinstance(y_train, pd.DataFrame):
            train_std = y_train.iloc[:, 0].std() if target_series is None else y_train[target_series].std()
        else:
            train_std = y_train.std()
        
        for horizon in horizons:
            horizon = int(horizon)
            try:
                # Get prediction for this specific horizon (index horizon-1 because fh starts at 1)
                if horizon <= len(y_pred_array):
                    y_pred_h = y_pred_array[horizon - 1]
                    
                    # Get corresponding actual value (if available)
                    if horizon <= len(y_test_array):
                        y_true_h = y_test_array[horizon - 1]
                        
                        # Check for NaN
                        if not (np.isnan(y_true_h) or np.isnan(y_pred_h)):
                            # Absolute errors
                            error = y_true_h - y_pred_h
                            absMSE = (error ** 2)
                            absMAE = np.abs(error)
                            absRMSE = np.sqrt(absMSE)
                            
                            if train_std > 0:
                                sMSE = (error ** 2) / (train_std ** 2)
                                sMAE = np.abs(error) / train_std
                                sRMSE = np.sqrt(sMSE)
                            else:
                                sMSE = np.nan
                                sMAE = np.nan
                                sRMSE = np.nan
                            
                            results[horizon] = {
                                'sMSE': float(sMSE) if not np.isnan(sMSE) else np.nan,
                                'sMAE': float(sMAE) if not np.isnan(sMAE) else np.nan,
                                'sRMSE': float(sRMSE) if not np.isnan(sRMSE) else np.nan,
                                'MAE': float(absMAE),
                                'MSE': float(absMSE),
                                'RMSE': float(absRMSE),
                                'n_valid': 1
                            }
                        else:
                            results[horizon] = {
                                'sMSE': np.nan,
                                'sMAE': np.nan,
                                'sRMSE': np.nan,
                                'MAE': np.nan,
                                'MSE': np.nan,
                                'RMSE': np.nan,
                                'n_valid': 0
                            }
                    else:
                        # Test data not available for this horizon
                        results[horizon] = {
                            'sMSE': np.nan,
                            'sMAE': np.nan,
                            'sRMSE': np.nan,
                            'MAE': np.nan,
                            'MSE': np.nan,
                            'RMSE': np.nan,
                            'n_valid': 0
                        }
                else:
                    # Prediction not available for this horizon
                    results[horizon] = {
                        'sMSE': np.nan,
                        'sMAE': np.nan,
                        'sRMSE': np.nan,
                        'MAE': np.nan,
                        'MSE': np.nan,
                        'RMSE': np.nan,
                        'n_valid': 0
                    }
            except Exception as e:
                logger.warning(f"Failed to evaluate horizon {horizon}: {e}")
                results[horizon] = {
                    'sMSE': np.nan,
                    'sMAE': np.nan,
                    'sRMSE': np.nan,
                    'MAE': np.nan,
                    'MSE': np.nan,
                    'RMSE': np.nan,
                    'n_valid': 0
                }
    except Exception as e:
        logger.error(f"Failed to generate forecasts: {e}")
        # Initialize all horizons as failed
        for horizon in horizons:
            results[int(horizon)] = {
                'sMSE': np.nan,
                'sMAE': np.nan,
                'sRMSE': np.nan,
                'n_valid': 0
            }
    
    return results


def compare_multiple_models(
    model_results: Dict[str, Dict[str, Any]],
    horizons: List[int],
    target_series: Optional[str] = None
) -> Dict[str, Any]:
    """Compare multiple models' performance.
    
    Parameters
    ----------
    model_results : Dict[str, Dict[str, Any]]
        Dictionary mapping model name to results (with 'metrics' key)
    horizons : List[int]
        Forecast horizons
    target_series : Optional[str]
        Target series name
        
    Returns
    -------
    Dict[str, Any]
        Comparison results with metrics table and summary
    """
    # Generate comparison table
    table = generate_comparison_table(model_results, horizons, target_series)
    
    # Find best model per horizon
    best_model_per_horizon = {}
    for horizon in horizons:
        horizon_str = str(horizon)
        best_model = None
        best_sRMSE = np.inf
        
        for model_name, result in model_results.items():
            metrics = result.get('metrics', {})
            forecast_metrics = metrics.get('forecast_metrics', {})
            horizon_metrics = forecast_metrics.get(horizon_str, {})
            sRMSE = horizon_metrics.get('sRMSE', np.nan)
            
            if not np.isnan(sRMSE) and sRMSE < best_sRMSE:
                best_sRMSE = sRMSE
                best_model = model_name
        
        if best_model:
            best_model_per_horizon[horizon] = best_model
    
    return {
        'metrics_table': table,
        'summary': f'Compared {len(model_results)} models across {len(horizons)} horizons',
        'best_model_per_horizon': best_model_per_horizon
    }


def generate_comparison_table(
    model_results: Dict[str, Dict[str, Any]],
    horizons: List[int],
    target_series: Optional[str] = None
) -> pd.DataFrame:
    """Generate comparison table for multiple models.
    
    Parameters
    ----------
    model_results : Dict[str, Dict[str, Any]]
        Dictionary mapping model name to results
    horizons : List[int]
        Forecast horizons
    target_series : Optional[str]
        Target series name
        
    Returns
    -------
    pd.DataFrame
        Comparison table with metrics per model and horizon
    """
    rows = []
    metrics = ['sMSE', 'sMAE', 'sRMSE']
    
    for model_name, result in model_results.items():
        model_metrics = result.get('metrics', {})
        forecast_metrics = model_metrics.get('forecast_metrics', {})
        
        for horizon in horizons:
            horizon_str = str(horizon)
            horizon_metrics = forecast_metrics.get(horizon_str, {})
            n_valid = horizon_metrics.get('n_valid', 0)
            
            if n_valid > 0:
                for metric in metrics:
                    value = horizon_metrics.get(metric, np.nan)
                    if not np.isnan(value):
                        rows.append({
                            'model': model_name,
                            'horizon': horizon,
                            'metric': metric,
                            'value': value,
                            'n_valid': n_valid
                        })
    
    if not rows:
        return pd.DataFrame(columns=['model', 'horizon', 'metric', 'value', 'n_valid'])
    
    df = pd.DataFrame(rows)
    return df

