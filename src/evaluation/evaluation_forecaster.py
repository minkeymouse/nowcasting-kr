"""Forecaster evaluation functions."""

from typing import Union, Dict, Optional, Any
import numpy as np
import pandas as pd
from .evaluation import _module_logger
from .evaluation_metrics import calculate_standardized_metrics

def evaluate_forecaster(
    forecaster,
    y_train: Union[pd.DataFrame, pd.Series],
    y_test: Union[pd.DataFrame, pd.Series],
    horizons: Union[list, np.ndarray],
    target_series: Optional[Union[str, int]] = None,
    y_recent: Optional[Union[pd.DataFrame, pd.Series]] = None
) -> Dict[int, Dict[str, float]]:
    """Evaluate a sktime-compatible forecaster on test data.
    
    This function implements a single-step evaluation design where each forecast
    horizon is evaluated using exactly one test point. This is an intentional design
    choice rather than a limitation, as it provides focused assessment of model
    performance at each specific forecast horizon.
    
    Evaluation Design:
        - For each horizon h, the function extracts exactly one test point at position
          test_pos = h - 1 (since test data starts at train_end+1, horizon h corresponds
          to position h-1 in test data).
        - This results in n_valid=1 for all horizons, which is expected behavior.
        - The single-step design is appropriate for nowcasting applications where
          we want to assess model performance at specific forecast horizons rather
          than aggregating across multiple test points.
    
    Why Single-Step Evaluation?
        - Data limitation: After 80/20 train/test split, the test set may be too small
          for multi-step evaluation, especially for longer horizons (e.g., h=22).
        - Focused assessment: Single-step evaluation provides a clear, focused assessment
          of model performance at each specific forecast horizon without aggregation
          effects that could mask horizon-specific performance characteristics.
        - Consistency: All models are evaluated using the same single-point design,
          ensuring fair comparison across models.
    
    Note: This design limitation is documented in the report methodology section.
    Multi-step evaluation (evaluating multiple test points per horizon) is not used
    due to the data limitation mentioned above.
    
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
        Dictionary with horizon as key and metrics dict as value.
        Each metrics dict contains: sMSE, sMAE, sRMSE, MSE, MAE, RMSE, sigma, n_valid.
        Note: n_valid=1 for all valid horizons due to single-step evaluation design.
    """
    logger = _module_logger
    
    # Ensure DFM/DDFM attributes exist to avoid AttributeError on non-factor models (e.g., ARIMA pipeline)
    if not hasattr(forecaster, "_dfm_model"):
        try:
            setattr(forecaster, "_dfm_model", None)
        except Exception:
            pass
    if not hasattr(forecaster, "_ddfm_model"):
        try:
            setattr(forecaster, "_ddfm_model", None)
        except Exception:
            pass
    
    # Defense-in-depth: Validate train-test split to prevent data leakage
    # This provides additional validation beyond the checks in train.py
    if hasattr(y_train, 'index') and hasattr(y_test, 'index'):
        train_max = y_train.index.max() if len(y_train) > 0 else None
        test_min = y_test.index.min() if len(y_test) > 0 else None
        if train_max is not None and test_min is not None:
            if train_max >= test_min:
                raise ValueError(
                    f"Data leakage detected in evaluate_forecaster: "
                    f"Training period ends at {train_max} but test period starts at {test_min}. "
                    f"There must be a gap between training and test periods to prevent data leakage."
                )
            logger.debug(f"Train-test split validated: train ends at {train_max}, test starts at {test_min}")
    
    # Check if forecaster is already fitted to avoid re-training
    is_fitted = False
    if hasattr(forecaster, 'is_fitted'):
        # Check if it's a method (callable) or property
        if callable(forecaster.is_fitted):
            try:
                is_fitted = forecaster.is_fitted()
            except Exception:
                # If calling fails, try as property/attribute
                is_fitted = bool(forecaster.is_fitted) if not callable(forecaster.is_fitted) else False
        else:
            is_fitted = bool(forecaster.is_fitted)
    elif hasattr(forecaster, '_is_fitted'):
        is_fitted = bool(forecaster._is_fitted)
    elif hasattr(forecaster, '_fitted_forecaster'):
        is_fitted = forecaster._fitted_forecaster is not None
    elif hasattr(forecaster, '_y'):
        is_fitted = forecaster._y is not None
    
    # Only fit if not already fitted (to avoid re-training loaded models)
    if not is_fitted:
        logger.info("Forecaster not fitted, fitting on training data...")
        forecaster.fit(y_train)
    else:
        logger.debug("Forecaster already fitted, skipping fit() to avoid re-training")
    
    # Detect model type for appropriate handling
    is_dfm = hasattr(forecaster, '_dfm_model') and forecaster._dfm_model is not None
    is_ddfm = hasattr(forecaster, '_ddfm_model') and forecaster._ddfm_model is not None

    # Capture the model's trained feature set (column order) so evaluation uses the same width/order
    model_feature_cols = None
    if is_dfm:
        if hasattr(forecaster, "_y") and isinstance(getattr(forecaster, "_y"), pd.DataFrame):
            model_feature_cols = list(forecaster._y.columns)
        if not model_feature_cols:
            try:
                cfg = forecaster._dfm_model.config
                if cfg and hasattr(cfg, "series"):
                    model_feature_cols = [s.series_id if hasattr(s, "series_id") else getattr(s, "series_id", None) for s in cfg.series]
                    model_feature_cols = [c for c in model_feature_cols if c]
            except Exception:
                model_feature_cols = model_feature_cols or None
    elif is_ddfm:
        if hasattr(forecaster, "_y") and isinstance(getattr(forecaster, "_y"), pd.DataFrame):
            model_feature_cols = list(forecaster._y.columns)
        if not model_feature_cols:
            try:
                cfg = forecaster._ddfm_model.config
                if cfg and hasattr(cfg, "series"):
                    model_feature_cols = [s.series_id if hasattr(s, "series_id") else getattr(s, "series_id", None) for s in cfg.series]
                    model_feature_cols = [c for c in model_feature_cols if c]
            except Exception:
                model_feature_cols = model_feature_cols or None
    
    # Detect ARIMA/VAR models (recursive prediction needed)
    is_arima = False
    is_var = False
    forecaster_type_str = str(type(forecaster)).lower()
    if 'arima' in forecaster_type_str:
        is_arima = True
    elif 'var' in forecaster_type_str:
        is_var = True
    else:
        # Check forecaster_ attribute in pipeline (sktime uses forecaster_ for fitted forecaster)
        if hasattr(forecaster, 'forecaster_'):
            forecaster_attr_type_str = str(type(forecaster.forecaster_)).lower()
            forecaster_attr_module = type(forecaster.forecaster_).__module__.lower()
            if 'arima' in forecaster_attr_type_str or 'arima' in forecaster_attr_module:
                is_arima = True
            elif 'var' in forecaster_attr_type_str or 'var' in forecaster_attr_module:
                is_var = True
        # Check _fitted_forecaster for pipeline models
        if not is_arima and not is_var and hasattr(forecaster, '_fitted_forecaster'):
            fitted_type_str = str(type(forecaster._fitted_forecaster)).lower()
            if 'arima' in fitted_type_str:
                is_arima = True
            elif 'var' in fitted_type_str:
                is_var = True
        # Check forecaster attribute in pipeline (fallback)
        if not is_arima and not is_var and hasattr(forecaster, 'forecaster'):
            forecaster_attr_type_str = str(type(forecaster.forecaster)).lower()
            if 'arima' in forecaster_attr_type_str:
                is_arima = True
            elif 'var' in forecaster_attr_type_str:
                is_var = True
    
    # For DFM/DDFM models, update factor state with recent data before prediction
    # This ensures predictions reflect the most recent information, not just training period
    if y_recent is not None and len(y_recent) > 0:
        if is_dfm or is_ddfm:
            dfm_model = forecaster._dfm_model if is_dfm else forecaster._ddfm_model
            
            # Get result for standardization parameters (Mx, Wx)
            result = dfm_model.result
            Mx = result.Mx  # Mean for standardization (N,)
            Wx = result.Wx  # Standard deviation for standardization (N,)
            
            # Prepare recent data for update
            # Ensure y_recent has same columns as training data
            if isinstance(y_recent, pd.DataFrame):
                # Use same columns as training data, in the same order
                if isinstance(y_train, pd.DataFrame):
                    # Align columns with training data
                    available_cols = [col for col in y_train.columns if col in y_recent.columns]
                    if len(available_cols) > 0:
                        y_recent_aligned = y_recent[available_cols].copy()
                        # Reorder to match training data column order
                        y_recent_aligned = y_recent_aligned[y_train.columns]
                    else:
                        logger.warning("No matching columns between y_recent and y_train, skipping update")
                        y_recent_aligned = None
                else:
                    y_recent_aligned = y_recent.copy()
            else:
                # y_recent is Series, convert to DataFrame
                y_recent_aligned = y_recent.to_frame() if isinstance(y_recent, pd.Series) else pd.DataFrame(y_recent)
            
            if y_recent_aligned is not None and len(y_recent_aligned) > 0:
                # If we know the model's feature columns, align and reorder to them to match Mx/Wx length
                if model_feature_cols:
                    available = [c for c in model_feature_cols if c in y_recent_aligned.columns]
                    if not available:
                        logger.warning("No overlap between y_recent and model feature columns; skipping update")
                        y_recent_aligned = None
                    else:
                        y_recent_aligned = y_recent_aligned[available]
                        # If some model columns are missing, warn but continue with available ones
                        missing = [c for c in model_feature_cols if c not in available]
                        if missing:
                            logger.warning(f"y_recent missing {len(missing)} model feature column(s); update uses {len(available)} columns")
                if y_recent_aligned is None or len(y_recent_aligned) == 0:
                    pass  # handled below
                else:
                    y_recent_all = y_recent_aligned.values  # (T x N) array
                    
                    # Standardize recent data using training parameters: (X - Mx) / Wx
                    # Ensure shapes are compatible by trimming/ordering to model_feature_cols if provided
                    if model_feature_cols:
                        # Trim Mx/Wx to available columns length if needed
                        Mx_use = Mx[: y_recent_all.shape[1]]
                        Wx_use = Wx[: y_recent_all.shape[1]]
                    else:
                        Mx_use = Mx
                        Wx_use = Wx
                    
                    X_recent_std = (y_recent_all - Mx_use) / np.where(Wx_use == 0, 1.0, Wx_use)
                    
                    # Handle NaN/Inf values
                    X_recent_std = np.where(np.isfinite(X_recent_std), X_recent_std, np.nan)
                    
                    # Update factor state with recent data using .update() method
                    # Use history=None to use all provided recent data
                    logger.info(f"Updating {('DFM' if is_dfm else 'DDFM')} factor state with {len(y_recent_aligned)} recent periods (using all data)")
                    try:
                        dfm_model.update(X_recent_std, history=None)
                        logger.debug(f"Successfully updated {('DFM' if is_dfm else 'DDFM')} factor state")
                    except Exception as e:
                        logger.warning(f"Failed to update {('DFM' if is_dfm else 'DDFM')} factor state: {e}, continuing with training state")
            
            # If alignment failed or no recent data left, skip update gracefully
            if y_recent_aligned is None or len(y_recent_aligned) == 0:
                logger.warning("Skipping factor state update: no aligned recent data available after filtering to model features")
        
        # For ARIMA/VAR models, update with recent data for recursive prediction
        # ARIMA/VAR need recent values as input (e.g., AR(1) needs 2023-12 to predict 2024-01)
        # Based on config: ARIMA order=[1,1,1] (AR(1)), VAR lag_order=1 (VAR(1))
        elif is_arima or is_var:
            # Prepare recent data for ARIMA/VAR
            if isinstance(y_recent, pd.DataFrame):
                # For VAR, use all columns; for ARIMA, use target series only
                if is_arima and target_series is not None:
                    # ARIMA: univariate, use target series only
                    if target_series in y_recent.columns:
                        y_recent_aligned = y_recent[[target_series]].copy()
                    else:
                        logger.warning(f"Target series '{target_series}' not in y_recent, skipping update")
                        y_recent_aligned = None
                else:
                    # VAR: multivariate, use all columns matching training data
                    if isinstance(y_train, pd.DataFrame):
                        available_cols = [col for col in y_train.columns if col in y_recent.columns]
                        if len(available_cols) > 0:
                            y_recent_aligned = y_recent[available_cols].copy()
                            y_recent_aligned = y_recent_aligned[y_train.columns]
                        else:
                            logger.warning("No matching columns between y_recent and y_train, skipping update")
                            y_recent_aligned = None
                    else:
                        y_recent_aligned = y_recent.copy()
            else:
                # y_recent is Series
                if is_arima:
                    y_recent_aligned = y_recent.copy()
                else:
                    # VAR expects DataFrame
                    y_recent_aligned = y_recent.to_frame() if isinstance(y_recent, pd.Series) else pd.DataFrame(y_recent)
            
            if y_recent_aligned is not None and len(y_recent_aligned) > 0:
                # Check minimum required data length based on model order
                # ARIMA(1,1,1): AR order=1, so need at least 1 recent value
                # VAR(1): lag_order=1, so need at least 1 recent value
                min_required = 1  # Default minimum
                
                # Try to get actual order/lag_order from forecaster if available
                if is_arima:
                    # Try to get ARIMA order from forecaster
                    try:
                        if hasattr(forecaster, 'forecaster_') and hasattr(forecaster.forecaster_, 'order'):
                            ar_order = forecaster.forecaster_.order[0] if isinstance(forecaster.forecaster_.order, (list, tuple)) else forecaster.forecaster_.order
                            min_required = max(min_required, int(ar_order) if ar_order else 1)
                        elif hasattr(forecaster, '_fitted_forecaster') and hasattr(forecaster._fitted_forecaster, 'order'):
                            ar_order = forecaster._fitted_forecaster.order[0] if isinstance(forecaster._fitted_forecaster.order, (list, tuple)) else forecaster._fitted_forecaster.order
                            min_required = max(min_required, int(ar_order) if ar_order else 1)
                    except Exception:
                        pass  # Use default
                    model_info = f"ARIMA (order=[1,1,1] from config, need ≥{min_required} recent values)"
                else:  # VAR
                    # Try to get VAR lag_order from forecaster
                    try:
                        if hasattr(forecaster, 'maxlags'):
                            var_lags = int(forecaster.maxlags) if forecaster.maxlags else 1
                            min_required = max(min_required, var_lags)
                        elif hasattr(forecaster, '_fitted_forecaster') and hasattr(forecaster._fitted_forecaster, 'k_ar'):
                            var_lags = int(forecaster._fitted_forecaster.k_ar) if forecaster._fitted_forecaster.k_ar else 1
                            min_required = max(min_required, var_lags)
                    except Exception:
                        pass  # Use default
                    model_info = f"VAR (lag_order=1 from config, need ≥{min_required} recent values)"
                
                if len(y_recent_aligned) < min_required:
                    logger.warning(f"{model_info}: Only {len(y_recent_aligned)} recent periods available, but need ≥{min_required}. Using available data.")
                
                # Convert frequency from MS (MonthStart) to ME (MonthEnd) for sktime compatibility
                # sktime forecasters use ME frequency internally, but data is often resampled with MS
                try:
                    if hasattr(y_recent_aligned.index, 'freq') and y_recent_aligned.index.freq is not None:
                        freq_str = str(y_recent_aligned.index.freq)
                        if 'MS' in freq_str or 'MonthBegin' in freq_str:
                            # Convert MS to ME: convert to Period then back to timestamp with ME
                            y_recent_aligned = y_recent_aligned.copy()
                            y_recent_aligned.index = y_recent_aligned.index.to_period('M').to_timestamp('M')
                            logger.debug(f"Converted y_recent frequency from MS to ME for sktime compatibility")
                except Exception as e:
                    logger.debug(f"Frequency conversion not needed or failed: {e}")
                
                logger.info(f"Updating {model_info} with {len(y_recent_aligned)} recent periods for recursive prediction")
                try:
                    # Use sktime's update() method to provide recent data
                    # This updates the forecaster's internal state with recent values
                    # For ARIMA(1,1,1): uses last value (2023-12) to predict 2024-01
                    # For VAR(1): uses last value (2023-12) to predict 2024-01
                    forecaster.update(y_recent_aligned, update_params=False)
                    logger.debug(f"Successfully updated {('ARIMA' if is_arima else 'VAR')} with recent data")
                except Exception as e:
                    logger.warning(f"Failed to update {('ARIMA' if is_arima else 'VAR')} with recent data: {e}, continuing with training data")
    
    # Calculate horizons for prediction
    # For ARIMA/VAR: if y_recent was provided (gap data), use relative horizons directly
    # For DFM/DDFM: use absolute horizons from train_end
    horizons_arr = np.asarray(horizons)
    horizons_arr = np.sort(horizons_arr)  # Sort horizons
    
    if is_arima or is_var:
        # ARIMA/VAR: If y_recent was provided, gap is filled, so use relative horizons
        # horizons=[1,2,...,22] means test period months 1,2,...,22
        # After update() with gap data, cutoff is at test_start-1, so relative horizon 1 predicts test_start
        use_relative_horizons = (y_recent is not None and len(y_recent) > 0)
        
        if use_relative_horizons:
            # Gap data was provided via update(), so use relative horizons directly
            # cutoff is now at test_start-1 (e.g., 2023-12), so fh=[1] predicts test_start (2024-01)
            prediction_horizons = horizons_arr.tolist()
            logger.info(f"ARIMA/VAR: Using relative horizons {prediction_horizons} (gap data provided via update())")
        else:
            # No gap data provided, calculate absolute horizons from train_end
            if hasattr(y_train, 'index') and hasattr(y_test, 'index') and len(y_train) > 0 and len(y_test) > 0:
                train_max = y_train.index.max()
                test_min = y_test.index.min()
                months_to_test_start = (test_min.year - train_max.year) * 12 + (test_min.month - train_max.month)
                prediction_horizons = (months_to_test_start + horizons_arr - 1).astype(int).tolist()
                logger.info(f"ARIMA/VAR: No gap data provided, using absolute horizons {prediction_horizons} from train_end")
            else:
                prediction_horizons = horizons_arr.tolist()
                logger.warning("ARIMA/VAR: Could not calculate horizons, using relative horizons")
    else:
        # DFM/DDFM: use absolute horizons from train_end
        if hasattr(y_train, 'index') and hasattr(y_test, 'index') and len(y_train) > 0 and len(y_test) > 0:
            train_max = y_train.index.max()
            test_min = y_test.index.min()
            months_to_test_start = (test_min.year - train_max.year) * 12 + (test_min.month - train_max.month)
            prediction_horizons = (months_to_test_start + horizons_arr - 1).astype(int).tolist()
            logger.info(f"DFM/DDFM: Using absolute horizons {prediction_horizons} from train_end")
        else:
            prediction_horizons = horizons_arr.tolist()
            logger.warning("DFM/DDFM: Could not calculate horizons, using relative horizons")
    
    # Generate predictions
    # For ARIMA/VAR: recursive prediction (each horizon uses previous predictions)
    # For DFM/DDFM: batch prediction (state space model, can predict all at once)
    pred_dict = {}
    
    if is_arima or is_var:
        # ARIMA/VAR: recursive prediction - each horizon uses previous predictions
        # If gap data was provided: horizon 1 uses y_recent[-1] (2023-12) → predicts 2024-01
        # If no gap data: horizon 1 uses train_end (2019-12) → predicts 2020-01 (wrong!)
        model_name = 'ARIMA(1,1,1)' if is_arima else 'VAR(1)'
        logger.info(f"Recursive prediction for {model_name}: predicting {len(prediction_horizons)} horizons sequentially")
        
        for i, pred_h in enumerate(prediction_horizons):
            rel_h = int(horizons_arr[i])  # Original relative horizon (for result dict key)
            try:
                # Predict one step ahead (relative horizon 1 from current state)
                # For first horizon: uses y_recent[-1] (provided via update()) or train_end
                # For subsequent horizons: uses previous prediction (provided via update() after each prediction)
                y_pred_h = forecaster.predict(fh=[1])
                
                # Store prediction using relative horizon as key (for consistency with DFM/DDFM)
                pred_dict[rel_h] = y_pred_h
                
                # Update forecaster with this prediction for next horizon (recursive prediction)
                if i < len(prediction_horizons) - 1:  # Don't update after last prediction
                    try:
                        # Convert prediction to ME frequency if needed (sktime compatibility)
                        if hasattr(y_pred_h.index, 'freq') and y_pred_h.index.freq is not None:
                            freq_str = str(y_pred_h.index.freq)
                            if 'MS' in freq_str or 'MonthBegin' in freq_str:
                                y_pred_h = y_pred_h.copy()
                                y_pred_h.index = y_pred_h.index.to_period('M').to_timestamp('M')
                        
                        forecaster.update(y_pred_h, update_params=False)
                        logger.debug(f"Horizon {rel_h} (pred_h={pred_h}, {i+1}/{len(prediction_horizons)}): predicted and updated for next horizon")
                    except Exception as e:
                        logger.warning(f"Failed to update forecaster with prediction for horizon {rel_h}: {e}")
            except Exception as e:
                logger.warning(f"Recursive prediction for horizon {rel_h} (pred_h={pred_h}) failed: {e}")
                pred_dict[rel_h] = None
    else:
        # DFM/DDFM: use dfm-python predict() per horizon
        # Note: Factor state update was already done above (lines 183-252) if y_recent was provided
        from src.models.models_utils import _convert_predictions_to_dataframe
        dfm_model = forecaster._dfm_model if is_dfm else forecaster._ddfm_model

        # Predict per horizon (per dfm-python pattern)
        # Use absolute horizons for DFM/DDFM (from train_end)
        # Map absolute horizons to relative horizons for result storage
        for i, abs_h in enumerate(prediction_horizons):
            rel_h = int(horizons_arr[i])  # Original relative horizon
            try:
                raw_pred = dfm_model.predict(horizon=int(abs_h))
                pred_df = _convert_predictions_to_dataframe(raw_pred, getattr(forecaster, "_y", y_train), int(abs_h))
                # Store using relative horizon as key (for consistency)
                pred_dict[rel_h] = pred_df
            except Exception as e:
                logger.warning(f"Prediction for horizon {rel_h} (abs_h={abs_h}) failed: {e}")
                pred_dict[rel_h] = None
    
    # Calculate metrics per horizon by matching indices
    results = {}
    for i, h in enumerate(horizons_arr):
        h = int(h)
        if h <= 0:
            continue
        
        # Get prediction from batch prediction results
        # For ARIMA/VAR: pred_dict uses relative horizons as keys
        # For DFM/DDFM: pred_dict uses relative horizons as keys (mapped from absolute horizons)
        y_pred_h = pred_dict.get(h)
        if y_pred_h is None:
            logger.warning(f"Horizon {h}: No prediction available, skipping")
            results[h] = {
                'sMSE': np.nan, 'sMAE': np.nan, 'sRMSE': np.nan,
                'MSE': np.nan, 'MAE': np.nan, 'RMSE': np.nan,
                'sigma': np.nan, 'n_valid': 0
            }
            continue
        
        logger.info(f"Horizon {h}: Using prediction from batch, type={type(y_pred_h)}, shape={getattr(y_pred_h, 'shape', 'N/A')}, length={len(y_pred_h) if hasattr(y_pred_h, '__len__') else 'N/A'}")
        
        try:
            # VAR-specific stability check: detect unstable forecasts before calculating metrics
            # VAR models can become numerically unstable for long horizons, producing extreme values
            # Note: is_var was already detected above (lines 149-179)
            if is_var and h > 1:
                # Check if forecast values are extreme (indicating numerical instability)
                # Extract numeric values from y_pred_h
                pred_values = None
                if isinstance(y_pred_h, pd.DataFrame):
                    pred_values = y_pred_h.values.flatten()
                elif isinstance(y_pred_h, pd.Series):
                    pred_values = y_pred_h.values
                elif hasattr(y_pred_h, '__array__'):
                    # Convert torch tensors to numpy array (handle CUDA tensors)
                    try:
                        import torch
                        if isinstance(y_pred_h, torch.Tensor):
                            pred_values = y_pred_h.cpu().numpy().flatten()
                        else:
                            pred_values = np.asarray(y_pred_h).flatten()
                    except ImportError:
                        pred_values = np.asarray(y_pred_h).flatten()
                
                if pred_values is not None and len(pred_values) > 0:
                    max_abs_pred = np.max(np.abs(pred_values))
                    # Use consistent threshold with aggregation (1e10) for consistency
                    # This ensures that prediction values that would lead to extreme metrics
                    # are caught early during evaluation, matching the validation in aggregation
                    VAR_PREDICTION_THRESHOLD = 1e10
                    if max_abs_pred > VAR_PREDICTION_THRESHOLD:
                        logger.warning(
                            f"Horizon {h}: VAR forecast contains extreme values (max_abs={max_abs_pred:.2e} > {VAR_PREDICTION_THRESHOLD:.0e}). "
                            f"This indicates numerical instability. Marking metrics as NaN."
                        )
                        results[h] = {
                            'sMSE': np.nan, 'sMAE': np.nan, 'sRMSE': np.nan,
                            'MSE': np.nan, 'MAE': np.nan, 'RMSE': np.nan,
                            'sigma': np.nan, 'n_valid': 0
                        }
                        continue
                    # Check for NaN or Inf in predictions
                    if np.any(~np.isfinite(pred_values)):
                        logger.warning(
                            f"Horizon {h}: VAR forecast contains NaN or Inf values. "
                            f"Marking metrics as NaN."
                        )
                        results[h] = {
                            'sMSE': np.nan, 'sMAE': np.nan, 'sRMSE': np.nan,
                            'MSE': np.nan, 'MAE': np.nan, 'RMSE': np.nan,
                            'sigma': np.nan, 'n_valid': 0
                        }
                        continue
            
            # Extract corresponding test data point using position-based matching
            # Relative horizon h corresponds to position h-1 in test data (0-indexed)
            test_pos = h - 1
            
            # Check if test data has enough points for this horizon
            if test_pos >= len(y_test):
                logger.warning(f"Horizon {h}: test_pos {test_pos} >= y_test length {len(y_test)}. Skipping horizon {h} - test set too small.")
                results[h] = {
                    'sMSE': np.nan, 'sMAE': np.nan, 'sRMSE': np.nan,
                    'MSE': np.nan, 'MAE': np.nan, 'RMSE': np.nan,
                    'sigma': np.nan, 'n_valid': 0
                }
                continue
            
            logger.info(f"Horizon {h}: test_pos={test_pos}, y_test length={len(y_test)}, y_test type={type(y_test)}, y_test shape={getattr(y_test, 'shape', 'N/A')}")
            
            # Extract prediction value(s) - handle both Series and DataFrame
            # For predict(fh=[h]), we want the prediction at horizon h
            # If it returns multiple predictions, take the last one (should be for horizon h)
            # If it returns a single prediction, use it directly
            if isinstance(y_pred_h, pd.DataFrame):
                if len(y_pred_h) > 0:
                    # Always take the last row (should be the prediction for horizon h)
                    y_pred_h = y_pred_h.iloc[-1:].copy()
                    # Extract target series if specified
                    if target_series is not None and target_series in y_pred_h.columns:
                        y_pred_h = y_pred_h[[target_series]]
                    elif target_series is not None and isinstance(y_test, pd.DataFrame) and target_series in y_test.columns:
                        # target_series not in y_pred_h.columns but in y_test.columns
                        # Use column index from y_test to extract from y_pred_h
                        col_idx = y_test.columns.get_loc(target_series)
                        if col_idx < y_pred_h.shape[1]:
                            y_pred_h = y_pred_h.iloc[:, [col_idx]]
                            logger.debug(f"Horizon {h}: Extracted target_series '{target_series}' from y_pred_h using column index {col_idx} (from y_test.columns)")
                        else:
                            logger.warning(f"Horizon {h}: Column index {col_idx} for target_series '{target_series}' out of bounds for y_pred_h (shape: {y_pred_h.shape})")
                            y_pred_h = pd.DataFrame()
                    elif isinstance(y_test, pd.DataFrame) and len(y_test.columns) == 1:
                        # Single column, use it
                        y_pred_h = y_pred_h.iloc[:, [0]]
                    elif target_series is not None:
                        # target_series specified but not found in either y_pred_h or y_test
                        logger.warning(f"Horizon {h}: target_series '{target_series}' not found in y_pred_h.columns={list(y_pred_h.columns)} or y_test.columns={list(y_test.columns) if isinstance(y_test, pd.DataFrame) else 'N/A'}")
                        # Try to keep all columns and let calculate_standardized_metrics handle it
                        pass
                else:
                    logger.warning(f"Horizon {h}: y_pred_h DataFrame is empty.")
                    y_pred_h = pd.DataFrame()
            elif isinstance(y_pred_h, pd.Series):
                if len(y_pred_h) > 0:
                    # Always take the last value (should be the prediction for horizon h)
                    y_pred_h = y_pred_h.iloc[-1:]
                else:
                    logger.warning(f"Horizon {h}: y_pred_h Series is empty.")
                    y_pred_h = pd.Series()
            else:
                # Handle numpy array or other types
                if hasattr(y_pred_h, '__len__') and len(y_pred_h) > 0:
                    # Convert to Series for consistent handling, take last value
                    y_pred_h = pd.Series([y_pred_h[-1]])
                else:
                    logger.warning(f"Horizon {h}: y_pred_h is empty or unsupported type: {type(y_pred_h)}")
                    y_pred_h = pd.Series()
            
            # Extract corresponding test data (test_pos already validated above)
            if isinstance(y_test, pd.DataFrame):
                y_true_h = y_test.iloc[test_pos:test_pos+1].copy()
                # Extract target series if specified
                if target_series is not None and target_series in y_true_h.columns:
                    y_true_h = y_true_h[[target_series]]
                elif len(y_test.columns) == 1:
                    # Single column, use it
                    y_true_h = y_true_h.iloc[:, [0]]
            elif isinstance(y_test, pd.Series):
                y_true_h = y_test.iloc[test_pos:test_pos+1]
            else:
                y_true_h = y_test[test_pos:test_pos+1]
            
            # Check if we have valid data
            has_pred = len(y_pred_h) > 0 if hasattr(y_pred_h, '__len__') else (y_pred_h.size > 0 if hasattr(y_pred_h, 'size') else False)
            has_true = len(y_true_h) > 0 if hasattr(y_true_h, '__len__') else (y_true_h.size > 0 if hasattr(y_true_h, 'size') else False)
            
            logger.info(f"Horizon {h}: After extraction - has_pred={has_pred}, has_true={has_true}")
            
            # Align shapes if both are valid
            if has_pred and has_true:
                if isinstance(y_pred_h, pd.DataFrame) and isinstance(y_true_h, pd.DataFrame):
                    # Align columns: use common columns
                    common_cols = [col for col in y_pred_h.columns if col in y_true_h.columns]
                    if len(common_cols) > 0:
                        y_pred_h = y_pred_h[common_cols].copy()
                        y_true_h = y_true_h[common_cols].copy()
                    elif y_pred_h.shape[1] != y_true_h.shape[1]:
                        logger.warning(f"Horizon {h}: Shape mismatch - cannot align columns")
                        has_pred = False
                elif isinstance(y_pred_h, pd.DataFrame) and isinstance(y_true_h, pd.Series):
                    y_true_h = y_true_h.to_frame()
                elif isinstance(y_pred_h, pd.Series) and isinstance(y_true_h, pd.DataFrame):
                    y_pred_h = y_pred_h.to_frame()
            
            if has_pred and has_true:
                try:
                    # Fix: If y_true_h is Series (not DataFrame), target_series must be None or int, not string
                    # When y_test is Series, we don't need to specify target_series since there's only one column
                    target_series_for_metrics = target_series
                    if isinstance(y_true_h, pd.Series) and isinstance(target_series, str):
                        # y_true_h is Series but target_series is string - set to None
                        target_series_for_metrics = None
                        logger.debug(f"Horizon {h}: y_true_h is Series, setting target_series=None (was string: {target_series})")
                    elif isinstance(y_true_h, pd.DataFrame) and len(y_true_h.columns) == 1:
                        # Single column DataFrame - can use None or column name
                        if isinstance(target_series, str) and target_series not in y_true_h.columns:
                            # target_series string doesn't match column name - set to None
                            target_series_for_metrics = None
                            logger.debug(f"Horizon {h}: y_true_h has 1 column but target_series '{target_series}' not in columns, setting target_series=None")
                    
                    # VAR-specific check for horizon 1: detect if VAR is just predicting persistence
                    # (last training value), which would result in suspiciously good results
                    var_persistence_detected = False
                    if is_var and h == 1:  # Note: is_var was already detected above (lines 149-179)
                        # Extract last training value for comparison
                        if isinstance(y_train, pd.DataFrame):
                            if target_series and target_series in y_train.columns:
                                last_train_val = y_train[target_series].iloc[-1]
                                train_std = y_train[target_series].std()
                            elif len(y_train.columns) == 1:
                                last_train_val = y_train.iloc[-1, 0]
                                train_std = y_train.iloc[:, 0].std()
                            else:
                                last_train_val = None
                                train_std = None
                        elif isinstance(y_train, pd.Series):
                            last_train_val = y_train.iloc[-1]
                            train_std = y_train.std()
                        else:
                            last_train_val = None
                            train_std = None
                        
                        # Extract prediction value
                        if isinstance(y_pred_h, pd.DataFrame):
                            if target_series and target_series in y_pred_h.columns:
                                pred_val = y_pred_h[target_series].iloc[0]
                            elif len(y_pred_h.columns) == 1:
                                pred_val = y_pred_h.iloc[0, 0]
                            else:
                                pred_val = None
                        elif isinstance(y_pred_h, pd.Series):
                            pred_val = y_pred_h.iloc[0]
                        else:
                            pred_val = None
                        
                        # Check if prediction is very close to last training value (persistence)
                        # Use both relative difference and absolute difference normalized by std
                        if last_train_val is not None and pred_val is not None:
                            if np.isfinite(last_train_val) and np.isfinite(pred_val):
                                abs_diff = abs(pred_val - last_train_val)
                                rel_diff = abs_diff / (abs(last_train_val) + 1e-10)
                                
                                # Check relative difference (original check)
                                if rel_diff < 1e-6:
                                    var_persistence_detected = True
                                    logger.warning(
                                        f"Horizon {h}: VAR prediction ({pred_val:.6f}) is essentially identical to last training value "
                                        f"({last_train_val:.6f}), suggesting VAR is predicting persistence. "
                                        f"Marking metrics as NaN due to model limitation."
                                    )
                                # Also check if absolute difference is very small compared to training std
                                # This catches cases where VAR predicts something very close to last value
                                # even if relative difference is larger (e.g., for small values)
                                elif train_std is not None and train_std > 0:
                                    std_normalized_diff = abs_diff / (train_std + 1e-10)
                                    if std_normalized_diff < 1e-4:  # Very small compared to std
                                        var_persistence_detected = True
                                        logger.warning(
                                            f"Horizon {h}: VAR prediction ({pred_val:.6f}) is very close to last training value "
                                            f"({last_train_val:.6f}), with std-normalized difference {std_normalized_diff:.2e}. "
                                            f"Marking metrics as NaN due to persistence prediction."
                                        )
                    
                    metrics = calculate_standardized_metrics(
                        y_true_h, y_pred_h, y_train=y_train, target_series=target_series_for_metrics
                    )
                    
                    # If VAR persistence detected, mark all metrics as NaN
                    if var_persistence_detected:
                        metrics = {
                            'sMSE': np.nan, 'sMAE': np.nan, 'sRMSE': np.nan,
                            'MSE': np.nan, 'MAE': np.nan, 'RMSE': np.nan,
                            'sigma': metrics.get('sigma', np.nan),
                            'n_valid': 0
                        }
                    
                    results[h] = metrics
                except Exception as e:
                    logger.warning(f"Horizon {h}: Error calculating metrics: {e}")
                    results[h] = {
                        'sMSE': np.nan, 'sMAE': np.nan, 'sRMSE': np.nan,
                        'MSE': np.nan, 'MAE': np.nan, 'RMSE': np.nan,
                        'sigma': np.nan, 'n_valid': 0
                    }
            else:
                logger.warning(f"Horizon {h}: Missing data - has_pred={has_pred}, has_true={has_true}")
                results[h] = {
                    'sMSE': np.nan, 'sMAE': np.nan, 'sRMSE': np.nan,
                    'MSE': np.nan, 'MAE': np.nan, 'RMSE': np.nan,
                    'sigma': np.nan, 'n_valid': 0
                }
        except Exception as e:
            # If prediction fails for this horizon, return NaN metrics
            logger.warning(f"Horizon {h}: Prediction failed with error: {e}")
            results[h] = {
                'sMSE': np.nan, 'sMAE': np.nan, 'sRMSE': np.nan,
                'MSE': np.nan, 'MAE': np.nan, 'RMSE': np.nan,
                'sigma': np.nan, 'n_valid': 0
            }
    
    return results
