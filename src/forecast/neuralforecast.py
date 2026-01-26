"""Forecasting functions for NeuralForecast models (PatchTST, TFT, iTransformer, TimeMixer)."""

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
import numpy as np

from src.utils import (
    get_target_series_from_model,
    filter_and_prepare_test_data,
    load_model_checkpoint,
    interpolate_missing_values,
    convert_to_neuralforecast_format,
    extract_neuralforecast_forecasts,
)
from src.train._common import get_processed_data_from_loader

logger = logging.getLogger(__name__)

def _ensure_primary_target_in_list(
    all_variables: List[str],
    available_targets: List[str],
    primary_target: Optional[str]
) -> Tuple[List[str], Optional[str]]:
    """Ensure primary target exists in available_targets for evaluation.

    If primary_target is missing, replace the last entry to keep n_series count.
    Falls back to the first available target if primary_target isn't in data.
    """
    if not primary_target:
        return available_targets, None
    if primary_target not in available_targets:
        if primary_target in all_variables:
            # Replace the last element to keep the expected n_series size.
            available_targets = available_targets[:-1] + [primary_target]
            logger.info(
                f"Multivariate model: Added primary target {primary_target} to available_targets "
                f"(replaced last element)"
            )
        else:
            logger.warning(
                f"Multivariate model: Config target_series {primary_target} not in test_data, "
                f"using {available_targets[0]} for evaluation"
            )
            primary_target = available_targets[0] if available_targets else None
    return available_targets, primary_target

def _resolve_forecast_targets(
    nf_model: Any,
    test_data: pd.DataFrame,
    target_series: Optional[List[str]],
    hist_exog_list: Optional[List[str]],
    is_multivariate: bool,
    model_targets: Optional[List[str]] = None
) -> Tuple[List[str], Optional[str]]:
    """Resolve available targets and primary target for evaluation.

    Keeps consistent logic across recursive and multi-horizon forecasting.
    """
    if is_multivariate:
        date_cols = {'date', 'date_w', 'date_m', 'Date', 'Date_w', 'Date_m', 'year', 'month', 'day'}
        all_variables = [col for col in test_data.columns if col not in date_cols]
        # Prefer the model's target ordering if available (ensures training/eval alignment)
        if model_targets:
            ordered_targets = [t for t in model_targets if t in all_variables]
            if ordered_targets:
                all_variables = ordered_targets
        model = nf_model.models[0]
        n_series = getattr(model, 'n_series', len(all_variables))
        if n_series > len(all_variables):
            logger.warning(
                f"Model expects {n_series} series but test_data only has {len(all_variables)}. "
                f"Using all available."
            )
        available_targets = all_variables[:n_series] if n_series <= len(all_variables) else all_variables
        primary_target = target_series[0] if target_series else None
        available_targets, primary_target = _ensure_primary_target_in_list(
            all_variables, available_targets, primary_target
        )
        if primary_target:
            logger.info(
                f"Multivariate model: Using {len(available_targets)} variables for prediction, "
                f"evaluating on {primary_target} (from config)"
            )
        else:
            logger.warning(
                "Multivariate model: No valid primary target found, evaluation may be skipped."
            )
        return available_targets, primary_target

    # TFT or univariate: primary target + covariates
    available_targets = get_target_series_from_model(nf_model, test_data, target_series)
    primary_target = available_targets[0] if available_targets else None
    if hist_exog_list:
        logger.info(
            f"TFT model: Primary target={available_targets}, "
            f"Covariates={len(hist_exog_list)} variables"
        )
    return available_targets, primary_target


def _extract_model_info(nf_model: Any) -> Tuple[List[str], Optional[List[str]], bool]:
    """Extract target series and covariate information from NeuralForecast model.
    
    Parameters
    ----------
    nf_model : Any
        NeuralForecast model instance
    
    Returns
    -------
    tuple
        (all_target_series, hist_exog_list, is_multivariate)
        - all_target_series: All series used by the model (targets + covariates for multivariate)
        - hist_exog_list: List of historical exogenous variables (for TFT), None if not applicable
        - is_multivariate: True if model uses multivariate forecasting (all as targets)
    """
    if not hasattr(nf_model, 'models') or len(nf_model.models) == 0:
        return [], None, False
    
    model = nf_model.models[0]
    
    # Check for hist_exog_list (TFT uses covariates)
    hist_exog_list = getattr(model, 'hist_exog_list', None)
    hist_exog_size = getattr(model, 'hist_exog_size', 0)
    
    # Check n_series to determine if multivariate
    n_series = getattr(model, 'n_series', 1)
    is_multivariate = (n_series > 1) or (hist_exog_size == 0 and hasattr(model, 'n_series'))
    
    # Try to extract target series from model's _y attribute
    all_target_series = []
    if hasattr(nf_model, '_y') and nf_model._y is not None:
        if isinstance(nf_model._y, pd.DataFrame):
            all_target_series = list(nf_model._y.columns)
        elif isinstance(nf_model._y, pd.Series):
            all_target_series = [nf_model._y.name] if nf_model._y.name else ['target']
    
    # If TFT with covariates, we need to know which are targets vs covariates
    # For multivariate models, all variables are targets
    if hist_exog_list and len(hist_exog_list) > 0:
        # TFT: primary target + covariates
        is_multivariate = False
    elif n_series > 1:
        # Multivariate: all variables are targets
        is_multivariate = True
    
    return all_target_series, hist_exog_list, is_multivariate


def forecast(
    checkpoint_path: Path,
    horizon: int,
    model_type: str,
    recursive: bool = False,
    test_data: Optional[pd.DataFrame] = None,
    update_params: bool = False
) -> None:
    """Load trained model and generate forecasts.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to model checkpoint
    horizon : int
        Forecast horizon
    model_type : str
        Model type ('patchtst', 'tft', 'itf', 'itransformer', 'timemixer')
    recursive : bool, default False
        Whether to use recursive forecasting
    test_data : pd.DataFrame, optional
        Test data for recursive forecasting
    update_params : bool, default False
        Whether to update model parameters (retrain) during recursive forecasting
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    if recursive and test_data is None:
        raise ValueError("test_data is required when recursive=True")
    
    logger.info(f"Loading {model_type.upper()} model from: {checkpoint_path}")
    
    if recursive:
        run_recursive_forecast_neuralforecast(
            checkpoint_path, test_data, horizon, model_type, update_params
        )
    else:
        _forecast_neuralforecast_models(checkpoint_path, horizon, model_type)


def run_recursive_forecast(
    checkpoint_path: Path,
    test_data: pd.DataFrame,
    start_date: str,
    end_date: str,
    model_type: str,
    target_series: Optional[List[str]] = None,
    data_loader: Optional[Any] = None,
    update_params: bool = False
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, List[str]]:
    """Run recursive forecasting experiment with weekly updates.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to trained model checkpoint
    test_data : pd.DataFrame
        Test data with datetime index
    start_date : str
        Start date for experiment (YYYY-MM-DD)
    end_date : str
        End date for experiment (YYYY-MM-DD)
    model_type : str
        Model type ('patchtst', 'tft', 'itf', 'itransformer', 'timemixer')
    target_series : list, optional
        List of target series names
    data_loader : optional
        Data loader object
    update_params : bool, default False
        If False, only move cutoff (no retraining). If True, full retraining.
    
    Returns
    -------
    tuple
        (predictions, actuals, dates, target_series)
    """
    return run_recursive_forecast_neuralforecast(
        checkpoint_path, test_data, horizon=1,
        model_type=model_type, update_params=update_params,
        start_date=start_date, end_date=end_date,
        target_series=target_series, data_loader=data_loader
    )


def run_multi_horizon_forecast(
    checkpoint_path: Path,
    horizons: List[int],
    start_date: str,
    test_data: Optional[pd.DataFrame] = None,
    model_type: str = "patchtst",
    target_series: Optional[List[str]] = None,
    data_loader: Optional[Any] = None,
    return_weekly_forecasts: bool = False
) -> Tuple[Dict[int, Any], List[str]]:
    """Run multi-horizon forecasting from fixed start point.
    
    Uses ONE model (trained with max horizon) to predict all horizons.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to trained model checkpoint (should be trained with prediction_length >= max(horizons))
    horizons : list
        List of forecast horizons (weeks), e.g. [4, 8, 12, ..., 40]
    start_date : str
        Fixed start date for all forecasts (YYYY-MM-DD)
    test_data : pd.DataFrame, optional
        Test data for extracting actuals
    model_type : str, default "patchtst"
        Model type ('patchtst', 'tft', 'itf', 'itransformer', 'timemixer')
    target_series : list, optional
        List of target series names
    data_loader : optional
        Data loader object
    return_weekly_forecasts : bool, default False
        If True, return weekly forecasts within each horizon month
    
    Returns
    -------
    tuple
        (horizon_forecasts dict, target_series list)
    """
    return _run_multi_horizon_forecast_neuralforecast(
        checkpoint_path, horizons, start_date, test_data,
        model_type=model_type, target_series=target_series,
        data_loader=data_loader, return_weekly_forecasts=return_weekly_forecasts
    )


def run_recursive_forecast_neuralforecast(
    checkpoint_path: Path,
    test_data: pd.DataFrame,
    horizon: int,
    model_type: str,
    update_params: bool = False,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    target_series: Optional[List[str]] = None,
    data_loader: Optional[Any] = None
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, List[str]]:
    """Run recursive forecasting for NeuralForecast models.
    
    NeuralForecast uses long format (unique_id, ds, y) and requires rebuilding
    the dataset for each prediction step.
    
    When update_params=True, training data from data_loader is included to provide
    sufficient historical context for retraining, especially important for models
    like iTransformer that use normalization statistics.
    """
    # Load model using helper
    logger.info(f"Loading {model_type.upper()} model for recursive forecasting...")
    nf_model = load_model_checkpoint(checkpoint_path)
    
    # Extract model information (targets, covariates, multivariate flag)
    model_targets, hist_exog_list, is_multivariate = _extract_model_info(nf_model)
    
    # Resolve targets for evaluation
    available_targets, primary_target = _resolve_forecast_targets(
        nf_model, test_data, target_series, hist_exog_list, is_multivariate, model_targets
    )
    
    # Convert dates to timestamps
    start_ts = pd.Timestamp(start_date) if start_date else test_data.index.min()
    end_ts = pd.Timestamp(end_date) if end_date else test_data.index.max()
    
    # Prepare data for model updates - include training data for sufficient context
    # Combine training data with test_data to provide historical context for first predictions
    if data_loader is not None and hasattr(data_loader, 'training_data') and data_loader.training_data is not None:
        # Get training data in original scale (same as what model was trained on)
        training_data_original = get_processed_data_from_loader(
            data_loader.training_data, data_loader, model_type.upper()
        )
        # CRITICAL: Filter available_targets to only include columns present in both training and test data
        # This must be done BEFORE using available_targets, to ensure we have exactly n_series columns
        model = nf_model.models[0]
        n_series = getattr(model, 'n_series', len(available_targets))
        
        # Get ALL common columns (exist in both datasets) - this is the source of truth
        date_cols = {'date', 'date_w', 'date_m', 'Date', 'Date_w', 'Date_m', 'year', 'month', 'day'}
        all_test_cols = [col for col in test_data.columns if col not in date_cols]
        all_train_cols = [col for col in training_data_original.columns if col not in date_cols]
        all_common_cols = [col for col in all_test_cols if col in all_train_cols]
        
        # Use common columns, ensuring we have exactly n_series
        if len(all_common_cols) >= n_series:
            # Prefer model_targets order if available, otherwise use common columns in their order
            if model_targets and len(model_targets) > 0:
                # Filter model_targets to only those in common_cols, preserving order
                model_targets_filtered = [col for col in model_targets if col in all_common_cols]
                if len(model_targets_filtered) >= n_series:
                    available_targets = model_targets_filtered[:n_series]
                    logger.info(f"Using {len(available_targets)} columns from model_targets (exact training order)")
                else:
                    available_targets = all_common_cols[:n_series]
                    logger.info(f"Using {len(available_targets)} common columns (model_targets had {len(model_targets_filtered)})")
            else:
                available_targets = all_common_cols[:n_series]
                logger.info(f"Using {len(available_targets)} common columns (no model_targets available)")
        else:
            available_targets = all_common_cols
            logger.warning(f"Not enough common columns ({len(all_common_cols)}) for model's n_series ({n_series}). Model may fail.")
        
        # Final validation
        if len(available_targets) != n_series:
            logger.error(f"Column count mismatch: available_targets has {len(available_targets)} columns but model expects {n_series}")
        missing = [col for col in available_targets if col not in training_data_original.columns or col not in test_data.columns]
        if missing:
            logger.error(f"Some available_targets missing from datasets: {missing}")
    
    # Filter test data to experiment date range (after available_targets is finalized)
    test_data_for_forecasts, _, _ = filter_and_prepare_test_data(
        test_data, start_date, end_date, available_targets
    )
    
    # Continue with data preparation
    if data_loader is not None and hasattr(data_loader, 'training_data') and data_loader.training_data is not None:
        # Get training data in original scale (same as what model was trained on)
        training_data_original = get_processed_data_from_loader(
            data_loader.training_data, data_loader, model_type.upper()
        )
        
        # Combine training + test data
        if is_multivariate:
            full_data_for_updates = pd.concat([
                training_data_original[available_targets],
                test_data[available_targets]
            ]).sort_index()
        else:
            if hist_exog_list:
                all_cols = available_targets + [col for col in hist_exog_list if col in test_data.columns]
                training_cols = [col for col in all_cols if col in training_data_original.columns and col in test_data.columns]
                test_cols = [col for col in all_cols if col in test_data.columns]
                full_data_for_updates = pd.concat([
                    training_data_original[training_cols],
                    test_data[test_cols]
                ]).sort_index()
            else:
                full_data_for_updates = pd.concat([
                    training_data_original[available_targets],
                    test_data[available_targets]
                ]).sort_index()
        
        # CRITICAL: Update full_data_for_updates with observed values from test_data
        # This ensures we use actual observed values instead of NaNs where available
        for date_idx in full_data_for_updates.index:
            if date_idx in test_data.index:
                for col_name in available_targets:
                    if col_name in test_data.columns:
                        observed_val = test_data.loc[date_idx, col_name]
                        if not pd.isna(observed_val):
                            full_data_for_updates.loc[date_idx, col_name] = observed_val
        
        logger.info(f"Combined training data ({len(training_data_original)}) with test data ({len(test_data)}) for forecasting context")
        logger.info(f"Updated full_data_for_updates with observed values from test_data")
    else:
        # Fallback: use only test_data if training data not available
        if is_multivariate:
            # Multivariate: use all variables
            full_data_for_updates = test_data[available_targets].copy()
        else:
            # TFT: use primary target + covariates
            if hist_exog_list:
                # Include covariates
                all_cols = available_targets + [col for col in hist_exog_list if col in test_data.columns]
                full_data_for_updates = test_data[all_cols].copy()
            else:
                # Only primary target
                full_data_for_updates = test_data[available_targets].copy()
        logger.warning("Training data not available - using only test_data (may cause poor first predictions)")
    
    # Use actual weekly dates from filtered test data
    weekly_dates = test_data_for_forecasts.index[(test_data_for_forecasts.index >= start_ts) & 
                                                   (test_data_for_forecasts.index <= end_ts)]
    weekly_dates = weekly_dates.sort_values()
    
    if len(weekly_dates) < 2:
        raise ValueError(f"Not enough weekly dates. Found {len(weekly_dates)} dates.")
    
    logger.info(f"Running recursive {model_type.upper()} forecast from {start_ts.date()} to {end_ts.date()} ({len(weekly_dates)} weeks)")
    
    predictions = []
    actuals = []
    forecast_dates = []
    
    # Loop over weekly cutoffs
    for i in range(len(weekly_dates) - 1):
        cutoff_date = weekly_dates[i]
        next_date = weekly_dates[i + 1]
        
        # Get data up to cutoff from FULL combined data (training + test)
        # Note: full_data_for_updates already has observed values from test_data (updated above)
        # Use data up to cutoff_date (not including next_date, as we're predicting it)
        train_data_up_to_cutoff = full_data_for_updates[full_data_for_updates.index <= cutoff_date].copy()
        
        # Impute missing values
        train_data_up_to_cutoff = interpolate_missing_values(train_data_up_to_cutoff, data_loader)
        
        # Convert to NeuralForecast long format
        if is_multivariate:
            # Multivariate: all variables as targets
            nf_df = convert_to_neuralforecast_format(train_data_up_to_cutoff, available_targets)
        else:
            # TFT: primary target + covariates
            if hist_exog_list and len(hist_exog_list) > 0:
                # Separate target and covariates
                target_data = train_data_up_to_cutoff[available_targets].copy()
                covariate_cols = [col for col in hist_exog_list if col in train_data_up_to_cutoff.columns]
                covariate_data = train_data_up_to_cutoff[covariate_cols].copy() if covariate_cols else None
                nf_df = convert_to_neuralforecast_format(
                    target_data, available_targets,
                    covariate_data=covariate_data,
                    covariate_names=covariate_cols
                )
            else:
                # Only primary target
                nf_df = convert_to_neuralforecast_format(train_data_up_to_cutoff, available_targets)
        
        # Predict
        forecast_df = nf_model.predict(df=nf_df)
        
        # Extract 1-step-ahead forecasts
        # For multivariate models, extract all targets but we only care about primary for evaluation
        # For TFT, extract the primary target
        # primary_target is already set above (from config for multivariate, from available_targets for TFT)
        if primary_target:
            # Extract forecast for primary target (from config for multivariate, first in available_targets for TFT)
            # For multivariate, the model predicts all series, but we extract only primary
            forecast_values = extract_neuralforecast_forecasts(forecast_df, [primary_target], horizon_idx=0)
            predictions.append(forecast_values)
        else:
            logger.warning(f"No primary target found for {next_date.date()}")
            predictions.append(np.array([np.nan]))
        
        forecast_dates.append(next_date)
        
        # Get actual values (only for primary target)
        if next_date in test_data_for_forecasts.index and primary_target:
            actual_values = test_data_for_forecasts.loc[next_date, [primary_target]].values
            actuals.append(actual_values)
            
            # Update full_data_for_updates with observed values for next iteration
            # This ensures subsequent predictions use the most recent observed data
            if next_date in full_data_for_updates.index:
                for col_idx, col_name in enumerate(available_targets):
                    if col_name in test_data_for_forecasts.columns:
                        observed_val = test_data_for_forecasts.loc[next_date, col_name]
                        if not pd.isna(observed_val):
                            full_data_for_updates.loc[next_date, col_name] = observed_val
                            logger.debug(f"Updated {col_name} at {next_date.date()} with observed value {observed_val:.2f}")
        else:
            logger.warning(f"No actual data for {next_date.date()}")
            actuals.append(np.full(1, np.nan))
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    forecast_dates = pd.DatetimeIndex(forecast_dates)
    
    # Return primary target series (from config for multivariate, first in available_targets for TFT)
    primary_target_list = [primary_target] if primary_target else []
    
    return predictions, actuals, forecast_dates, primary_target_list


def _forecast_neuralforecast_models(checkpoint_path: Path, horizon: int, model_type: str) -> None:
    """Forecast using NeuralForecast models (PatchTST, TFT, iTransformer, TimeMixer).
    
    Note: NeuralForecast models require data (df) for prediction. This simple mode
    logs that the model is ready for forecasting. Use experiment modes for actual forecasts.
    """
    from src.utils import load_model_checkpoint
    
    logger.info(f"Loading {model_type.upper()} model...")
    nf_model = load_model_checkpoint(checkpoint_path)
    
    logger.info(f"Model loaded successfully. Model is ready for forecasting with horizon={horizon}.")
    logger.info("Note: NeuralForecast models require data (df) for prediction. "
                "Use experiment modes (short_term/long_term) for actual forecasts with data.")
    
    # Save a placeholder to indicate forecast mode was run
    output_dir = checkpoint_path.parent / "forecasts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple status file
    status_file = output_dir / "forecast_status.txt"
    with open(status_file, 'w') as f:
        f.write(f"Model: {model_type}\n")
        f.write(f"Horizon: {horizon}\n")
        f.write(f"Status: Model loaded and ready for forecasting\n")
        f.write(f"Note: Use experiment modes for actual forecasts with data\n")
    
    logger.info(f"Forecast status saved to: {output_dir}")


def _run_multi_horizon_forecast_recursive(
    checkpoint_path: Path,
    horizons: List[int],
    start_date: str,
    test_data: Optional[pd.DataFrame] = None,
    model_type: str = "patchtst",
    target_series: Optional[List[str]] = None,
    data_loader: Optional[Any] = None,
    return_weekly_forecasts: bool = False
) -> Tuple[Dict[int, Any], List[str]]:
    """Run multi-horizon forecasting using recursive approach for models with h=1.
    
    For models trained with prediction_length=1, we recursively predict step-by-step
    to reach longer horizons.
    """
    from src.utils import (
        get_target_series_from_model,
        load_model_checkpoint,
        interpolate_missing_values,
        convert_to_neuralforecast_format,
        extract_neuralforecast_forecasts,
    )
    from src.train._common import get_processed_data_from_loader
    
    # Load model
    logger.info(f"Loading {model_type.upper()} model for recursive multi-horizon forecasting from {start_date}")
    model = load_model_checkpoint(checkpoint_path)
    
    # Extract model information
    model_targets, hist_exog_list, is_multivariate = _extract_model_info(model)
    
    start_date_ts = pd.Timestamp(start_date)
    forecasts = {}
    
    # Get actual target series
    if test_data is not None:
        available_targets, primary_target = _resolve_forecast_targets(
            model, test_data, target_series, hist_exog_list, is_multivariate, model_targets
        )
    else:
        available_targets = None
        primary_target = None
    
    if test_data is None or not available_targets:
        logger.warning("No test_data or available_targets for recursive multi-horizon forecasting")
        return {}, []
    
    # Prepare initial data: combine training + test data up to start_date
    if data_loader is not None:
        # Get training data in original scale
        training_data_original = get_processed_data_from_loader(
            test_data, data_loader, model_type.upper()
        )
        # Combine training + test data
        full_data = pd.concat([training_data_original, test_data]).sort_index()
        full_data = full_data[~full_data.index.duplicated(keep='last')].sort_index()
    else:
        full_data = test_data
    
    # For each horizon, recursively predict step-by-step
    for horizon in horizons:
        # Get data up to start_date
        data_up_to_start = full_data[full_data.index < start_date_ts][available_targets].copy()
        data_up_to_start = interpolate_missing_values(data_up_to_start, data_loader)
        
        # Convert to NeuralForecast format
        if is_multivariate:
            nf_df = convert_to_neuralforecast_format(data_up_to_start, available_targets)
        else:
            if hist_exog_list and len(hist_exog_list) > 0:
                all_cols = available_targets + [col for col in hist_exog_list if col in full_data.columns]
                train_data_full = full_data[full_data.index < start_date_ts][all_cols].copy()
                train_data_full = interpolate_missing_values(train_data_full, data_loader)
                target_data = train_data_full[available_targets].copy()
                covariate_cols = [col for col in hist_exog_list if col in train_data_full.columns]
                covariate_data = train_data_full[covariate_cols].copy() if covariate_cols else None
                nf_df = convert_to_neuralforecast_format(
                    target_data, available_targets,
                    covariate_data=covariate_data,
                    covariate_names=covariate_cols
                )
            else:
                nf_df = convert_to_neuralforecast_format(data_up_to_start, available_targets)
        
        # Build up predictions recursively
        current_data = nf_df.copy()
        weekly_predictions = []
        weekly_dates = []
        
        for step in range(horizon):
            # Predict 1 step ahead
            forecast_df = model.predict(df=current_data)
            
            # Extract the 1-step forecast (horizon_idx=0)
            step_forecast = extract_neuralforecast_forecasts(forecast_df, available_targets, horizon_idx=0)
            step_date = start_date_ts + pd.Timedelta(weeks=step + 1)
            
            weekly_predictions.append(step_forecast)
            weekly_dates.append(step_date)
            
            # Update current_data with the prediction for next step (recursive)
            # Append the forecast to the data for next prediction
            if len(step_forecast) > 0:
                new_row = {
                    'unique_id': current_data['unique_id'].iloc[0],
                    'ds': step_date,
                    'y': step_forecast[0] if len(step_forecast) > 0 else np.nan
                }
                # Add covariates if present (use last known values)
                if hist_exog_list and len(hist_exog_list) > 0:
                    for col in hist_exog_list:
                        if col in current_data.columns and len(current_data) > 0:
                            new_row[col] = current_data[col].iloc[-1]
                
                current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
        
        if return_weekly_forecasts:
            # For monthly aggregation, get all weeks in the target month
            horizon_date = start_date_ts + pd.Timedelta(weeks=horizon)
            month_start = pd.Timestamp(year=horizon_date.year, month=horizon_date.month, day=1)
            month_end = month_start + pd.offsets.MonthEnd(0)
            
            month_weekly_dates = []
            month_forecast_values = []
            
            for pred_date, pred_val in zip(weekly_dates, weekly_predictions):
                if month_start <= pred_date <= month_end:
                    month_weekly_dates.append(pred_date)
                    month_forecast_values.append(pred_val)
            
            if month_forecast_values:
                forecasts[horizon] = {
                    'weekly_forecasts': np.array(month_forecast_values),
                    'dates': pd.DatetimeIndex(month_weekly_dates),
                    '_available_targets': available_targets,
                    '_primary_target': primary_target
                }
            else:
                # Fallback: use the horizon step prediction
                forecasts[horizon] = weekly_predictions[-1] if weekly_predictions else np.full(len(available_targets), np.nan)
        else:
            # Return the prediction at the horizon step
            forecasts[horizon] = weekly_predictions[-1] if weekly_predictions else np.full(len(available_targets), np.nan)
    
    # Return series list for evaluation
    if is_multivariate and primary_target:
        series_for_outputs = [primary_target]
    else:
        series_for_outputs = [available_targets[0]] if available_targets else []
    
    return forecasts, series_for_outputs


def _run_multi_horizon_forecast_neuralforecast(
    checkpoint_path: Path,
    horizons: List[int],
    start_date: str,
    test_data: Optional[pd.DataFrame] = None,
    model_type: str = "patchtst",
    target_series: Optional[List[str]] = None,
    data_loader: Optional[Any] = None,
    return_weekly_forecasts: bool = False
) -> Tuple[Dict[int, Any], List[str]]:
    """Run multi-horizon forecasting from fixed start point for NeuralForecast models."""
    from src.utils import (
        get_target_series_from_model,
        load_model_checkpoint,
        interpolate_missing_values,
        convert_to_neuralforecast_format,
        extract_neuralforecast_forecasts,
    )
    
    # Load model
    logger.info(f"Loading {model_type.upper()} model for multi-horizon forecasting from {start_date}")
    model = load_model_checkpoint(checkpoint_path)
    
    # Extract model information
    model_targets, hist_exog_list, is_multivariate = _extract_model_info(model)
    
    # Check if model supports multi-horizon (has h > 1) or needs recursive forecasting
    if hasattr(model, 'models') and len(model.models) > 0:
        nf_model = model.models[0]
        model_h = getattr(nf_model, 'h', 1)
        if model_h == 1:
            # Model only supports 1-step ahead - use recursive forecasting for long-term
            logger.info(f"Model has h=1, using recursive forecasting for long-term horizons")
            return _run_multi_horizon_forecast_recursive(
                checkpoint_path, horizons, start_date, test_data,
                model_type=model_type, target_series=target_series,
                data_loader=data_loader, return_weekly_forecasts=return_weekly_forecasts
            )
    
    start_date_ts = pd.Timestamp(start_date)
    forecasts = {}
    
    # Get actual target series
    if test_data is not None:
        available_targets, primary_target = _resolve_forecast_targets(
            model, test_data, target_series, hist_exog_list, is_multivariate, model_targets
        )
    else:
        available_targets = None
        primary_target = None
    
    # Rebuild dataset for each horizon and predict
    for horizon in horizons:
        if test_data is not None and available_targets:
            # Prepare data based on model type
            if is_multivariate:
                # Multivariate: all variables
                train_data = test_data[test_data.index < start_date_ts][available_targets].copy()
                train_data = interpolate_missing_values(train_data, data_loader)
                nf_df = convert_to_neuralforecast_format(train_data, available_targets)
            else:
                # TFT: primary target + covariates
                if hist_exog_list and len(hist_exog_list) > 0:
                    all_cols = available_targets + [col for col in hist_exog_list if col in test_data.columns]
                    train_data_full = test_data[test_data.index < start_date_ts][all_cols].copy()
                    train_data_full = interpolate_missing_values(train_data_full, data_loader)
                    target_data = train_data_full[available_targets].copy()
                    covariate_cols = [col for col in hist_exog_list if col in train_data_full.columns]
                    covariate_data = train_data_full[covariate_cols].copy() if covariate_cols else None
                    nf_df = convert_to_neuralforecast_format(
                        target_data, available_targets,
                        covariate_data=covariate_data,
                        covariate_names=covariate_cols
                    )
                else:
                    train_data = test_data[test_data.index < start_date_ts][available_targets].copy()
                    train_data = interpolate_missing_values(train_data, data_loader)
                    nf_df = convert_to_neuralforecast_format(train_data, available_targets)
            
            forecast_df = model.predict(df=nf_df)
            
            if return_weekly_forecasts:
                # Get all weeks in the month containing horizon_date
                horizon_date = start_date_ts + pd.Timedelta(weeks=horizon)
                month_start = pd.Timestamp(year=horizon_date.year, month=horizon_date.month, day=1)
                month_end = month_start + pd.offsets.MonthEnd(0)
                
                month_weekly_dates = []
                month_forecast_values = []
                
                # Find the first week in the month
                first_week_in_month = month_start
                while first_week_in_month < start_date_ts:
                    first_week_in_month += pd.Timedelta(weeks=1)
                
                # Get all weeks from first_week_in_month to month_end
                current_date = first_week_in_month
                while current_date <= month_end:
                    weeks_from_start = (current_date - start_date_ts).days // 7
                    if 0 <= weeks_from_start < max(horizons):
                        # NeuralForecast models handle scaling internally, no inverse transform needed
                        # Extract all targets for aggregation (needs all series for tent kernel)
                        forecast_val = extract_neuralforecast_forecasts(
                            forecast_df, available_targets, horizon_idx=weeks_from_start
                        )
                        month_forecast_values.append(forecast_val)
                        month_weekly_dates.append(current_date)
                    current_date += pd.Timedelta(weeks=1)
                    if weeks_from_start >= max(horizons):
                        break
                
                if month_forecast_values:
                    forecasts[horizon] = {
                        'weekly_forecasts': np.array(month_forecast_values),
                        'dates': pd.DatetimeIndex(month_weekly_dates)
                    }
                else:
                    # Fallback: single forecast (primary target only)
                    # Use primary_target from config (set above) instead of available_targets[0]
                    primary_target_list = [primary_target] if primary_target else ([available_targets[0]] if available_targets else [])
                    forecast_values = extract_neuralforecast_forecasts(forecast_df, primary_target_list, horizon_idx=horizon - 1)
                    forecasts[horizon] = forecast_values
            else:
                # Extract forecast (primary target only)
                # Use primary_target from config (set above) instead of available_targets[0]
                primary_target_list = [primary_target] if primary_target else ([available_targets[0]] if available_targets else [])
                forecast_values = extract_neuralforecast_forecasts(forecast_df, primary_target_list, horizon_idx=horizon - 1)
                forecasts[horizon] = forecast_values
        else:
            logger.warning(f"No test_data available for horizon {horizon}w")
            forecasts[horizon] = np.full(len(available_targets) if available_targets else 1, np.nan)
    
    # Return forecasts and the series list that matches forecast output columns.
    #
    # IMPORTANT:
    # - For multivariate models (PatchTST / iTransformer / TimeMixer), `weekly_forecasts` contains
    #   predictions for ALL series in `available_targets` (shape: [n_weeks_in_month, n_series]).
    #   Downstream aggregation (`aggregate_weekly_to_monthly_tent_kernel`) needs all series for
    #   proper aggregation, so we need to return available_targets when weekly_forecasts are used.
    # - For evaluation, we use primary_target (from config) instead of available_targets[0]
    # - The return value should be: (forecasts_dict, [primary_target]) for evaluation
    #   But weekly_forecasts dict internally contains all available_targets for aggregation
    if is_multivariate:
        # For multivariate, return primary_target for evaluation (from config)
        # Note: When return_weekly_forecasts=True, the weekly_forecasts dict contains all
        # available_targets, but we return [primary_target] for evaluation.
        # The aggregation function will need to be called with available_targets, not primary_target.
        if primary_target:
            series_for_outputs = [primary_target]
        else:
            series_for_outputs = [available_targets[0]] if available_targets else []
    else:
        if available_targets:
            series_for_outputs = [available_targets[0]]
        elif target_series:
            series_for_outputs = [target_series[0]] if isinstance(target_series, list) and len(target_series) > 0 else []
        else:
            series_for_outputs = []
    
    # Store available_targets in forecasts dict for aggregation (if weekly_forecasts are used)
    # This allows main.py to access all series for aggregation while using primary_target for evaluation
    if is_multivariate and any(isinstance(v, dict) and 'weekly_forecasts' in v for v in forecasts.values()):
        # Add metadata to all forecast dicts so aggregation has consistent target mapping.
        for horizon, forecast_data in forecasts.items():
            if isinstance(forecast_data, dict) and 'weekly_forecasts' in forecast_data:
                forecast_data['_available_targets'] = available_targets
                forecast_data['_primary_target'] = primary_target
    
    return forecasts, series_for_outputs
