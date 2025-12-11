"""
Chronos forecasting model implementation.

Chronos is a pre-trained time series forecasting model from Amazon.
This module provides training and forecasting functions for Chronos models
using sktime's ChronosForecaster wrapper.

예측 결과 파일 (predictions/chronos/ 디렉토리):
================================================================================
1. chronos_weekly_forecasts.csv
   - 세 타겟(KOIPALL.G, KOEQUIPTE, KOWRCCNSE)의 주별 예측값 통합 파일
   - 형식: CSV (date, KOIPALL.G, KOEQUIPTE, KOWRCCNSE)
   - 예측 기간: 2024-01-01 ~ 2025-09-01 (88주)
   - 총 89행 × 4열 (date 포함)

2. {target}_chronos_weekly.csv
   - 각 타겟별 주별 예측값 개별 파일
   - 형식: CSV (date, {target})
   - 예: KOIPALL.G_chronos_weekly.csv, KOEQUIPTE_chronos_weekly.csv, KOWRCCNSE_chronos_weekly.csv

3. {target}_chronos_monthly.csv
   - 각 타겟별 월별 예측값 (주별 예측값을 월별로 집계)
   - 형식: CSV (date, {target})
   - 예: KOIPALL.G_chronos_monthly.csv, KOEQUIPTE_chronos_monthly.csv, KOWRCCNSE_chronos_monthly.csv

4. chronos_training_info.json
   - 하이퍼파라미터 정보 및 학습 시간 메타데이터
   - 포함 정보:
     * 모델 타입: chronos
     * 예측 기간: 2024-01-01 ~ 2025-09-01 (88주)
     * 각 타겟별:
       - model_name: amazon/chronos-t5-tiny
       - context_length: None (모델 기본값)
       - scaler_type: robust (RobustScaler)
       - training_data_length: 학습 데이터 길이
       - training_time_seconds: 학습 시간 (초)
       - training_time_minutes: 학습 시간 (분)
       - checkpoint_path: 체크포인트 경로
       - log_file: 로그 파일 경로
     * 요약 통계:
       - total_targets: 3
       - average_training_time_seconds: 평균 학습 시간
       - total_training_time_seconds: 총 학습 시간
================================================================================
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from sktime.forecasting.compose import ForecastingPipeline
from sktime.transformations.series.impute import Imputer
from sktime.forecasting.naive import NaiveForecaster
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.utils import ValidationError, SCALER_ROBUST, SCALER_STANDARD
from .base import save_model_checkpoint, load_model_checkpoint

logger = logging.getLogger(__name__)


def train_chronos(
    y_train: pd.Series,
    target_series: str,
    config: Dict[str, Any],
    checkpoint_dir: Path,
    X_train: Optional[pd.DataFrame] = None,
    y_train_original: Optional[pd.Series] = None
) -> Tuple[Any, Dict[str, Any]]:
    """Train Chronos model.
    
    Parameters
    ----------
    y_train : pd.Series
        Preprocessed training data (already transformed, resampled, and imputed)
    target_series : str
        Target series name
    config : Dict[str, Any]
        Model configuration (model_name, scaler, etc.)
    checkpoint_dir : Path
        Directory to save checkpoint
    X_train : Optional[pd.DataFrame]
        Covariates (not used for Chronos, but kept for API consistency)
    y_train_original : Optional[pd.Series]
        Original data before transformations (for scaler fitting to original scale)
        
    Returns
    -------
    Tuple[Any, Dict[str, Any]]
        (trained_model, metadata) tuple
    """
    try:
        from sktime.forecasting.chronos import ChronosForecaster
    except ImportError as e:
        raise ImportError(
            f"ChronosForecaster is not available. "
            f"Install with: pip install 'sktime[all-extras]' or pip install chronos-forecasting"
        ) from e
    
    # Get model parameters
    model_name = config.get('model_name', 'amazon/chronos-t5-tiny')
    context_length = config.get('context_length', None)
    prediction_length = config.get('prediction_length', None)
    
    # Get scaler type
    scaler_type = config.get('scaler', SCALER_ROBUST)
    
    # Create scaler transformer for pipeline
    scaler_transformer = None
    scaler = None
    if scaler_type and scaler_type != 'null':
        scaler_type_lower = scaler_type.lower()
        if scaler_type_lower == SCALER_ROBUST:
            sklearn_scaler = RobustScaler()
        elif scaler_type_lower == SCALER_STANDARD:
            sklearn_scaler = StandardScaler()
        else:
            logger.warning(f"Unknown scaler type '{scaler_type}', using RobustScaler")
            sklearn_scaler = RobustScaler()
        
        # Fit scaler to original data if available (for proper inverse transform)
        if y_train_original is not None and len(y_train_original) > 0:
            sklearn_scaler.fit(y_train_original.values.reshape(-1, 1))
            logger.info(f"Fitted {scaler_type} scaler to original data ({len(y_train_original)} observations)")
        else:
            # Fit to transformed data as fallback
            sklearn_scaler.fit(y_train.values.reshape(-1, 1))
            logger.info(f"Fitted {scaler_type} scaler to transformed data ({len(y_train)} observations)")
        
        scaler_transformer = TabularToSeriesAdaptor(sklearn_scaler)
        scaler = sklearn_scaler
        logger.info(f"Will apply {scaler_type} scaling in ForecastingPipeline")
    else:
        logger.info("No scaling applied (scaler='null')")
    
    logger.info(f"Training Chronos on {target_series} ({len(y_train)} observations)")
    logger.info(f"Model: {model_name}")
    
    # Create config dict for ChronosForecaster
    # Note: context_length and prediction_length are not direct parameters
    # They may be handled internally by the model
    chronos_config = {}
    if context_length is not None:
        chronos_config['context_length'] = context_length
    if prediction_length is not None:
        chronos_config['prediction_length'] = prediction_length
    
    # Create base forecaster
    # ChronosForecaster uses model_path (not model_name) and optional config dict
    base_forecaster = ChronosForecaster(
        model_path=model_name,
        config=chronos_config if chronos_config else None
    )
    
    # Create pipeline with imputation and scaling
    pipeline_steps = [
        ('imputer_ffill', Imputer(method="ffill")),
        ('imputer_bfill', Imputer(method="bfill")),
        ('imputer_forecaster', Imputer(method="forecaster", forecaster=NaiveForecaster(strategy="last"))),
    ]
    
    # Add scaler to pipeline if specified
    if scaler_transformer is not None:
        pipeline_steps.append(('scaler', scaler_transformer))
    
    pipeline_steps.append(('forecaster', base_forecaster))
    
    forecaster = ForecastingPipeline(pipeline_steps)
    
    # Train (scaler will be fitted automatically during pipeline.fit())
    logger.info("Calling forecaster.fit()...")
    try:
        forecaster.fit(y_train)
        logger.info("forecaster.fit() completed successfully")
    except Exception as e:
        logger.error(f"forecaster.fit() failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    logger.info("Chronos training completed")
    
    # The scaler in the pipeline is already fitted after forecaster.fit()
    # Extract the fitted scaler for metadata
    fitted_scaler = None
    if scaler_transformer is not None:
        # Get the scaler from the pipeline
        try:
            fitted_scaler = forecaster.steps[-2][1].transformer  # Second to last step is scaler
        except (IndexError, AttributeError):
            # Fallback: use the scaler we created
            fitted_scaler = scaler
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / "model.pkl"
    metadata = {
        'model_type': 'chronos',
        'target_series': target_series,
        'model_name': model_name,
        'scaler_type': scaler_type,
        'training_data_index': y_train.index.tolist(),
        'training_data_length': len(y_train),
        'scaler': fitted_scaler,  # Save scaler for inverse transform
    }
    
    save_model_checkpoint(forecaster, checkpoint_path, metadata)
    logger.info(f"Saved Chronos checkpoint to {checkpoint_path}")
    
    return forecaster, metadata


def forecast_chronos(
    model: Any,
    horizon: int,
    last_date: Optional[pd.Timestamp] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> pd.Series:
    """Generate forecasts from a trained Chronos model.
    
    Parameters
    ----------
    model : Any
        Trained Chronos forecaster (ForecastingPipeline)
    horizon : int
        Forecast horizon in weeks
    last_date : Optional[pd.Timestamp]
        Last date from training data (for creating forecast index)
    metadata : Optional[Dict[str, Any]]
        Model metadata (for scaler inverse transform)
        
    Returns
    -------
    pd.Series
        Forecasted values with DatetimeIndex (weekly frequency)
    """
    from sktime.forecasting.base import ForecastingHorizon
    
    logger = logging.getLogger(__name__)
    
    # CRITICAL: Fix pipeline cutoff frequency BEFORE creating ForecastingHorizon
    # Pipeline cutoff might not have freq set, which causes ForecastingHorizon to fail
    # Fix cutoff first, then create ForecastingHorizon
    try:
        # Update model's cutoff with proper freq (weekly)
        if hasattr(model, 'forecaster_') and hasattr(model.forecaster_, '_cutoff'):
            cutoff = model.forecaster_.cutoff
            if isinstance(cutoff, pd.DatetimeIndex) and len(cutoff) > 0:
                cutoff_date = cutoff[0]
                # Adjust to nearest Sunday (W-SUN) for weekly frequency
                days_since_sunday = cutoff_date.weekday()  # Monday=0, Sunday=6
                if days_since_sunday == 6:  # Already Sunday
                    adjusted_cutoff = cutoff_date
                else:
                    # Move to next Sunday
                    adjusted_cutoff = cutoff_date + pd.Timedelta(days=6 - days_since_sunday)
                
                # Create new cutoff with weekly frequency
                cutoff_weekly = pd.DatetimeIndex([adjusted_cutoff], freq='W-SUN')
                model.forecaster_._cutoff = cutoff_weekly
                logger.debug(f"Fixed cutoff frequency: {cutoff_date} -> {adjusted_cutoff} (W-SUN)")
        elif hasattr(model, 'cutoff'):
            # Direct cutoff access
            cutoff = model.cutoff
            if isinstance(cutoff, pd.DatetimeIndex) and len(cutoff) > 0:
                cutoff_date = cutoff[0]
                days_since_sunday = cutoff_date.weekday()
                if days_since_sunday == 6:
                    adjusted_cutoff = cutoff_date
                else:
                    adjusted_cutoff = cutoff_date + pd.Timedelta(days=6 - days_since_sunday)
                cutoff_weekly = pd.DatetimeIndex([adjusted_cutoff], freq='W-SUN')
                model._cutoff = cutoff_weekly
                logger.debug(f"Fixed direct cutoff frequency: {cutoff_date} -> {adjusted_cutoff} (W-SUN)")
    except Exception as e:
        logger.warning(f"Could not fix cutoff frequency: {e}, continuing anyway")
    
    # Create forecasting horizon with explicit freq if possible
    # Get freq from model's cutoff if available
    freq = None
    try:
        if hasattr(model, 'forecaster_') and hasattr(model.forecaster_, '_cutoff'):
            cutoff = model.forecaster_.cutoff
            if hasattr(cutoff, 'freq') and cutoff.freq is not None:
                freq = cutoff.freq
        elif hasattr(model, 'cutoff'):
            cutoff = model.cutoff
            if hasattr(cutoff, 'freq') and cutoff.freq is not None:
                freq = cutoff.freq
    except:
        pass
    
    # If no freq from cutoff, use weekly as default
    if freq is None:
        freq = 'W-SUN'
    
    # CRITICAL: Update model with latest data before forecasting
    # Load recent data (2020-01-01 to 2023-12-31) and use lookback window
    context_length = metadata.get('context_length') if metadata else None
    # Chronos default context_length varies by model, use reasonable default
    lookback_window = context_length if context_length is not None else 96  # Default: 96 weeks
    target_series = metadata.get('target_series') if metadata else None
    
    # Track whether model was successfully updated
    model_updated = False
    updated_cutoff = None
    
    try:
        from src.utils import get_project_root, resolve_data_path, RECENT_START, RECENT_END, TEST_START
        from src.train.preprocess import prepare_univariate_data, set_dataframe_frequency, impute_missing_values
        
        project_root = get_project_root()
        data_path = resolve_data_path()
        
        if data_path.exists() and target_series:
            # Load full data
            full_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            
            if target_series in full_data.columns:
                # Get recent data (2020-01-01 to 2023-12-31)
                recent_data = full_data[
                    (full_data.index >= RECENT_START) & 
                    (full_data.index <= RECENT_END)
                ]
                
                if len(recent_data) >= lookback_window:
                    # Use last lookback_window periods for context
                    y_recent = prepare_univariate_data(recent_data, target_series)
                    y_recent = set_dataframe_frequency(y_recent.to_frame()).iloc[:, 0]
                    
                    # Get last lookback_window periods
                    y_lookback = y_recent.iloc[-lookback_window:].copy()
                    
                    # Ensure index is sorted and has proper frequency
                    y_lookback = y_lookback.sort_index()
                    
                    # CRITICAL: Force W-SUN frequency to match model's expected frequency
                    # The data might have W-THU or other weekday, but model expects W-SUN
                    try:
                        # Try to resample to W-SUN (this will align dates to Sundays)
                        y_lookback = y_lookback.asfreq('W-SUN', method='ffill')
                        logger.debug(f"Resampled to W-SUN frequency: {y_lookback.index[0]} to {y_lookback.index[-1]}")
                    except (TypeError, ValueError) as e:
                        # If asfreq fails, create new index with W-SUN
                        logger.debug(f"asfreq failed: {e}, creating new W-SUN index")
                        try:
                            # Create new DatetimeIndex with W-SUN frequency
                            start_date = y_lookback.index[0]
                            # Adjust to nearest Sunday
                            days_since_sunday = start_date.weekday()  # Monday=0, Sunday=6
                            if days_since_sunday != 6:
                                start_sunday = start_date - pd.Timedelta(days=days_since_sunday + 1)
                            else:
                                start_sunday = start_date
                            
                            new_index = pd.date_range(
                                start=start_sunday,
                                periods=len(y_lookback),
                                freq='W-SUN'
                            )
                            y_lookback.index = new_index
                            logger.debug(f"Created new W-SUN index: {new_index[0]} to {new_index[-1]}")
                        except Exception as e2:
                            logger.warning(f"Could not create W-SUN index: {e2}, trying to set freq attribute")
                            # Last resort: try to set freq attribute directly
                            if isinstance(y_lookback.index, pd.DatetimeIndex):
                                y_lookback.index.freq = 'W-SUN'
                    
                    # Impute missing values
                    if y_lookback.isnull().any():
                        y_lookback = impute_missing_values(y_lookback.to_frame()).iloc[:, 0]
                        # Re-sort after imputation
                        y_lookback = y_lookback.sort_index()
                        # Ensure W-SUN frequency is still set after imputation
                        if not isinstance(y_lookback.index, pd.DatetimeIndex) or y_lookback.index.freq != 'W-SUN':
                            try:
                                y_lookback = y_lookback.asfreq('W-SUN', method='ffill')
                            except:
                                if isinstance(y_lookback.index, pd.DatetimeIndex):
                                    y_lookback.index.freq = 'W-SUN'
                    
                    logger.info(f"Updating Chronos model with {len(y_lookback)} weeks of recent data: {y_lookback.index[0]} to {y_lookback.index[-1]}")
                    
                    # Get cutoff before update for comparison
                    old_cutoff = None
                    try:
                        if hasattr(model, 'forecaster_') and hasattr(model.forecaster_, '_cutoff'):
                            old_cutoff = model.forecaster_.cutoff
                        elif hasattr(model, 'cutoff'):
                            old_cutoff = model.cutoff
                    except:
                        pass
                    
                    if old_cutoff is not None:
                        logger.info(f"Model cutoff before update: {old_cutoff}")
                    
                    # Update model with recent data using update() or fit()
                    # NOTE: ChronosForecaster does not have a custom update() implementation.
                    # When update() is called, it internally calls refit() (same as fit()).
                    # This is intentional - we want to refit the model with latest data before forecasting.
                    if hasattr(model, 'update'):
                        try:
                            # update() will internally refit the model with new data
                            # This ensures the model uses the latest lookback window for prediction
                            model.update(y_lookback)
                            model_updated = True
                            logger.info("✓ Model updated successfully using update() (internally refit with latest data)")
                        except Exception as e:
                            logger.warning(f"update() failed: {e}, trying fit()")
                            # Fallback: directly refit with latest data
                            model.fit(y_lookback)
                            model_updated = True
                            logger.info("✓ Model updated successfully using fit()")
                    else:
                        # Fallback: use fit() to refit model with latest data
                        model.fit(y_lookback)
                        model_updated = True
                        logger.info("✓ Model updated successfully using fit()")
                    
                    # After update/fit, check and reset cutoff frequency to weekly
                    try:
                        if hasattr(model, 'forecaster_') and hasattr(model.forecaster_, '_cutoff'):
                            cutoff = model.forecaster_.cutoff
                            if isinstance(cutoff, pd.DatetimeIndex) and len(cutoff) > 0:
                                cutoff_date = cutoff[0]
                                days_since_sunday = cutoff_date.weekday()
                                if days_since_sunday == 6:
                                    adjusted_cutoff = cutoff_date
                                else:
                                    adjusted_cutoff = cutoff_date + pd.Timedelta(days=6 - days_since_sunday)
                                cutoff_weekly = pd.DatetimeIndex([adjusted_cutoff], freq='W-SUN')
                                model.forecaster_._cutoff = cutoff_weekly
                                updated_cutoff = cutoff_weekly[0]
                                logger.info(f"✓ Model cutoff after update: {updated_cutoff} (W-SUN)")
                                logger.debug(f"Reset cutoff frequency after update: {adjusted_cutoff} (W-SUN)")
                        elif hasattr(model, 'cutoff'):
                            cutoff = model.cutoff
                            if isinstance(cutoff, pd.DatetimeIndex) and len(cutoff) > 0:
                                cutoff_date = cutoff[0]
                                days_since_sunday = cutoff_date.weekday()
                                if days_since_sunday == 6:
                                    adjusted_cutoff = cutoff_date
                                else:
                                    adjusted_cutoff = cutoff_date + pd.Timedelta(days=6 - days_since_sunday)
                                cutoff_weekly = pd.DatetimeIndex([adjusted_cutoff], freq='W-SUN')
                                model._cutoff = cutoff_weekly
                                updated_cutoff = cutoff_weekly[0]
                                logger.info(f"✓ Model cutoff after update: {updated_cutoff} (W-SUN)")
                                logger.debug(f"Reset direct cutoff frequency after update: {adjusted_cutoff} (W-SUN)")
                    except Exception as e:
                        logger.warning(f"Could not reset cutoff after update: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())
                else:
                    logger.warning(f"Not enough recent data ({len(recent_data)} < {lookback_window}), using original model state")
            else:
                logger.warning(f"Target series '{target_series}' not found in data, using original model state")
        else:
            logger.warning("Could not load recent data, using original model state")
    except Exception as e:
        logger.warning(f"Failed to update model with recent data: {e}, using original model state")
        import traceback
        logger.debug(traceback.format_exc())
    
    # Log update status
    if not model_updated:
        logger.warning("⚠ Model was NOT updated with recent data - forecast may be based on old training data (2019-12-31)")
    else:
        logger.info(f"✓ Model successfully updated - cutoff is now at: {updated_cutoff}")
    
    # Get current cutoff to determine forecast start
    current_cutoff = None
    try:
        if hasattr(model, 'forecaster_') and hasattr(model.forecaster_, '_cutoff'):
            current_cutoff = model.forecaster_.cutoff
        elif hasattr(model, 'cutoff'):
            current_cutoff = model.cutoff
        
        if current_cutoff is not None:
            if isinstance(current_cutoff, pd.DatetimeIndex) and len(current_cutoff) > 0:
                current_cutoff = current_cutoff[0]
            logger.info(f"Current model cutoff: {current_cutoff}")
    except Exception as e:
        logger.warning(f"Could not get current cutoff: {e}")
    
    # Target forecast start date: 2024-01-01
    forecast_start_date = pd.Timestamp('2024-01-01')
    
    # Calculate gap between cutoff and forecast start
    if current_cutoff is not None and isinstance(current_cutoff, pd.Timestamp):
        # Calculate weeks between cutoff and forecast start
        weeks_to_forecast_start = (forecast_start_date - current_cutoff).days / 7
        logger.info(f"Gap between cutoff ({current_cutoff}) and forecast start ({forecast_start_date}): {weeks_to_forecast_start:.1f} weeks")
        
        # If cutoff is before forecast start, we need to predict from cutoff to forecast_start, then from forecast_start
        if weeks_to_forecast_start > 0:
            # Use absolute horizon: predict from cutoff to forecast_start + horizon
            # This ensures we predict the correct period
            total_weeks_needed = int(weeks_to_forecast_start) + horizon
            logger.info(f"Using absolute horizon: predicting {total_weeks_needed} weeks from cutoff ({current_cutoff})")
            
            # Create absolute forecasting horizon
            forecast_end_date = forecast_start_date + pd.Timedelta(weeks=horizon)
            forecast_end_sunday = forecast_end_date
            days_since_sunday = forecast_end_date.weekday()
            if days_since_sunday != 6:
                forecast_end_sunday = forecast_end_date + pd.Timedelta(days=6 - days_since_sunday)
            
            # Use absolute dates for forecasting horizon
            fh = ForecastingHorizon(
                pd.date_range(start=current_cutoff + pd.Timedelta(weeks=1), 
                            end=forecast_end_sunday, freq='W-SUN'),
                is_relative=False
            )
        else:
            # Cutoff is at or after forecast start, use relative horizon
            logger.info(f"Cutoff ({current_cutoff}) is at or after forecast start ({forecast_start_date}), using relative horizon")
            fh = ForecastingHorizon(range(1, horizon + 1), is_relative=True, freq=freq)
    else:
        # Fallback: use relative horizon
        logger.warning("Could not determine cutoff, using relative horizon (may cause date mismatch)")
        fh = ForecastingHorizon(range(1, horizon + 1), is_relative=True, freq=freq)
    
    # RECURSIVE PREDICTION: Predict in 24-week chunks with update after each chunk
    chunk_size = 24  # weeks
    forecast_start_sunday = forecast_start_date
    days_since_sunday = forecast_start_date.weekday()  # Monday=0, Sunday=6
    if days_since_sunday != 6:  # Not Sunday
        forecast_start_sunday = forecast_start_date + pd.Timedelta(days=6 - days_since_sunday)
    
    if current_cutoff is None:
        current_cutoff = forecast_start_sunday - pd.Timedelta(weeks=1)
        logger.warning(f"Could not get cutoff, using {current_cutoff} as starting point")
    
    logger.info(f"Starting recursive prediction: {horizon} weeks in {chunk_size}-week chunks")
    logger.info(f"Forecast start: {forecast_start_sunday}, Current cutoff: {current_cutoff}")
    
    # Load actual data for updates (if available)
    actual_data = None
    try:
        from src.utils import TEST_START, TEST_END
        if data_path.exists() and target_series:
            full_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            if target_series in full_data.columns:
                # Get actual data in forecast period (if available)
                actual_data = full_data[
                    (full_data.index >= TEST_START) & 
                    (full_data.index <= TEST_END)
                ][target_series]
                if len(actual_data) > 0:
                    logger.info(f"Found {len(actual_data)} actual data points for updates")
    except Exception as e:
        logger.debug(f"Could not load actual data for updates: {e}")
    
    # Collect forecast chunks
    forecast_chunks = []
    current_date = forecast_start_sunday
    remaining_horizon = horizon
    
    while remaining_horizon > 0:
        # Determine chunk size (last chunk may be smaller)
        current_chunk_size = min(chunk_size, remaining_horizon)
        logger.info(f"Predicting chunk: {current_chunk_size} weeks starting from {current_date}")
        
        # Get current cutoff for this chunk
        try:
            if hasattr(model, 'forecaster_') and hasattr(model.forecaster_, '_cutoff'):
                chunk_cutoff = model.forecaster_.cutoff
            elif hasattr(model, 'cutoff'):
                chunk_cutoff = model.cutoff
            else:
                chunk_cutoff = pd.DatetimeIndex([current_date - pd.Timedelta(weeks=1)], freq='W-SUN')
            
            if isinstance(chunk_cutoff, pd.DatetimeIndex) and len(chunk_cutoff) > 0:
                chunk_cutoff_date = chunk_cutoff[0]
            else:
                chunk_cutoff_date = current_date - pd.Timedelta(weeks=1)
        except:
            chunk_cutoff_date = current_date - pd.Timedelta(weeks=1)
        
        # Calculate gap for this chunk
        weeks_to_chunk_start = (current_date - chunk_cutoff_date).days / 7
        if weeks_to_chunk_start > 0:
            total_weeks = int(weeks_to_chunk_start) + current_chunk_size
            forecast_end_date = current_date + pd.Timedelta(weeks=current_chunk_size)
            forecast_end_sunday = forecast_end_date
            days_since_sunday = forecast_end_date.weekday()
            if days_since_sunday != 6:
                forecast_end_sunday = forecast_end_date + pd.Timedelta(days=6 - days_since_sunday)
            
            fh = ForecastingHorizon(
                pd.date_range(start=chunk_cutoff_date + pd.Timedelta(weeks=1), 
                            end=forecast_end_sunday, freq='W-SUN'),
                is_relative=False
            )
        else:
            fh = ForecastingHorizon(range(1, current_chunk_size + 1), is_relative=True, freq=freq)
        
        # Predict current chunk
        chunk_forecast = model.predict(fh=fh)
        
        # Extract only the portion for this chunk
        chunk_index = pd.date_range(
            start=current_date,
            periods=current_chunk_size,
            freq='W-SUN'
        )
        
        if len(chunk_forecast) > current_chunk_size:
            chunk_forecast = chunk_forecast.iloc[:current_chunk_size]
        elif len(chunk_forecast) < current_chunk_size:
            # Pad with last value
            last_value = chunk_forecast.iloc[-1] if len(chunk_forecast) > 0 else 0
            padding = pd.Series([last_value] * (current_chunk_size - len(chunk_forecast)), 
                              index=chunk_index[len(chunk_forecast):])
            chunk_forecast = pd.concat([chunk_forecast, padding])
        
        chunk_forecast.index = chunk_index
        forecast_chunks.append(chunk_forecast)
        logger.info(f"✓ Predicted {current_chunk_size} weeks: {chunk_index[0]} to {chunk_index[-1]}")
        
        # Update model with actual data or forecast for next chunk
        update_data = None
        chunk_end_date = chunk_index[-1] + pd.Timedelta(weeks=1)  # Next week after chunk
        
        # Try to get actual data for update
        if actual_data is not None and len(actual_data) > 0:
            # Get actual data up to chunk_end_date
            available_actual = actual_data[actual_data.index <= chunk_end_date]
            if len(available_actual) > 0:
                # Use last lookback_window periods for update
                update_data = available_actual.iloc[-lookback_window:].copy() if len(available_actual) >= lookback_window else available_actual
                logger.info(f"Using actual data for update: {len(update_data)} periods from {update_data.index[0]} to {update_data.index[-1]}")
        
        # If no actual data, use forecast values for update
        if update_data is None or len(update_data) < lookback_window:
            # Use forecast chunk for update
            update_data = chunk_forecast.iloc[-min(lookback_window, len(chunk_forecast)):].copy()
            logger.info(f"Using forecast values for update: {len(update_data)} periods")
        
        # Ensure W-SUN frequency
        try:
            update_data = update_data.asfreq('W-SUN', method='ffill')
        except:
            if isinstance(update_data.index, pd.DatetimeIndex):
                update_data.index.freq = 'W-SUN'
        
        # Update model
        try:
            if hasattr(model, 'update'):
                model.update(update_data)
                logger.info(f"✓ Model updated after chunk prediction")
            else:
                model.fit(update_data)
                logger.info(f"✓ Model refitted after chunk prediction")
            
            # Reset cutoff frequency after update
            try:
                if hasattr(model, 'forecaster_') and hasattr(model.forecaster_, '_cutoff'):
                    cutoff = model.forecaster_.cutoff
                    if isinstance(cutoff, pd.DatetimeIndex) and len(cutoff) > 0:
                        cutoff_date = cutoff[0]
                        days_since_sunday = cutoff_date.weekday()
                        if days_since_sunday == 6:
                            adjusted_cutoff = cutoff_date
                        else:
                            adjusted_cutoff = cutoff_date + pd.Timedelta(days=6 - days_since_sunday)
                        cutoff_weekly = pd.DatetimeIndex([adjusted_cutoff], freq='W-SUN')
                        model.forecaster_._cutoff = cutoff_weekly
            except:
                pass
        except Exception as e:
            logger.warning(f"Failed to update model after chunk: {e}")
        
        # Move to next chunk
        current_date = chunk_index[-1] + pd.Timedelta(weeks=1)
        remaining_horizon -= current_chunk_size
    
    # Combine all chunks
    forecast = pd.concat(forecast_chunks)
    
    # Ensure proper index
    forecast_index = pd.date_range(
        start=forecast_start_sunday,
        periods=horizon,
        freq='W-SUN'
    )
    forecast.index = forecast_index[:len(forecast)]
    
    logger.info(f"✓ Completed recursive prediction: {len(forecast)} weeks from {forecast.index[0]} to {forecast.index[-1]}")
    
    # Inverse transform if scaler is available in metadata
    if metadata and 'scaler' in metadata and metadata['scaler'] is not None:
        scaler = metadata['scaler']
        try:
            # Inverse transform the forecast
            forecast_values = forecast.values.reshape(-1, 1)
            forecast_inverse = scaler.inverse_transform(forecast_values)
            forecast = pd.Series(
                forecast_inverse.flatten(),
                index=forecast.index,
                name=forecast.name
            )
            logger.info("Applied inverse transform to return to original scale")
        except Exception as e:
            logger.warning(f"Failed to apply inverse transform: {e}")
    
    return forecast
