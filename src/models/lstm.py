"""LSTM model training and forecasting.

Uses sktime's LSTM forecaster.

예측 결과 파일 (predictions/lstm/ 디렉토리):
================================================================================
1. lstm_weekly_forecasts.csv
   - 세 타겟(KOIPALL.G, KOEQUIPTE, KOWRCCNSE)의 주별 예측값
   - 형식: CSV (date, KOIPALL.G, KOEQUIPTE, KOWRCCNSE)
   - 예측 기간: 2024-01-01 ~ 2025-09-01 (88주)
   - 총 89행 × 4열 (date 포함)

2. lstm_training_info.json
   - 하이퍼파라미터 정보 및 학습 시간 메타데이터
   - 포함 정보:
     * 모델 타입: lstm
     * 예측 기간: 2024-01-01 ~ 2025-09-01 (88주)
     * 각 타겟별:
       - units: LSTM units per layer
       - epochs: 학습 에포크 수
       - batch_size: 배치 크기
       - learning_rate: 학습률
       - input_size: 입력 윈도우 크기 (주)
       - scaler_type: 스케일러 타입 (robust)
       - training_data_length: 학습 데이터 길이
       - num_covariates: 공변량 개수
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
# Data preprocessing handled in train_sktime.py
from .base import save_model_checkpoint, load_model_checkpoint

logger = logging.getLogger(__name__)


def train_lstm(
    y_train: pd.Series,
    target_series: str,
    config: Dict[str, Any],
    checkpoint_dir: Path,
    X_train: Optional[pd.DataFrame] = None,
    y_train_original: Optional[pd.Series] = None
) -> Tuple[Any, Dict[str, Any]]:
    """Train LSTM model.
    
    Parameters
    ----------
    y_train : pd.Series
        Preprocessed training data (already transformed and resampled)
    target_series : str
        Target series name
    config : Dict[str, Any]
        Model configuration (units, epochs, etc.)
    checkpoint_dir : Path
        Directory to save checkpoint
    X_train : Optional[pd.DataFrame]
        Covariates (all other series except target). If None, no covariates used.
        
    Returns
    -------
    Tuple[Any, Dict[str, Any]]
        (trained_model, metadata) tuple
    """
    try:
        from sktime.forecasting.neuralforecast import NeuralForecastLSTM
    except ImportError:
        try:
            from sktime.forecasting.deep import LSTM
            use_neuralforecast = False
        except ImportError as e:
            raise ImportError(
                f"LSTM forecaster is not available in sktime. "
                f"Install with: pip install 'sktime[all-extras]' or pip install neuralforecast"
            ) from e
    else:
        use_neuralforecast = True
    
    # Data preprocessing is handled in train_sktime.py for unified preprocessing
    
    # Get model parameters
    units = config.get('units', [50, 50])  # LSTM units per layer
    epochs = config.get('epochs', config.get('max_epochs', 100))
    batch_size = config.get('batch_size', 32)
    learning_rate = config.get('learning_rate', 0.001)
    horizon = config.get('horizon', 24)  # Default: 24 weeks per chunk for recursive prediction
    
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
    
    logger.info(f"Training LSTM on {target_series} ({len(y_train)} observations)")
    
    # Horizon is now in weeks in config (88 weeks = 2024 JAN to 2025 OCT)
    # horizon is already in weeks from config, no conversion needed
    
    # Validate parameters
    input_size = config.get('input_size', 96)
    # NeuralForecastLSTM supports direct long-horizon forecasting, so we validate with full horizon
    total_required = input_size + horizon
    if len(y_train) < total_required:
        raise ValueError(
            f"LSTM training failed: Insufficient data. "
            f"Need at least {total_required} observations (input_size={input_size} + horizon={horizon}), "
            f"but only have {len(y_train)} observations. "
            f"Reduce input_size or horizon, or use more training data."
        )
    
    # Ensure y_train has proper frequency for NeuralForecastLSTM
    if use_neuralforecast and isinstance(y_train.index, pd.DatetimeIndex):
        if y_train.index.freq is None:
            # Try to infer frequency
            inferred_freq = pd.infer_freq(y_train.index)
            if inferred_freq:
                # Create new index with inferred frequency
                y_train.index = pd.DatetimeIndex(y_train.index, freq=inferred_freq)
                logger.info(f"Inferred frequency: {inferred_freq}")
            else:
                # Create new index with weekly frequency as default
                y_train.index = pd.date_range(start=y_train.index[0], periods=len(y_train), freq='W')
                logger.info("Set default frequency: W (weekly)")
    
    # Create base forecaster
    if use_neuralforecast:
        # NeuralForecastLSTM doesn't have 'h' parameter - horizon is set during prediction via fh
        # context_size defaults to input_size if not specified
        # Get frequency from y_train index if available
        freq = None
        if isinstance(y_train.index, pd.DatetimeIndex) and y_train.index.freq is not None:
            freq = y_train.index.freq
            logger.info(f"Using frequency: {freq} for NeuralForecastLSTM")
        
        base_forecaster = NeuralForecastLSTM(
            input_size=config.get('input_size', 96),  # Default: 96 weeks = 2 years
            encoder_hidden_size=units[0] if isinstance(units, list) else units,
            encoder_n_layers=len(units) if isinstance(units, list) else 1,
            learning_rate=learning_rate,
            max_steps=epochs,
            batch_size=batch_size,
            freq=freq  # Pass frequency explicitly
        )
    else:
        base_forecaster = LSTM(
            units=units,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
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
    
    # Train
    # NeuralForecastLSTM requires the same fh in fit() and predict()
    # So we must use the full horizon during training
    X_train_aligned = None
    if use_neuralforecast:
        from sktime.forecasting.base import ForecastingHorizon
        # Use full horizon for training (NeuralForecastLSTM requires same fh in fit and predict)
        training_fh = ForecastingHorizon(range(1, horizon + 1), is_relative=True)
        logger.info(f"Using full horizon ({horizon} weeks) for training (required by NeuralForecastLSTM)")
        if X_train is not None:
            # Remove duplicates from both y_train and X_train indices
            y_train_clean = y_train[~y_train.index.duplicated(keep='first')]
            X_train_clean = X_train[~X_train.index.duplicated(keep='first')]
            # Align covariates with target series index
            X_train_aligned = X_train_clean.reindex(y_train_clean.index, method='ffill').bfill()
            logger.info(f"Training LSTM with {len(X_train_aligned.columns)} covariates")
            forecaster.fit(y_train_clean, fh=training_fh, X=X_train_aligned)
        else:
            # Remove duplicates from y_train index first
            y_train_clean = y_train[~y_train.index.duplicated(keep='first')]
            forecaster.fit(y_train_clean, fh=training_fh)
    else:
        if X_train is not None:
            # Remove duplicates from both y_train and X_train indices
            y_train_clean = y_train[~y_train.index.duplicated(keep='first')]
            X_train_clean = X_train[~X_train.index.duplicated(keep='first')]
            # Align covariates with target series index
            X_train_aligned = X_train_clean.reindex(y_train_clean.index, method='ffill').bfill()
            logger.info(f"Training LSTM with {len(X_train_aligned.columns)} covariates")
            forecaster.fit(y_train_clean, X=X_train_aligned)
        else:
            # Remove duplicates from y_train index first
            y_train_clean = y_train[~y_train.index.duplicated(keep='first')]
            forecaster.fit(y_train_clean)
    
    logger.info("LSTM training completed")
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / "model.pkl"
    metadata = {
        'model_type': 'lstm',
        'target_series': target_series,
        'units': units,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'scaler': scaler,
        'scaler_type': scaler_type if scaler else None,
        'training_data_shape': y_train.shape,
        'training_data_index': list(y_train.index),
        'covariates_used': list(X_train_aligned.columns) if X_train_aligned is not None else None,
        'num_covariates': len(X_train_aligned.columns) if X_train_aligned is not None else 0
    }
    save_model_checkpoint(forecaster, checkpoint_path, metadata)
    
    return forecaster, metadata


def forecast_lstm(
    model: Any,
    horizon: int,
    last_date: Optional[pd.Timestamp] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Generate LSTM forecasts in weekly frequency using direct long-horizon forecasting.
    
    Parameters
    ----------
    model : Any
        Trained LSTM forecaster (trained on weekly data)
    horizon : int
        Forecast horizon in weeks (88 weeks = 2024 JAN to 2025 OCT)
    last_date : Optional[pd.Timestamp]
        Last date in training data
    metadata : Optional[Dict[str, Any]]
        Model metadata (for scaler inverse transform)
        
    Returns
    -------
    pd.DataFrame
        Forecasted values with DatetimeIndex (weekly frequency)
    """
    from sktime.forecasting.base import ForecastingHorizon
    
    logger = logging.getLogger(__name__)
    
    # Horizon is already in weeks
    horizon_weeks = horizon
    
    # CRITICAL: Update model with latest data before forecasting
    # Load recent data (2020-01-01 to 2023-12-31) and use lookback window
    input_size = metadata.get('input_size', 96) if metadata else 96  # Default: 96 weeks
    target_series = metadata.get('target_series') if metadata else None
    
    try:
        from src.utils import get_project_root, resolve_data_path, RECENT_START, RECENT_END
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
                
                if len(recent_data) >= input_size:
                    # Use last input_size periods for lookback window
                    y_recent = prepare_univariate_data(recent_data, target_series)
                    y_recent = set_dataframe_frequency(y_recent.to_frame()).iloc[:, 0]
                    
                    # Get last input_size periods
                    y_lookback = y_recent.iloc[-input_size:].copy()
                    
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
                    
                    logger.info(f"Updating LSTM model with {len(y_lookback)} weeks of recent data: {y_lookback.index[0]} to {y_lookback.index[-1]}")
                    
                    # Update model with recent data using update() or fit()
                    # NOTE: NeuralForecastLSTM does not have a custom update() implementation.
                    # When update() is called, it internally calls refit() (same as fit()).
                    # This is intentional - we want to refit the model with latest data before forecasting.
                    update_success = False
                    if hasattr(model, 'update'):
                        try:
                            # update() will internally refit the model with new data
                            # This ensures the model uses the latest lookback window for prediction
                            model.update(y_lookback, update_params=True)
                            logger.info("Model updated successfully using update() (internally refit with latest data)")
                            update_success = True
                        except Exception as e:
                            logger.warning(f"update() failed: {e}, trying fit()")
                            try:
                                # Fallback: directly refit with latest data
                                model.fit(y_lookback)
                                logger.info("Model updated successfully using fit()")
                                update_success = True
                            except Exception as e2:
                                logger.error(f"fit() also failed: {e2}, using original model state")
                    else:
                        # Fallback: use fit() to refit model with latest data
                        try:
                            model.fit(y_lookback)
                            logger.info("Model updated successfully using fit()")
                            update_success = True
                        except Exception as e:
                            logger.error(f"fit() failed: {e}, using original model state")
                    
                    if not update_success:
                        logger.warning("Model update failed, will use original trained model state for prediction")
                else:
                    logger.warning(f"Not enough recent data ({len(recent_data)} < {input_size}), using original model state")
            else:
                logger.warning(f"Target series '{target_series}' not found in data, using original model state")
        else:
            logger.warning("Could not load recent data, using original model state")
    except Exception as e:
        logger.warning(f"Failed to update model with recent data: {e}, using original model state")
    
    # RECURSIVE PREDICTION: Predict in 24-week chunks with update after each chunk
    chunk_size = 24  # weeks
    forecast_start_date = pd.Timestamp('2024-01-01')
    days_since_sunday = forecast_start_date.weekday()  # Monday=0, Sunday=6
    if days_since_sunday != 6:  # Not Sunday
        forecast_start_sunday = forecast_start_date + pd.Timedelta(days=6 - days_since_sunday)
    else:
        forecast_start_sunday = forecast_start_date
    
    # Get current cutoff
    current_cutoff = None
    try:
        if hasattr(model, 'forecaster_') and hasattr(model.forecaster_, '_cutoff'):
            current_cutoff = model.forecaster_.cutoff
        elif hasattr(model, 'cutoff'):
            current_cutoff = model.cutoff
        if isinstance(current_cutoff, pd.DatetimeIndex) and len(current_cutoff) > 0:
            current_cutoff = current_cutoff[0]
    except:
        pass
    
    if current_cutoff is None:
        current_cutoff = forecast_start_sunday - pd.Timedelta(weeks=1)
        logger.warning(f"Could not get cutoff, using {current_cutoff} as starting point")
    
    logger.info(f"Starting recursive prediction: {horizon_weeks} weeks in {chunk_size}-week chunks")
    logger.info(f"Forecast start: {forecast_start_sunday}, Current cutoff: {current_cutoff}")
    
    # Load actual data for updates (if available)
    actual_data = None
    try:
        from src.utils import get_project_root, resolve_data_path, TEST_START, TEST_END
        data_path = resolve_data_path()
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
    remaining_horizon = horizon_weeks
    
    while remaining_horizon > 0:
        # Determine chunk size (last chunk may be smaller)
        current_chunk_size = min(chunk_size, remaining_horizon)
        logger.info(f"Predicting chunk: {current_chunk_size} weeks starting from {current_date}")
        
        # Predict current chunk
        fh = ForecastingHorizon(range(1, current_chunk_size + 1), is_relative=True)
        chunk_forecast = model.predict(fh=fh)
        
        # Create index for this chunk
        chunk_index = pd.date_range(
            start=current_date,
            periods=current_chunk_size,
            freq='W-SUN'
        )
        
        if isinstance(chunk_forecast, pd.Series):
            chunk_forecast.index = chunk_index
        elif isinstance(chunk_forecast, pd.DataFrame):
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
                # Use last input_size periods for update
                update_data = available_actual.iloc[-input_size:].copy() if len(available_actual) >= input_size else available_actual
                logger.info(f"Using actual data for update: {len(update_data)} periods from {update_data.index[0]} to {update_data.index[-1]}")
        
        # If no actual data, use forecast values for update
        if update_data is None or len(update_data) < input_size:
            # Use forecast chunk + previous data for update
            if isinstance(chunk_forecast, pd.Series):
                forecast_values = chunk_forecast
            else:
                forecast_values = chunk_forecast.iloc[:, 0]
            
            # Combine with previous data if available (from model's cutoff)
            # For simplicity, use forecast chunk as update data
            update_data = forecast_values.iloc[-min(input_size, len(forecast_values)):].copy()
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
                model.update(update_data, update_params=True)
                logger.info(f"✓ Model updated after chunk prediction")
            else:
                model.fit(update_data)
                logger.info(f"✓ Model refitted after chunk prediction")
        except Exception as e:
            logger.warning(f"Failed to update model after chunk: {e}")
        
        # Move to next chunk
        current_date = chunk_index[-1] + pd.Timedelta(weeks=1)
        remaining_horizon -= current_chunk_size
    
    # Combine all chunks
    if isinstance(forecast_chunks[0], pd.Series):
        forecast = pd.concat(forecast_chunks)
    else:
        forecast = pd.concat(forecast_chunks)
    
    # Ensure proper index
    forecast_index = pd.date_range(
        start=forecast_start_sunday,
        periods=horizon_weeks,
        freq='W-SUN'
    )
    forecast.index = forecast_index[:len(forecast)]
    
    logger.info(f"✓ Completed recursive prediction: {len(forecast)} weeks from {forecast.index[0]} to {forecast.index[-1]}")
    
    # Apply inverse transform if scaler is available in metadata
    if metadata and 'scaler' in metadata and metadata['scaler'] is not None:
        scaler = metadata['scaler']
        try:
            # Inverse transform the forecast
            if isinstance(forecast, pd.Series):
                forecast_values = forecast.values.reshape(-1, 1)
                forecast_inverse = scaler.inverse_transform(forecast_values)
                forecast = pd.Series(
                    forecast_inverse.flatten(),
                    index=forecast.index,
                    name=forecast.name
                )
                logger.info("Applied inverse transform to return to original scale")
            elif isinstance(forecast, pd.DataFrame):
                forecast_values = forecast.values
                forecast_inverse = scaler.inverse_transform(forecast_values)
                forecast = pd.DataFrame(
                    forecast_inverse,
                    index=forecast.index,
                    columns=forecast.columns
                )
                logger.info("Applied inverse transform to return to original scale")
        except Exception as e:
            logger.warning(f"Failed to apply inverse transform: {e}")
    
    return forecast if isinstance(forecast, pd.DataFrame) else forecast.to_frame()
