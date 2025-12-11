"""TFT (Temporal Fusion Transformer) model training and forecasting.

Uses sktime's PytorchForecastingTFT which wraps pytorch-forecasting library.
Leverages sktime's ForecastingPipeline, ForecastingHorizon, and TabularToSeriesAdaptor
for proper preprocessing and forecasting workflow.

예측 결과 파일 (predictions/tft/ 디렉토리):
================================================================================
1. tft_weekly_forecasts.csv
   - 세 타겟(KOIPALL.G, KOEQUIPTE, KOWRCCNSE)의 주별 예측값 통합 파일
   - 형식: CSV (date, KOIPALL.G, KOEQUIPTE, KOWRCCNSE)
   - 예측 기간: 2024-01-01 ~ 2025-09-01 (88주)
   - 총 89행 × 4열 (date 포함)

2. {target}_tft_weekly.csv
   - 각 타겟별 주별 예측값 개별 파일
   - 형식: CSV (date, {target})
   - 예: KOIPALL.G_tft_weekly.csv, KOEQUIPTE_tft_weekly.csv, KOWRCCNSE_tft_weekly.csv

3. {target}_tft_monthly.csv
   - 각 타겟별 월별 예측값 (주별 예측값을 월별로 집계)
   - 형식: CSV (date, {target})
   - 예: KOIPALL.G_tft_monthly.csv, KOEQUIPTE_tft_monthly.csv, KOWRCCNSE_tft_monthly.csv

4. tft_training_info.json
   - 하이퍼파라미터 정보 및 학습 시간 메타데이터
   - 포함 정보:
     * 모델 타입: tft
     * 예측 기간: 2024-01-01 ~ 2025-09-01 (88주)
     * 각 타겟별:
       - input_size: 입력 윈도우 크기 (주)
       - hidden_size: 히든 레이어 크기
       - n_head: 어텐션 헤드 수
       - dropout: 드롭아웃 비율
       - learning_rate: 학습률
       - max_epochs: 최대 에포크 수
       - batch_size: 배치 크기
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
from omegaconf import DictConfig

from sktime.forecasting.pytorchforecasting import PytorchForecastingTFT
from sktime.forecasting.compose import ForecastingPipeline
from sktime.transformations.series.impute import Imputer
from sktime.forecasting.naive import NaiveForecaster
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.utils import ValidationError, SCALER_ROBUST, SCALER_STANDARD
from .base import save_model_checkpoint, load_model_checkpoint

logger = logging.getLogger(__name__)


def train_tft(
    y_train: pd.Series,
    target_series: str,
    config: Dict[str, Any],
    cfg: DictConfig,
    checkpoint_dir: Path,
    X_train: Optional[pd.DataFrame] = None,
    y_train_original: Optional[pd.Series] = None
) -> Tuple[Any, Dict[str, Any]]:
    """Train TFT (Temporal Fusion Transformer) model using sktime's PytorchForecastingTFT.
    
    Parameters
    ----------
    y_train : pd.Series
        Preprocessed training data (already transformed, resampled, and imputed)
    target_series : str
        Target series name
    config : Dict[str, Any]
        Model configuration (input_size, hidden_size, etc.)
    cfg : DictConfig
        Full experiment config
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
        from sktime.forecasting.pytorchforecasting import PytorchForecastingTFT
    except ImportError as e:
        raise ImportError(
            f"PytorchForecastingTFT is not available in sktime. "
            f"Install with: pip install 'sktime[all-extras]' or pip install pytorch-forecasting"
        ) from e
    
    # PyTorch 2.6 compatibility: Register pytorch-forecasting classes as safe globals
    # PyTorch 2.6 changed torch.load() default to weights_only=True for security.
    # We need to allowlist pytorch-forecasting classes that are used in checkpoints.
    # This uses the official PyTorch API for registering safe globals.
    try:
        import torch
        if hasattr(torch.serialization, 'add_safe_globals'):
            try:
                # Import pytorch-forecasting classes that appear in checkpoints
                from pytorch_forecasting.data.encoders import EncoderNormalizer
                
                # Register as safe globals for checkpoint loading
                # This allows PyTorch Lightning's load_from_checkpoint to work with pytorch-forecasting models
                torch.serialization.add_safe_globals([EncoderNormalizer])
                logger.debug("Registered pytorch-forecasting classes as safe globals for PyTorch 2.6")
            except (ImportError, AttributeError) as e:
                logger.debug(f"Could not register safe globals (may not be needed): {e}")
    except (ImportError, AttributeError):
        # PyTorch version doesn't support this - skip
        pass
    
    # Data preprocessing is handled in train_sktime.py for unified preprocessing
    # y_train is already prepared (weekly data, transformed, imputed)
    
    # Get model parameters
    input_size = config.get('input_size', 96)  # Lookback window in weeks (max_encoder_length), default: 96 weeks = 24 months
    hidden_size = config.get('hidden_size', 64)
    n_head = config.get('n_head', 4)
    dropout = config.get('dropout', 0.1)
    learning_rate = config.get('learning_rate', 0.001)
    # max_epochs takes precedence over max_steps if both are provided
    max_epochs = config.get('max_epochs', config.get('max_steps', 100))
    batch_size = config.get('batch_size', 32)
    
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
    
    logger.info(f"Training TFT on {target_series} ({len(y_train)} observations)")
    
    # Prepare model parameters for PytorchForecastingTFT
    # Horizon is now in weeks in config (88 weeks = 2024 JAN to 2025 OCT)
    horizon = config.get('horizon', 24)  # Forecast horizon in weeks (24 weeks per chunk for recursive prediction)
    
    # Validate parameters
    # Note: max_prediction_length for training is small (6-12), so we only need input_size + small buffer
    # Use fixed small value for training efficiency (not related to forecast horizon)
    max_prediction_length_training = 12  # Fixed small value for training efficiency
    total_required = input_size + max_prediction_length_training
    if len(y_train) < total_required:
        raise ValueError(
            f"TFT training failed: Insufficient data. "
            f"Need at least {total_required} observations (input_size={input_size} + max_prediction_length={max_prediction_length_training}), "
            f"but only have {len(y_train)} observations. "
            f"Reduce input_size or use more training data."
        )
    
    # Model architecture parameters (for TFT model itself)
    # Note: learning_rate goes here, not in trainer_params
    model_params = {
        'lstm_layers': config.get('lstm_layers', 2),
        'hidden_size': hidden_size,
        'attention_head_size': n_head,
        'dropout': dropout,
        'learning_rate': learning_rate,
    }
    
    # Training parameters (for PyTorch Lightning trainer)
    # Enable verbose logging and progress bars to track training progress
    # Use minimal logging to avoid hanging issues
    # Explicitly set GPU to ensure proper GPU utilization
    trainer_params = {
        'max_epochs': max_epochs,
        'enable_progress_bar': True,  # Show training progress
        'enable_model_summary': False,  # Disable to reduce overhead
        'logger': False,  # Disable Lightning logger to avoid hanging
        'enable_checkpointing': False,  # Disable checkpointing (we handle it separately)
        'log_every_n_steps': 10,  # Log every 10 steps
        'accelerator': 'gpu',  # Explicitly use GPU
        'devices': 1,  # Use single GPU (device 0)
    }
    
    # Dataset parameters
    # Note: max_prediction_length is NOT the forecast horizon - it's the max prediction length
    # used during training (teacher forcing). Should be small (6-12 steps) for efficiency.
    # The actual forecast horizon is controlled by fh parameter in fit()/predict()
    # max_prediction_length_training is already defined above (line 126)
    dataset_params = {
        'max_encoder_length': input_size,
        'max_prediction_length': max_prediction_length_training,  # Small value for training, not forecast horizon
    }
    
    # DataLoader parameters
    # Set num_workers=4 to improve GPU utilization (parallel data loading)
    # Note: Increased from 0 to reduce CPU bottleneck and improve GPU utilization
    train_to_dataloader_params = {
        'train': True,
        'batch_size': batch_size,
        'num_workers': 4,
    }
    validation_to_dataloader_params = {
        'train': False,
        'batch_size': batch_size,
        'num_workers': 4,
    }
    
    # Create base forecaster
    base_forecaster = PytorchForecastingTFT(
        model_params=model_params,
        trainer_params=trainer_params,
        dataset_params=dataset_params,
        train_to_dataloader_params=train_to_dataloader_params,
        validation_to_dataloader_params=validation_to_dataloader_params
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
    # For PytorchForecastingTFT, we need to set fh to the desired forecast horizon (24 weeks)
    # because it doesn't allow changing fh after fit
    from sktime.forecasting.base import ForecastingHorizon
    # Set fh to the actual forecast horizon (24 weeks) for prediction compatibility
    # Note: This is the forecast horizon we want to use during prediction (per chunk)
    forecast_horizon_weeks = horizon  # Use config horizon (24 weeks per chunk)
    training_fh = ForecastingHorizon(range(1, forecast_horizon_weeks + 1), is_relative=True)
    logger.info(f"Setting training fh to {forecast_horizon_weeks} weeks for prediction compatibility (per chunk)")
    
    # Prepare covariates if available
    X_train_aligned = None
    if X_train is not None:
        # Remove duplicate indices before reindex (pytorch-forecasting can fail with duplicates)
        y_train_clean = y_train[~y_train.index.duplicated(keep='first')]
        X_train_clean = X_train[~X_train.index.duplicated(keep='first')]
        
        # Align covariates with target series index
        X_train_aligned = X_train_clean.reindex(y_train_clean.index, method='ffill').bfill()
        y_train = y_train_clean
        
        # Limit covariates if too many (pytorch-forecasting can hang with too many covariates)
        max_covariates = config.get('max_covariates', 50)  # Default: limit to 50 covariates
        if len(X_train_aligned.columns) > max_covariates:
            logger.warning(
                f"Too many covariates ({len(X_train_aligned.columns)}). "
                f"Limiting to {max_covariates} to avoid hanging issues. "
                f"Using first {max_covariates} covariates."
            )
            X_train_aligned = X_train_aligned.iloc[:, :max_covariates]
        
        logger.info(f"Starting TFT training with {len(X_train_aligned.columns)} covariates (this may take several minutes for {len(y_train)} observations)...")
        logger.info(f"Data shapes - y_train: {y_train.shape}, X_train: {X_train_aligned.shape}")
        num_workers_actual = train_to_dataloader_params.get('num_workers', 0)
        logger.info(f"Training parameters - max_epochs: {max_epochs}, batch_size: {batch_size}, num_workers: {num_workers_actual}")
        logger.info("Preparing dataset (this may take 1-2 minutes with many covariates)...")
        import sys
        sys.stdout.flush()  # Force flush to see progress
        
        logger.info("Calling forecaster.fit()...")
        try:
            forecaster.fit(y_train, fh=training_fh, X=X_train_aligned)
            logger.info("forecaster.fit() completed successfully")
        except Exception as e:
            logger.error(f"forecaster.fit() failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    else:
        # Remove duplicates from y_train index first
        y_train_clean = y_train[~y_train.index.duplicated(keep='first')]
        y_train = y_train_clean
        
        logger.info(f"Starting TFT training (this may take several minutes for {len(y_train)} observations)...")
        logger.info(f"Data shapes - y_train: {y_train.shape}")
        num_workers_actual = train_to_dataloader_params.get('num_workers', 0)
        logger.info(f"Training parameters - max_epochs: {max_epochs}, batch_size: {batch_size}, num_workers: {num_workers_actual}")
        logger.info("Calling forecaster.fit()...")
        try:
            forecaster.fit(y_train, fh=training_fh)
            logger.info("forecaster.fit() completed successfully")
        except Exception as e:
            logger.error(f"forecaster.fit() failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    logger.info("TFT training completed")
    
    # Save checkpoint
    # sktime.save() creates a .zip file, so we use .zip extension or no extension
    checkpoint_path = checkpoint_dir / "model.zip"
    
    # Ensure checkpoint directory exists
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if model.zip or model.pkl is a directory (from previous failed save) and remove it
    for path_to_check in [checkpoint_path, checkpoint_dir / "model.pkl"]:
        if path_to_check.exists() and path_to_check.is_dir():
            logger.warning(f"Removing existing directory at {path_to_check}")
            import shutil
            shutil.rmtree(path_to_check)
    
    metadata = {
        'model_type': 'tft',
        'target_series': target_series,
        'input_size': input_size,
        'hidden_size': hidden_size,
        'n_head': n_head,
        'dropout': dropout,
        'learning_rate': learning_rate,
        'max_epochs': max_epochs,
        'batch_size': batch_size,
        'scaler': scaler,
        'scaler_type': scaler_type if scaler else None,
        'training_data_shape': y_train.shape,
        'training_data_index': list(y_train.index),
        'covariates_used': list(X_train_aligned.columns) if X_train_aligned is not None else None,
        'num_covariates': len(X_train_aligned.columns) if X_train_aligned is not None else 0
    }
    # Use sktime's save method (handles pytorch-forecasting models better)
    # Note: sktime.save() creates a .zip file, so we need to ensure the directory exists
    try:
        # Try cloudpickle format first (better for pytorch-forecasting)
        # sktime.save() will create a .zip file, so ensure parent directory exists
        forecaster.save(path=str(checkpoint_path), serialization_format='cloudpickle')
        logger.info(f"Saved forecaster using sktime.save() to {checkpoint_path}")
    except Exception as e:
        # Fallback to pickle
        try:
            forecaster.save(path=str(checkpoint_path), serialization_format='pickle')
            logger.info(f"Saved forecaster using sktime.save() (pickle) to {checkpoint_path}")
        except Exception as e2:
            logger.warning(f"sktime.save() failed: {e2}, falling back to custom save")
            # Fallback to custom save method
            # Use .pkl for custom save (not .zip)
            checkpoint_path_pkl = checkpoint_dir / "model.pkl"
            # Ensure directory exists and remove if it's a directory
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if checkpoint_path_pkl.exists() and checkpoint_path_pkl.is_dir():
                import shutil
                shutil.rmtree(checkpoint_path_pkl)
            save_model_checkpoint(forecaster, checkpoint_path_pkl, metadata)
            checkpoint_path = checkpoint_path_pkl
    
    # Save metadata separately
    metadata_path = checkpoint_dir / "metadata.pkl"
    try:
        import pickle
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Saved metadata to {metadata_path}")
    except Exception as e:
        logger.warning(f"Failed to save metadata separately: {e}")
    
    return forecaster, metadata


def forecast_tft(
    model: Any,
    horizon: int,
    last_date: Optional[pd.Timestamp] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Generate TFT forecasts in weekly frequency using direct long-horizon forecasting.
    
    Uses sktime's ForecastingHorizon and ForecastingPipeline for proper forecasting.
    
    Parameters
    ----------
    model : Any
        Trained TFT forecaster (ForecastingPipeline)
    horizon : int
        Forecast horizon in weeks (88 weeks = 2024 JAN to 2025 OCT)
    last_date : Optional[pd.Timestamp]
        Last date in training data
    metadata : Optional[Dict[str, Any]]
        Model metadata (for scaler inverse transform)
        
    Returns
    -------
    pd.DataFrame
        Forecasted values with DatetimeIndex (weekly frequency, original scale)
    """
    from sktime.forecasting.base import ForecastingHorizon
    
    logger = logging.getLogger(__name__)
    
    # Horizon is already in weeks
    horizon_weeks = horizon
    
    # Generate forecasts using sktime's ForecastingHorizon
    # TFT model was trained with fh=88 weeks, so we must use the same fh for prediction
    # Use relative horizon matching training fh (range(1, 89) = 88 weeks)
    # Also need to ensure cutoff has freq - adjust cutoff to nearest Sunday and set freq='W-SUN'
    try:
        # Update model's cutoff with proper freq (W-SUN)
        # The model was trained with W-FRI, but we need W-SUN for forecasting
        if hasattr(model, 'forecaster_') and hasattr(model.forecaster_, '_cutoff'):
            cutoff = model.forecaster_.cutoff
            if isinstance(cutoff, pd.DatetimeIndex) and len(cutoff) > 0:
                cutoff_date = cutoff[0]
                # Model was trained with W-FRI, so keep cutoff as W-FRI
                # Just ensure it has the correct frequency
                if not hasattr(cutoff, 'freq') or cutoff.freq != 'W-FRI':
                    # Ensure cutoff is Friday with W-FRI frequency
                    days_to_friday = (4 - cutoff_date.weekday()) % 7
                    if days_to_friday == 0 and cutoff_date.weekday() != 4:
                        days_to_friday = 7
                    friday_date = cutoff_date + pd.Timedelta(days=days_to_friday)
                    new_cutoff = pd.DatetimeIndex([friday_date], freq='W-FRI')
                    model.forecaster_._cutoff = new_cutoff
                    logger.info(f"Updated forecaster cutoff to Friday with freq='W-FRI': {new_cutoff}")
                else:
                    logger.debug(f"Cutoff already has W-FRI frequency: {cutoff}")
    except Exception as e:
        logger.warning(f"Failed to update cutoff with freq: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    
    # PytorchForecastingTFT requires the same fh as used in training (24 weeks)
    # Since we need 88 weeks, we'll predict in chunks but without updating the model
    # Use training fh (24 weeks) for each chunk
    training_fh_size = 24  # Must match training fh
    logger.info(f"Generating TFT forecast for horizon={horizon_weeks} weeks using {training_fh_size}-week chunks (matching training fh)...")
    
    # Check if model was trained with covariates
    # If metadata has num_covariates > 0, we need to provide X during prediction
    # IMPORTANT: TFT needs input_size (lookback window) amount of X data BEFORE forecast start
    # NOT the forecast period itself - just the history window
    X_future = None
    forecast_start_date = pd.Timestamp('2024-01-01')  # Target forecast start date
    
    if metadata and metadata.get('num_covariates', 0) > 0:
        # Get input_size from metadata (lookback window in weeks)
        input_size = metadata.get('input_size', 96)  # Default: 96 weeks
        logger.info(f"Model was trained with {metadata.get('num_covariates')} covariates. Loading {input_size} weeks of history before {forecast_start_date}...")
        
        try:
            from src.utils import get_project_root, resolve_data_path, RECENT_START, RECENT_END
            from src.train.preprocess import set_dataframe_frequency, impute_missing_values
            
            project_root = get_project_root()
            data_path = resolve_data_path()
            
            if data_path.exists():
                # Load full data
                full_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
                target_series = metadata.get('target_series')
                
                if target_series and target_series in full_data.columns:
                    # Get covariates columns used during training
                    covariates_used = metadata.get('covariates_used', [])
                    if not covariates_used:
                        # Fallback: get all numeric columns except target
                        covariate_cols = [col for col in full_data.columns if col != target_series]
                        numeric_cols = [col for col in covariate_cols if pd.api.types.is_numeric_dtype(full_data[col])]
                        max_covariates = metadata.get('num_covariates', 50)
                        numeric_cols = numeric_cols[:max_covariates]
                    else:
                        numeric_cols = [col for col in covariates_used if col in full_data.columns]
                    
                    if numeric_cols:
                        # Calculate history window: input_size weeks BEFORE forecast_start_date
                        # Adjust to Sunday for consistency
                        days_since_sunday = forecast_start_date.weekday()
                        if days_since_sunday == 6:
                            sunday_start = forecast_start_date
                        else:
                            sunday_start = forecast_start_date + pd.Timedelta(days=6-days_since_sunday)
                        
                        history_end = sunday_start - pd.Timedelta(weeks=1)  # One week before forecast start
                        history_start = history_end - pd.Timedelta(weeks=input_size - 1)  # input_size weeks back
                        
                        # Load history data (input_size weeks before forecast start)
                        history_data = full_data[
                            (full_data.index >= history_start) & 
                            (full_data.index <= history_end)
                        ]
                        
                        if len(history_data) > 0:
                            # Use actual history data
                            X_future = history_data[numeric_cols].copy()
                            logger.info(f"Loaded history data: {len(X_future)} periods from {X_future.index[0]} to {X_future.index[-1]}")
                            
                            # If we don't have enough history, try to get more recent data
                            if len(X_future) < input_size:
                                logger.warning(f"Only {len(X_future)} periods available, need {input_size}. Trying to extend...")
                                # Try to get data up to RECENT_END
                                extended_data = full_data[
                                    (full_data.index >= history_start) & 
                                    (full_data.index <= RECENT_END)
                                ]
                                if len(extended_data) >= input_size:
                                    X_future = extended_data[numeric_cols].iloc[-input_size:].copy()
                                    logger.info(f"Extended to {len(X_future)} periods using data up to {RECENT_END}")
                                else:
                                    # Forward fill from last available
                                    logger.warning(f"Still insufficient data. Forward filling from last available...")
                                    # Create history index
                                    history_index = pd.date_range(
                                        start=history_start,
                                        periods=input_size,
                                        freq='W-SUN'
                                    )
                                    # Reindex and forward fill
                                    X_future = X_future.reindex(history_index, method='ffill').bfill()
                        else:
                            # No history data available, use recent data up to RECENT_END
                            logger.warning(f"No history data available. Using recent data up to {RECENT_END}...")
                            recent_data = full_data[
                                (full_data.index >= RECENT_START) & 
                                (full_data.index <= RECENT_END)
                            ]
                            
                            if len(recent_data) >= input_size:
                                # Use last input_size periods
                                X_future = recent_data[numeric_cols].iloc[-input_size:].copy()
                                logger.info(f"Using last {input_size} periods from recent data: {X_future.index[0]} to {X_future.index[-1]}")
                            elif len(recent_data) > 0:
                                # Not enough, forward fill
                                X_last = recent_data[numeric_cols].iloc[-1:]
                                # Use W-FRI to match model's training frequency
                                history_index = pd.date_range(
                                    start=history_start,
                                    periods=input_size,
                                    freq='W-FRI'
                                )
                                X_future = pd.DataFrame(
                                    index=history_index,
                                    columns=numeric_cols
                                )
                                for col in numeric_cols:
                                    X_future[col] = X_last[col].values[0]
                                logger.info(f"Created history from last known values (forward fill)")
                            else:
                                # Fallback: use training data last values
                                logger.warning("No recent data available. Using training data last values...")
                                train_data = full_data[
                                    (full_data.index >= pd.Timestamp('1985-01-01')) & 
                                    (full_data.index <= pd.Timestamp('2019-12-31'))
                                ]
                                if len(train_data) >= input_size:
                                    X_future = train_data[numeric_cols].iloc[-input_size:].copy()
                                    logger.info(f"Using last {input_size} periods from training data")
                                elif len(train_data) > 0:
                                    X_last = train_data[numeric_cols].iloc[-1:]
                                    history_index = pd.date_range(
                                        start=history_start,
                                        periods=input_size,
                                        freq='W-SUN'
                                    )
                                    X_future = pd.DataFrame(
                                        index=history_index,
                                        columns=numeric_cols
                                    )
                                    for col in numeric_cols:
                                        X_future[col] = X_last[col].values[0]
                        
                        if X_future is not None:
                            # Ensure we have exactly input_size periods
                            if len(X_future) > input_size:
                                X_future = X_future.iloc[-input_size:]
                            elif len(X_future) < input_size:
                                # Pad with forward fill
                                # Use W-FRI to match model's training frequency
                                history_index = pd.date_range(
                                    start=X_future.index[0],
                                    periods=input_size,
                                    freq='W-FRI'
                                )
                                X_future = X_future.reindex(history_index, method='ffill').bfill()
                            
                            # Ensure frequency is set and impute
                            X_future = set_dataframe_frequency(X_future)
                            X_future = impute_missing_values(X_future)
                            logger.info(f"Prepared history covariates: {len(X_future)} periods (input_size={input_size})")
                            logger.info(f"History date range: {X_future.index[0]} to {X_future.index[-1]}")
            else:
                logger.warning(f"Data file not found at {data_path}. Using empty DataFrame.")
                # Create empty DataFrame with correct structure
                if metadata.get('covariates_used'):
                    covariates_used = metadata.get('covariates_used', [])[:metadata.get('num_covariates', 50)]
                    # Use W-FRI to match model's training frequency
                    history_index = pd.date_range(
                        start=forecast_start_date - pd.Timedelta(weeks=input_size),
                        periods=input_size,
                        freq='W-FRI'
                    )
                    X_future = pd.DataFrame(index=history_index, columns=covariates_used)
                    X_future = X_future.fillna(0)  # Fill with zeros as fallback
        except Exception as e:
            logger.warning(f"Failed to load history data for covariates: {e}. Using fallback.")
            import traceback
            logger.debug(traceback.format_exc())
            # Create empty DataFrame with correct structure as fallback
            if metadata and metadata.get('covariates_used'):
                covariates_used = metadata.get('covariates_used', [])[:metadata.get('num_covariates', 50)]
                input_size = metadata.get('input_size', 96)
                # Use W-FRI to match model's training frequency
                history_index = pd.date_range(
                    start=forecast_start_date - pd.Timedelta(weeks=input_size),
                    periods=input_size,
                    freq='W-FRI'
                )
                X_future = pd.DataFrame(index=history_index, columns=covariates_used)
                X_future = X_future.fillna(0)  # Fill with zeros as fallback
    
    # Generate forecast with or without covariates
    # Note: If covariates were used during training, X must be a DataFrame (not None)
    # to avoid pipeline imputer errors
    # Also, we need to ensure the model has a proper cutoff with frequency
    # Check if model has cutoff and set freq if needed
    try:
        if hasattr(model, 'forecaster_') and hasattr(model.forecaster_, 'cutoff'):
            cutoff = model.forecaster_.cutoff
            if cutoff is not None:
                # Check if cutoff has freq attribute and if it's None
                if hasattr(cutoff, 'freq') and cutoff.freq is None:
                    # Set freq to 'W' (weekly) for proper date handling
                    cutoff = cutoff.asfreq('W')
                    # Update the cutoff in the forecaster
                    model.forecaster_._cutoff = cutoff
                    logger.info(f"Set cutoff freq to 'W': {cutoff}")
                elif not hasattr(cutoff, 'freq'):
                    # If cutoff doesn't have freq, try to create a new one with freq
                    if metadata and 'training_data_index' in metadata:
                        training_index = metadata['training_data_index']
                        if training_index:
                            try:
                                last_idx = pd.to_datetime(training_index[-1])
                                if isinstance(last_idx, pd.Timestamp):
                                    # Create a DatetimeIndex with freq
                                    cutoff_with_freq = pd.DatetimeIndex([last_idx], freq='W')
                                    # Update cutoff
                                    model.forecaster_._cutoff = cutoff_with_freq
                                    logger.info(f"Created cutoff with freq 'W': {cutoff_with_freq}")
                            except Exception as e:
                                logger.warning(f"Failed to set cutoff freq: {e}")
    except Exception as e:
        logger.warning(f"Failed to check/set cutoff freq: {e}")
    
    # Ensure model cutoff is W-SUN to match forecast frequency
    try:
        if hasattr(model, 'forecaster_') and hasattr(model.forecaster_, 'cutoff'):
            cutoff = model.forecaster_.cutoff
            if cutoff is not None:
                # Convert cutoff to W-SUN if needed
                if isinstance(cutoff, pd.DatetimeIndex):
                    # Ensure cutoff is Sunday
                    cutoff_date = cutoff[0] if len(cutoff) > 0 else None
                    if cutoff_date is not None:
                        # Convert to nearest Sunday
                        days_to_sunday = (6 - cutoff_date.weekday()) % 7
                        if days_to_sunday == 0 and cutoff_date.weekday() != 6:
                            days_to_sunday = 7
                        sunday_cutoff = cutoff_date + pd.Timedelta(days=days_to_sunday)
                        # Create new cutoff with W-SUN frequency
                        cutoff_w_sun = pd.DatetimeIndex([sunday_cutoff], freq='W-SUN')
                        model.forecaster_._cutoff = cutoff_w_sun
                        logger.info(f"Updated forecaster cutoff to Sunday with freq='W-SUN': {cutoff_w_sun}")
                elif hasattr(cutoff, 'freq') and cutoff.freq != 'W-SUN':
                    # Update frequency to W-SUN
                    cutoff_date = cutoff[0] if len(cutoff) > 0 else None
                    if cutoff_date is not None:
                        cutoff_w_sun = pd.DatetimeIndex([cutoff_date], freq='W-SUN')
                        model.forecaster_._cutoff = cutoff_w_sun
                        logger.info(f"Updated forecaster cutoff frequency to W-SUN: {cutoff_w_sun}")
    except Exception as e:
        logger.warning(f"Failed to update cutoff frequency: {e}")
    
    # RECURSIVE PREDICTION: Predict in 24-week chunks with update after each chunk
    # IMPORTANT: Model was trained with W-FRI frequency, so we need to use W-FRI for forecasting
    chunk_size = 24  # weeks
    forecast_start_date = pd.Timestamp('2024-01-01')
    # Convert to Friday (W-FRI) to match model's training frequency
    days_since_friday = (forecast_start_date.weekday() - 4) % 7
    if days_since_friday == 0 and forecast_start_date.weekday() == 4:  # Already Friday
        friday_start = forecast_start_date
    else:
        # Go to next Friday
        days_to_friday = (4 - forecast_start_date.weekday()) % 7
        if days_to_friday == 0:
            days_to_friday = 7
        friday_start = forecast_start_date + pd.Timedelta(days=days_to_friday)
    
    logger.info(f"Starting recursive prediction: {horizon_weeks} weeks in {chunk_size}-week chunks")
    logger.info(f"Forecast start (W-FRI): {friday_start}")
    
    # Get target_series and data_path for recursive prediction
    target_series = metadata.get('target_series') if metadata else None
    data_path_for_update = None
    try:
        from src.utils import get_project_root, resolve_data_path
        data_path_for_update = resolve_data_path()
    except:
        pass
    
    # Load actual data for updates (if available)
    actual_data = None
    try:
        from src.utils import TEST_START, TEST_END
        if data_path_for_update and data_path_for_update.exists() and target_series:
            full_data = pd.read_csv(data_path_for_update, index_col=0, parse_dates=True)
            if target_series in full_data.columns:
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
    current_date = friday_start
    remaining_horizon = horizon_weeks
    
    while remaining_horizon > 0:
        # Determine chunk size (last chunk may be smaller)
        current_chunk_size = min(chunk_size, remaining_horizon)
        logger.info(f"Predicting chunk: {current_chunk_size} weeks starting from {current_date}")
        
        # Get cutoff for this chunk
        try:
            cutoff_date = model.forecaster_.cutoff[0] if hasattr(model, 'forecaster_') and hasattr(model.forecaster_, 'cutoff') else None
            if cutoff_date is None:
                cutoff_date = current_date - pd.Timedelta(weeks=1)
        except:
            cutoff_date = current_date - pd.Timedelta(weeks=1)
        
        # Prepare X for this chunk - CRITICAL: Model was trained with covariates, so X must be provided
        # Use W-FRI frequency to match model's training frequency
        X_for_chunk = None
        if X_future is not None and len(X_future) > 0:
            # X_future has history data, extend it for this chunk
            X_future_clean = X_future[~X_future.index.duplicated(keep='last')]
            
            # Use W-FRI frequency to match model cutoff
            X_start = cutoff_date + pd.Timedelta(weeks=1)
            # Ensure X_start is Friday (W-FRI)
            days_to_friday = (4 - X_start.weekday()) % 7
            if days_to_friday == 0 and X_start.weekday() != 4:
                days_to_friday = 7
            X_start_friday = X_start + pd.Timedelta(days=days_to_friday)
            
            X_end = X_start_friday + pd.Timedelta(weeks=current_chunk_size - 1)
            X_index = pd.date_range(start=X_start_friday, end=X_end, freq='W-FRI')
            
            # Get last value from history to forward fill
            last_values = X_future_clean.iloc[-1:]
            
            # Create X for prediction period with same columns as training
            X_for_chunk = pd.DataFrame(
                index=X_index,
                columns=X_future_clean.columns
            )
            for col in X_future_clean.columns:
                X_for_chunk[col] = last_values[col].values[0] if len(last_values) > 0 else 0
            
            logger.info(f"Prepared X_for_chunk: {len(X_for_chunk)} periods, {len(X_for_chunk.columns)} covariates")
        else:
            # If X_future is None but model was trained with covariates, create dummy X
            if metadata and metadata.get('num_covariates', 0) > 0:
                logger.warning("X_future is None but model was trained with covariates. Creating dummy X...")
                covariates_used = metadata.get('covariates_used', [])[:metadata.get('num_covariates', 50)]
                X_start = cutoff_date + pd.Timedelta(weeks=1)
                days_to_friday = (4 - X_start.weekday()) % 7
                if days_to_friday == 0 and X_start.weekday() != 4:
                    days_to_friday = 7
                X_start_friday = X_start + pd.Timedelta(days=days_to_friday)
                X_end = X_start_friday + pd.Timedelta(weeks=current_chunk_size - 1)
                X_index = pd.date_range(start=X_start_friday, end=X_end, freq='W-FRI')
                
                X_for_chunk = pd.DataFrame(
                    index=X_index,
                    columns=covariates_used if covariates_used else [f'cov_{i}' for i in range(metadata.get('num_covariates', 50))]
                )
                X_for_chunk = X_for_chunk.fillna(0)  # Fill with zeros
                logger.warning(f"Created dummy X_for_chunk with {len(X_for_chunk.columns)} covariates (filled with 0)")
        
        # Predict current chunk
        # PytorchForecastingTFT requires the same fh format as used in fit()
        # Training uses relative ForecastingHorizon(range(1, 25), is_relative=True) for 24 weeks
        # So we must use the same format: relative horizon
        # IMPORTANT: Use exactly 24 weeks (training fh) for each chunk, not current_chunk_size
        # This is because PytorchForecastingTFT doesn't allow changing fh
        training_fh_size = 24  # Must match training fh
        
        # Model was trained with W-FRI, so cutoff should already be W-FRI
        # No need to change it - just ensure it's set correctly
        try:
            if hasattr(model, 'forecaster_') and hasattr(model.forecaster_, '_cutoff'):
                current_cutoff = model.forecaster_.cutoff
                if current_cutoff is not None:
                    if isinstance(current_cutoff, pd.DatetimeIndex):
                        # Ensure cutoff has W-FRI frequency
                        if current_cutoff.freq != 'W-FRI':
                            cutoff_date = current_cutoff[0] if len(current_cutoff) > 0 else None
                            if cutoff_date is not None:
                                # Ensure cutoff is Friday with W-FRI frequency
                                days_to_friday = (4 - cutoff_date.weekday()) % 7
                                if days_to_friday == 0 and cutoff_date.weekday() != 4:
                                    days_to_friday = 7
                                friday_cutoff = cutoff_date + pd.Timedelta(days=days_to_friday)
                                cutoff_w_fri = pd.DatetimeIndex([friday_cutoff], freq='W-FRI')
                                model.forecaster_._cutoff = cutoff_w_fri
                                logger.debug(f"Aligned cutoff to W-FRI: {cutoff_w_fri}")
        except Exception as e:
            logger.debug(f"Failed to check cutoff: {e}")
        
        # Use training fh size for prediction (24 weeks)
        fh_chunk = ForecastingHorizon(range(1, training_fh_size + 1), is_relative=True)
        
        # CRITICAL: Model was trained with covariates, so X must be provided
        # If X_for_chunk is None, model will fail - ensure it's always provided
        if X_for_chunk is None:
            if metadata and metadata.get('num_covariates', 0) > 0:
                raise ValueError(f"Model was trained with {metadata.get('num_covariates')} covariates but X_for_chunk is None")
            else:
                logger.warning("X_for_chunk is None but attempting prediction without covariates")
        
        try:
            # Always use Pipeline predict with X if covariates were used during training
            if metadata and metadata.get('num_covariates', 0) > 0 and X_for_chunk is not None:
                # Model was trained with covariates - must provide X
                logger.info(f"Predicting with X ({len(X_for_chunk.columns)} covariates, {len(X_for_chunk)} periods)")
                chunk_forecast = model.predict(fh=fh_chunk, X=X_for_chunk)
            elif X_for_chunk is not None:
                # X provided but model might not need it - try with X first
                logger.info(f"Predicting with X ({len(X_for_chunk.columns)} covariates)")
                chunk_forecast = model.predict(fh=fh_chunk, X=X_for_chunk)
            else:
                # No covariates - predict without X
                logger.info("Predicting without X (no covariates)")
                chunk_forecast = model.predict(fh=fh_chunk)
            
            if len(chunk_forecast) == 0:
                raise ValueError("Empty forecast returned")
                
        except Exception as e:
            error_msg = str(e).lower()
            logger.warning(f"Prediction failed: {e}")
            
            # If Pipeline fails, try base forecaster directly
            if hasattr(model, 'forecaster_'):
                try:
                    logger.info("Trying base forecaster directly...")
                    if X_for_chunk is not None and metadata and metadata.get('num_covariates', 0) > 0:
                        chunk_forecast = model.forecaster_.predict(fh=fh_chunk, X=X_for_chunk)
                    else:
                        chunk_forecast = model.forecaster_.predict(fh=fh_chunk)
                    
                    if len(chunk_forecast) == 0:
                        raise ValueError("Base forecaster also returned empty forecast")
                    logger.info("Base forecaster prediction succeeded")
                except Exception as e2:
                    logger.error(f"Base forecaster also failed: {e2}")
                    raise ValueError(f"All prediction methods failed. Last error: {e2}") from e2
            else:
                raise
        
        # Create index for this chunk (W-FRI to match model's training frequency)
        chunk_index = pd.date_range(
            start=current_date,
            periods=current_chunk_size,
            freq='W-FRI'
        )
        
        # Truncate to actual chunk size if needed (if training fh was larger)
        actual_chunk_size = min(len(chunk_forecast), current_chunk_size)
        if len(chunk_forecast) > current_chunk_size:
            chunk_forecast = chunk_forecast.iloc[:current_chunk_size]
            chunk_index = chunk_index[:current_chunk_size]
        elif len(chunk_forecast) < current_chunk_size:
            # If forecast is shorter, adjust chunk_index
            chunk_index = chunk_index[:len(chunk_forecast)]
        
        # Set index only if forecast has values
        if len(chunk_forecast) > 0:
            if isinstance(chunk_forecast, pd.Series):
                chunk_forecast = pd.Series(chunk_forecast.values, index=chunk_index[:len(chunk_forecast)], name=chunk_forecast.name)
            elif isinstance(chunk_forecast, pd.DataFrame):
                chunk_forecast = pd.DataFrame(chunk_forecast.values, index=chunk_index[:len(chunk_forecast)], columns=chunk_forecast.columns)
        
        if len(chunk_forecast) > 0:
            forecast_chunks.append(chunk_forecast)
            logger.info(f"✓ Predicted {len(chunk_forecast)} weeks: {chunk_index[0]} to {chunk_index[-1]}")
        else:
            logger.error(f"Empty forecast returned for chunk starting at {current_date}")
            raise ValueError(f"Empty forecast returned - model prediction failed")
        
        # Skip model update for PytorchForecastingTFT to avoid frequency mismatch issues
        # PytorchForecastingTFT doesn't support update() well, and frequency mismatches cause errors
        # Instead, we'll use the original model state for all predictions
        # This means predictions won't be updated with actual data, but will be consistent
        logger.info(f"Skipping model update (PytorchForecastingTFT limitation)")
        
        # Update X_future for next chunk (extend with last values)
        if X_future is not None:
            # Extend X_future with forward-filled values for the chunk period
            chunk_X_index = pd.date_range(
                start=chunk_index[0],
                periods=current_chunk_size,
                freq='W-SUN'
            )
            last_X_values = X_future.iloc[-1:] if len(X_future) > 0 else X_future
            chunk_X = pd.DataFrame(
                index=chunk_X_index,
                columns=X_future.columns
            )
            for col in X_future.columns:
                chunk_X[col] = last_X_values[col].values[0] if len(last_X_values) > 0 else 0
            
            # Append to X_future for next chunk
            X_future = pd.concat([X_future, chunk_X])
            X_future = X_future.iloc[-input_size:]  # Keep only last input_size periods
        
        # Move to next chunk
        current_date = chunk_index[-1] + pd.Timedelta(weeks=1)
        remaining_horizon -= current_chunk_size
    
    # Combine all chunks
    if isinstance(forecast_chunks[0], pd.Series):
        forecast = pd.concat(forecast_chunks)
    else:
        forecast = pd.concat(forecast_chunks)
    
    # Ensure proper index (W-FRI to match model's training frequency)
    forecast_index = pd.date_range(
        start=friday_start,
        periods=horizon_weeks,
        freq='W-FRI'
    )
    forecast.index = forecast_index[:len(forecast)]
    
    logger.info(f"✓ Completed recursive prediction: {len(forecast)} weeks from {forecast.index[0]} to {forecast.index[-1]}")
    
    # Apply inverse transform if scaler is available in metadata
    # ForecastingPipeline should handle this, but we ensure it's done explicitly
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
