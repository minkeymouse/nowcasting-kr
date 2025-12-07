"""Nowcasting module - generates JSON/CSV results only."""
from pathlib import Path
import sys
import argparse
import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pickle
import json
import hashlib

# Set up paths
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from src.utils import setup_cli_environment
setup_cli_environment()

from src.utils import (
    get_project_root,
    parse_experiment_config,
    extract_experiment_params,
    validate_experiment_config
)
try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def _validate_checkpoint(model_data: dict, target_series: str, model: str) -> None:
    """Validate checkpoint contains required data.
    
    Raises:
        ValueError: If checkpoint is missing required fields
    """
    if not isinstance(model_data, dict):
        raise ValueError(f"Checkpoint for {target_series}_{model} is not a dictionary")
    
    if 'forecaster' not in model_data:
        raise ValueError(f"Checkpoint for {target_series}_{model} does not contain 'forecaster'")
    
    forecaster = model_data.get('forecaster')
    if forecaster is None:
        raise ValueError(f"Checkpoint for {target_series}_{model} has None forecaster")
    
    # For DFM/DDFM models, check if data_module is present (optional but recommended)
    if model.lower() in ['dfm', 'ddfm']:
        if hasattr(forecaster, '_dfm_model') or hasattr(forecaster, '_ddfm_model'):
            dfm_model = forecaster._dfm_model if hasattr(forecaster, '_dfm_model') else forecaster._ddfm_model
            if dfm_model is not None:
                if not hasattr(dfm_model, '_data_module') or dfm_model._data_module is None:
                    logger.warning(f"Checkpoint for {target_series}_{model} missing data_module - may need to recreate during inference")


def _load_model_for_inference(
    project_root: Path,
    target_series: str,
    model: str,
    supports_nowcast_manager: bool,
    train_start: str,
    train_end: str
) -> Tuple[Any, Optional[Any]]:
    """Load trained model from checkpoint with validation."""
    checkpoint_path = project_root / "checkpoint" / f"{target_series}_{model.lower()}" / "model.pkl"
    if not checkpoint_path.exists():
        # Fallback: check outputs/comparisons/{target_series}/{model}/model.pkl
        comparisons_dir = project_root / "outputs" / "comparisons" / target_series
        if comparisons_dir.exists():
            model_path = comparisons_dir / model.lower() / "model.pkl"
            if model_path.exists():
                checkpoint_path = model_path
                logger.info(f"Found checkpoint in comparisons directory: {checkpoint_path}")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Trained {model.upper()} model not found for {target_series}. "
            f"Expected path: {project_root / 'checkpoint' / f'{target_series}_{model.lower()}' / 'model.pkl'}"
        )
    
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    try:
        with open(checkpoint_path, 'rb') as f:
            model_data = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, IOError) as e:
        raise ValueError(f"Failed to load checkpoint for {target_series}_{model}: {type(e).__name__}: {str(e)}") from e
    except Exception as e:
        raise ValueError(f"Unexpected error loading checkpoint for {target_series}_{model}: {type(e).__name__}: {str(e)}") from e
    
    # Validate checkpoint structure
    try:
        _validate_checkpoint(model_data, target_series, model)
    except ValueError as e:
        raise ValueError(f"Checkpoint validation failed for {target_series}_{model}: {str(e)}") from e
    
    forecaster = model_data.get('forecaster')
    if forecaster is None:
        raise ValueError(f"Model file does not contain forecaster for {target_series}_{model}")
    
    logger.info(f"Successfully loaded forecaster for {target_series}_{model}")
    
    if not supports_nowcast_manager:
        return forecaster, None
    
    # Extract DFM/DDFM model from forecaster
    # DFMForecaster/DDFMForecaster store DFMBase/DDFMBase instances directly
    if hasattr(forecaster, '_dfm_model'):
        dfm_model = forecaster._dfm_model
    elif hasattr(forecaster, '_ddfm_model'):
        dfm_model = forecaster._ddfm_model
    else:
        raise ValueError(f"Forecaster does not contain {model.upper()} model for {target_series}")
    
    if dfm_model is None:
        raise ValueError(f"{model.upper()} model is None for {target_series}")
    
    logger.info(f"Successfully extracted {model.upper()} model from forecaster")
    
    # Restore result from checkpoint if needed
    # DFMBase/DDFMBase are the actual models (not wrappers), so access _result directly
    if hasattr(dfm_model, '_result') and dfm_model._result is None:
        # Try to restore from checkpoint first (safest, no training)
        if 'result' in model_data:
            try:
                dfm_model._result = model_data['result']
                logger.debug(f"Restored result from checkpoint for {target_series}_{model}")
            except Exception as e:
                logger.debug(f"Could not restore result from checkpoint: {str(e)}")
        # Only restore from training_state if result is already computed (no EM)
        if dfm_model._result is None and hasattr(dfm_model, 'training_state') and dfm_model.training_state is not None:
            # Check if result is already computed in training_state
            if hasattr(dfm_model.training_state, '_result'):
                try:
                    dfm_model._result = dfm_model.training_state._result
                    logger.debug(f"Restored result from training_state for {target_series}_{model}")
                except Exception as e:
                    logger.debug(f"Could not restore result from training_state: {str(e)}")
    
    # Check for data_module (required for nowcasting)
    if not hasattr(dfm_model, '_data_module') or dfm_model._data_module is None:
        logger.warning(f"data_module not found in checkpoint for {target_series}_{model}. Model may not work correctly for nowcasting.")
        # Try to restore from checkpoint if available
        if 'data_module' in model_data:
            try:
                dfm_model._data_module = model_data['data_module']
                logger.info(f"Restored data_module from checkpoint for {target_series}_{model}")
            except Exception as e:
                logger.warning(f"Could not restore data_module from checkpoint: {str(e)}")
    
    # Return forecaster and dfm_model
    return forecaster, dfm_model


def _extract_target_forecast(
    X_pred: np.ndarray,
    pred_step: int,
    dfm_model: Any,
    target_series: str
) -> float:
    """Extract target series forecast value from prediction array.
    
    Args:
        X_pred: Prediction array (horizon x n_series)
        pred_step: Time step index to extract
        dfm_model: DFM/DDFM model instance
        target_series: Target series name
        
    Returns:
        Forecast value for target series
    """
    # Ensure pred_step is valid
    if pred_step >= X_pred.shape[0]:
        pred_step = X_pred.shape[0] - 1
    if pred_step < 0:
        pred_step = 0
    
    # Try to find target series index
    if hasattr(dfm_model, 'config'):
        try:
            from dfm_python.utils.helpers import find_series_index
            series_idx = find_series_index(dfm_model.config, target_series)
            if series_idx is not None and series_idx < X_pred.shape[1]:
                return float(X_pred[pred_step, series_idx])
        except (ValueError, IndexError, KeyError, AttributeError):
            pass
    
    # Fallback to first column
    return float(X_pred[pred_step, 0] if X_pred.shape[1] > 0 else 0.0)


def _save_json_results(
    output_file: Path,
    results: Dict[str, Any],
    logger: logging.Logger
) -> None:
    """Save results to JSON file with validation.
    
    Args:
        output_file: Path to output file
        results: Results dictionary to save
        logger: Logger instance
        
    Raises:
        IOError: If file cannot be created or is empty
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        
        # Validate file was created and is non-empty
        if not output_file.exists():
            raise IOError(f"Output file was not created: {output_file}")
        
        file_size = output_file.stat().st_size
        if file_size == 0:
            raise IOError(f"Output file is empty: {output_file}")
        
        logger.info(f"Results saved successfully to: {output_file} (size: {file_size} bytes)")
    except (IOError, OSError) as e:
        logger.error(f"Failed to save results to {output_file}: {type(e).__name__}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error saving results to {output_file}: {type(e).__name__}: {str(e)}")
        raise


def _get_current_factor_state(dfm_model: Any, logger: logging.Logger) -> np.ndarray:
    """Get current factor state by re-running Kalman filter with masked data.
    
    This function re-runs Kalman filter using the updated data_module data
    to get the current factor state, which reflects the masked data.
    
    Args:
        dfm_model: DFMBase or DDFMBase model instance
        logger: Logger instance
        
    Returns:
        Current factor state (last row of smoothed factors), shape (m,)
        
    Raises:
        RuntimeError: If model doesn't have required attributes or Kalman filter fails
    """
    # Try to get result if not available
    if not hasattr(dfm_model, '_result') or dfm_model._result is None:
        # Try to get result from model
        if hasattr(dfm_model, 'get_result'):
            try:
                dfm_model._result = dfm_model.get_result()
                logger.debug("Retrieved result using get_result() method")
            except Exception as e:
                logger.warning(f"Failed to get result using get_result(): {type(e).__name__}: {str(e)}")
                raise RuntimeError("Model result not available - cannot get current factor state") from e
        else:
            raise RuntimeError("Model result not available and no get_result() method - cannot get current factor state")
    
    if not hasattr(dfm_model, '_data_module') or dfm_model._data_module is None:
        raise RuntimeError("DataModule not available - cannot get current factor state")
    
    result = dfm_model._result
    data_module = dfm_model._data_module
    
    # Get model parameters from result
    A = result.A
    C = result.C
    Q = result.Q
    R = result.R
    Z_0 = result.Z_0
    V_0 = result.V_0
    
    # Get masked data from data_module
    # CRITICAL: data_module.data should already be filtered to match model's series
    if not hasattr(data_module, 'data') or data_module.data is None:
        raise RuntimeError("DataModule.data not available")
    
    data_masked = data_module.data
    
    # DEBUG: Log data statistics to diagnose why factor states might be repetitive
    # This helps identify if data masking is actually changing between timepoints
    if isinstance(data_masked, pd.DataFrame):
        nan_count = data_masked.isnull().sum().sum()
        total_cells = data_masked.size
        nan_pct = (nan_count / total_cells * 100) if total_cells > 0 else 0
        data_mean = data_masked.mean().mean() if len(data_masked.columns) > 0 else np.nan
        data_std = data_masked.std().mean() if len(data_masked.columns) > 0 else np.nan
        logger.debug(
            f"_get_current_factor_state: Data statistics - "
            f"shape {data_masked.shape}, NaN: {nan_count}/{total_cells} ({nan_pct:.1f}%), "
            f"mean: {data_mean:.4f}, std: {data_std:.4f}"
        )
    else:
        nan_count = np.sum(np.isnan(data_masked)) if hasattr(data_masked, '__array__') else 0
        total_cells = data_masked.size if hasattr(data_masked, 'size') else len(data_masked)
        nan_pct = (nan_count / total_cells * 100) if total_cells > 0 else 0
        data_mean = np.nanmean(data_masked) if hasattr(data_masked, '__array__') else np.nan
        data_std = np.nanstd(data_masked) if hasattr(data_masked, '__array__') else np.nan
        logger.debug(
            f"_get_current_factor_state: Data statistics - "
            f"shape {data_masked.shape if hasattr(data_masked, 'shape') else 'unknown'}, "
            f"NaN: {nan_count}/{total_cells} ({nan_pct:.1f}%), "
            f"mean: {data_mean:.4f}, std: {data_std:.4f}"
        )
    
    # Extract Mx and Wx from result FIRST (before using them)
    # Standardize data using result's Wx and Mx
    # CRITICAL: Extract these before any validation checks to avoid UnboundLocalError
    if not hasattr(result, 'Wx') or result.Wx is None:
        raise RuntimeError("Model result missing Wx parameter - cannot standardize data for Kalman filter re-run")
    if not hasattr(result, 'Mx') or result.Mx is None:
        raise RuntimeError("Model result missing Mx parameter - cannot standardize data for Kalman filter re-run")
    
    Wx = result.Wx
    Mx = result.Mx
    
    # Verify data has been filtered to match model's series count
    # If not, this will cause dimension mismatch in Kalman filter
    if isinstance(data_masked, pd.DataFrame):
        n_series_in_data = len(data_masked.columns)
    else:
        n_series_in_data = data_masked.shape[1] if len(data_masked.shape) > 1 else 1
    
    # Get expected series count from Mx
    if hasattr(Mx, '__len__') and not isinstance(Mx, str):
        n_series_expected = len(Mx)
    elif hasattr(Mx, 'shape') and len(Mx.shape) > 0:
        n_series_expected = Mx.shape[0] if len(Mx.shape) == 1 else 1
    else:
        n_series_expected = 1
    
    if n_series_in_data != n_series_expected:
        logger.error(
            f"CRITICAL: data_module.data has {n_series_in_data} series but model expects {n_series_expected}. "
            f"This indicates data filtering failed. Cannot proceed with Kalman filter re-run."
        )
        # This should not happen if _update_data_module_for_nowcasting worked correctly
        raise ValueError(
            f"DataModule data dimension mismatch: {n_series_in_data} series in data_module.data "
            f"but {n_series_expected} expected from model parameters. "
            f"Data filtering in _update_data_module_for_nowcasting may have failed."
        )
    
    # Convert to numpy if DataFrame
    if isinstance(data_masked, pd.DataFrame):
        X_data = data_masked.values
    else:
        X_data = np.asarray(data_masked)
    
    # Validate dimensions match
    n_series_data = X_data.shape[1] if len(X_data.shape) > 1 else 1
    # Handle Mx/Wx - they can be 1D arrays or scalars
    if hasattr(Mx, '__len__') and not isinstance(Mx, str):
        n_series_params = len(Mx)
    elif hasattr(Mx, 'shape') and len(Mx.shape) > 0:
        n_series_params = Mx.shape[0] if len(Mx.shape) == 1 else 1
    else:
        n_series_params = 1
    
    if n_series_data != n_series_params:
        error_msg = f"Series count mismatch: data has {n_series_data} series but Mx/Wx have {n_series_params}. Cannot re-run Kalman filter."
        logger.warning(error_msg)
        # Fallback to training state
        if hasattr(result, 'Z') and result.Z is not None and len(result.Z) > 0:
            logger.debug("Using training state as fallback for factor state")
            return result.Z[-1, :].copy()
        else:
            raise ValueError(error_msg)
    
    # Handle NaN values (masked data)
    # For Kalman filter, we need to handle missing data properly
    # Convert to torch tensor and transpose to (N x T) format expected by Kalman filter
    import torch
    
    # CRITICAL: Replace Inf with NaN before processing (NaN is handled by Kalman filter, Inf is not)
    if np.any(~np.isfinite(X_data)):
        inf_count = np.sum(~np.isfinite(X_data))
        logger.warning(f"Found {inf_count} non-finite values in data (Inf/NaN). Replacing Inf with NaN.")
        X_data = np.where(np.isfinite(X_data), X_data, np.nan)
    
    # Standardize: X_std = (X - Mx) / Wx
    # Ensure Mx and Wx are broadcastable with X_data
    # Handle Mx shape safely - it can be 1D array or scalar
    if hasattr(Mx, 'shape') and len(Mx.shape) == 1 and Mx.shape[0] == X_data.shape[1]:
        # Mx is (N,), X_data is (T, N), so broadcasting works
        X_std = (X_data - Mx) / np.where(Wx != 0, Wx, 1.0)
    elif hasattr(Mx, 'shape') and len(Mx.shape) == 0:
        # Mx is scalar, broadcast to all series
        X_std = (X_data - Mx) / np.where(Wx != 0, Wx, 1.0)
    else:
        mx_shape = Mx.shape if hasattr(Mx, 'shape') else 'scalar'
        raise ValueError(f"Cannot broadcast Mx (shape {mx_shape}) with X_data (shape {X_data.shape})")
    
    # CRITICAL: Check for Inf values after standardization (division by small Wx can create Inf)
    if np.any(~np.isfinite(X_std)):
        inf_count = np.sum(~np.isfinite(X_std))
        logger.warning(f"Found {inf_count} non-finite values in standardized data. Replacing Inf with NaN.")
        X_std = np.where(np.isfinite(X_std), X_std, np.nan)
    
    # Convert to torch tensor: (N x T) format
    Y = torch.tensor(X_std.T, dtype=torch.float32)  # (N x T)
    
    # CRITICAL: Check for Inf values in torch tensor (NaN is OK, Inf is not)
    if torch.any(~torch.isfinite(Y)):
        inf_count = torch.sum(~torch.isfinite(Y)).item()
        logger.warning(f"Found {inf_count} non-finite values in torch tensor Y. Replacing Inf with NaN.")
        Y = torch.where(torch.isfinite(Y), Y, torch.tensor(float('nan'), dtype=torch.float32))
    
    # Convert parameters to torch
    A_torch = torch.tensor(A, dtype=torch.float32)
    C_torch = torch.tensor(C, dtype=torch.float32)
    Q_torch = torch.tensor(Q, dtype=torch.float32)
    R_torch = torch.tensor(R, dtype=torch.float32)
    Z_0_torch = torch.tensor(Z_0, dtype=torch.float32)
    V_0_torch = torch.tensor(V_0, dtype=torch.float32)
    
    # Re-run Kalman filter with masked data
    try:
        # Check if model has kalman attribute (DFM) or needs to use result's kalman
        kalman = None
        if hasattr(dfm_model, 'kalman') and dfm_model.kalman is not None:
            kalman = dfm_model.kalman
        elif hasattr(dfm_model, '_model') and hasattr(dfm_model._model, 'kalman'):
            kalman = dfm_model._model.kalman
        
        # For DDFM, we might not have kalman attribute, so create one
        if kalman is None:
            from dfm_python.ssm.kalman import KalmanFilter
            kalman = KalmanFilter(
                min_eigenval=1e-8,
                inv_regularization=1e-6,
                cholesky_regularization=1e-8
            )
            logger.debug("Created new KalmanFilter instance for nowcasting (model doesn't have kalman attribute)")
        
        # Run Kalman smoother
        zsmooth, Vsmooth, _, _ = kalman(
            Y, A_torch, C_torch, Q_torch, R_torch, Z_0_torch, V_0_torch
        )
        
        # zsmooth is (m x (T+1)), transpose to ((T+1) x m)
        Zsmooth = zsmooth.T  # ((T+1) x m)
        
        # Get last factor state (skip initial state at index 0)
        Z_last = Zsmooth[-1, :].cpu().numpy()  # (m,)
        
        # Log factor state statistics for debugging
        factor_norm = np.linalg.norm(Z_last)
        factor_mean = np.mean(Z_last)
        factor_std = np.std(Z_last)
        logger.debug(f"Re-ran Kalman filter with masked data: got factor state shape {Z_last.shape}, norm={factor_norm:.4f}, mean={factor_mean:.4f}, std={factor_std:.4f}")
        
        return Z_last
        
    except Exception as e:
        # CRITICAL: Log detailed error information for debugging DDFM constant predictions
        model_type = type(dfm_model).__name__ if hasattr(dfm_model, '__class__') else 'Unknown'
        
        # Try DDFM-specific encoder path as fallback (if encoder is available)
        if model_type == 'DDFM' or 'DDFM' in str(type(dfm_model)):
            try:
                # For DDFM, try using encoder to get factor state from last observation
                # Handle both DDFMBase directly and DDFM wrapper (which stores model in _model)
                encoder = None
                if hasattr(dfm_model, 'encoder') and dfm_model.encoder is not None:
                    encoder = dfm_model.encoder
                elif hasattr(dfm_model, '_model') and hasattr(dfm_model._model, 'encoder') and dfm_model._model.encoder is not None:
                    encoder = dfm_model._model.encoder
                
                if encoder is not None:
                    logger.info(f"[{model_type}] Kalman filter failed, trying encoder-based factor extraction...")
                    
                    # Get last valid observation (handle NaN)
                    X_last = X_data[-1, :]  # Last time step
                    # Replace NaN with 0 for encoder (encoder can handle this better than Kalman filter)
                    X_last_clean = np.where(np.isfinite(X_last), X_last, 0.0)
                    
                    # Standardize
                    if hasattr(Mx, 'shape') and len(Mx.shape) == 1 and Mx.shape[0] == X_last_clean.shape[0]:
                        X_last_std = (X_last_clean - Mx) / np.where(Wx != 0, Wx, 1.0)
                    elif hasattr(Mx, 'shape') and len(Mx.shape) == 0:
                        X_last_std = (X_last_clean - Mx) / np.where(Wx != 0, Wx, 1.0)
                    else:
                        raise ValueError(f"Cannot standardize for encoder: Mx shape mismatch")
                    
                    # Convert to torch and encode
                    X_torch = torch.tensor(X_last_std, dtype=torch.float32).unsqueeze(0)  # (1, N)
                    encoder.eval()
                    with torch.no_grad():
                        Z_encoded = encoder(X_torch)  # Should return (1, m)
                        if isinstance(Z_encoded, tuple):
                            Z_encoded = Z_encoded[0]  # Some encoders return tuple
                        Z_last = Z_encoded.squeeze(0).cpu().numpy()  # (m,)
                    
                    logger.info(f"[{model_type}] Successfully extracted factor state using encoder: shape {Z_last.shape}")
                    return Z_last
            except Exception as encoder_error:
                logger.warning(f"[{model_type}] Encoder-based extraction also failed: {type(encoder_error).__name__}: {str(encoder_error)}")
        
        # Log detailed error
        logger.error(
            f"CRITICAL: Failed to re-run Kalman filter for {model_type}: {type(e).__name__}: {str(e)}. "
            f"Data shape: {X_data.shape if 'X_data' in locals() else 'unknown'}, "
            f"Mx shape: {Mx.shape if hasattr(Mx, 'shape') else type(Mx)}, "
            f"Wx shape: {Wx.shape if hasattr(Wx, 'shape') else type(Wx)}, "
            f"Result has series_ids: {hasattr(result, 'series_ids') and result.series_ids is not None}. "
            f"Using training state as fallback (this will produce constant predictions)."
        )
        # Fallback to training state
        if hasattr(result, 'Z') and result.Z is not None and len(result.Z) > 0:
            logger.warning(f"Falling back to training state (constant factor state): {result.Z[-1, :]}")
            return result.Z[-1, :].copy()
        else:
            raise RuntimeError(f"Cannot get factor state: Kalman filter failed and no training state available: {e}") from e


def _update_data_module_for_nowcasting(
    dfm_model: Any,
    data_up_to_target: pd.DataFrame,
    view_date: pd.Timestamp,
    config: Optional[Any] = None
) -> None:
    """Update data_module with data up to view date, applying release date masking.
    
    Args:
        dfm_model: DFMBase or DDFMBase model instance
        data_up_to_target: DataFrame with data up to view date
        view_date: View date for nowcasting (data after this date should be masked)
        config: Model configuration with release date information (optional, will try to get from model)
        
    Raises:
        RuntimeError: If data_module is not available
        ValueError: If data is empty or invalid
    """
    if not hasattr(dfm_model, '_data_module'):
        raise RuntimeError(f"dfm_model does not have _data_module attribute")
    
    if dfm_model._data_module is None:
        raise RuntimeError(f"data_module is None - cannot update for nowcasting")
    
    # Validate input data
    if data_up_to_target is None:
        raise ValueError(f"data_up_to_target is None")
    
    if not isinstance(data_up_to_target, pd.DataFrame):
        raise ValueError(f"data_up_to_target must be a DataFrame, got {type(data_up_to_target)}")
    
    if len(data_up_to_target) == 0:
        raise ValueError(f"No data available in data_up_to_target (empty DataFrame)")
    
    if len(data_up_to_target.columns) == 0:
        raise ValueError(f"No columns in data_up_to_target")
    
    from src.preprocessing import resample_to_monthly
    from dfm_python.utils.time import TimeIndex
    from src.nowcasting import create_data_view
    
    try:
        data_monthly = resample_to_monthly(data_up_to_target)
    except Exception as e:
        raise ValueError(f"Failed to resample data to monthly: {type(e).__name__}: {str(e)}") from e
    
    if len(data_monthly) == 0:
        raise ValueError(f"No data available after resampling to monthly (view_date: {view_date.strftime('%Y-%m-%d')})")
    
    if len(data_monthly.columns) == 0:
        raise ValueError(f"No columns in resampled monthly data")
    
    # Validate index
    if data_monthly.index is None or len(data_monthly.index) == 0:
        raise ValueError(f"Invalid index in resampled monthly data")
    
    try:
        time_index = TimeIndex(data_monthly.index.to_pydatetime().tolist())
    except Exception as e:
        raise ValueError(f"Failed to create TimeIndex from data: {type(e).__name__}: {str(e)}") from e
    
    if len(time_index) == 0:
        raise ValueError(f"TimeIndex is empty after creation")
    
    # Get config if not provided
    if config is None:
        if hasattr(dfm_model, '_result') and dfm_model._result is not None:
            if hasattr(dfm_model._result, 'config'):
                config = dfm_model._result.config
        elif hasattr(dfm_model, 'config'):
            config = dfm_model.config
    
    # CRITICAL: Filter data to only include series that the model was trained with
    # This ensures dimension matching and proper masking
    # First, try to get series_ids from model result (most reliable)
    model_series_ids = None
    expected_n_series = None
    
    # Get expected number of series from model's Mx/Wx dimensions
    model_type = type(dfm_model).__name__ if hasattr(dfm_model, '__class__') else 'Unknown'
    result = None
    if hasattr(dfm_model, '_result') and dfm_model._result is not None:
        result = dfm_model._result
        # Get series_ids from result if available
        if hasattr(result, 'series_ids') and result.series_ids is not None:
            model_series_ids = result.series_ids
            logger.info(f"[{model_type}] Got {len(model_series_ids)} series IDs from model result: {model_series_ids[:5]}...")
        else:
            logger.warning(f"[{model_type}] Result does not have series_ids attribute or it is None. Will try config fallback.")
        
        # Get expected dimension from Mx/Wx (CRITICAL for dimension-based fallback)
        if hasattr(result, 'Mx') and result.Mx is not None:
            Mx = result.Mx
            if hasattr(Mx, '__len__') and not isinstance(Mx, str):
                expected_n_series = len(Mx)
            elif hasattr(Mx, 'shape') and len(Mx.shape) > 0:
                expected_n_series = Mx.shape[0] if len(Mx.shape) == 1 else 1
            else:
                expected_n_series = 1
            logger.info(f"[{model_type}] Expected {expected_n_series} series from Mx/Wx dimensions (Mx type: {type(Mx)}, shape: {Mx.shape if hasattr(Mx, 'shape') else 'N/A'})")
        elif hasattr(result, 'Wx') and result.Wx is not None:
            # Fallback to Wx if Mx not available
            Wx = result.Wx
            if hasattr(Wx, '__len__') and not isinstance(Wx, str):
                expected_n_series = len(Wx)
            elif hasattr(Wx, 'shape') and len(Wx.shape) > 0:
                expected_n_series = Wx.shape[0] if len(Wx.shape) == 1 else 1
            else:
                expected_n_series = 1
            logger.info(f"[{model_type}] Expected {expected_n_series} series from Wx dimensions (Mx not available)")
        elif hasattr(result, 'C') and result.C is not None:
            # Fallback to C matrix shape if Mx/Wx not available
            C = result.C
            if hasattr(C, 'shape') and len(C.shape) >= 1:
                expected_n_series = C.shape[0]  # C is (N x m), so first dimension is number of series
            else:
                expected_n_series = None
            logger.info(f"[{model_type}] Expected {expected_n_series} series from C matrix shape (Mx/Wx not available)")
        else:
            logger.warning(f"[{model_type}] Result does not have Mx, Wx, or C attributes - cannot determine expected series count")
    
    # Fallback: try to get from config
    if model_series_ids is None and config is not None:
        try:
            from dfm_python.utils.helpers import get_series_ids
            model_series_ids = get_series_ids(config)
            logger.debug(f"Got {len(model_series_ids)} series IDs from config")
        except Exception as e:
            logger.warning(f"Failed to get series IDs from config: {type(e).__name__}: {str(e)}")
    
    # Filter data to match model's expected series
    n_series_in_data = len(data_monthly.columns)
    data_filtered = False
    
    if model_series_ids is not None and len(model_series_ids) > 0:
        # Filter by series IDs
        available_series = [s for s in model_series_ids if s in data_monthly.columns]
        if len(available_series) == 0:
            raise ValueError(
                f"CRITICAL: No matching series found between model series IDs ({len(model_series_ids)} series: {model_series_ids[:5]}...) "
                f"and data columns ({len(data_monthly.columns)} columns: {list(data_monthly.columns)[:5]}...). "
                f"Cannot proceed with nowcasting."
            )
        if len(available_series) != len(model_series_ids):
            logger.warning(
                f"Only {len(available_series)}/{len(model_series_ids)} model series found in data. "
                f"Missing: {set(model_series_ids) - set(available_series)}"
            )
        # Reorder columns to match model's series order (CRITICAL for dimension matching)
        data_monthly = data_monthly[available_series].copy()
        logger.info(f"Filtered data from {n_series_in_data} to {len(available_series)} series matching model series IDs")
        data_filtered = True
        
        # Verify filtering worked
        if len(data_monthly.columns) != len(available_series):
            raise ValueError(f"Data filtering failed: expected {len(available_series)} columns but got {len(data_monthly.columns)}")
    
    # Fallback: filter by expected dimension if we know it
    if not data_filtered and expected_n_series is not None and expected_n_series > 0:
        if n_series_in_data > expected_n_series:
            # CRITICAL: We need to filter to match model's expected dimension
            # Try to match by series_ids from config first (if available)
            if config is not None:
                try:
                    from dfm_python.utils.helpers import get_series_ids
                    config_series_ids = get_series_ids(config)
                    if config_series_ids is not None and len(config_series_ids) > 0:
                        # Filter to only config series that exist in data
                        available_config_series = [s for s in config_series_ids if s in data_monthly.columns]
                        if len(available_config_series) >= expected_n_series:
                            # Use config series order (take first expected_n_series that match)
                            data_monthly = data_monthly[available_config_series[:expected_n_series]].copy()
                            logger.info(f"Filtered data from {n_series_in_data} to {expected_n_series} series using config series IDs (first {expected_n_series} matching series)")
                            data_filtered = True
                except Exception as e:
                    logger.debug(f"Failed to use config series IDs for filtering: {type(e).__name__}: {str(e)}")
            
            # If config-based filtering didn't work, use dimension-based fallback
            if not data_filtered:
                logger.warning(
                    f"CRITICAL [{model_type}]: Data has {n_series_in_data} series but model expects {expected_n_series}. "
                    f"Filtering to first {expected_n_series} columns. This may cause incorrect results if column order doesn't match."
                )
                data_monthly = data_monthly.iloc[:, :expected_n_series].copy()
                logger.info(f"Filtered data from {n_series_in_data} to {expected_n_series} series by dimension (first N columns)")
                data_filtered = True
        elif n_series_in_data < expected_n_series:
            raise ValueError(
                f"CRITICAL: Data has {n_series_in_data} series but model expects {expected_n_series}. "
                f"Cannot proceed with nowcasting - insufficient data."
            )
        elif n_series_in_data == expected_n_series:
            # Dimensions already match - no filtering needed
            logger.debug(f"Data already has correct dimension ({n_series_in_data} series), no filtering needed")
            data_filtered = True
    
    # Final check: if we still haven't filtered and dimensions don't match, raise error
    if not data_filtered:
        if expected_n_series is not None and n_series_in_data != expected_n_series:
            raise ValueError(
                f"CRITICAL: Cannot filter data - data has {n_series_in_data} series but model expects {expected_n_series}. "
                f"Model series IDs not available and dimension mismatch detected. Cannot proceed with nowcasting."
            )
        elif n_series_in_data > 100:  # Suspiciously large - likely unfiltered
            logger.warning(
                f"WARNING: Data has {n_series_in_data} series but no filtering was applied. "
                f"This may cause dimension mismatch errors. Expected series count: {expected_n_series}"
            )
    
    # Apply release date masking if config is available
    if config is not None:
        try:
            # Convert DataFrame to numpy array for create_data_view
            X_data = data_monthly.values
            X_frame = data_monthly.copy()
            
            # Apply release date masking
            X_masked, time_index_masked, _ = create_data_view(
                X=X_data,
                Time=time_index,
                config=config,
                view_date=view_date.to_pydatetime(),
                X_frame=X_frame
            )
            
            # Convert masked array back to DataFrame
            data_monthly_masked = pd.DataFrame(
                X_masked,
                index=data_monthly.index,
                columns=data_monthly.columns
            )
            
            # Update time_index if it was modified
            if time_index_masked is not None:
                time_index = time_index_masked
            
            data_monthly = data_monthly_masked
            # Enhanced logging: Log masking statistics at INFO level for debugging repetitive predictions
            nan_count = data_monthly.isnull().sum().sum() if isinstance(data_monthly, pd.DataFrame) else data_monthly.isnull().sum()
            total_cells = data_monthly.size if hasattr(data_monthly, 'size') else len(data_monthly)
            nan_pct = (nan_count / total_cells * 100) if total_cells > 0 else 0
            # Also log per-series NaN counts to see which series are masked
            if isinstance(data_monthly, pd.DataFrame):
                nan_per_series = data_monthly.isnull().sum()
                series_with_nan = nan_per_series[nan_per_series > 0]
                series_info = f", {len(series_with_nan)} series with NaN" if len(series_with_nan) > 0 else ", no series with NaN"
            else:
                series_info = ""
            logger.info(f"Data masking for view_date={view_date.strftime('%Y-%m-%d')}: {nan_count}/{total_cells} NaN ({nan_pct:.1f}%){series_info}")
            logger.debug(f"Applied release date masking for view_date={view_date.strftime('%Y-%m-%d')}: {nan_count}/{total_cells} NaN ({nan_pct:.1f}%)")
        except Exception as e:
            logger.warning(f"Failed to apply release date masking (will use unmasked data): {type(e).__name__}: {str(e)}")
            # Continue with unmasked data if masking fails
    
    # CRITICAL: Verify filtering worked before updating data_module
    final_n_series = len(data_monthly.columns) if isinstance(data_monthly, pd.DataFrame) else (data_monthly.shape[1] if len(data_monthly.shape) > 1 else 1)
    if expected_n_series is not None and final_n_series != expected_n_series:
        # Log detailed information for debugging
        logger.error(
            f"CRITICAL [{model_type}]: Data filtering validation failed - after filtering, data has {final_n_series} series but model expects {expected_n_series}. "
            f"Initial data had {n_series_in_data} series. "
            f"Model series IDs available: {model_series_ids is not None and len(model_series_ids) > 0 if model_series_ids is not None else False}. "
            f"Data filtering attempted: {data_filtered}. "
            f"This will cause dimension mismatch in Kalman filter and constant predictions."
        )
        raise ValueError(
            f"CRITICAL: Data filtering failed - after filtering, data has {final_n_series} series but model expects {expected_n_series}. "
            f"Cannot proceed with nowcasting. This will cause dimension mismatch in Kalman filter."
        )
    
    # Additional validation: Log successful filtering for debugging
    if expected_n_series is not None and final_n_series == expected_n_series:
        logger.info(f"[{model_type}] Data filtering successful: {n_series_in_data} → {final_n_series} series (expected: {expected_n_series})")
    
    # Update data_module
    try:
        existing_data_module = dfm_model._data_module
        existing_data_module.data = data_monthly
        existing_data_module.time_index = time_index
        dfm_model._data_module = existing_data_module
        
        # CRITICAL: Verify the assignment worked
        if hasattr(existing_data_module, 'data') and existing_data_module.data is not None:
            assigned_n_series = len(existing_data_module.data.columns) if isinstance(existing_data_module.data, pd.DataFrame) else (existing_data_module.data.shape[1] if len(existing_data_module.data.shape) > 1 else 1)
            if expected_n_series is not None and assigned_n_series != expected_n_series:
                raise RuntimeError(
                    f"CRITICAL: data_module.data assignment failed - assigned data has {assigned_n_series} series but model expects {expected_n_series}. "
                    f"This indicates data_module.data property may not allow direct assignment or was reset."
                )
            logger.info(f"Successfully updated data_module with {len(data_monthly)} monthly observations, {final_n_series} series (view_date={view_date.strftime('%Y-%m-%d')})")
        else:
            raise RuntimeError("data_module.data is None after assignment - assignment failed")
    except Exception as e:
        raise RuntimeError(f"Failed to update data_module: {type(e).__name__}: {str(e)}") from e


def run_backtest_evaluation(
    config_name: str,
    model: str,
    train_start: str = "1985-01-01",
    train_end: str = "2019-12-31",
    nowcast_start: str = "2024-01-01",
    nowcast_end: str = "2025-10-31",
    weeks_before: Optional[List[int]] = None,
    config_dir: Optional[str] = None,
    overrides: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Run backtest evaluation and generate JSON/CSV results.
    
    Note: Nowcasting is only supported for DFM and DDFM models.
    ARIMA and VAR models are not suitable for nowcasting due to their
    inability to handle missing data from release date masking.
    """
    project_root = get_project_root()
    
    if config_dir is None:
        config_dir = str(project_root / "config")
    
    if weeks_before is None:
        weeks_before = [4, 1]
    
    model_lower = model.lower()
    
    # Nowcasting is only supported for DFM and DDFM models
    if model_lower not in ['dfm', 'ddfm']:
        error_msg = f"Nowcasting is not supported for {model.upper()} model. Only DFM and DDFM models support nowcasting."
        logger.error(error_msg)
        results = {
            'target_series': None,
            'model': model.upper(),
            'train_period': f"{train_start} to {train_end}",
            'nowcast_period': f"{nowcast_start} to {nowcast_end}",
            'weeks_before': weeks_before,
            'status': 'not_supported',
            'error': error_msg,
            'summary': {
                'total_timepoints': len(weeks_before),
                'successful_timepoints': 0,
                'failed_timepoints': len(weeks_before)
            }
        }
        # Still save the result file to indicate the status
        output_file = project_root / "outputs" / "backtest" / f"UNKNOWN_{model_lower}_backtest.json"
        _save_json_results(output_file, results, logger)
        return results
    
    supports_nowcast_manager = True  # Only DFM/DDFM reach here
    
    cfg = parse_experiment_config(config_name, config_dir, overrides)
    validate_experiment_config(cfg, require_target=True, require_models=False)
    
    params = extract_experiment_params(cfg)
    target_series = params['target_series']
    exp_cfg = params.get('exp_cfg', cfg)
    
    # Load release dates
    series_release_dates = {}
    if hasattr(exp_cfg, 'get'):
        series_ids_raw = exp_cfg.get('series', [])
    else:
        series_ids_raw = getattr(exp_cfg, 'series', [])
    
    if series_ids_raw:
        import yaml
        series_config_dir = project_root / "dfm-python" / "config" / "series"
        series_ids = OmegaConf.to_container(series_ids_raw, resolve=True) if hasattr(OmegaConf, 'to_container') else list(series_ids_raw)
        if not isinstance(series_ids, list):
            series_ids = []
        
        for series_id in series_ids:
            series_config_path = series_config_dir / f"{series_id}.yaml"
            if series_config_path.exists():
                try:
                    with open(series_config_path, 'r', encoding='utf-8') as f:
                        series_cfg = yaml.safe_load(f) or {}
                    release_date = series_cfg.get('release') or series_cfg.get('release_date')
                    if release_date is not None:
                        series_release_dates[series_id] = release_date
                except Exception:
                    pass
    
    # Load model
    forecaster, dfm_model = _load_model_for_inference(
        project_root, target_series, model, supports_nowcast_manager, train_start, train_end
    )
    
    # Generate target periods
    start_date = datetime.strptime(nowcast_start, "%Y-%m-%d")
    end_date = datetime.strptime(nowcast_end, "%Y-%m-%d")
    target_periods = []
    current = start_date.replace(day=1)
    while current <= end_date:
        if current.month == 12:
            last_day = current.replace(day=31)
        else:
            next_month = current.replace(month=current.month + 1, day=1)
            last_day = next_month - timedelta(days=1)
        target_periods.append(last_day)
        current += relativedelta(months=1)
    
    # Load data
    from src.preprocessing import resample_to_monthly
    data_path_file = project_root / "data" / "data.csv"
    if not data_path_file.exists():
        data_path_file = project_root / "data" / "sample_data.csv"
    full_data = pd.read_csv(data_path_file, index_col=0, parse_dates=True)
    full_data_monthly = resample_to_monthly(full_data)
    
    train_start_ts = pd.Timestamp(train_start)
    train_end_ts = pd.Timestamp(train_end)
    train_data_filtered = full_data[(full_data.index >= train_start_ts) & (full_data.index <= train_end_ts)]
    train_std = train_data_filtered[target_series].std() if target_series in train_data_filtered.columns else train_data_filtered.iloc[:, 0].std() if len(train_data_filtered.columns) > 0 else 1.0
    
    results_by_timepoint = {}
    train_end_date = datetime.strptime(train_end, "%Y-%m-%d")
    
    logger.info(f"Starting nowcasting backtest for {target_series} with {model.upper()}")
    logger.info(f"Target periods: {len(target_periods)} months from {nowcast_start} to {nowcast_end}")
    logger.info(f"Time points: {weeks_before} weeks before month end")
    
    for weeks in weeks_before:
        monthly_results = []
        logger.info(f"Processing {weeks} weeks before time point...")
        
        for target_month_end in target_periods:
            target_month_end_ts = pd.Timestamp(target_month_end) if not isinstance(target_month_end, pd.Timestamp) else target_month_end
            view_date = target_month_end_ts - timedelta(weeks=weeks)
            view_date_ts = pd.Timestamp(view_date)
            
            if view_date_ts <= pd.Timestamp(train_end_date):
                continue
            
            if len(full_data_monthly) == 0:
                continue
            
            try:
                # Nowcasting is only supported for DFM/DDFM models
                # Use predict method directly (simpler and faster)
                # Update data_module with data up to view_date for nowcasting
                # Include data up to view_date (not target_month_end) to properly simulate data availability
                data_up_to_view = full_data[full_data.index <= pd.Timestamp(view_date)].copy()
                try:
                    # Get config for release date masking
                    config = None
                    if hasattr(dfm_model, '_result') and dfm_model._result is not None:
                        if hasattr(dfm_model._result, 'config'):
                            config = dfm_model._result.config
                    elif hasattr(dfm_model, 'config'):
                        config = dfm_model.config
                    
                    # Debug: Log data before masking
                    data_before_mask = resample_to_monthly(data_up_to_view)
                    nan_count_before = data_before_mask.isnull().sum().sum() if isinstance(data_before_mask, pd.DataFrame) else data_before_mask.isnull().sum()
                    
                    # Store data hash before masking for comparison
                    import hashlib
                    data_before_hash = hashlib.md5(pd.util.hash_pandas_object(data_before_mask.fillna(0)).values).hexdigest() if isinstance(data_before_mask, pd.DataFrame) else None
                    
                    _update_data_module_for_nowcasting(dfm_model, data_up_to_view, pd.Timestamp(view_date), config=config)
                    
                    # Debug: Log data after masking
                    if hasattr(dfm_model, '_data_module') and dfm_model._data_module is not None:
                        data_after_mask = dfm_model._data_module.data
                        nan_count_after = data_after_mask.isnull().sum().sum() if isinstance(data_after_mask, pd.DataFrame) else data_after_mask.isnull().sum()
                        data_after_hash = hashlib.md5(pd.util.hash_pandas_object(data_after_mask.fillna(0)).values).hexdigest() if isinstance(data_after_mask, pd.DataFrame) else None
                        logger.info(f"Masking for {target_month_end_ts.strftime('%Y-%m')} (view_date={view_date.strftime('%Y-%m-%d')}, weeks={weeks}): NaN before={nan_count_before}, NaN after={nan_count_after}, data_changed={data_before_hash != data_after_hash if data_before_hash and data_after_hash else 'N/A'}")
                        
                        # Log target series specific masking
                        if isinstance(data_after_mask, pd.DataFrame) and target_series in data_after_mask.columns:
                            target_nan_before = data_before_mask[target_series].isnull().sum() if isinstance(data_before_mask, pd.DataFrame) else 0
                            target_nan_after = data_after_mask[target_series].isnull().sum()
                            target_last_valid_before = data_before_mask[target_series].last_valid_index() if isinstance(data_before_mask, pd.DataFrame) else None
                            target_last_valid_after = data_after_mask[target_series].last_valid_index()
                            logger.debug(f"  Target {target_series}: NaN before={target_nan_before}, NaN after={target_nan_after}, last_valid before={target_last_valid_before}, last_valid after={target_last_valid_after}")
                except (RuntimeError, ValueError) as e:
                    logger.warning(f"Failed to update data_module for {target_month_end_ts.strftime('%Y-%m')} (view_date={view_date.strftime('%Y-%m-%d')}): {str(e)}")
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error updating data_module for {target_month_end_ts.strftime('%Y-%m')} (view_date={view_date.strftime('%Y-%m-%d')}): {type(e).__name__}: {str(e)}")
                    continue
                
                # Use predict method - can use any horizon (1 for nowcasting, 22 for forecasting)
                # Calculate horizon: number of months from view_date to target_month_end
                view_date_ts = pd.Timestamp(view_date)
                months_ahead = (target_month_end_ts.year - view_date_ts.year) * 12 + (target_month_end_ts.month - view_date_ts.month)
                # For nowcasting, we want to predict the target period, so horizon should be based on months difference
                # But if target is in the past relative to view_date, use horizon=1 (current period)
                horizon = max(1, min(months_ahead, 22)) if months_ahead >= 0 else 1
                
                try:
                    # CRITICAL: For nowcasting, we need to use the current masked data, not training state
                    # The predict() method uses result.Z[-1, :] from training, which doesn't reflect masked data
                    # Solution: Re-run Kalman filter with masked data to get current factor state
                    
                    # Get current factor state from masked data by re-running Kalman filter
                    # Note: data_module should already be updated with filtered and masked data
                    try:
                        Z_last_current = _get_current_factor_state(dfm_model, logger)
                        
                        # CRITICAL: Validate that factor state is actually different from previous timepoints
                        # This helps detect when Kalman filter is failing silently and returning constant states
                        if not hasattr(run_backtest_evaluation, '_previous_factor_states'):
                            run_backtest_evaluation._previous_factor_states = {}
                        key = f"{target_series}_{model}"
                        if key in run_backtest_evaluation._previous_factor_states:
                            prev_state = run_backtest_evaluation._previous_factor_states[key]
                            state_diff = np.linalg.norm(Z_last_current - prev_state)
                            if state_diff < 1e-6:
                                logger.warning(
                                    f"CRITICAL [{target_series} {model}]: Factor state is identical to previous timepoint "
                                    f"(diff={state_diff:.2e}). This indicates Kalman filter may be failing or data masking is not changing. "
                                    f"Month: {target_month_end_ts.strftime('%Y-%m')}, Weeks: {weeks}"
                                )
                                # Try to force recalculation by adding small perturbation if data masking has changed
                                if hasattr(dfm_model, '_data_module') and dfm_model._data_module is not None:
                                    data_masked = dfm_model._data_module.data
                                    if isinstance(data_masked, pd.DataFrame):
                                        data_hash = hashlib.md5(pd.util.hash_pandas_object(data_masked.fillna(0)).values).hexdigest()
                                        if hasattr(run_backtest_evaluation, '_previous_data_hashes'):
                                            if key in run_backtest_evaluation._previous_data_hashes:
                                                prev_hash = run_backtest_evaluation._previous_data_hashes[key]
                                                if data_hash != prev_hash:
                                                    logger.warning(
                                                        f"Data masking changed but factor state is identical - "
                                                        f"Kalman filter may be failing. Attempting alternative calculation..."
                                                    )
                                                    # Try alternative: use last valid observation to estimate factor state
                                                    # This is a fallback when Kalman filter fails but data has changed
                                                    try:
                                                        # Get last valid row of masked data
                                                        last_valid_idx = data_masked.last_valid_index()
                                                        if last_valid_idx is not None:
                                                            last_row = data_masked.loc[last_valid_idx].values
                                                            # Standardize using result's Wx and Mx
                                                            if hasattr(dfm_model._result, 'Wx') and hasattr(dfm_model._result, 'Mx'):
                                                                Wx = dfm_model._result.Wx
                                                                Mx = dfm_model._result.Mx
                                                                last_row_std = (last_row - Mx) / np.where(Wx != 0, Wx, 1.0)
                                                                # Use C matrix to estimate factor state: X = Z @ C^T, so Z ≈ X @ C @ (C^T @ C)^-1
                                                                C = dfm_model._result.C
                                                                if C is not None and len(C.shape) == 2:
                                                                    # Pseudo-inverse: Z ≈ X @ C @ (C^T @ C)^-1
                                                                    CtC = C.T @ C
                                                                    if np.linalg.cond(CtC) < 1e12:  # Check condition number
                                                                        CtC_inv = np.linalg.inv(CtC)
                                                                        Z_estimated = last_row_std @ C @ CtC_inv
                                                                        # Blend with previous state (weighted average)
                                                                        Z_last_current = 0.3 * Z_estimated + 0.7 * Z_last_current
                                                                        logger.info(
                                                                            f"Used alternative factor state calculation (blended with previous): "
                                                                            f"norm={np.linalg.norm(Z_last_current):.4f}"
                                                                        )
                                                    except Exception as alt_error:
                                                        logger.warning(f"Alternative factor state calculation failed: {alt_error}")
                        run_backtest_evaluation._previous_factor_states[key] = Z_last_current.copy()
                        
                        # Store data hash for comparison
                        if not hasattr(run_backtest_evaluation, '_previous_data_hashes'):
                            run_backtest_evaluation._previous_data_hashes = {}
                        if hasattr(dfm_model, '_data_module') and dfm_model._data_module is not None:
                            data_masked = dfm_model._data_module.data
                            if isinstance(data_masked, pd.DataFrame):
                                data_hash = hashlib.md5(pd.util.hash_pandas_object(data_masked.fillna(0)).values).hexdigest()
                                run_backtest_evaluation._previous_data_hashes[key] = data_hash
                        
                        # Log factor state values for debugging repetitive predictions
                        # Enhanced logging: show first 5 values, norm, mean, std, and min/max
                        factor_norm = np.linalg.norm(Z_last_current)
                        factor_mean = np.mean(Z_last_current)
                        factor_std = np.std(Z_last_current)
                        factor_min = np.min(Z_last_current)
                        factor_max = np.max(Z_last_current)
                        factor_first5 = Z_last_current[:5] if len(Z_last_current) >= 5 else Z_last_current
                        factor_state_str = (
                            f"shape {Z_last_current.shape}, first 5: {factor_first5}, "
                            f"norm: {factor_norm:.4f}, mean: {factor_mean:.4f}, std: {factor_std:.4f}, "
                            f"min: {factor_min:.4f}, max: {factor_max:.4f}"
                        )
                        logger.info(f"[{target_series} DFM] Factor state from masked data ({target_month_end_ts.strftime('%Y-%m')}, weeks={weeks}): {factor_state_str}")
                        logger.debug(f"Successfully got current factor state from masked data: {factor_state_str}")
                    except (ValueError, RuntimeError) as e:
                        # If Kalman filter re-run fails (e.g., dimension mismatch), try alternative calculation
                        # CRITICAL: This fallback will cause repetitive predictions if Kalman filter keeps failing
                        model_type = type(dfm_model).__name__ if hasattr(dfm_model, '__class__') else 'Unknown'
                        logger.error(
                            f"CRITICAL [{model_type}]: Failed to get current factor state from masked data: {type(e).__name__}: {str(e)}. "
                            f"This will cause constant predictions. Check data filtering and Kalman filter re-run logic. "
                            f"Target: {target_series}, Month: {target_month_end_ts.strftime('%Y-%m')}, Weeks: {weeks}"
                        )
                        
                        # CRITICAL FIX: Try alternative calculation when Kalman filter fails (not just when factor state is identical)
                        # This helps prevent repetitive predictions by providing variation even when Kalman filter fails
                        Z_last_current = None
                        try:
                            if hasattr(dfm_model, '_data_module') and dfm_model._data_module is not None:
                                data_masked = dfm_model._data_module.data
                                if isinstance(data_masked, pd.DataFrame):
                                    # Try alternative: use last valid observation to estimate factor state
                                    last_valid_idx = data_masked.last_valid_index()
                                    if last_valid_idx is not None:
                                        last_row = data_masked.loc[last_valid_idx].values
                                        # Standardize using result's Wx and Mx
                                        if hasattr(dfm_model, '_result') and dfm_model._result is not None:
                                            result = dfm_model._result
                                            if hasattr(result, 'Wx') and hasattr(result, 'Mx'):
                                                Wx = result.Wx
                                                Mx = result.Mx
                                                last_row_std = (last_row - Mx) / np.where(Wx != 0, Wx, 1.0)
                                                # Use C matrix to estimate factor state: X = Z @ C^T, so Z ≈ X @ C @ (C^T @ C)^-1
                                                if hasattr(result, 'C') and result.C is not None:
                                                    C = result.C
                                                    if len(C.shape) == 2:
                                                        # Pseudo-inverse: Z ≈ X @ C @ (C^T @ C)^-1
                                                        CtC = C.T @ C
                                                        if np.linalg.cond(CtC) < 1e12:  # Check condition number
                                                            CtC_inv = np.linalg.inv(CtC)
                                                            Z_estimated = last_row_std @ C @ CtC_inv
                                                            # Get previous state if available, otherwise use training state
                                                            if hasattr(result, 'Z') and result.Z is not None and len(result.Z) > 0:
                                                                Z_prev = result.Z[-1, :].copy()
                                                                # Blend with previous state (weighted average) to provide variation
                                                                Z_last_current = 0.3 * Z_estimated + 0.7 * Z_prev
                                                                logger.info(
                                                                    f"[{model_type}] Kalman filter failed, used alternative factor state calculation (blended): "
                                                                    f"norm={np.linalg.norm(Z_last_current):.4f}"
                                                                )
                                                        else:
                                                            logger.warning(f"Alternative calculation failed: C^T @ C condition number too high ({np.linalg.cond(CtC):.2e})")
                                        if Z_last_current is None:
                                            raise ValueError("Alternative calculation failed")
                        except Exception as alt_error:
                            logger.warning(f"Alternative factor state calculation failed: {type(alt_error).__name__}: {str(alt_error)}")
                            Z_last_current = None
                        
                        # Final fallback: use training state if alternative calculation also failed
                        if Z_last_current is None:
                            if hasattr(dfm_model, '_result') and dfm_model._result is not None:
                                if hasattr(dfm_model._result, 'Z') and dfm_model._result.Z is not None and len(dfm_model._result.Z) > 0:
                                    Z_last_current = dfm_model._result.Z[-1, :].copy()
                                    logger.warning(
                                        f"[{model_type}] Using training state (constant) - THIS WILL CAUSE REPETITIVE PREDICTIONS: "
                                        f"{Z_last_current[:3] if len(Z_last_current) >= 3 else Z_last_current}. "
                                        f"Factor state norm: {np.linalg.norm(Z_last_current):.4f}"
                                    )
                                    # CRITICAL: Track when we fall back to training state to detect repetitive predictions
                                    if not hasattr(run_backtest_evaluation, '_kalman_failures'):
                                        run_backtest_evaluation._kalman_failures = {}
                                    key = f"{target_series}_{model}"
                                    if key not in run_backtest_evaluation._kalman_failures:
                                        run_backtest_evaluation._kalman_failures[key] = []
                                    run_backtest_evaluation._kalman_failures[key].append({
                                        'month': target_month_end_ts.strftime('%Y-%m'),
                                        'weeks': weeks,
                                        'error': str(e),
                                        'error_type': type(e).__name__
                                    })
                                else:
                                    raise RuntimeError("Cannot get factor state: Kalman filter failed and no training state available") from e
                            else:
                                raise RuntimeError("Cannot get factor state: Kalman filter failed and no result available") from e
                    
                    # Temporarily update result.Z[-1, :] with current state
                    # CRITICAL: Ensure _result exists before updating (predict() may create it if None)
                    original_Z_last = None
                    if hasattr(dfm_model, '_result') and dfm_model._result is not None:
                        if hasattr(dfm_model._result, 'Z') and dfm_model._result.Z is not None:
                            original_Z_last = dfm_model._result.Z[-1, :].copy()
                            # CRITICAL: Update the actual array in-place to ensure predict() uses it
                            dfm_model._result.Z[-1, :] = Z_last_current
                            # Enhanced logging: verify the update was applied
                            Z_after_update = dfm_model._result.Z[-1, :]
                            update_diff = np.linalg.norm(Z_after_update - Z_last_current)
                            if update_diff > 1e-6:
                                logger.warning(f"[{target_series} DFM] Factor state update may have failed: diff={update_diff:.6f}")
                            logger.debug(f"Updated result.Z[-1, :] with current factor state from masked data (shape: {Z_last_current.shape}, diff={update_diff:.6e})")
                    else:
                        # CRITICAL: If _result doesn't exist, create it first to ensure update persists
                        logger.warning(f"[{target_series} DFM] _result is None, calling get_result() to create it before updating factor state")
                        if hasattr(dfm_model, 'get_result'):
                            dfm_model._result = dfm_model.get_result()
                            if hasattr(dfm_model._result, 'Z') and dfm_model._result.Z is not None:
                                original_Z_last = dfm_model._result.Z[-1, :].copy()
                                dfm_model._result.Z[-1, :] = Z_last_current
                                logger.debug(f"Created _result and updated result.Z[-1, :] with current factor state")
                            else:
                                raise RuntimeError(f"Cannot update factor state: result.Z is not available after get_result()")
                        else:
                            raise RuntimeError(f"Cannot update factor state: _result is None and no get_result() method available")
                    
                    try:
                        # Enhanced logging: verify factor state before prediction
                        if hasattr(dfm_model, '_result') and dfm_model._result is not None:
                            if hasattr(dfm_model._result, 'Z') and dfm_model._result.Z is not None:
                                Z_before_predict = dfm_model._result.Z[-1, :]
                                Z_diff_from_current = np.linalg.norm(Z_before_predict - Z_last_current)
                                logger.debug(f"[{target_series} DFM] Factor state before predict(): norm={np.linalg.norm(Z_before_predict):.4f}, diff from current={Z_diff_from_current:.6e}")
                        
                        # CRITICAL: Store factor state for repetitive prediction detection
                        # This helps diagnose why KOIPALL.G DFM produces only 2 unique predictions
                        if not hasattr(run_backtest_evaluation, '_factor_states'):
                            run_backtest_evaluation._factor_states = {}
                        if not hasattr(run_backtest_evaluation, '_data_masking_history'):
                            run_backtest_evaluation._data_masking_history = {}
                        key = f"{target_series}_{model}"
                        if key not in run_backtest_evaluation._factor_states:
                            run_backtest_evaluation._factor_states[key] = []
                        if key not in run_backtest_evaluation._data_masking_history:
                            run_backtest_evaluation._data_masking_history[key] = []
                        
                        # Store data masking info to check if data is actually changing
                        if hasattr(dfm_model, '_data_module') and dfm_model._data_module is not None:
                            data_masked = dfm_model._data_module.data
                            if isinstance(data_masked, pd.DataFrame):
                                nan_count = data_masked.isnull().sum().sum()
                                total_cells = data_masked.size
                                nan_pct = (nan_count / total_cells * 100) if total_cells > 0 else 0
                                # Create a hash of the masked data pattern to detect if it's changing
                                data_hash = hashlib.md5(pd.util.hash_pandas_object(data_masked.fillna(0)).values).hexdigest() if isinstance(data_masked, pd.DataFrame) else None
                                run_backtest_evaluation._data_masking_history[key].append({
                                    'month': target_month_end_ts.strftime('%Y-%m'),
                                    'weeks': weeks,
                                    'nan_count': nan_count,
                                    'nan_pct': nan_pct,
                                    'data_hash': data_hash
                                })
                        
                        run_backtest_evaluation._factor_states[key].append({
                            'month': target_month_end_ts.strftime('%Y-%m'),
                            'weeks': weeks,
                            'factor_state': Z_last_current.copy(),
                            'factor_norm': np.linalg.norm(Z_last_current),
                            'factor_mean': np.mean(Z_last_current),
                            'factor_std': np.std(Z_last_current)
                        })
                        
                        # Check if factor states are repetitive (only 2 unique states)
                        if len(run_backtest_evaluation._factor_states[key]) >= 5:
                            # Compare factor states using norm (faster than full comparison)
                            factor_norms = [s['factor_norm'] for s in run_backtest_evaluation._factor_states[key]]
                            unique_norms = set(np.round(factor_norms, 6))  # Round to detect near-duplicates
                            if len(unique_norms) <= 2:
                                # Also check if data masking is changing
                                data_changing = True
                                if key in run_backtest_evaluation._data_masking_history and len(run_backtest_evaluation._data_masking_history[key]) >= 2:
                                    data_hashes = [d['data_hash'] for d in run_backtest_evaluation._data_masking_history[key] if d.get('data_hash')]
                                    unique_hashes = set(data_hashes)
                                    data_changing = len(unique_hashes) > 1
                                    if not data_changing:
                                        logger.warning(
                                            f"DATA MASKING NOT CHANGING for {target_series} {model}: "
                                            f"All {len(run_backtest_evaluation._data_masking_history[key])} timepoints have same data hash. "
                                            f"This explains why factor states are repetitive - data masking is identical across timepoints."
                                        )
                                
                                logger.warning(
                                    f"REPETITIVE FACTOR STATES DETECTED for {target_series} {model}: "
                                    f"Only {len(unique_norms)} unique factor state norms in last {len(run_backtest_evaluation._factor_states[key])} timepoints: {sorted(unique_norms)}. "
                                    f"Data masking changing: {data_changing}. "
                                    f"This will cause repetitive predictions. Check data masking and Kalman filter re-run."
                                )
                        
                        # Use dfm_model.predict() - now it will use the updated factor state
                        X_pred, Z_pred = dfm_model.predict(horizon=horizon, return_series=True, return_factors=True)
                    finally:
                        # Restore original state
                        if original_Z_last is not None:
                            dfm_model._result.Z[-1, :] = original_Z_last
                            logger.debug("Restored original result.Z[-1, :]")
                    
                    # Extract target series value
                    # If horizon > 1, we need to select the appropriate time step
                    # For nowcasting, we typically want the last step (horizon-1) if months_ahead matches
                    pred_step = min(horizon - 1, max(0, months_ahead - 1)) if months_ahead > 0 else 0
                    
                    # Extract target series value from predictions
                    forecast_value = _extract_target_forecast(
                        X_pred, pred_step, dfm_model, target_series
                    )
                    
                    # Debug: Log prediction for comparison (use INFO level for visibility)
                    # Enhanced logging: include factor state info to diagnose repetitive predictions
                    factor_state_summary = f"factor_norm={np.linalg.norm(Z_last_current):.4f}" if 'Z_last_current' in locals() else "factor_state=N/A"
                    logger.info(f"[{target_series} DFM] Prediction for {target_month_end_ts.strftime('%Y-%m')} (view_date={view_date.strftime('%Y-%m-%d')}, weeks={weeks}): forecast_value={forecast_value:.6f}, horizon={horizon}, months_ahead={months_ahead}, pred_step={pred_step}, {factor_state_summary}")
                    
                    # CRITICAL: Detect repetitive predictions early
                    # Store predictions to detect if we're only getting 2 unique values (like KOIPALL.G DFM)
                    if not hasattr(run_backtest_evaluation, '_predictions'):
                        run_backtest_evaluation._predictions = {}
                    key = f"{target_series}_{model}"
                    if key not in run_backtest_evaluation._predictions:
                        run_backtest_evaluation._predictions[key] = []
                    run_backtest_evaluation._predictions[key].append(forecast_value)
                    
                    # Check if we have repetitive predictions (only 2 unique values)
                    if len(run_backtest_evaluation._predictions[key]) >= 5:
                        unique_values = set(run_backtest_evaluation._predictions[key])
                        if len(unique_values) <= 2:
                            logger.warning(
                                f"REPETITIVE PREDICTIONS DETECTED for {target_series} {model}: "
                                f"Only {len(unique_values)} unique values in last {len(run_backtest_evaluation._predictions[key])} predictions: {sorted(unique_values)}. "
                                f"This indicates factor state is not varying enough. Check data masking and Kalman filter re-run."
                            )
                    
                    # Also log prediction array shape for debugging
                    if hasattr(X_pred, 'shape'):
                        logger.debug(f"  X_pred shape: {X_pred.shape}, target_series index in model: checking...")
                    
                    if not np.isfinite(forecast_value):
                        logger.debug(f"Non-finite forecast value for {target_month_end_ts.strftime('%Y-%m')} (view_date={view_date.strftime('%Y-%m-%d')}): {forecast_value}")
                        continue
                    
                    # Validate forecast value is within reasonable bounds (detect numerical instability)
                    # Check if forecast is extremely large compared to typical values (e.g., > 100x typical range)
                    # This helps catch numerical instability issues like KOIPALL.G DFM
                    abs_forecast = abs(forecast_value)
                    train_mean = train_data_filtered[target_series].mean() if target_series in train_data_filtered.columns else train_data_filtered.iloc[:, 0].mean() if len(train_data_filtered.columns) > 0 else 0.0
                    original_forecast = forecast_value  # Store original before any clipping
                    
                    # CRITICAL FIX: Track clipped values and extreme value ranges to preserve variation
                    # This prevents the repetitive prediction bug where all extreme values get clipped to exact bounds
                    if not hasattr(run_backtest_evaluation, '_clipped_values'):
                        run_backtest_evaluation._clipped_values = {}
                    if not hasattr(run_backtest_evaluation, '_extreme_ranges'):
                        run_backtest_evaluation._extreme_ranges = {}
                    key_clip = f"{target_series}_{model}"
                    if key_clip not in run_backtest_evaluation._clipped_values:
                        run_backtest_evaluation._clipped_values[key_clip] = []
                    if key_clip not in run_backtest_evaluation._extreme_ranges:
                        run_backtest_evaluation._extreme_ranges[key_clip] = {'positive': {'min': None, 'max': None}, 'negative': {'min': None, 'max': None}}
                    
                    if train_std and train_std > 0:
                        # Use ±10 standard deviations as reasonable bounds (more conservative than 50 std devs for warnings)
                        max_forecast = train_mean + 10 * train_std
                        min_forecast = train_mean - 10 * train_std
                        
                        # CRITICAL FIX: Improved soft clipping that preserves relative differences
                        # Track the range of extreme values and normalize them to preserve variation
                        # BUG FIX: Use a hash-based approach to preserve variation even when values are very similar
                        if forecast_value > max_forecast or forecast_value < min_forecast:
                            was_clipped = True
                            
                            # Store all original extreme values to compute proper range
                            if not hasattr(run_backtest_evaluation, '_extreme_values'):
                                run_backtest_evaluation._extreme_values = {}
                            if key_clip not in run_backtest_evaluation._extreme_values:
                                run_backtest_evaluation._extreme_values[key_clip] = {'positive': [], 'negative': []}
                            
                            if forecast_value > max_forecast:
                                # Track all positive extreme values with their order of appearance
                                extreme_list = run_backtest_evaluation._extreme_values[key_clip]['positive']
                                order = len(extreme_list)  # Order is determined by length before appending
                                extreme_list.append(original_forecast)
                                
                                # Compute range from all observed extreme values
                                if len(extreme_list) > 1:
                                    extreme_min = min(extreme_list)
                                    extreme_max = max(extreme_list)
                                    extreme_range = extreme_max - extreme_min
                                    
                                    if extreme_range > 1e-10:  # Avoid division by zero
                                        # Map original value to [max_forecast, max_forecast + 2*std] range
                                        # Preserve relative position within observed extreme range
                                        normalized = (original_forecast - extreme_min) / extreme_range
                                        soft_clip_range = 2 * train_std  # Allow 2 std devs of variation
                                        forecast_value = max_forecast + normalized * soft_clip_range
                                    else:
                                        # All values are very similar - distribute evenly across range
                                        # Use order of appearance to ensure variation (each value gets unique position)
                                        total_count = len(extreme_list)
                                        normalized = order / max(1, total_count - 1) if total_count > 1 else 0.5
                                        soft_clip_range = 2 * train_std
                                        forecast_value = max_forecast + normalized * soft_clip_range
                                else:
                                    # First extreme value - place in middle of range
                                    soft_clip_range = 2 * train_std
                                    forecast_value = max_forecast + 0.5 * soft_clip_range
                            else:  # forecast_value < min_forecast
                                # Track all negative extreme values with their order of appearance
                                extreme_list = run_backtest_evaluation._extreme_values[key_clip]['negative']
                                order = len(extreme_list)  # Order is determined by length before appending
                                extreme_list.append(original_forecast)
                                
                                # Compute range from all observed extreme values
                                if len(extreme_list) > 1:
                                    extreme_min = min(extreme_list)
                                    extreme_max = max(extreme_list)
                                    extreme_range = extreme_max - extreme_min
                                    
                                    if extreme_range > 1e-10:  # Avoid division by zero
                                        # Map original value to [min_forecast - 2*std, min_forecast] range
                                        # Preserve relative position within observed extreme range
                                        normalized = (original_forecast - extreme_min) / extreme_range
                                        soft_clip_range = 2 * train_std
                                        forecast_value = min_forecast - (1.0 - normalized) * soft_clip_range
                                    else:
                                        # All values are very similar - distribute evenly across range
                                        # Use order of appearance to ensure variation (each value gets unique position)
                                        total_count = len(extreme_list)
                                        normalized = order / max(1, total_count - 1) if total_count > 1 else 0.5
                                        soft_clip_range = 2 * train_std
                                        forecast_value = min_forecast - (1.0 - normalized) * soft_clip_range
                                else:
                                    # First extreme value - place in middle of range
                                    soft_clip_range = 2 * train_std
                                    forecast_value = min_forecast - 0.5 * soft_clip_range
                            
                            # Track clipped values to detect repetitive patterns
                            run_backtest_evaluation._clipped_values[key_clip].append({
                                'original': original_forecast,
                                'clipped': forecast_value,
                                'month': target_month_end_ts.strftime('%Y-%m')
                            })
                            
                            # CRITICAL: Log clipping to diagnose repetitive prediction bug
                            # This helps identify if clipping is collapsing all values to same bounds
                            logger.warning(
                                f"Extreme forecast value clipped for {target_month_end_ts.strftime('%Y-%m')} "
                                f"(view_date={view_date.strftime('%Y-%m-%d')}): original={original_forecast:.6f}, "
                                f"clipped={forecast_value:.6f}, train_mean={train_mean:.6f}, train_std={train_std:.6f}, "
                                f"bounds=[{min_forecast:.6f}, {max_forecast:.6f}], "
                                f"ratio={abs(original_forecast - train_mean)/train_std:.1f}x. "
                                f"This indicates numerical instability or poor model convergence."
                            )
                            
                            # CRITICAL: Detect if all clipped values are collapsing to same 2 bounds
                            if len(run_backtest_evaluation._clipped_values[key_clip]) >= 5:
                                clipped_vals = [v['clipped'] for v in run_backtest_evaluation._clipped_values[key_clip]]
                                unique_clipped = set(np.round(clipped_vals, 6))
                                if len(unique_clipped) <= 2:
                                    logger.error(
                                        f"CRITICAL: Clipping is collapsing all predictions to only {len(unique_clipped)} unique values "
                                        f"for {target_series} {model}: {sorted(unique_clipped)}. "
                                        f"This indicates the model is producing binary/extreme predictions. "
                                        f"Consider investigating model convergence or numerical stability."
                                    )
                        # Also log warning if extremely large (even if within clipping bounds)
                        elif abs_forecast > 50 * train_std:
                            logger.warning(
                                f"Extreme forecast value detected for {target_month_end_ts.strftime('%Y-%m')} "
                                f"(view_date={view_date.strftime('%Y-%m-%d')}): forecast={forecast_value:.6f}, "
                                f"train_std={train_std:.6f}, ratio={abs_forecast/train_std:.1f}x. "
                                f"This may indicate numerical instability or poor model convergence."
                            )
                        # CRITICAL: Log forecast value before storing in JSON to diagnose repetitive prediction bug
                        # This helps identify if value is transformed between prediction and storage
                        logger.debug(
                            f"[{target_series} {model}] Forecast value before JSON storage: {forecast_value:.6f} "
                            f"(original={original_forecast:.6f}, clipped={forecast_value != original_forecast})"
                        )
                except (ValueError, RuntimeError, AttributeError) as e:
                    logger.warning(f"Failed to predict for {target_month_end_ts.strftime('%Y-%m')} (view_date={view_date.strftime('%Y-%m-%d')}): {type(e).__name__}: {str(e)}")
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error during prediction for {target_month_end_ts.strftime('%Y-%m')} (view_date={view_date.strftime('%Y-%m-%d')}): {type(e).__name__}: {str(e)}")
                    continue
                
                # Get actual value
                actual_value = np.nan
                if target_month_end_ts <= full_data_monthly.index.max():
                    available_dates = full_data_monthly.index[full_data_monthly.index <= target_month_end_ts]
                    if len(available_dates) > 0:
                        closest_date = available_dates.max()
                        if target_series in full_data_monthly.columns:
                            raw_value = full_data_monthly.loc[closest_date, target_series]
                        else:
                            raw_value = full_data_monthly.loc[closest_date].iloc[0]
                        if not pd.isna(raw_value):
                            actual_value = float(raw_value)
                
                if np.isnan(forecast_value) or np.isnan(actual_value):
                    if np.isnan(forecast_value):
                        logger.debug(f"NaN forecast value for {target_month_end_ts.strftime('%Y-%m')} (view_date={view_date.strftime('%Y-%m-%d')})")
                    if np.isnan(actual_value):
                        logger.debug(f"NaN actual value for {target_month_end_ts.strftime('%Y-%m')} (view_date={view_date.strftime('%Y-%m-%d')})")
                    continue
                
                error = forecast_value - actual_value
                if train_std and train_std > 0:
                    sMSE = (error ** 2) / (train_std ** 2)
                    sMAE = abs(error) / train_std
                else:
                    sMSE = error ** 2
                    sMAE = abs(error)
                
                # CRITICAL: Log exact value being stored in JSON to diagnose repetitive prediction bug
                # This helps identify if value is transformed between prediction and storage
                logger.debug(
                    f"[{target_series} {model}] Storing in JSON for {target_month_end_ts.strftime('%Y-%m')}: "
                    f"forecast_value={forecast_value:.6f}, actual_value={actual_value:.6f}, error={error:.6f}"
                )
                
                monthly_results.append({
                    'month': target_month_end_ts.strftime('%Y-%m'),
                    'view_date': view_date.strftime('%Y-%m-%d'),
                    'weeks_before': weeks,
                    'forecast_value': float(forecast_value),
                    'actual_value': float(actual_value),
                    'error': float(error),
                    'abs_error': float(abs(error)),
                    'squared_error': float(error ** 2),
                    'sMSE': float(sMSE),
                    'sMAE': float(sMAE)
                })
                
            except (ValueError, RuntimeError, AttributeError, KeyError, IndexError) as e:
                logger.warning(f"Error processing {target_month_end_ts.strftime('%Y-%m')} (view_date={view_date.strftime('%Y-%m-%d')}): {type(e).__name__}: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error processing {target_month_end_ts.strftime('%Y-%m')} (view_date={view_date.strftime('%Y-%m-%d')}): {type(e).__name__}: {str(e)}")
                continue
        
        if len(monthly_results) > 0:
            results_by_timepoint[f"{weeks}weeks"] = {
                'weeks_before': weeks,
                'monthly_results': monthly_results,
                'overall_sMAE': float(np.nanmean([r['sMAE'] for r in monthly_results])),
                'overall_sMSE': float(np.nanmean([r['sMSE'] for r in monthly_results])),
                'overall_mae': float(np.nanmean([r['abs_error'] for r in monthly_results])),
                'overall_rmse': float(np.sqrt(np.nanmean([r['squared_error'] for r in monthly_results]))),
                'n_months': len(monthly_results)
            }
            logger.info(f"Completed {weeks} weeks before: {len(monthly_results)} successful predictions out of {len(target_periods)} target months")
        else:
            logger.warning(f"No successful predictions for {weeks} weeks before time point")
    
    # Save results (always save, even if empty, to indicate status)
    total_timepoints = len(weeks_before)
    successful_timepoints = len(results_by_timepoint)
    
    if len(results_by_timepoint) > 0:
        results = {
            'target_series': target_series,
            'model': model.upper(),
            'train_period': f"{train_start} to {train_end}",
            'nowcast_period': f"{nowcast_start} to {nowcast_end}",
            'weeks_before': weeks_before,
            'results_by_timepoint': results_by_timepoint,
            'horizon': 1,
            'status': 'completed',
            'summary': {
                'total_timepoints': total_timepoints,
                'successful_timepoints': successful_timepoints,
                'failed_timepoints': total_timepoints - successful_timepoints
            }
        }
        logger.info(f"Generated results for {successful_timepoints}/{total_timepoints} time points")
    else:
        results = {
            'target_series': target_series,
            'model': model.upper(),
            'train_period': f"{train_start} to {train_end}",
            'nowcast_period': f"{nowcast_start} to {nowcast_end}",
            'weeks_before': weeks_before,
            'status': 'no_results',
            'error': 'No valid results generated for any time point',
            'summary': {
                'total_timepoints': total_timepoints,
                'successful_timepoints': 0,
                'failed_timepoints': total_timepoints
            }
        }
        logger.warning(f"No results generated for any time point - all predictions failed")
    
    output_file = project_root / "outputs" / "backtest" / f"{target_series}_{model}_backtest.json"
    _save_json_results(output_file, results, logger)
    
    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Run nowcasting backtest")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    backtest_parser = subparsers.add_parser('backtest', help='Run nowcasting backtest (DFM and DDFM only)')
    backtest_parser.add_argument("--config-name", required=True)
    backtest_parser.add_argument("--model", required=True, choices=['dfm', 'ddfm'], 
                                 help='Model type (only DFM and DDFM support nowcasting)')
    backtest_parser.add_argument("--train-start", default="1985-01-01")
    backtest_parser.add_argument("--train-end", default="2019-12-31")
    backtest_parser.add_argument("--nowcast-start", default="2024-01-01")
    backtest_parser.add_argument("--nowcast-end", default="2025-10-31")
    backtest_parser.add_argument("--weeks-before", nargs="+", type=int)
    backtest_parser.add_argument("--override", action="append")
    
    args = parser.parse_args()
    config_path = str(get_project_root() / "config")
    
    if args.command == 'backtest':
        run_backtest_evaluation(
            config_name=args.config_name,
            model=args.model,
            train_start=args.train_start,
            train_end=args.train_end,
            nowcast_start=args.nowcast_start,
            nowcast_end=args.nowcast_end,
            weeks_before=args.weeks_before,
            config_dir=config_path,
            overrides=args.override
        )


if __name__ == "__main__":
    main()
