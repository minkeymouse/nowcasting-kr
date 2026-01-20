"""Training function for DFM (Dynamic Factor Model)."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
import pandas as pd
import numpy as np
import joblib

from src.utils import get_project_root

logger = logging.getLogger(__name__)

# Try to import DictConfig for type checking
try:
    from omegaconf import DictConfig
    _HAS_OMEGACONF = True
except ImportError:
    DictConfig = type(None)  # Dummy type if not available
    _HAS_OMEGACONF = False


def train_dfm_model(
    model_type: str,
    cfg: Any,
    data: pd.DataFrame,
    model_name: str,
    outputs_dir: Path = None,
    model_params: Optional[Dict[str, Any]] = None,
    data_loader: Optional[Any] = None
) -> None:
    """Train DFM model.
    
    Note: DFM is a dynamic factor model that trains a single model capable of
    forecasting any horizon. Unlike attention-based models, DFM doesn't use
    prediction_length - the model structure defines forecast dynamics.
    
    Parameters
    ----------
    model_type : str
        Model type: 'dfm'
    cfg : Any
        Hydra config object
    data : pd.DataFrame
        Preprocessed training data (standardized, without date columns)
    model_name : str
        Model name for saving
    outputs_dir : Path
        Directory to save trained model
    model_params : dict, optional
        Model parameters dictionary
    data_loader : optional
        Data loader object for metadata access
    """
    try:
        from dfm_python import DFM, DFMDataset
        from dfm_python.config import DFMConfig
        import joblib
    except ImportError as e:
        raise ImportError(f"dfm_python not available: {e}")
    
    logger.info(f"Training DFM model (data shape: {data.shape})")
    
    # Convert DictConfig to dict if needed
    if model_params is not None and _HAS_OMEGACONF:
        try:
            from omegaconf import DictConfig, OmegaConf
            if isinstance(model_params, DictConfig):
                model_params = OmegaConf.to_container(model_params, resolve=True)
        except (ImportError, TypeError):
            pass
    
    if not model_params:
        raise ValueError("DFM config must be provided via model_params.")
    
    if not isinstance(model_params, dict):
        model_params = dict(model_params)
    model_params.setdefault('model_type', 'dfm')
    
    # Create config
    config = DFMConfig.from_dict(model_params)
    
    # Get target series
    target_series = model_params.get('target_series')
    available_targets = [t for t in target_series if t in data.columns] if target_series else None
    if target_series and not available_targets:
        logger.warning("No target series found in data. Using all columns as targets.")
        available_targets = None
    
    # Target scaler: DO NOT create for DFM.
    # 
    # DFM's predict() returns predictions in transformed space (chg/logdiff), NOT standardized space.
    # Unlike DDFM, DFM doesn't standardize data internally - it uses transformed data as-is.
    # Therefore, we should NOT use target_scaler.inverse_transform() in DFM's predict().
    # Instead, predictions are already in transformed space, and we only need to reverse
    # metadata transformations (chg → level, logdiff → level) in the evaluation code.
    # 
    # Setting target_scaler = None ensures DFM.predict() returns predictions in transformed space,
    # which matches what inverse_transform_predictions() expects.
    target_scaler = None
    # region agent log
    try:
        import json, time
        with open("/data/nowcasting-kr/.cursor/debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "dfm-train",
                "hypothesisId": "H1_target_scaler_creation",
                "location": "src/train/dfm.py:target_scaler_set_to_none",
                "message": "target_scaler set to None for DFM (predictions in transformed space)",
                "data": {
                    "available_targets": available_targets,
                    "available_targets_count": len(available_targets) if available_targets else 0,
                },
                "timestamp": int(time.time() * 1000),
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # endregion
    
    logger.info("DFM target_scaler set to None - predictions will be in transformed space (chg/logdiff)")
    
    # Configure dfm_python logging first
    try:
        from dfm_python.logger import configure_logging
        from pathlib import Path
        
        # Create log/log_dfm directory for core debugging logs
        log_dir = get_project_root() / "log"
        log_dfm_dir = log_dir / "log_dfm"
        log_dir.mkdir(parents=True, exist_ok=True)
        configure_logging(
            level=logging.INFO,
            log_dfm_dir=log_dfm_dir,
            enable_core_log=True,
            log_file_prefix="dfm_core"
        )
    except Exception as e:
        logger.warning(f"Could not configure dfm_python logging: {e}")
    
    # Create dataset
    try:
        dataset = DFMDataset(config=config, data=data, target_series=available_targets)
        # Set target_scaler as attribute (DFMDataset.get_initialization_params() will pick it up)
        if target_scaler is not None:
            dataset.target_scaler = target_scaler
            logger.info("Set target_scaler on DFMDataset for inverse transformation during prediction")
        
        X = dataset.get_processed_data()
        missing_pct = data.isnull().sum().sum() / (data.shape[0] * data.shape[1]) * 100
        logger.info(f"Dataset ready: {X.shape}, missing: {missing_pct:.1f}%")
        
        # region agent log
        try:
            import json, time
            # Check what space the training data is in
            if available_targets and len(available_targets) > 0:
                # Find target index in X (need to match column order)
                target_col = available_targets[0]
                if target_col in data.columns:
                    # Get the column index in the original data
                    col_idx = list(data.columns).index(target_col)
                    # X should have same column order as data
                    if col_idx < X.shape[1]:
                        target_data = X[:, col_idx]
                        # Check if data has NaNs
                        target_data_clean = target_data[~np.isnan(target_data)]
                        with open("/data/nowcasting-kr/.cursor/debug.log", "a", encoding="utf-8") as f:
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "dfm-train",
                                "hypothesisId": "H7_training_data_space",
                                "location": "src/train/dfm.py:training_data_check",
                                "message": "Checking training data space (X from dataset)",
                                "data": {
                                    "X_shape": X.shape,
                                    "target_col": target_col,
                                    "target_col_idx": col_idx,
                                    "target_data_mean": float(np.mean(target_data_clean)) if len(target_data_clean) > 0 else None,
                                    "target_data_std": float(np.std(target_data_clean)) if len(target_data_clean) > 0 else None,
                                    "target_data_min": float(np.min(target_data_clean)) if len(target_data_clean) > 0 else None,
                                    "target_data_max": float(np.max(target_data_clean)) if len(target_data_clean) > 0 else None,
                                    "target_data_nan_count": int(np.isnan(target_data).sum()),
                                    "transformed_data_mean": float(data[target_col].mean()) if target_col in data.columns else None,
                                    "transformed_data_std": float(data[target_col].std()) if target_col in data.columns else None,
                                },
                                "timestamp": int(time.time() * 1000),
                            }, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Could not log training data space: {e}")
        # endregion
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}", exc_info=True)
        raise
    
    # Create and train model
    try:
        model = DFM(config)
        logger.info(f"Starting DFM training (max_iter={model.max_iter}, threshold={model.threshold:.2e})")
        
        # Create checkpoint callback for periodic saving (every 5 iterations)
        def save_checkpoint(iteration: int, state: Dict[str, Any]) -> None:
            """Save checkpoint every 5 iterations, overriding previous checkpoint."""
            try:
                # Import DFMStateSpaceParams for checkpoint saving
                from dfm_python.config.schema.params import DFMStateSpaceParams
                
                # Update model parameters from current state
                model._update_parameters(
                    state['A'], state['C'], state['Q'],
                    state['R'], state['Z_0'], state['V_0']
                )
                # Create training_state for save() method
                model.training_state = DFMStateSpaceParams(
                    A=state['A'], C=state['C'], Q=state['Q'],
                    R=state['R'], Z_0=state['Z_0'], V_0=state['V_0']
                )
                # Store training metadata
                model._training_loglik = state.get('loglik')
                model._training_num_iter = state.get('num_iter')
                model._training_converged = state.get('converged', False)
                
                # Save checkpoint using model's save() method (overwrites previous)
                model_path = outputs_dir / "model.pkl"
                model.save(model_path)
                logger.info(f"✓ Checkpoint saved at iteration {iteration} (overwrites previous): {model_path}")
            except Exception as e:
                logger.warning(f"Failed to save checkpoint at iteration {iteration}: {e}", exc_info=True)
        
        model.fit(X=X, dataset=dataset, checkpoint_callback=save_checkpoint)
        
        # Save model after training (checkpoint callback only saves every 5 iterations)
        model_path = outputs_dir / "model.pkl"
        model.save(model_path)
        logger.info(f"Model saved: {model_path}")
        
        # region agent log
        try:
            import json, time, joblib
            # Check if target_scaler was saved in checkpoint
            checkpoint = joblib.load(model_path)
            has_target_scaler = 'target_scaler' in checkpoint and checkpoint['target_scaler'] is not None
            with open("/data/nowcasting-kr/.cursor/debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "dfm-train",
                    "hypothesisId": "H2_target_scaler_saved",
                    "location": "src/train/dfm.py:model_saved",
                    "message": "Checking if target_scaler was saved to checkpoint",
                    "data": {
                        "has_target_scaler": has_target_scaler,
                        "target_scaler_in_checkpoint": has_target_scaler,
                        "model_has_target_scaler": hasattr(model, 'target_scaler') and model.target_scaler is not None,
                        "result_has_target_scaler": hasattr(model, '_result') and model._result is not None and hasattr(model._result, 'target_scaler') and model._result.target_scaler is not None,
                    },
                    "timestamp": int(time.time() * 1000),
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass
        # endregion
        
        # Log results
        if hasattr(model, '_training_converged'):
            logger.info(f"Training completed: converged={model._training_converged}, "
                      f"iterations={model._training_num_iter}, "
                      f"loglik={model._training_loglik:.4f}")
        elif hasattr(model, 'result') and model.result:
            logger.info(f"Training completed: converged={model.result.converged}, "
                      f"iterations={model.result.num_iter}, "
                      f"loglik={model.result.loglik:.4f}")
        else:
            logger.warning("Training completed but no result available")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    
    # Save dataset
    dataset_path = outputs_dir / "dataset.pkl"
    try:
        joblib.dump(dataset, dataset_path)
        logger.info(f"Dataset saved: {dataset_path}")
    except Exception as e:
        logger.warning(f"Failed to save dataset: {e}")
