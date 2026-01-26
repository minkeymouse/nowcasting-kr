"""Training function for DFM (Dynamic Factor Model)."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
import pandas as pd
import numpy as np

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
    
    # Get covariates from config
    covariates = model_params.get('covariates')
    
    # Filter covariates to only those available in data
    available_covariates = None
    if covariates:
        available_covariates = [c for c in covariates if c in data.columns]
        if len(available_covariates) < len(covariates):
            missing = [c for c in covariates if c not in data.columns]
            logger.warning(f"Some covariates not found in data: {missing}")
    
    # Scaler: DO NOT create for DFM (model will create internally).
    # 
    # DFM's predict() returns predictions in transformed space (chg/logdiff), NOT standardized space.
    # DFM standardizes data internally during training (creates scaler if not provided).
    # Predictions are unstandardized using the scaler, then returned in transformed space.
    # We only need to reverse metadata transformations (chg → level, logdiff → level) in evaluation.
    # 
    # Setting scaler = None ensures DFM creates its own scaler internally during fit(),
    # which will be used for standardization and inverse transformation.
    scaler = None
    logger.info("DFM scaler set to None - model will create scaler internally during training")
    
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
        # Pass covariates to dataset
        if available_covariates is not None:
            dataset = DFMDataset(config=config, data=data, covariates=available_covariates)
        else:
            # No covariates: all series are targets (default)
            dataset = DFMDataset(config=config, data=data)
        # Note: scaler is always None for DFM (model creates internally)
        
        X = dataset.get_processed_data()
        missing_pct = data.isnull().sum().sum() / (data.shape[0] * data.shape[1]) * 100
        logger.info(f"Dataset ready: {X.shape}, missing: {missing_pct:.1f}%")
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
        # Check if checkpoint exists despite error (partial save)
        checkpoint_path = outputs_dir / "model.pkl"
        if checkpoint_path.exists():
            logger.warning(f"Checkpoint exists despite training error: {checkpoint_path}. "
                         f"This may be from a previous run or partial save.")
        raise
    
    # Dataset metadata is now saved in model.pkl, so we don't need to save dataset.pkl separately
    # This reduces file size and eliminates redundancy
