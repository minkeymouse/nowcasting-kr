"""Training function for DDFM (Deep Dynamic Factor Model)."""

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


def train_ddfm_model(
    model_type: str,
    cfg: Any,
    data: pd.DataFrame,
    model_name: str,
    outputs_dir: Path = None,
    model_params: Optional[Dict[str, Any]] = None,
    data_loader: Optional[Any] = None
) -> None:
    """Train DDFM model.
    
    Note: DDFM is a deep dynamic factor model that trains a single model capable of
    forecasting any horizon.
    
    Parameters
    ----------
    model_type : str
        Model type: 'ddfm'
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
        from dfm_python import DDFM, DDFMDataset
        from dfm_python.config import DDFMConfig
        import joblib
    except ImportError as e:
        raise ImportError(f"dfm_python not available: {e}")
    
    logger.info(f"Training DDFM model (data shape: {data.shape})")
    
    # Convert DictConfig to dict if needed
    if model_params is not None and _HAS_OMEGACONF:
        try:
            from omegaconf import DictConfig, OmegaConf
            if isinstance(model_params, DictConfig):
                model_params = OmegaConf.to_container(model_params, resolve=True)
        except (ImportError, TypeError):
            pass
    
    if not model_params:
        raise ValueError("DDFM config must be provided via model_params.")
    
    if not isinstance(model_params, dict):
        model_params = dict(model_params)
    model_params.setdefault('model_type', 'ddfm')
    
    # Create config
    config = DDFMConfig.from_dict(model_params)
    
    # Get target series
    target_series = model_params.get('target_series')
    available_targets = [t for t in target_series if t in data.columns] if target_series else None
    if target_series and not available_targets:
        logger.warning("No target series found in data. Using all columns as targets.")
        available_targets = None
    
    # Configure dfm_python logging first
    try:
        from dfm_python.logger import configure_logging
        from pathlib import Path
        
        # Create log/log_ddfm directory for core debugging logs
        log_dir = get_project_root() / "log"
        log_ddfm_dir = log_dir / "log_ddfm"
        log_dir.mkdir(parents=True, exist_ok=True)
        configure_logging(
            level=logging.INFO,
            log_dfm_dir=log_ddfm_dir,
            enable_core_log=True,
            log_file_prefix="ddfm_core"
        )
    except Exception as e:
        logger.warning(f"Could not configure dfm_python logging: {e}")
    
    # Create scaler for inverse transformation during prediction
    target_scaler = None
    if data_loader is not None:
        original_data = data_loader.training_data
        original_targets = original_data[available_targets] if available_targets else original_data
        original_targets = original_targets.select_dtypes(include=[np.number]).reset_index(drop=True)
        
        scaler_type = model_params.get('target_scaler', 'standard')
        if scaler_type == 'robust':
            from sklearn.preprocessing import RobustScaler
            target_scaler = RobustScaler()
        else:
            from sklearn.preprocessing import StandardScaler
            target_scaler = StandardScaler()
        from sklearn.preprocessing import StandardScaler as FeatureScaler
        feature_scaler = FeatureScaler()
        
        target_scaler.fit(original_targets.to_numpy(dtype=np.float64))
    
    dataset = DDFMDataset(
        data=data,
        time_idx='index',
        target_series=available_targets,
        target_scaler=target_scaler,
        feature_scaler=feature_scaler
    )
    
    # Create DDFM model
    encoder_layers = model_params.get('encoder_layers', [64, 32])
    decoder_type = model_params.get('decoder_type', 'linear')
    model = DDFM(
        dataset=dataset,
        config=config,
        encoder_size=tuple(encoder_layers) if isinstance(encoder_layers, list) else encoder_layers,
        decoder_type=decoder_type,
        activation=model_params.get('activation', 'relu'),
        learning_rate=model_params.get('learning_rate', 0.005),
        optimizer='Adam',
        n_mc_samples=model_params.get('n_mc_samples', 10),
        window_size=model_params.get('window_size', 100),
        max_iter=model_params.get('max_epoch', 50),
        tolerance=model_params.get('tolerance', 0.0005),
        disp=model_params.get('disp', 10),
        seed=model_params.get('seed', None)
    )
    
    logger.info(f"Starting DDFM training (max_iter={model.max_iter})...")
    model.fit()
    logger.info("DDFM training completed. Building state-space model...")
    model.build_state_space()
    logger.info("State-space model built successfully")
    
    # Save model
    model_path = outputs_dir / "model.pkl"
    if hasattr(model, 'save') and callable(model.save):
        model.save(model_path)
        logger.info(f"Model saved: {model_path}")
    else:
        joblib.dump(model, model_path)
        logger.info(f"Model saved: {model_path}")
    
    # Save dataset
    dataset_path = outputs_dir / "dataset.pkl"
    try:
        joblib.dump(dataset, dataset_path)
    except Exception as e:
        logger.warning(f"Failed to save dataset: {e}")
