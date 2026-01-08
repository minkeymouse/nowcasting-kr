"""Training functions for DFM and DDFM models."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import DictConfig for type checking
try:
    from omegaconf import DictConfig
    _HAS_OMEGACONF = True
except ImportError:
    DictConfig = type(None)  # Dummy type if not available
    _HAS_OMEGACONF = False


def train_dfm_python_model(
    model_type: str,
    config_name: str,
    cfg: Any,
    data: pd.DataFrame,
    model_name: str,
    horizons: Optional[int] = None,
    outputs_dir: Path = None,
    model_cfg_dict: Optional[Union[Dict[str, Any], DictConfig]] = None,
    data_loader: Optional[Any] = None,
    metadata: Optional[pd.DataFrame] = None  # Unused but kept for compatibility
) -> None:
    """Train DFM or DDFM model.
    
    Parameters
    ----------
    model_type : str
        Model type: 'dfm' or 'ddfm'
    config_name : str
        Config name (unused, kept for compatibility)
    cfg : Any
        Hydra config object
    data : pd.DataFrame
        Preprocessed training data (standardized, without date columns)
    model_name : str
        Model name for saving
    horizons : int, optional
        Forecast horizons (unused for training)
    outputs_dir : Path
        Directory to save trained model
    model_cfg_dict : dict, optional
        Model configuration dictionary
    """
    try:
        from dfm_python import DFM, DDFM, DFMDataset, DDFMDataset
        from dfm_python.config import DFMConfig, DDFMConfig
        import joblib
    except ImportError as e:
        raise ImportError(f"dfm_python not available: {e}")
    
    logger.info(f"Training {model_type.upper()} model...")
    logger.info(f"Training data shape: {data.shape}")
    
    # Convert DictConfig to dict if needed (consistent with dfm-python DictSource pattern)
    if model_cfg_dict is not None and _HAS_OMEGACONF:
        try:
            from omegaconf import DictConfig, OmegaConf
            if isinstance(model_cfg_dict, DictConfig):
                model_cfg_dict = OmegaConf.to_container(model_cfg_dict, resolve=True)
        except (ImportError, TypeError):
            pass  # Not a DictConfig or OmegaConf not available, use as-is
    
    # Validate config is provided (should always be from main.py)
    if not model_cfg_dict:
        raise ValueError(f"{model_type.upper()} config must be provided via model_cfg_dict from Hydra config.")
    
    # Ensure model_type is set for correct config detection (convert to dict first if needed)
    if not isinstance(model_cfg_dict, dict):
        model_cfg_dict = dict(model_cfg_dict)
    model_cfg_dict.setdefault('model_type', model_type.lower())
    
    # Create config from Hydra config dict - use dfm-python as-is
    model_type_lower = model_type.lower()
    if model_type_lower == 'dfm':
        config = DFMConfig.from_dict(model_cfg_dict)
    elif model_type_lower == 'ddfm':
        config = DDFMConfig.from_dict(model_cfg_dict)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Get target series from config (None means use all series)
    target_series = model_cfg_dict.get('target_series')
    available_targets = [t for t in target_series if t in data.columns] if target_series else None
    if target_series and not available_targets:
        logger.warning("No target series found in data. Using all columns as targets.")
        available_targets = None
    
    logger.info(f"Target series: {len(available_targets) if available_targets else 'all'} series" + 
                (f" ({available_targets[:3]}...)" if available_targets and len(available_targets) > 3 else 
                 (f" ({available_targets})" if available_targets else "")))
    
    # Train model (DFM and DDFM have different initialization patterns)
    if model_type_lower == 'dfm':
        dataset = DFMDataset(
            config=config,
            data=data,
            target_series=available_targets
        )
        
        # Create and train DFM model
        model = DFM(config)
        
        # Get processed data from dataset
        X = dataset.get_processed_data()
        logger.info(f"Training DFM with data shape: {X.shape}")
        
        # Fit model
        model.fit(X=X, dataset=dataset)
        
        logger.info("DFM training completed!")
        if hasattr(model, 'result') and model.result:
            logger.info(f"Converged: {model.result.converged}")
            logger.info(f"Iterations: {model.result.num_iter}")
            logger.info(f"Log-likelihood: {model.result.loglik:.4f}")
    
    elif model_type_lower == 'ddfm':
        # Create target scaler if specified in config
        scaler_type = model_cfg_dict.get('target_scaler')
        target_scaler = None
        if scaler_type == 'robust':
            from sklearn.preprocessing import RobustScaler
            target_scaler = RobustScaler()
        elif scaler_type == 'standard':
            from sklearn.preprocessing import StandardScaler
            target_scaler = StandardScaler()
        elif scaler_type and scaler_type != 'null':
            logger.warning(f"Unknown scaler type: {scaler_type}. Using RobustScaler.")
            from sklearn.preprocessing import RobustScaler
            target_scaler = RobustScaler()
        
        dataset = DDFMDataset(data=data, time_idx='index', target_series=available_targets, target_scaler=target_scaler)
        
        # Create DDFM model - get parameters directly from config dict
        encoder_layers = model_cfg_dict.get('encoder_layers', [64, 32])
        model = DDFM(
            dataset=dataset,
            config=config,
            encoder_size=tuple(encoder_layers) if isinstance(encoder_layers, list) else encoder_layers,
            decoder_type="linear",
            activation=model_cfg_dict.get('activation', 'relu'),
            learning_rate=model_cfg_dict.get('learning_rate', 0.005),
            optimizer='Adam',
            n_mc_samples=model_cfg_dict.get('n_mc_samples', 10),
            window_size=model_cfg_dict.get('window_size', 100),
            max_iter=model_cfg_dict.get('max_epoch', 50),
            tolerance=model_cfg_dict.get('tolerance', 0.0005),
            disp=model_cfg_dict.get('disp', 10),
            seed=model_cfg_dict.get('seed', None)
        )
        
        logger.info(f"Starting DDFM training (max {model.max_iter} iterations)...")
        model.fit()
        logger.info("DDFM training completed!")
        
        logger.info("Building state-space model...")
        model.build_state_space()
        logger.info("State-space model built successfully")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Save model (dfm-python doesn't have model.save() method, use joblib)
    model_path = outputs_dir / "model.pkl"
    logger.info(f"Saving model checkpoint to: {model_path.resolve()}")
    try:
        joblib.dump(model, model_path)
        logger.info(f"Model checkpoint saved successfully: {model_path}")
    except Exception as e:
        logger.warning(f"Failed to save model (pickling error): {e}. Model training completed successfully.")
    
    # Save dataset for reference
    dataset_path = outputs_dir / "dataset.pkl"
    try:
        joblib.dump(dataset, dataset_path)
        logger.info(f"Dataset saved to {dataset_path}")
    except Exception as e:
        logger.warning(f"Failed to save dataset: {e}")
