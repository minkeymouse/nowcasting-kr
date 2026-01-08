from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import hydra
import logging

from src.preprocess import ConsumptionData, InvestmentData, ProductionData
from src.utils import setup_logging, get_project_root
from src.train.sktime import train_sktime_model
from src.train.dfm_python import train_dfm_python_model

# Import forecast functions only when needed
def _import_forecast_funcs():
    """Lazy import forecast functions to avoid errors if not implemented."""
    try:
        from src.forecast.sktime import forecast
        from src.forecast.dfm_python import forecast as forecast_dfm
        return forecast, forecast_dfm
    except ImportError:
        # If forecast modules don't exist, create no-op stubs
        def forecast_stub(*args, **kwargs):
            raise NotImplementedError("Forecast functionality not yet implemented")
        return forecast_stub, forecast_stub

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> None:
    """Main entry point for training and forecasting.
    
    Uses Hydra's nested config groups:
    - DFM with data variants: model=dfm/investment, model=dfm/consumption, model=dfm/production
    - Other models: model=ddfm, model=itf, etc. (uses model/{name}/default.yaml)
    - Override via CLI: python -m src.main model=dfm/consumption data=consumption
    """
    setup_logging(log_dir=get_project_root() / "log", force=True)
    
    # Get data model type from config (consumption, investment, or production)
    data_model = cfg.get('data', 'investment').lower()
    
    # Get model name and horizon from config
    # Handle both flat (cfg.model.name) and nested (cfg.model.{model_name}.name) structures
    if hasattr(cfg.model, 'name'):
        model_name = cfg.model.name.lower()
        model_cfg = cfg.model
    else:
        # Nested structure: find the actual model config
        model_cfg = None
        for key in ['dfm', 'ddfm', 'itf', 'itransformer', 'patchtst', 'tft', 'timemixer']:
            if hasattr(cfg.model, key):
                model_cfg = getattr(cfg.model, key)
                model_name = key.lower()
                break
        
        if model_cfg is None:
            # Try to infer from dict keys
            model_dict = OmegaConf.to_container(cfg.model, resolve=True)
            if isinstance(model_dict, dict) and len(model_dict) == 1:
                model_name = list(model_dict.keys())[0].lower()
                model_cfg = cfg.model[model_name]
            else:
                raise ValueError(f"Could not determine model name from config structure: {list(model_dict.keys()) if isinstance(model_dict, dict) else 'non-dict'}")
    
    horizon = model_cfg.get('horizon', cfg.get('horizon', 88))
    
    # Warn if DFM model variant doesn't match data type (helpful for debugging)
    if model_name == 'dfm':
        target_series = model_cfg.get('target_series', [])
        if target_series:
            if 'KOEQUIPTE' in target_series and data_model != 'investment':
                logger.warning(
                    f"DFM config targets KOEQUIPTE (investment) but data={data_model}. "
                    f"Consider using: model=dfm/investment data=investment"
                )
            elif 'KOWRCCNSE' in target_series and data_model != 'consumption':
                logger.warning(
                    f"DFM config targets KOWRCCNSE (consumption) but data={data_model}. "
                    f"Consider using: model=dfm/consumption data=consumption"
                )
            elif 'KOIPALL.G' in target_series and data_model != 'production':
                logger.warning(
                    f"DFM config targets KOIPALL.G (production) but data={data_model}. "
                    f"Consider using: model=dfm/production data=production"
                )
    
    # Select appropriate data class based on config
    data_class_map = {
        'consumption': ConsumptionData,
        'investment': InvestmentData,
        'production': ProductionData,
    }
    
    if data_model not in data_class_map:
        raise ValueError(
            f"Unknown data model: {data_model}. "
            f"Must be one of: {list(data_class_map.keys())}"
        )
    
    # Load preprocessed data (all preprocessing done automatically)
    logger.info(f"Loading and preprocessing {data_model} model data...")
    data_loader = data_class_map[data_model]()
    training_data = data_loader.training_data
    logger.info(f"Training data shape: {training_data.shape}")
    logger.info(f"Number of series: {len(training_data.columns)}")
    
    outputs_dir = get_project_root() / "checkpoints" / data_model / model_name
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {outputs_dir.resolve()}")
    
    # Training
    if cfg.train:
        logger.info(f"Training {model_name} model...")
        
        if model_name in ['dfm', 'ddfm']:
            train_dfm_python_model(
                model_type=model_name,
                config_name="default",
                cfg=cfg,
                data=training_data,  # Pass preprocessed DataFrame
                model_name=model_name,
                horizons=None,
                outputs_dir=outputs_dir,
                model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True),
                data_loader=data_loader,  # Pass data_loader for metadata access
                metadata=data_loader.metadata if hasattr(data_loader, 'metadata') else None
            )
        elif model_name in ['itf', 'itransformer', 'patchtst', 'tft', 'timemixer']:
            train_sktime_model(
                model_type=model_name,
                config_name="default",
                cfg=cfg,
                data=training_data,  # Pass preprocessed DataFrame
                model_name=model_name,
                horizons=None,
                outputs_dir=outputs_dir,
                model_params=OmegaConf.to_container(model_cfg, resolve=True),
                data_loader=data_loader,  # Pass data_loader for datetime index preservation
                metadata=data_loader.metadata if hasattr(data_loader, 'metadata') else None
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    # Forecasting
    if cfg.forecast:
        logger.info(f"Forecasting with {model_name} model...")
        
        checkpoint_path = outputs_dir / "model.pkl"
        if not checkpoint_path.exists():
            checkpoint_path = outputs_dir / "model.zip"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {outputs_dir}")
        
        # Import forecast functions when needed
        forecast, forecast_dfm = _import_forecast_funcs()
        
        if model_name in ['itf', 'itransformer', 'patchtst', 'tft', 'timemixer']:
            forecast(checkpoint_path=checkpoint_path, horizon=horizon, model_type=model_name)
        elif model_name in ['dfm', 'ddfm']:
            forecast_dfm(checkpoint_path=checkpoint_path, horizon=horizon, model_type=model_name)


if __name__ == "__main__":
    main()
