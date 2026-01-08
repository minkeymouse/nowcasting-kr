from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import hydra
import logging

from src.preprocess import NowcastingData
from src.utils import setup_logging, get_project_root
from src.train.train_sktime import train_sktime_model
from src.train.train_dfm_python import train_dfm_python_model
from src.evalutate.forecast_sktime import forecast
from src.evalutate.forecast_dfm_python import forecast as forecast_dfm

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig) -> None:
    """Main entry point for training and forecasting."""
    setup_logging(log_dir=get_project_root() / "log", force=True)
    
    # Get model name and horizon from config
    model_name = cfg.model.name.lower()
    horizon = cfg.model.get('horizon', cfg.get('horizon', 88))
    
    data = NowcastingData()

    outputs_dir = get_project_root() / "checkpoints" / model_name
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Training
    if cfg.train:
        logger.info(f"Training {model_name} model...")
        
        if model_name in ['dfm', 'ddfm']:
            train_dfm_python_model(
                model_type=model_name,
                config_name="default",
                cfg=cfg,
                data_file="data/data.csv",
                model_name=model_name,
                horizons=None,
                outputs_dir=outputs_dir,
                model_cfg_dict=OmegaConf.to_container(cfg.model, resolve=True)
            )
        elif model_name in ['itf', 'itransformer', 'patchtst', 'tft', 'timemixer', 'lstm', 'chronos']:
            train_sktime_model(
                model_type=model_name,
                config_name="default",
                cfg=cfg,
                data_file="data/data.csv",
                model_name=model_name,
                horizons=None,
                outputs_dir=outputs_dir,
                model_params=OmegaConf.to_container(cfg.model, resolve=True)
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
        
        if model_name in ['itf', 'itransformer', 'patchtst', 'tft', 'timemixer', 'lstm', 'chronos']:
            forecast(checkpoint_path=checkpoint_path, horizon=horizon, model_type=model_name)
        elif model_name in ['dfm', 'ddfm']:
            forecast_dfm(checkpoint_path=checkpoint_path, horizon=horizon, model_type=model_name)


if __name__ == "__main__":
    main()
