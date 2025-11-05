"""Hydra-enabled script for DFM estimation with experiment management.

This script is for training DFM models, typically run on-demand or periodically.
For regular data ingestion (GitHub Actions), use scripts/ingest_data.py instead.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import sys
import pandas as pd
import pickle

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.nowcasting import load_config, load_data, dfm, load_data_from_db, load_model_config_from_hydra
from src.nowcasting.config import ModelConfig, DataConfig, DFMConfig, AppConfig
from src.utils import summarize
import logging

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config", config_name="defaults")
def main(cfg: DictConfig) -> None:
    """Run DFM estimation with Hydra configuration.
    
    Usage:
        python train_dfm.py                          # Use defaults
        python train_dfm.py dfm.threshold=1e-4       # Override threshold
        python train_dfm.py model=us_full            # Use different model
        python train_dfm.py data.vintage=2016-12-23  # Use different vintage
        python train_dfm.py --multirun dfm.threshold=1e-5,1e-4,1e-3  # Sweep
    """
    # Load model configuration - prefer CSV if config_path provided, otherwise use YAML
    # Researchers update migrations/001_initial_spec.csv for model specifications
    model_cfg = load_model_config_from_hydra(cfg.model)
    data_cfg = DataConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    dfm_cfg = DFMConfig(**OmegaConf.to_container(cfg.dfm, resolve=True))
    
    print(f"\n{'='*70}")
    print(f"DFM Estimation - Experiment: {cfg.get('experiment_name', 'default')}")
    print(f"Model Config: Loaded from Hydra (not database)")
    print(f"{'='*70}\n")
    
    # Load data - check if using database or file-based
    sample_start = pd.to_datetime(data_cfg.sample_start) if data_cfg.sample_start else None
    
    if data_cfg.use_database:
        logger.info("Loading data from database...")
        X, Time, Z = load_data_from_db(
            vintage_id=data_cfg.vintage_id,
            vintage_date=data_cfg.vintage,
            config=model_cfg,
            config_id=data_cfg.config_id,
            start_date=data_cfg.start_date,
            end_date=data_cfg.end_date,
            sample_start=sample_start,
            strict_mode=data_cfg.strict_mode
        )
        logger.info("Data loaded successfully from database")
    else:
        logger.info("Loading data from file...")
        # Construct data file path if not provided
        if data_cfg.data_path:
            data_file = Path(data_cfg.data_path)
        else:
            base_dir = Path(__file__).parent.parent.parent
            data_file = base_dir / 'data' / data_cfg.country / f'{data_cfg.vintage}.csv'
            
            if not data_file.exists():
                logger.error(f"CSV file not found: {data_file}")
                logger.error("Please use database mode (data.use_database=true) or provide a CSV file")
                raise FileNotFoundError(f"Data file not found: {data_file}")
        
        X, Time, Z = load_data(data_file, model_cfg, sample_start=sample_start)
        logger.info("Data loaded successfully from file")
    
    # Summarize data
    vintage_label = data_cfg.vintage or f"vintage_id={data_cfg.vintage_id}" if data_cfg.vintage_id else "unknown"
    summarize(X, Time, model_cfg, vintage_label)
    
    # Run DFM estimation
    Res = dfm(X, model_cfg, threshold=dfm_cfg.threshold)
    
    # Save results
    output_dir = Path(cfg.get('output_dir', '.'))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'ResDFM.pkl'
    
    with open(output_file, 'wb') as f:
        pickle.dump({'Res': Res, 'Config': model_cfg}, f)
    
    print(f'\nResults saved to {output_file}')
    print(f"Output directory: {output_dir}")
    
    # Optionally save to database if using database mode
    if data_cfg.use_database:
        try:
            from database import save_model_weights, get_client
            from src.nowcasting.data_loader import _resolve_vintage_id as resolve_vintage
            
            client = get_client()
            
            # Get config_id for saving trained model
            # Note: Model config is from Hydra, but we need config_id to link trained_models
            # For now, we'll use the model config name as identifier or skip config_id
            config_id = data_cfg.config_id  # Can be set manually if needed
            # TODO: Could create a mapping table or use config_name as identifier
            
            # Save model weights if we have config_id and vintage_id
            if config_id:
                resolved_vintage_id = resolve_vintage(
                    vintage_id=data_cfg.vintage_id,
                    vintage_date=data_cfg.vintage,
                    client=client
                )
                if resolved_vintage_id:
                    try:
                        # Extract parameters from DFMResult
                        parameters = {
                            'A': Res.A,
                            'C': Res.C,
                            'Q': Res.Q,
                            'R': Res.R,
                            'Z_0': Res.Z_0,
                            'V_0': Res.V_0,
                            'Mx': Res.Mx,
                            'Wx': Res.Wx
                        }
                        saved_model = save_model_weights(
                            config_id=config_id,
                            vintage_id=resolved_vintage_id,
                            parameters=parameters,
                            threshold=dfm_cfg.threshold,
                            log_likelihood=getattr(Res, 'loglik', None),
                            client=client
                        )
                        if saved_model:
                            logger.info(f"Saved model weights to database: model_id={saved_model.get('model_id')}")
                    except Exception as e:
                        logger.warning(f"Could not save model weights to database: {e}")
        except ImportError:
            logger.warning("Database module not available. Skipping database save.")
        except Exception as e:
            logger.warning(f"Error saving to database: {e}")
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()

