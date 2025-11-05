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

from src.nowcasting import load_config, load_data, dfm
from src.nowcasting.config import ModelConfig, DataConfig, DFMConfig, AppConfig
from src.utils import summarize


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
    # Convert OmegaConf to Pydantic models for validation
    model_cfg = ModelConfig.from_dict(OmegaConf.to_container(cfg.model, resolve=True))
    data_cfg = DataConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    dfm_cfg = DFMConfig(**OmegaConf.to_container(cfg.dfm, resolve=True))
    
    print(f"\n{'='*70}")
    print(f"DFM Estimation - Experiment: {cfg.get('experiment_name', 'default')}")
    print(f"{'='*70}\n")
    
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
    
    # Load data
    sample_start = pd.to_datetime(data_cfg.sample_start) if data_cfg.sample_start else None
    X, Time, Z = load_data(data_file, model_cfg, sample_start=sample_start)
    
    # Summarize data
    summarize(X, Time, model_cfg, data_cfg.vintage)
    
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
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()

