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

from src.nowcasting import load_config, load_data, dfm, load_model_config_from_hydra
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
    # Load model configuration - prefer CSV if config_path provided, otherwise use YAML
    # Researchers update src/spec/001_initial_spec.csv for model specifications
    model_cfg = load_model_config_from_hydra(cfg.model)
    
    # Load data and DFM configs (use OmegaConf directly, no Pydantic classes needed)
    data_cfg_dict = OmegaConf.to_container(cfg.data, resolve=True)
    dfm_cfg_dict = OmegaConf.to_container(cfg.dfm, resolve=True)
    
    print(f"\n{'='*70}")
    print(f"DFM Estimation - Experiment: {cfg.get('experiment_name', 'default')}")
    print(f"{'='*70}\n")
    
    # Extract settings from config dicts
    use_database = data_cfg_dict.get('use_database', True)
    data_path = data_cfg_dict.get('data_path')
    country = data_cfg_dict.get('country', 'KR')
    vintage = data_cfg_dict.get('vintage')
    sample_start = data_cfg_dict.get('sample_start')
    config_id = data_cfg_dict.get('config_id')
    strict_mode = data_cfg_dict.get('strict_mode', False)
    threshold = dfm_cfg_dict.get('threshold', 1e-5)
    
    # Load data
    if use_database:
        from src.nowcasting import load_data_from_db
        sample_start_dt = pd.to_datetime(sample_start) if sample_start else None
        X, Time, Z = load_data_from_db(
            vintage_date=vintage,
            config=model_cfg,
            config_id=config_id,
            sample_start=sample_start_dt,
            strict_mode=strict_mode
        )
    else:
        # File-based loading
        if data_path:
            data_file = Path(data_path)
        else:
            base_dir = Path(__file__).parent.parent.parent
            data_file = base_dir / 'data' / country / f'{vintage}.csv'
        
        if not data_file.exists():
            raise FileNotFoundError(
                f"CSV file not found: {data_file}\n"
                f"Use database mode (data.use_database=true) or provide CSV file"
            )
        
        sample_start_dt = pd.to_datetime(sample_start) if sample_start else None
        X, Time, Z = load_data(data_file, model_cfg, sample_start=sample_start_dt)
    
    # Summarize data
    summarize(X, Time, model_cfg, vintage)
    
    # Run DFM estimation
    Res = dfm(X, model_cfg, threshold=threshold)
    
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

