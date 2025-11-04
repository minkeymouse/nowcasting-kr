"""Hydra-enabled script for nowcasting with experiment management."""

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import sys
import pandas as pd
import pickle

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.nowcasting import load_config, load_data, dfm, update_nowcast
from src.nowcasting.config import ModelConfig, DataConfig, DFMConfig


@hydra.main(version_base=None, config_path="../../config", config_name="defaults")
def main(cfg: DictConfig) -> None:
    """Run nowcasting with Hydra configuration.
    
    Usage:
        python run_nowcast.py
        python run_nowcast.py data.vintage_old=2016-12-16 data.vintage_new=2016-12-23
        python run_nowcast.py series=GDPC1 period=2016q4
    """
    # Convert OmegaConf to Pydantic models
    model_cfg = ModelConfig.from_dict(OmegaConf.to_container(cfg.model, resolve=True))
    data_cfg = DataConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    
    # Nowcast parameters (can be overridden via CLI)
    series = cfg.get('series', 'GDPC1')
    period = cfg.get('period', '2016q4')
    vintage_old = cfg.get('vintage_old', data_cfg.vintage)
    vintage_new = cfg.get('vintage_new', data_cfg.vintage)
    
    print(f"\n{'='*70}")
    print(f"Nowcasting - Series: {series}, Period: {period}")
    print(f"{'='*70}\n")
    
    # Load DFM results or estimate
    base_dir = Path(__file__).parent.parent.parent
    res_file = base_dir / 'ResDFM.pkl'
    
    try:
        with open(res_file, 'rb') as f:
            data = pickle.load(f)
            # Check config consistency
            saved_config = data.get('Config', data.get('Spec'))
            if saved_config and 'Res' in data:
                if hasattr(saved_config, 'SeriesID') and saved_config.SeriesID != model_cfg.SeriesID:
                    print('Warning: Configuration mismatch. Re-estimating...')
                    raise FileNotFoundError
                Res = data
            else:
                Res = data.get('Res', data)
    except FileNotFoundError:
        # Re-estimate if file not found or config mismatch
        print('Estimating DFM model...')
        data_file = base_dir / 'data' / data_cfg.country / f'{vintage_new}.xls'
        X, Time, Z = load_data(data_file, model_cfg, load_excel=data_cfg.load_excel)
        dfm_cfg = DFMConfig(**OmegaConf.to_container(cfg.dfm, resolve=True))
        Res = dfm(X, model_cfg, threshold=dfm_cfg.threshold)
        with open(res_file, 'wb') as f:
            pickle.dump({'Res': Res, 'Config': model_cfg}, f)
    
    # Load datasets for each vintage
    datafile_old = base_dir / 'data' / data_cfg.country / f'{vintage_old}.xls'
    datafile_new = base_dir / 'data' / data_cfg.country / f'{vintage_new}.xls'
    
    X_old, Time_old, _ = load_data(datafile_old, model_cfg, load_excel=data_cfg.load_excel)
    X_new, Time, _ = load_data(datafile_new, model_cfg, load_excel=data_cfg.load_excel)
    
    # Update nowcast
    update_nowcast(X_old, X_new, Time, model_cfg, Res, series, period,
                   vintage_old, vintage_new)
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()

