"""Main entry point for nowcasting (GitHub Actions).

This script runs nowcasting using the latest data vintage and generates forecasts.
"""

import sys
import os
import logging
import argparse
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.nowcasting import load_config, load_data, dfm, update_nowcast, load_model_config_from_hydra
from src.nowcasting.config import ModelConfig, DataConfig, DFMConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="defaults")
def main(cfg: DictConfig) -> None:
    """Run nowcasting with Hydra configuration.
    
    This script is designed for GitHub Actions automation.
    It loads the latest data vintage and generates nowcasts.
    
    Usage:
        python run_nowcast.py
        python run_nowcast.py data.vintage_old=2016-12-16 data.vintage_new=2016-12-23
        python run_nowcast.py series=GDPC1 period=2016q4
    """
    # GitHub Actions context
    github_run_id = os.getenv('GITHUB_RUN_ID')
    github_run_url = os.getenv('GITHUB_RUN_URL') or (
        f"{os.getenv('GITHUB_SERVER_URL', '')}/{os.getenv('GITHUB_REPOSITORY', '')}/actions/runs/{github_run_id}"
        if github_run_id else None
    )
    
    if github_run_id:
        logger.info(f"GitHub Actions Run ID: {github_run_id}")
        if github_run_url:
            logger.info(f"Workflow URL: {github_run_url}")
    
    try:
        # Convert OmegaConf to Pydantic models
        # Load model configuration - prefer CSV if config_path provided, otherwise use YAML
        # Researchers update migrations/001_initial_spec.csv for model specifications
        model_cfg = load_model_config_from_hydra(cfg.model)
        data_cfg = DataConfig(**OmegaConf.to_container(cfg.data, resolve=True))
        dfm_cfg = DFMConfig(**OmegaConf.to_container(cfg.dfm, resolve=True))
        
        # Nowcast parameters (can be overridden via CLI)
        series = cfg.get('series', 'GDPC1')
        period = cfg.get('period', '2016q4')
        vintage_old = cfg.get('vintage_old', data_cfg.vintage)
        vintage_new = cfg.get('vintage_new', data_cfg.vintage)
        
        logger.info("=" * 80)
        logger.info(f"Nowcasting - Series: {series}, Period: {period}")
        logger.info(f"Vintage (old): {vintage_old}, Vintage (new): {vintage_new}")
        logger.info("=" * 80)
        
        # Load DFM results or estimate
        base_dir = Path(__file__).parent.parent
        res_file = base_dir / 'ResDFM.pkl'
        
        try:
            logger.info(f"Loading DFM results from {res_file}")
            with open(res_file, 'rb') as f:
                data = pickle.load(f)
                # Check config consistency
                saved_config = data.get('Config', data.get('Spec'))
                if saved_config and 'Res' in data:
                    if hasattr(saved_config, 'SeriesID') and saved_config.SeriesID != model_cfg.SeriesID:
                        logger.warning('Configuration mismatch. Re-estimating...')
                        raise FileNotFoundError
                    Res = data
                else:
                    Res = data.get('Res', data)
            logger.info("DFM results loaded successfully")
        except FileNotFoundError:
            # Re-estimate if file not found or config mismatch
            logger.info('Estimating DFM model...')
            data_file = base_dir / 'data' / data_cfg.country / f'{vintage_new}.csv'
            if not data_file.exists():
                logger.error(f"CSV file not found: {data_file}")
                logger.error("Please use database mode (data.use_database=true) or provide a CSV file")
                raise FileNotFoundError(f"Data file not found: {data_file}")
            X, Time, Z = load_data(data_file, model_cfg)
            Res = dfm(X, model_cfg, threshold=dfm_cfg.threshold)
            with open(res_file, 'wb') as f:
                pickle.dump({'Res': Res, 'Config': model_cfg}, f)
            logger.info(f"DFM model estimated and saved to {res_file}")
        
        # Load datasets for each vintage
        datafile_old = base_dir / 'data' / data_cfg.country / f'{vintage_old}.csv'
        datafile_new = base_dir / 'data' / data_cfg.country / f'{vintage_new}.csv'
        
        if not datafile_old.exists() or not datafile_new.exists():
            logger.error(f"CSV files not found: {datafile_old} or {datafile_new}")
            logger.error("Please use database mode (data.use_database=true) or provide CSV files")
            raise FileNotFoundError("CSV data files not found")
        
        logger.info(f"Loading data from {datafile_old} and {datafile_new}")
        X_old, Time_old, _ = load_data(datafile_old, model_cfg)
        X_new, Time, _ = load_data(datafile_new, model_cfg)
        
        # Update nowcast
        logger.info(f"Running nowcast for {series}, period {period}")
        update_nowcast(X_old, X_new, Time, model_cfg, Res, series, period,
                       vintage_old, vintage_new)
        
        logger.info("=" * 80)
        logger.info("Nowcasting completed successfully")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Fatal error in nowcasting: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

