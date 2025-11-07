"""Hydra-enabled script for DFM estimation with experiment management.

This script is for training DFM models, typically run on-demand or periodically.
For regular data ingestion (GitHub Actions), use scripts/ingest_data.py instead.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Optional
import sys
import pandas as pd
import pickle
import numpy as np

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.nowcasting import load_data, dfm
from src.utils import summarize
from scripts.utils import load_model_config_from_hydra, get_db_client


@hydra.main(version_base=None, config_path="../config", config_name="defaults")
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
    model_cfg = load_model_config_from_hydra(cfg.model, script_path=Path(__file__))
    
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
    max_iter = dfm_cfg_dict.get('max_iter', 5000)
    
    # Load data
    if use_database:
        from adapters.adapter_database import load_data_from_db
        sample_start_dt = pd.to_datetime(sample_start) if sample_start else None
        
        # Use latest vintage if not specified
        if vintage is None:
            try:
                from database import get_latest_vintage_id
                client = get_db_client()
                latest_vintage_id = get_latest_vintage_id(client=client)
                if latest_vintage_id:
                    print(f"   Using latest vintage_id: {latest_vintage_id}")
                    vintage = latest_vintage_id  # Use vintage_id instead
                else:
                    raise ValueError("No vintage available in database")
            except Exception as e:
                raise ValueError(f"Must specify vintage_date or ensure database has vintages: {e}")
        
        # Derive config_name from CSV filename if available
        # But only use it if blocks table has data for this config
        config_name = None
        if hasattr(cfg.model, 'config_path') and cfg.model.config_path:
            config_file = Path(cfg.model.config_path)
            if config_file.suffix.lower() == '.csv':
                config_name = config_file.stem.replace('_', '-')
                # Check if blocks table has data for this config_name
                try:
                    from database.helpers import get_series_ids_for_config
                    client = get_db_client()
                    series_ids = get_series_ids_for_config(config_name, client=client)
                    if not series_ids:
                        # No blocks data, don't use config_name
                        config_name = None
                except Exception:
                    # If check fails, don't use config_name
                    config_name = None
        
        data_df, Time, Z_df, series_metadata = load_data_from_db(
            vintage_id=vintage if isinstance(vintage, int) else None,
            vintage_date=vintage if not isinstance(vintage, int) else None,
            config=model_cfg,
            config_name=config_name,
            config_id=config_id,
            sample_start=sample_start_dt,
            strict_mode=strict_mode
        )
        # Convert DataFrame to numpy array for DFM
        X = data_df.values
        Z = Z_df.values if Z_df is not None else None
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
    
    # Pre-flight data validation
    print(f"\n{'='*70}")
    print("Data Validation")
    print(f"{'='*70}\n")
    
    # Check data completeness
    total_obs = X.shape[0] * X.shape[1]
    finite_obs = np.sum(np.isfinite(X))
    completeness_pct = (finite_obs / total_obs * 100) if total_obs > 0 else 0.0
    print(f"Data completeness: {finite_obs}/{total_obs} ({completeness_pct:.1f}%)")
    
    # Check minimum observations per series
    min_obs = 20
    insufficient = [(model_cfg.SeriesID[i], np.sum(np.isfinite(X[:, i]))) 
                     for i in range(len(model_cfg.SeriesID))
                     if np.sum(np.isfinite(X[:, i])) < min_obs]
    
    if insufficient:
        print(f"⚠️  {len(insufficient)} series have <{min_obs} observations:")
        for series_id, count in insufficient[:5]:
            print(f"   - {series_id}: {count} obs")
        if len(insufficient) > 5:
            print(f"   ... and {len(insufficient) - 5} more")
    
    # Block coverage summary
    if hasattr(model_cfg, 'block_names') and model_cfg.block_names:
        blocks = np.array([[s.blocks[i] if hasattr(s, 'blocks') and i < len(s.blocks) else 0
                          for i in range(len(model_cfg.block_names))] 
                         for s in model_cfg.series])
        print(f"\nBlock coverage:")
        for i, block_name in enumerate(model_cfg.block_names):
            block_series = np.where(blocks[:, i] == 1)[0]
            if len(block_series) > 0:
                block_obs = np.sum(np.isfinite(X[:, block_series]), axis=0)
                print(f"   {block_name}: {len(block_series)} series, "
                      f"obs: {np.min(block_obs)}-{np.max(block_obs)} (avg {np.mean(block_obs):.0f})")
    
    if completeness_pct < 50.0:
        print(f"\n⚠️  Warning: Low data completeness may affect estimation")
    
    print(f"{'='*70}\n")
    
    # Run DFM estimation
    Res = dfm(X, model_cfg, threshold=threshold, max_iter=max_iter)
    
    # Save results to pickle file
    output_dir = Path(cfg.get('output_dir', '.'))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'ResDFM.pkl'
    
    with open(output_file, 'wb') as f:
        pickle.dump({'Res': Res, 'Config': model_cfg}, f)
    
    print(f'\nResults saved to {output_file}')
    
    # Save model weights to model/ directory
    base_dir = Path(__file__).parent.parent.parent
    model_dir = base_dir / 'model'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model filename based on vintage and config
    model_filename = f"dfm_{vintage or 'default'}.pkl"
    if config_id:
        model_filename = f"dfm_config_{config_id}_{vintage or 'default'}.pkl"
    
    model_file = model_dir / model_filename
    
    # Save model weights (parameters only, not full results)
    model_weights = {
        'C': Res.C,
        'R': Res.R,
        'A': Res.A,
        'Q': Res.Q,
        'Z_0': Res.Z_0,
        'V_0': Res.V_0,
        'Mx': Res.Mx,
        'Wx': Res.Wx,
        'threshold': threshold,
        'vintage': vintage,
        'config_id': config_id,
        'convergence_iter': getattr(Res, 'convergence_iter', None),
        'log_likelihood': getattr(Res, 'loglik', None),
    }
    
    with open(model_file, 'wb') as f:
        pickle.dump(model_weights, f)
    
    print(f'Model weights saved to {model_file}')
    print(f"Output directory: {output_dir}")
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()

