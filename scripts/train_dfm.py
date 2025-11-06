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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.nowcasting import load_data, dfm, load_config, ModelConfig
from src.utils import summarize


def load_model_config_from_hydra(
    cfg_model: DictConfig,
    use_db: bool = True,
    script_path: Optional[Path] = None
) -> ModelConfig:
    """Load model configuration from database (latest) or CSV/YAML file.
    
    Application-specific config loader for Hydra workflows.
    Priority: Database → CSV → YAML
    
    Parameters
    ----------
    cfg_model : DictConfig
        Hydra model configuration dict
    use_db : bool, default=True
        Whether to try loading from database first
    script_path : Path, optional
        Path to the calling script (for relative path resolution)
        If None, uses current working directory
    
    Returns
    -------
    ModelConfig
        Loaded model configuration
    
    Raises
    ------
    FileNotFoundError
        If config file is specified but not found
    """
    # Try database first (if enabled)
    if use_db:
        try:
            from database import get_client, load_model_config
            
            client = get_client()
            config_name = cfg_model.get('config_name', '001-initial-spec')
            
            db_config = load_model_config(config_name, client=client)
            if db_config and 'config_json' in db_config:
                config_dict = db_config['config_json']
                if 'block_names' not in config_dict and 'block_names' in db_config:
                    config_dict['block_names'] = db_config['block_names']
                return ModelConfig.from_dict(config_dict)
        except (ImportError, Exception):
            pass  # Fall back to file
    
    # Load from CSV or YAML file
    model_config_path = cfg_model.get('config_path')
    if model_config_path:
        config_file = Path(model_config_path)
        
        # Resolve relative paths
        if not config_file.is_absolute():
            # Start from script location or current directory
            if script_path:
                base_dir = script_path.parent.parent
            else:
                base_dir = Path.cwd()
            
            # Try multiple possible locations
            search_paths = [
                base_dir / model_config_path,  # Relative to project root
                Path.cwd() / model_config_path,  # Relative to current dir
            ]
            
            # Also try parent directories
            for parent in [base_dir] + list(base_dir.parents):
                search_paths.append(parent / model_config_path)
            
            # Find first existing path
            for candidate in search_paths:
                if candidate.exists():
                    config_file = candidate
                    break
            else:
                # If not found, use default location relative to script
                if script_path:
                    config_file = script_path.parent.parent / model_config_path
                else:
                    config_file = Path(model_config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(
                f"Model config file not found: {config_file}\n"
                f"Researchers should update: src/spec/001_initial_spec.csv"
            )
        
        # Load config from file
        model_config = load_config(config_file)
        
        # If loading from CSV and use_db is enabled, save blocks to database
        if use_db and config_file.suffix.lower() == '.csv':
            try:
                from adapters.database import save_blocks_to_db
                
                # Derive config_name from CSV filename
                # Example: '001_initial_spec.csv' → '001-initial-spec'
                config_name = config_file.stem.replace('_', '-')
                
                # Save blocks to database
                save_blocks_to_db(model_config, config_name)
            except (ImportError, Exception) as e:
                # Log warning but don't fail - block saving is optional
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Could not save blocks to database for {config_file.name}: {e}. "
                    f"Continuing without saving blocks."
                )
        
        return model_config
    else:
        # Fallback to YAML config (convert DictConfig to dict, then to ModelConfig)
        return ModelConfig.from_dict(OmegaConf.to_container(cfg_model, resolve=True))


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
    
    # Load data
    if use_database:
        from adapters.database import load_data_from_db
        sample_start_dt = pd.to_datetime(sample_start) if sample_start else None
        
        # Use latest vintage if not specified
        if vintage is None:
            try:
                from database import get_latest_vintage_id
                from adapters.database import _get_db_client
                client = _get_db_client()
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
                    from adapters.database import _get_db_client
                    from database.helpers import get_series_ids_for_config
                    client = _get_db_client()
                    series_ids = get_series_ids_for_config(config_name, client=client)
                    if not series_ids:
                        # No blocks data, don't use config_name
                        config_name = None
                except Exception:
                    # If check fails, don't use config_name
                    config_name = None
        
        X, Time, Z, series_metadata = load_data_from_db(
            vintage_id=vintage if isinstance(vintage, int) else None,
            vintage_date=vintage if not isinstance(vintage, int) else None,
            config=model_cfg,
            config_name=config_name,
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

