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

from dfm_python import load_data, dfm
from src.utils import summarize
from scripts.utils import load_model_config_from_hydra, get_db_client


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> None:
    """Run DFM estimation with Hydra configuration.
    
    Usage:
        python train_dfm.py                          # Use defaults
        python train_dfm.py dfm.threshold=1e-4       # Override threshold
        python train_dfm.py model=us_full            # Use different model
        python train_dfm.py data.vintage=2016-12-23  # Use different vintage
        python train_dfm.py --multirun dfm.threshold=1e-5,1e-4,1e-3  # Sweep
    """
    # Load model configuration with priority: CSV from DB storage → Hydra YAML
    # Application-specific: Try database storage first, then fallback to Hydra YAML
    try:
        model_cfg = load_model_config_from_hydra(cfg.model, script_path=Path(__file__))
    except (ValueError, FileNotFoundError) as e:
        # If CSV loading fails, try Hydra YAML structure
        logger.info("CSV config not found, trying Hydra YAML structure...")
        try:
            from dfm_python.config import DFMConfig
            from omegaconf import OmegaConf
            
            # Combine series and model configs
            series_dict = OmegaConf.to_container(cfg.series, resolve=True) if hasattr(cfg, 'series') else {}
            model_dict = OmegaConf.to_container(cfg.model, resolve=True) if hasattr(cfg, 'model') else {}
            
            # Merge: series from cfg.series.series, blocks from cfg.model.blocks
            combined_dict = {
                'series': series_dict.get('series', {}),
                'blocks': model_dict.get('blocks', {}),
                'block_names': model_dict.get('block_names', None),
                'factors_per_block': model_dict.get('factors_per_block', None)
            }
            
            # Try loading from combined structure
            model_cfg = DFMConfig.from_dict(combined_dict)
            logger.info("✅ Loaded config from Hydra YAML structure")
        except Exception as e2:
            logger.error(f"Failed to load config from CSV storage and Hydra YAML: {e2}")
            raise
    
    # Merge DFM estimation parameters from Hydra into model config
    dfm_cfg_dict = OmegaConf.to_container(cfg.dfm, resolve=True)
    if dfm_cfg_dict:
        # Update model_cfg with estimation parameters from Hydra
        for key in ['ar_lag', 'threshold', 'max_iter', 'nan_method', 'nan_k', 'factors_per_block']:
            if key in dfm_cfg_dict and dfm_cfg_dict[key] is not None:
                setattr(model_cfg, key, dfm_cfg_dict[key])
    
    # Load data config (use OmegaConf directly, no Pydantic classes needed)
    data_cfg_dict = OmegaConf.to_container(cfg.data, resolve=True)
    
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
    
    # Initialize database client (reused throughout script if needed)
    db_client = None
    if use_database:
        try:
            db_client = get_db_client()
        except Exception:
            pass  # Will handle errors in specific operations
    
    # Load data
    if use_database:
        from adapters.adapter_database import load_data_from_db
        sample_start_dt = pd.to_datetime(sample_start) if sample_start else None
        
        # Use latest vintage if not specified
        if vintage is None:
            try:
                from database import get_latest_vintage_id
                if db_client is None:
                    db_client = get_db_client()
                latest_vintage_id = get_latest_vintage_id(client=db_client)
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
                    if db_client is None:
                        db_client = get_db_client()
                    series_ids = get_series_ids_for_config(config_name, client=db_client)
                    if not series_ids:
                        # No blocks data, don't use config_name
                        config_name = None
                except Exception:
                    # If check fails, don't use config_name
                    config_name = None
        
        X, Time, Z = load_data_from_db(
            vintage_id=vintage if isinstance(vintage, int) else None,
            vintage_date=vintage if not isinstance(vintage, int) else None,
            config=model_cfg,
            config_name=config_name,
            config_id=config_id,
            sample_start=sample_start_dt,
            strict_mode=strict_mode
        )
        # X and Z are already numpy arrays, no conversion needed
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
    # Convert vintage to string if it's an int (vintage_id)
    vintage_str = str(vintage) if vintage is not None else None
    summarize(X, Time, model_cfg, vintage_str)
    
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
    # Ensure we don't exceed X dimensions (in case of missing series)
    n_series = min(len(model_cfg.SeriesID), X.shape[1])
    insufficient = [(model_cfg.SeriesID[i], np.sum(np.isfinite(X[:, i]))) 
                     for i in range(n_series)
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
    
    # Run DFM estimation - config already contains all parameters (merged ModelConfig + DFMConfig)
    # Override threshold and max_iter if provided via Hydra command line
    Res = dfm(X, model_cfg, threshold=threshold, max_iter=max_iter)
    
    # Save results to pickle file (legacy support)
    output_dir = Path(cfg.get('output_dir', '.'))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'ResDFM.pkl'
    
    with open(output_file, 'wb') as f:
        pickle.dump({'Res': Res, 'Config': model_cfg}, f)
    
    print(f'\nResults saved to {output_file}')
    
    # Save factors, factor_values, and factor_loadings to database for frontend visualization
    if use_database:
        try:
            from adapters.adapter_database import save_factors_to_db
            from database import get_vintage
            
            # Generate model_id from config_id or hash
            if config_id:
                model_id = int(config_id) if isinstance(config_id, (int, str)) and str(config_id).isdigit() else hash(str(config_id)) % 2147483647
            else:
                model_id = hash(str(model_cfg.SeriesID)) % 2147483647
            
            # Get vintage_id for factor_values
            vintage_id = None
            if isinstance(vintage, int):
                vintage_id = vintage
            else:
                if db_client is None:
                    db_client = get_db_client()
                vintage_info = get_vintage(
                    vintage_id=None,
                    vintage_date=vintage,
                    client=db_client
                )
                if vintage_info:
                    vintage_id = vintage_info['vintage_id']
            
            if vintage_id:
                save_factors_to_db(
                    Res=Res,
                    model_id=model_id,
                    config=model_cfg,
                    vintage_id=vintage_id,
                    Time=Time,
                    client=db_client if db_client else get_db_client()
                )
                print(f'✅ Saved factors, factor_values, and factor_loadings to database for model_id={model_id}')
            else:
                print(f'⚠️  Could not resolve vintage_id for {vintage}. Skipping factor save.')
        except ImportError:
            print('⚠️  Warning: Database module not available. Cannot save factors to database.')
        except Exception as e:
            print(f'⚠️  Warning: Failed to save factors to database: {e}')
    
    # Save model weights to Supabase storage
    try:
        from adapters.adapter_database import upload_model_weights_to_storage
        
        # Create model filename based on vintage and config
        if isinstance(vintage, int):
            # If vintage is an ID, get the date
            try:
                from database import get_vintage
                if db_client is None:
                    db_client = get_db_client()
                vintage_info = get_vintage(vintage_id=vintage, client=db_client)
                vintage_str = vintage_info.get('vintage_date', str(vintage)) if vintage_info else str(vintage)
            except Exception:
                vintage_str = str(vintage)
        else:
            vintage_str = str(vintage) if vintage else 'default'
        
        model_filename = f"dfm_{vintage_str}.pkl"
        if config_id:
            model_filename = f"dfm_config_{config_id}_{vintage_str}.pkl"
        
        # Prepare model weights (parameters only, not full results)
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
            'vintage': vintage_str,
            'config_id': config_id,
            'convergence_iter': getattr(Res, 'convergence_iter', None),
            'log_likelihood': getattr(Res, 'loglik', None),
        }
        
        # Upload to Supabase storage
        storage_url = upload_model_weights_to_storage(
            model_weights=model_weights,
            filename=model_filename,
            bucket_name="model-weights",
            client=db_client if db_client else get_db_client()
        )
        
        print(f'Model weights uploaded to Supabase storage: {storage_url}')
        
    except ImportError:
        print('⚠️  Warning: Could not upload model weights to storage (database module not available)')
        # Fallback to local save
        base_dir = Path(__file__).parent.parent.parent
        model_dir = base_dir / 'model'
        model_dir.mkdir(parents=True, exist_ok=True)
        model_filename = f"dfm_{vintage or 'default'}.pkl"
        if config_id:
            model_filename = f"dfm_config_{config_id}_{vintage or 'default'}.pkl"
        model_file = model_dir / model_filename
        model_weights = {
            'C': Res.C, 'R': Res.R, 'A': Res.A, 'Q': Res.Q,
            'Z_0': Res.Z_0, 'V_0': Res.V_0, 'Mx': Res.Mx, 'Wx': Res.Wx,
            'threshold': threshold, 'vintage': vintage, 'config_id': config_id,
            'convergence_iter': getattr(Res, 'convergence_iter', None),
            'log_likelihood': getattr(Res, 'loglik', None),
        }
        with open(model_file, 'wb') as f:
            pickle.dump(model_weights, f)
        print(f'Model weights saved locally to {model_file}')
    except Exception as e:
        print(f'⚠️  Warning: Failed to upload model weights to storage: {e}')
        print('   Continuing with local save only...')
    
    print(f"Output directory: {output_dir}")
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()

