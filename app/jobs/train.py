"""Job 2: Train DFM model and save weights to storage.

This job:
- Queries database for latest data vintage
- Loads model configuration (from DB storage CSV or Hydra YAML)
- Trains DFM model using EM algorithm
- Saves model weights to Supabase storage bucket
- Saves factors to database

Usage:
    python -m app.jobs.train --config-name=test series=test_series
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Optional
import sys
import os
import pandas as pd
import pickle
import numpy as np
import logging

# Add project root to path (script is in app/jobs/ directory)
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Optional dotenv import (for local development only)
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False
    load_dotenv = None

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env.local file (local development only)
# In GitHub Actions, environment variables come from secrets
env_loaded = False

if HAS_DOTENV and not os.getenv('GITHUB_ACTIONS'):
    # Use .env.local for local development
    env_locations = [
        project_root / '.env.local',
        Path('.env.local'),  # Current directory
    ]
    
    for env_path in env_locations:
        if env_path.exists():
            load_dotenv(env_path, override=True)
            logger.info(f"✅ Loaded environment from: {env_path}")
            env_loaded = True
            break

if not env_loaded and os.getenv('GITHUB_ACTIONS'):
    logger.info("Running in GitHub Actions - using environment variables from secrets")
    env_loaded = True  # In GitHub Actions, we use secrets, so consider it "loaded"
elif not env_loaded and not HAS_DOTENV:
    logger.info("dotenv not available - using environment variables from system")
    env_loaded = True  # If no dotenv, we rely on system env vars

from dfm_python import load_data, dfm
from app.utils import summarize
from app.utils import (
    load_model_config_with_hydra_fallback,
    get_db_client,
    get_latest_vintage_with_fallback,
    extract_hydra_config_dicts
)
import logging

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../app/config", config_name="default")
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
    model_cfg = load_model_config_with_hydra_fallback(cfg, script_path=Path(__file__))
    
    # Extract and merge config dicts
    config_dicts = extract_hydra_config_dicts(cfg, sections=['data', 'dfm'])
    data_cfg_dict = config_dicts['data']
    dfm_cfg_dict = config_dicts['dfm']
    
    # Merge DFM estimation parameters from Hydra into model config
    if dfm_cfg_dict:
        # Core DFM parameters
        core_keys = ['ar_lag', 'threshold', 'max_iter', 'nan_method', 'nan_k', 'clock']
        # Numerical stability parameters (dfm-python 0.1.5+)
        stability_keys = [
            'clip_ar_coefficients', 'ar_clip_min', 'ar_clip_max', 'warn_on_ar_clip',
            'clip_data_values', 'data_clip_threshold', 'warn_on_data_clip',
            'use_regularization', 'regularization_scale', 'min_eigenvalue', 'max_eigenvalue', 'warn_on_regularization',
            'use_damped_updates', 'damping_factor', 'warn_on_damped_update'
        ]
        for key in core_keys + stability_keys:
            if key in dfm_cfg_dict and dfm_cfg_dict[key] is not None:
                setattr(model_cfg, key, dfm_cfg_dict[key])
    
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
        from app.adapters.adapter_database import load_data_from_db
        sample_start_dt = pd.to_datetime(sample_start) if sample_start else None
        
        # Use latest vintage if not specified
        if vintage is None:
            try:
                if db_client is None:
                    db_client = get_db_client()
                result = get_latest_vintage_with_fallback(client=db_client)
                if result:
                    latest_vintage_id, vintage_info = result
                    print(f"   Using latest vintage_id: {latest_vintage_id}")
                    vintage = latest_vintage_id
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
                    from app.database.helpers import get_series_ids_for_config
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
    
    # Validate minimum data requirements before estimation
    # Check if we have enough series with sufficient data
    series_with_data = sum(1 for i in range(X.shape[1]) if np.sum(np.isfinite(X[:, i])) >= 20)
    min_series_required = max(3, len(model_cfg.block_names))  # At least 3 series or number of blocks
    
    if series_with_data < min_series_required:
        error_msg = (
            f"\n❌ ERROR: Insufficient data for DFM estimation\n"
            f"   Series with ≥20 observations: {series_with_data}\n"
            f"   Minimum required: {min_series_required}\n"
            f"   Data completeness: {completeness_pct:.1f}%\n\n"
            f"Possible solutions:\n"
            f"  1. Use a different vintage that has more data\n"
            f"  2. Check why series are missing from vintage {vintage}\n"
            f"  3. Reduce the number of series in the config\n"
            f"  4. Run data ingestion to populate missing series\n"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Check if any series have data (to avoid empty array issues)
    if series_with_data == 0:
        error_msg = (
            f"\n❌ ERROR: No series have sufficient data\n"
            f"   All {X.shape[1]} series have <20 observations\n"
            f"   Cannot estimate DFM model\n"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
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
            from app.adapters.adapter_database import save_factors_to_db
            from app.database import get_vintage
            
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
                
                # Cleanup old models (keep only latest 1 training run)
                # This prevents database from growing unbounded
                try:
                    from app.adapters.adapter_database import cleanup_old_models
                    cleanup_result = cleanup_old_models(
                        keep_latest=1,
                        client=db_client if db_client else get_db_client()
                    )
                    if cleanup_result['deleted_count'] > 0:
                        print(f'✅ Cleaned up old models: kept {len(cleanup_result["kept_models"])} latest, deleted {cleanup_result["deleted_count"]} old models')
                        print(f'   (Deleted {cleanup_result["deleted_factors"]} factors, ~{cleanup_result["deleted_factor_values"]} factor_values, ~{cleanup_result["deleted_factor_loadings"]} factor_loadings)')
                        logger.info(f"Cleaned up models: kept {len(cleanup_result['kept_models'])} latest, deleted {cleanup_result['deleted_count']}")
                    else:
                        print(f'✅ No cleanup needed: only {cleanup_result["total_models"]} model(s) found (keeping all)')
                except Exception as cleanup_error:
                    logger.error(f"Failed to cleanup old models: {cleanup_error}", exc_info=True)
                    print(f'⚠️  Warning: Failed to cleanup old models: {cleanup_error}')
            else:
                print(f'⚠️  Could not resolve vintage_id for {vintage}. Skipping factor save.')
        except ImportError:
            print('⚠️  Warning: Database module not available. Cannot save factors to database.')
        except Exception as e:
            print(f'⚠️  Warning: Failed to save factors to database: {e}')
    
    # Save model weights to Supabase storage
    try:
        from app.adapters.adapter_database import upload_model_weights_to_storage, cleanup_old_model_weights
        
        # Create model filename based on vintage and config
        if isinstance(vintage, int):
            # If vintage is an ID, get the date
            try:
                from app.database import get_vintage
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
        
        # Cleanup old model weights (keep only latest 3)
        try:
            cleanup_result = cleanup_old_model_weights(
                keep_latest=3,
                bucket_name="model-weights",
                client=db_client if db_client else get_db_client()
            )
            if cleanup_result['deleted_count'] > 0:
                print(f'Cleaned up old model weights: kept {len(cleanup_result["kept_files"])} latest, deleted {cleanup_result["deleted_count"]} old files')
                logger.info(f"Cleaned up model weights: kept {len(cleanup_result['kept_files'])} latest, deleted {cleanup_result['deleted_count']} old")
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup old model weights: {cleanup_error}")
        
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

