"""Common training functions - unified interface for all model types."""

from pathlib import Path
import sys
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import numpy as np
import pandas as pd

from src.utils import (
    ValidationError,
    get_experiment_cfg,
    get_project_root,
    get_config_path,
    disable_omegaconf_struct,
    resolve_data_path,
    load_and_filter_data,
    get_checkpoint_path,
    detect_model_type,
    setup_paths,
    TRAIN_START,
    TRAIN_END,
    RECENT_START,
    RECENT_END,
    TEST_START,
    TEST_END,
    MODEL_DFM,
    MODEL_DDFM
)

# Setup paths
setup_paths(include_dfm_python=True, include_src=True)
project_root = get_project_root()
from src.train.train_sktime import train_sktime_model
from src.train.train_dfm_python import train_dfm_python_model

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except ImportError as e:
    raise ImportError(f"Required dependencies not available: {e}")

logger = logging.getLogger(__name__)


def _prepare_dfm_recent_data(
    forecaster: Any,
    full_data: pd.DataFrame,
    target_series: str,
    config_path: str
) -> Optional[pd.DataFrame]:
    """Prepare recent data for DFM/DDFM state update."""
    recent_data = full_data[(full_data.index >= RECENT_START) & (full_data.index <= RECENT_END)]

    if len(recent_data) == 0:
        return None

    # Get actual model
    dfm_model = forecaster
    if hasattr(forecaster, '_dfm_model'):
        dfm_model = forecaster._dfm_model
    elif hasattr(forecaster, '_ddfm_model'):
        dfm_model = forecaster._ddfm_model
    else:
        return None

    if not hasattr(dfm_model, 'config') or not hasattr(dfm_model.config, 'series'):
        return None

    # Get trained series and clock
    trained_series_ids = [s.series_id for s in dfm_model.config.series]
    clock = getattr(dfm_model.config, 'clock', 'm')
    
    # Filter to available series
    available_series = [s for s in trained_series_ids if s in recent_data.columns]
    if target_series in recent_data.columns and target_series not in available_series:
        available_series.append(target_series)
    
    if len(available_series) == 0:
        return None
    
    # Apply unified preprocessing (same as training)
    # All models use weekly data - no resampling needed
    from src.train.preprocess import apply_transformations, set_dataframe_frequency
    
    selected_data = recent_data[available_series].dropna(how='all')
    
    # NOTE: For DFM/DDFM models, do NOT apply transformations here
    # Transformations (differencing, etc.) are handled by the preprocessing pipeline
    # (model.preprocess) which is applied in update() method
    # This matches the training data preparation in prepare_multivariate_data
    # Applying transformations here would cause double differencing when pipeline is applied
    # For other models (VAR, etc.), apply transformations here as before
    # Check if this is DFM/DDFM model
    is_dfm_ddfm = (hasattr(dfm_model, '__class__') and 
                   ('dfm' in dfm_model.__class__.__name__.lower() or 
                    'ddfm' in dfm_model.__class__.__name__.lower()))
    
    if not is_dfm_ddfm:
        try:
            selected_data = apply_transformations(selected_data, config_path=config_path, series_ids=available_series)
        except Exception as e:
            raise
    
    # Use weekly data directly (unified preprocessing - same as training)
    processed_data = selected_data
    
    # Maintain series order
    final_order = [s for s in trained_series_ids if s in processed_data.columns]
    final_order.extend([c for c in processed_data.columns if c not in final_order])
    
    result = set_dataframe_frequency(processed_data[final_order])
    
    return result


def train(
    cfg: DictConfig,
    model_name: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    horizons: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Train a single forecasting model using Hydra configuration."""
    config_name = getattr(cfg, '_name_', None) or cfg.get('_name_', 'experiment/consumption_kowrccnse_report')
    checkpoint_model_name = cfg.get('checkpoint_model_name')
    
    if not model_name:
        model_name = cfg.get('model')
    
    if not checkpoint_dir:
        checkpoint_dir = cfg.get('checkpoint_dir', 'checkpoint')
    if not checkpoint_dir:
        raise ValueError("checkpoint_dir must be specified in config or via override.")
    
    # BUG FIX: If model_name contains target_series prefix (from train_model), use it for checkpoint path
    # Otherwise, use checkpoint_model_name from config if available
    if not checkpoint_model_name and model_name and '_' in model_name:
        # Check if model_name looks like "TARGET_MODEL" format
        # Extract potential target prefix (everything before last underscore)
        parts = model_name.split('_')
        if len(parts) >= 2:
            # Use the full model_name as checkpoint_model_name if it has target prefix
            checkpoint_model_name = model_name
    
    outputs_dir = Path(checkpoint_dir)
    if checkpoint_model_name:
        outputs_dir = outputs_dir / checkpoint_model_name
    
    import os
    try:
        outputs_dir.mkdir(parents=True, exist_ok=True)
        if not outputs_dir.exists() or not os.access(outputs_dir, os.W_OK):
            raise ValidationError(f"Cannot create or write to checkpoint directory: {outputs_dir}")
    except (OSError, PermissionError) as e:
        raise ValidationError(f"Cannot create or write to checkpoint directory {outputs_dir}: {e}") from e
    
    disable_omegaconf_struct(cfg)
    
    if model_name:
        model_type = detect_model_type(model_name)
        
        if not model_type:
            raise ValueError(f"Cannot detect model type from name: {model_name}")
    else:
        raise ValueError("model_name is required (set via parameter or cfg.model)")
    
    exp_cfg = get_experiment_cfg(cfg)
    model_overrides_raw = exp_cfg.get('model_overrides', {})
    model_overrides = OmegaConf.to_container(model_overrides_raw, resolve=True) if model_overrides_raw else {}
    if not isinstance(model_overrides, dict):
        model_overrides = {}
    
    model_specific_overrides = model_overrides.get(model_type, {}) or {}
    model_config = model_specific_overrides.copy()
    
    if 'series' in exp_cfg:
        series_raw = exp_cfg.get('series', [])
        model_config['series'] = OmegaConf.to_container(series_raw, resolve=True) if series_raw else []
    
    data_file = exp_cfg.get('data_path')
    if not data_file:
        raise ValidationError(f"data_path required. Config: {config_name}")
    
    # Route to appropriate training module
    if model_type in [MODEL_DFM, MODEL_DDFM]:
        return train_dfm_python_model(
            model_type=model_type,
            config_name=config_name,
            cfg=cfg,
            data_file=data_file,
            model_name=model_name,
            horizons=horizons,
            outputs_dir=outputs_dir,
            model_cfg_dict=model_config
        )
    elif model_type in ['arima', 'var', 'tft', 'lstm', 'chronos']:
        return train_sktime_model(
            model_type=model_type,
            config_name=config_name,
            cfg=cfg,
            data_file=data_file,
            model_name=model_name,
            horizons=horizons,
            outputs_dir=outputs_dir,
            model_params=model_specific_overrides
        )
    else:
        raise ValidationError(f"Unknown model type: {model_type}")


def compare_models(
    target_series: str,
    models: List[str],
    horizons: List[int] = list(range(1, 25)),
    data_path: Optional[str] = None,
    config_dir: Optional[str] = None,
    config_name: Optional[str] = None,
    config_overrides: Optional[List[str]] = None,
    checkpoint_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Train multiple models and compare their performance."""
    config_dir = config_dir or str(get_config_path())
    data_file_path = resolve_data_path(data_path) if data_path else resolve_data_path()
    
    output_dir = project_root / "outputs" / "comparisons" / target_series
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info(f"Comparing models for {target_series}")
    logger.info(f"Models: {', '.join(models)} | Horizons: {horizons}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 70)
    
    model_results = {}
    failed_models = []
    
    if config_name is None:
        report_map = {
            "KOEQUIPTE": "experiment/investment_koequipte_report",
            "KOWRCCNSE": "experiment/consumption_kowrccnse_report",
            "KOIPALL.G": "experiment/production_koipallg_report"
        }
        config_name = report_map.get(target_series, f"experiment/{target_series.lower().replace('...', '').replace('.', '')}_report")
    
    if horizons is None or len(horizons) == 0:
        import hydra
        try:
            with hydra.initialize_config_dir(config_dir=config_dir, version_base="1.3"):
                cfg = hydra.compose(config_name=config_name, overrides=config_overrides or [])
                disable_omegaconf_struct(cfg)
                exp_cfg = get_experiment_cfg(cfg)
                horizons = exp_cfg.get('forecast_horizons', list(range(1, 25)))
            logger.info(f"Extracted horizons from config: {len(horizons)} horizons ({min(horizons)}-{max(horizons)})")
        except Exception as e:
            logger.warning(f"Failed to load horizons from config '{config_name}': {e}. Using default horizons.")
            horizons = list(range(1, 25))
    
    for i, model_name in enumerate(models, 1):
        logger.info(f"[{i}/{len(models)}] {model_name.upper()}...")
        
        if config_overrides is None:
            config_overrides = []
        
        try:
            # Get checkpoint path
            checkpoint_path = get_checkpoint_path(target_series, model_name, checkpoint_dir, config_overrides)
            
            if not checkpoint_path.exists():
                logger.warning(f"Skipping: Checkpoint not found ({checkpoint_path})")
                result = {
                    'status': 'skipped',
                    'model_name': f"{target_series}_{model_name}",
                    'model_type': model_name,
                    'target_series': target_series,
                    'error': f'Checkpoint not found: {checkpoint_path}',
                    'metrics': None
                }
                failed_models.append(model_name)
                model_results[model_name] = result
                continue
            
            # Load checkpoint
            logger.info(f"Loading model from checkpoint: {checkpoint_path}")
            from src.models import load_model_checkpoint
            
            try:
                forecaster, checkpoint_metadata = load_model_checkpoint(checkpoint_path)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint for {target_series}_{model_name}: {e}")
                result = {
                    'status': 'failed',
                    'model_name': f"{target_series}_{model_name}",
                    'error': f'Failed to load checkpoint: {str(e)}',
                    'metrics': None
                }
                failed_models.append(model_name)
                model_results[model_name] = result
                continue
            
            if forecaster is None:
                logger.warning(f"Forecaster is None in checkpoint for {target_series}_{model_name}")
                result = {
                    'status': 'failed',
                    'model_name': f"{target_series}_{model_name}",
                    'error': 'Forecaster is None in checkpoint',
                    'metrics': None
                }
                failed_models.append(model_name)
                model_results[model_name] = result
                continue
            
            # Extract metrics from metadata if available
            metrics = checkpoint_metadata.get('metrics') or {}
            
            result = {
                'status': 'completed',
                'model_name': f"{target_series}_{model_name}",
                'model_dir': str(checkpoint_path.parent),
                'model_type': model_name,
                'target_series': target_series,
                'metrics': metrics,
                'checkpoint_loaded': True
            }
            
            # Evaluate if horizons provided
            if horizons:
                logger.info(f"Evaluating {model_name.upper()} model on {len(horizons)} horizons")
                
                # Load and prepare data
                data_file_path = resolve_data_path(data_path)
                full_data = pd.read_csv(data_file_path, index_col=0, parse_dates=True)
                
                train_data = load_and_filter_data(data_file_path, TRAIN_START, TRAIN_END, resample_freq='ME')
                test_data = load_and_filter_data(data_file_path, TEST_START, TEST_END, resample_freq='ME')
                
                if len(train_data) == 0 or len(test_data) == 0:
                    raise ValidationError(f"Insufficient data: train={len(train_data)}, test={len(test_data)}")
                
                if train_data.index.max() >= test_data.index.min():
                    raise ValidationError(f"Data leakage: train ends {train_data.index.max()}, test starts {test_data.index.min()}")
                
                # VAR: drop NaN rows
                if model_name.lower() == 'var':
                    train_data = train_data.dropna()
                    test_data = test_data.dropna()
                    if len(train_data) == 0 or len(test_data) == 0:
                        raise ValidationError("VAR: Data empty after dropna")
                
                # Prepare recent data for DFM/DDFM
                y_recent = None
                if model_name.lower() in [MODEL_DFM, MODEL_DDFM]:
                    config_path = str(get_config_path())
                    try:
                        y_recent = _prepare_dfm_recent_data(forecaster, full_data, target_series, config_path)
                        if y_recent is not None:
                            logger.info(f"Prepared {len(y_recent)} recent periods for {model_name.upper()} update")
                        else:
                            logger.warning(f"Could not prepare recent data for {model_name.upper()}")
                    except Exception as e:
                        logger.warning(f"Failed to prepare recent data for {model_name.upper()}: {e}")
                        y_recent = None
                
                # Evaluate
                from src.evalutate.evaluate import evaluate_forecaster
                try:
                    forecast_metrics_raw = evaluate_forecaster(
                        forecaster, train_data, test_data, horizons,
                        target_series=target_series, y_recent=y_recent
                    )
                    forecast_metrics = {str(k): v for k, v in forecast_metrics_raw.items()}
                    result['metrics'] = {'forecast_metrics': forecast_metrics}
                    logger.info(f"Successfully evaluated {model_name.upper()}")
                except Exception as e:
                    import traceback
                    error_trace = traceback.format_exc()
                    logger.error(f"Failed to evaluate {model_name.upper()}: {e}")
                    logger.debug(f"Full traceback:\n{error_trace}")
                    result['status'] = 'failed'
                    result['error'] = f"Evaluation failed: {str(e)}"
                    failed_models.append(model_name)
                    model_results[model_name] = result
                    continue
            
            model_results[model_name] = result
            
        except Exception as e:
            logger.error(f"Failed: {str(e)}")
            failed_models.append(model_name)
            model_results[model_name] = {'status': 'failed', 'error': str(e), 'metrics': None}
    
    # Compare results
    comparison = None
    successful_count = len(model_results) - len(failed_models)
    if successful_count > 0:
        logger.info(f"Comparing {successful_count} successful models...")
        comparison = _compare_results(model_results, horizons, target_series)
        
        if comparison and comparison.get('metrics_table') is not None:
            table_path = output_dir / "comparison_table.csv"
            try:
                comparison['metrics_table'].to_csv(table_path, index=False)
                logger.info(f"Table: {table_path}")
            except Exception as e:
                logger.warning(f"Failed to save comparison table: {e}")
    
    # Save results
    comparison_data = {
        'target_series': target_series,
        'models': models,
        'horizons': horizons,
        'results': model_results,
        'comparison': comparison,
        'output_dir': str(output_dir),
        'failed_models': failed_models
    }
    
    results_file = output_dir / "comparison_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"Results saved to: {results_file}")
    
    skipped_models = [name for name, result in model_results.items() if result.get('status') == 'skipped']
    actual_failed = [name for name in failed_models if name not in skipped_models]
    
    if skipped_models:
        logger.warning(f"Skipped (no checkpoint): {', '.join(skipped_models)}")
    if actual_failed:
        logger.error(f"Failed: {', '.join(actual_failed)}")
    logger.info("=" * 70)
    
    return comparison_data


def _compare_results(
    results: Dict[str, Dict[str, Any]],
    horizons: List[int],
    target_series: str
) -> Dict[str, Any]:
    """Compare results from multiple models."""
    from src.evalutate.evaluate import compare_multiple_models
    
    successful_results = {
        name: result for name, result in results.items()
        if result.get('status') == 'completed' and result.get('metrics') is not None
    }
    
    if not successful_results:
        return {
            'metrics_table': None,
            'summary': 'No successful models to compare',
            'best_model_per_horizon': {}
        }
    
    return compare_multiple_models(
        model_results=successful_results,
        horizons=horizons,
        target_series=target_series
    )


def train_model(
    cfg: DictConfig,
    model_name: Optional[str] = None,
    checkpoint_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Train a single model using Hydra config."""
    
    
    exp_cfg = get_experiment_cfg(cfg)
    models = exp_cfg.get('models', [])
    target_series = exp_cfg.get('target_series')
    
    
    
    if model_name is None:
        model_name = models[0] if models else None
        if len(models) > 1:
            logger.warning(f"Config specifies {len(models)} models, training only first: {model_name}")
    
    if not model_name:
        raise ValueError(f"Config must specify at least one model in 'models' list")
    
    checkpoint_model_name = f"{target_series}_{model_name}" if (checkpoint_dir and target_series) else model_name
    
    result = train(cfg=cfg, model_name=checkpoint_model_name, checkpoint_dir=checkpoint_dir)
    
    if checkpoint_dir and 'model_dir' in result:
        result['model_dir'] = str(Path(checkpoint_dir) / checkpoint_model_name)
    
    return result


def compare_models_by_config(
    cfg: DictConfig,
    models_filter: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Compare multiple models from Hydra config."""
    exp_cfg = get_experiment_cfg(cfg)
    config_name = getattr(cfg, '_name_', None) or cfg.get('_name_', 'experiment/consumption_kowrccnse_report')
    config_dir = str(get_config_path())
    
    models_to_run = exp_cfg.get('models', [])
    if models_filter:
        models_to_run = [m for m in models_to_run if m.lower() in [mf.lower() for mf in models_filter]]
        if not models_to_run:
            raise ValueError(f"No models match filter {models_filter}. Available models: {exp_cfg.get('models', [])}")
    
    return compare_models(
        target_series=exp_cfg.get('target_series'),
        models=models_to_run,
        horizons=exp_cfg.get('forecast_horizons', []),
        data_path=exp_cfg.get('data_path'),
        config_dir=config_dir,
        config_name=config_name,
        config_overrides=None,
        checkpoint_dir=exp_cfg.get('checkpoint_dir', 'checkpoints')
    )


@hydra.main(
    config_path=str(get_config_path()),
    config_name="experiment/consumption_kowrccnse_report",
    version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    """CLI entry point for training using Hydra decorator."""
    from src.utils import setup_logging
    
    exp_cfg = get_experiment_cfg(cfg)
    log_dir = exp_cfg.get('log_dir', None)
    if log_dir:
        log_dir = project_root / log_dir if isinstance(log_dir, str) else Path(log_dir)
    else:
        log_dir = project_root / "log"
    
    # Get target_series and model_name for consistent log naming
    target_series = exp_cfg.get('target_series', 'unknown')
    model_name = exp_cfg.get('model', 'unknown')
    
    # Setup logging with model-specific log file (aligned with DFM/DDFM pattern)
    # Pattern: {TARGET}_{MODEL}_{TIMESTAMP}.log (same as run_train.sh)
    # Only create model-specific log if both target_series and model_name are available
    if target_series != 'unknown' and model_name != 'unknown':
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_log_file = log_dir / f"{target_series}_{model_name}_{timestamp}.log"
        setup_logging(log_dir=log_dir, force=True, log_file=model_log_file)
    else:
        # Fallback to default train_{timestamp}.log pattern
        setup_logging(log_dir=log_dir, force=True)
    command = exp_cfg.get('command', 'train')
    model_name = exp_cfg.get('model')
    checkpoint_dir = exp_cfg.get('checkpoint_dir')
    models_filter = exp_cfg.get('models')
    
    config_name = getattr(cfg, '_name_', None) or cfg.get('_name_', 'experiment/consumption_kowrccnse_report')
    config_path = str(get_config_path())
    
    if command == 'train':
        if not checkpoint_dir:
            raise ValueError("checkpoint_dir must be specified in config or via override.")
        result = train_model(cfg=cfg, model_name=model_name, checkpoint_dir=checkpoint_dir)
        print(f"\n✓ Model saved to: {result['model_dir']}")
        
    elif command == 'compare':
        result = compare_models_by_config(cfg=cfg, models_filter=models_filter)
        print(f"\n✓ Comparison saved to: {result['output_dir']}")
        if result.get('failed_models'):
            print(f"  Failed: {', '.join(result['failed_models'])}")
    else:
        raise ValueError(f"Unknown command: {command}. Supported commands: 'train', 'compare'")


if __name__ == "__main__":
    main()
