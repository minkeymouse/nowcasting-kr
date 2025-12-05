"""Training execution module for DFM/DDFM models."""

from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import numpy as np
import sys

# Add project root to path for direct execution
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(project_root / "dfm-python" / "src"))

# Set up paths using centralized utility
try:
    from .utils.path_setup import setup_paths
except ImportError:
    from src.utils.path_setup import setup_paths

setup_paths(include_dfm_python=True, include_src=True, include_app=True)

try:
    from app.utils import (
        ValidationError,
        DEFAULT_DDFM_ENCODER_LAYERS, DEFAULT_DDFM_NUM_FACTORS, DEFAULT_DDFM_EPOCHS
    )
except ImportError:
    from app.utils import (
        ValidationError,
        DEFAULT_DDFM_ENCODER_LAYERS, DEFAULT_DDFM_NUM_FACTORS, DEFAULT_DDFM_EPOCHS
    )

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except ImportError as e:
    raise ImportError(f"Required dependencies not available: {e}")

# Import model wrappers
try:
    from model.dfm import DFM
    from model.ddfm import DDFM
except ImportError:
    try:
        from src.model.dfm import DFM
        from src.model.ddfm import DDFM
    except ImportError as e:
        raise ImportError(
            f"Model wrapper not available: {e}\n"
            "This usually means:\n"
            "  1. dfm-python package is not installed or not in path\n"
            "  2. Path setup failed - check src/utils/path_setup.py\n"
            "  3. Missing dependencies - run: uv pip install -e dfm-python/"
        )


def _detect_model_type_from_config(cfg: DictConfig) -> str:
    """Detect model type from Hydra config."""
    # Check defaults for model override
    defaults = cfg.get('defaults', [])
    for default in defaults:
        if isinstance(default, dict) and default.get('override') == '/model':
            model_override = default.get('_target_', '')
            if 'ddfm' in model_override.lower():
                return "ddfm"
    
    # Check model_type field
    model_type = cfg.get('model_type', '').lower()
    if model_type in ('ddfm', 'deep'):
        return "ddfm"
    
    # Check for DDFM-specific parameters
    ddfm_params = ['encoder_layers', 'epochs', 'learning_rate', 'batch_size']
    if any(key in cfg for key in ddfm_params):
        return "ddfm"
    
    # Default to DFM
    return "dfm"


def _extract_ddfm_params_from_hydra(cfg: DictConfig) -> Dict[str, Any]:
    """Extract DDFM parameters from Hydra config."""
    params = {
        "encoder_layers": DEFAULT_DDFM_ENCODER_LAYERS,
        "num_factors": DEFAULT_DDFM_NUM_FACTORS,
        "epochs": DEFAULT_DDFM_EPOCHS
    }
    
    cfg_dict = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg
    
    if cfg_dict and isinstance(cfg_dict, dict):
        if "encoder_layers" in cfg_dict:
            encoder_layers = cfg_dict["encoder_layers"]
            if isinstance(encoder_layers, list) and all(isinstance(x, int) for x in encoder_layers):
                params["encoder_layers"] = encoder_layers
        
        if "num_factors" in cfg_dict:
            num_factors = cfg_dict["num_factors"]
            if isinstance(num_factors, int) and num_factors > 0:
                params["num_factors"] = num_factors
        
        if "epochs" in cfg_dict:
            epochs = cfg_dict["epochs"]
            if isinstance(epochs, int) and epochs > 0:
                params["epochs"] = epochs
    
    return params


def train(
    config_name: str,
    config_path: Optional[str] = None,
    data_path: Optional[str] = None,
    model_name: Optional[str] = None,
    config_overrides: Optional[list] = None
) -> Dict[str, Any]:
    """Train model with Hydra configuration and save to outputs."""
    config_path = config_path or str(Path(__file__).parent.parent / "config")
    outputs_dir = Path(__file__).parent.parent / "outputs" / "models"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    with hydra.initialize_config_dir(config_dir=config_path, version_base="1.3"):
        cfg = hydra.compose(config_name=config_name, overrides=config_overrides or [])
        
        model_type = _detect_model_type_from_config(cfg)
        model = DDFM(**_extract_ddfm_params_from_hydra(cfg)) if model_type == "ddfm" else DFM()
        
        # Load config - dfm-python expects series info from experiment config
        # Extract series from preprocess.series and merge with model config
        try:
            # Extract series - check multiple possible locations
            # Note: Hydra composes configs, so template content is at top level after composition
            series_list = []
            # print(f"DEBUG: cfg keys: {list(cfg.keys())[:15]}")
            try:
                # After Hydra composition, template content should be at top level
                if 'preprocess' in cfg:
                    print(f"DEBUG: preprocess keys: {list(cfg.preprocess.keys())[:10]}")
                    if 'series' in cfg.preprocess:
                        series_list = OmegaConf.to_container(cfg.preprocess.series, resolve=True)
                        print(f"Found {len(series_list)} series in preprocess.series")
                # Also check if preprocess is nested under experiment
                elif 'experiment' in cfg and 'preprocess' in cfg.experiment:
                    if 'series' in cfg.experiment.preprocess:
                        series_list_raw = OmegaConf.to_container(cfg.experiment.preprocess.series, resolve=True)
                        # Filter out dfm-python unsupported fields (only keep supported ones)
                        dfm_series_fields = ['series_id', 'frequency', 'transformation', 'blocks', 'block']
                        series_list = []
                        for s in series_list_raw:
                            filtered_series = {k: v for k, v in s.items() if k in dfm_series_fields}
                            # Convert 'block' to 'blocks' if needed
                            if 'block' in filtered_series and 'blocks' not in filtered_series:
                                filtered_series['blocks'] = filtered_series.pop('block')
                            series_list.append(filtered_series)
                        # print(f"Found {len(series_list)} series in experiment.preprocess.series (filtered)")
                # Try top-level series (if template has series at root)
                if not series_list and 'series' in cfg:
                    series_list = OmegaConf.to_container(cfg.series, resolve=True)
                    print(f"Found {len(series_list)} series in top-level")
            except Exception as e:
                print(f"Warning: Could not extract series: {e}")
                import traceback
                traceback.print_exc()
            
            # Build model config dict (exclude experiment-specific sections)
            model_cfg_dict = {}
            if series_list:
                model_cfg_dict['series'] = series_list
            
            # Extract blocks - check multiple locations
            blocks_dict = None
            if 'blocks' in cfg:
                blocks_dict = OmegaConf.to_container(cfg.blocks, resolve=True)
                print(f"Found blocks in cfg")
            elif 'experiment' in cfg and 'blocks' in cfg.experiment:
                blocks_dict = OmegaConf.to_container(cfg.experiment.blocks, resolve=True)
                print(f"Found blocks in experiment")
            # Blocks will be loaded from model config file below
            if blocks_dict:
                model_cfg_dict['blocks'] = blocks_dict
            
            # Load model-specific config and merge
            config_path_obj = Path(config_path)
            model_config_path = config_path_obj / "model" / f"{model_type}.yaml"
            if model_config_path.exists():
                import yaml
                with open(model_config_path, 'r') as f:
                    model_yaml = yaml.safe_load(f) or {}
                # Remove experiment-specific keys from model_yaml too
                excluded_keys = ['experiment', 'preprocess', 'forecast', 'evaluation', 'nowcasting', 'ablation', 'output', 'name', 'description', 'model_type', 'target_series', 'defaults']
                model_yaml = {k: v for k, v in model_yaml.items() if k not in excluded_keys}
                # Merge: model config first (provides blocks), then experiment config (experiment overrides)
                # This ensures blocks from model config are included
                # For DDFM, blocks might not be in model config, so use DFM blocks as fallback
                if 'blocks' not in model_cfg_dict and 'blocks' in model_yaml:
                    model_cfg_dict['blocks'] = model_yaml['blocks']
                model_cfg_dict = {**model_yaml, **model_cfg_dict}
                # For DDFM, if still no blocks, try loading from DFM config
                if 'blocks' not in model_cfg_dict and model_type == 'ddfm':
                    dfm_config_path = config_path_obj / "model" / "dfm.yaml"
                    if dfm_config_path.exists():
                        import yaml
                        with open(dfm_config_path, 'r') as f:
                            dfm_yaml = yaml.safe_load(f) or {}
                        if 'blocks' in dfm_yaml:
                            model_cfg_dict['blocks'] = dfm_yaml['blocks']
                            # print(f"Using blocks from DFM config for DDFM: {list(model_cfg_dict.get('blocks', {}).keys())}")
                # if 'blocks' in model_cfg_dict:
                #     print(f"Using blocks: {list(model_cfg_dict.get('blocks', {}).keys())}")
            
            # Remove experiment-specific keys that dfm-python doesn't understand
            excluded_keys = ['experiment', 'preprocess', 'forecast', 'evaluation', 'nowcasting', 'ablation', 'output', 'name', 'description', 'model_type', 'target_series', 'defaults']
            model_cfg_dict = {k: v for k, v in model_cfg_dict.items() if k not in excluded_keys}
            
            if model_cfg_dict and 'series' in model_cfg_dict:
                model.load_config(mapping=model_cfg_dict)
            else:
                raise ValidationError(f"No series found in config. Config: {config_name}")
        except Exception as e:
            # If mapping fails, don't try hydra (it will have same issue)
            raise ValidationError(f"Failed to load config: {e}")
        
        # Get data path - use provided data_path or from config
        data_file = data_path
        if not data_file:
            try:
                data_file = cfg.get('data', {}).get('path')
            except:
                pass
        if not data_file:
            try:
                data_file = cfg.get('preprocess', {}).get('metadata', {}).get('data_path')
            except:
                pass
        if not data_file:
            raise ValidationError(f"data_path required. Config: {config_name}")
        
        # Verify config is loaded before training
        # Note: get_config is called inside train() via create_data_module_impl
        # So we don't need to check here - it will fail there if config not loaded
        
        if model_type == "ddfm":
            model.train(data_path=data_file)
        else:
            max_iter = cfg.get('max_iter', 5000)
            threshold = cfg.get('threshold', 1e-5)
            model.train(data_path=data_file, max_iter=max_iter, threshold=threshold)
        
        result = model.get_result()
        metadata = model.get_metadata()
        
        # Calculate forecast metrics if horizons are specified
        forecast_metrics = {}
        horizons = cfg.get('forecast', {}).get('horizons', [1, 7, 28])
        if horizons:
            try:
                import pandas as pd
                from .evaluation import calculate_metrics_per_horizon
                
                # Load data for evaluation
                data = pd.read_csv(data_file, index_col=0, parse_dates=True)
                target_col = cfg.get('preprocess', {}).get('metadata', {}).get('target_column')
                if not target_col:
                    target_col = cfg.get('target_series')
                
                if target_col and target_col in data.columns:
                    # Use last portion of data for evaluation (simulate out-of-sample)
                    # Model was trained on full data, but we evaluate on last part
                    split_idx = int(len(data) * 0.8)
                    y_train_data = data.iloc[:split_idx]
                    y_test_data = data.iloc[split_idx:]
                    
                    # Generate predictions for max horizon
                    max_horizon = max(horizons)
                    try:
                        pred_result = model.predict(horizon=max_horizon)
                        if isinstance(pred_result, tuple):
                            X_pred, Z_pred = pred_result
                        else:
                            X_pred = pred_result
                        
                        # Convert to DataFrame with proper index
                        if isinstance(X_pred, np.ndarray):
                            # Create index for predictions (starting from last training point)
                            last_train_idx = y_train_data.index[-1]
                            if isinstance(data.index, pd.DatetimeIndex) and isinstance(last_train_idx, pd.Timestamp):
                                freq = pd.infer_freq(data.index) if isinstance(data.index, pd.DatetimeIndex) else 'D'
                                pred_index = pd.date_range(
                                    start=last_train_idx + pd.Timedelta(days=1),
                                    periods=min(len(X_pred), len(y_test_data)),
                                    freq=freq or 'D'
                                )
                            else:
                                pred_index = range(int(last_train_idx) + 1, int(last_train_idx) + 1 + len(X_pred))
                            
                            if X_pred.ndim == 1:
                                y_pred = pd.Series(X_pred[:len(pred_index)], index=pred_index[:len(X_pred)])
                            else:
                                y_pred = pd.DataFrame(X_pred[:len(pred_index)], index=pred_index[:len(X_pred)], columns=data.columns)
                        else:
                            y_pred = X_pred
                        
                        # Align test data with predictions
                        if isinstance(y_pred, pd.DataFrame):
                            y_test_aligned = y_test_data[target_col] if target_col in y_test_data.columns else y_test_data.iloc[:, 0]
                            y_pred_aligned = y_pred[target_col] if target_col in y_pred.columns else y_pred.iloc[:, 0]
                        else:
                            y_test_aligned = y_test_data[target_col] if target_col in y_test_data.columns else y_test_data.iloc[:, 0]
                            y_pred_aligned = y_pred
                        
                        # Align indices
                        common_idx = y_test_aligned.index.intersection(y_pred_aligned.index)
                        if len(common_idx) > 0:
                            y_test_aligned = y_test_aligned.loc[common_idx]
                            y_pred_aligned = y_pred_aligned.loc[common_idx]
                            
                            # Calculate metrics per horizon
                            forecast_metrics = calculate_metrics_per_horizon(
                                y_test_aligned,
                                y_pred_aligned,
                                horizons,
                                y_train=y_train_data[target_col] if target_col in y_train_data.columns else y_train_data.iloc[:, 0]
                            )
                    except Exception as e:
                        print(f"Warning: Could not generate predictions: {e}")
                        import traceback
                        traceback.print_exc()
            except Exception as e:
                print(f"Warning: Could not calculate forecast metrics: {e}")
        
        metrics = {
            'converged': result.converged if hasattr(result, 'converged') else False,
            'num_iter': result.num_iter if hasattr(result, 'num_iter') else 0,
            'loglik': result.loglik if hasattr(result, 'loglik') else np.nan,
            'training_completed': metadata.get('training_completed', True),
            'model_type': metadata.get('model_type', 'dfm'),
            'forecast_metrics': forecast_metrics
        }
        
        final_model_name = model_name or cfg.get('model_name') or f"{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_dir = model.save_to_outputs(
            model_name=final_model_name,
            outputs_dir=outputs_dir,
            config_path=None
        )
        
        return {
            'status': 'completed',
            'model_name': final_model_name,
            'model_dir': str(model_dir),
            'metrics': metrics,
            'result': result,
            'metadata': metadata
        }



def compare_models(
    target_series: str,
    models: List[str],
    horizons: List[int] = [1, 7, 28],
    data_path: Optional[str] = None,
    config_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Train multiple models and compare results."""
    config_dir = config_dir or str(Path(__file__).parent.parent / "config")
    data_path = data_path or str(Path(__file__).parent.parent / "data" / "sample_data.csv")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent / "outputs" / "comparisons" / f"{target_series}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"Comparing models for {target_series}")
    print(f"Models: {', '.join(models)} | Horizons: {horizons}")
    print("=" * 70)
    
    model_results = {}
    failed_models = []
    
    for i, model_name in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] {model_name.upper()}...")
        experiment_id = f"{model_name}_{target_series}"
        
        # Build config overrides - override model in defaults using absolute path
        config_overrides = [f"+defaults.0=/model/{model_name}"]
        if model_name == "ddfm":
            config_overrides.append("+epochs=1")  # Quick test
        
        # Map target series to template name (experiment/ subdirectory)
        template_map = {
            "KOGDP...D": "experiment/kogdp_template",
            "KOCNPER.D": "experiment/kocnper_template",
            "KOGFCF..D": "experiment/kogfcf_template"
        }
        template_name = template_map.get(target_series, f"experiment/{target_series.lower().replace('...', '').replace('.', '')}_template")
        
        try:
            # Try experiment-specific config first
            exp_config_name = f"experiment/{experiment_id}"
            try:
                result = train(
                    config_name=exp_config_name,
                    config_path=config_dir,
                    data_path=data_path,
                    model_name=f"{model_name}_{target_series}_{timestamp}",
                    config_overrides=config_overrides
                )
            except Exception as config_error:
                # Fallback to template if specific config doesn't exist
                if "Cannot find primary config" in str(config_error) or "Could not find" in str(config_error):
                    print(f"  Using template: {template_name}")
                    result = train(
                        config_name=template_name,
                        config_path=config_dir,
                        data_path=data_path,
                        model_name=f"{model_name}_{target_series}_{timestamp}",
                        config_overrides=config_overrides
                    )
                else:
                    raise
            
            model_results[model_name] = result
            print(f"  ✓ Completed")
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            failed_models.append(model_name)
            model_results[model_name] = {'status': 'failed', 'error': str(e), 'metrics': None}
    
    comparison = None
    successful_count = len(model_results) - len(failed_models)
    if successful_count > 0:
        print(f"\nComparing {successful_count} successful models...")
        comparison = _compare_results(model_results, horizons, target_series)
        
        if comparison and comparison.get('metrics_table') is not None:
            from .evaluation import generate_comparison_table
            
            table_path = output_dir / "comparison_table.csv"
            generate_comparison_table(comparison, output_path=str(table_path))
            print(f"  Table: {table_path}")
    
    comparison_data = {
        'target_series': target_series,
        'models': models,
        'horizons': horizons,
        'results': model_results,
        'comparison': comparison,
        'timestamp': datetime.now().isoformat(),
        'output_dir': str(output_dir),
        'failed_models': failed_models
    }
    
    results_file = output_dir / "comparison_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nResults: {results_file}")
    if failed_models:
        print(f"Failed: {', '.join(failed_models)}")
    print("=" * 70)
    
    return comparison_data


def _compare_results(
    results: Dict[str, Dict[str, Any]],
    horizons: List[int],
    target_series: str
) -> Dict[str, Any]:
    """Compare results from multiple models."""
    from .evaluation import compare_multiple_models
    
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train models")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    train_parser = subparsers.add_parser('train', help='Train single model')
    train_parser.add_argument("config_name", help="Experiment config name")
    train_parser.add_argument("--data-path", help="Path to data file")
    train_parser.add_argument("--model-name", help="Model name (default: auto-generated)")
    train_parser.add_argument("--override", action="append", help="Hydra config override")
    
    compare_parser = subparsers.add_parser('compare', help='Compare multiple models')
    compare_parser.add_argument("--target-series", required=True, help="Target series (e.g., KOGDP...D)")
    compare_parser.add_argument("--models", nargs="+", required=True,
                               choices=['arima', 'var', 'vecm', 'dfm', 'ddfm', 'xgboost', 'lightgbm', 'deepar', 'tft'],
                               help="Models to compare")
    compare_parser.add_argument("--horizons", nargs="+", type=int, default=[1, 7, 28],
                               help="Forecast horizons (default: 1 7 28)")
    compare_parser.add_argument("--data-path", help="Path to data file")
    
    args = parser.parse_args()
    
    if args.command == 'train':
        result = train(
            config_name=args.config_name,
            data_path=args.data_path,
            model_name=args.model_name,
            config_overrides=args.override
        )
        print(f"\n✓ Model saved to: {result['model_dir']}")
        
    elif args.command == 'compare':
        result = compare_models(
            target_series=args.target_series,
            models=args.models,
            horizons=args.horizons,
            data_path=args.data_path
        )
        print(f"\n✓ Comparison saved to: {result['output_dir']}")
        if result.get('failed_models'):
            print(f"  Failed: {', '.join(result['failed_models'])}")



