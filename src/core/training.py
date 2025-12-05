"""Training execution module for DFM/DDFM models."""

from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import numpy as np
import sys

# Set up paths using centralized utility
from ..utils.config_parser import setup_paths
setup_paths(include_dfm_python=True, include_src=True, include_app=True)

from ..utils.config_parser import (
    ValidationError,
    DEFAULT_DDFM_ENCODER_LAYERS, DEFAULT_DDFM_NUM_FACTORS, DEFAULT_DDFM_EPOCHS
)

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except ImportError as e:
    raise ImportError(f"Required dependencies not available: {e}")

# Import model wrappers
from ..model.dfm import DFM
from ..model.ddfm import DDFM


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


def _train_forecaster(
    model_type: str,
    config_name: str,
    cfg: DictConfig,
    data_file: str,
    model_name: Optional[str],
    horizons: Optional[List[int]],
    outputs_dir: Path,
    model_cfg_dict: Optional[dict] = None
) -> Dict[str, Any]:
    """Train any model using sktime forecaster interface.
    
    This function provides a unified interface for training all models (ARIMA, VAR, DFM, DDFM)
    using sktime forecasters, ensuring consistent training, evaluation, and prediction.
    
    Parameters
    ----------
    model_type : str
        Model type: 'arima' or 'var'
    config_name : str
        Config name
    cfg : DictConfig
        Hydra config
    data_file : str
        Path to data file
    model_name : Optional[str]
        Model name for saving
    horizons : Optional[List[int]]
        Forecast horizons
    outputs_dir : Path
        Output directory
        
    Returns
    -------
    Dict[str, Any]
        Training results in same format as DFM/DDFM
    """
    try:
        import pandas as pd
        import numpy as np
        from sktime.forecasting.arima import ARIMA as SktimeARIMA
        from sktime.forecasting.var import VAR as SktimeVAR
        from sktime.transformations.series.impute import Imputer
        try:
            from src.model.sktime_forecaster import DFMForecaster, DDFMForecaster
        except ImportError:
            try:
                from model.sktime_forecaster import DFMForecaster, DDFMForecaster
            except ImportError:
                from ..model.sktime_forecaster import DFMForecaster, DDFMForecaster
    except ImportError as e:
        raise ImportError(f"sktime is required for {model_type} models. Install with: pip install sktime[forecasting]") from e
    
    # Load data
    data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    # Get target series
    source_cfg = cfg.experiment if 'experiment' in cfg else cfg
    target_col = None
    if 'experiment' in cfg and 'target_series' in cfg.experiment:
        target_col = cfg.experiment.target_series
    elif 'target_series' in cfg:
        target_col = cfg.target_series
    
    if not target_col or target_col not in data.columns:
        raise ValidationError(f"target_series '{target_col}' not found in data. Available columns: {list(data.columns)}")
    
    # Get model parameters from config
    model_params = {}
    if 'model_overrides' in source_cfg:
        model_overrides_dict = OmegaConf.to_container(source_cfg.model_overrides, resolve=True)
        if isinstance(model_overrides_dict, dict) and model_type in model_overrides_dict:
            model_params = model_overrides_dict[model_type] or {}
    
    # Create and train model
    print(f"\n{'='*70}")
    print(f"Training {model_type.upper()} model")
    print(f"{'='*70}")
    print(f"Config: {config_name}")
    print(f"Data: {data_file}")
    
    if model_type == 'dfm':
        # DFM using sktime forecaster
        max_iter = model_params.get('max_iter', 5000)
        threshold = model_params.get('threshold', 1e-5)
        
        # Prepare config_dict for DFMForecaster
        config_dict = model_cfg_dict if model_cfg_dict else {}
        if not config_dict or 'series' not in config_dict:
            raise ValidationError(f"No series found in config. Config: {config_name}")
        
        forecaster = DFMForecaster(
            config_dict=config_dict,
            max_iter=max_iter,
            threshold=threshold
        )
        
        # DFM uses multivariate data (all series from config)
        # IMPORTANT: Filter data columns to match filtered_series_list from config_dict
        # This ensures data shape matches the series IDs in the config after frequency hierarchy filtering
        filtered_series_ids = []
        if config_dict and 'series' in config_dict:
            filtered_series_ids = [s.get('series_id', s) if isinstance(s, dict) else s 
                                  for s in config_dict['series']]
        
        if filtered_series_ids:
            # Use only series IDs from filtered config
            available_series = [s for s in filtered_series_ids if s in data.columns]
            if len(available_series) > 0:
                y_train = data[available_series].dropna()
            else:
                raise ValidationError(f"DFM: No filtered series found in data columns. Filtered series IDs: {filtered_series_ids[:5]}..., Data columns: {list(data.columns)[:5]}...")
        elif 'experiment' in cfg and 'series' in cfg.experiment:
            series_ids = OmegaConf.to_container(cfg.experiment.series, resolve=True)
            if isinstance(series_ids, list):
                available_series = [s for s in series_ids if s in data.columns]
                if len(available_series) > 0:
                    y_train = data[available_series].dropna()
                else:
                    y_train = data.select_dtypes(include=[np.number]).dropna()
        else:
            y_train = data.select_dtypes(include=[np.number]).dropna()
        
        print(f"Max iterations: {max_iter}, Threshold: {threshold}, Series: {y_train.shape[1]} (filtered from config)")
        
    elif model_type == 'ddfm':
        # DDFM using sktime forecaster
        epochs = model_params.get('epochs', 100)
        encoder_layers = model_params.get('encoder_layers', [64, 32])
        num_factors = model_params.get('num_factors', 1)
        learning_rate = model_params.get('learning_rate', 0.001)
        batch_size = model_params.get('batch_size', 32)
        
        # Prepare config_dict for DDFMForecaster
        config_dict = model_cfg_dict if model_cfg_dict else {}
        if not config_dict or 'series' not in config_dict:
            raise ValidationError(f"No series found in config. Config: {config_name}")
        
        forecaster = DDFMForecaster(
            config_dict=config_dict,
            encoder_layers=encoder_layers,
            num_factors=num_factors,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size
        )
        
        # DDFM uses multivariate data (all series from config)
        # IMPORTANT: Filter data columns to match filtered_series_list from config_dict
        # This ensures data shape matches the series IDs in the config after frequency hierarchy filtering
        filtered_series_ids = []
        if config_dict and 'series' in config_dict:
            filtered_series_ids = [s.get('series_id', s) if isinstance(s, dict) else s 
                                  for s in config_dict['series']]
        
        if filtered_series_ids:
            # Use only series IDs from filtered config
            available_series = [s for s in filtered_series_ids if s in data.columns]
            if len(available_series) > 0:
                y_train = data[available_series].dropna()
            else:
                raise ValidationError(f"DDFM: No filtered series found in data columns. Filtered series IDs: {filtered_series_ids[:5]}..., Data columns: {list(data.columns)[:5]}...")
        elif 'experiment' in cfg and 'series' in cfg.experiment:
            series_ids = OmegaConf.to_container(cfg.experiment.series, resolve=True)
            if isinstance(series_ids, list):
                available_series = [s for s in series_ids if s in data.columns]
                if len(available_series) > 0:
                    y_train = data[available_series].dropna()
                else:
                    y_train = data.select_dtypes(include=[np.number]).dropna()
        else:
            y_train = data.select_dtypes(include=[np.number]).dropna()
        
        print(f"Epochs: {epochs}, Encoder layers: {encoder_layers}, Factors: {num_factors}, Series: {y_train.shape[1]} (filtered from config)")
        
    elif model_type == 'arima':
        # Prepare training data for ARIMA (univariate)
        if 'experiment' in cfg and 'target_series' in cfg.experiment:
            target_col = cfg.experiment.target_series
        elif 'target_series' in cfg:
            target_col = cfg.target_series
        else:
            raise ValidationError(f"target_series required for ARIMA. Config: {config_name}")
        
        y_train = data[target_col].dropna()
        if len(y_train) == 0:
            raise ValidationError(f"No valid data for target series '{target_col}'")
        
        print(f"Target series: {target_col}")
        # ARIMA parameters
        order = model_params.get('order', [1, 1, 1])
        auto_arima = model_params.get('auto_arima', {})
        
        if auto_arima and auto_arima.get('enabled', False):
            from sktime.forecasting.arima import AutoARIMA
            forecaster = AutoARIMA(
                max_p=auto_arima.get('max_p', 5),
                max_d=auto_arima.get('max_d', 2),
                max_q=auto_arima.get('max_q', 5),
                information_criterion=auto_arima.get('information_criterion', 'aic')
            )
        else:
            forecaster = SktimeARIMA(order=tuple(order) if isinstance(order, list) else order)
        print(f"Order: {order}")
    elif model_type == 'var':
        # VAR parameters
        lag_order = model_params.get('lag_order')
        auto_lag = model_params.get('auto_lag', {})
        trend = model_params.get('trend', 'c')
        
        # Ensure trend is not None
        if trend is None:
            trend = 'c'
        
        # VAR requires multivariate data - use all series or specified series
        if 'experiment' in cfg and 'series' in cfg.experiment:
            series_ids = OmegaConf.to_container(cfg.experiment.series, resolve=True)
            if isinstance(series_ids, list):
                available_series = [s for s in series_ids if s in data.columns]
                if len(available_series) > 1:
                    y_train = data[available_series].copy()
                else:
                    # Fallback: use all numeric columns
                    y_train = data.select_dtypes(include=[np.number]).copy()
        else:
            # Use all numeric columns
            y_train = data.select_dtypes(include=[np.number]).copy()
        
        # Handle missing values with forward-fill imputation (preserves more data than dropping)
        if y_train.isnull().any().any():
            print(f"Warning: VAR data contains NaN values. Applying forward-fill imputation...")
            # Use sktime Imputer for forward-fill (then backward-fill for leading NaNs)
            imputer_ffill = Imputer(method="ffill")
            imputer_bfill = Imputer(method="bfill")
            
            # Apply imputation to each column
            for col in y_train.columns:
                col_series = y_train[[col]]
                # Forward-fill first
                col_imputed = imputer_ffill.fit_transform(col_series)
                # Then backward-fill any remaining leading NaNs
                if col_imputed.isnull().any().any():
                    col_imputed = imputer_bfill.fit_transform(col_imputed)
                y_train[col] = col_imputed[col]
            
            # If any NaNs remain after imputation, drop those rows as last resort
            if y_train.isnull().any().any():
                print(f"Warning: Some NaN values remain after imputation. Dropping remaining rows with NaN...")
                y_train = y_train.dropna()
        
        if len(y_train) == 0:
            raise ValidationError(f"VAR: No valid data after imputation.")
        
        if y_train.shape[1] < 2:
            raise ValidationError(f"VAR requires at least 2 series. Found {y_train.shape[1]} series.")
        
        # VAR requires frequency to be set on the index for prediction
        # Set frequency if not already set
        # IMPORTANT: Apply asfreq() BEFORE final imputation, as asfreq() can introduce new NaNs
        if isinstance(y_train.index, pd.DatetimeIndex):
            if y_train.index.freq is None:
                # Try to infer frequency
                inferred_freq = pd.infer_freq(y_train.index)
                if inferred_freq:
                    # Use fill_method='ffill' to forward-fill during frequency setting
                    y_train = y_train.asfreq(inferred_freq, fill_method='ffill')
                else:
                    # Default to daily frequency for daily data
                    y_train = y_train.asfreq('D', fill_method='ffill')
        
        # Final imputation check: asfreq() may have introduced new NaNs, so re-impute if needed
        if y_train.isnull().any().any():
            print(f"Warning: VAR data contains NaN values after asfreq(). Applying final imputation...")
            # Use sktime Imputer for forward-fill (then backward-fill for leading NaNs)
            imputer_ffill = Imputer(method="ffill")
            imputer_bfill = Imputer(method="bfill")
            
            # Apply imputation to each column
            for col in y_train.columns:
                col_series = y_train[[col]]
                # Forward-fill first
                col_imputed = imputer_ffill.fit_transform(col_series)
                # Then backward-fill any remaining leading NaNs
                if col_imputed.isnull().any().any():
                    col_imputed = imputer_bfill.fit_transform(col_imputed)
                y_train[col] = col_imputed[col]
            
            # If any NaNs remain after imputation, drop those rows as last resort
            if y_train.isnull().any().any():
                print(f"Warning: Some NaN values remain after final imputation. Dropping remaining rows with NaN...")
                y_train = y_train.dropna()
        
        # Final validation: ensure no NaNs before VAR model creation
        if y_train.isnull().any().any():
            nan_count = y_train.isnull().sum().sum()
            raise ValidationError(f"VAR cannot handle missing data. Found {nan_count} NaN values after all imputation attempts.")
        
        # VAR uses maxlags parameter (not lag_order)
        # Ensure maxlags is a valid integer
        if lag_order is None and auto_lag and auto_lag.get('enabled', False):
            # Auto-lag selection
            maxlags = auto_lag.get('maxlags', 12)
            if maxlags is None:
                maxlags = 12
            ic = auto_lag.get('ic', 'aic')
            if ic is None:
                ic = 'aic'
            forecaster = SktimeVAR(maxlags=int(maxlags), trend=str(trend), ic=str(ic))
        else:
            maxlags = lag_order if lag_order is not None else 1
            if maxlags is None:
                maxlags = 1
            forecaster = SktimeVAR(maxlags=int(maxlags), trend=str(trend))
        print(f"Max lags: {maxlags}, Trend: {trend}, Series: {y_train.shape[1]}")
    
    # Fit model
    forecaster.fit(y_train)
    print(f"{'='*70}\n")
    
    # Get horizons
    if horizons is None:
        if 'experiment' in cfg and 'forecast_horizons' in cfg.experiment:
            horizons_raw = OmegaConf.to_container(cfg.experiment.forecast_horizons, resolve=True)
        elif 'forecast_horizons' in cfg:
            horizons_raw = OmegaConf.to_container(cfg.forecast_horizons, resolve=True)
        else:
            horizons_raw = None
        
        if horizons_raw is not None:
            if isinstance(horizons_raw, list):
                horizons = [int(str(h)) for h in horizons_raw]
            else:
                horizons = [int(str(horizons_raw))]
        else:
            horizons = [1, 7, 28]
    
    # Calculate forecast metrics
    forecast_metrics = {}
    try:
        from src.eval.evaluation import evaluate_forecaster
    except ImportError:
        try:
            from eval.evaluation import evaluate_forecaster
        except ImportError:
            from ..eval.evaluation import evaluate_forecaster
    
    # Get target series for evaluation and naming
    target_col = None
    if 'experiment' in cfg and 'target_series' in cfg.experiment:
        target_col = cfg.experiment.target_series
    elif 'target_series' in cfg:
        target_col = cfg.target_series
    
    # Split data for evaluation
    split_idx = int(len(y_train) * 0.8)
    y_train_eval = y_train.iloc[:split_idx]
    y_test_eval = y_train.iloc[split_idx:]
    
    if len(y_test_eval) > 0:
        forecast_metrics = evaluate_forecaster(
            forecaster, y_train_eval, y_test_eval, horizons, target_series=target_col
        )
    
    # Save model
    if target_col:
        final_model_name = model_name or f"{model_type}_{target_col}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        final_model_name = model_name or f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_dir = outputs_dir / final_model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    import pickle
    with open(model_dir / "model.pkl", 'wb') as f:
        pickle.dump({
            'forecaster': forecaster,
            'model_type': model_type,
            'target_series': target_col,
            'config': OmegaConf.to_container(cfg, resolve=True)
        }, f)
    
    # Create result object compatible with DFM/DDFM format
    # For DFM/DDFM forecasters, extract results from underlying model
    if model_type in ['dfm', 'ddfm']:
        # Access underlying model from forecaster
        if hasattr(forecaster, '_dfm_model'):
            underlying_model = forecaster._dfm_model
        elif hasattr(forecaster, '_ddfm_model'):
            underlying_model = forecaster._ddfm_model
        else:
            underlying_model = None
        
        if underlying_model:
            try:
                result = underlying_model.get_result()
                metadata = underlying_model.get_metadata()
            except:
                result = None
                metadata = {}
        else:
            result = None
            metadata = {}
        
        # Extract metrics from result if available
        if result:
            metrics = {
                'converged': result.converged if hasattr(result, 'converged') else True,
                'num_iter': result.num_iter if hasattr(result, 'num_iter') else 0,
                'loglik': result.loglik if hasattr(result, 'loglik') else np.nan,
                'training_completed': metadata.get('training_completed', True),
                'model_type': model_type,
                'forecast_metrics': forecast_metrics
            }
        else:
            metrics = {
                'converged': True,
                'num_iter': 0,
                'loglik': np.nan,
                'training_completed': True,
                'model_type': model_type,
                'forecast_metrics': forecast_metrics
            }
    else:
        # For ARIMA/VAR, create simple result object
        class SktimeResult:
            def __init__(self):
                self.converged = True
                self.num_iter = 0
                self.loglik = np.nan
        
        result = SktimeResult()
        metadata = {
            'created_at': datetime.now().isoformat(),
            'model_type': model_type,
            'training_completed': True,
            'target_series': target_col
        }
        
        metrics = {
            'converged': True,
            'num_iter': 0,
            'loglik': np.nan,
            'training_completed': True,
            'model_type': model_type,
            'forecast_metrics': forecast_metrics
        }
    
    return {
        'status': 'completed',
        'model_name': final_model_name,
        'model_dir': str(model_dir),
        'metrics': metrics,
        'result': result,
        'metadata': metadata
    }


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
    config_overrides: Optional[list] = None,
    horizons: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Train model with Hydra configuration and save to outputs."""
    config_path = config_path or str(Path(__file__).parent.parent.parent / "config")
    outputs_dir = Path(__file__).parent.parent / "outputs" / "models"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    with hydra.initialize_config_dir(config_dir=config_path, version_base="1.3"):
        cfg = hydra.compose(config_name=config_name, overrides=config_overrides or [])
        # Convert to regular DictConfig (not struct) to allow overrides
        if OmegaConf.is_struct(cfg):
            OmegaConf.set_struct(cfg, False)
        # Also handle nested experiment config
        if 'experiment' in cfg and OmegaConf.is_struct(cfg.experiment):
            OmegaConf.set_struct(cfg.experiment, False)
        
        # Detect model type: prioritize model_name if provided, otherwise use config
        if model_name:
            # Extract model type from model_name (format: "model_type_target_timestamp")
            # Examples: "arima_KOGDP...D_20251205_164431" -> "arima"
            model_type_from_name = model_name.split('_')[0].lower()
            if model_type_from_name in ['arima', 'var', 'dfm', 'ddfm', 'vecm', 'xgboost', 'lightgbm', 'deepar', 'tft']:
                model_type = model_type_from_name
            else:
                # Fallback to config detection
                model_type = _detect_model_type_from_config(cfg)
        else:
            # No model_name provided, use config detection
            model_type = _detect_model_type_from_config(cfg)
        
        # All models use sktime forecaster interface for consistency
        # This ensures unified training, evaluation, and prediction across all models
        
        # Load config - extract series from new config format
        # New format: series is a simple list of series IDs at top level
        # Note: Hydra may wrap config in 'experiment' key when loading from experiment/ directory
        series_list = []  # Make series_list available in outer scope
        try:
            # Extract series - check both top-level and experiment key (Hydra wrapping)
            series_ids_raw = None
            if 'experiment' in cfg and 'series' in cfg.experiment:
                series_ids_raw = OmegaConf.to_container(cfg.experiment.series, resolve=True)
            elif 'series' in cfg:
                series_ids_raw = OmegaConf.to_container(cfg.series, resolve=True)
            
            if series_ids_raw and isinstance(series_ids_raw, list):
                series_ids = series_ids_raw
                # New format: series is list of strings (series IDs)
                # Need to convert to dfm-python format by loading from series config files
                series_list = []
                config_path_obj = Path(config_path)
                series_dir = config_path_obj / "series"
                
                # Get series_overrides from experiment config (if any)
                series_overrides = {}
                if 'experiment' in cfg and 'series_overrides' in cfg.experiment:
                    try:
                        series_overrides_raw = OmegaConf.to_container(cfg.experiment.series_overrides, resolve=True)
                        if isinstance(series_overrides_raw, dict) and series_overrides_raw:
                            series_overrides = series_overrides_raw
                    except (ValueError, TypeError):
                        # Empty or None - ignore
                        pass
                elif 'series_overrides' in cfg:
                    try:
                        series_overrides_raw = OmegaConf.to_container(cfg.series_overrides, resolve=True)
                        if isinstance(series_overrides_raw, dict) and series_overrides_raw:
                            series_overrides = series_overrides_raw
                    except (ValueError, TypeError):
                        # Empty or None - ignore
                        pass
                
                for series_id in series_ids:
                        series_config_path = series_dir / f"{series_id}.yaml"
                        if series_config_path.exists():
                            import yaml
                            with open(series_config_path, 'r') as f:
                                series_cfg = yaml.safe_load(f) or {}
                            
                            # Apply series_overrides if specified for this series
                            if series_id in series_overrides:
                                override = series_overrides[series_id]
                                if isinstance(override, dict):
                                    # Merge override into series_cfg (override takes precedence)
                                    series_cfg = {**series_cfg, **override}
                            
                            # Extract only dfm-python supported fields
                            # Note: transformation codes must match exactly as specified in series config
                            # Do not modify transformation codes - they are applied in preprocessing pipeline
                            dfm_series = {
                                'series_id': series_cfg.get('series_id', series_id),
                                'frequency': series_cfg.get('frequency', 'm'),
                                'transformation': series_cfg.get('transformation', 'lin'),
                            }
                            # Blocks will be converted to binary vector later after we know block order
                            # Store block names/indices temporarily
                            if 'blocks' in series_cfg:
                                dfm_series['_block_names'] = series_cfg['blocks']
                            elif 'block' in series_cfg:
                                dfm_series['_block_names'] = series_cfg['block']
                            else:
                                dfm_series['_block_names'] = ['Block_Global']  # Default
                            series_list.append(dfm_series)
                        else:
                            # Fallback: create minimal series config
                            # But still check for series_overrides
                            dfm_series = {
                                'series_id': series_id,
                                'frequency': 'm',  # Default to monthly
                                'transformation': 'lin',
                                '_block_names': ['Block_Global']  # Default
                            }
                            # Apply series_overrides if specified for this series
                            if series_id in series_overrides:
                                override = series_overrides[series_id]
                                if isinstance(override, dict):
                                    # Update dfm_series with override values
                                    if 'frequency' in override:
                                        dfm_series['frequency'] = override['frequency']
                                    if 'transformation' in override:
                                        dfm_series['transformation'] = override['transformation']
                                    if 'blocks' in override:
                                        dfm_series['_block_names'] = override['blocks']
                                    elif 'block' in override:
                                        dfm_series['_block_names'] = override['block']
                            series_list.append(dfm_series)
        except Exception as e:
            print(f"Warning: Could not extract series: {e}")
            import traceback
            traceback.print_exc()
        
        # Build model config dict
        model_cfg_dict = {}
        
        # Extract blocks from model config file (not from experiment config)
        # Blocks are model-specific, not experiment-specific
        config_path_obj = Path(config_path)
        model_config_path = config_path_obj / "model" / f"{model_type}.yaml"
        
        # Load model-specific config to get block structure
        block_names_order = []  # Order of blocks as they appear in model config
        if model_config_path.exists():
            import yaml
            with open(model_config_path, 'r') as f:
                model_yaml = yaml.safe_load(f) or {}
            # Get block names in order (from model config)
            if 'blocks' in model_yaml and isinstance(model_yaml['blocks'], dict):
                block_names_order = list(model_yaml['blocks'].keys())
            # Remove experiment-specific keys from model_yaml
            excluded_keys = ['models', 'series', 'target_series', 'data_path', 'start_date', 'end_date', 
                            'forecast_horizons', 'evaluation_metrics', 'test_size', 'output_path', 
                            'name', 'description', 'defaults']
            model_yaml = {k: v for k, v in model_yaml.items() if k not in excluded_keys}
            # Merge: model config first (provides blocks), then experiment config (experiment overrides)
            model_cfg_dict = {**model_yaml, **model_cfg_dict}
        
        # Extract model config overrides from experiment config
        # Support both old format (flat) and new format (model_overrides namespace)
        experiment_model_overrides = {}
        
        # Get config source (experiment key or top-level)
        source_cfg = cfg.experiment if 'experiment' in cfg else cfg
        
        # Check for new format: model_overrides namespace (recommended)
        if 'model_overrides' in source_cfg:
            model_overrides_dict = OmegaConf.to_container(source_cfg.model_overrides, resolve=True)
            if isinstance(model_overrides_dict, dict) and model_type in model_overrides_dict:
                # Get model-specific overrides from namespace
                model_specific_overrides = model_overrides_dict[model_type]
                if isinstance(model_specific_overrides, dict):
                    experiment_model_overrides = model_specific_overrides
        else:
            # Fallback to old format: flat structure (for backward compatibility)
            # DFM/DDFM model config keys that can be overridden
            dfm_config_keys = ['max_iter', 'threshold', 'ar_lag', 'regularization_scale', 
                               'nan_method', 'nan_k', 'clip_ar_coefficients', 'augment_idio',
                               'augment_idio_slow', 'idio_min_var', 'idio_rho0', 'use_regularization',
                               'epochs', 'learning_rate', 'batch_size', 'tolerance',
                               'encoder_layers', 'num_factors', 'factor_order', 'use_idiosyncratic']
            # sktime forecaster config keys (ARIMA, VAR, VECM)
            sktime_config_keys = [
                # ARIMA
                'order', 'auto_arima', 'max_iter', 'threshold', 'method',
                # VAR
                'lag_order', 'auto_lag', 'trend', 'enforce_stationarity',
                # VECM
                'coint_rank', 'coint_test', 'k_ar_diff', 'deterministic',
                # Common
                'method'
            ]
            # Combine all config keys
            model_config_keys = dfm_config_keys + sktime_config_keys
            for key in model_config_keys:
                if key in source_cfg:
                    experiment_model_overrides[key] = OmegaConf.to_container(source_cfg[key], resolve=True)
        
        # Apply experiment config overrides to model_cfg_dict
        if experiment_model_overrides:
            model_cfg_dict = {**model_cfg_dict, **experiment_model_overrides}
        
        # For DDFM, if still no blocks, try loading from DFM config
        if not block_names_order and model_type == 'ddfm':
            dfm_config_path = config_path_obj / "model" / "dfm.yaml"
            if dfm_config_path.exists():
                import yaml
                with open(dfm_config_path, 'r') as f:
                    dfm_yaml = yaml.safe_load(f) or {}
                if 'blocks' in dfm_yaml and isinstance(dfm_yaml['blocks'], dict):
                    block_names_order = list(dfm_yaml['blocks'].keys())
                    if 'blocks' not in model_cfg_dict:
                        model_cfg_dict['blocks'] = dfm_yaml['blocks']
        
        # Get block clock frequencies from model config to filter series
        block_clocks = {}  # Map block name to clock frequency
        if 'blocks' in model_cfg_dict and isinstance(model_cfg_dict['blocks'], dict):
            for block_name, block_cfg in model_cfg_dict['blocks'].items():
                if isinstance(block_cfg, dict) and 'clock' in block_cfg:
                    block_clocks[block_name] = block_cfg['clock']
        
        # Frequency hierarchy: 'd' < 'w' < 'm' < 'q' < 'sa' < 'a'
        # A series can only be in a block if its frequency is >= block clock frequency
        freq_hierarchy = {'d': 1, 'w': 2, 'm': 3, 'q': 4, 'sa': 5, 'a': 6}
        
        # Convert series block names to binary vectors and filter incompatible series
        filtered_series_list = []
        if block_names_order and series_list:
            for series_item in series_list:
                series_freq = series_item.get('frequency', 'm').lower()
                series_freq_level = freq_hierarchy.get(series_freq, 3)  # Default to monthly level
                
                if '_block_names' in series_item:
                    block_names = series_item.pop('_block_names')
                    # Check if series frequency is compatible with block clocks
                    compatible_blocks = []
                    for block_spec in block_names:
                        block_name = None
                        if isinstance(block_spec, str):
                            block_name = block_spec
                        elif isinstance(block_spec, int) and 0 <= block_spec < len(block_names_order):
                            block_name = block_names_order[block_spec]
                        
                        if block_name:
                            block_clock = block_clocks.get(block_name, 'm')  # Default to monthly
                            block_clock_level = freq_hierarchy.get(block_clock.lower(), 3)
                            # Series can be in block if series frequency level >= block clock level
                            if series_freq_level >= block_clock_level:
                                compatible_blocks.append(block_spec)
                    
                    # Only add series if it has at least one compatible block
                    if compatible_blocks:
                        # Convert compatible block names to binary vector
                        block_vector = [0] * len(block_names_order)
                        for block_spec in compatible_blocks:
                            if isinstance(block_spec, str):
                                if block_spec in block_names_order:
                                    block_idx = block_names_order.index(block_spec)
                                    block_vector[block_idx] = 1
                            elif isinstance(block_spec, int):
                                block_idx = block_spec - 1 if block_spec > 0 else block_spec
                                if 0 <= block_idx < len(block_names_order):
                                    block_vector[block_idx] = 1
                        series_item['blocks'] = block_vector
                        filtered_series_list.append(series_item)
                    else:
                        # Series frequency incompatible with all blocks - skip it
                        print(f"Warning: Series {series_item.get('series_id', 'unknown')} with frequency '{series_freq}' is incompatible with block clocks. Skipping.")
                else:
                    # No block info: default to first block (Block_Global)
                    # Check compatibility with default block
                    default_block_clock = block_clocks.get(block_names_order[0] if block_names_order else 'Block_Global', 'm')
                    default_block_clock_level = freq_hierarchy.get(default_block_clock.lower(), 3)
                    if series_freq_level >= default_block_clock_level:
                        series_item['blocks'] = [1] + [0] * (len(block_names_order) - 1) if block_names_order else [1]
                        filtered_series_list.append(series_item)
                    else:
                        print(f"Warning: Series {series_item.get('series_id', 'unknown')} with frequency '{series_freq}' is incompatible with default block clock '{default_block_clock}'. Skipping.")
        else:
            # No block structure - include all series
            filtered_series_list = series_list
        
        # Add filtered series to model config
        if filtered_series_list:
            model_cfg_dict['series'] = filtered_series_list
        else:
            raise ValidationError(f"No compatible series found after filtering by block clock frequencies. Check series frequencies against block clocks.")
        
        # Remove experiment-specific keys that dfm-python doesn't understand
        # But keep model config keys (max_iter, threshold, ar_lag, etc.) as they are valid model config
        excluded_keys = ['models', 'target_series', 'data_path', 'start_date', 'end_date', 
                        'forecast_horizons', 'evaluation_metrics', 'test_size', 'output_path', 
                        'name', 'description', 'defaults']
        model_cfg_dict = {k: v for k, v in model_cfg_dict.items() if k not in excluded_keys}
        
        # Get data path - use provided data_path or from config (new format)
        # Check both top-level and experiment key (Hydra wrapping)
        data_file = data_path
        if not data_file:
            if 'experiment' in cfg and 'data_path' in cfg.experiment:
                data_file = cfg.experiment.data_path
            elif 'data_path' in cfg:
                data_file = cfg.data_path
        if not data_file:
            raise ValidationError(f"data_path required. Config: {config_name}")
        
        # All models use sktime forecaster interface for unified training and evaluation
        return _train_forecaster(
            model_type=model_type,
            config_name=config_name,
            cfg=cfg,
            data_file=data_file,
            model_name=model_name,
            horizons=horizons,
            outputs_dir=outputs_dir,
            model_cfg_dict=model_cfg_dict if 'model_cfg_dict' in locals() else {}
        )
        
        # Calculate forecast metrics if horizons are specified
        # Get horizons from parameter, config (new format), or use default
        # Check both top-level and experiment key (Hydra wrapping)
        forecast_metrics = {}
        if horizons is None:
            horizons_raw = None
            if 'experiment' in cfg and 'forecast_horizons' in cfg.experiment:
                horizons_raw = OmegaConf.to_container(cfg.experiment.forecast_horizons, resolve=True)
            elif 'forecast_horizons' in cfg:
                horizons_raw = OmegaConf.to_container(cfg.forecast_horizons, resolve=True)
            # Ensure horizons is a list of integers
            if horizons_raw is not None:
                if isinstance(horizons_raw, list):
                    horizons = [int(str(h)) for h in horizons_raw]
                else:
                    horizons = [int(str(horizons_raw))]
        if horizons is None:
            # Default horizons
            horizons = [1, 7, 28]
        if horizons:
            try:
                import pandas as pd
                try:
                    from src.eval.evaluation import calculate_metrics_per_horizon
                except ImportError:
                    try:
                        from eval.evaluation import calculate_metrics_per_horizon
                    except ImportError:
                        from .eval.evaluation import calculate_metrics_per_horizon
                
                # Load data for evaluation
                data = pd.read_csv(data_file, index_col=0, parse_dates=True)
                # Check both top-level and experiment key (Hydra wrapping)
                if 'experiment' in cfg and 'target_series' in cfg.experiment:
                    target_col = cfg.experiment.target_series
                else:
                    target_col = cfg.get('target_series')
                
                if target_col and target_col in data.columns:
                    # For forecast evaluation, we need to:
                    # 1. Train on subset of data (train set)
                    # 2. Predict future values from end of train set
                    # 3. Compare with actual test set values
                    # However, model is already trained on full data, so we use a different approach:
                    # - Use last portion as test set
                    # - Predict from the point just before test set starts
                    # - Note: This is not ideal but works for evaluation
                    split_idx = int(len(data) * 0.8)
                    y_train_data = data.iloc[:split_idx]
                    y_test_data = data.iloc[split_idx:]
                    
                    # Generate predictions for max horizon
                    # Note: predict() returns forecasts starting from the last training point
                    max_horizon = max(horizons)
                    y_pred = None  # Initialize to avoid unbound variable
                    try:
                        # Predict from current state (model was trained on full data)
                        # This gives us forecasts for the next max_horizon periods
                        pred_result = model.predict(horizon=max_horizon)
                        if isinstance(pred_result, tuple):
                            X_pred, Z_pred = pred_result
                        else:
                            X_pred = pred_result
                        
                        # DFM predict returns (horizon, N) where N is number of series
                        if isinstance(X_pred, np.ndarray):
                            # Handle NaN predictions - check if all NaN
                            if np.all(np.isnan(X_pred)):
                                print(f"Warning: All predictions are NaN. Shape: {X_pred.shape}. Skipping forecast metrics calculation.")
                                forecast_metrics = {}
                                y_pred = pd.Series([], dtype=float)
                            else:
                                # X_pred shape: (horizon, N) where N is number of series
                                # Get series list from config - reuse the same logic as model loading
                                # Extract series IDs from config (new format: simple list)
                                # Check both top-level and experiment key (Hydra wrapping)
                                target_idx = 0
                                series_ids = []
                                if 'experiment' in cfg and 'series' in cfg.experiment:
                                    series_ids_raw = OmegaConf.to_container(cfg.experiment.series, resolve=True)
                                    if isinstance(series_ids_raw, list):
                                        series_ids = [str(s) for s in series_ids_raw]
                                elif 'series' in cfg:
                                    series_ids_raw = OmegaConf.to_container(cfg.series, resolve=True)
                                    if isinstance(series_ids_raw, list):
                                        series_ids = [str(s) for s in series_ids_raw]
                                
                                # Also check series_list from model loading (contains dict format)
                                if not series_ids and 'series_list' in locals() and series_list:
                                    for s in series_list:
                                        if isinstance(s, dict):
                                            series_ids.append(s.get('series_id', str(s)))
                                        else:
                                            series_ids.append(str(s))
                                    
                                    if target_col in series_ids:
                                        target_idx = series_ids.index(target_col)
                                    elif len(series_ids) > 0:
                                        # Match by name similarity
                                        norm_target = target_col.replace('...', '').replace('.', '').lower()
                                        for idx, s_id in enumerate(series_ids):
                                            norm_s = str(s_id).replace('...', '').replace('.', '').lower()
                                            if norm_target in norm_s or norm_s in norm_target:
                                                target_idx = idx
                                                break
                                
                                # Extract predictions for target series
                                if X_pred.ndim == 2 and X_pred.shape[1] > target_idx:
                                    y_pred_values = X_pred[:, target_idx]
                                elif X_pred.ndim == 2 and X_pred.shape[1] > 0:
                                    y_pred_values = X_pred[:, 0]  # Fallback
                                elif X_pred.ndim == 1:
                                    y_pred_values = X_pred
                                else:
                                    forecast_metrics = {}
                                    y_pred_values = np.array([])
                                
                                # Create index aligned with test data
                                if len(y_pred_values) > 0 and len(y_test_data) > 0:
                                    n_pred = min(len(y_pred_values), len(y_test_data))
                                    if n_pred > 0:
                                        pred_index = y_test_data.index[:n_pred]
                                        y_pred = pd.Series(y_pred_values[:n_pred], index=pred_index)
                                    else:
                                        y_pred = pd.Series([], dtype=float)
                                else:
                                    y_pred = pd.Series([], dtype=float)
                        else:
                            # Already a DataFrame/Series
                            y_pred = X_pred
                        
                        # Align test data with predictions
                        if y_pred is not None and isinstance(y_pred, pd.DataFrame):
                            y_test_aligned = y_test_data[target_col] if target_col in y_test_data.columns else y_test_data.iloc[:, 0]
                            y_pred_aligned = y_pred[target_col] if target_col in y_pred.columns else y_pred.iloc[:, 0]
                        elif y_pred is not None:
                            y_test_aligned = y_test_data[target_col] if target_col in y_test_data.columns else y_test_data.iloc[:, 0]
                            y_pred_aligned = y_pred
                        else:
                            y_test_aligned = pd.Series([], dtype=float)
                            y_pred_aligned = pd.Series([], dtype=float)
                        
                        # Align indices and check for valid predictions
                        if y_pred is not None and len(y_pred_aligned) > 0 and len(y_test_aligned) > 0:
                            common_idx = y_test_aligned.index.intersection(y_pred_aligned.index)
                            if len(common_idx) > 0:
                                y_test_aligned = y_test_aligned.loc[common_idx]
                                y_pred_aligned = y_pred_aligned.loc[common_idx]
                                
                                # Check for NaN predictions
                                if isinstance(y_pred_aligned, pd.Series):
                                    valid_mask = ~y_pred_aligned.isna()
                                else:
                                    valid_mask = ~np.isnan(y_pred_aligned)
                                
                                if valid_mask.sum() > 0:
                                    # Calculate metrics per horizon
                                    y_train_series = y_train_data[target_col] if target_col in y_train_data.columns else y_train_data.iloc[:, 0]
                                    forecast_metrics = calculate_metrics_per_horizon(
                                        y_test_aligned,
                                        y_pred_aligned,
                                        horizons,
                                        y_train=y_train_series,
                                        target_series=target_col
                                    )
                                else:
                                    forecast_metrics = {}
                            else:
                                forecast_metrics = {}
                        else:
                            forecast_metrics = {}
                    except Exception:
                        forecast_metrics = {}
            except Exception:
                forecast_metrics = {}
        
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
    config_dir: Optional[str] = None,
    config_name: Optional[str] = None,
    config_overrides: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Train multiple models and compare results (internal function).
    
    This is the low-level implementation. For programmatic use, prefer:
    - train_model() from src.train for single model training
    - compare_models_by_config() from src.train for model comparison
    """
    config_dir = config_dir or str(Path(__file__).parent.parent.parent / "config")
    data_path = data_path or str(Path(__file__).parent.parent.parent / "data" / "sample_data.csv")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent.parent / "outputs" / "comparisons" / f"{target_series}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"Comparing models for {target_series}")
    print(f"Models: {', '.join(models)} | Horizons: {horizons}")
    print("=" * 70)
    
    model_results = {}
    failed_models = []
    
    # Use provided config_name or derive from target_series
    if config_name is None:
        # Fallback: map target series to report config name
        report_map = {
            "KOGDP...D": "experiment/kogdp_report",
            "KOCNPER.D": "experiment/kocnper_report",
            "KOGFCF..D": "experiment/kogfcf_report"
        }
        config_name = report_map.get(target_series, f"experiment/{target_series.lower().replace('...', '').replace('.', '')}_report")
    
    for i, model_name in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] {model_name.upper()}...")
        
        # Build config overrides - use the same config for all models
        # Model-specific overrides are handled in train() function via model_overrides namespace
        # Use provided overrides or empty list
        if config_overrides is None:
            config_overrides = []
        
        try:
            # Use the provided config_name for all models
            # Each model will use the same experiment config, but model-specific settings
            # are handled via model_overrides namespace in the config
            # Pass config_overrides to train() so CLI overrides are preserved
            result = train(
                config_name=config_name,
                config_path=config_dir,
                data_path=data_path,
                model_name=f"{model_name}_{target_series}_{timestamp}",
                config_overrides=config_overrides or [],
                horizons=horizons
            )
            
            if result is not None:
                model_results[model_name] = result
            print(f"  ✓ Completed")
        except Exception as e:
            import traceback
            print(f"  ✗ Failed: {str(e)}")
            print("Full traceback:")
            traceback.print_exc()
            failed_models.append(model_name)
            model_results[model_name] = {'status': 'failed', 'error': str(e), 'metrics': None}
    
    comparison = None
    successful_count = len(model_results) - len(failed_models)
    if successful_count > 0:
        print(f"\nComparing {successful_count} successful models...")
        comparison = _compare_results(model_results, horizons, target_series)
        
        if comparison and comparison.get('metrics_table') is not None:
            # Import with fallback for both script and package execution
            try:
                from src.eval.evaluation import generate_comparison_table
            except ImportError:
                try:
                    from eval.evaluation import generate_comparison_table
                except ImportError:
                    from ..eval.evaluation import generate_comparison_table
            
            if generate_comparison_table:
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
    # Import with fallback for both script and package execution
    try:
        from src.eval.evaluation import compare_multiple_models
    except ImportError:
        try:
            from eval.evaluation import compare_multiple_models
        except ImportError:
            from ..eval.evaluation import compare_multiple_models
    
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





