"""Helper functions to reduce code duplication across experiment modules.

Consolidates common patterns:
- Checkpoint path finding
- Experiment config parsing
- Frequency inference
- Experiment type detection
"""

import logging
from pathlib import Path
from typing import Any, Optional, Dict, Literal
import pandas as pd

logger = logging.getLogger(__name__)


def find_checkpoint_path(outputs_dir: Path, error_msg: Optional[str] = None) -> Path:
    """Find model checkpoint file with .pkl/.zip fallback.
    
    Parameters
    ----------
    outputs_dir : Path
        Directory containing checkpoint files
    error_msg : str, optional
        Custom error message if checkpoint not found
        
    Returns
    -------
    Path
        Path to checkpoint file
        
    Raises
    ------
    FileNotFoundError
        If neither .pkl nor .zip checkpoint exists
    """
    checkpoint_path = outputs_dir / "model.pkl"
    if not checkpoint_path.exists():
        checkpoint_path = outputs_dir / "model.zip"
    
    if not checkpoint_path.exists():
        # Check if directory exists but is empty (training may have failed)
        if outputs_dir.exists() and len(list(outputs_dir.iterdir())) == 0:
            default_msg = (
                f"Model checkpoint not found at {outputs_dir}. "
                f"Training may have failed - check training logs for errors. "
                f"Train the model first with train=true."
            )
        else:
            default_msg = f"Model checkpoint not found at {outputs_dir}. Train the model first with train=true."
        raise FileNotFoundError(error_msg or default_msg)
    
    return checkpoint_path


def determine_experiment_type(experiment_cfg: Any) -> Literal["short_term", "long_term"]:
    """Determine experiment type from config structure.
    
    Parameters
    ----------
    experiment_cfg : Any
        Experiment configuration (dict, DictConfig, or str)
        
    Returns
    -------
    str
        Experiment type: "short_term" or "long_term"
        
    Raises
    ------
    ValueError
        If experiment type cannot be determined
    """
    # Handle string type (just experiment name)
    if isinstance(experiment_cfg, str):
        if experiment_cfg in ["short_term", "long_term"]:
            return experiment_cfg
        else:
            raise ValueError(f"Unknown experiment type string: {experiment_cfg}")
    
    # Handle dict-like config
    if isinstance(experiment_cfg, dict) or hasattr(experiment_cfg, 'get'):
        # short_term has 'horizon' (singular) and/or 'update_frequency'
        if experiment_cfg.get('horizon') is not None or experiment_cfg.get('update_frequency') is not None:
            return "short_term"
        # long_term has 'horizons' (plural)
        elif experiment_cfg.get('horizons') is not None:
            return "long_term"
    
    raise ValueError(f"Could not determine experiment type from config: {experiment_cfg}")


def parse_experiment_config(
    exp_cfg: Any,
    experiment_type: Literal["short_term", "long_term"]
) -> Dict[str, Any]:
    """Parse experiment configuration.
    
    Parameters
    ----------
    exp_cfg : Any
        Experiment configuration (dict, DictConfig, or str)
    experiment_type : str
        Experiment type: "short_term" or "long_term"
        
    Returns
    -------
    dict
        Parsed config:
        - short_term: start_date, end_date, update_params
        - long_term: start_date, horizons
        
    Raises
    ------
    ValueError
        If required config values are missing
    """
    from src.utils import get_experiment_dates
    
    # Handle string type (load from config file via get_experiment_dates)
    if isinstance(exp_cfg, str):
        dates = get_experiment_dates(experiment_type)
        if experiment_type == "short_term":
            return {
                "start_date": dates["start_date"],
                "end_date": dates["end_date"],
                "update_params": False  # Default for string type
            }
        else:  # long_term
            # Load horizons from config file
            try:
                from omegaconf import OmegaConf
                from src.utils import get_project_root
                project_root = get_project_root()
                config_path = project_root / "config" / "experiment" / "long_term.yaml"
                if config_path.exists():
                    cfg = OmegaConf.load(config_path)
                    horizons = cfg.get("horizons", [])
                    if horizons:
                        return {
                            "start_date": dates["start_date"],
                            "horizons": list(horizons) if hasattr(horizons, '__iter__') else [horizons]
                        }
            except Exception:
                pass
            # If config loading fails, raise error to force explicit config
            raise ValueError(
                "long_term experiment requires explicit config with 'horizons'. "
                "Use experiment=long_term (with config file) instead of string, "
                "or ensure config/experiment/long_term.yaml exists."
            )
    
    # Handle dict-like config (must have required fields)
    if hasattr(exp_cfg, 'get'):
        if experiment_type == "short_term":
            start_date = exp_cfg.get('start_date')
            end_date = exp_cfg.get('end_date')
            if start_date is None or end_date is None:
                # Fallback to get_experiment_dates if missing
                dates = get_experiment_dates("short_term")
                start_date = start_date or dates["start_date"]
                end_date = end_date or dates["end_date"]
            return {
                "start_date": start_date,
                "end_date": end_date,
                "update_params": exp_cfg.get('update_params', False)
            }
        else:  # long_term
            start_date = exp_cfg.get('start_date')
            horizons = exp_cfg.get('horizons')
            if start_date is None:
                # Fallback to get_experiment_dates if missing
                dates = get_experiment_dates("long_term")
                start_date = dates["start_date"]
            if horizons is None:
                raise ValueError(
                    "long_term experiment config must specify 'horizons'. "
                    f"Config keys: {list(exp_cfg.keys()) if hasattr(exp_cfg, 'keys') else 'unknown'}"
                )
            # Handle comma-separated string
            if isinstance(horizons, str):
                horizons = [int(h.strip()) for h in horizons.split(',')]
            return {
                "start_date": start_date,
                "horizons": horizons
            }
    
    # Empty or invalid config
    raise ValueError(
        f"Cannot parse experiment config for {experiment_type}. "
        f"Expected dict/DictConfig or string, got: {type(exp_cfg)}"
    )


def infer_frequency(data_or_model: Any, default: str = 'W') -> str:
    """Infer frequency string from data or model.
    
    Parameters
    ----------
    data_or_model : Any
        Either a DataFrame with DatetimeIndex or a model with _y attribute
    default : str, default 'W'
        Default frequency if inference fails
        
    Returns
    -------
    str
        Frequency string (e.g., 'W', 'W-FRI', 'M')
    """
    freq = None
    
    # Try to get frequency from DataFrame index (if data_or_model is a DataFrame)
    if isinstance(data_or_model, pd.DataFrame):
        if isinstance(data_or_model.index, pd.DatetimeIndex):
            if data_or_model.index.freq is not None:
                freq = data_or_model.index.freqstr or str(data_or_model.index.freq)
            else:
                freq = pd.infer_freq(data_or_model.index)
            if freq:
                return str(freq) if not isinstance(freq, str) else freq
    
    # Try to get frequency from model's training data (if data_or_model is a model)
    if hasattr(data_or_model, '_y') and data_or_model._y is not None:
        if isinstance(data_or_model._y, pd.DataFrame) and isinstance(data_or_model._y.index, pd.DatetimeIndex):
            if data_or_model._y.index.freq is not None:
                freq = data_or_model._y.index.freqstr or str(data_or_model._y.index.freq)
            else:
                freq = pd.infer_freq(data_or_model._y.index)
            if freq:
                return str(freq) if not isinstance(freq, str) else freq
    
    # Fallback to default
    logger.warning(f"Could not infer frequency, defaulting to '{default}'")
    return default


def extract_target_series_from_config(cfg: Any) -> Optional[list]:
    """Extract target series from model config.
    
    Checks for covariates first (new approach), then falls back to target_series (backward compatibility).
    If covariates is found, returns None (targets will be computed as all_series - covariates).
    
    Parameters
    ----------
    cfg : Any
        Model configuration (DictConfig or dict)
        
    Returns
    -------
    list or None
        List of target series names (from target_series), or None if covariates found or not found
    """
    # Extract model config from nested structure
    model_cfg = cfg.get('model', cfg) if hasattr(cfg, 'get') else cfg
    
    # Check for covariates first (new approach)
    if hasattr(model_cfg, 'covariates'):
        covariates = model_cfg.covariates
        if covariates is not None:
            # If covariates found, return None (targets will be computed as all_series - covariates)
            return None
    
    if isinstance(model_cfg, dict) and 'covariates' in model_cfg:
        covariates = model_cfg['covariates']
        if covariates is not None:
            # If covariates found, return None (targets will be computed as all_series - covariates)
            return None
    
    # Fallback to target_series (backward compatibility)
    if hasattr(model_cfg, 'target_series'):
        target_series = model_cfg.target_series
        if target_series is not None:
            return list(target_series) if not isinstance(target_series, list) else target_series
    
    if isinstance(model_cfg, dict) and 'target_series' in model_cfg:
        target_series = model_cfg['target_series']
        if target_series is not None:
            return list(target_series) if not isinstance(target_series, list) else target_series
    
    return None
