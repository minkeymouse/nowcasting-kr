"""Training execution module for DFM/DDFM models.

This module provides functions to run training experiments using Hydra configuration.
Also includes experiment execution helpers (merged from experiment.py).

The module supports both direct model training and sktime-compatible forecasting workflows
using ForecastingPipeline and splitters for temporal cross-validation.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

# Set up paths using centralized utility (relative import since we're in src/)
from .utils.path_setup import setup_paths
setup_paths(include_dfm_python=True, include_src=True, include_app=True)

# Import custom exceptions for error handling
from app.utils import (
    TrainingError, ConfigError, ValidationError, 
    DEFAULT_DDFM_ENCODER_LAYERS, DEFAULT_DDFM_NUM_FACTORS, DEFAULT_DDFM_EPOCHS,
    validate_config_file
)

try:
    import hydra
    from hydra.utils import get_original_cwd
    from omegaconf import DictConfig, OmegaConf
except ImportError as e:
    raise ImportError(
        f"Required dependencies not available: {e}\n"
        "Please install: uv pip install hydra-core omegaconf\n"
        "Or install all dependencies: uv pip install -e ."
    )

# Import model wrappers
try:
    from model.dfm import DFM
    from model.ddfm import DDFM
except ImportError as e:
    raise ImportError(
        f"Model wrapper not available: {e}\n"
        "This usually means:\n"
        "  1. dfm-python package is not installed or not in path\n"
        "  2. Path setup failed - check src/utils/path_setup.py\n"
        "  3. Missing dependencies - run: uv pip install -e dfm-python/"
    )


def _detect_model_type_from_config(cfg: DictConfig) -> str:
    """Detect model type from Hydra config.
    
    Args:
        cfg: Hydra DictConfig object
        
    Returns:
        Model type string ("dfm" or "ddfm")
    """
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
    """Extract DDFM parameters from Hydra config.
    
    Args:
        cfg: Hydra DictConfig object
        
    Returns:
        Dictionary with DDFM parameters (encoder_layers, num_factors, epochs)
        Uses defaults if parameters not found in config
    """
    params = {
        "encoder_layers": DEFAULT_DDFM_ENCODER_LAYERS,
        "num_factors": DEFAULT_DDFM_NUM_FACTORS,
        "epochs": DEFAULT_DDFM_EPOCHS
    }
    
    # Convert to dict if needed
    if isinstance(cfg, DictConfig):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    else:
        cfg_dict = cfg
    
    # Extract encoder_layers
    if "encoder_layers" in cfg_dict:
        encoder_layers = cfg_dict["encoder_layers"]
        if isinstance(encoder_layers, list) and all(isinstance(x, int) for x in encoder_layers):
            params["encoder_layers"] = encoder_layers
    
    # Extract num_factors
    if "num_factors" in cfg_dict:
        num_factors = cfg_dict["num_factors"]
        if isinstance(num_factors, int) and num_factors > 0:
            params["num_factors"] = num_factors
    
    # Extract epochs
    if "epochs" in cfg_dict:
        epochs = cfg_dict["epochs"]
        if isinstance(epochs, int) and epochs > 0:
            params["epochs"] = epochs
    
    return params


def run_training_experiment(
    config_name: str,
    config_path: Optional[str] = None,
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    config_overrides: Optional[list] = None
) -> Dict[str, Any]:
    """Run a training experiment with Hydra configuration.
    
    Parameters
    ----------
    config_name : str
        Name of the experiment config (without .yaml)
    config_path : str, optional
        Path to config directory. If None, uses ../config
    data_path : str, optional
        Path to data file. If None, uses config's data.path
    output_dir : str, optional
        Output directory for results. If None, uses Hydra default
    config_overrides : list, optional
        List of Hydra config overrides (e.g., ["max_iter=1000"])
        
    Returns
    -------
    dict
        Dictionary with training results and metrics
    """
    # Set up Hydra config path
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "config")
    
    # Initialize Hydra
    with hydra.initialize_config_dir(
        config_dir=config_path,
        version_base="1.3"
    ):
        # Compose config
        cfg = hydra.compose(
            config_name=config_name,
            overrides=config_overrides or []
        )
        
        # Override data path if provided
        if data_path:
            cfg.data.path = data_path
        
        # Override output directory if provided
        if output_dir:
            cfg.hydra.run.dir = output_dir
        
        # Detect model type from config
        model_type = _detect_model_type_from_config(cfg)
        print(f"Detected model type: {model_type}")
        
        # Initialize appropriate model wrapper
        if model_type == "ddfm":
            # Extract DDFM parameters from config
            ddfm_params = _extract_ddfm_params_from_hydra(cfg)
            print(f"DDFM parameters: encoder_layers={ddfm_params.get('encoder_layers')}, "
                  f"num_factors={ddfm_params.get('num_factors')}, epochs={ddfm_params.get('epochs')}")
            model = DDFM(**ddfm_params)
        else:
            model = DFM()
        
        # Load configuration
        print(f"Loading configuration: {config_name}")
        # Pass Hydra config using hydra= keyword (wrapper now supports this)
        model.load_config(hydra=cfg)
        
        # Get data path
        data_file = cfg.get('data', {}).get('path') or data_path
        if not data_file:
            raise ValidationError(
                "data_path must be provided either:\n"
                "  1. As a parameter: run_training_experiment(..., data_path='path/to/data.csv')\n"
                "  2. In config file: data.path: 'path/to/data.csv'\n"
                f"Current config: {config_name}, config_path: {config_path}"
            )
        
        print(f"Training model with data from: {data_file}")
        
        # Train model using new API (data_path will create DFMDataModule automatically)
        if model_type == "ddfm":
            # DDFM training (epochs passed in constructor, train() accepts data_path)
            model.train(data_path=data_file)
        else:
            # DFM training (needs max_iter and threshold)
            max_iter = cfg.get('max_iter', 5000)
            threshold = cfg.get('threshold', 1e-5)
            print(f"Training parameters: max_iter={max_iter}, threshold={threshold}")
            model.train(data_path=data_file, max_iter=max_iter, threshold=threshold)
        
        # Get results and metadata
        result = model.get_result()
        metadata = model.get_metadata()
        metrics = {
            'converged': result.converged,
            'num_iter': result.num_iter,
            'loglik': result.loglik,
            'training_completed': metadata.get('training_completed', True),
            'model_type': metadata.get('model_type', 'dfm')
        }
        
        print(f"Training completed:")
        print(f"  - Converged: {metrics['converged']}")
        print(f"  - Iterations: {metrics['num_iter']}")
        print(f"  - Log-likelihood: {metrics['loglik']:.4f}")
        
        return {
            'status': 'completed',
            'metrics': metrics,
            'result': result,
            'metadata': metadata
        }


if __name__ == "__main__":
    # CLI entry point
    import argparse
    
    parser = argparse.ArgumentParser(description="Run DFM training experiment")
    parser.add_argument("config_name", help="Experiment config name")
    parser.add_argument("--data-path", help="Path to data file")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--override", action="append", help="Hydra config override")
    
    args = parser.parse_args()
    
    result = run_training_experiment(
        config_name=args.config_name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        config_overrides=args.override
    )
    
    print(f"\nExperiment complete: {result['status']}")


# ========================================================================
# Experiment Execution Helpers (merged from experiment.py)
# ========================================================================

def run_experiment_from_config(
    experiment_id: str,
    config_dir: Optional[str] = None,
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    config_overrides: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Run an experiment from config directory.
    
    Parameters
    ----------
    experiment_id : str
        Experiment ID (config name without .yaml)
    config_dir : str, optional
        Path to config directory. If None, uses ../config
    data_path : str, optional
        Path to data file. If None, uses config's data.path
    output_dir : str, optional
        Output directory. If None, uses Hydra default
    config_overrides : list, optional
        List of Hydra config overrides
        
    Returns
    -------
    dict
        Experiment results
    """
    if config_dir is None:
        config_dir = str(Path(__file__).parent.parent / "config")
    
    # Check if experiment config exists
    exp_config_path = Path(config_dir) / "experiment" / f"{experiment_id}.yaml"
    validate_config_file(exp_config_path)
    
    # Run training
    result = run_training_experiment(
        config_name=experiment_id,
        config_path=config_dir,
        data_path=data_path,
        output_dir=output_dir,
        config_overrides=config_overrides
    )
    
    return {
        'experiment_id': experiment_id,
        'status': result['status'],
        'metrics': result['metrics'],
        'timestamp': datetime.now().isoformat()
    }


def save_experiment_results(
    results: Dict[str, Any],
    output_dir: str,
    filename: Optional[str] = None
) -> Path:
    """Save experiment results to JSON file.
    
    Parameters
    ----------
    results : dict
        Experiment results dictionary
    output_dir : str
        Output directory path
    filename : str, optional
        Output filename. If None, uses timestamp
        
    Returns
    -------
    Path
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_results_{timestamp}.json"
    
    filepath = output_path / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    return filepath


# ========================================================================
# sktime Integration Functions
# ========================================================================

def run_training_with_sktime(
    experiment_id: str,
    data_path: str,
    config_path: Optional[str] = None,
    fh: Optional[Any] = None,
    cv_splitter: Optional[Any] = None,
    scoring_metrics: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """Run training experiment using sktime's forecasting API.
    
    This function integrates DFM/DDFM models with sktime's forecasting pipeline,
    enabling use of splitters, cross-validation, and evaluation metrics.
    
    Parameters
    ----------
    experiment_id : str
        Experiment ID (config name without .yaml)
    data_path : str
        Path to data file (CSV)
    config_path : str, optional
        Path to config directory. If None, uses ../config
    fh : array-like or ForecastingHorizon, optional
        Forecasting horizon. If None, uses default from config
    cv_splitter : sktime splitter, optional
        Cross-validation splitter (e.g., ExpandingWindowSplitter).
        If None, uses single train-test split
    scoring_metrics : list, optional
        List of sktime scoring metrics. If None, uses default metrics
        
    Returns
    -------
    dict
        Dictionary with training results, metrics, and evaluation scores
        
    Examples
    --------
    >>> from src.training import run_training_with_sktime
    >>> from sktime.split import ExpandingWindowSplitter
    >>> import numpy as np
    >>> 
    >>> # Define forecasting horizon
    >>> fh = np.arange(1, 13)  # Next 12 steps
    >>> 
    >>> # Create splitter for cross-validation
    >>> cv = ExpandingWindowSplitter(
    >>>     fh=fh,
    >>>     initial_window=100,
    >>>     step_length=12
    >>> )
    >>> 
    >>> # Run training with cross-validation
    >>> results = run_training_with_sktime(
    >>>     experiment_id="my_experiment",
    >>>     data_path="data/sample_data.csv",
    >>>     fh=fh,
    >>>     cv_splitter=cv
    >>> )
    """
    try:
        from sktime.forecasting.model_evaluation import evaluate
        from sktime.split import temporal_train_test_split
        from sktime.performance_metrics.forecasting import (
            MeanAbsoluteError,
            MeanAbsolutePercentageError
        )
        import pandas as pd
        import numpy as np
    except ImportError as e:
        raise ImportError(
            f"sktime is required for sktime integration: {e}\n"
            "Install it with: pip install sktime[forecasting]"
        )
    
    # Set up config path
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "config")
    
    # Load data
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Get target column from config if available
    # For now, use all columns (multivariate forecasting)
    y = data
    
    # Set default forecasting horizon
    if fh is None:
        fh = np.arange(1, 13)  # Default: next 12 steps
    
    # Run training experiment (using existing function)
    training_results = run_training_experiment(
        config_name=experiment_id,
        config_path=config_path,
        data_path=data_path
    )
    
    # If splitter provided, perform cross-validation evaluation
    if cv_splitter is not None:
        # Note: This requires the model to be sktime-compatible
        # For now, we'll use the trained model for evaluation
        # Future: Integrate with DFMForecaster/DDFMForecaster
        
        if scoring_metrics is None:
            scoring_metrics = [MeanAbsoluteError(), MeanAbsolutePercentageError()]
        
        # For now, return training results with note about CV
        # Full CV integration requires sktime-compatible forecaster wrapper
        return {
            **training_results,
            'cv_evaluation': 'Cross-validation requires sktime-compatible forecaster wrapper',
            'note': 'Use DFMForecaster or DDFMForecaster for full sktime integration'
        }
    else:
        # Simple train-test split evaluation
        train_size = int(len(y) * 0.8)
        y_train, y_test = temporal_train_test_split(y, train_size=train_size)
        
        # Evaluate on test set
        # Note: This is a placeholder - full integration requires forecaster wrapper
        return {
            **training_results,
            'evaluation': {
                'train_size': train_size,
                'test_size': len(y_test),
                'note': 'Full evaluation requires sktime-compatible forecaster wrapper'
            }
        }

