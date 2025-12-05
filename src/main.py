"""Main entry point for research experiments using Hydra."""

import sys
from pathlib import Path

# Set up paths using centralized utility (relative import since we're in src/)
from .utils.path_setup import setup_paths
setup_paths(include_dfm_python=True, include_src=True, include_app=True)

# Import custom exceptions for error handling
from app.utils import TrainingError

try:
    import hydra
    from hydra.utils import get_original_cwd
    from omegaconf import DictConfig, OmegaConf
except ImportError as e:
    print(f"Required dependencies not available: {e}")
    print("Please install: pip install hydra-core omegaconf")
    sys.exit(1)

# Import model wrappers
try:
    from model.dfm import DFM
    from model.ddfm import DDFM
except ImportError as e:
    print(f"Model wrapper not available: {e}")
    print("This usually means dfm-python package is not installed or not in path")
    sys.exit(1)

# Import helper functions from training.py
from .training import _detect_model_type_from_config, _extract_ddfm_params_from_hydra


@hydra.main(config_path="../config", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run DFM or DDFM experiment with Hydra configuration.
    
    This uses the model wrapper (DFM or DDFM) which provides metadata tracking
    and outputs/ structure support. Results are saved to outputs/ or 
    nowcasting-report/ directories.
    
    The model type (DFM or DDFM) is automatically detected from the config.
    """
    print("=" * 70)
    print("DFM/DDFM Research Experiment")
    print("=" * 70)
    
    # Load configuration first to detect model type
    print(f"\nLoading configuration from: {cfg.get('config_name', 'default')}")
    
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
    
    # Load configuration into model
    model.load_config(hydra=cfg)
    
    # Get data path
    data_path = cfg.get('data_path')
    if not data_path:
        # Try alternative location in config
        data_path = cfg.get('data', {}).get('path')
    if not data_path:
        print("\nNo data_path specified in config")
        return
    
    print(f"\nTraining model with data from: {data_path}")
    
    # Train model using new API (data_path will create DFMDataModule automatically)
    if model_type == "ddfm":
        # DDFM training (epochs passed in constructor, train() accepts data_path)
        model.train(data_path=data_path)
    else:
        # DFM training (needs max_iter and threshold)
        max_iter = cfg.get('max_iter', 5000)
        threshold = cfg.get('threshold', 1e-5)
        print(f"Training parameters: max_iter={max_iter}, threshold={threshold}")
        model.train(data_path=data_path, max_iter=max_iter, threshold=threshold)
    
    # Get results
    result = model.get_result()
    metadata = model.get_metadata()
    print(f"\nTraining completed:")
    print(f"  - Model type: {metadata.get('model_type', 'dfm')}")
    
    # Display results (DDFM results may not have all attributes)
    if hasattr(result, 'converged'):
        print(f"  - Converged: {result.converged}")
    if hasattr(result, 'num_iter'):
        print(f"  - Iterations: {result.num_iter}")
    if hasattr(result, 'loglik'):
        print(f"  - Log-likelihood: {result.loglik:.4f}")
    
    # Display DDFM-specific metadata if available
    if model_type == "ddfm" and 'encoder_layers' in metadata:
        print(f"  - Encoder layers: {metadata.get('encoder_layers')}")
        print(f"  - Number of factors: {metadata.get('num_factors')}")
    
    # Forecast
    horizon = cfg.get('forecast_horizon', None)
    if horizon:
        print(f"\nGenerating forecasts (horizon={horizon})...")
        X_forecast, Z_forecast = model.predict(horizon=horizon)
        print(f"  - Forecast shape: {X_forecast.shape}")
    
    # Optionally save model to outputs/ structure
    model_name = cfg.get('model_name') or f"experiment_{cfg.get('config_name', 'default')}"
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        config_path = cfg.get('config_path')  # Optional config file path to copy
        output_path = model.save_to_outputs(
            model_name=model_name,
            outputs_dir=outputs_dir,
            config_path=config_path
        )
        print(f"\nModel saved to: {output_path}")
    except Exception as e:
        # Convert to TrainingError for consistency, but don't raise (just warn)
        error_msg = f"Could not save model to outputs/ structure: {e}"
        print(f"\nWarning: {error_msg}")
        # Fallback: just note where results would be
        output_dir = outputs_dir / "experiments"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results available in: {output_dir}")
        # Note: We don't raise here as this is a non-critical operation
    
    print("=" * 70)


if __name__ == "__main__":
    main()

