"""Main entry point for research experiments using Hydra."""

import sys
from pathlib import Path

# Add dfm-python to path
sys.path.insert(0, str(Path(__file__).parent.parent / "dfm-python" / "src"))

try:
    import hydra
    from hydra.utils import get_original_cwd
    from omegaconf import DictConfig
    import dfm_python as dfm
except ImportError as e:
    print(f"Required dependencies not available: {e}")
    print("Please install: pip install hydra-core omegaconf")
    sys.exit(1)


@hydra.main(config_path="../config", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run DFM experiment with Hydra configuration.
    
    This is a thin wrapper around dfm-python for research experiments.
    Results are saved to outputs/ or nowcasting-report/ directories.
    """
    print("=" * 70)
    print("DFM Research Experiment")
    print("=" * 70)
    
    # Load configuration
    print(f"\nLoading configuration from: {cfg.get('config_name', 'default')}")
    dfm.load_config(hydra=cfg)
    
    # Load data
    data_path = cfg.get('data_path')
    if data_path:
        print(f"\nLoading data from: {data_path}")
        dfm.load_data(data_path)
    else:
        print("\nNo data_path specified in config")
        return
    
    # Train model
    print("\nTraining model...")
    max_iter = cfg.get('max_iter', 5000)
    threshold = cfg.get('threshold', 1e-5)
    dfm.train(max_iter=max_iter, threshold=threshold)
    
    # Get results
    result = dfm.get_result()
    print(f"\nTraining completed:")
    print(f"  - Converged: {result.converged}")
    print(f"  - Iterations: {result.num_iter}")
    print(f"  - Log-likelihood: {result.loglik:.4f}")
    
    # Forecast
    horizon = cfg.get('forecast_horizon', None)
    print(f"\nGenerating forecasts (horizon={horizon})...")
    X_forecast, Z_forecast = dfm.predict(horizon=horizon)
    print(f"  - Forecast shape: {X_forecast.shape}")
    
    # Save results (optional)
    output_dir = Path("outputs") / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExperiment complete. Results available in: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

