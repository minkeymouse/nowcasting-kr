"""Test Hydra configuration loading with new nested config structure."""

import sys
from pathlib import Path
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_hydra_config_loading():
    """Test that Hydra configs can be loaded with nested paths."""
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
    except ImportError:
        pytest.skip("Hydra not available")
    
    config_dir = project_root / "config"
    config_dir_abs = config_dir.resolve()  # Get absolute path
    
    with initialize_config_dir(config_dir=str(config_dir_abs), version_base=None):
        # Test DFM nested config
        cfg = compose(config_name="default", overrides=["model=dfm/investment", "data=investment"])
        
        assert cfg.model.name == "dfm", "Model name should be dfm"
        assert cfg.data == "investment", "Data should be investment"
        assert "target_series" in cfg.model, "Model config should have target_series"
        assert "KOEQUIPTE" in cfg.model.target_series, "Investment config should have KOEQUIPTE"
        
        # Test DFM consumption variant
        cfg = compose(config_name="default", overrides=["model=dfm/consumption", "data=consumption"])
        assert "KOWRCCNSE" in cfg.model.target_series, "Consumption config should have KOWRCCNSE"
        
        # Test DFM production variant
        cfg = compose(config_name="default", overrides=["model=dfm/production", "data=production"])
        assert "KOIPALL.G" in cfg.model.target_series, "Production config should have KOIPALL.G"
        
        # Test DDFM (uses default.yaml in model/ddfm/ directory)
        # Note: For models with only default.yaml, we still reference as model=ddfm
        # but Hydra looks for model/ddfm/default.yaml automatically
        try:
            cfg = compose(config_name="default", overrides=["model=ddfm", "data=investment"])
            assert cfg.model.name == "ddfm", "Model name should be ddfm"
        except Exception:
            # If direct reference doesn't work, try with explicit default
            cfg = compose(config_name="default", overrides=["model=ddfm/default", "data=investment"])
            assert cfg.model.name == "ddfm", "Model name should be ddfm"


def test_all_model_configs_loadable():
    """Test that all model configs can be loaded."""
    try:
        from hydra import compose, initialize_config_dir
    except ImportError:
        pytest.skip("Hydra not available")
    
    config_dir = project_root / "config"
    config_dir_abs = config_dir.resolve()
    
    models_to_test = [
        "dfm/investment",
        "dfm/consumption", 
        "dfm/production",
        "ddfm",
        "itf",
        "patchtst",
        "tft",
        "timemixer",
    ]
    
    with initialize_config_dir(config_dir=str(config_dir_abs), version_base=None):
        for model in models_to_test:
            try:
                cfg = compose(config_name="default", overrides=[f"model={model}", "data=investment"])
                assert cfg.model is not None, f"Config for {model} should load"
                assert hasattr(cfg.model, 'name') or 'name' in cfg.model, f"Config for {model} should have name"
            except Exception as e:
                pytest.fail(f"Failed to load config for {model}: {e}")


def test_config_structure_consistency():
    """Test that configs have consistent structure."""
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
    except ImportError:
        pytest.skip("Hydra not available")
    
    config_dir = project_root / "config"
    config_dir_abs = config_dir.resolve()
    
    with initialize_config_dir(config_dir=str(config_dir_abs), version_base=None):
        # Test DFM config structure
        cfg = compose(config_name="default", overrides=["model=dfm/investment"])
        
        # Convert to dict for easier inspection
        model_dict = OmegaConf.to_container(cfg.model, resolve=True)
        
        # DFM should have required fields
        assert "name" in model_dict, "DFM config should have 'name'"
        assert "target_series" in model_dict, "DFM config should have 'target_series'"
        assert "blocks" in model_dict, "DFM config should have 'blocks'"
        assert "clock" in model_dict, "DFM config should have 'clock'"
        
        # Test DDFM config structure
        cfg = compose(config_name="default", overrides=["model=ddfm"])
        model_dict = OmegaConf.to_container(cfg.model, resolve=True)
        
        assert "name" in model_dict, "DDFM config should have 'name'"
        assert "encoder_layers" in model_dict, "DDFM config should have 'encoder_layers'"
        assert "num_factors" in model_dict, "DDFM config should have 'num_factors'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
