"""Test training functions for DFM and DDFM models."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile
import shutil

# Add paths: project root and dfm-python src
project_root = Path(__file__).parent.parent.parent
dfm_python_src = project_root / "dfm-python" / "src"
sys.path.insert(0, str(dfm_python_src))
sys.path.insert(0, str(project_root))

from src.train.dfm_python import train_dfm_python_model


def test_train_ddfm_with_single_target():
    """Test DDFM training with single target series."""
    # Create synthetic data
    np.random.seed(42)
    n_samples, n_features = 100, 5
    data = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=['KOEQUIPTE', 'KOWRCCNSE', 'KOIPALL.G', 'A001', 'GSCITOT']
    )
    
    # Config with single target
    model_cfg_dict = {
        'name': 'ddfm',
        'target_series': ['KOEQUIPTE'],
        'encoder_layers': [16, 8],  # Smaller for faster testing
        'num_factors': 2,
        'max_epoch': 2,  # Minimal epochs for testing
        'learning_rate': 0.01,
        'target_scaler': 'robust'
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        train_dfm_python_model(
            model_type='ddfm',
            config_name='test',
            cfg=None,
            data=data,
            model_name='ddfm_test',
            horizons=None,
            outputs_dir=outputs_dir,
            model_cfg_dict=model_cfg_dict
        )
        
        # Verify model was saved
        model_path = outputs_dir / "model.pkl"
        dataset_path = outputs_dir / "dataset.pkl"
        
        assert model_path.exists(), "Model file should be created"
        assert dataset_path.exists(), "Dataset file should be created"
        
        # Load and verify
        import joblib
        model = joblib.load(model_path)
        dataset = joblib.load(dataset_path)
        
        assert hasattr(model, 'fit'), "Model should have fit method"
        assert hasattr(dataset, 'target_series'), "Dataset should have target_series"
        assert 'KOEQUIPTE' in dataset.target_series, "Dataset should contain target series"


def test_train_ddfm_with_all_targets():
    """Test DDFM training with all series as targets (target_series=null)."""
    # Create synthetic data
    np.random.seed(42)
    n_samples, n_features = 100, 5
    data = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=['KOEQUIPTE', 'KOWRCCNSE', 'KOIPALL.G', 'A001', 'GSCITOT']
    )
    
    # Config with null target_series (use all)
    model_cfg_dict = {
        'name': 'ddfm',
        'target_series': None,  # None means use all
        'encoder_layers': [16, 8],
        'num_factors': 2,
        'max_epoch': 2,
        'learning_rate': 0.01,
        'target_scaler': 'robust'
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        train_dfm_python_model(
            model_type='ddfm',
            config_name='test',
            cfg=None,
            data=data,
            model_name='ddfm_test_all',
            horizons=None,
            outputs_dir=outputs_dir,
            model_cfg_dict=model_cfg_dict
        )
        
        # Verify model was saved
        model_path = outputs_dir / "model.pkl"
        dataset_path = outputs_dir / "dataset.pkl"
        
        assert model_path.exists(), "Model file should be created"
        assert dataset_path.exists(), "Dataset file should be created"
        
        # Load and verify all columns are targets
        import joblib
        dataset = joblib.load(dataset_path)
        
        assert len(dataset.target_series) == n_features, f"All {n_features} columns should be targets"
        assert dataset.feature_shape[1] == 0, "No feature columns when all are targets"


def test_train_ddfm_with_dictconfig():
    """Test DDFM training with Hydra DictConfig."""
    try:
        from omegaconf import DictConfig, OmegaConf
    except ImportError:
        # Skip if OmegaConf not available
        return
    
    # Create synthetic data
    np.random.seed(42)
    data = pd.DataFrame(
        np.random.randn(100, 5),
        columns=['KOEQUIPTE', 'KOWRCCNSE', 'KOIPALL.G', 'A001', 'GSCITOT']
    )
    
    # Create DictConfig (simulating Hydra)
    cfg_dict = {
        'name': 'ddfm',
        'target_series': ['KOEQUIPTE'],
        'encoder_layers': [16, 8],
        'num_factors': 2,
        'max_epoch': 2,
        'learning_rate': 0.01
    }
    dict_config = OmegaConf.create(cfg_dict)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        train_dfm_python_model(
            model_type='ddfm',
            config_name='test',
            cfg=None,
            data=data,
            model_name='ddfm_test_dictconfig',
            horizons=None,
            outputs_dir=outputs_dir,
            model_cfg_dict=dict_config  # Pass DictConfig directly
        )
        
        model_path = outputs_dir / "model.pkl"
        assert model_path.exists(), "Model should be created from DictConfig"


def test_train_dfm_with_targets():
    """Test DFM training with target series."""
    # Create synthetic data with proper structure
    np.random.seed(42)
    n_samples, n_features = 100, 5
    data = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=['KOEQUIPTE', 'KOWRCCNSE', 'KOIPALL.G', 'A001', 'GSCITOT']
    )
    
    # DFM config needs blocks structure
    model_cfg_dict = {
        'name': 'dfm',
        'target_series': ['KOEQUIPTE'],
        'clock': 'w',
        'blocks': {
            'Block_Global': {
                'num_factors': 2,
                'series': list(data.columns)
            }
        },
        'ar_order': 1,
        'max_iter': 5,  # Minimal iterations for testing
        'threshold': 1e-3
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        train_dfm_python_model(
            model_type='dfm',
            config_name='test',
            cfg=None,
            data=data,
            model_name='dfm_test',
            horizons=None,
            outputs_dir=outputs_dir,
            model_cfg_dict=model_cfg_dict
        )
        
        # Verify model was saved
        model_path = outputs_dir / "model.pkl"
        assert model_path.exists(), "DFM model file should be created"
        
        import joblib
        model = joblib.load(model_path)
        assert hasattr(model, 'fit'), "DFM model should have fit method"


def test_target_series_filtering():
    """Test that target series filtering works correctly."""
    np.random.seed(42)
    data = pd.DataFrame(
        np.random.randn(100, 3),
        columns=['KOEQUIPTE', 'KOWRCCNSE', 'KOIPALL.G']
    )
    
    # Target series that includes non-existent column
    model_cfg_dict = {
        'name': 'ddfm',
        'target_series': ['KOEQUIPTE', 'NONEXISTENT'],
        'encoder_layers': [16, 8],
        'num_factors': 2,
        'max_epoch': 2
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        train_dfm_python_model(
            model_type='ddfm',
            config_name='test',
            cfg=None,
            data=data,
            model_name='ddfm_test_filter',
            horizons=None,
            outputs_dir=outputs_dir,
            model_cfg_dict=model_cfg_dict
        )
        
        # Should filter to only existing columns
        import joblib
        dataset = joblib.load(outputs_dir / "dataset.pkl")
        assert 'KOEQUIPTE' in dataset.target_series
        assert 'NONEXISTENT' not in dataset.target_series


def test_empty_target_series_falls_back_to_all():
    """Test that empty target_series uses all columns."""
    np.random.seed(42)
    data = pd.DataFrame(
        np.random.randn(100, 3),
        columns=['KOEQUIPTE', 'KOWRCCNSE', 'KOIPALL.G']
    )
    
    # Empty target_series
    model_cfg_dict = {
        'name': 'ddfm',
        'target_series': [],
        'encoder_layers': [16, 8],
        'num_factors': 2,
        'max_epoch': 2
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        train_dfm_python_model(
            model_type='ddfm',
            config_name='test',
            cfg=None,
            data=data,
            model_name='ddfm_test_empty',
            horizons=None,
            outputs_dir=outputs_dir,
            model_cfg_dict=model_cfg_dict
        )
        
        import joblib
        dataset = joblib.load(outputs_dir / "dataset.pkl")
        assert len(dataset.target_series) == 3, "Should use all 3 columns when empty list"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
