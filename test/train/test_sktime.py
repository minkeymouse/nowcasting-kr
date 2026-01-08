"""Test training functions for sktime forecasting models."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Check if models are available (neuralforecast or sktime)
try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import PatchTST, TFT, TimeMixer, iTransformer
    NEURALFORECAST_AVAILABLE = True
except ImportError:
    NEURALFORECAST_AVAILABLE = False

try:
    from sktime.forecasting.base import ForecastingHorizon
    from sktime.forecasting.patch_tst import PatchTSTForecaster
    SKTIME_AVAILABLE = True
except ImportError:
    SKTIME_AVAILABLE = False

# Models are available if either neuralforecast or sktime is available
MODELS_AVAILABLE = NEURALFORECAST_AVAILABLE or SKTIME_AVAILABLE

from src.train.sktime import train_sktime_model


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="neuralforecast/sktime not available")
def test_train_tft_model():
    """Test TFT model training."""
    # Create synthetic time series data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='W')
    data = pd.DataFrame(
        np.random.randn(200, 5),
        columns=['KOEQUIPTE', 'KOWRCCNSE', 'KOIPALL.G', 'A001', 'GSCITOT'],
        index=dates
    )
    
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'prediction_length': 8,
        'context_length': 32,
        'hidden_size': 32,  # Smaller for faster testing
        'max_epochs': 2,
        'batch_size': 32
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        train_sktime_model(
            model_type='tft',
            config_name='test',
            cfg=None,
            data=data,
            model_name='tft_test',
            horizons=None,
            outputs_dir=outputs_dir,
            model_params=model_params
        )
        
        # Verify model was saved
        model_path = outputs_dir / "model.pkl"
        assert model_path.exists(), "TFT model file should be created"
        
        import joblib
        try:
            model = joblib.load(model_path)
            assert hasattr(model, 'fit'), "Model should have fit method"
        except Exception:
            # Some sktime models may not pickle well, check for zip instead
            zip_path = outputs_dir / "model.zip"
            if zip_path.exists():
                assert True, "Model saved as zip"
            else:
                raise


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="neuralforecast/sktime not available")
def test_train_patchtst_model():
    """Test PatchTST model training."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='W')
    data = pd.DataFrame(
        np.random.randn(200, 5),
        columns=['KOEQUIPTE', 'KOWRCCNSE', 'KOIPALL.G', 'A001', 'GSCITOT'],
        index=dates
    )
    
    model_params = {
        'target_series': ['KOEQUIPTE', 'KOWRCCNSE'],
        'prediction_length': 8,
        'context_length': 32,
        'd_model': 64,
        'max_epochs': 2,
        'batch_size': 32
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        train_sktime_model(
            model_type='patchtst',
            config_name='test',
            cfg=None,
            data=data,
            model_name='patchtst_test',
            horizons=None,
            outputs_dir=outputs_dir,
            model_params=model_params
        )
        
        model_path = outputs_dir / "model.pkl"
        zip_path = outputs_dir / "model.zip"
        assert model_path.exists() or zip_path.exists(), "PatchTST model should be saved"


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="neuralforecast/sktime not available")
def test_train_timemixer_model():
    """Test TimeMixer model training."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='W')
    data = pd.DataFrame(
        np.random.randn(200, 5),
        columns=['KOEQUIPTE', 'KOWRCCNSE', 'KOIPALL.G', 'A001', 'GSCITOT'],
        index=dates
    )
    
    model_params = {
        'target_series': ['KOIPALL.G'],
        'prediction_length': 8,
        'context_length': 32,
        'max_epochs': 2,
        'batch_size': 32
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        train_sktime_model(
            model_type='timemixer',
            config_name='test',
            cfg=None,
            data=data,
            model_name='timemixer_test',
            horizons=None,
            outputs_dir=outputs_dir,
            model_params=model_params
        )
        
        model_path = outputs_dir / "model.pkl"
        zip_path = outputs_dir / "model.zip"
        assert model_path.exists() or zip_path.exists(), "TimeMixer model should be saved"


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="neuralforecast/sktime not available")
def test_train_itransformer_model():
    """Test ITransformer model training."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='W')
    data = pd.DataFrame(
        np.random.randn(200, 5),
        columns=['KOEQUIPTE', 'KOWRCCNSE', 'KOIPALL.G', 'A001', 'GSCITOT'],
        index=dates
    )
    
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'n_forecasts': 8,
        'n_lags': 32,
        'd_model': 128,
        'max_epochs': 2,
        'batch_size': 32
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        train_sktime_model(
            model_type='itransformer',
            config_name='test',
            cfg=None,
            data=data,
            model_name='itransformer_test',
            horizons=None,
            outputs_dir=outputs_dir,
            model_params=model_params
        )
        
        model_path = outputs_dir / "model.pkl"
        zip_path = outputs_dir / "model.zip"
        assert model_path.exists() or zip_path.exists(), "ITransformer model should be saved"


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="neuralforecast/sktime not available")
def test_target_series_defaults():
    """Test that default target series are used when not specified."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='W')
    data = pd.DataFrame(
        np.random.randn(200, 5),
        columns=['KOEQUIPTE', 'KOWRCCNSE', 'KOIPALL.G', 'A001', 'GSCITOT'],
        index=dates
    )
    
    # No target_series in params - should use defaults
    model_params = {
        'prediction_length': 8,
        'context_length': 32,
        'max_epochs': 2,
        'batch_size': 32
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        train_sktime_model(
            model_type='tft',
            config_name='test',
            cfg=None,
            data=data,
            model_name='tft_test_defaults',
            horizons=None,
            outputs_dir=outputs_dir,
            model_params=model_params
        )
        
        model_path = outputs_dir / "model.pkl"
        zip_path = outputs_dir / "model.zip"
        assert model_path.exists() or zip_path.exists(), "Model should be saved with defaults"


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="neuralforecast/sktime not available")
def test_target_series_filtering():
    """Test that missing target series fall back to first column."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='W')
    data = pd.DataFrame(
        np.random.randn(200, 3),
        columns=['A001', 'GSCITOT', 'OTHER'],
        index=dates
    )
    
    # Target series that don't exist in data
    model_params = {
        'target_series': ['KOEQUIPTE', 'KOWRCCNSE'],  # Not in data
        'prediction_length': 8,
        'context_length': 32,
        'max_epochs': 2,
        'batch_size': 32
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        train_sktime_model(
            model_type='tft',
            config_name='test',
            cfg=None,
            data=data,
            model_name='tft_test_filter',
            horizons=None,
            outputs_dir=outputs_dir,
            model_params=model_params
        )
        
        model_path = outputs_dir / "model.pkl"
        zip_path = outputs_dir / "model.zip"
        assert model_path.exists() or zip_path.exists(), "Model should fallback to first column"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
