"""Tests for Mamba model training."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
import tempfile
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Check model availability
try:
    from mamba_ssm import Mamba2
    MAMBA_AVAILABLE = True
except ImportError as e:
    MAMBA_AVAILABLE = False
    IMPORT_ERROR = str(e)

from src.train.mamba import train_mamba_model, MambaForecaster, TimeSeriesDataset


@pytest.fixture
def sample_data():
    """Create sample time series data."""
    dates = pd.date_range('2020-01-01', periods=150, freq='W')
    return pd.DataFrame(
        np.random.randn(150, 3),
        columns=['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
        index=dates
    )


@pytest.fixture
def mock_data_loader():
    """Create a mock data loader."""
    class MockDataLoader:
        def __init__(self):
            dates = pd.date_range('2020-01-01', periods=150, freq='W')
            self.original = pd.DataFrame(
                np.random.randn(150, 3) * 10 + 100,
                columns=['KOEQUIPTE', 'KOWRCCNSE', 'A001'],
                index=dates
            )
            self.processed = self.original.copy()
    return MockDataLoader()


@pytest.mark.skipif(not MAMBA_AVAILABLE, reason=f"Mamba not available: {IMPORT_ERROR if not MAMBA_AVAILABLE else ''}")
def test_train_mamba_basic(sample_data, mock_data_loader):
    """Test basic Mamba model training."""
    model_params = {
        'target_series': ['KOEQUIPTE'],
        'prediction_length': 4,
        'context_length': 32,
        'n_layers': 2,
        'd_model': 64,  # Works with Mamba2 default headdim=64
        'd_state': 64,  # Mamba2 default
        'max_epochs': 1,
        'batch_size': 8,
        'learning_rate': 0.001,
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs_dir = Path(tmpdir)
        
        train_mamba_model(
            model_type='mamba',
            cfg=None,
            data=sample_data,
            model_name='mamba_test',
            outputs_dir=outputs_dir,
            model_params=model_params,
            data_loader=mock_data_loader
        )
        
        # Verify model was saved
        model_path = outputs_dir / "model.pkl"
        assert model_path.exists(), "Mamba model should be saved"
        
        # Verify metadata was saved
        metadata_path = outputs_dir / "metadata.pkl"
        assert metadata_path.exists(), "Metadata should be saved"


@pytest.mark.skipif(not MAMBA_AVAILABLE, reason=f"Mamba not available: {IMPORT_ERROR if not MAMBA_AVAILABLE else ''}")
def test_mamba_forecaster_forward():
    """Test MambaForecaster forward pass."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use d_model that works with Mamba2's default headdim=64
    # d_ssm = expand * d_model = 2 * 64 = 128, which is divisible by headdim=64
    model = MambaForecaster(
        d_model=64,
        n_layers=2,
        context_length=32,
        prediction_length=4,
        d_state=64,  # Mamba2 default
        d_conv=4,
        expand=2,
        dropout=0.1,
        device=device
    ).to(device)
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 32, 64).to(device)
    
    with torch.no_grad():
        y = model(x)
    
    assert y.shape == (batch_size, 4, 64), f"Expected shape (2, 4, 64), got {y.shape}"


@pytest.mark.skipif(not MAMBA_AVAILABLE, reason=f"Mamba not available: {IMPORT_ERROR if not MAMBA_AVAILABLE else ''}")
def test_mamba_forecaster_with_projection():
    """Test MambaForecaster with input/output projection."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_features = 3
    d_model = 64  # Works with Mamba2 default headdim=64 (d_ssm=128, 128%64=0)
    
    model = MambaForecaster(
        d_model=d_model,
        n_layers=2,
        context_length=32,
        prediction_length=4,
        d_state=64,  # Mamba2 default
        device=device
    ).to(device)
    
    input_proj = torch.nn.Linear(n_features, d_model).to(device)
    output_proj = torch.nn.Linear(d_model, n_features).to(device)
    
    # Test forward pass with projection
    batch_size = 2
    x = torch.randn(batch_size, 32, n_features).to(device)
    
    with torch.no_grad():
        x_proj = input_proj(x)  # (B, 32, d_model)
        y = model(x_proj)  # (B, 4, d_model)
        y_proj = output_proj(y)  # (B, 4, n_features)
    
    assert y_proj.shape == (batch_size, 4, n_features), f"Expected shape (2, 4, 3), got {y_proj.shape}"


@pytest.mark.skipif(not MAMBA_AVAILABLE, reason=f"Mamba not available: {IMPORT_ERROR if not MAMBA_AVAILABLE else ''}")
def test_time_series_dataset():
    """Test TimeSeriesDataset."""
    data = np.random.randn(100, 3).astype(np.float32)
    context_length = 32
    prediction_length = 4
    
    dataset = TimeSeriesDataset(data, context_length, prediction_length)
    
    assert len(dataset) > 0, "Dataset should have samples"
    
    # Test getting a sample
    x, y = dataset[0]
    
    assert x.shape == (context_length, 3), f"Expected x shape ({context_length}, 3), got {x.shape}"
    assert y.shape == (prediction_length, 3), f"Expected y shape ({prediction_length}, 3), got {y.shape}"


@pytest.mark.skipif(not MAMBA_AVAILABLE, reason=f"Mamba not available: {IMPORT_ERROR if not MAMBA_AVAILABLE else ''}")
def test_mamba2_direct_usage():
    """Test Mamba2 direct usage (as shown in README).
    
    Note: This test may fail if causal-conv1d is not installed.
    Mamba2 uses causal-conv1d for optimized operations, but can fallback
    to PyTorch conv1d if use_mem_eff_path=False.
    """
    # Check if causal-conv1d is available
    try:
        from causal_conv1d import causal_conv1d_fn
        has_causal_conv1d = True
    except ImportError:
        has_causal_conv1d = False
    
    if not has_causal_conv1d:
        pytest.skip("causal-conv1d not available. Install it for full Mamba2 support.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    batch, length, dim = 2, 64, 16
    x = torch.randn(batch, length, dim).to(device)
    
    # Mamba2 requires: d_ssm % headdim == 0
    # d_ssm = d_inner = expand * d_model = 2 * 16 = 32
    # headdim must divide 32, so use headdim=16 or 32
    model = Mamba2(
        d_model=dim,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=16,  # Must divide d_ssm (32)
        device=device
    ).to(device)
    
    y = model(x)
    assert y.shape == x.shape, f"Mamba2 should maintain input shape: {x.shape} -> {y.shape}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
