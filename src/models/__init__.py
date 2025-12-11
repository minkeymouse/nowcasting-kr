"""Model implementations for ARIMA, VAR, DFM, DDFM, TFT, LSTM, and Chronos.

This package provides model-specific training and forecasting functionality.
"""

from .base import save_model_checkpoint, load_model_checkpoint

# Import available models (some may not exist)
try:
    from .arima import train_arima, forecast_arima
except ImportError:
    train_arima = None
    forecast_arima = None

try:
    from .var import train_var, forecast_var
except ImportError:
    train_var = None
    forecast_var = None

from .dfm import train_dfm, forecast_dfm
from .ddfm import train_ddfm, forecast_ddfm
from .tft import train_tft, forecast_tft
from .lstm import train_lstm, forecast_lstm
from .chronos import train_chronos, forecast_chronos

__all__ = [
    'save_model_checkpoint', 'load_model_checkpoint',
    'train_dfm', 'forecast_dfm',
    'train_ddfm', 'forecast_ddfm',
    'train_tft', 'forecast_tft',
    'train_lstm', 'forecast_lstm',
    'train_chronos', 'forecast_chronos',
]

# Add optional imports if available
if train_arima is not None:
    __all__.extend(['train_arima', 'forecast_arima'])
if train_var is not None:
    __all__.extend(['train_var', 'forecast_var'])

