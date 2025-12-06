"""Model wrappers for dfm-python package."""

from .dfm_models import DFM, DDFM

# Optional sktime forecasters (require sktime to be installed)
try:
    from .sktime_forecaster import DFMForecaster, DDFMForecaster
    __all__ = ['DFM', 'DDFM', 'DFMForecaster', 'DDFMForecaster']
except ImportError:
    __all__ = ['DFM', 'DDFM']

