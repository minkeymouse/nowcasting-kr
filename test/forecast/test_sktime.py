"""Forecast-specific tests for sktime models.

These tests focus on forecasting functionality. For comprehensive model tests,
see test/test_sktime_models.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile
import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from sktime.forecasting.base import ForecastingHorizon
    from src.forecast.sktime import run_recursive_forecast, run_multi_horizon_forecast
    from src.train.sktime import train_sktime_model
    from src.utils import load_model_checkpoint
    SKTIME_AVAILABLE = True
except ImportError:
    SKTIME_AVAILABLE = False
