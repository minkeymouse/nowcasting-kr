"""Evaluation helper functions for forecasting experiments.

This module provides standardized metric calculation functions for evaluating
forecasting model performance, including standardized MSE, MAE, and RMSE.

All functions are re-exported from sub-modules for backward compatibility.
"""

import logging
from src.utils import setup_logging

setup_logging()
_module_logger = logging.getLogger(__name__)
EXTREME_VALUE_THRESHOLD = 1e10

from .evaluation_metrics import (
    calculate_standardized_metrics,
    calculate_metrics_per_horizon
)
from .evaluation_forecaster import evaluate_forecaster
from .evaluation_aggregation import (
    compare_multiple_models,
    generate_comparison_table,
    collect_all_comparison_results,
    aggregate_overall_performance,
    analyze_correlation_structure,
    main_aggregator
)

__all__ = [
    'calculate_standardized_metrics',
    'calculate_metrics_per_horizon',
    'evaluate_forecaster',
    'compare_multiple_models',
    'generate_comparison_table',
    'collect_all_comparison_results',
    'aggregate_overall_performance',
    'analyze_correlation_structure',
    'main_aggregator',
    'EXTREME_VALUE_THRESHOLD'
]
