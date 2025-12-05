"""Evaluation and result aggregation modules."""

from .evaluation import (
    calculate_standardized_metrics,
    calculate_metrics_per_horizon,
    evaluate_forecaster,
    compare_multiple_models,
    generate_comparison_table,
    save_comparison_plots,
    collect_all_comparison_results,
    aggregate_overall_performance,
    main_aggregator
)

__all__ = [
    'calculate_standardized_metrics',
    'calculate_metrics_per_horizon',
    'evaluate_forecaster',
    'compare_multiple_models',
    'generate_comparison_table',
    'save_comparison_plots',
    'collect_all_comparison_results',
    'aggregate_overall_performance',
    'main_aggregator'
]

