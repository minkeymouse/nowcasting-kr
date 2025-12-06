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
    main_aggregator,
    generate_latex_table_overall_metrics,
    generate_latex_table_by_target,
    generate_latex_table_by_horizon,
    generate_latex_table_36_rows,
    generate_latex_table_dataset_params,
    generate_all_latex_tables
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
    'main_aggregator',
    'generate_latex_table_overall_metrics',
    'generate_latex_table_by_target',
    'generate_latex_table_by_horizon',
    'generate_latex_table_36_rows',
    'generate_latex_table_dataset_params',
    'generate_all_latex_tables'
]

