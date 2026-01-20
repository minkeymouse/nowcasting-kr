"""Plot forecasted vs actual values for model outputs.

Creates time series plots showing original and forecasted values for comparison.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Dict
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.utils import get_project_root, load_data

logger = logging.getLogger(__name__)


def get_default_target_series(data_model: str) -> list[str]:
    """Get default target series for a given dataset.
    
    Parameters
    ----------
    data_model : str
        Dataset name: "investment" or "production"
    
    Returns
    -------
    list of str
        Default target series for the dataset
    """
    target_map = {
        'investment': ['KOEQUIPTE'],
        'production': ['KOIPALL.G']
    }
    
    data_model_lower = data_model.lower()
    if data_model_lower not in target_map:
        raise ValueError(f"Unknown data_model: {data_model}. Must be 'investment' or 'production'")
    
    return target_map[data_model_lower]


def load_forecast_data(
    data_model: str,
    model_name: str,
    experiment_type: str = "short_term"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load predictions and actuals from output files.
    
    Parameters
    ----------
    data_model : str
        Dataset name: "investment" or "production"
    model_name : str
        Model name: "tft", "patchtst", "itf", etc.
    experiment_type : str, default "short_term"
        Experiment type: "short_term" or "long_term"
    
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (predictions, actuals) DataFrames with datetime index
    """
    project_root = get_project_root()
    output_dir = project_root / "outputs" / experiment_type / data_model / model_name
    
    pred_file = output_dir / "predictions.csv"
    actual_file = output_dir / "actuals.csv"
    
    if not pred_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_file}")
    if not actual_file.exists():
        raise FileNotFoundError(f"Actuals file not found: {actual_file}")
    
    # Load data
    predictions = pd.read_csv(pred_file, index_col=0, parse_dates=True)
    actuals = pd.read_csv(actual_file, index_col=0, parse_dates=True)
    
    logger.info(f"Loaded {model_name} predictions: {len(predictions)} rows, {len(predictions.columns)} series")
    logger.info(f"Loaded {model_name} actuals: {len(actuals)} rows, {len(actuals.columns)} series")
    
    return predictions, actuals


def load_historical_data(data_model: str, target_series: list[str]) -> pd.DataFrame:
    """Load historical data from original CSV file.
    
    Parameters
    ----------
    data_model : str
        Dataset name: "investment" or "production"
    target_series : list of str
        Target series names to extract
    
    Returns
    -------
    pd.DataFrame
        Historical data with datetime index, containing target_series columns
    """
    project_root = get_project_root()
    data_path = project_root / "data" / f"{data_model}.csv"
    
    if not data_path.exists():
        logger.warning(f"Historical data file not found: {data_path}")
        return pd.DataFrame()
    
    data = load_data(str(data_path))
    
    # Set datetime index
    if 'date_w' in data.columns:
        data['date_w'] = pd.to_datetime(data['date_w'], errors='coerce')
        data = data.set_index('date_w')
        data.index.name = 'date'
    elif 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        data = data.set_index('date')
    else:
        logger.warning("No date column found in historical data")
        return pd.DataFrame()
    
    # Filter to target series that exist
    available_series = [s for s in target_series if s in data.columns]
    if not available_series:
        logger.warning(f"No target series found in historical data")
        return pd.DataFrame()
    
    historical = data[available_series].copy()
    historical = historical.sort_index()
    
    logger.info(f"Loaded historical data: {len(historical)} rows, date range: {historical.index.min()} to {historical.index.max()}")
    
    return historical


def load_attention_model_forecasts(
    data_model: str,
    experiment_type: str = "short_term"
) -> Dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Load predictions and actuals for attention models.
    
    Parameters
    ----------
    data_model : str
        Dataset name: "investment" or "production"
    experiment_type : str, default "short_term"
        Experiment type: "short_term" or "long_term"
    
    Returns
    -------
    dict
        Dictionary mapping model_name -> (predictions, actuals)
    """
    models = ['tft', 'patchtst', 'itf']
    results = {}
    
    for model in models:
        try:
            preds, acts = load_forecast_data(data_model, model, experiment_type)
            results[model] = (preds, acts)
        except FileNotFoundError as e:
            logger.warning(f"Skipping {model}: {e}")
            continue
    
    return results


def load_ssm_model_forecasts(
    data_model: str,
    experiment_type: str = "short_term"
) -> Dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Load predictions and actuals for SSM models (DFM, DDFM, Mamba).
    
    Parameters
    ----------
    data_model : str
        Dataset name: "investment" or "production"
    experiment_type : str, default "short_term"
        Experiment type: "short_term" or "long_term"
    
    Returns
    -------
    dict
        Dictionary mapping model_name -> (predictions, actuals)
    """
    models = ['dfm', 'ddfm', 'mamba']
    results = {}
    
    for model in models:
        try:
            preds, acts = load_forecast_data(data_model, model, experiment_type)
            results[model] = (preds, acts)
        except FileNotFoundError as e:
            logger.warning(f"Skipping {model}: {e}")
            continue
    
    return results


def plot_combined_attention_forecasts(
    n_months: int = 20,
    experiment_type: str = "short_term",
    output_path: Optional[Path] = None
) -> None:
    """Create combined forecast plot with production (left) and investment (right).
    
    Creates a figure with 1x2 subplots (1 row, 2 columns):
    - Left: Production data - Actual + all three models
    - Right: Investment data - Actual + all three models
    
    Parameters
    ----------
    n_months : int, default 20
        Number of months to display for forecast period (from the end)
    experiment_type : str, default "short_term"
        Experiment type: "short_term" or "long_term"
    output_path : Path, optional
        Path to save the plot. If None, displays interactively.
    """
    # Use non-interactive backend if saving to file
    if output_path:
        matplotlib.use('Agg')
    
    # Model colors and styles
    model_styles = {
        'tft': {'color': '#F77F00', 'marker': '^', 'label': 'TFT'},
        'patchtst': {'color': '#FCBF49', 'marker': 's', 'label': 'PatchTST'},
        'itf': {'color': '#D62828', 'marker': 'D', 'label': 'iTransformer'}
    }
    
    data_models = ['production', 'investment']
    target_map = {
        'production': 'KOIPALL.G',
        'investment': 'KOEQUIPTE'
    }
    
    # Create figure with 1 row, 2 columns (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    
    for idx, data_model in enumerate(data_models):
        ax = axes[idx]
        target_series = get_default_target_series(data_model)
        
        # Load all model forecasts
        model_data = load_attention_model_forecasts(data_model, experiment_type)
        if not model_data:
            logger.warning(f"No model forecasts found for {data_model}")
            continue
        
        # Get actuals from any model
        first_model = list(model_data.keys())[0]
        _, actuals = model_data[first_model]
        
        # Get target series
        series_name = target_series[0] if target_series and target_series[0] in actuals.columns else None
        if not series_name:
            logger.warning(f"Target series not found for {data_model}")
            continue
        
        # Get actuals for forecast period
        actual_series = actuals[series_name].dropna()
        
        # Get last n_months of forecast period
        if len(actual_series) > n_months:
            forecast_period = actual_series.tail(n_months)
        else:
            forecast_period = actual_series
        
        # Plot actual
        ax.plot(forecast_period.index, forecast_period.values, 
                'o-', color='#2E86AB', linewidth=2, markersize=6, 
                label='Actual', alpha=0.9, zorder=10)
        
        # Plot each model's forecasts
        for model_name, (preds, _) in model_data.items():
            if series_name in preds.columns:
                pred_series = preds[series_name].dropna()
                # Align with forecast_period dates
                common_dates = pred_series.index.intersection(forecast_period.index)
                if len(common_dates) > 0:
                    pred_aligned = pred_series.loc[common_dates]
                    if len(pred_aligned) > n_months:
                        pred_aligned = pred_aligned.tail(n_months)
                    
                    style = model_styles.get(model_name, {'color': '#000000', 'marker': 'o', 'label': model_name.upper()})
                    ax.plot(pred_aligned.index, pred_aligned.values, 
                            f'{style["marker"]}--', color=style['color'], linewidth=2, 
                            markersize=6, label=style['label'], alpha=0.8)
        
        # Set title and labels
        ax.set_title(f'{data_model.title()}: {series_name}', fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=9, loc='best', ncol=4)
        
        # Format x-axis dates
        n_ticks = min(6, len(forecast_period))
        interval = max(1, len(forecast_period) // n_ticks) if len(forecast_period) > 0 else 1
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Adjust layout (no title)
    plt.tight_layout()
    
    # Save or show
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_forecasts(
    data_model: str,
    n_months: int = 20,
    experiment_type: str = "short_term",
    output_path: Optional[Path] = None
) -> None:
    """Create forecast plots with attention models (legacy function)."""
    # Redirect to combined plot if both datasets are requested
    # For now, keep original behavior but can be deprecated
    plot_combined_attention_forecasts(n_months, experiment_type, output_path)


def plot_combined_ssm_forecasts(
    n_months: int = 20,
    experiment_type: str = "short_term",
    output_path: Optional[Path] = None
) -> None:
    """Create combined forecast plot with production (left) and investment (right).
    
    Shows DFM, DDFM, and Mamba forecasts with actuals.
    """
    if output_path:
        matplotlib.use('Agg')
    
    model_styles = {
        'dfm': {'color': '#06A77D', 'marker': 'o', 'label': 'DFM'},
        'ddfm': {'color': '#4D908E', 'marker': 's', 'label': 'DDFM'},
        'mamba': {'color': '#577590', 'marker': 'D', 'label': 'Mamba'}
    }
    
    data_models = ['production', 'investment']
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    
    for idx, data_model in enumerate(data_models):
        ax = axes[idx]
        target_series = get_default_target_series(data_model)
        
        model_data = load_ssm_model_forecasts(data_model, experiment_type)
        if not model_data:
            logger.warning(f"No model forecasts found for {data_model}")
            continue
        
        first_model = list(model_data.keys())[0]
        _, actuals = model_data[first_model]
        
        series_name = target_series[0] if target_series and target_series[0] in actuals.columns else None
        if not series_name:
            logger.warning(f"Target series not found for {data_model}")
            continue
        
        actual_series = actuals[series_name].dropna()
        forecast_period = actual_series.tail(n_months) if len(actual_series) > n_months else actual_series
        
        ax.plot(
            forecast_period.index,
            forecast_period.values,
            'o-',
            color='#2E86AB',
            linewidth=2,
            markersize=6,
            label='Actual',
            alpha=0.9,
            zorder=10
        )
        
        for model_name, (preds, _) in model_data.items():
            if series_name in preds.columns:
                pred_series = preds[series_name].dropna()
                common_dates = pred_series.index.intersection(forecast_period.index)
                if len(common_dates) > 0:
                    pred_aligned = pred_series.loc[common_dates]
                    if len(pred_aligned) > n_months:
                        pred_aligned = pred_aligned.tail(n_months)
                    
                    style = model_styles.get(model_name, {'color': '#000000', 'marker': 'o', 'label': model_name.upper()})
                    ax.plot(
                        pred_aligned.index,
                        pred_aligned.values,
                        f'{style["marker"]}--',
                        color=style['color'],
                        linewidth=2,
                        markersize=6,
                        label=style['label'],
                        alpha=0.8
                    )
        
        ax.set_title(f'{data_model.title()}: {series_name}', fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=9, loc='best', ncol=4)
        
        n_ticks = min(6, len(forecast_period))
        interval = max(1, len(forecast_period) // n_ticks) if len(forecast_period) > 0 else 1
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main entry point for plotting forecasts."""
    parser = argparse.ArgumentParser(
        description="Plot forecasted vs actual values for model outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create combined plot (production top, investment bottom)
  python -m src.paper.plot_forecast
  
  # Custom output location
  python -m src.paper.plot_forecast --output custom/path/plot.png
  
  # Last 15 months
  python -m src.paper.plot_forecast --n-months 15
        """
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for saving the plot. Default: nowcasting-report/images/combined_attention_forecast.png'
    )
    
    parser.add_argument(
        '--n-months',
        type=int,
        default=20,
        help='Number of months to display for forecast period (from the end). Default: 20'
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        default='short_term',
        choices=['short_term', 'long_term'],
        help='Experiment type: short_term or long_term. Default: short_term'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='attention',
        choices=['attention', 'ssm'],
        help='Plot group: attention or ssm. Default: attention'
    )
    
    parser.add_argument(
        '--ssm',
        action='store_true',
        help='Deprecated: use --model ssm. Generates combined SSM forecast plot.'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set default output path if not provided
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = get_project_root() / "nowcasting-report" / "images" / "combined_attention_forecast.png"
    
    plot_group = 'ssm' if args.ssm else args.model
    
    if plot_group == 'ssm':
        plot_combined_ssm_forecasts(
            n_months=args.n_months,
            experiment_type=args.experiment,
            output_path=output_path
        )
    else:
        # Create combined plot with production (left) and investment (right)
        plot_combined_attention_forecasts(
            n_months=args.n_months,
            experiment_type=args.experiment,
            output_path=output_path
        )


if __name__ == '__main__':
    main()
