"""Plot creation code for forecast vs actual comparisons.

This module generates plots comparing forecasts from different models with actual values:
1. forecast_vs_actual_all: Comparison of all models' forecasts vs actual values for all targets

All plots are saved to nowcasting-report/forecast/images/ directory as *.png files.
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
_script_dir = Path(__file__).parent
PROJECT_ROOT = _script_dir.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import utilities after adding to path
try:
    from src.utils import get_project_root
    PROJECT_ROOT = get_project_root()
except ImportError:
    # Fallback: use relative path
    PROJECT_ROOT = _script_dir.parent.parent
IMAGES_DIR = PROJECT_ROOT / "nowcasting-report" / "forecast" / "images"
DATA_DIR = PROJECT_ROOT / "data"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13
plt.rcParams['font.family'] = 'DejaVu Sans'

# Target variables
TARGETS = ['KOIPALL.G', 'KOEQUIPTE', 'KOWRCCNSE']
TARGET_DISPLAY_NAMES = {
    'KOIPALL.G': 'Production (KOIPALL.G)',
    'KOEQUIPTE': 'Investment (KOEQUIPTE)',
    'KOWRCCNSE': 'Consumption (KOWRCCNSE)'
}

# Model configurations
MODELS = ['dfm', 'ddfm', 'chronos', 'lstm', 'tft']
MODEL_DISPLAY_NAMES = {
    'dfm': 'DFM',
    'ddfm': 'DDFM',
    'chronos': 'Chronos',
    'lstm': 'LSTM',
    'tft': 'TFT'
}
MODEL_COLORS = {
    'dfm': '#1f77b4',      # Blue
    'ddfm': '#ff7f0e',     # Orange
    'chronos': '#2ca02c',  # Green
    'lstm': '#d62728',     # Red
    'tft': '#9467bd'       # Purple
}

# Test period
TEST_START = pd.Timestamp('2024-01-01')
TEST_END = pd.Timestamp('2025-10-31')


def load_actual_data() -> pd.DataFrame:
    """Load actual values from data.csv, filter to test period, using weekly data from date_w column."""
    data_file = DATA_DIR / "data.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    df = pd.read_csv(data_file)
    
    # Use date_w for weekly alignment (format: MM/DD/YYYY)
    if 'date_w' in df.columns:
        df['date_weekly'] = pd.to_datetime(df['date_w'], format='%m/%d/%Y', errors='coerce')
        # Drop rows where date_weekly is null
        df = df[df['date_weekly'].notna()].copy()
        df = df.set_index('date_weekly')
    else:
        # Fallback to date column
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    
    # Filter to test period
    test_data = df[(df.index >= TEST_START) & (df.index <= TEST_END)]
    
    return test_data


def load_forecasts(model_name: str) -> Optional[pd.DataFrame]:
    """Load weekly forecasts for a specific model by combining individual target files."""
    # Load weekly forecasts for each target and combine
    all_forecasts = {}
    
    for target in TARGETS:
        weekly_file = PREDICTIONS_DIR / model_name / f"{target}_{model_name}_weekly.csv"
        
        if weekly_file.exists():
            try:
                forecast_df = pd.read_csv(weekly_file, index_col=0, parse_dates=True)
                # Get the target column (first column if target name doesn't match)
                if target in forecast_df.columns:
                    forecast_series = forecast_df[target]
                else:
                    forecast_series = forecast_df.iloc[:, 0]
                
                all_forecasts[target] = forecast_series
            except Exception as e:
                print(f"Warning: Failed to load {target} weekly forecast for {model_name}: {e}")
                continue
    
    if len(all_forecasts) == 0:
        return None
    
    # Combine into single DataFrame
    combined_df = pd.DataFrame(all_forecasts)
    return combined_df


def plot_forecast_vs_actual_all(save_path: Optional[Path] = None):
    """Create plot comparing all models' forecasts vs actual values.
    
    Layout: 3 subplots (one for each target)
    - Each subplot shows actual values (black solid line) and model forecasts (colored dashed lines)
    - X-axis: Date (2024-01 to 2025-10)
    - Y-axis: Value
    """
    if save_path is None:
        save_path = IMAGES_DIR / "forecast_vs_actual_all.png"
    
    # Load actual data
    print("Loading actual data...")
    actual_data = load_actual_data()
    
    # Load forecasts for all models
    print("Loading forecasts...")
    forecasts = {}
    for model in MODELS:
        forecast_df = load_forecasts(model)
        if forecast_df is not None:
            forecasts[model] = forecast_df
            print(f"  Loaded {model}: {forecast_df.shape}")
        else:
            print(f"  Warning: {model} forecasts not found")
    
    if len(forecasts) == 0:
        print("Error: No forecasts found")
        return
    
    # Create figure with 3 subplots (one for each target)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Forecast vs Actual: All Models and Targets', fontsize=14, fontweight='bold')
    
    # Plot each target
    for idx, target in enumerate(TARGETS):
        ax = axes[idx]
        
        # Plot actual values as scattered points (sparse, since actual data is weekly but not all weeks have data)
        if target in actual_data.columns:
            actual_series = actual_data[target].dropna()
            if len(actual_series) > 0:
                ax.scatter(actual_series.index, actual_series.values, 
                          color='black', s=30, marker='o', label='Actual', 
                          alpha=0.7, zorder=5, edgecolors='black', linewidths=0.5)
        
        # Plot forecasts for each model
        for model in MODELS:
            if model not in forecasts:
                continue
            
            forecast_df = forecasts[model]
            
            # Check if target exists in forecast
            if target in forecast_df.columns:
                forecast_series = forecast_df[target].dropna()
            else:
                # Try to find matching column
                matching_cols = [col for col in forecast_df.columns if target in str(col)]
                if matching_cols:
                    forecast_series = forecast_df[matching_cols[0]].dropna()
                else:
                    continue
            
            if len(forecast_series) > 0:
                # Align forecast dates with actual data dates for better comparison
                # Filter to overlapping period
                if len(actual_series) > 0:
                    # Find overlapping date range
                    overlap_start = max(actual_series.index.min(), forecast_series.index.min())
                    overlap_end = min(actual_series.index.max(), forecast_series.index.max())
                    
                    # Filter both series to overlapping period
                    forecast_aligned = forecast_series[
                        (forecast_series.index >= overlap_start) & 
                        (forecast_series.index <= overlap_end)
                    ]
                else:
                    forecast_aligned = forecast_series
                
                if len(forecast_aligned) > 0:
                    ax.plot(forecast_aligned.index, forecast_aligned.values,
                           '--', linewidth=1.5, alpha=0.7,
                           color=MODEL_COLORS.get(model, 'gray'),
                           label=MODEL_DISPLAY_NAMES.get(model, model))
        
        # Customize subplot
        ax.set_title(TARGET_DISPLAY_NAMES.get(target, target), fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', ncol=3, framealpha=0.9)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
    print(f"Saved plot to: {save_path}")
    plt.close()
    
    return save_path


def main():
    """Main function to generate all forecast plots."""
    print("="*70)
    print("Generating Forecast vs Actual Plots")
    print("="*70)
    
    try:
        plot_forecast_vs_actual_all()
        print("\n✓ All plots generated successfully!")
    except Exception as e:
        print(f"\n✗ Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
