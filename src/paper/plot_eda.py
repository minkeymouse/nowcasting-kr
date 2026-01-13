"""Plot creation code for EDA visualizations.

This module generates a combined 2x4 plot (2 rows x 4 columns) showing:
- Investment dataset (left 4 columns)
- Production dataset (right 4 columns)

Each dataset has 4 subplots:
1. Target series (original scale) time series
2. Correlation heatmap of top 10 variables (monthly data only)
3. Top NaN proportion by variable (monthly aggregation)
4. Target series distribution (kernel density, standardized scale)

All plots are saved to nowcasting-report/images/ directory.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing utilities
from src.preprocess import InvestmentData, ProductionData
from src.utils import get_project_root

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['font.family'] = 'DejaVu Sans'

# Base paths
PROJECT_ROOT = get_project_root()
IMAGES_DIR = PROJECT_ROOT / "nowcasting-report" / "images"

# Target variables for each dataset
TARGETS = {
    'investment': 'KOEQUIPTE',
    'production': 'KOIPALL.G'
}
DATASET_DISPLAY_NAMES = {
    'investment': 'Investment',
    'production': 'Production'
}


def _load_data(dataset: str):
    """Load data for a specific dataset."""
    if dataset == 'investment':
        data = InvestmentData()
    elif dataset == 'production':
        data = ProductionData()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return data.original, data.processed, data.standardized, data.metadata


def _separate_weekly_monthly(data: pd.DataFrame, metadata: pd.DataFrame):
    """Separate data into weekly and monthly series."""
    series_col = 'Series_ID' if 'Series_ID' in metadata.columns else 'SeriesID'
    
    monthly_series = []
    weekly_series = []
    
    for col in data.columns:
        meta_row = metadata[metadata[series_col] == col]
        if len(meta_row) > 0:
            freq = str(meta_row.iloc[0].get('Frequency', 'w')).lower()
            if freq == 'm':
                monthly_series.append(col)
            else:
                weekly_series.append(col)
        else:
            weekly_series.append(col)
    
    monthly_data = data[monthly_series] if monthly_series else pd.DataFrame()
    weekly_data = data[weekly_series] if weekly_series else pd.DataFrame()
    
    return monthly_data, weekly_data, monthly_series, weekly_series


def _aggregate_to_monthly(weekly_data: pd.DataFrame):
    """Aggregate weekly data to monthly by taking last value of each month."""
    if weekly_data.empty or not isinstance(weekly_data.index, pd.DatetimeIndex):
        return pd.DataFrame()
    
    # Resample to monthly, taking last value
    monthly = weekly_data.resample('M').last()
    return monthly


def _calculate_nan_proportion_monthly(data: pd.DataFrame):
    """Calculate NaN proportion for each variable after aggregating to monthly."""
    if data.empty:
        return pd.Series(dtype=float)
    
    monthly_data = _aggregate_to_monthly(data) if not isinstance(data.index, pd.DatetimeIndex) or data.index.freq != 'M' else data
    if monthly_data.empty:
        return pd.Series(dtype=float)
    
    nan_prop = monthly_data.isna().sum() / len(monthly_data)
    return nan_prop.sort_values(ascending=False)


def plot_combined_eda(save_path: Optional[Path] = None):
    """Create combined 2x4 plot for both datasets."""
    if save_path is None:
        save_path = IMAGES_DIR / "combined_eda.png"
    
    try:
        # Create figure: 2 rows x 4 columns (2 datasets side by side)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        datasets = ['production', 'investment']  # Production on left, Investment on right
        
        for col_idx, dataset in enumerate(datasets):
            # Load data
            original, processed, standardized, metadata = _load_data(dataset)
            target = TARGETS[dataset]
            
            # Separate weekly and monthly data
            monthly_data, weekly_data, monthly_cols, weekly_cols = _separate_weekly_monthly(original, metadata)
            
            # Set index if needed
            if 'date_w' in original.columns and not isinstance(original.index, pd.DatetimeIndex):
                idx = pd.to_datetime(original['date_w'], errors='coerce')
                original.index = idx
                processed.index = idx
                standardized.index = idx
                monthly_data.index = idx
                weekly_data.index = idx
            
            # Plot 1: Target series (original scale) - Top row
            ax = axes[0, col_idx * 2]
            if target in original.columns:
                target_series = original[target].dropna()
                if len(target_series) > 0:
                    ax.plot(target_series.index, target_series.values, 
                           linewidth=1.5, alpha=0.8, color='#2ca02c', marker='o', markersize=3)
                    ax.set_xlabel('Date', fontsize=9)
                    ax.set_ylabel('Original Value', fontsize=9)
                    ax.set_title(f'{DATASET_DISPLAY_NAMES[dataset]} - Target Series', fontsize=10, fontweight='bold')
                    ax.grid(alpha=0.3)
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Plot 2: Correlation heatmap (top 10 variables, monthly data only) - Top row
            ax = axes[0, col_idx * 2 + 1]
            if target in monthly_data.columns:
                # Get monthly data with target
                monthly_with_target = monthly_data.copy()
                if target not in monthly_with_target.columns:
                    # Add target from original if not in monthly
                    target_monthly = original[target].dropna()
                    monthly_with_target = monthly_with_target.reindex(target_monthly.index)
                    monthly_with_target[target] = target_monthly
                
                # Calculate correlations using monthly observations only
                monthly_clean = monthly_with_target.dropna()
                if len(monthly_clean) > 10:  # Need enough observations
                    corr_series = monthly_clean.corr()[target].drop(target)
                    top_10 = corr_series.abs().nlargest(10).index.tolist()
                    features_for_matrix = [target] + top_10[:9]  # Target + top 9 = 10 total
                    
                    corr_matrix = monthly_clean[features_for_matrix].corr()
                    
                    # Create heatmap
                    im = ax.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
                    ax.set_xticks(range(len(features_for_matrix)))
                    ax.set_yticks(range(len(features_for_matrix)))
                    feature_labels = [f.replace('KO', '').replace('KOW', 'W')[:6] for f in features_for_matrix]
                    ax.set_xticklabels(feature_labels, rotation=45, ha='right', fontsize=7)
                    ax.set_yticklabels(feature_labels, fontsize=7)
                    
                    # Add correlation values
                    for i in range(len(features_for_matrix)):
                        for j in range(len(features_for_matrix)):
                            corr_val = corr_matrix.iloc[i, j]
                            if abs(corr_val) > 0.3:  # Only show significant correlations
                                text_color = 'white' if abs(corr_val) > 0.6 else 'black'
                                ax.text(j, i, f'{corr_val:.2f}', ha='center', va='center',
                                       color=text_color, fontsize=6, fontweight='bold')
                    
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    ax.set_title(f'{DATASET_DISPLAY_NAMES[dataset]} - Top 10 Correlations (Monthly)', 
                               fontsize=10, fontweight='bold')
                else:
                    ax.text(0.5, 0.5, 'Insufficient monthly data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=9)
                    ax.set_title(f'{DATASET_DISPLAY_NAMES[dataset]} - Correlations', fontsize=10)
            else:
                ax.text(0.5, 0.5, 'Target not in monthly data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=9)
                ax.set_title(f'{DATASET_DISPLAY_NAMES[dataset]} - Correlations', fontsize=10)
            
            # Plot 3: Top NaN proportion (bottom row, left)
            ax = axes[1, col_idx * 2]
            # Combine weekly and monthly, then aggregate to monthly
            all_data = pd.concat([weekly_data, monthly_data], axis=1)
            nan_prop = _calculate_nan_proportion_monthly(all_data)
            
            if len(nan_prop) > 0:
                top_10_nan = nan_prop.head(10)
                colors_bar = plt.cm.Reds([v/max(top_10_nan) if max(top_10_nan) > 0 else 0 for v in top_10_nan.values])
                bars = ax.barh(range(len(top_10_nan)), top_10_nan.values, color=colors_bar, alpha=0.7, edgecolor='black')
                ax.set_yticks(range(len(top_10_nan)))
                var_labels = [v.replace('KO', '').replace('KOW', 'W')[:10] for v in top_10_nan.index]
                ax.set_yticklabels(var_labels, fontsize=7)
                ax.set_xlabel('NaN Proportion', fontsize=9)
                ax.set_title(f'{DATASET_DISPLAY_NAMES[dataset]} - Top 10 NaN Proportion', 
                           fontsize=10, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                
                # Add value labels
                for i, (bar, val) in enumerate(zip(bars, top_10_nan.values)):
                    ax.text(val + 0.02, i, f'{val:.2f}', va='center', fontsize=7)
            else:
                ax.text(0.5, 0.5, 'No data for NaN calculation', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=9)
                ax.set_title(f'{DATASET_DISPLAY_NAMES[dataset]} - NaN Proportion', fontsize=10)
            
            # Plot 4: Target distribution (kernel density, standardized) - bottom row, right
            ax = axes[1, col_idx * 2 + 1]
            # Use original data, aggregate to monthly, then standardize for visualization
            if target in original.columns:
                target_orig = original[target].copy()
                
                # Ensure datetime index
                if not isinstance(target_orig.index, pd.DatetimeIndex) or target_orig.index.freq != 'ME':
                    if 'date_w' in original.columns:
                        idx = pd.to_datetime(original['date_w'], errors='coerce')
                        target_orig.index = idx
                        target_orig = target_orig[~target_orig.index.isna()]
                
                # Aggregate to monthly (take last value per month)
                target_monthly = target_orig.resample('ME').last().dropna()
                
                if len(target_monthly) > 5:  # Need enough points for KDE
                    # Manual standardization for visualization
                    from sklearn.preprocessing import StandardScaler
                    values = target_monthly.values.reshape(-1, 1)
                    scaler = StandardScaler()
                    target_std_values = scaler.fit_transform(values).flatten()
                    
                    # Kernel density estimation
                    kde = gaussian_kde(target_std_values)
                    x_range = np.linspace(target_std_values.min(), target_std_values.max(), 200)
                    density = kde(x_range)
                    
                    ax.fill_between(x_range, density, alpha=0.6, color='#3498db')
                    ax.plot(x_range, density, linewidth=2, color='#2980b9')
                    ax.axvline(target_std_values.mean(), color='red', linestyle='--', linewidth=1.5, 
                              label=f'Mean: {target_std_values.mean():.2f}')
                    ax.axvline(np.median(target_std_values), color='green', linestyle='--', linewidth=1.5, 
                              label=f'Median: {np.median(target_std_values):.2f}')
                    
                    ax.set_xlabel('Standardized Value', fontsize=9)
                    ax.set_ylabel('Density', fontsize=9)
                    ax.set_title(f'{DATASET_DISPLAY_NAMES[dataset]} - Target Distribution (Monthly)', 
                               fontsize=10, fontweight='bold')
                    ax.legend(fontsize=7, loc='best')
                    ax.grid(alpha=0.3)
                else:
                    ax.text(0.5, 0.5, f'Insufficient monthly data for KDE (n={len(target_monthly)})', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=9)
                    ax.set_title(f'{DATASET_DISPLAY_NAMES[dataset]} - Distribution', fontsize=10)
            else:
                ax.text(0.5, 0.5, 'Target not found', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=9)
                ax.set_title(f'{DATASET_DISPLAY_NAMES[dataset]} - Distribution', fontsize=10)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Generated: {save_path.name}")
        
    except Exception as e:
        print(f"Error generating combined EDA plot: {e}")
        import traceback
        traceback.print_exc()
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()


def generate_plots():
    """Generate combined EDA plot."""
    print("=" * 70)
    print("Generating Combined EDA Plot")
    print("=" * 70)
    
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating combined 2x4 plot...")
    plot_combined_eda()
    
    print("\n" + "=" * 70)
    print("Plot generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    generate_plots()
