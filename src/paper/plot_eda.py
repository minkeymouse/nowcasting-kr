"""Plot creation code for EDA visualizations.

This module generates plots for exploratory data analysis:
1. preprocessed_targets: Preprocessed target variables (KOIPALL.G, KOEQUIPTE, KOWRCCNSE)

All plots are saved to nowcasting-report/forecast/images/ directory as *.png files.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing utilities
from src.train.preprocess import apply_transformations, impute_missing_values, apply_scaling
from src.utils import get_project_root, setup_paths, SCALER_ROBUST
from sklearn.preprocessing import RobustScaler

# Setup paths first
setup_paths(include_dfm_python=True, include_src=True)

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

# Base paths
PROJECT_ROOT = get_project_root()
IMAGES_DIR = PROJECT_ROOT / "nowcasting-report" / "forecast" / "images"
DATA_DIR = PROJECT_ROOT / "data"

# Target variables
TARGETS = ['KOEQUIPTE', 'KOWRCCNSE', 'KOIPALL.G']
TARGET_DISPLAY_NAMES = {
    'KOEQUIPTE': 'Investment (KOEQUIPTE)',
    'KOWRCCNSE': 'Consumption (KOWRCCNSE)',
    'KOIPALL.G': 'Production (KOIPALL.G)'
}


def _load_data() -> pd.DataFrame:
    """Load full dataset."""
    data_file = DATA_DIR / "data.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    return data


def plot_preprocessed_targets(save_path: Optional[Path] = None):
    """Create plot showing preprocessed target variables with correlation tables.
    
    Layout: 3x2 grid
    - Left column (3 plots): Time series plots (wide)
    - Right column (3 tables): Top 7 correlation variables for each target (square)
    """
    if save_path is None:
        save_path = IMAGES_DIR / "preprocessed_targets.png"
    
    try:
        from src.utils import get_config_path
        
        # Load raw data
        data = _load_data()
        
        # Get target variables
        available_targets = [t for t in TARGETS if t in data.columns]
        
        if len(available_targets) == 0:
            print("Warning: No target variables found in data")
            return
        
        # Get config path
        config_path = str(get_config_path())
        
        # Prepare preprocessed data for each target
        target_data = {}
        for target in available_targets:
            # Extract target series
            target_series = data[[target]].copy()
            
            # Apply transformations
            try:
                target_transformed = apply_transformations(
                    target_series, 
                    config_path=config_path, 
                    series_ids=[target]
                )
            except Exception as e:
                print(f"Warning: Transformation failed for {target}: {e}")
                target_transformed = target_series.copy()
            
            # Impute missing values
            try:
                target_imputed = impute_missing_values(target_transformed, model_type="preprocessing")
            except Exception as e:
                print(f"Warning: Imputation failed for {target}: {e}")
                target_imputed = target_transformed.fillna(method='ffill').fillna(method='bfill')
            
            # Scale data (RobustScaler)
            try:
                target_scaled, _ = apply_scaling(target_imputed, scaler_type=SCALER_ROBUST)
            except Exception as e:
                print(f"Warning: Scaling failed for {target}: {e}")
                target_scaled = target_imputed.copy()
            
            target_data[target] = target_scaled
        
        # Calculate correlations for each target
        # Use preprocessed target data and calculate correlation with other preprocessed variables
        correlations = {}
        top_features = {}
        for target in available_targets:
            if target not in target_data:
                correlations[target] = pd.Series()
                top_features[target] = []
                continue
            
            # Get preprocessed target series
            target_series = target_data[target][target]
            
            # Calculate correlation with all other variables in original data
            # We'll use simple correlation on numeric columns only
            numeric_data = data.select_dtypes(include=[np.number])
            
            if target not in numeric_data.columns:
                correlations[target] = pd.Series()
                top_features[target] = []
                continue
            
            # Align indices and calculate correlation
            target_aligned = target_series.reindex(numeric_data.index)
            numeric_aligned = numeric_data.reindex(target_aligned.index)
            
            # Calculate pairwise correlation
            corr_series = numeric_aligned.corrwith(target_aligned)
            # Remove self-correlation and NaN values
            corr_series = corr_series.drop(target)
            corr_series = corr_series.dropna()
            # Sort by absolute value
            corr_series = corr_series.reindex(corr_series.abs().sort_values(ascending=False).index)
            # Get top 7
            top_7 = corr_series.head(7)
            correlations[target] = top_7
            top_features[target] = top_7.index.tolist()
        
        # Compute correlation matrix for top features for each target
        correlation_matrices = {}
        for target in available_targets:
            if target not in top_features or len(top_features[target]) == 0:
                correlation_matrices[target] = None
                continue
            
            # Get top features plus target
            features_for_matrix = [target] + top_features[target]
            # Filter to features that exist in numeric data
            features_for_matrix = [f for f in features_for_matrix if f in numeric_data.columns]
            
            if len(features_for_matrix) < 2:
                correlation_matrices[target] = None
                continue
            
            # Align all features
            aligned_data = numeric_data[features_for_matrix].reindex(target_aligned.index)
            # Drop rows with any NaN
            aligned_data = aligned_data.dropna()
            
            if len(aligned_data) < 100:
                correlation_matrices[target] = None
                continue
            
            # Compute correlation matrix
            corr_matrix = aligned_data.corr()
            correlation_matrices[target] = {
                'matrix': corr_matrix,
                'features': features_for_matrix
            }
        
        # Create figure with 3x2 grid
        # Use gridspec for better control of subplot sizes
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 2, figure=fig, width_ratios=[2, 1], hspace=0.3, wspace=0.3)
        
        for idx, target in enumerate(available_targets):
            # Left column: Time series plot (wide)
            ax_left = fig.add_subplot(gs[idx, 0])
            
            if target not in target_data:
                ax_left.text(0.5, 0.5, f'No data for {target}', ha='center', va='center', transform=ax_left.transAxes)
                ax_left.set_title(TARGET_DISPLAY_NAMES.get(target, target))
            else:
                # Get preprocessed data
                preprocessed = target_data[target]
                
                # Plot
                ax_left.plot(preprocessed.index, preprocessed[target].values, 
                           linewidth=1.5, alpha=0.8, color='#2ca02c')
                
                # Formatting
                ax_left.set_xlabel('Date', fontsize=10)
                ax_left.set_ylabel('Preprocessed Value', fontsize=10)
                ax_left.set_title(TARGET_DISPLAY_NAMES.get(target, target), fontsize=11, fontweight='bold')
                ax_left.grid(alpha=0.3)
                fig.autofmt_xdate()
            
            # Right column: Correlation heatmap (square)
            ax_right = fig.add_subplot(gs[idx, 1])
            
            if target in correlation_matrices and correlation_matrices[target] is not None:
                corr_info = correlation_matrices[target]
                corr_matrix = corr_info['matrix']
                features = corr_info['features']
                
                # Create heatmap
                im = ax_right.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
                
                # Set ticks and labels
                ax_right.set_xticks(range(len(features)))
                ax_right.set_yticks(range(len(features)))
                # Shorten feature names for display
                feature_labels = [f.replace('KO', '').replace('KOW', 'W')[:8] for f in features]
                ax_right.set_xticklabels(feature_labels, rotation=45, ha='right', fontsize=7)
                ax_right.set_yticklabels(feature_labels, fontsize=7)
                
                # Add correlation values as text
                for i in range(len(features)):
                    for j in range(len(features)):
                        corr_val = corr_matrix.iloc[i, j]
                        text_color = 'white' if abs(corr_val) > 0.5 else 'black'
                        ax_right.text(j, i, f'{corr_val:.2f}', ha='center', va='center',
                                     color=text_color, fontsize=6, fontweight='bold')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax_right, fraction=0.046, pad=0.04)
                cbar.set_label('Correlation', fontsize=8)
                
                ax_right.set_title(f'Top 7 Correlations', fontsize=10, fontweight='bold', pad=10)
            else:
                ax_right.axis('off')
                ax_right.text(0.5, 0.5, 'No correlation data', ha='center', va='center', 
                            transform=ax_right.transAxes, fontsize=10)
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Generated: {save_path.name}")
        
    except Exception as e:
        print(f"Error generating preprocessed targets plot: {e}")
        import traceback
        traceback.print_exc()
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(18, 12))
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()


def plot_data_quality_dashboard(save_path: Optional[Path] = None):
    """Create data quality and statistics dashboard.
    
    Shows:
    - Normalized time series comparison of 3 targets
    - Distribution/boxplot of variables
    - Missing value patterns
    - Key statistics summary
    """
    if save_path is None:
        save_path = IMAGES_DIR / "data_quality_dashboard.png"
    
    try:
        from src.utils import get_config_path
        
        # Load raw data
        data = _load_data()
        
        # Get target variables
        available_targets = [t for t in TARGETS if t in data.columns]
        
        if len(available_targets) == 0:
            print("Warning: No target variables found in data")
            return
        
        # Get config path
        config_path = str(get_config_path())
        
        # Prepare preprocessed data for each target
        target_data = {}
        for target in available_targets:
            target_series = data[[target]].copy()
            
            try:
                target_transformed = apply_transformations(
                    target_series, 
                    config_path=config_path, 
                    series_ids=[target]
                )
            except Exception as e:
                print(f"Warning: Transformation failed for {target}: {e}")
                target_transformed = target_series.copy()
            
            try:
                target_imputed = impute_missing_values(target_transformed, model_type="preprocessing")
            except Exception as e:
                print(f"Warning: Imputation failed for {target}: {e}")
                target_imputed = target_transformed.fillna(method='ffill').fillna(method='bfill')
            
            try:
                target_scaled, _ = apply_scaling(target_imputed, scaler_type=SCALER_ROBUST)
            except Exception as e:
                print(f"Warning: Scaling failed for {target}: {e}")
                target_scaled = target_imputed.copy()
            
            target_data[target] = target_scaled
        
        # Create figure with 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Data Quality and Statistics Dashboard', fontsize=14, fontweight='bold', y=0.995)
        
        # 1. Normalized time series comparison (top left)
        ax = axes[0, 0]
        for target in available_targets:
            if target in target_data:
                preprocessed = target_data[target]
                ax.plot(preprocessed.index, preprocessed[target].values, 
                       linewidth=1.5, alpha=0.7, 
                       label=TARGET_DISPLAY_NAMES.get(target, target))
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Normalized Value (RobustScaler)', fontsize=10)
        ax.set_title('Normalized Target Variables Comparison', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        
        # 2. Distribution/Boxplot (top right)
        ax = axes[0, 1]
        box_data = []
        box_labels = []
        for target in available_targets:
            if target in target_data:
                values = target_data[target][target].values
                values = values[~np.isnan(values)]
                if len(values) > 0:
                    box_data.append(values)
                    box_labels.append(TARGET_DISPLAY_NAMES.get(target, target).split('(')[0].strip())
        
        if len(box_data) > 0:
            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
            colors = ['#2ca02c', '#d62728', '#1f77b4']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_ylabel('Normalized Value', fontsize=10)
            ax.set_title('Distribution of Target Variables', fontsize=11, fontweight='bold')
            ax.grid(alpha=0.3, axis='y')
        
        # 3. Data coverage / variance comparison (bottom left)
        ax = axes[1, 0]
        # Calculate coefficient of variation (CV) for each target as a measure of variability
        cv_data = []
        cv_labels = []
        for target in available_targets:
            if target in target_data:
                values = target_data[target][target].values
                values = values[~np.isnan(values)]
                if len(values) > 0 and np.mean(np.abs(values)) > 1e-10:
                    cv = np.std(values) / np.mean(np.abs(values)) if np.mean(np.abs(values)) > 0 else 0
                    cv_data.append(cv)
                    cv_labels.append(TARGET_DISPLAY_NAMES.get(target, target).split('(')[0].strip())
        
        if len(cv_data) > 0:
            colors_cv = plt.cm.viridis([v/max(cv_data) if max(cv_data) > 0 else 0 for v in cv_data])
            bars = ax.barh(cv_labels, cv_data, color=colors_cv, alpha=0.7, edgecolor='black', linewidth=1)
            ax.set_xlabel('Coefficient of Variation', fontsize=10)
            ax.set_title('Variability Comparison', fontsize=11, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            for bar, cv_val in zip(bars, cv_data):
                ax.text(cv_val + max(cv_data)*0.02, bar.get_y() + bar.get_height()/2, 
                       f'{cv_val:.2f}', va='center', fontsize=9, fontweight='bold')
        
        # 4. Key statistics summary - visual bar chart style (bottom right)
        ax = axes[1, 1]
        
        # Calculate statistics for each target
        stats_dict = {}
        for target in available_targets:
            if target in target_data:
                values = target_data[target][target].values
                values = values[~np.isnan(values)]
                if len(values) > 0:
                    var_name = TARGET_DISPLAY_NAMES.get(target, target).split('(')[0].strip()
                    stats_dict[var_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
        
        if len(stats_dict) > 0:
            # Create grouped bar chart for statistics
            variables = list(stats_dict.keys())
            x = np.arange(len(variables))
            width = 0.2
            
            # Normalize statistics for visualization (use relative scale)
            means = [stats_dict[v]['mean'] for v in variables]
            stds = [stats_dict[v]['std'] for v in variables]
            mins = [stats_dict[v]['min'] for v in variables]
            maxs = [stats_dict[v]['max'] for v in variables]
            
            # Normalize to 0-1 scale for better visualization
            all_vals = means + stds + [abs(m) for m in mins] + [abs(m) for m in maxs]
            max_abs = max([abs(v) for v in all_vals]) if len(all_vals) > 0 else 1
            
            means_norm = [abs(m)/max_abs for m in means]
            stds_norm = [s/max_abs for s in stds]
            mins_norm = [abs(m)/max_abs for m in mins]
            maxs_norm = [abs(m)/max_abs for m in maxs]
            
            bars1 = ax.bar(x - 1.5*width, means_norm, width, label='Mean (abs)', 
                          color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=0.5)
            bars2 = ax.bar(x - 0.5*width, stds_norm, width, label='Std', 
                          color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
            bars3 = ax.bar(x + 0.5*width, mins_norm, width, label='Min (abs)', 
                          color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)
            bars4 = ax.bar(x + 1.5*width, maxs_norm, width, label='Max (abs)', 
                          color='#f39c12', alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Variable', fontsize=10)
            ax.set_ylabel('Normalized Value', fontsize=10)
            ax.set_title('Key Statistics Comparison', fontsize=11, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(variables, fontsize=9, rotation=15, ha='right')
            ax.legend(fontsize=8, loc='upper left', ncol=2)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bars, vals, orig_vals in [(bars1, means_norm, means), (bars2, stds_norm, stds), 
                                          (bars3, mins_norm, mins), (bars4, maxs_norm, maxs)]:
                for bar, val, orig in zip(bars, vals, orig_vals):
                    if val > 0.05:  # Only label if bar is visible
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{orig:.2f}', ha='center', va='bottom', fontsize=7)
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Generated: {save_path.name}")
        
    except Exception as e:
        print(f"Error generating data quality dashboard: {e}")
        import traceback
        traceback.print_exc()
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(16, 12))
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()


def generate_plots():
    """Generate all EDA plots."""
    print("=" * 70)
    print("Generating EDA Plots")
    print("=" * 70)
    
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n1. Preprocessed Target Variables")
    print("   - preprocessed_targets.png")
    plot_preprocessed_targets()
    
    print("\n2. Data Quality Dashboard")
    print("   - data_quality_dashboard.png")
    plot_data_quality_dashboard()
    
    print("\n" + "=" * 70)
    print("Plot generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    generate_plots()
