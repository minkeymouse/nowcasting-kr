"""Table creation code for forecast metrics by horizon.

This module generates tables comparing forecast metrics across different models and horizons:
1. For each target variable, create a table with horizon × metrics structure
2. Metrics include: MAE, RMSE, MAPE, sMAE, sRMSE
3. Tables are saved to tables/ directory as CSV files

Each table shows:
- Rows: Horizon (1, 2, 3, ..., up to forecast horizon)
- Columns: Models (CHRONOS, DDFM, DFM, LSTM, TFT) × Metrics (MAE, RMSE, MAPE, sMAE, sRMSE)
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
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

TABLES_DIR = PROJECT_ROOT / "nowcasting-report" / "forecast" / "tables"
DATA_DIR = PROJECT_ROOT / "data"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"

# Target variables
TARGETS = ['KOIPALL.G', 'KOEQUIPTE', 'KOWRCCNSE']

# Models
MODELS = ['chronos', 'ddfm', 'dfm', 'lstm', 'tft']

# Metrics to calculate
METRICS = ['MAE', 'RMSE', 'MAPE', 'sMAE', 'sRMSE', 'sMSE']

# Metrics to display in tables
TABLE_METRICS = ['sMAE', 'sMSE']  # Only show sMAE and sMSE in tables


def load_actual_data(data_path: Path) -> pd.DataFrame:
    """Load actual data from CSV file.
    
    IMPORTANT: Uses monthly date column (not date_w) because target variables are monthly data.
    The date_w column shows when data was published (often in the next month), but the actual
    values correspond to the month in the 'date' column. For example, January 2024 data appears
    in date_w as February 2, 2024, but the date column correctly shows 2024-01-31.
    
    Since the same monthly date appears in multiple rows (one per week), we group by month
    and take the last non-null value for each month.
    
    Parameters
    ----------
    data_path : Path
        Path to data CSV file
        
    Returns
    -------
    pd.DataFrame
        Actual data with monthly date index and target columns (one row per month)
    """
    df = pd.read_csv(data_path)
    
    # Use monthly date column (not date_w) because targets are monthly data
    # date_w shows publication date, but date shows the actual month the data represents
    if 'date' in df.columns:
        df['date_monthly'] = pd.to_datetime(df['date'], errors='coerce')
        # Drop rows where date_monthly is null
        df = df[df['date_monthly'].notna()].copy()
        
        # Group by monthly date and take last non-null value for each month
        # This handles the case where the same month appears in multiple rows (weekly data)
        # groupby automatically sets the grouped column as index
        monthly_df = df.groupby('date_monthly').last()
        
        return monthly_df
    elif 'date_w' in df.columns:
        # Fallback to date_w if date column doesn't exist
        df['date_weekly'] = pd.to_datetime(df['date_w'], format='%m/%d/%Y', errors='coerce')
        df = df[df['date_weekly'].notna()].copy()
        df = df.set_index('date_weekly')
        return df
    else:
        raise ValueError("Neither 'date' nor 'date_w' column found in data file")


def load_forecast_data(model: str, predictions_dir: Path) -> Optional[pd.DataFrame]:
    """Load forecast data for a model.
    
    IMPORTANT: Uses weekly forecast files (same as plot_forecasts.py) to ensure consistency.
    predictions/{model}/{target}_{model}_weekly.csv is the source of truth.
    
    Parameters
    ----------
    model : str
        Model name (chronos, ddfm, dfm, lstm, tft)
    predictions_dir : Path
        Directory containing predictions
        
    Returns
    -------
    Optional[pd.DataFrame]
        Forecast data with weekly date index, or None if not found
    """
    # Load weekly forecasts for each target (same as plot_forecasts.py)
    all_forecasts = {}
    
    for target in TARGETS:
        weekly_file = predictions_dir / model / f"{target}_{model}_weekly.csv"
        
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
                print(f"Warning: Failed to load {target} weekly forecast for {model}: {e}")
                continue
    
    if len(all_forecasts) == 0:
        return None
    
    # Combine into single DataFrame (same as plot_forecasts.py)
    combined_df = pd.DataFrame(all_forecasts)
    return combined_df


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    train_std: float = 1.0
) -> Dict[str, float]:
    """Calculate forecast metrics.
    
    Parameters
    ----------
    actual : np.ndarray
        Actual values
    predicted : np.ndarray
        Predicted values
    train_std : float
        Training data standard deviation for standardized metrics
        
    Returns
    -------
    Dict[str, float]
        Dictionary of metrics
    """
    # Remove NaN values
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    if mask.sum() == 0:
        return {metric: np.nan for metric in METRICS}
    
    actual_clean = actual[mask]
    predicted_clean = predicted[mask]
    
    # Calculate errors
    errors = actual_clean - predicted_clean
    
    # MAE
    mae = np.mean(np.abs(errors))
    
    # RMSE
    rmse = np.sqrt(np.mean(errors ** 2))
    
    # MAPE (avoid division by zero)
    mape_mask = actual_clean != 0
    if mape_mask.sum() > 0:
        mape = np.mean(np.abs((actual_clean[mape_mask] - predicted_clean[mape_mask]) / actual_clean[mape_mask])) * 100
    else:
        mape = np.nan
    
    # Standardized metrics
    if train_std > 0:
        smae = mae / train_std
        srmse = rmse / train_std
        smse = (rmse / train_std) ** 2  # sMSE = sRMSE^2
    else:
        smae = np.nan
        srmse = np.nan
        smse = np.nan
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'sMAE': smae,
        'sRMSE': srmse,
        'sMSE': smse
    }


def align_forecast_with_actual(
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    target: str,
    tolerance_days: int = 3,
    return_dates: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Align forecast data with actual data by date.
    
    Uses nearest date matching if exact dates don't match.
    For weekly data, uses week-based alignment as fallback.
    
    Parameters
    ----------
    actual_df : pd.DataFrame
        Actual data with date index
    forecast_df : pd.DataFrame
        Forecast data with date index
    target : str
        Target variable name
    tolerance_days : int
        Maximum days difference for matching dates
        
    return_dates : bool
        If True, also return aligned dates
        
    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
        (actual_values, forecast_values, dates) aligned by date, or (None, None, None) if no overlap
        If return_dates=False, dates will be None
    """
    if target not in actual_df.columns or target not in forecast_df.columns:
        return (None, None, None) if return_dates else (None, None)
    
    # Try exact match first
    common_dates = actual_df.index.intersection(forecast_df.index)
    
    if len(common_dates) == 0:
        # Try nearest date matching with increased tolerance for weekly data
        actual_values_list = []
        forecast_values_list = []
        
        # Sort indices for proper matching
        actual_sorted = actual_df.sort_index()
        forecast_sorted = forecast_df.sort_index()
        
        # For weekly forecasts, use larger tolerance (up to 7 days)
        effective_tolerance = max(tolerance_days, 7) if len(forecast_sorted) > 20 else tolerance_days
        
        for fc_date in forecast_sorted.index:
            # Find closest actual date
            date_diffs = np.abs((actual_sorted.index - fc_date).days)
            closest_idx_pos = date_diffs.argmin()
            closest_idx = actual_sorted.index[closest_idx_pos]
            closest_diff = date_diffs.iloc[closest_idx_pos] if hasattr(date_diffs, 'iloc') else date_diffs[closest_idx_pos]
            
            if closest_diff <= effective_tolerance:
                actual_val = actual_sorted.loc[closest_idx, target]
                forecast_val = forecast_sorted.loc[fc_date, target]
                
                if pd.notna(actual_val) and pd.notna(forecast_val):
                    actual_values_list.append(actual_val)
                    forecast_values_list.append(forecast_val)
                    if return_dates:
                        # Store forecast dates as we match
                        if 'forecast_dates_list' not in locals():
                            forecast_dates_list = []
                        forecast_dates_list.append(fc_date)
        
        if len(actual_values_list) > 0:
            if return_dates:
                if 'forecast_dates_list' in locals() and len(forecast_dates_list) == len(actual_values_list):
                    return np.array(actual_values_list), np.array(forecast_values_list), np.array(forecast_dates_list)
                else:
                    # Fallback: use forecast dates from sorted index
                    forecast_dates = np.array([forecast_sorted.index[i] for i in range(len(forecast_sorted)) 
                                              if i < len(actual_values_list)])
                    return np.array(actual_values_list), np.array(forecast_values_list), forecast_dates
            else:
                return np.array(actual_values_list), np.array(forecast_values_list)
        
        # Fallback: Try week-based alignment for weekly data
        try:
            actual_series = actual_sorted[target].dropna()
            forecast_series = forecast_sorted[target].dropna()
            
            if len(actual_series) > 0 and len(forecast_series) > 0:
                # Resample actual to weekly if needed, but keep forecast as-is if already weekly
                actual_weekly = actual_series.resample('W-SUN').last()
                
                # Use forecast as-is (it's already weekly)
                forecast_weekly = forecast_series
                
                # Find overlapping period (extend by 2 weeks for flexibility)
                actual_start = actual_weekly.index.min()
                actual_end = actual_weekly.index.max()
                forecast_start = forecast_weekly.index.min()
                forecast_end = forecast_weekly.index.max()
                
                # Extend overlap window
                overlap_start = max(actual_start, forecast_start - pd.Timedelta(days=14))
                overlap_end = min(actual_end, forecast_end + pd.Timedelta(days=14))
                
                if overlap_start <= overlap_end:
                    actual_aligned = actual_weekly[(actual_weekly.index >= overlap_start) & (actual_weekly.index <= overlap_end)]
                    forecast_aligned = forecast_weekly[(forecast_weekly.index >= overlap_start) & (forecast_weekly.index <= overlap_end)]
                    
                    # Match by finding closest dates (within 7 days for same week)
                    aligned_actual = []
                    aligned_forecast = []
                    used_actual_indices = set()  # Avoid using same actual value twice
                    
                    # Sort both by date
                    actual_sorted_weeks = actual_aligned.sort_index()
                    forecast_sorted_weeks = forecast_aligned.sort_index()
                    
                    # For each forecast date, find closest actual date (within 7 days)
                    aligned_dates = []  # Store dates for return_dates
                    for fc_date in forecast_sorted_weeks.index:
                        # Find closest actual date that hasn't been used
                        date_diffs = np.abs((actual_sorted_weeks.index - fc_date).days)
                        
                        # Sort by date difference and find first unused
                        sorted_indices = date_diffs.argsort()
                        for idx_pos in sorted_indices:
                            actual_idx = actual_sorted_weeks.index[idx_pos]
                            if actual_idx not in used_actual_indices:
                                date_diff = date_diffs.iloc[idx_pos] if hasattr(date_diffs, 'iloc') else date_diffs[idx_pos]
                                
                                if date_diff <= 7:  # Within same week
                                    actual_val = actual_sorted_weeks.loc[actual_idx]
                                    forecast_val = forecast_sorted_weeks.loc[fc_date]
                                    
                                    if pd.notna(actual_val) and pd.notna(forecast_val):
                                        aligned_actual.append(actual_val)
                                        aligned_forecast.append(forecast_val)
                                        if return_dates:
                                            aligned_dates.append(fc_date)
                                        used_actual_indices.add(actual_idx)
                                        break
                    
                    if len(aligned_actual) > 0:
                        if return_dates:
                            return np.array(aligned_actual), np.array(aligned_forecast), np.array(aligned_dates)
                        else:
                            return np.array(aligned_actual), np.array(aligned_forecast)
        except Exception as e:
            # Silently fail and return None
            pass
        
        return (None, None, None) if return_dates else (None, None)
    
    # Use exact matches
    actual_values = actual_df.loc[common_dates, target].values
    forecast_values = forecast_df.loc[common_dates, target].values
    
    # Remove NaN pairs
    mask = ~(np.isnan(actual_values) | np.isnan(forecast_values))
    if mask.sum() == 0:
        return (None, None, None) if return_dates else (None, None)
    
    if return_dates:
        dates = common_dates[mask].values if hasattr(common_dates, 'values') else np.array(common_dates)[mask]
        return actual_values[mask], forecast_values[mask], dates
    else:
        return actual_values[mask], forecast_values[mask]


def calculate_horizon_metrics(
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    target: str,
    max_horizon: int
) -> Dict[int, Dict[str, float]]:
    """Calculate metrics for each horizon.
    
    For each horizon h, we calculate metrics by comparing:
    - actual[t+h] with forecast[t+h] for all available forecast origins t
    
    Parameters
    ----------
    actual_df : pd.DataFrame
        Actual data
    forecast_df : pd.DataFrame
        Forecast data
    target : str
        Target variable name
    max_horizon : int
        Maximum horizon to calculate
        
    Returns
    -------
    Dict[int, Dict[str, float]]
        Dictionary mapping horizon to metrics
    """
    if target not in actual_df.columns:
        return {}
    
    # Get training data standard deviation for standardized metrics
    # Use data before forecast period for training std
    if len(forecast_df) > 0:
        forecast_start = forecast_df.index.min()
        train_data = actual_df[actual_df.index < forecast_start]
        if len(train_data) > 0 and target in train_data.columns:
            train_std = train_data[target].std()
        else:
            train_std = actual_df[target].std()
    else:
        train_std = actual_df[target].std()
    
    if train_std == 0 or np.isnan(train_std):
        train_std = 1.0
    
    # actual_df is now monthly (from date column), forecast_df is weekly
    # Aggregate forecast weekly to monthly, then match with monthly actual data
    
    if target not in forecast_df.columns:
        return {}
    
    forecast_series = forecast_df[target].dropna()
    if len(forecast_series) == 0:
        return {}
    
    # Aggregate forecast weekly to monthly (ME = Month End, using mean)
    forecast_monthly = forecast_series.resample('ME').mean()
    
    # Get actual data for matching
    # actual_df is already monthly (from date column), so no need to resample
    if target not in actual_df.columns:
        return {}
    
    actual_series = actual_df[target]  # Already monthly, don't dropna() to preserve structure
    if len(actual_series) == 0:
        return {}
    
    # actual_series is already monthly (indexed by monthly dates from 'date' column)
    # No need to resample - it's already in monthly format
    actual_monthly = actual_series
    
    # Filter to forecast period (2024-01 to 2025-09)
    forecast_start = forecast_monthly.index.min()
    forecast_end = pd.Timestamp('2025-09-30')  # Up to 2025-09
    forecast_monthly = forecast_monthly[(forecast_monthly.index >= forecast_start) & 
                                        (forecast_monthly.index <= forecast_end)]
    
    # Match months and calculate metrics
    results = {}
    horizon = 1
    
    for forecast_date in forecast_monthly.index:
        if horizon > max_horizon:
            break
        
        # Find closest actual month (within 15 days)
        date_diffs = np.abs((actual_monthly.index - forecast_date).days)
        if len(date_diffs) > 0:
            closest_idx = date_diffs.argmin()
            closest_diff = date_diffs.iloc[closest_idx] if hasattr(date_diffs, 'iloc') else date_diffs[closest_idx]
            
            if closest_diff <= 15:  # Within same month
                actual_val = actual_monthly.iloc[closest_idx]
                forecast_val = forecast_monthly.loc[forecast_date]
                
                if pd.notna(actual_val) and pd.notna(forecast_val):
                    # Calculate metrics for this month
                    metrics = calculate_metrics(
                        np.array([actual_val]),
                        np.array([forecast_val]),
                        train_std
                    )
                    results[horizon] = metrics
                else:
                    # If either is NaN, still include but with NaN metrics
                    results[horizon] = {metric: np.nan for metric in METRICS}
                horizon += 1
            else:
                # No matching actual month, include forecast only
                forecast_val = forecast_monthly.loc[forecast_date]
                if pd.notna(forecast_val):
                    results[horizon] = {metric: np.nan for metric in METRICS}
                horizon += 1
        else:
            # No actual data available, include forecast only
            forecast_val = forecast_monthly.loc[forecast_date]
            if pd.notna(forecast_val):
                results[horizon] = {metric: np.nan for metric in METRICS}
            horizon += 1
    
    return results
    
    # Fallback: use original approach if monthly aggregation fails
    results = {}
    for i, horizon in enumerate(range(1, min(max_horizon + 1, len(actual_values) + 1)), 1):
        if i <= len(actual_values):
            actual_h = actual_values[i - 1]
            forecast_h = forecast_values[i - 1]
            
            metrics = calculate_metrics(
                np.array([actual_h]),
                np.array([forecast_h]),
                train_std
            )
            results[horizon] = metrics
    
    return results


def create_target_table(
    target: str,
    actual_df: pd.DataFrame,
    predictions_dir: Path,
    max_horizon: int = 88
) -> pd.DataFrame:
    """Create a table for a target variable with horizon × (model × metric) structure.
    
    Parameters
    ----------
    target : str
        Target variable name
    actual_df : pd.DataFrame
        Actual data
    predictions_dir : Path
        Directory containing predictions
    max_horizon : int
        Maximum horizon to include
        
    Returns
    -------
    pd.DataFrame
        Table with horizons as rows and (model × metric) as columns
    """
    # Collect metrics for all models
    all_metrics = {}
    
    for model in MODELS:
        forecast_df = load_forecast_data(model, predictions_dir)
        if forecast_df is None:
            continue
        
        horizon_metrics = calculate_horizon_metrics(
            actual_df, forecast_df, target, max_horizon
        )
        
        # Store metrics for this model
        for horizon, metrics in horizon_metrics.items():
            if horizon not in all_metrics:
                all_metrics[horizon] = {}
            for metric_name, metric_value in metrics.items():
                col_name = f"{model.upper()}_{metric_name}"
                all_metrics[horizon][col_name] = metric_value
    
    # Create DataFrame
    if not all_metrics:
        return pd.DataFrame()
    
    # Create list of all columns (model × metric combinations)
    # Only include metrics that will be displayed in tables
    all_columns = []
    for model in MODELS:
        for metric in METRICS:  # Calculate all metrics
            all_columns.append(f"{model.upper()}_{metric}")
    
    # Create rows for each horizon
    rows = []
    for horizon in range(1, max_horizon + 1):
        row = {'Horizon': horizon}
        for col in all_columns:
            if horizon in all_metrics and col in all_metrics[horizon]:
                row[col] = all_metrics[horizon][col]
            else:
                row[col] = np.nan
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.set_index('Horizon')
    
    return df


def dataframe_to_latex(
    df: pd.DataFrame,
    target: str,
    caption: Optional[str] = None,
    label: Optional[str] = None
) -> str:
    """Convert DataFrame to LaTeX table format with proper multirow headers.
    
    Only shows sMAE and sMSE metrics, displays months instead of horizons.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to convert
    target : str
        Target variable name
    caption : Optional[str]
        Table caption
    label : Optional[str]
        Table label for referencing
        
    Returns
    -------
    str
        LaTeX table code
    """
    if df.empty:
        return ""
    
    # Filter to only show sMAE and sMSE metrics
    filtered_cols = []
    for col in df.columns:
        parts = col.split("_")
        if len(parts) >= 2:
            metric = "_".join(parts[1:])
            if metric in TABLE_METRICS:
                filtered_cols.append(col)
    
    if not filtered_cols:
        return ""
    
    df_filtered = df[filtered_cols].copy()
    
    # Format numbers: round to 3 decimal places, handle NaN
    df_formatted = df_filtered.copy()
    for col in df_formatted.columns:
        df_formatted[col] = df_formatted[col].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) and not np.isinf(x) else "---"
        )
    
    # Group columns by model, only for sMAE and sMSE
    model_groups = {}
    for col in df_formatted.columns:
        parts = col.split("_")
        if len(parts) >= 2:
            model = parts[0]
            metric = "_".join(parts[1:])
            if metric in TABLE_METRICS:
                if model not in model_groups:
                    model_groups[model] = []
                model_groups[model].append((col, metric))
    
    # Generate month labels: 2024-01 to 2025-09 (21 months)
    from datetime import datetime
    month_labels = []
    start_date = datetime(2024, 1, 1)
    for i in range(21):
        month_date = datetime(start_date.year + (start_date.month + i - 1) // 12,
                             ((start_date.month + i - 1) % 12) + 1, 1)
        month_labels.append(month_date.strftime("%Y-%m"))
    
    # Generate LaTeX code
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("    \\centering")
    latex_lines.append("    \\normalsize")
    
    # Determine column alignment
    n_cols = len(df_formatted.columns) + 1  # +1 for Month column
    col_spec = "l" + "r" * (n_cols - 1)  # Left align Month, right align numbers
    
    latex_lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("        \\toprule")
    
    # First header row: model names
    header1 = "        \\multirow{2}{*}{Month}"
    for model in sorted(model_groups.keys()):
        n_metrics = len(model_groups[model])
        if n_metrics > 1:
            header1 += f" & \\multicolumn{{{n_metrics}}}{{c}}{{{model}}}"
        else:
            header1 += f" & {model}"
    header1 += " \\\\"
    latex_lines.append(header1)
    
    # Second header row: metrics (sMAE, sMSE)
    header2 = "        "
    for model in sorted(model_groups.keys()):
        for col, metric in sorted(model_groups[model], key=lambda x: TABLE_METRICS.index(x[1]) if x[1] in TABLE_METRICS else 999):
            header2 += f" & {metric}"
    header2 += " \\\\"
    latex_lines.append(header2)
    latex_lines.append("        \\midrule")
    
    # Data rows - only show first 21 horizons (months)
    max_display = 21
    
    for idx in range(1, max_display + 1):
        if idx not in df_formatted.index:
            continue
        row = df_formatted.loc[idx]
        month_label = month_labels[idx - 1] if idx <= len(month_labels) else f"Horizon {idx}"
        row_str = f"        {month_label}"
        for model in sorted(model_groups.keys()):
            for col, metric in sorted(model_groups[model], key=lambda x: TABLE_METRICS.index(x[1]) if x[1] in TABLE_METRICS else 999):
                row_str += f" & {row[col]}"
        row_str += " \\\\"
        latex_lines.append(row_str)
    
    latex_lines.append("        \\bottomrule")
    latex_lines.append("    \\end{tabular}")
    
    if caption:
        latex_lines.append(f"    \\caption{{{caption}}}")
    else:
        latex_lines.append(f"    \\caption{{Forecast metrics by month for {target} (2024-01 to 2025-09).}}")
    
    if label:
        latex_lines.append(f"    \\label{{{label}}}")
    else:
        # Create label from target name
        label_name = target.replace(".", "").replace("_", "").lower()
        latex_lines.append(f"    \\label{{tab:{label_name}_forecasts}}")
    
    latex_lines.append("\\end{table}")
    
    return "\n".join(latex_lines)


def create_all_tables(
    data_path: Optional[Path] = None,
    predictions_dir: Optional[Path] = None,
    tables_dir: Optional[Path] = None,
    max_horizon: int = 88
) -> Dict[str, pd.DataFrame]:
    """Create tables for all target variables.
    
    Parameters
    ----------
    data_path : Optional[Path]
        Path to actual data CSV file
    predictions_dir : Optional[Path]
        Directory containing predictions
    tables_dir : Optional[Path]
        Directory to save tables
    max_horizon : int
        Maximum horizon to include
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping target name to table DataFrame
    """
    # Set default paths
    if data_path is None:
        data_path = DATA_DIR / "data.csv"
    if predictions_dir is None:
        predictions_dir = PREDICTIONS_DIR
    if tables_dir is None:
        tables_dir = TABLES_DIR
    
    # Create tables directory
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Load actual data
    print(f"Loading actual data from {data_path}...")
    actual_df = load_actual_data(data_path)
    print(f"Loaded actual data: {len(actual_df)} rows, columns: {list(actual_df.columns[:5])}...")
    
    # Create tables for each target
    tables = {}
    for target in TARGETS:
        print(f"\nCreating table for {target}...")
        table = create_target_table(target, actual_df, predictions_dir, max_horizon)
        
        if not table.empty:
            # Save to LaTeX only
            output_file_tex = tables_dir / f"{target}_forecasts.tex"
            latex_code = dataframe_to_latex(table, target)
            with open(output_file_tex, 'w', encoding='utf-8') as f:
                f.write(latex_code)
            print(f"  Saved LaTeX to {output_file_tex}")
            
            print(f"  Shape: {table.shape}")
            print(f"  Columns: {list(table.columns[:5])}...")
            tables[target] = table
        else:
            print(f"  No data available for {target}")
    
    return tables


def main():
    """Main function to create all tables."""
    print("=" * 80)
    print("Creating forecast metrics tables by horizon")
    print("=" * 80)
    
    tables = create_all_tables()
    
    print("\n" + "=" * 80)
    print(f"Created {len(tables)} tables")
    print("=" * 80)
    
    for target, table in tables.items():
        print(f"\n{target}:")
        print(f"  Shape: {table.shape}")
        print(f"  Non-null values: {table.notna().sum().sum()} / {table.size}")
        print(f"  Sample (first 3 horizons, first 5 columns):")
        print(table.iloc[:3, :5])


if __name__ == "__main__":
    main()
