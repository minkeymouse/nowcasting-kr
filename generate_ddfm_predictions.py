#!/usr/bin/env python3
"""Generate DDFM predictions for 2024-01 to 2025-10 using weekly data."""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import torch

targets = ['KOEQUIPTE', 'KOIPALL.G', 'KOWRCCNSE']
checkpoint_dir = Path('checkpoints')
data_path = Path('data/data.csv')
output_path = Path('predictions/ddfm.csv')

# Load data - keep weekly frequency
print("Loading data...")
data = pd.read_csv(data_path, index_col=0, parse_dates=True)
# Drop non-numeric columns
data = data.select_dtypes(include=[np.number])
data = data.sort_index()
data = data[data.index >= pd.Timestamp('1985-01-01')]

# Keep weekly data (don't resample to monthly)
weekly_data = data.copy()

# Also create monthly for actual values
monthly = data.resample('MS').mean()

# Generate predictions
rows = []
for target in targets:
    print(f"\nProcessing {target}...")
    checkpoint_path = checkpoint_dir / f"{target}_ddfm" / "model.pkl"
    
    if not checkpoint_path.exists():
        print(f"  Checkpoint not found: {checkpoint_path}")
        continue
    
    # Load checkpoint
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    model = checkpoint_data.get('model') or checkpoint_data.get('forecaster')
    metadata = checkpoint_data.get('metadata', {})
    
    if model is None:
        print(f"  Model not found in checkpoint")
        continue
    
    # Use CPU for stability - force model to CPU
    device = "cpu"
    try:
        if hasattr(model, 'to'):
            model.to(device)
        elif hasattr(model, 'cpu'):
            model.cpu()
    except:
        pass
    
    # Get scaler info from checkpoint metadata or model
    Mx = None
    Wx = None
    
    # Try to get from model's training state
    if hasattr(model, 'training_state') and model.training_state is not None:
        if hasattr(model.training_state, 'result') and hasattr(model.training_state.result, 'Mx'):
            Mx = np.asarray(model.training_state.result.Mx)
            Wx = np.asarray(model.training_state.result.Wx)
    
    # Try to get from model.result
    if Mx is None and hasattr(model, 'result') and hasattr(model.result, 'Mx'):
        Mx = np.asarray(model.result.Mx)
        Wx = np.asarray(model.result.Wx)
    
    # Get model series names
    try:
        series_names = [s.series_id for s in model.config.series]
    except:
        series_names = list(weekly_data.columns)
    
    # Filter available series
    available_series = [s for s in series_names if s in weekly_data.columns]
    if target not in available_series:
        available_series.append(target)
    
    # Generate predictions for each month (but use weekly data)
    months = pd.date_range('2024-01-01', '2025-10-01', freq='MS')
    
    for month in months:
        # Get data up to 4 weeks before (for 4weeks forecast)
        cutoff_4w = month - pd.Timedelta(weeks=4)
        cutoff_1w = month - pd.Timedelta(weeks=1)
        
        for weeks_before, cutoff in [('4weeks', cutoff_4w), ('1weeks', cutoff_1w)]:
            try:
                # Get available weekly data up to cutoff
                y_available = weekly_data.loc[:cutoff, available_series].copy()
                y_available = y_available.dropna(how='all')
                
                if len(y_available) == 0:
                    rows.append({
                        'target': target,
                        'weeks_before': weeks_before,
                        'date': month.strftime('%Y-%m-%d'),
                        'month': month.strftime('%Y-%m'),
                        'forecast': None,
                        'actual': None,
                        'error': None,
                        'status': 'failed',
                        'error_msg': 'no data'
                    })
                    continue
                
                # Get original scale from training data (1985-2019) for inverse transform
                train_data = weekly_data.loc[:pd.Timestamp('2019-12-31'), available_series]
                orig_Mx_dict = {s: train_data[s].mean() for s in available_series}
                orig_Wx_dict = {s: train_data[s].std() for s in available_series}
                for s in orig_Wx_dict:
                    if orig_Wx_dict[s] == 0:
                        orig_Wx_dict[s] = 1.0
                
                # Get model series order
                model_series = [s.series_id for s in model.config.series]
                
                # Get standardized scaler (Mx/Wx) for update
                if Mx is None or Wx is None:
                    # Use training period statistics for standardization
                    Mx_use = train_data.mean().values
                    Wx_use = train_data.std().values
                    Wx_use = np.where(Wx_use == 0, 1.0, Wx_use)
                else:
                    Mx_use = Mx[:len(available_series)] if len(Mx) >= len(available_series) else Mx
                    Wx_use = Wx[:len(available_series)] if len(Wx) >= len(available_series) else Wx
                    Wx_use = np.where(Wx_use == 0, 1.0, Wx_use)
                
                # Standardize all available data (full history for update)
                y_vals = y_available.values
                if len(Mx_use) >= y_vals.shape[1]:
                    X_std = (y_vals - Mx_use[:y_vals.shape[1]]) / Wx_use[:y_vals.shape[1]]
                else:
                    X_std = y_vals
                X_std = np.where(np.isfinite(X_std), X_std, 0.0)
                
                # Convert to tensor
                X_std_tensor = torch.tensor(X_std, dtype=torch.float32, device=device)
                
                # Update and predict (predict 4 weeks to cover the month)
                # Note: update expects standardized data, predict returns unstandardized
                model.update(X_std_tensor, history=None)
                pred = model.predict(horizon=4, return_series=True, return_factors=False)
                
                if isinstance(pred, tuple):
                    pred = pred[0]
                
                # Convert tensor to numpy if needed
                if isinstance(pred, torch.Tensor):
                    pred = pred.detach().cpu().numpy()
                elif hasattr(pred, 'cpu'):
                    pred = pred.cpu().numpy()
                elif not isinstance(pred, np.ndarray):
                    pred = np.asarray(pred)
                
                pred = np.asarray(pred)
                
                # predict() returns standardized values, need inverse transform to original scale
                # pred columns match model.config.series order
                # Get original scale for each series in model order
                orig_Mx_model = []
                orig_Wx_model = []
                for s in model_series:
                    if s in orig_Mx_dict:
                        orig_Mx_model.append(orig_Mx_dict[s])
                        orig_Wx_model.append(orig_Wx_dict[s])
                    else:
                        # Use default if series not in training data
                        orig_Mx_model.append(0.0)
                        orig_Wx_model.append(1.0)
                
                orig_Mx_model = np.array(orig_Mx_model)
                orig_Wx_model = np.array(orig_Wx_model)
                orig_Wx_model = np.where(orig_Wx_model == 0, 1.0, orig_Wx_model)
                
                # Inverse transform: pred_raw = pred * Wx + Mx
                if len(orig_Mx_model) >= pred.shape[1]:
                    Mx_pred = orig_Mx_model[:pred.shape[1]]
                    Wx_pred = orig_Wx_model[:pred.shape[1]]
                    pred_raw = pred * Wx_pred + Mx_pred
                else:
                    pred_raw = pred
                
                # Get target series prediction (average of 4 weekly predictions for the month)
                try:
                    target_idx = model_series.index(target)
                    pred_vals = pred_raw[:, target_idx] if pred_raw.shape[1] > target_idx else pred_raw[:, 0]
                except (ValueError, IndexError):
                    # Fallback: use first column
                    pred_vals = pred_raw[:, 0] if pred_raw.shape[1] > 0 else pred_raw.flatten()
                
                # Average weekly predictions to get monthly prediction
                pred_val = float(np.mean(pred_vals))
                
                # Get actual value from monthly data
                actual_val = None
                if target in monthly.columns and month in monthly.index:
                    actual_val = float(monthly.loc[month, target]) if not pd.isna(monthly.loc[month, target]) else None
                
                error = None
                if actual_val is not None and pred_val is not None:
                    error = float(pred_val - actual_val)
                
                rows.append({
                    'target': target,
                    'weeks_before': weeks_before,
                    'date': month.strftime('%Y-%m-%d'),
                    'month': month.strftime('%Y-%m'),
                    'forecast': pred_val,
                    'actual': actual_val,
                    'error': error,
                    'status': 'ok',
                    'error_msg': None
                })
                
            except Exception as e:
                rows.append({
                    'target': target,
                    'weeks_before': weeks_before,
                    'date': month.strftime('%Y-%m-%d'),
                    'month': month.strftime('%Y-%m'),
                    'forecast': None,
                    'actual': None,
                    'error': None,
                    'status': 'failed',
                    'error_msg': str(e)
                })
                print(f"  Error for {month} ({weeks_before}): {e}")

# Save results - pivot to date, target columns format
output_path.parent.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(rows)

# Filter to only 4weeks (or use 1weeks if preferred)
df_filtered = df[df['weeks_before'] == '4weeks'].copy()

# Pivot: date as index, targets as columns
df_pivot = df_filtered.pivot_table(
    index='date',
    columns='target',
    values='forecast',
    aggfunc='first'  # Use first value if duplicates
).reset_index()

# Rename columns to match target names exactly
df_pivot.columns.name = None
df_pivot = df_pivot.rename_axis(None, axis=1)

# Sort by date
df_pivot['date'] = pd.to_datetime(df_pivot['date'])
df_pivot = df_pivot.sort_values('date')
df_pivot['date'] = df_pivot['date'].dt.strftime('%Y-%m-%d')

# Save
df_pivot.to_csv(output_path, index=False)
print(f"\n✓ Saved {len(df_pivot)} predictions to {output_path}")
print(f"  Shape: {df_pivot.shape}")
print(f"  Columns: {list(df_pivot.columns)}")
print(f"  Date range: {df_pivot['date'].iloc[0]} to {df_pivot['date'].iloc[-1]}")
