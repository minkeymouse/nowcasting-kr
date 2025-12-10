#!/usr/bin/env python3
"""Calculate DDFM forecasting metrics and update LaTeX tables."""

import pandas as pd
import numpy as np
from pathlib import Path
import re

# Load predictions
pred_df = pd.read_csv('predictions/ddfm.csv')
pred_df['date'] = pd.to_datetime(pred_df['date'])

# Load actual data
data = pd.read_csv('data/data.csv', index_col=0, parse_dates=True)
data = data.select_dtypes(include=[np.number])
data = data.sort_index()

# Resample to monthly
monthly = data.resample('MS').mean()

# Filter to 2024-01 to 2025-10
monthly = monthly[(monthly.index >= pd.Timestamp('2024-01-01')) & 
                  (monthly.index <= pd.Timestamp('2025-10-01'))]

targets = ['KOEQUIPTE', 'KOIPALL.G', 'KOWRCCNSE']

def calculate_metrics(pred, actual):
    """Calculate sMAE and sMSE (standardized by actual values)."""
    if len(pred) == 0 or len(actual) == 0:
        return None, None
    
    # Filter to valid pairs
    valid_mask = ~(pd.isna(pred) | pd.isna(actual))
    pred_valid = pred[valid_mask]
    actual_valid = actual[valid_mask]
    
    if len(pred_valid) == 0:
        return None, None
    
    # Calculate errors
    errors = pred_valid - actual_valid
    
    # Standardize by actual values (sMAE, sMSE)
    # sMAE = mean(|error| / |actual|) * 100
    # sMSE = mean((error / actual)^2) * 10000
    abs_actual = np.abs(actual_valid)
    abs_actual = np.where(abs_actual == 0, 1.0, abs_actual)  # Avoid division by zero
    
    sMAE = np.mean(np.abs(errors) / abs_actual) * 100
    sMSE = np.mean((errors / abs_actual) ** 2) * 10000
    
    return sMAE, sMSE

# Calculate metrics for each target and horizon
results = {}
for target in targets:
    results[target] = {}
    
    # Get actual values
    if target not in monthly.columns:
        print(f"Warning: {target} not found in monthly data")
        continue
    
    actual = monthly[target].values
    
    # Get predictions
    if target not in pred_df.columns:
        print(f"Warning: {target} not found in predictions")
        continue
    
    pred = pred_df[target].values
    
    # Calculate metrics for each horizon (month)
    for horizon in range(1, min(len(pred), len(actual)) + 1):
        pred_h = pred[:horizon]
        actual_h = actual[:horizon]
        
        sMAE, sMSE = calculate_metrics(pd.Series(pred_h), pd.Series(actual_h))
        results[target][horizon] = {'sMAE': sMAE, 'sMSE': sMSE}

# Calculate average across all targets
results['all'] = {}
for horizon in range(1, 23):  # 1 to 22
    sMAE_list = []
    sMSE_list = []
    for target in targets:
        if horizon in results[target]:
            if results[target][horizon]['sMAE'] is not None:
                sMAE_list.append(results[target][horizon]['sMAE'])
            if results[target][horizon]['sMSE'] is not None:
                sMSE_list.append(results[target][horizon]['sMSE'])
    
    if len(sMAE_list) > 0:
        results['all'][horizon] = {
            'sMAE': np.mean(sMAE_list),
            'sMSE': np.mean(sMSE_list)
        }

# Update LaTeX tables
def format_value(val):
    """Format value for LaTeX table."""
    if val is None or np.isnan(val):
        return '-'
    return f"{val:.2f}"

def update_table(target_name, results_dict):
    """Update LaTeX table for a target."""
    if target_name == 'all':
        file_path = Path('nowcasting-report/tables/tab_appendix_forecasting_all.tex')
    elif target_name == 'KOEQUIPTE':
        file_path = Path('nowcasting-report/tables/tab_appendix_forecasting_koequipte.tex')
    elif target_name == 'KOIPALL.G':
        file_path = Path('nowcasting-report/tables/tab_appendix_forecasting_koipall_g.tex')
    elif target_name == 'KOWRCCNSE':
        file_path = Path('nowcasting-report/tables/tab_appendix_forecasting_kowrccnse.tex')
    else:
        return
    
    if not file_path.exists():
        print(f"Warning: {file_path} not found")
        return
    
    # Read existing table
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace DDFM values for each horizon
    for horizon in range(1, 23):
        sMAE = format_value(results_dict.get(horizon, {}).get('sMAE'))
        sMSE = format_value(results_dict.get(horizon, {}).get('sMSE'))
        
        # Pattern: horizon number followed by columns ending with DDFM values
        # Match: "horizon & ... & - & - \\" or "horizon & ... & num & num \\"
        pattern = rf'({horizon}\s*&[^&]*&[^&]*&[^&]*&[^&]*&[^&]*&[^&]*&)\s*([^&]*)\s*&\s*([^&]*)\s*\\\\'
        
        def replace_ddfm(match):
            prefix = match.group(1)
            return f"{prefix} {sMAE} & {sMSE} \\\\"
        
        content = re.sub(pattern, replace_ddfm, content)
    
    # Write updated table
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Updated {file_path}")

# Update all tables
for target in ['all'] + targets:
    if target in results:
        update_table(target, results[target])

print("\n✓ All tables updated")
