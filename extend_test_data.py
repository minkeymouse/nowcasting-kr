"""Temporary script to extend test data with 2021-2023 from original CSV files.

This provides sufficient historical context for models like iTransformer
without requiring retraining.
"""

import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent

# Process both investment and production datasets
for data_model in ['investment', 'production']:
    print(f"\n{'='*80}")
    print(f"Processing {data_model} data...")
    print(f"{'='*80}")
    
    # Paths
    original_path = project_root / "data" / f"{data_model}.csv"
    test_path = project_root / "data" / f"test_{data_model}.csv"
    
    # Load original data
    print(f"Loading original data from: {original_path}")
    original_data = pd.read_csv(original_path)
    
    # Load test data
    print(f"Loading test data from: {test_path}")
    test_data = pd.read_csv(test_path)
    
    # Convert date columns to datetime
    if 'date_w' in original_data.columns:
        original_data['date_w'] = pd.to_datetime(original_data['date_w'], errors='coerce')
        test_data['date_w'] = pd.to_datetime(test_data['date_w'], errors='coerce')
        date_col = 'date_w'
    elif 'date' in original_data.columns:
        original_data['date'] = pd.to_datetime(original_data['date'], errors='coerce')
        test_data['date'] = pd.to_datetime(test_data['date'], errors='coerce')
        date_col = 'date'
    else:
        raise ValueError(f"No date column found in {data_model} data")
    
    # Filter 2021-2023 from original data
    original_data['year'] = original_data[date_col].dt.year
    historical_data = original_data[(original_data['year'] >= 2021) & (original_data['year'] <= 2023)].copy()
    historical_data = historical_data.drop(columns=['year'])
    
    print(f"Extracted {len(historical_data)} rows from 2021-2023")
    print(f"Original test data: {len(test_data)} rows")
    
    # Get date range
    if date_col in historical_data.columns:
        print(f"Historical data date range: {historical_data[date_col].min()} to {historical_data[date_col].max()}")
    
    # Check for overlap
    if date_col in historical_data.columns and date_col in test_data.columns:
        hist_max = historical_data[date_col].max()
        test_min = test_data[date_col].min()
        if hist_max >= test_min:
            print(f"WARNING: Overlap detected! Historical max: {hist_max}, Test min: {test_min}")
            # Remove overlapping rows from historical data
            historical_data = historical_data[historical_data[date_col] < test_min].copy()
            print(f"After removing overlap: {len(historical_data)} rows")
    
    # Concatenate: historical (2021-2023) + test data
    extended_test_data = pd.concat([historical_data, test_data], ignore_index=True)
    
    # Sort by date
    extended_test_data = extended_test_data.sort_values(by=date_col)
    
    print(f"Extended test data: {len(extended_test_data)} rows")
    print(f"Date range: {extended_test_data[date_col].min()} to {extended_test_data[date_col].max()}")
    
    # Save back to test file
    print(f"Saving extended test data to: {test_path}")
    extended_test_data.to_csv(test_path, index=False)
    print(f"✓ Saved {len(extended_test_data)} rows to {test_path}")

print(f"\n{'='*80}")
print("Done! Test data files have been extended with 2021-2023 data.")
print(f"{'='*80}")
