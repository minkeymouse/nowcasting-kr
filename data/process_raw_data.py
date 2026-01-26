#!/usr/bin/env python3
"""
Process raw_data.csv to create train and test datasets for production and investment models.

Steps:
1. Load raw_data.csv (mixed weekly/monthly series)
2. Apply e^{y/100} transformation to target series (KOIPALL.G, KOEQUIPTE) to reverse log*100
3. Split into train/test at cutoff date (2024-01-01)
4. Filter to production vs investment series based on metadata
5. Save train_production.csv, test_production.csv, train_investment.csv, test_investment.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Target series that need e^{y/100} transformation
TARGET_SERIES = ['KOIPALL.G', 'KOEQUIPTE']

# Cutoff date for train/test split
CUTOFF_DATE = pd.Timestamp('2024-01-01')

def load_metadata(metadata_path: str) -> pd.DataFrame:
    """Load metadata file."""
    metadata = pd.read_csv(metadata_path)
    if 'SeriesID' in metadata.columns and 'Series_ID' not in metadata.columns:
        metadata = metadata.rename(columns={'SeriesID': 'Series_ID'})
    return metadata

def apply_target_inverse_transform(data: pd.DataFrame, target_series: list) -> pd.DataFrame:
    """Apply e^{y/100} transformation to target series to reverse log*100.
    
    Parameters
    ----------
    data : pd.DataFrame
        Raw data with target series in log*100 format
    target_series : list
        List of target series names to transform
        
    Returns
    -------
    pd.DataFrame
        Data with target series transformed back to original scale
    """
    data = data.copy()
    for series in target_series:
        if series in data.columns:
            # Apply e^{y/100} to reverse log*100 transformation
            mask = data[series].notna()
            if mask.any():
                original_values = np.exp(data.loc[mask, series] / 100.0)
                data.loc[mask, series] = original_values
                logger.info(f"Applied e^{{y/100}} to {series}: transformed {mask.sum()} values")
                logger.info(f"  Sample: {data[series].dropna().head(3).values}")
            else:
                logger.warning(f"{series} has no non-null values")
        else:
            logger.warning(f"{series} not found in data columns")
    return data

def filter_series_by_metadata(data: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Filter data to only include series present in metadata.
    
    Parameters
    ----------
    data : pd.DataFrame
        Full dataset
    metadata : pd.DataFrame
        Metadata with Series_ID column
        
    Returns
    -------
    pd.DataFrame
        Filtered data with only series in metadata
    """
    series_col = 'Series_ID' if 'Series_ID' in metadata.columns else 'SeriesID'
    valid_series = set(metadata[series_col].values)
    
    # Keep date columns
    date_cols = ['date_m', 'date_w', 'date']
    valid_cols = [col for col in data.columns if col in date_cols or col in valid_series]
    
    filtered_data = data[valid_cols].copy()
    logger.info(f"Filtered to {len(valid_cols)} columns ({len(valid_cols) - len(date_cols)} data series)")
    
    return filtered_data

def split_train_test(data: pd.DataFrame, cutoff_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets at cutoff date.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with date_w as datetime index
    cutoff_date : pd.Timestamp
        Cutoff date (exclusive for train, inclusive for test)
        
    Returns
    -------
    tuple
        (train_data, test_data)
    """
    if 'date_w' in data.columns:
        data = data.set_index('date_w')
    elif not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have date_w column or datetime index")
    
    train_data = data[data.index < cutoff_date].copy()
    test_data = data[data.index >= cutoff_date].copy()
    
    logger.info(f"Split at {cutoff_date.date()}: train={len(train_data)} rows, test={len(test_data)} rows")
    logger.info(f"  Train range: {train_data.index.min().date()} to {train_data.index.max().date()}")
    logger.info(f"  Test range: {test_data.index.min().date()} to {test_data.index.max().date()}")
    
    return train_data, test_data

def main():
    """Main processing function."""
    data_dir = Path(__file__).parent
    
    # Paths
    raw_data_path = data_dir / 'raw_data.csv'
    production_metadata_path = data_dir / 'production_metadata.csv'
    investment_metadata_path = data_dir / 'investment_metadata.csv'
    
    logger.info("=" * 80)
    logger.info("Processing raw_data.csv to create train/test datasets")
    logger.info("=" * 80)
    
    # Step 1: Load raw data
    logger.info("\n[Step 1] Loading raw_data.csv...")
    raw_data = pd.read_csv(raw_data_path)
    logger.info(f"Loaded {len(raw_data)} rows, {len(raw_data.columns)} columns")
    logger.info(f"Date range: {raw_data['date_w'].min()} to {raw_data['date_w'].max()}")
    
    # Convert date_w to datetime
    raw_data['date_w'] = pd.to_datetime(raw_data['date_w'], errors='coerce')
    raw_data = raw_data.sort_values('date_w')
    
    # Step 2: Apply e^{y/100} to target series
    logger.info("\n[Step 2] Applying e^{y/100} transformation to target series...")
    logger.info(f"Target series: {TARGET_SERIES}")
    processed_data = apply_target_inverse_transform(raw_data, TARGET_SERIES)
    
    # Step 3: Split into train/test
    logger.info(f"\n[Step 3] Splitting into train/test at {CUTOFF_DATE.date()}...")
    train_data, test_data = split_train_test(processed_data, CUTOFF_DATE)
    
    # Step 4: Load metadata
    logger.info("\n[Step 4] Loading metadata files...")
    production_metadata = load_metadata(production_metadata_path)
    investment_metadata = load_metadata(investment_metadata_path)
    logger.info(f"Production metadata: {len(production_metadata)} series")
    logger.info(f"Investment metadata: {len(investment_metadata)} series")
    
    # Step 5: Filter and save production datasets
    logger.info("\n[Step 5] Creating production datasets...")
    train_production = filter_series_by_metadata(train_data.reset_index(), production_metadata)
    test_production = filter_series_by_metadata(test_data.reset_index(), production_metadata)
    
    # Ensure date_w column exists
    if 'date_w' not in train_production.columns and train_production.index.name == 'date_w':
        train_production = train_production.reset_index()
    if 'date_w' not in test_production.columns and test_production.index.name == 'date_w':
        test_production = test_production.reset_index()
    
    train_production_path = data_dir / 'train_production.csv'
    test_production_path = data_dir / 'test_production.csv'
    train_production.to_csv(train_production_path, index=False)
    test_production.to_csv(test_production_path, index=False)
    logger.info(f"Saved: {train_production_path} ({len(train_production)} rows)")
    logger.info(f"Saved: {test_production_path} ({len(test_production)} rows)")
    
    # Step 6: Filter and save investment datasets
    logger.info("\n[Step 6] Creating investment datasets...")
    train_investment = filter_series_by_metadata(train_data.reset_index(), investment_metadata)
    test_investment = filter_series_by_metadata(test_data.reset_index(), investment_metadata)
    
    # Ensure date_w column exists
    if 'date_w' not in train_investment.columns and train_investment.index.name == 'date_w':
        train_investment = train_investment.reset_index()
    if 'date_w' not in test_investment.columns and test_investment.index.name == 'date_w':
        test_investment = test_investment.reset_index()
    
    train_investment_path = data_dir / 'train_investment.csv'
    test_investment_path = data_dir / 'test_investment.csv'
    train_investment.to_csv(train_investment_path, index=False)
    test_investment.to_csv(test_investment_path, index=False)
    logger.info(f"Saved: {train_investment_path} ({len(train_investment)} rows)")
    logger.info(f"Saved: {test_investment_path} ({len(test_investment)} rows)")
    
    # Step 7: Verify target series values
    logger.info("\n[Step 7] Verifying target series values...")
    for series in TARGET_SERIES:
        if series in train_production.columns:
            train_vals = train_production[series].dropna()
            test_vals = test_production[series].dropna()
            logger.info(f"{series} (production):")
            logger.info(f"  Train: min={train_vals.min():.2f}, max={train_vals.max():.2f}, mean={train_vals.mean():.2f}")
            logger.info(f"  Test: min={test_vals.min():.2f}, max={test_vals.max():.2f}, mean={test_vals.mean():.2f}")
        if series in train_investment.columns:
            train_vals = train_investment[series].dropna()
            test_vals = test_investment[series].dropna()
            logger.info(f"{series} (investment):")
            logger.info(f"  Train: min={train_vals.min():.2f}, max={train_vals.max():.2f}, mean={train_vals.mean():.2f}")
            logger.info(f"  Test: min={test_vals.min():.2f}, max={test_vals.max():.2f}, mean={test_vals.mean():.2f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Processing complete!")
    logger.info("=" * 80)

if __name__ == '__main__':
    main()
