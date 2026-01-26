# Data Processing Steps

## Overview
This document records the steps for processing `raw_data.csv` to create train and test datasets for production and investment models.

## Raw Data Structure
- **File**: `raw_data.csv`
- **Format**: Mixed weekly and monthly series
- **Date columns**: `date_m` (monthly), `date_w` (weekly)
- **Total rows**: 2136
- **Date range**: 1985-01-04 to 2025-12-05
- **Total columns**: 127 series

## Key Transformation: Target Series (log*100 → Original Scale)
**Target series are stored in log*100 format in raw_data.csv:**
- `KOIPALL.G` (production target): Values ~396-475 (log*100 format)
- `KOEQUIPTE` (investment target): Values ~374-381 (log*100 format)

**Inverse transformation**: Apply `e^{y/100}` to get original scale
- `KOIPALL.G`: After e^{y/100} → 52-115 (original index values)
- `KOEQUIPTE`: After e^{y/100} → 42-128 (original index values)

**Rationale**: Target series were transformed as `log(x) * 100` for stable training. To get back to original scale, we apply `e^{y/100}`.

## Processing Steps

### Step 1: Load Raw Data
- Load `raw_data.csv`
- Convert `date_w` to datetime
- Sort by date

### Step 2: Apply e^{y/100} to Target Series
- For `KOIPALL.G`: Apply `e^{y/100}` to reverse log*100 transformation
- For `KOEQUIPTE`: Apply `e^{y/100}` to reverse log*100 transformation
- This converts target series from log*100 space back to original scale

### Step 3: Split Train/Test
- **Cutoff date**: 2024-01-01
- **Train**: All data before 2024-01-01 (1985-01-04 to 2023-12-29)
- **Test**: All data from 2024-01-01 onwards (2024-01-05 to 2025-12-05)

### Step 4: Filter by Metadata
- **Production dataset**: Filter to series in `production_metadata.csv`
  - 42 series in metadata → 41 data series + date columns
- **Investment dataset**: Filter to series in `investment_metadata.csv`
  - 40 series in metadata → 39 data series + date columns

### Step 5: Save Datasets
- `train_production.csv`: Production training data (2035 rows)
- `test_production.csv`: Production test data (101 rows)
- `train_investment.csv`: Investment training data (2035 rows)
- `test_investment.csv`: Investment test data (101 rows)

## Verification
After processing:
- **KOIPALL.G (production)**:
  - Train: min=52.50, max=114.00, mean=84.70
  - Test: min=112.40, max=115.80, mean=113.61
- **KOEQUIPTE (production/investment)**:
  - Train: min=26.10, max=122.90, mean=73.65
  - Test: min=98.80, max=128.10, mean=112.58

## Notes
- Target series are now in original scale (not log*100) in train/test CSV files
- Other series remain as-is (they were not log*100 transformed)
- Mixed frequency (weekly/monthly) is preserved in the datasets
- Preprocessing pipeline (`src/preprocess.py`) will apply transformations (chg, logdiff, etc.) as specified in metadata
