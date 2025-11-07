# DFM-Python 0.1.3: New Patterns and Clock-Based Framework

## Overview

Version 0.1.3 introduces a **clock-based mixed-frequency framework** that synchronizes all latent factors to a common base frequency. This document summarizes the new patterns and how to integrate them into our codebase.

## Key New Features

### 1. Clock Parameter

**What it is:**
- Base frequency at which all latent factors evolve
- Default: `'m'` (monthly)
- Options: `'d'`, `'w'`, `'m'`, `'q'`, `'sa'`, `'a'`

**Where to set it:**
```yaml
dfm:
  clock: "m"  # Monthly clock (default, recommended)
  threshold: 1e-5
  max_iter: 5000
  nan_method: 2
  nan_k: 3
  ar_lag: 1
```

**In DFMConfig:**
```python
config = DFMConfig(
    series=series_list,
    clock="m",  # Base frequency for all factors
    # ... other params
)
```

### 2. Clock-Based Synchronization

**How it works:**
1. **Global Clock**: All latent factors (global and block-level) evolve at the clock frequency
2. **Tent Kernels**: Lower-frequency observations (e.g., quarterly) are mapped to higher-frequency latent states using deterministic tent kernels
3. **Missing Data**: Higher-frequency observations (e.g., daily) are handled via missing data when clock is slower

**Example:**
- Clock = `'m'` (monthly)
- Quarterly GDP → mapped to 5 monthly latent states with weights `[1, 2, 3, 2, 1]`
- Daily exchange rate → handled as missing data on non-month-end days

### 3. Tent Kernel Aggregation

**Supported frequency pairs:**
- Quarterly → Monthly: `[1, 2, 3, 2, 1]` (5 periods)
- Semi-annual → Monthly: `[1, 2, 3, 4, 3, 2, 1]` (7 periods)
- Annual → Monthly: `[1, 2, 3, 4, 5, 4, 3, 2, 1]` (9 periods)

**Maximum tent size:** 12 periods (larger gaps use missing data approach)

## Integration with Our Codebase

### Current State

✅ **What we're doing correctly:**
1. Resampling all data to monthly frequency (`'MS'`) in `get_vintage_data()`
2. Handling quarterly data with tent structure (NaN between quarters)
3. Daily data aggregated to monthly (last value of month)

❌ **What needs updating:**
1. **Missing clock parameter** in our config files
2. **Clock parameter not passed** to DFMConfig when loading from database
3. **No explicit clock handling** in data loading functions

### Required Changes

#### 1. Update Config Files

**File: `app/config/default.yaml`**
```yaml
dfm:
  clock: "m"  # ADD THIS - Monthly clock (default)
  threshold: 1e-5
  max_iter: 5000
  nan_method: 2
  nan_k: 3
  ar_lag: 1
```

**File: `app/config/test.yaml`**
```yaml
dfm:
  clock: "m"  # ADD THIS
  threshold: 1e-5
  max_iter: 100
  nan_method: 2
  nan_k: 3
  ar_lag: 1
```

#### 2. Update Config Loading

**File: `app/utils/utils.py`** - In `load_model_config_with_hydra_fallback()`:
```python
# When merging DFM config, include clock parameter
if dfm_cfg_dict:
    for key in ['ar_lag', 'threshold', 'max_iter', 'nan_method', 'nan_k', 'clock']:  # ADD 'clock'
        if key in dfm_cfg_dict and dfm_cfg_dict[key] is not None:
            setattr(model_cfg, key, dfm_cfg_dict[key])
```

**File: `app/jobs/train.py`** - In `main()`:
```python
# Merge DFM estimation parameters from Hydra into model config
if dfm_cfg_dict:
    for key in ['ar_lag', 'threshold', 'max_iter', 'nan_method', 'nan_k', 'clock']:  # ADD 'clock'
        if key in dfm_cfg_dict and dfm_cfg_dict[key] is not None:
            setattr(model_cfg, key, dfm_cfg_dict[key])
```

#### 3. Update CSV Spec to Hydra Config Conversion

**File: `app/adapters/adapter_database.py`** - In `csv_spec_to_hydra_config()`:
```python
dfm_defaults = {
    'ar_lag': 1,
    'factors_per_block': None,
    'threshold': 1e-5,
    'max_iter': 5000,
    'nan_method': 2,
    'nan_k': 3,
    'clock': 'm'  # ADD THIS - default to monthly
}
```

#### 4. Verify Data Alignment Matches Clock

**File: `app/database/operations.py`** - In `get_vintage_data()`:

Our current implementation already aligns with `clock='m'`:
- ✅ Resamples to monthly (`'MS'`)
- ✅ Quarterly data uses tent structure (NaN between quarters)
- ✅ Daily data aggregated to monthly

**No changes needed** - our data alignment is correct for monthly clock.

## Data Format Requirements

### Quarterly Series (with clock='m')

**Current format (correct):**
```csv
Date,gdp_real
2000-01-01,        # Empty - not quarter-end
2000-02-01,        # Empty - not quarter-end
2000-03-01,100.5   # Q1 value (quarter-end)
2000-04-01,        # Empty - not quarter-end
2000-05-01,        # Empty - not quarter-end
2000-06-01,101.2   # Q2 value (quarter-end)
```

**Why this works:**
- DFM uses tent kernels to map quarterly values to 5 monthly latent states
- NaN values between quarters are expected (tent structure)
- Only quarter-end months (March, June, September, December) have values

### Monthly Series (with clock='m')

**Format:**
- Include values for all months
- No special handling needed (already at clock frequency)

### Daily Series (with clock='m')

**Current format (correct):**
- Aggregated to monthly using `.resample('MS').last()`
- Only month-end values are kept
- This matches the missing data approach for higher-frequency data

## API Changes

### DFMConfig

**New attribute:**
```python
config.clock  # str: Base frequency ('m' by default)
```

**Validation:**
- Clock frequency is validated on initialization
- Must be one of: `'d'`, `'w'`, `'m'`, `'q'`, `'sa'`, `'a'`

### dfm() Function

**No API changes** - clock is passed via DFMConfig:
```python
result = dfm(X, config, threshold=1e-4, max_iter=1000)
# config.clock is used internally
```

## Benefits of Clock-Based Framework

1. **Simplified Kalman Filter**: Single unified filter at clock frequency
2. **Consistent Factor Evolution**: All factors evolve synchronously
3. **Robust Aggregation**: Deterministic tent kernels ensure proper aggregation
4. **Flexible**: Works with any frequency combination

## Testing Checklist

- [ ] Add `clock: "m"` to all config files
- [ ] Update config loading to include clock parameter
- [ ] Verify data alignment matches clock frequency
- [ ] Test with quarterly series (should use tent kernels)
- [ ] Test with daily series (should aggregate to monthly)
- [ ] Test with monthly series (should work directly)
- [ ] Verify DFM estimation works with clock parameter

## Migration Notes

1. **Backward Compatibility**: If clock is not specified, defaults to `'m'` (monthly)
2. **Data Format**: No changes needed - our current data format is compatible
3. **Config Files**: Add `clock: "m"` to be explicit (recommended)

## References

- README: Clock-Based Mixed-Frequency Framework section
- DFMConfig: `clock` parameter (line 24 in config.py)
- Default: `'m'` (monthly)

---

**Status**: Ready for implementation  
**Priority**: Medium (adds explicit clock parameter, but defaults work)  
**Breaking Changes**: None (backward compatible)

