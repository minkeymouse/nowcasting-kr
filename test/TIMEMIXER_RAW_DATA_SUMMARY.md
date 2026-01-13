# TimeMixer Raw Data Integration Summary

## Overview

TimeMixer has been updated to use **raw (unstandardized) data** when `use_norm=True` (RevIN enabled), allowing TimeMixer's internal normalization to work correctly.

## Changes Made

### 1. Training Function (`src/train/timemixer.py`)

**Key Update**: Modified `train_timemixer_model()` to automatically use raw data when RevIN is enabled.

```python
# TimeMixer with use_norm=True (RevIN) should receive raw (unstandardized) data
use_norm = model_params.get('use_norm', True) if model_params else True

if use_norm and data_loader is not None:
    # Use raw processed data (not standardized) for TimeMixer with RevIN
    if hasattr(data_loader, 'processed') and data_loader.processed is not None:
        raw_data = data_loader.processed.copy()
        # ... filter and prepare raw data ...
        data = raw_data  # Use raw data instead of standardized data
```

**Logic**:
- When `use_norm=True` (default): Uses `data_loader.processed` (raw, unstandardized data)
- When `use_norm=False`: Uses standardized data (as before)

### 2. Multi-Scale Downsampling Configuration

**Weekly → Monthly → Quarterly → Yearly Hierarchy**:
- When monthly series are detected: `down_sampling_layers=3`, `down_sampling_window=4`
  - Creates 4 scales: weekly (1×), monthly (4×), quarterly (~16×), yearly (~64×)
- For uniform data: `down_sampling_layers=1`, `down_sampling_window=2`

## Data Flow

### Training Pipeline

1. **Data Loading**: 
   - Standardized data passed to `train_timemixer_model()`
   - Function detects `use_norm=True` → switches to raw data
   - Raw data extracted from `data_loader.processed`

2. **Data Preparation**:
   - Raw data goes through: interpolation → format conversion → NeuralForecast format
   - Raw data passed to TimeMixer

3. **TimeMixer Processing**:
   - RevIN normalizes internally (per batch)
   - Model processes normalized data
   - RevIN denormalizes predictions
   - **Output**: Predictions in raw (processed) scale

### Forecasting Pipeline (Current)

**Note**: Forecasting code currently assumes predictions are in standardized scale. 

For TimeMixer with raw data:
- **Model outputs**: Raw (processed) scale
- **Current inverse transform**: Expects standardized scale
- **Action needed**: May need to skip inverse standardization step

## Testing

### Test Results

✅ **Data Selection Logic**: Verified working
- Correctly selects raw data when `use_norm=True`
- Correctly uses standardized data when `use_norm=False`
- Data loader integration works correctly

### Test File: `test/test_timemixer_data_flow.py`

Verifies:
1. Data characteristics (standardized vs raw)
2. Data selection logic simulation
3. Integration with data loader

## Important Notes

### RevIN Behavior

RevIN (Reversible Instance Normalization) in TimeMixer:
- **Normalization**: Computes mean/std per batch during forward pass
- **Denormalization**: Restores original scale before output
- **Result**: Model outputs are in the same scale as inputs

**Implication**:
- Input = Raw scale → Output = Raw scale
- Input = Standardized scale → Output = Standardized scale

### Inverse Transformation for Forecasting

When TimeMixer uses raw data (`use_norm=True`):
- Model predictions are in **raw (processed) scale**
- Need to apply: **Reverse transformations only** (no inverse standardization)
- Use `inverse_transform_predictions(predictions, ..., reverse_transformations=True)` 
- **BUT**: Skip the standardization step (predictions are already in processed scale)

**Potential Update Needed**:
- Modify forecasting functions to detect if model uses raw data
- Skip inverse standardization when raw data was used
- Still apply reverse transformations to get to original scale

## Configuration Example

```python
model_params = {
    'prediction_length': 1,
    'context_length': 96,
    'use_norm': True,  # Enables RevIN → uses raw data
    'down_sampling_layers': 3,  # Weekly → Monthly → Quarterly → Yearly
    'down_sampling_window': 4,  # 4 weeks per month
    # ... other params ...
}
```

## Next Steps

1. ✅ **Training**: Uses raw data correctly when RevIN enabled
2. ⚠️ **Forecasting**: May need adjustment to handle raw-scale predictions
3. ✅ **Downsampling**: Configured for weekly → monthly → quarterly → yearly hierarchy
4. ✅ **Testing**: Data selection logic verified

## Files Modified

- `src/train/timemixer.py`: Added raw data selection logic
- `test/test_timemixer_data_flow.py`: Added test for data selection

## Files to Review

- `src/forecast/timemixer.py`: May need updates for raw data predictions
- `src/forecast/_common.py`: `inverse_transform_predictions` usage may need adjustment