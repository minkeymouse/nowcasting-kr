
## DDFM debug investigation summary (2026-01-15)

### Problem
- DDFM training produced NaN loss and extreme sMSE/sMAE.
- Short-term recursive forecasting showed unstable factors and suspicious metrics.

### Key findings (runtime evidence)
- Target series had very high missingness (around 92-94%), so any leakage or scaling issue amplified instability.
- NaNs were originally flowing into autoencoder inputs during MCMC; switching to interpolated inputs removed NaNs.
- Target scaler was being refit on tiny/NaN-heavy update datasets, corrupting mean/scale.
- The largest remaining blow-up came from `update()` being fed raw-scale inputs (not aligned with training space), which exploded factors and predictions.

### Fixes applied
- **Prevent scaler corruption**: if a target scaler is already fitted, do not refit it in `DDFMDataset`; only transform using the existing scaler.
- **Imputation consistency**: use interpolated denoised data for MCMC training inputs to avoid NaNs in the autoencoder.
- **Feature scaling in dataset**: added optional feature scaling so encoder inputs don't dwarf targets.
- **Update path fixed**: recursive forecasting now updates DDFM using processed data (aligned with training space) instead of raw test data.
  - Implemented in `src/forecast/ddfm.py` via `_get_update_data_source(...)` and applied to both recursive and multi-horizon flows.

### Post-fix outcome
- Factors stayed in a reasonable range during prediction.
- Predictions and actuals were on comparable scales.
- Short-term metrics improved to:
  - Production: sMSE ~4.41, sMAE ~1.69
  - Investment: sMSE ~1.64, sMAE ~0.99

### Files touched
- `dfm-python/src/dfm_python/dataset/ddfm_dataset.py`
  - scaler refit guard, feature scaling support (kept)
- `dfm-python/src/dfm_python/models/ddfm.py`
  - MCMC input interpolation (kept)
- `src/forecast/ddfm.py`
  - processed update data source for `update()` (kept)
- `src/train/ddfm.py`, `dfm-python/src/dfm_python/encoder/simple_autoencoder.py`, `src/main.py`
  - debug instrumentation added during investigation and later removed
