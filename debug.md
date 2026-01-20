
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

## DFM debug investigation summary (2026-01-15)

### Problem
- DFM metrics were missing in the report; evaluation outputs existed but scales were inconsistent.
- Training/forecasting is slow; debug runs used 1 iteration and half dataset for feasibility.

### Key findings (runtime evidence)
- Recursive update inputs contained NaNs (expected for mixed-frequency), but were masked correctly.
- Predictions after inverse-transform were exploding (up to 1e12), indicating double-scaling.
- Target series resolution was stable (dataset + config both had 1 target).

### Fixes applied
- **Scaling fix**: stop fitting a new `target_scaler` in DFM training unless preprocessing actually provides a scaler (`data_loader.scaler` is not None). This avoids double-scaling at predict time.
- **Long-term consistency**: inverse-transform DFM/DDFM weekly forecasts before monthly aggregation, matching short-term evaluation.
- **Stability**: ensure `use_cholesky_filter: true` for DFM investment config (Cholesky path avoids SVD/pinv failures).

### Post-fix outcome
- DFM short-term metrics are now produced and added to the report:
  - Production: sMSE 1.955, sMAE 1.165
  - Investment: sMSE 1.657, sMAE 1.006
- Long-term DFM metrics generated for available horizons and added to appendix.

### Files touched
- `src/train/dfm.py`
  - target_scaler creation guard (kept)
- `src/main.py`
  - DFM/DDFM long-term inverse-transform before monthly aggregation (kept)
- `config/model/dfm/investment.yaml`
  - `use_cholesky_filter: true` (kept)
