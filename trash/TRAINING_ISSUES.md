# Training Code Issues Identified

## Critical Issues

### 1. **Hardcoded Target Series in `sktime.py` (Line 53)**
**Issue:** Default target series are hardcoded to `['KOEQUIPTE', 'KOWRCCNSE', 'KOIPALL.G']` which don't exist in all model data files.

**Impact:**
- Consumption model: Has `KOEQUIPTE`, `KOWRCCNSE` but NOT `KOIPALL.G`
- Investment model: Only has `KOEQUIPTE`
- Production model: Has all three

**Location:** `src/train/sktime.py:53`

**Fix:** Should determine target series from config or metadata, not hardcode defaults.

---

### 2. **Hardcoded Frequency 'w' in `dfm_python.py` (Lines 109, 128)**
**Issue:** When creating minimal DFM/DDFM configs, all series are hardcoded to frequency='w' (weekly), ignoring actual mixed frequencies.

**Impact:**
- Monthly series will be treated as weekly
- Incorrect state space calculation
- Wrong aggregation behavior

**Locations:**
- `src/train/dfm_python.py:109` (DFM)
- `src/train/dfm_python.py:128` (DDFM)

**Fix:** Should read frequencies from metadata or use frequency dict from config.

---

### 3. **Missing Metadata Access in Training Functions**
**Issue:** Training functions receive preprocessed `data` DataFrame but don't receive `metadata` information about series frequencies/transformations.

**Impact:**
- Cannot properly determine series frequencies for DFM config
- Cannot validate target series against metadata
- Loss of information about data characteristics

**Locations:**
- `src/train/dfm_python.py:19-28` (function signature)
- `src/train/sktime.py:11-20` (function signature)

**Fix:** Pass metadata or data_loader object through to training functions.

---

### 4. **Arbitrary DateTime Index Creation in `sktime.py` (Lines 82-86)**
**Issue:** Creates fake datetime index starting from '2020-01-01' if data doesn't have DatetimeIndex, losing actual temporal information.

**Impact:**
- Wrong temporal relationships in neural models
- Incorrect frequency handling
- Loss of actual date context

**Location:** `src/train/sktime.py:82-86`

**Fix:** Should preserve original index from data_loader (which has proper datetime index from date_w).

---

## Medium Priority Issues

### 5. **Data Modification In-Place in `sktime.py` (Line 86)**
**Issue:** `data = data.copy()` followed by `data.index = dates` modifies the copy but could cause confusion.

**Impact:**
- Potential for bugs if original data is referenced elsewhere
- Index modification happens after data preparation

**Location:** `src/train/sktime.py:86`

**Fix:** Ensure clean separation between original and processed data.

---

### 6. **Inconsistent Target Series Handling**
**Issue:** 
- DFM uses `target_series` from config correctly but falls back to None (all series)
- Sktime uses hardcoded defaults if config doesn't specify

**Impact:**
- Different behavior between model types
- Sktime models may fail if defaults don't exist in data

**Locations:**
- `src/train/dfm_python.py:71-85`
- `src/train/sktime.py:53-59`

**Fix:** Standardize target series resolution logic.

---

### 7. **Missing Data Model Context**
**Issue:** Training functions don't know which data model (consumption/investment/production) they're training on.

**Impact:**
- Cannot adapt behavior based on data model
- Cannot validate targets against model-specific metadata
- Logging less informative

**Location:** Both training functions

**Fix:** Pass data_model parameter from main.py.

---

## Low Priority Issues

### 8. **Unused Parameters**
**Issue:** Some parameters are passed but not used:
- `config_name` in both functions (kept for compatibility but unused)
- `horizons` parameter (documented as unused)

**Impact:** Code clarity, but not breaking.

**Locations:**
- `src/train/dfm_python.py:21`
- `src/train/sktime.py:13, 17`

---

### 9. **Error Handling for Missing Target Series**
**Issue:** Sktime falls back to first column if no targets found, which may not be the desired target.

**Impact:**
- Training might proceed with wrong target
- No clear error if expected targets are missing

**Location:** `src/train/sktime.py:57-59`

**Fix:** Should raise error or at least log clear warning about fallback.

---

## Recommended Fixes Priority

1. **HIGH:** Fix hardcoded frequencies in dfm_python.py (Issue #2)
2. **HIGH:** Fix hardcoded target series in sktime.py (Issue #1)
3. **MEDIUM:** Pass metadata/data_loader to training functions (Issue #3)
4. **MEDIUM:** Fix datetime index creation in sktime.py (Issue #4)
5. **LOW:** Standardize target series handling (Issue #6)
6. **LOW:** Pass data_model context (Issue #7)
