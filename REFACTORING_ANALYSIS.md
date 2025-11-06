# DFM Module Refactoring Analysis

## Application-Specific Code to Move from `src/` to `scripts/`

### 1. **data_loader.py** - Database Functions (~357 lines)

**Functions to move:**
- `load_data_from_db()` (lines 680-826, ~146 lines) - Main database loading
- `_get_db_client()` (line 470) - Database client helper
- `_resolve_vintage_id()` (line 496) - Vintage ID resolution
- `_fetch_vintage_data()` (line 621) - Database data fetching
- `_apply_transformations_from_metadata()` (line 523) - Uses DB metadata (but also works with config)

**Dependencies:**
- Imports `database` module (application-specific)
- Uses Supabase client
- Calls `get_vintage_data()`, `get_series_metadata_bulk()`, etc.

**Impact:**
- Currently exported via `__init__.py` as `load_data_from_db`
- Used in: `scripts/train_dfm.py`, `scripts/forecast_dfm.py`
- Referenced in: `src/nowcasting/dfm.py` docstrings

### 2. **news.py** - Database Saving (~150 lines)

**Functions to move:**
- `_save_nowcast_to_db()` (line 321) - Database saving logic
- Call to `_save_nowcast_to_db()` in `update_nowcast()` (line 707)

**Dependencies:**
- Imports `database` module
- Uses `save_forecast()`, `get_client()`, `TABLES`

**Impact:**
- Called from `update_nowcast()` which is core DFM functionality
- Should be made optional via callback pattern

### 3. **config.py** - Minor Reference

- `use_database` field in `DataConfig` - This is fine, just a flag
- No database imports in config.py ✓

## Generic Code to Keep in DFM Module

### data_loader.py (keep these):
- `load_config()`, `load_config_from_csv()`, `load_config_from_yaml()` - Generic file loading
- `load_data()` - Generic file-based data loading (CSV, Excel, MATLAB)
- `transform_data()` - Generic transformations (pure numpy/pandas)
- `_transform_series()`, `sort_data()`, etc. - Pure transformation logic

### Core algorithms (all generic):
- `dfm.py` - Pure DFM estimation algorithm
- `kalman.py` - Pure Kalman filtering/smoothing
- `news.py` - News decomposition (make saving optional)
- `config.py` - Data structures (dataclasses)

## Proposed Solution

### Option 1: Create `scripts/db_adapters.py` (Recommended)

**Move to scripts/db_adapters.py:**
- All database functions from `data_loader.py`
- `_save_nowcast_to_db()` from `news.py`

**Refactor `news.py`:**
- Make `update_nowcast()` accept optional `save_callback` parameter
- If `save_callback` provided, call it instead of `_save_nowcast_to_db()`
- Scripts provide callback that uses `db_adapters.py`

**Refactor `data_loader.py`:**
- Remove all database functions
- Keep only file-based loading

**Update scripts:**
- Import `load_data_from_db` from `scripts.db_adapters` instead of `src.nowcasting`

### Option 2: Keep Thin Interface (Alternative)

**Keep in DFM module:**
- Minimal interface: `load_data_from_db()` that accepts pre-loaded data
- Scripts handle database → DataFrame conversion
- DFM module only works with DataFrames/numpy arrays

**Move to scripts:**
- All database-specific query logic
- Database → DataFrame conversion

## Recommendation

**Option 1 is cleaner** - completely separate database code from DFM module.

**Benefits:**
- DFM module becomes truly generic (no database dependencies)
- Can be used in any project without database module
- Clear separation: core algorithms vs. application adapters
- Easier testing (can test DFM without database)

**Migration Steps:**
1. Create `scripts/db_adapters.py` with all database functions
2. Update `news.py` to use callback pattern for saving
3. Remove database functions from `data_loader.py`
4. Update `__init__.py` to remove `load_data_from_db`
5. Update scripts to import from `db_adapters.py`

