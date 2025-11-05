# Database Integration Proposals for DFM Forecasting Module

## Overview
This document outlines proposals for integrating Supabase database queries into the DFM forecasting workflow. The goal is to replace file-based data loading (`load_data()` from Excel files) with database-backed data loading for both training and forecasting.

## Current State Analysis

### Existing Functions

#### Data Loading (`src/nowcasting/data_loader.py`)
- `load_data(datafile, config, ...)`: Loads data from Excel files, returns (X, Time, Z)
  - X: Transformed data matrix (T x N)
  - Time: pd.DatetimeIndex
  - Z: Raw untransformed data (T x N)
- `load_config(configfile)`: Loads ModelConfig from Excel/YAML files
- `transform_data(Z, Time, config)`: Applies transformations to data

#### Database Operations (`database/operations.py`)
- `get_vintage_data(vintage_id, config_series_ids, client)`: Returns (df, Time) - partial implementation
- `get_observations(...)`: Returns DataFrame with observations
- `list_series(...)`: Lists all series
- `get_series(series_id)`: Gets single series metadata
- `save_model_config(...)`: Saves model configuration
- `load_model_config(config_name)`: Loads model configuration
- `save_forecast(...)`: Saves forecast results

### Usage Points
1. **`scripts/setup/train_dfm.py`**: Uses `load_data()` and `load_config()` from files
2. **`scripts/run_nowcast.py`**: Uses `load_data()` for old and new vintages
3. **`scripts/forecast_dfm.py`**: (To be confirmed - likely similar usage)

---

## Proposals

### Proposal 1: Database-Backed Data Loader Function

**Location**: `src/nowcasting/data_loader.py` or `src/utils/data_utils.py`

**New Function**: `load_data_from_db()`

```python
def load_data_from_db(
    vintage_id: Optional[int] = None,
    vintage_date: Optional[date] = None,
    config: Optional[ModelConfig] = None,
    config_series_ids: Optional[List[str]] = None,
    sample_start: Optional[Union[pd.Timestamp, str]] = None,
    client: Optional[Client] = None
) -> Tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """
    Load vintage data from database and format for DFM.
    
    Parameters
    ----------
    vintage_id : int, optional
        Vintage ID (if None, uses latest vintage)
    vintage_date : date, optional
        Vintage date (alternative to vintage_id)
    config : ModelConfig, optional
        Model configuration (if None, loads from database)
    config_series_ids : List[str], optional
        List of series IDs to include (if None, uses config.SeriesID)
    sample_start : Timestamp or str, optional
        Start date for estimation sample
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    X : np.ndarray
        Transformed data matrix (T x N)
    Time : pd.DatetimeIndex
        Time index for observations
    Z : np.ndarray
        Raw (untransformed) data matrix (T x N)
    """
```

**Implementation Steps**:
1. Get vintage_id (from parameter or latest vintage)
2. Call `get_vintage_data()` to get raw data DataFrame
3. Get series metadata via `list_series()` or `get_series()` for each series_id
4. Map series to config.SeriesID order
5. Apply transformations based on series.transformation field
6. Return (X, Time, Z) matching current `load_data()` signature

**Dependencies**:
- Enhance `get_vintage_data()` to return Z (raw data) in addition to formatted data
- Ensure series metadata includes transformation codes

---

### Proposal 2: Database-Backed Config Loader Function

**Location**: `src/nowcasting/data_loader.py`

**New Function**: `load_config_from_db()`

```python
def load_config_from_db(
    config_name: str,
    client: Optional[Client] = None
) -> ModelConfig:
    """
    Load model configuration from database.
    
    Parameters
    ----------
    config_name : str
        Configuration name stored in database
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    ModelConfig
        Model configuration object
    """
```

**Implementation Steps**:
1. Call `load_model_config(config_name)` to get config JSON
2. Get block assignments from `model_block_assignments` table
3. Get series metadata for all series in config
4. Construct ModelConfig from database data
5. Validate configuration

**Database Schema Requirements**:
- `model_configs` table should store:
  - `config_name`: Unique identifier
  - `config_json`: Full ModelConfig structure
  - `block_names`: List of block names
  - `series_ids`: List of series IDs in order
  - `country`: Country code

---

### Proposal 3: Enhanced `get_vintage_data()` Function

**Location**: `database/operations.py`

**Current State**: Returns `(DataFrame, DatetimeIndex)` but needs to match `load_data()` output format

**Proposed Enhancement**:

```python
def get_vintage_data(
    vintage_id: int,
    config_series_ids: Optional[List[str]] = None,
    return_raw: bool = True,
    client: Optional[Client] = None
) -> Tuple[pd.DataFrame, pd.DatetimeIndex, Optional[pd.DataFrame]]:
    """
    Get all data for a vintage, formatted for DFM input.
    
    Parameters
    ----------
    vintage_id : int
        Vintage ID
    config_series_ids : List[str], optional
        List of series IDs to include (maintains order)
    return_raw : bool, default True
        If True, also returns raw data before transformations
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    tuple
        (Z_raw, Time, Z_transformed) where:
        - Z_raw: Raw data DataFrame (T x N) - columns in config order
        - Time: DatetimeIndex
        - Z_transformed: Transformed data (if return_raw=False, this is primary output)
    """
```

**Implementation Steps**:
1. Get observations via `get_observations(vintage_id=vintage_id)`
2. Pivot to wide format (date x series_id)
3. Sort by date
4. Reorder columns to match config_series_ids order
5. Return as numpy arrays matching `load_data()` format

---

### Proposal 4: Integration into Training Script

**Location**: `scripts/setup/train_dfm.py`

**Changes**:
1. Add database loading option (configurable via Hydra config)
2. Use `load_data_from_db()` when `data.use_database=True`
3. Save trained model to database via `save_model_weights()`
4. Save model config to database via `save_model_config()`

**Proposed Config Addition** (`config/data/`):
```yaml
data:
  use_database: true  # or false for file-based (backward compatibility)
  vintage_id: null    # null = use latest vintage
  vintage_date: null  # alternative to vintage_id
  config_name: "kr_dfm_v1"  # for load_config_from_db()
  # ... existing config options
```

---

### Proposal 5: Integration into Forecasting Scripts

**Location**: `scripts/run_nowcast.py` and `scripts/forecast_dfm.py`

**Changes**:
1. Load old and new vintages from database
2. Use `load_data_from_db()` instead of `load_data()`
3. Save forecasts to database via `save_forecast()`
4. Create forecast_run record for tracking

**Proposed Flow**:
```python
# Load old vintage
X_old, Time_old, Z_old = load_data_from_db(
    vintage_date=vintage_old,
    config=model_cfg
)

# Load new vintage
X_new, Time_new, Z_new = load_data_from_db(
    vintage_date=vintage_new,
    config=model_cfg
)

# Run nowcast
# ... nowcast logic ...

# Save forecasts
for series_id, forecast_value in forecasts.items():
    save_forecast(
        model_id=trained_model_id,
        series_id=series_id,
        forecast_date=target_date,
        forecast_value=forecast_value,
        ...
    )
```

---

### Proposal 6: Series Metadata Mapping

**Location**: `src/utils/data_utils.py` (new function)

**New Function**: `map_series_to_config()`

```python
def map_series_to_config(
    series_ids: List[str],
    config: ModelConfig,
    client: Optional[Client] = None
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Map database series to ModelConfig structure.
    
    Returns
    -------
    tuple
        (series_order, series_metadata_list) where:
        - series_order: Indices for ordering series to match config
        - series_metadata_list: List of series metadata dicts
    """
```

**Purpose**: Ensures series from database are ordered correctly and include transformation metadata needed for `transform_data()`.

---

## Database Schema Enhancements Needed

### 1. Series Table
- Ensure `transformation` field matches ModelConfig transformation codes
- Ensure `frequency` field matches ModelConfig frequency codes
- Add `block_assignment` field if not exists (for block structure)

### 2. Model Configs Table
- Verify `config_json` can store full ModelConfig structure
- Ensure `block_names` and `series_ids` are properly stored

### 3. Observations Table
- Verify date handling and timezone consistency
- Ensure efficient querying by vintage_id and series_id

---

## Implementation Priority

### Phase 1: Core Data Loading (High Priority)
1. Enhance `get_vintage_data()` in `database/operations.py`
2. Create `load_data_from_db()` in `src/nowcasting/data_loader.py`
3. Test with existing vintage data

### Phase 2: Config Loading (High Priority)
1. Create `load_config_from_db()` in `src/nowcasting/data_loader.py`
2. Ensure model configs are properly saved to database
3. Test config loading and validation

### Phase 3: Script Integration (Medium Priority)
1. Update `scripts/setup/train_dfm.py` to use database
2. Update `scripts/run_nowcast.py` to use database
3. Update `scripts/forecast_dfm.py` to use database

### Phase 4: Forecast Saving (Medium Priority)
1. Integrate `save_forecast()` into forecasting scripts
2. Create forecast_run tracking
3. Add forecast retrieval functions

### Phase 5: Backward Compatibility (Low Priority)
1. Maintain file-based loading as fallback
2. Add config option to switch between file/database modes
3. Update documentation

---

## Testing Requirements

1. **Unit Tests**:
   - Test `load_data_from_db()` returns same format as `load_data()`
   - Test `load_config_from_db()` returns valid ModelConfig
   - Test series ordering matches config

2. **Integration Tests**:
   - Test full training workflow with database
   - Test forecasting workflow with database
   - Test vintage comparison (old vs new)

3. **Data Validation**:
   - Verify data matches between file-based and database-based loading
   - Verify transformations are applied correctly
   - Verify date alignment

---

## Open Questions for Database Agent

1. **Vintage Selection**: How should we determine which vintage to use if not specified?
   - Always use latest?
   - Use latest completed vintage?
   - Allow date-based selection?

2. **Series Ordering**: How do we ensure series from database match ModelConfig.SeriesID order?
   - Should we store order in model_configs table?
   - Or rely on series_id matching?

3. **Missing Data Handling**: How should we handle series that exist in config but not in database?
   - Skip with warning?
   - Fill with NaN?
   - Raise error?

4. **Transformation Codes**: Do database series.transformation codes match ModelConfig transformation codes?
   - Mapping needed?
   - Standardization required?

5. **Performance**: For large vintages (many series, long time periods), should we:
   - Cache data in memory?
   - Use batch queries?
   - Implement pagination?

6. **Model Config Storage**: Should model_configs table store:
   - Full JSON structure?
   - Normalized structure?
   - Both for flexibility?

---

## Notes

- All changes should maintain backward compatibility with file-based loading
- Database integration should be opt-in via configuration
- Error handling should gracefully fall back to file-based loading if database fails
- Logging should clearly indicate when database vs file loading is used

