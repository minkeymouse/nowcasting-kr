"""Database operations for all tables."""

import logging
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Union, Callable
from datetime import date, datetime
from supabase import Client

from .client import get_client  # Only used for ensure_client fallback
from .helpers import (
    batch_query_in,
    batch_insert,
    build_query,
    DatabaseError,
    NotFoundError,
    ValidationError,
    ensure_client,
    get_series_ids_for_config,
    get_block_assignments_for_config,
    get_block_names_for_config,
    resolve_config_name,
)

logger = logging.getLogger(__name__)
from .models import (
    SeriesModel,
    VintageModel,
    ObservationModel,
    ForecastModel,
    StatisticsMetadataModel,
    StatisticsItemModel,
    TABLES,
)


# ============================================================================
# Transformation Helpers (for DFM data loading)
# ============================================================================

def _transform_series(z: np.ndarray, formula: str, freq: str, step: int) -> np.ndarray:
    """
    Apply transformation to a single series.
    
    This is a port of the transformation logic from src/nowcasting/data_loader.py
    to work with database-loaded data.
    
    Parameters
    ----------
    z : np.ndarray
        Raw series data (1D array)
    formula : str
        Transformation code: 'lin', 'chg', 'ch1', 'pch', 'pc1', 'pca', 'log'
    freq : str
        Frequency code: 'q', 'm', 'd', etc.
    step : int
        Step size for transformations (3 for quarterly, 1 for monthly/daily)
        
    Returns
    -------
    np.ndarray
        Transformed series
    """
    T = len(z)
    X = np.full(T, np.nan)
    t1 = step
    n = step / 12
    
    if formula == 'lin':
        X[:] = z
    elif formula == 'chg':
        idx = np.arange(t1, T, step)
        if len(idx) > 1:
            X[idx[0]] = np.nan
            X[idx[1:]] = z[idx[1:]] - z[idx[:-1]]
    elif formula == 'ch1':
        idx = np.arange(12 + t1, T, step)
        if len(idx) > 0:
            X[idx] = z[idx] - z[idx - 12]
    elif formula == 'pch':
        idx = np.arange(t1, T, step)
        if len(idx) > 1:
            X[idx[0]] = np.nan
            X[idx[1:]] = 100 * (z[idx[1:]] / z[idx[:-1]] - 1)
    elif formula == 'pc1':
        idx = np.arange(12 + t1, T, step)
        if len(idx) > 0:
            with np.errstate(divide='ignore', invalid='ignore'):
                X[idx] = 100 * (z[idx] / z[idx - 12] - 1)
            X[np.isinf(X)] = np.nan
    elif formula == 'pca':
        idx = np.arange(t1, T, step)
        if len(idx) > 1:
            X[idx[0]] = np.nan
            with np.errstate(divide='ignore', invalid='ignore'):
                X[idx[1:]] = 100 * ((z[idx[1:]] / z[idx[:-1]]) ** (1/n) - 1)
            X[np.isinf(X)] = np.nan
    elif formula == 'log':
        with np.errstate(invalid='ignore'):
            X[:] = np.log(z)
    else:
        # Default to linear if unknown transformation
        X[:] = z
    
    return X


def _apply_transformations(
    Z: pd.DataFrame,
    series_metadata: pd.DataFrame,
    drop_initial: int = 4
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply transformations to data based on series metadata.
    
    Parameters
    ----------
    Z : pd.DataFrame
        Raw data (T x N) with series as columns
    series_metadata : pd.DataFrame
        Metadata with columns: series_id, transformation, frequency
    drop_initial : int
        Number of initial observations to drop (default: 4)
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (X, Z) where X is transformed data and Z is raw data (after dropping)
    """
    X = pd.DataFrame(index=Z.index, columns=Z.columns, dtype=float)
    
    for series_id in Z.columns:
        if series_id not in series_metadata['series_id'].values:
            # If metadata missing, use linear transformation
            X[series_id] = Z[series_id]
            continue
        
        meta = series_metadata[series_metadata['series_id'] == series_id].iloc[0]
        transformation = meta.get('transformation', 'lin')
        frequency = meta.get('frequency', 'm')
        
        # Determine step size
        step = 3 if frequency == 'q' else 1
        
        # Apply transformation
        z_values = Z[series_id].values
        x_values = _transform_series(z_values, transformation, frequency, step)
        X[series_id] = x_values
    
    # Drop initial observations (as per DFM convention)
    if len(X) > drop_initial:
        X = X.iloc[drop_initial:]
        Z = Z.iloc[drop_initial:]
    
    return X, Z




# ============================================================================
# Data Source Operations
# ============================================================================

# ============================================================================
# Series Operations
# ============================================================================

def generate_series_id(
    source_code: str,
    stat_code: str,
    item_code: Optional[str] = None
) -> str:
    """
    Generate consistent series_id.
    
    Format: {SOURCE}_{STAT_CODE} or {SOURCE}_{STAT_CODE}_{ITEM_CODE}
    Examples:
      - BOK_200Y105
      - BOK_200Y105_1101
    """
    if item_code:
        return f"{source_code}_{stat_code}_{item_code}"
    return f"{source_code}_{stat_code}"


def create_or_get_series(
    source_code: str,
    stat_code: str,
    series_name: str,
    frequency: str,
    client: Optional[Client] = None,
    statistics_metadata_id: Optional[int] = None,
    item_code: Optional[str] = None,
    item_name: Optional[str] = None,
    units: Optional[str] = None,
    category: Optional[str] = None,
    api_source: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create or get existing series.
    
    Parameters
    ----------
    source_code : str
        Source code (e.g., 'BOK', 'KOSIS')
    stat_code : str
        Statistic code (e.g., '200Y105')
    series_name : str
        Human-readable series name
    frequency : str
        Frequency code (d, w, m, q, sa, a)
    client : Client, optional
        Supabase client instance
    statistics_metadata_id : int, optional
        Reference to statistics_metadata
    item_code : str, optional
        Item code if this is an item-specific series
    item_name : str, optional
        Item name for display
    units : str, optional
        Units of measurement
    category : str, optional
        Category classification
    api_source : str, optional
        API source identifier (defaults to source_code if not provided)
        
    Returns
    -------
    Dict[str, Any]
        Series record from database
    """
    client = ensure_client(client)
    
    series_id = generate_series_id(source_code, stat_code, item_code)
    
    # Check if exists
    existing = get_series(series_id=series_id, client=client)
    if existing:
        return existing
    
    # Create new series
    # Use source_code as api_source if not provided
    api_source_value = api_source or source_code
    
    series_model = SeriesModel(
        series_id=series_id,
        series_name=series_name,
        frequency=frequency,
        api_source=api_source_value,
        api_code=stat_code,
        units=units,
        category=category
    )
    
    result = upsert_series(
        series=series_model,
        statistics_metadata_id=statistics_metadata_id,
        item_code=item_code,
        client=client
    )
    
    logger.debug(f"Created series: {series_id}")
    return result


def create_series_from_item(
    source_code: str,
    stat_code: str,
    stat_name: str,
    item: Dict[str, Any],
    client: Optional[Client] = None,
    statistics_metadata_id: Optional[int] = None,
    frequency_mapper: Optional[Callable[[str], str]] = None
) -> Dict[str, Any]:
    """
    Create series from statistics item.
    
    Parameters
    ----------
    source_code : str
        Source code (e.g., 'BOK')
    stat_code : str
        Statistic code
    stat_name : str
        Statistic name
    item : Dict[str, Any]
        Item dictionary with item_code, item_name, cycle, unit_name
    client : Client, optional
        Supabase client instance
    statistics_metadata_id : int, optional
        Reference to statistics_metadata
    frequency_mapper : callable, optional
        Function to map cycle to frequency code
        
    Returns
    -------
    Dict[str, Any]
        Series record
    """
    client = ensure_client(client)
    
    from .helpers import map_frequency_to_code
    
    item_code = item.get('item_code')
    item_name = item.get('item_name', '')
    cycle = item.get('cycle', '')
    unit_name = item.get('unit_name')
    
    # Map frequency
    frequency = (frequency_mapper or map_frequency_to_code)(cycle) if cycle else 'm'
    
    # Build series name
    if item_name:
        series_name = f"{stat_name} - {item_name}"
    elif item_code:
        series_name = f"{stat_name} - {item_code}"
    else:
        series_name = stat_name
    
    return create_or_get_series(
        source_code=source_code,
        stat_code=stat_code,
        series_name=series_name,
        frequency=frequency,
        statistics_metadata_id=statistics_metadata_id,
        item_code=item_code,
        item_name=item_name,
        units=unit_name,
        category=stat_name,
        client=client
    )


def get_series(series_id: str, client: Optional[Client] = None) -> Optional[Dict[str, Any]]:
    client = ensure_client(client)
    if not series_id:
        return None
    result = client.table(TABLES['series']).select('*').eq('series_id', series_id).execute()
    return result.data[0] if result.data else None


def upsert_series(
    series: SeriesModel,
    client: Optional[Client] = None,
    statistics_metadata_id: Optional[int] = None,
    item_code: Optional[str] = None
) -> Dict[str, Any]:
    """Insert or update a series."""
    client = ensure_client(client)
    data = series.model_dump(exclude_none=True)
    # Remove any fields that don't exist in the database table
    # (updated_at and created_at are managed by triggers)
    data.pop('updated_at', None)
    data.pop('created_at', None)
    if statistics_metadata_id is not None:
        data['statistics_metadata_id'] = statistics_metadata_id
    if item_code is not None:
        data['item_code'] = item_code
    
    result = client.table(TABLES['series']).upsert(data, on_conflict='series_id').execute()
    return result.data[0] if result.data else None


def list_series(client: Optional[Client] = None, api_source: Optional[str] = None) -> List[Dict[str, Any]]:
    client = ensure_client(client)
    query = client.table(TABLES['series']).select('*')
    if api_source:
        query = query.eq('api_source', api_source)
    result = query.execute()
    return result.data


def get_series_metadata_bulk(
    series_ids: List[str],
    client: Optional[Client] = None
) -> pd.DataFrame:
    """
    Get metadata for multiple series.
    
    Parameters
    ----------
    series_ids : List[str]
        List of series IDs to get metadata for
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: series_id, transformation, frequency, units, category, etc.
    """
    client = ensure_client(client)
    if not series_ids:
        return pd.DataFrame(columns=['series_id', 'transformation', 'frequency', 'units', 'category'])
    
    # Use batch query helper
    all_data = batch_query_in(
        client=client,
        table_name=TABLES['series'],
        column='series_id',
        values=series_ids,
        batch_size=100,
        select='series_id, series_name, transformation, frequency, units, category, api_source, data_code'
    )
    
    if not all_data:
        return pd.DataFrame(columns=['series_id', 'transformation', 'frequency', 'units', 'category'])
    
    df = pd.DataFrame(all_data)
    
    # Ensure required columns exist
    required_cols = ['series_id', 'transformation', 'frequency']
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
    
    return df


# ============================================================================
# Vintage Operations
# ============================================================================

def create_vintage(
    vintage_date: date,
    client: Optional[Client] = None,
    country: str = 'KR',
    github_run_id: Optional[str] = None,
    github_workflow_run_url: Optional[str] = None
) -> Dict[str, Any]:
    """Create a new data vintage."""
    client = ensure_client(client)
    data = {
        'vintage_date': vintage_date.isoformat(),
        'country': country,
        'fetch_status': 'in_progress',
        'fetch_started_at': datetime.now().isoformat(),
        'github_run_id': github_run_id,
        'github_workflow_run_url': github_workflow_run_url,
    }
    result = client.table(TABLES['vintages']).insert(data).execute()
    return result.data[0] if result.data else None


def get_vintage(
    client: Optional[Client] = None,
    vintage_date: Optional[date] = None,
    vintage_id: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Get a vintage by date or ID."""
    client = ensure_client(client)
    if vintage_date:
        result = client.table(TABLES['vintages']).select('*').eq('vintage_date', vintage_date.isoformat()).execute()
        return result.data[0] if result.data else None
    elif vintage_id:
        result = client.table(TABLES['vintages']).select('*').eq('vintage_id', vintage_id).execute()
        return result.data[0] if result.data else None
    return None


def get_latest_vintage(client: Optional[Client] = None) -> Optional[Dict[str, Any]]:
    result = (
        client.table(TABLES['vintages'])
        .select('*')
        .eq('fetch_status', 'completed')
        .order('vintage_date', desc=True)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


def get_latest_vintage_id(
    vintage_date: Optional[date] = None,
    country: str = 'KR',
    status: Optional[str] = 'completed',
    client: Optional[Client] = None
) -> Optional[int]:
    """
    Get latest vintage ID by date and country.
    
    Parameters
    ----------
    vintage_date : date, optional
        Get latest vintage on or before this date. If None, gets latest overall.
    country : str
        Country code (default: 'KR')
    status : str, optional
        Filter by fetch_status (default: 'completed'). If None, any status.
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    int, optional
        Vintage ID if found, None otherwise
    """
    query = (
        client.table(TABLES['vintages'])
        .select('vintage_id')
        .eq('country', country)
    )
    
    if status:
        query = query.eq('fetch_status', status)
    
    if vintage_date:
        query = query.lte('vintage_date', vintage_date.isoformat())
    
    result = (
        query
        .order('vintage_date', desc=True)
        .limit(1)
        .execute()
    )
    
    if result.data:
        return result.data[0]['vintage_id']
    return None


def update_vintage_status(
    vintage_id: int,
    status: str,
    error_message: Optional[str] = None,
    client: Optional[Client] = None
) -> Dict[str, Any]:
    """Update vintage fetch status."""
    
    update_data = {
        'fetch_status': status,
        'fetch_completed_at': datetime.now().isoformat() if status in ('completed', 'failed') else None,
    }
    
    if error_message:
        update_data['error_message'] = error_message
    
    result = (
        client.table(TABLES['vintages'])
        .update(update_data)
        .eq('vintage_id', vintage_id)
        .execute()
    )
    return result.data[0] if result.data else None


# ============================================================================
# Observation Operations (Pandas-Optimized)
# ============================================================================

def insert_observations_from_dataframe(
    df: pd.DataFrame,
    vintage_id: int,
    github_run_id: Optional[str] = None,
    api_source: Optional[str] = None,
    batch_size: int = 1000,
    client: Optional[Client] = None
) -> int:
    """
    Bulk insert observations from pandas DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with observations. Expected columns:
        - series_id: str
        - date: date/pd.Timestamp
        - value: float
        - Optional: item_code1-4, item_name1-4, weight, github_run_id, api_source
    vintage_id : int
        Vintage ID for the observations
    github_run_id : str, optional
        GitHub Actions run ID
    api_source : str, optional
        API source identifier
    batch_size : int
        Batch size for insertion (default: 1000)
    client : Client, optional
        Supabase client instance. If None, uses get_client()
        
    Returns
    -------
    int
        Number of observations inserted
        
    Raises
    ------
    ValueError
        If required columns are missing
    Exception
        If database insertion fails
    """
    client = ensure_client(client)
    
    # Prepare data
    df = df.copy()
    df['vintage_id'] = vintage_id
    if github_run_id:
        df['github_run_id'] = github_run_id
    if api_source:
        df['api_source'] = api_source
    
    # Convert date to string
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.date.astype(str)
    
    # Select columns - required + optional
    required_cols = ['series_id', 'date', 'value', 'vintage_id']  # vintage_id is required
    optional_cols = ['github_run_id', 'api_source', 'weight', 'is_forecast'] + \
                   [f'item_code{i}' for i in range(1, 5)] + \
                   [f'item_name{i}' for i in range(1, 5)]
    
    columns = required_cols + [col for col in optional_cols if col in df.columns]
    
    df_clean = df[columns].copy()
    
    # Remove NaN values
    df_clean = df_clean.dropna(subset=['series_id', 'date', 'value'])
    
    # Convert to records
    records = df_clean.to_dict('records')
    
    # Use batch insert helper
    total_inserted = batch_insert(
        client=client,
        table_name=TABLES['observations'],
        records=records,
        batch_size=batch_size,
        on_conflict='series_id,vintage_id,date'
    )
    
    return total_inserted


def get_observations(
    client: Optional[Client] = None,
    series_id: Optional[str] = None,
    vintage_id: Optional[int] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> pd.DataFrame:
    """Get observations as a pandas DataFrame."""
    
    query = client.table(TABLES['observations']).select('*')
    
    if series_id:
        query = query.eq('series_id', series_id)
    if vintage_id:
        query = query.eq('vintage_id', vintage_id)
    if start_date:
        query = query.gte('date', start_date.isoformat())
    if end_date:
        query = query.lte('date', end_date.isoformat())
    
    result = query.execute()
    
    if not result.data:
        return pd.DataFrame()
    
    df = pd.DataFrame(result.data)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    return df


def get_latest_observation_date(
    series_id: str,
    vintage_id: Optional[int] = None,
    client: Optional[Client] = None
) -> Optional[date]:
    """
    Get the latest observation date for a series.
    
    Parameters
    ----------
    series_id : str
        Series ID
    vintage_id : int, optional
        Vintage ID (if None, checks all vintages)
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    date or None
        Latest observation date, or None if no observations exist
    """
    query = client.table(TABLES['observations']).select('date').eq('series_id', series_id).order('date', desc=True).limit(1)
    
    if vintage_id:
        query = query.eq('vintage_id', vintage_id)
    
    result = query.execute()
    
    if not result.data:
        return None
    
    latest_date = result.data[0]['date']
    if isinstance(latest_date, str):
        return datetime.fromisoformat(latest_date).date()
    return latest_date


def check_series_exists(
    series_id: str,
    client: Optional[Client] = None
) -> bool:
    """
    Check if a series exists in the database.
    
    Parameters
    ----------
    series_id : str
        Series ID to check
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    bool
        True if series exists, False otherwise
    """
    series = get_series(series_id, client=client)
    return series is not None


def get_vintage_data(
    vintage_id: int,
    config_series_ids: Optional[List[str]] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    strict_mode: bool = False,
    ensure_series_order: bool = False,
    client: Optional[Client] = None
) -> Tuple[pd.DataFrame, pd.DatetimeIndex, pd.DataFrame, pd.DataFrame]:
    """
    Get all data for a vintage, formatted for DFM input with transformations.
    
    Parameters
    ----------
    vintage_id : int
        Vintage ID
    config_series_ids : List[str], optional
        List of series IDs to include (if None, gets all series in vintage)
    start_date : date, optional
        Start date filter
    end_date : date, optional
        End date filter
    strict_mode : bool
        If True, raise error for missing series. If False, fill with NaN and warn.
    ensure_series_order : bool
        If True, ensure series order matches config_series_ids order
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DatetimeIndex, pd.DataFrame, pd.DataFrame]
        (X, Time, Z, series_metadata) where:
        - X: Transformed data (T x N) DataFrame
        - Time: DatetimeIndex
        - Z: Raw untransformed data (T x N) DataFrame
        - series_metadata: DataFrame with series metadata
    """
    
    # Get observations
    df = get_observations(
        client=client,
        vintage_id=vintage_id,
        start_date=start_date,
        end_date=end_date
    )
    
    if df.empty:
        empty_df = pd.DataFrame()
        empty_time = pd.DatetimeIndex([])
        empty_metadata = pd.DataFrame(columns=['series_id', 'transformation', 'frequency'])
        return empty_df, empty_time, empty_df, empty_metadata
    
    # Determine which series to fetch
    available_series = df['series_id'].unique().tolist()
    
    if config_series_ids:
        requested_series = config_series_ids
        missing_series = set(requested_series) - set(available_series)
        
        if missing_series:
            if strict_mode:
                raise ValueError(
                    f"Missing series in vintage {vintage_id}: {missing_series}"
                )
            else:
                logger.warning(
                    f"Series not found in vintage {vintage_id}: {missing_series}. "
                    "Filling with NaN."
                )
        
        # Filter to requested series (may include missing ones)
        df = df[df['series_id'].isin(requested_series)]
    else:
        requested_series = available_series
    
    # Get series metadata
    series_metadata = get_series_metadata_bulk(requested_series, client=client)
    
    # Ensure all requested series have metadata entries (fill missing with defaults)
    if config_series_ids:
        missing_metadata = set(requested_series) - set(series_metadata['series_id'].values)
        if missing_metadata:
            default_metadata = pd.DataFrame({
                'series_id': list(missing_metadata),
                'transformation': 'lin',
                'frequency': 'm',
                'units': None,
                'category': None,
                'series_name': None,
                'api_source': None,
                'api_code': None
            })
            series_metadata = pd.concat([series_metadata, default_metadata], ignore_index=True)
            
            if strict_mode:
                logger.warning(f"Series missing metadata, defaulting to 'lin': {missing_metadata}")
    
    # Pivot to wide format: rows = dates, columns = series
    df_pivot = df.pivot(index='date', columns='series_id', values='value')
    
    # Ensure all requested series are present as columns (fill missing with NaN)
    if config_series_ids:
        for series_id in requested_series:
            if series_id not in df_pivot.columns:
                df_pivot[series_id] = np.nan
    
    # Order columns if requested
    if ensure_series_order and config_series_ids:
        # Order columns to match config_series_ids
        existing_cols = [s for s in config_series_ids if s in df_pivot.columns]
        missing_cols = [s for s in config_series_ids if s not in df_pivot.columns]
        df_pivot = df_pivot[existing_cols + missing_cols]
    
    # Sort by date
    df_pivot = df_pivot.sort_index()
    
    # Get Time index
    Time = df_pivot.index
    
    # Get raw data (Z) - before transformations
    Z_raw = df_pivot.copy()
    
    # Apply transformations to get X
    X, Z = _apply_transformations(Z_raw, series_metadata)
    
    # Update Time index after dropping initial observations
    # _apply_transformations drops initial rows, so Time needs to match
    if len(Time) > len(X):
        Time = Time[len(Time) - len(X):]
    
    return X, Time, Z, series_metadata


def get_model_config_series_ids(
    config_name: Optional[str] = None,
    config_id: Optional[int] = None,
    client: Optional[Client] = None
) -> List[str]:
    """
    Get ordered list of series IDs for a model configuration.
    
    Orders by blocks.series_order to maintain DFM block structure.
    
    Parameters
    ----------
    config_name : str, optional
        Model configuration name (e.g., '001-initial-spec')
    config_id : int, optional
        Model configuration ID (deprecated, use config_name instead)
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    List[str]
        Ordered list of series IDs
    """
    client = ensure_client(client)
    
    # Resolve config_name if only config_id provided
    if not config_name:
        if config_id is not None:
            config_name = resolve_config_name(config_id=config_id, client=client)
            if not config_name:
                logger.warning(f"Could not resolve config_name from config_id {config_id}")
                return []
        else:
            raise ValueError("Either config_name or config_id must be provided")
    
    return get_series_ids_for_config(config_name, client=client)


def get_vintage_data_for_config(
    config_name: str,
    vintage_id: int,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    strict_mode: bool = False,
    client: Optional[Client] = None
) -> Tuple[pd.DataFrame, pd.DatetimeIndex, pd.DataFrame, pd.DataFrame]:
    """
    Get vintage data ordered by model configuration block assignments.
    
    This function uses blocks.series_order to order series,
    ensuring proper block structure for DFM.
    
    Parameters
    ----------
    config_name : str
        Model configuration name (e.g., '001-initial-spec')
    vintage_id : int
        Vintage ID
    start_date : date, optional
        Start date filter
    end_date : date, optional
        End date filter
    strict_mode : bool
        If True, raise error for missing series. If False, fill with NaN and warn.
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DatetimeIndex, pd.DataFrame, pd.DataFrame]
        (X, Time, Z, series_metadata) ordered by series_order
    """
    client = ensure_client(client)
    
    # Get ordered series IDs from blocks table
    config_series_ids = get_model_config_series_ids(config_name, client=client)
    
    if not config_series_ids:
        raise ValueError(f"No series assigned to config_name {config_name}")
    
    # Get data with ordering
    X, Time, Z, series_metadata = get_vintage_data(
        vintage_id=vintage_id,
        config_series_ids=config_series_ids,
        start_date=start_date,
        end_date=end_date,
        strict_mode=strict_mode,
        ensure_series_order=True,
        client=client
    )
    
    return X, Time, Z, series_metadata


# ============================================================================
# Model Operations
# ============================================================================

def save_model_config(
    config_name: str,
    config_json: Dict[str, Any],
    block_names: List[str],
    description: Optional[str] = None,
    country: str = 'KR',
    client: Optional[Client] = None
) -> Dict[str, Any]:
    """Save a model configuration.
    
    Note: model_configs table may not exist in all database schemas.
    If the table doesn't exist, this function will log a warning and return None.
    """
    client = ensure_client(client)
    
    data = {
        'config_name': config_name,
        'config_json': config_json,
        'block_names': block_names,
        'description': description,
        'country': country,
    }
    
    try:
        result = client.table('model_configs').upsert(data, on_conflict='config_name').execute()
        return result.data[0] if result.data else None
    except Exception as e:
        error_str = str(e).lower()
        if 'not found' in error_str or 'pgrst205' in error_str.lower():
            logger.warning(f"model_configs table not found in database. Skipping config save. Error: {e}")
            logger.info("Model configuration will be stored in blocks table only.")
            return None
        else:
            raise


def load_model_config(
    config_name: Optional[str] = None,
    config_id: Optional[int] = None,
    client: Optional[Client] = None
) -> Optional[Dict[str, Any]]:
    """Load model configuration by config_name or config_id."""
    client = ensure_client(client)
    
    if config_name:
        result = client.table(TABLES['model_configs']).select('*').eq('config_name', config_name).execute()
    elif config_id:
        result = client.table(TABLES['model_configs']).select('*').eq('config_id', config_id).execute()
    else:
        raise ValueError("Either config_name or config_id must be provided")
    
    return result.data[0] if result.data else None


def save_model_weights(
    config_id: int,
    vintage_id: int,
    parameters: Dict[str, Any],
    threshold: Optional[float] = None,
    convergence_iter: Optional[int] = None,
    log_likelihood: Optional[float] = None,
    client: Optional[Client] = None
) -> Dict[str, Any]:
    """
    Save trained model weights (DFMResult).
    
    DEPRECATED: Model weights are now stored in Supabase storage (model-weights bucket).
    This function is kept for backward compatibility but will raise an error if called.
    Use adapters.adapter_database.upload_model_weights_to_storage() instead.
    """
    logger.warning(
        "save_model_weights() is deprecated. Model weights are stored in Supabase storage. "
        "Use adapters.adapter_database.upload_model_weights_to_storage() instead."
    )
    raise NotImplementedError(
        "trained_models table does not exist. Model weights are stored in Supabase storage. "
        "Use adapters.adapter_database.upload_model_weights_to_storage() instead."
    )


def load_model_weights(model_id: int, client: Optional[Client] = None) -> Optional[Dict[str, Any]]:
    """
    Load trained model weights.
    
    DEPRECATED: Model weights are now stored in Supabase storage (model-weights bucket).
    This function is kept for backward compatibility but will raise an error if called.
    Use adapters.adapter_database.download_model_weights_from_storage() instead.
    """
    logger.warning(
        "load_model_weights() is deprecated. Model weights are stored in Supabase storage. "
        "Use adapters.adapter_database.download_model_weights_from_storage() instead."
    )
    raise NotImplementedError(
        "trained_models table does not exist. Model weights are stored in Supabase storage. "
        "Use adapters.adapter_database.download_model_weights_from_storage() instead."
    )


# ============================================================================
# Cleanup Operations
# ============================================================================

def delete_old_vintages(
    months: int = 6,
    dry_run: bool = False,
    client: Optional[Client] = None
) -> Dict[str, Any]:
    """
    Delete vintages older than specified months.
    
    This function deletes old vintages and their associated observations
    (via CASCADE). Use with caution as this is irreversible.
    
    Parameters
    ----------
    months : int, default=6
        Number of months to keep (vintages older than this will be deleted)
    dry_run : bool, default=False
        If True, only count and return without deleting
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    Dict[str, Any]
        Summary of deletion operation:
        - deleted_count: Number of vintages deleted
        - cutoff_date: Date cutoff used
        - dry_run: Whether this was a dry run
    """
    client = ensure_client(client)
    
    from datetime import date, timedelta
    
    # Calculate cutoff date (approximate: 30 days per month)
    cutoff_date = date.today() - timedelta(days=months * 30)
    
    # Find old vintages
    result = client.table(TABLES['vintages']).select(
        'vintage_id,vintage_date'
    ).lt('vintage_date', cutoff_date.isoformat()).execute()
    
    old_vintages = result.data
    count = len(old_vintages)
    
    if dry_run:
        logger.info(f"DRY RUN: Would delete {count} vintages older than {cutoff_date}")
        return {
            'deleted_count': 0,
            'would_delete_count': count,
            'cutoff_date': cutoff_date.isoformat(),
            'dry_run': True,
            'old_vintages': old_vintages
        }
    
    if count == 0:
        logger.info(f"No vintages older than {cutoff_date} to delete")
        return {
            'deleted_count': 0,
            'cutoff_date': cutoff_date.isoformat(),
            'dry_run': False
        }
    
    # Delete old vintages (CASCADE will delete observations)
    vintage_ids = [v['vintage_id'] for v in old_vintages]
    
    for vintage_id in vintage_ids:
        try:
            client.table(TABLES['vintages']).delete().eq('vintage_id', vintage_id).execute()
            logger.debug(f"Deleted vintage {vintage_id}")
        except Exception as e:
            logger.error(f"Error deleting vintage {vintage_id}: {e}")
    
    logger.info(f"Deleted {count} vintages older than {cutoff_date}")
    
    return {
        'deleted_count': count,
        'cutoff_date': cutoff_date.isoformat(),
        'dry_run': False,
        'deleted_vintage_ids': vintage_ids
    }


# ============================================================================
# Forecast Operations
# ============================================================================

def save_forecast(
    model_id: int,
    series_id: str,
    forecast_date: date,
    forecast_value: float,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    confidence_level: float = 0.95,
    run_type: Optional[str] = None,
    vintage_id_old: Optional[int] = None,
    vintage_id_new: Optional[int] = None,
    github_run_id: Optional[str] = None,
    metadata_json: Optional[Dict[str, Any]] = None,
    client: Optional[Client] = None
) -> Dict[str, Any]:
    """Save a forecast to forecasts table.
    
    Parameters
    ----------
    model_id : int
        Model ID (models stored as pkl files, not in DB)
    series_id : str
        Series ID for the forecast
    forecast_date : date
        Target date for forecast
    forecast_value : float
        Forecast value
    lower_bound : float, optional
        Lower confidence bound
    upper_bound : float, optional
        Upper confidence bound
    confidence_level : float, default=0.95
        Confidence level
    run_type : str, optional
        Type of forecast run ('nowcast', 'forecast', 'batch')
    vintage_id_old : int, optional
        Old vintage ID (for nowcasting)
    vintage_id_new : int, optional
        New vintage ID (for nowcasting)
    github_run_id : str, optional
        GitHub Actions run ID
    metadata_json : dict, optional
        Additional metadata as JSON
    client : Client, optional
        Database client
    """
    data = {
        'model_id': model_id,
        'series_id': series_id,
        'forecast_date': forecast_date.isoformat(),
        'forecast_value': forecast_value,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'confidence_level': confidence_level,
    }
    
    # Add optional fields if provided
    if run_type is not None:
        data['run_type'] = run_type
    if vintage_id_old is not None:
        data['vintage_id_old'] = vintage_id_old
    if vintage_id_new is not None:
        data['vintage_id_new'] = vintage_id_new
    if github_run_id is not None:
        data['github_run_id'] = github_run_id
    if metadata_json is not None:
        data['metadata_json'] = metadata_json
    
    result = client.table(TABLES['forecasts']).insert(data).execute()
    return result.data[0] if result.data else None


def get_forecast(
    client: Optional[Client] = None,
    model_id: Optional[int] = None,
    series_id: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> pd.DataFrame:
    """Get forecasts as a pandas DataFrame."""
    
    query = client.table(TABLES['forecasts']).select('*')
    
    if model_id:
        query = query.eq('model_id', model_id)
    if series_id:
        query = query.eq('series_id', series_id)
    if start_date:
        query = query.gte('forecast_date', start_date.isoformat())
    if end_date:
        query = query.lte('forecast_date', end_date.isoformat())
    
    result = query.execute()
    
    if not result.data:
        return pd.DataFrame()
    
    df = pd.DataFrame(result.data)
    if 'forecast_date' in df.columns:
        df['forecast_date'] = pd.to_datetime(df['forecast_date'])
    
    return df


def ensure_vintage_and_job(
    vintage_date: Optional[date] = None,
    client: Optional[Client] = None,
    dry_run: bool = False,
    github_run_id: Optional[str] = None
) -> Optional[int]:
    """
    Ensure a vintage exists for the given date.
    
    This is a convenience function that creates or retrieves a vintage.
    Ingestion job tracking is now integrated into data_vintages table.
    Used by scripts that need to set up ingestion tracking without using the full orchestrator.
    
    Parameters
    ----------
    vintage_date : date, optional
        Vintage date (default: today)
    client : Client, optional
        Supabase client (default: get_client())
    dry_run : bool
        If True, don't create (returns None)
    github_run_id : str, optional
        GitHub Actions run ID (defaults to env var or manual run ID)
        
    Returns
    -------
    Optional[int]
        Vintage ID, or None if dry_run
    """
    if dry_run:
        return None
    
    client = ensure_client(client)
    
    if vintage_date is None:
        vintage_date = date.today()
    
    # Use provided run_id or default
    import os
    run_id = github_run_id or os.getenv('GITHUB_RUN_ID', f'manual-{vintage_date.isoformat()}')
    
    # Create or get vintage
    vintage_id = None
    try:
        vintage_result = create_vintage(
            vintage_date=vintage_date,
            country='KR',
            github_run_id=run_id,
            client=client
        )
        vintage_id = vintage_result['vintage_id']
        logger.info(f"Created vintage {vintage_id} for {vintage_date}")
    except Exception as e:
        # Vintage might already exist
        error_str = str(e).lower()
        if 'already exists' in error_str or 'duplicate' in error_str or '23505' in error_str:
            logger.info(f"Vintage for {vintage_date} already exists, retrieving...")
            # Try to get existing vintage directly from DB
            try:
                result = client.table(TABLES['vintages']).select('vintage_id').eq('vintage_date', vintage_date.isoformat()).eq('country', 'KR').execute()
                if result.data and len(result.data) > 0:
                    vintage_id = result.data[0]['vintage_id']
                    logger.info(f"Retrieved existing vintage {vintage_id}")
                    # Update github_run_id if provided
                    if run_id:
                        try:
                            client.table(TABLES['vintages']).update({
                                'github_run_id': run_id
                            }).eq('vintage_id', vintage_id).execute()
                        except Exception as update_err:
                            logger.warning(f"Could not update github_run_id for vintage {vintage_id}: {update_err}")
                else:
                    # Fallback to get_latest_vintage_id
                    vintage_id = get_latest_vintage_id(vintage_date=vintage_date, client=client)
                    if vintage_id:
                        logger.info(f"Retrieved existing vintage {vintage_id} via get_latest_vintage_id")
            except Exception as retrieve_err:
                logger.warning(f"Failed to retrieve existing vintage: {retrieve_err}")
                # Fallback to get_latest_vintage_id
                vintage_id = get_latest_vintage_id(vintage_date=vintage_date, client=client)
                if vintage_id:
                    logger.info(f"Retrieved existing vintage {vintage_id} via get_latest_vintage_id")
        else:
            logger.error(f"Failed to create vintage: {e}")
            raise
    
    return vintage_id


def finalize_ingestion_job(
    vintage_id: Optional[int],
    status: str = 'completed',
    successful_series: Optional[int] = None,
    failed_series: Optional[int] = None,
    total_series: Optional[int] = None,
    error_message: Optional[str] = None,
    client: Optional[Client] = None
) -> None:
    """
    Finalize an ingestion by updating vintage status and statistics.
    
    This is a convenience function for scripts to update vintage status.
    Ingestion job tracking is now integrated into data_vintages table.
    
    Parameters
    ----------
    vintage_id : int, optional
        Vintage ID to finalize
    status : str
        Final status of the vintage (e.g., 'completed', 'failed'). Defaults to 'completed'.
    successful_series : int, optional
        Number of series successfully processed.
    failed_series : int, optional
        Number of series that failed to process.
    total_series : int, optional
        Total number of series attempted.
    error_message : str, optional
        Error message if status is 'failed'.
    client : Client, optional
        Supabase client (default: get_client())
    """
    if vintage_id is None:
        return
    
    client = ensure_client(client)
    
    try:
        update_data = {
            'fetch_status': status,
            'fetch_completed_at': datetime.now().isoformat(),
        }
        
        if successful_series is not None:
            update_data['successful_series'] = successful_series
        if failed_series is not None:
            update_data['failed_series'] = failed_series
        if total_series is not None:
            update_data['total_series'] = total_series
        if error_message is not None:
            update_data['error_message'] = error_message
        
        client.table(TABLES['vintages']).update(update_data).eq('vintage_id', vintage_id).execute()
        logger.info(f"Updated vintage {vintage_id} to {status}")
    except Exception as e:
        logger.warning(f"Could not update vintage {vintage_id}: {e}")


def get_latest_forecasts(client: Optional[Client] = None, limit: int = 100) -> pd.DataFrame:
    
    result = (
        client.table('latest_forecasts_view')
        .select('*')
        .limit(limit)
        .execute()
    )
    
    if not result.data:
        return pd.DataFrame()
    
    df = pd.DataFrame(result.data)
    if 'forecast_date' in df.columns:
        df['forecast_date'] = pd.to_datetime(df['forecast_date'])
    
    return df



# ============================================================================
# Statistics Metadata Operations
# ============================================================================

def upsert_statistics_metadata(
    metadata: StatisticsMetadataModel,
    client: Optional[Client] = None
) -> Dict[str, Any]:
    """Insert or update statistics metadata."""
    data = metadata.model_dump(exclude_none=True, exclude={'id'})
    # Convert date/datetime to ISO format strings
    for field in ['last_data_fetch_date', 'data_start_date', 'data_end_date', 
                  'last_observation_date', 'vintage_date', 'forecast_date']:
        if field in data and data[field] and isinstance(data[field], date):
            data[field] = data[field].isoformat()
    for field in ['dfm_selected_at', 'created_at', 'updated_at']:
        if field in data and data[field] and isinstance(data[field], datetime):
            data[field] = data[field].isoformat()
    
    result = (
        client.table(TABLES['statistics_metadata'])
        .upsert(data, on_conflict='source_id,source_stat_code')
        .execute()
    )
    return result.data[0] if result.data else None


def get_statistics_metadata(
    client: Optional[Client] = None,
    source_id: Optional[int] = None,
    source_stat_code: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Get statistics metadata by source_id and source_stat_code."""
    query = client.table(TABLES['statistics_metadata']).select('*')
    
    if source_id and source_stat_code:
        query = query.eq('source_id', source_id).eq('source_stat_code', source_stat_code)
        result = query.execute()
        return result.data[0] if result.data else None
    elif source_id:
        query = query.eq('source_id', source_id)
    
    result = query.execute()
    return result.data if isinstance(result.data, list) else result.data



def get_statistics_metadata_bulk(
    source_stat_codes: List[str],
    source_id: int,
    client: Optional[Client] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Get statistics metadata for multiple statistics codes in a single query.
    
    This is more efficient than calling get_statistics_metadata() multiple times.
    
    Parameters
    ----------
    source_stat_codes : List[str]
        List of source statistic codes to fetch
    source_id : int
        Source ID
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping source_stat_code to metadata dict
    """
    client = ensure_client(client)
    
    if not source_stat_codes:
        return {}
    
    # Use batch query helper
    from .helpers import batch_query_in
    
    all_data = batch_query_in(
        client=client,
        table_name=TABLES['statistics_metadata'],
        column='source_stat_code',
        values=source_stat_codes,
        batch_size=100,
        select='*',
        additional_filters={'source_id': source_id}
    )
    
    # Build result dictionary
    result = {}
    for item in all_data:
        stat_code = item.get('source_stat_code')
        if stat_code:
            result[stat_code] = item
    
    return result


def list_dfm_selected_statistics(
    client: Optional[Client] = None,
    source_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """List all DFM-selected statistics."""
    query = (
        client.table(TABLES['statistics_metadata'])
        .select('*')
        .eq('is_dfm_selected', True)
        .order('dfm_priority')
    )
    if source_id:
        query = query.eq('source_id', source_id)
    result = query.execute()
    return result.data


def update_statistics_metadata_status(
    source_id: int,
    source_stat_code: str,
    client: Optional[Client] = None,
    last_data_fetch_date: Optional[date] = None,
    last_data_fetch_status: Optional[str] = None,
    data_start_date: Optional[date] = None,
    data_end_date: Optional[date] = None,
    total_observations: Optional[int] = None,
    last_observation_date: Optional[date] = None
) -> Optional[Dict[str, Any]]:
    """Update statistics metadata fetch status and dates."""
    update_data = {
        'updated_at': datetime.now().isoformat()
    }
    
    if last_data_fetch_date:
        update_data['last_data_fetch_date'] = last_data_fetch_date.isoformat()
    if last_data_fetch_status:
        update_data['last_data_fetch_status'] = last_data_fetch_status
    if data_start_date:
        update_data['data_start_date'] = data_start_date.isoformat()
    if data_end_date:
        update_data['data_end_date'] = data_end_date.isoformat()
    if total_observations is not None:
        update_data['total_observations'] = total_observations
    if last_observation_date:
        update_data['last_observation_date'] = last_observation_date.isoformat()
    
    if len(update_data) == 1:  # Only updated_at
        return None
    
    result = (
        client.table(TABLES['statistics_metadata'])
        .update(update_data)
        .eq('source_id', source_id)
        .eq('source_stat_code', source_stat_code)
        .execute()
    )
    return result.data[0] if result.data else None


# ============================================================================
# Statistics Items Operations
# ============================================================================

def upsert_statistics_items(
    items: List[StatisticsItemModel],
    client: Optional[Client] = None
) -> List[Dict[str, Any]]:
    """Batch insert or update statistics items."""
    if not items:
        return []
    
    records = [item.model_dump(exclude_none=True, exclude={'id'}) for item in items]
    result = (
        client.table(TABLES['statistics_items'])
        .upsert(records, on_conflict='statistics_metadata_id,item_code,cycle')
        .execute()
    )
    return result.data if result.data else []


def get_statistics_items(
    statistics_metadata_id: int,
    client: Optional[Client] = None,
    cycle: Optional[str] = None,
    is_active: Optional[bool] = None
) -> List[Dict[str, Any]]:
    """Get statistics items for a given statistic."""
    query = (
        client.table(TABLES['statistics_items'])
        .select('*')
        .eq('statistics_metadata_id', statistics_metadata_id)
    )
    if cycle:
        query = query.eq('cycle', cycle)
    if is_active is not None:
        query = query.eq('is_active', is_active)
    result = query.execute()
    return result.data if result.data else []


def get_active_items_for_statistic(
    statistics_metadata_id: int,
    client: Optional[Client] = None,
    cycle: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get active items for data collection."""
    return get_statistics_items(statistics_metadata_id, client, cycle, is_active=True)


# ============================================================================
# DFM Data Loading Wrapper (load_data_from_db)
# ============================================================================

def load_data_from_db(
    vintage_id: Optional[int] = None,
    vintage_date: Optional[date] = None,
    config_series_ids: Optional[List[str]] = None,
    config_name: Optional[str] = None,
    config_id: Optional[int] = None,  # Deprecated: use config_name instead
    sample_start: Optional[Union[date, str, pd.Timestamp]] = None,
    strict_mode: bool = False,
    client: Optional[Client] = None
) -> Tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """
    Load data from database for DFM, matching load_data() signature from Excel.
    
    This is the main entry point for loading data from database instead of Excel files.
    It returns data in the same format as load_data() for compatibility.
    
    Parameters
    ----------
    vintage_id : int, optional
        Vintage ID (if None, uses latest vintage or vintage_date)
    vintage_date : date, optional
        Vintage date (alternative to vintage_id)
    config_series_ids : List[str], optional
        List of series IDs to include (if None and config_name provided, uses config)
    config_name : str, optional
        Model configuration name (e.g., '001-initial-spec') - uses blocks table for ordering
    config_id : int, optional, deprecated
        Model configuration ID (deprecated: use config_name instead)
    sample_start : date, str, or Timestamp, optional
        Start date for estimation sample
    strict_mode : bool
        If True, raise error for missing series. If False, fill with NaN and warn.
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    Tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]
        (X, Time, Z) where:
        - X: Transformed data (T x N) numpy array
        - Time: DatetimeIndex
        - Z: Raw untransformed data (T x N) numpy array
    """
    client = ensure_client(client)
    
    # Resolve vintage_id
    if vintage_id is None:
        if vintage_date:
            vintage_id = get_latest_vintage_id(vintage_date=vintage_date, client=client)
        else:
            vintage_id = get_latest_vintage_id(client=client)
        
        if vintage_id is None:
            raise ValueError("No vintage found. Create a vintage first or specify vintage_id/vintage_date.")
    
    # Convert sample_start to date if needed
    start_date = None
    if sample_start:
        if isinstance(sample_start, str):
            start_date = pd.to_datetime(sample_start).date()
        elif isinstance(sample_start, pd.Timestamp):
            start_date = sample_start.date()
        elif isinstance(sample_start, date):
            start_date = sample_start
    
    # Get data
    # Resolve config_name if only config_id provided
    if config_id and not config_name:
        logger.warning("config_id is deprecated. Use config_name instead.")
        config_name = resolve_config_name(config_id=config_id, client=client)
        if not config_name:
            raise ValueError(f"Could not resolve config_name from config_id {config_id}")
    
    if config_name:
        # Use block-ordered data from blocks table
        X, Time, Z, _ = get_vintage_data_for_config(
            config_name=config_name,
            vintage_id=vintage_id,
            start_date=start_date,
            strict_mode=strict_mode,
            client=client
        )
    else:
        # Use config_series_ids ordering
        X, Time, Z, _ = get_vintage_data(
            vintage_id=vintage_id,
            config_series_ids=config_series_ids,
            start_date=start_date,
            strict_mode=strict_mode,
            ensure_series_order=bool(config_series_ids),
            client=client
        )
    
    # Convert to numpy arrays (matching load_data() return format)
    X_array = X.values if isinstance(X, pd.DataFrame) else X
    Z_array = Z.values if isinstance(Z, pd.DataFrame) else Z
    
    return X_array, Time, Z_array
