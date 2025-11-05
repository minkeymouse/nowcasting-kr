"""Database operations for all tables."""

import logging
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Callable
from datetime import date, datetime
from functools import wraps
from supabase import Client

from .client import get_client

logger = logging.getLogger(__name__)
from .models import (
    SeriesModel,
    VintageModel,
    IngestionJobModel,
    ObservationModel,
    ForecastModel,
    StatisticsMetadataModel,
    StatisticsItemModel,
    TABLES,
)

# ============================================================================
# Constants
# ============================================================================

# Series metadata columns for DataFrame construction
SERIES_METADATA_COLUMNS = [
    'series_id', 'series_name', 'transformation', 'frequency', 
    'units', 'category', 'api_source', 'api_code'
]

# Series metadata fields for database queries
SERIES_METADATA_FIELDS = 'series_id, series_name, frequency, transformation, units, category, api_source, api_code'


# ============================================================================
# Helper Functions
# ============================================================================

def _ensure_client(func: Callable) -> Callable:
    """Decorator to ensure client is provided."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if client is None in kwargs or if client is a positional arg
        if 'client' not in kwargs or kwargs['client'] is None:
            kwargs['client'] = get_client()
        return func(*args, **kwargs)
    return wrapper




# ============================================================================
# Data Source Operations
# ============================================================================

@_ensure_client
def get_source_id(source_code: str, client: Optional[Client] = None) -> int:
    """
    Get source_id from source_code.
    
    Parameters
    ----------
    source_code : str
        Source code (e.g., 'BOK', 'KOSIS')
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    int
        Source ID
        
    Raises
    ------
    ValueError
        If source_code not found
    """
    result = client.table('data_sources').select('id').eq('source_code', source_code).execute()
    if result.data:
        return result.data[0]['id']
    raise ValueError(f"Data source '{source_code}' not found in database")


@_ensure_client
def get_source_code(source_id: int, client: Optional[Client] = None) -> str:
    """
    Get source_code from source_id.
    
    Parameters
    ----------
    source_id : int
        Source ID
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    str
        Source code (e.g., 'BOK', 'KOSIS')
        
    Raises
    ------
    ValueError
        If source_id not found
    """
    result = client.table('data_sources').select('source_code').eq('id', source_id).execute()
    if result.data:
        return result.data[0]['source_code']
    raise ValueError(f"Data source with ID {source_id} not found in database")


# ============================================================================
# Series Operations
# ============================================================================

@_ensure_client
def get_series(series_id: str, client: Optional[Client] = None) -> Optional[Dict[str, Any]]:
    """Get a single series by ID."""
    if not series_id:
        return None
    result = client.table(TABLES['series']).select('*').eq('series_id', series_id).execute()
    return result.data[0] if result.data else None


@_ensure_client
def upsert_series(
    series: SeriesModel,
    client: Optional[Client] = None,
    statistics_metadata_id: Optional[int] = None,
    item_code: Optional[str] = None
) -> Dict[str, Any]:
    """Insert or update a series."""
    data = series.model_dump(exclude_none=True)
    if statistics_metadata_id is not None:
        data['statistics_metadata_id'] = statistics_metadata_id
    if item_code is not None:
        data['item_code'] = item_code
    
    result = client.table(TABLES['series']).upsert(data, on_conflict='series_id').execute()
    return result.data[0] if result.data else None


@_ensure_client
def list_series(client: Optional[Client] = None, api_source: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all series, optionally filtered by API source."""
    query = client.table(TABLES['series']).select('*')
    if api_source:
        query = query.eq('api_source', api_source)
    result = query.execute()
    return result.data


# ============================================================================
# Vintage Operations
# ============================================================================

@_ensure_client
def create_vintage(
    vintage_date: date,
    client: Optional[Client] = None,
    country: str = 'KR',
    github_run_id: Optional[str] = None,
    github_workflow_run_url: Optional[str] = None
) -> Dict[str, Any]:
    """Create a new data vintage."""
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


@_ensure_client
def get_vintage(
    client: Optional[Client] = None,
    vintage_date: Optional[date] = None,
    vintage_id: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Get a vintage by date or ID."""
    if vintage_date:
        result = client.table(TABLES['vintages']).select('*').eq('vintage_date', vintage_date.isoformat()).execute()
        return result.data[0] if result.data else None
    elif vintage_id:
        result = client.table(TABLES['vintages']).select('*').eq('vintage_id', vintage_id).execute()
        return result.data[0] if result.data else None
    return None


@_ensure_client
def get_latest_vintage(client: Optional[Client] = None) -> Optional[Dict[str, Any]]:
    """Get the latest completed vintage."""
    result = (
        client.table(TABLES['vintages'])
        .select('*')
        .eq('fetch_status', 'completed')
        .order('vintage_date', desc=True)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


@_ensure_client
def get_latest_vintage_id(
    vintage_date: Optional[date] = None,
    status: Optional[str] = 'completed',
    country: str = 'KR',
    client: Optional[Client] = None
) -> Optional[int]:
    """
    Get latest vintage ID.
    
    Parameters
    ----------
    vintage_date : date, optional
        If provided, get vintage for this date (or latest before this date)
        If None, get absolute latest vintage
    status : str, optional
        Filter by fetch_status ('completed', 'in_progress', etc.)
        If None, return regardless of status
    country : str, default 'KR'
        Country code
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    int or None
        Vintage ID, or None if no vintage found
    """
    query = (
        client.table(TABLES['vintages'])
        .select('vintage_id')
        .eq('country', country)
        .order('vintage_date', desc=True)
        .limit(1)
    )
    
    if vintage_date:
        # Get vintage for this date or latest before this date
        query = query.lte('vintage_date', vintage_date.isoformat())
    
    if status:
        query = query.eq('fetch_status', status)
    
    result = query.execute()
    return result.data[0]['vintage_id'] if result.data else None


@_ensure_client
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
# Ingestion Job Operations
# ============================================================================

@_ensure_client
def create_ingestion_job(
    github_run_id: str,
    vintage_date: date,
    github_workflow_run_url: Optional[str] = None,
    client: Optional[Client] = None
) -> Dict[str, Any]:
    """Create a new ingestion job for GitHub Actions tracking."""
    
    data = {
        'github_run_id': github_run_id,
        'vintage_date': vintage_date.isoformat(),
        'status': 'running',
        'github_workflow_run_url': github_workflow_run_url,
    }
    
    result = client.table(TABLES['ingestion_jobs']).insert(data).execute()
    return result.data[0] if result.data else None


@_ensure_client
def update_ingestion_job(
    job_id: int,
    status: Optional[str] = None,
    total_series: Optional[int] = None,
    successful_series: Optional[int] = None,
    failed_series: Optional[int] = None,
    error_message: Optional[str] = None,
    logs_json: Optional[Dict[str, Any]] = None,
    client: Optional[Client] = None
) -> Dict[str, Any]:
    """Update an ingestion job."""
    
    update_data = {}
    if status:
        update_data['status'] = status
        if status in ('completed', 'failed', 'cancelled'):
            update_data['completed_at'] = datetime.now().isoformat()
    if total_series is not None:
        update_data['total_series'] = total_series
    if successful_series is not None:
        update_data['successful_series'] = successful_series
    if failed_series is not None:
        update_data['failed_series'] = failed_series
    if error_message:
        update_data['error_message'] = error_message
    if logs_json:
        update_data['logs_json'] = logs_json
    
    result = (
        client.table(TABLES['ingestion_jobs'])
        .update(update_data)
        .eq('job_id', job_id)
        .execute()
    )
    return result.data[0] if result.data else None


def get_ingestion_job(client: Optional[Client] = None, github_run_id: Optional[str] = None, job_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Get an ingestion job by run ID or job ID."""
    if client is None:
        client = get_client()
    
    if github_run_id:
        result = client.table(TABLES['ingestion_jobs']).select('*').eq('github_run_id', github_run_id).execute()
        return result.data[0] if result.data else None
    elif job_id:
        result = client.table(TABLES['ingestion_jobs']).select('*').eq('job_id', job_id).execute()
        return result.data[0] if result.data else None
    return None


# ============================================================================
# Observation Operations (Pandas-Optimized)
# ============================================================================

def insert_observations_from_dataframe(
    df: pd.DataFrame,
    vintage_id: int,
    job_id: Optional[int] = None,
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
        - Optional: item_code1-4, item_name1-4, weight, job_id, api_source
    vintage_id : int
        Vintage ID for the observations
    job_id : int, optional
        Ingestion job ID
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
    if client is None:
        client = get_client()
    
    # Prepare data
    df = df.copy()
    df['vintage_id'] = vintage_id
    if job_id:
        df['job_id'] = job_id
    if api_source:
        df['api_source'] = api_source
    
    # Convert date to string
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.date.astype(str)
    
    # Select columns - required + optional
    required_cols = ['series_id', 'date', 'value']
    optional_cols = ['job_id', 'api_source', 'weight'] + \
                   [f'item_code{i}' for i in range(1, 5)] + \
                   [f'item_name{i}' for i in range(1, 5)]
    
    columns = required_cols + [col for col in optional_cols if col in df.columns]
    
    df_clean = df[columns].copy()
    
    # Remove NaN values
    df_clean = df_clean.dropna(subset=['series_id', 'date', 'value'])
    
    # Convert to records
    records = df_clean.to_dict('records')
    
    # Batch insert with upsert (handle duplicates)
    total_inserted = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            result = (
                client.table(TABLES['observations'])
                .upsert(batch, on_conflict='series_id,vintage_id,date')
                .execute()
            )
            total_inserted += len(batch)
        except Exception as e:
            logger.error(f"Error inserting batch {i//batch_size + 1}: {e}", exc_info=True)
            raise
    
    return total_inserted


@_ensure_client
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


def _create_empty_metadata() -> pd.DataFrame:
    """Create empty series metadata DataFrame with correct columns."""
    return pd.DataFrame(columns=SERIES_METADATA_COLUMNS)


def _create_missing_series_metadata(
    missing_series_ids: List[str],
    strict_mode: bool = False
) -> pd.DataFrame:
    """
    Create metadata entries for missing series.
    
    Parameters
    ----------
    missing_series_ids : List[str]
        Series IDs that are missing
    strict_mode : bool
        If True, don't set default transformation
        
    Returns
    -------
    pd.DataFrame
        DataFrame with metadata entries for missing series
    """
    if not missing_series_ids:
        return _create_empty_metadata()
    
    rows = []
    for series_id in missing_series_ids:
        rows.append({
            'series_id': series_id,
            'series_name': None,
            'transformation': 'lin' if not strict_mode else None,
            'frequency': None,
            'units': None,
            'category': None,
            'api_source': None,
            'api_code': None
        })
    return pd.DataFrame(rows)


def _order_metadata_by_series_ids(
    metadata_df: pd.DataFrame,
    series_ids: List[str]
) -> pd.DataFrame:
    """
    Order metadata DataFrame to match series_ids order.
    
    Parameters
    ----------
    metadata_df : pd.DataFrame
        Metadata DataFrame
    series_ids : List[str]
        Desired order of series IDs
        
    Returns
    -------
    pd.DataFrame
        Ordered metadata DataFrame
    """
    if metadata_df.empty or not series_ids:
        return metadata_df
    
    series_order = {s: i for i, s in enumerate(series_ids)}
    metadata_df['_order'] = metadata_df['series_id'].map(series_order)
    metadata_df = metadata_df.sort_values('_order').drop('_order', axis=1).reset_index(drop=True)
    return metadata_df


def _ensure_metadata_completeness(
    metadata_df: pd.DataFrame,
    requested_series_ids: List[str],
    strict_mode: bool = False
) -> pd.DataFrame:
    """
    Ensure metadata DataFrame contains all requested series IDs.
    
    Adds missing series with None/default values.
    
    Parameters
    ----------
    metadata_df : pd.DataFrame
        Current metadata DataFrame
    requested_series_ids : List[str]
        All series IDs that should be in metadata
    strict_mode : bool
        If True, don't set default transformation for missing series
        
    Returns
    -------
    pd.DataFrame
        Complete metadata DataFrame with all requested series
    """
    if not requested_series_ids:
        return metadata_df
    
    # Get series that are already in metadata
    existing_series = set(metadata_df['series_id'].tolist()) if not metadata_df.empty else set()
    missing_series = set(requested_series_ids) - existing_series
    
    if missing_series:
        missing_metadata = _create_missing_series_metadata(
            sorted(missing_series),
            strict_mode=strict_mode
        )
        if metadata_df.empty:
            metadata_df = missing_metadata
        else:
            metadata_df = pd.concat([metadata_df, missing_metadata], ignore_index=True)
    
    # Order to match requested_series_ids
    return _order_metadata_by_series_ids(metadata_df, requested_series_ids)


def _pivot_observations_to_wide(
    df: pd.DataFrame,
    requested_series: List[str],
    missing_series: set,
    ensure_order: bool = True
) -> pd.DataFrame:
    """
    Pivot observations DataFrame to wide format with proper ordering.
    
    Parameters
    ----------
    df : pd.DataFrame
        Observations DataFrame with columns: date, series_id, value
    requested_series : List[str]
        All requested series IDs in desired order
    missing_series : set
        Set of series IDs that are missing from observations
    ensure_order : bool
        If True, ensure columns match requested_series order
        
    Returns
    -------
    pd.DataFrame
        Pivoted DataFrame with dates as index and series as columns
    """
    if df.empty:
        # Create empty DataFrame with correct column structure
        df_pivot = pd.DataFrame(columns=requested_series)
        df_pivot.index = pd.DatetimeIndex([])
        return df_pivot
    
    # Pivot to wide format
    df_pivot = df.pivot(index='date', columns='series_id', values='value')
    
    # Add missing series as NaN columns
    for missing_id in missing_series:
        if missing_id not in df_pivot.columns:
            df_pivot[missing_id] = np.nan
    
    # Ensure series order matches requested order
    if ensure_order and requested_series:
        # Add any series that weren't in pivot
        for series_id in requested_series:
            if series_id not in df_pivot.columns:
                df_pivot[series_id] = np.nan
        # Reorder columns
        df_pivot = df_pivot[requested_series]
    
    # Sort by date
    if not df_pivot.empty:
        df_pivot = df_pivot.sort_index()
    
    return df_pivot


def _validate_series_metadata(
    metadata_df: pd.DataFrame,
    vintage_id: int,
    strict_mode: bool = False
) -> pd.DataFrame:
    """
    Validate and fix series metadata (transformation and frequency codes).
    
    Parameters
    ----------
    metadata_df : pd.DataFrame
        DataFrame with series metadata
    vintage_id : int
        Vintage ID for error messages
    strict_mode : bool
        If True, raise errors for missing required fields
        
    Returns
    -------
    pd.DataFrame
        Validated metadata DataFrame
    """
    if metadata_df.empty:
        return metadata_df
    
    # Check for missing frequency codes (always required)
    missing_freq = metadata_df['frequency'].isna() | (metadata_df['frequency'] == '')
    if missing_freq.any():
        missing_series = metadata_df[missing_freq]['series_id'].tolist()
        raise ValueError(
            f"Series missing frequency codes in vintage {vintage_id}: {missing_series}"
        )
    
    # Check for missing transformation codes
    missing_transforms = metadata_df['transformation'].isna() | (metadata_df['transformation'] == '')
    if missing_transforms.any():
        missing_series = metadata_df[missing_transforms]['series_id'].tolist()
        if strict_mode:
            raise ValueError(
                f"Series missing transformation codes in vintage {vintage_id}: {missing_series}"
            )
        else:
            # Default to 'lin' for missing transformations
            metadata_df.loc[missing_transforms, 'transformation'] = 'lin'
            logger.warning(
                f"Series missing transformation codes in vintage {vintage_id}, "
                f"defaulting to 'lin': {missing_series}"
            )
    
    return metadata_df


@_ensure_client
def get_series_metadata_bulk(
    series_ids: List[str],
    client: Optional[Client] = None
) -> pd.DataFrame:
    """
    Get metadata for multiple series in bulk.
    
    Parameters
    ----------
    series_ids : List[str]
        List of series IDs to query
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - series_id
        - series_name
        - frequency (d, w, m, q, sa, a)
        - transformation (lin, chg, ch1, pch, etc.)
        - units
        - category
        - api_source
        - api_code
    """
    if not series_ids:
        return _create_empty_metadata()
    
    # Single query for all series_ids
    result = (
        client.table(TABLES['series'])
        .select(SERIES_METADATA_FIELDS)
        .in_('series_id', series_ids)
        .execute()
    )
    
    if not result.data:
        return _create_empty_metadata()
    
    # Convert to DataFrame and maintain order
    metadata_df = pd.DataFrame(result.data)
    return _order_metadata_by_series_ids(metadata_df, series_ids)


@_ensure_client
def get_vintage_data(
    vintage_id: int,
    config_series_ids: Optional[List[str]] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    strict_mode: bool = False,
    ensure_series_order: bool = True,
    client: Optional[Client] = None
) -> tuple[pd.DataFrame, pd.DatetimeIndex, pd.DataFrame]:
    """
    Get all data for a vintage, formatted for DFM input.
    
    Optimized for forecasting queries with efficient indexing and data alignment.
    
    Parameters
    ----------
    vintage_id : int
        Vintage ID to retrieve data for
    config_series_ids : List[str], optional
        List of series IDs to include. If provided, ensures series order matches config.
        If None, returns all series for the vintage.
    start_date : date, optional
        Start date for data filtering
    end_date : date, optional
        End date for data filtering
    strict_mode : bool, default False
        If True, raise errors for missing series. If False, fill with NaN and log warnings.
    ensure_series_order : bool
        If True and config_series_ids provided, ensures output columns match config order
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    tuple
        (data_df, time_index, series_metadata_df) where:
        - data_df is (T x N) DataFrame with dates as index and series as columns
        - time_index is DatetimeIndex
        - series_metadata_df is DataFrame with series metadata (series_id, transformation, frequency, etc.)
    """
    
    # Get observations with efficient query
    df = get_observations(
        client=client, 
        vintage_id=vintage_id,
        start_date=start_date,
        end_date=end_date
    )
    
    # Determine which series to include
    if config_series_ids:
        requested_series = config_series_ids
    else:
        # If no config provided, use all series in vintage
        requested_series = df['series_id'].unique().tolist() if not df.empty else []
    
    if not requested_series:
        # Return empty results with proper shape
        return pd.DataFrame(), pd.DatetimeIndex([]), _create_empty_metadata()
    
    # Get series metadata
    series_metadata = get_series_metadata_bulk(requested_series, client=client)
    
    # Get available series from observations
    available_series = df['series_id'].unique().tolist() if not df.empty else []
    missing_series = set(requested_series) - set(available_series)
    
    # Handle missing series
    if missing_series:
        if strict_mode:
            raise ValueError(
                f"Missing series in vintage {vintage_id}: {sorted(missing_series)}"
            )
        else:
            logger.warning(f"Missing series in vintage {vintage_id}: {sorted(missing_series)}")
            # Create NaN columns for missing series to maintain shape
            if df.empty:
                # Need to create a minimal DataFrame structure
                # This will be handled by pivot below
                pass
    
    # Filter observations by requested series
    if not df.empty:
        df = df[df['series_id'].isin(requested_series)]
    
    # Pivot to wide format with proper ordering
    df_pivot = _pivot_observations_to_wide(
        df=df,
        requested_series=requested_series,
        missing_series=missing_series,
        ensure_order=ensure_series_order
    )
    
    # Get Time index
    Time = df_pivot.index
    
    # Ensure metadata contains all requested series and is properly ordered
    series_metadata = _ensure_metadata_completeness(
        series_metadata, 
        requested_series, 
        strict_mode=strict_mode
    )
    
    # Validate metadata (after adding any missing series entries)
    series_metadata = _validate_series_metadata(series_metadata, vintage_id, strict_mode)
    
    return df_pivot, Time, series_metadata


@_ensure_client
def get_vintage_data_for_config(
    vintage_id: int,
    config_id: int,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    strict_mode: bool = False,
    client: Optional[Client] = None
) -> tuple[pd.DataFrame, pd.DatetimeIndex, pd.DataFrame]:
    """
    Get vintage data aligned with a model configuration.
    
    Retrieves series IDs from model_block_assignments and ensures proper ordering.
    
    Parameters
    ----------
    vintage_id : int
        Vintage ID to retrieve data for
    config_id : int
        Model configuration ID
    start_date : date, optional
        Start date for data filtering
    end_date : date, optional
        End date for data filtering
    strict_mode : bool, default False
        If True, raise errors for missing series. If False, fill with NaN and log warnings.
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    tuple
        (data_df, time_index, series_metadata_df) where:
        - data_df is (T x N) DataFrame with dates as index and series as columns
        - time_index is DatetimeIndex
        - series_metadata_df is DataFrame with series metadata (series_id, transformation, frequency, etc.)
    """
    # Get series IDs from model configuration
    result = (
        client.table(TABLES['model_block_assignments'])
        .select('series_id')
        .eq('config_id', config_id)
        .order('block_index, series_id')
        .execute()
    )
    
    if not result.data:
        logger.warning(f"No series found for config_id {config_id}")
        return pd.DataFrame(), pd.DatetimeIndex([]), _create_empty_metadata()
    
    config_series_ids = [row['series_id'] for row in result.data]
    
    # Get data with proper ordering and metadata
    Z, Time, series_metadata = get_vintage_data(
        vintage_id=vintage_id,
        config_series_ids=config_series_ids,
        start_date=start_date,
        end_date=end_date,
        strict_mode=strict_mode,
        ensure_series_order=True,
        client=client
    )
    
    return Z, Time, series_metadata


@_ensure_client
def get_series_metadata_for_config(
    config_id: int,
    client: Optional[Client] = None
) -> pd.DataFrame:
    """
    Get series metadata aligned with a model configuration.
    
    Returns series information (IDs, names, frequencies, etc.) in the order
    specified by the model configuration.
    
    Parameters
    ----------
    config_id : int
        Model configuration ID
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    pd.DataFrame
        DataFrame with series metadata, ordered by config block assignments
    """
    # Get series IDs from model configuration with block information
    result = (
        client.table(TABLES['model_block_assignments'])
        .select('series_id, block_name, block_index')
        .eq('config_id', config_id)
        .order('block_index, series_id')
        .execute()
    )
    
    if not result.data:
        return pd.DataFrame()
    
    # Get series metadata
    series_ids = [row['series_id'] for row in result.data]
    series_list = []
    
    for series_id in series_ids:
        series = get_series(series_id, client=client)
        if series:
            # Add block information
            block_info = next((r for r in result.data if r['series_id'] == series_id), None)
            if block_info:
                series['block_name'] = block_info['block_name']
                series['block_index'] = block_info['block_index']
            series_list.append(series)
    
    if not series_list:
        return pd.DataFrame()
    
    return pd.DataFrame(series_list)


@_ensure_client
def compare_vintages(
    vintage_id_old: int,
    vintage_id_new: int,
    series_ids: Optional[List[str]] = None,
    client: Optional[Client] = None
) -> pd.DataFrame:
    """
    Compare observations between two vintages.
    
    Useful for nowcast updates to see what data changed.
    
    Parameters
    ----------
    vintage_id_old : int
        Old vintage ID
    vintage_id_new : int
        New vintage ID
    series_ids : List[str], optional
        Series IDs to compare. If None, compares all overlapping series.
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: series_id, date, value_old, value_new, diff
    """
    # Get old vintage data
    df_old = get_observations(
        client=client,
        vintage_id=vintage_id_old,
        series_id=None  # Will filter by series_ids later if provided
    )
    
    # Get new vintage data
    df_new = get_observations(
        client=client,
        vintage_id=vintage_id_new,
        series_id=None
    )
    
    if df_old.empty or df_new.empty:
        return pd.DataFrame()
    
    # Filter by series if provided
    if series_ids:
        df_old = df_old[df_old['series_id'].isin(series_ids)]
        df_new = df_new[df_new['series_id'].isin(series_ids)]
    
    # Merge on series_id and date
    df_merged = pd.merge(
        df_old[['series_id', 'date', 'value']],
        df_new[['series_id', 'date', 'value']],
        on=['series_id', 'date'],
        how='outer',
        suffixes=('_old', '_new')
    )
    
    # Calculate difference
    df_merged['diff'] = df_merged['value_new'] - df_merged['value_old']
    df_merged = df_merged.rename(columns={'value_old': 'value_old', 'value_new': 'value_new'})
    
    return df_merged.sort_values(['series_id', 'date'])


# ============================================================================
# Model Operations
# ============================================================================

@_ensure_client
def save_model_config(
    config_name: str,
    config_json: Dict[str, Any],
    block_names: List[str],
    series_block_assignments: Optional[Dict[str, str]] = None,
    description: Optional[str] = None,
    country: str = 'KR',
    client: Optional[Client] = None
) -> Dict[str, Any]:
    """
    Save a model configuration.
    
    Parameters
    ----------
    config_name : str
        Unique configuration name
    config_json : Dict[str, Any]
        Full ModelConfig structure as JSON
    block_names : List[str]
        List of block names in order
    series_block_assignments : Dict[str, str], optional
        Dictionary mapping series_id to block_name.
        If not provided, will attempt to extract from config_json.
    description : str, optional
        Configuration description
    country : str, default 'KR'
        Country code
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    Dict[str, Any]
        Saved configuration record with config_id
    """
    # Save model config
    data = {
        'config_name': config_name,
        'config_json': config_json,
        'block_names': block_names,
        'description': description,
        'country': country,
    }
    
    result = client.table(TABLES['model_configs']).upsert(data, on_conflict='config_name').execute()
    config_record = result.data[0] if result.data else None
    
    if not config_record:
        raise ValueError(f"Failed to save model configuration: {config_name}")
    
    config_id = config_record['config_id']
    
    # Save block assignments if provided
    if series_block_assignments:
        # Delete existing assignments for this config
        client.table(TABLES['model_block_assignments']).delete().eq('config_id', config_id).execute()
        
        # Create block index mapping
        block_index_map = {block_name: idx for idx, block_name in enumerate(block_names)}
        
        # Insert new assignments
        assignments = []
        for series_id, block_name in series_block_assignments.items():
            if block_name not in block_index_map:
                logger.warning(f"Block '{block_name}' not found in block_names for config {config_name}")
                continue
            
            assignments.append({
                'config_id': config_id,
                'series_id': series_id,
                'block_name': block_name,
                'block_index': block_index_map[block_name]
            })
        
        if assignments:
            # Batch insert (Supabase supports up to 1000 rows)
            batch_size = 1000
            for i in range(0, len(assignments), batch_size):
                batch = assignments[i:i + batch_size]
                client.table(TABLES['model_block_assignments']).insert(batch).execute()
    
    return config_record


@_ensure_client
def load_model_config(config_name: str, client: Optional[Client] = None) -> Optional[Dict[str, Any]]:
    """Load a model configuration."""
    
    result = client.table(TABLES['model_configs']).select('*').eq('config_name', config_name).execute()
    return result.data[0] if result.data else None


@_ensure_client
def get_config_id(config_name: str, client: Optional[Client] = None) -> Optional[int]:
    """
    Get config_id from config_name.
    
    Parameters
    ----------
    config_name : str
        Model configuration name
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    int or None
        Configuration ID, or None if not found
    """
    config = load_model_config(config_name, client=client)
    return config['config_id'] if config else None


@_ensure_client
def list_model_configs(
    country: Optional[str] = None,
    client: Optional[Client] = None
) -> List[Dict[str, Any]]:
    """
    List all model configurations.
    
    Parameters
    ----------
    country : str, optional
        Filter by country code
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    List[Dict[str, Any]]
        List of model configuration records
    """
    query = client.table(TABLES['model_configs']).select('*')
    
    if country:
        query = query.eq('country', country)
    
    result = query.order('config_name').execute()
    return result.data if result.data else []


@_ensure_client
def get_model_config_series_ids(
    config_name: str,
    client: Optional[Client] = None
) -> Optional[List[str]]:
    """
    Get ordered list of series IDs from model configuration.
    
    Retrieves series IDs from model_block_assignments table, ordered by block_index.
    
    Parameters
    ----------
    config_name : str
        Model configuration name
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    List[str] or None
        Ordered list of series IDs, or None if config not found
    """
    config_id = get_config_id(config_name, client=client)
    if config_id is None:
        return None
    
    # Get series IDs from model_block_assignments
    result = (
        client.table(TABLES['model_block_assignments'])
        .select('series_id')
        .eq('config_id', config_id)
        .order('block_index, series_id')
        .execute()
    )
    
    if not result.data:
        return []
    
    return [row['series_id'] for row in result.data]


@_ensure_client
def save_model_weights(
    config_id: int,
    vintage_id: int,
    parameters: Dict[str, Any],
    threshold: Optional[float] = None,
    convergence_iter: Optional[int] = None,
    log_likelihood: Optional[float] = None,
    client: Optional[Client] = None
) -> Dict[str, Any]:
    """Save trained model weights (DFMResult)."""
    
    # Serialize numpy arrays to lists
    def serialize_array(arr):
        if isinstance(arr, np.ndarray):
            return arr.tolist()
        return arr
    
    data = {
        'config_id': config_id,
        'vintage_id': vintage_id,
        'parameters_json': {k: serialize_array(v) for k, v in parameters.items()},
        'threshold': threshold,
        'convergence_iter': convergence_iter,
        'log_likelihood': log_likelihood,
    }
    
    result = client.table(TABLES['trained_models']).insert(data).execute()
    return result.data[0] if result.data else None


@_ensure_client
def load_model_weights(model_id: int, client: Optional[Client] = None) -> Optional[Dict[str, Any]]:
    """Load trained model weights."""
    
    result = client.table(TABLES['trained_models']).select('*').eq('model_id', model_id).execute()
    return result.data[0] if result.data else None


# ============================================================================
# Forecast Operations
# ============================================================================

@_ensure_client
def save_forecast(
    model_id: int,
    series_id: str,
    forecast_date: date,
    forecast_value: float,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    confidence_level: float = 0.95,
    client: Optional[Client] = None
) -> Dict[str, Any]:
    """Save a forecast."""
    
    data = {
        'model_id': model_id,
        'series_id': series_id,
        'forecast_date': forecast_date.isoformat(),
        'forecast_value': forecast_value,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'confidence_level': confidence_level,
    }
    
    result = client.table(TABLES['forecasts']).insert(data).execute()
    return result.data[0] if result.data else None


@_ensure_client
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


@_ensure_client
def get_latest_forecasts(client: Optional[Client] = None, limit: int = 100) -> pd.DataFrame:
    """Get latest forecasts for dashboard (using view)."""
    
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


@_ensure_client
def create_forecast_run(
    model_id: int,
    vintage_id_new: int,
    run_type: str = 'nowcast',
    vintage_id_old: Optional[int] = None,
    target_series_id: Optional[str] = None,
    target_period: Optional[str] = None,
    github_run_id: Optional[str] = None,
    github_workflow_run_url: Optional[str] = None,
    client: Optional[Client] = None
) -> Dict[str, Any]:
    """
    Create a forecast run record.
    
    Parameters
    ----------
    model_id : int
        Trained model ID
    vintage_id_new : int
        New vintage ID for forecast
    run_type : str
        Type of run: 'nowcast', 'forecast', 'batch'
    vintage_id_old : int, optional
        Old vintage ID (for nowcast comparisons)
    target_series_id : str, optional
        Target series being forecasted
    target_period : str, optional
        Target period (e.g., '2016q4')
    github_run_id : str, optional
        GitHub Actions run ID
    github_workflow_run_url : str, optional
        GitHub Actions workflow URL
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    Dict[str, Any]
        Created forecast run record
    """
    data = {
        'model_id': model_id,
        'vintage_id_new': vintage_id_new,
        'run_type': run_type,
        'status': 'pending',
        'started_at': datetime.now().isoformat(),
    }
    
    if vintage_id_old is not None:
        data['vintage_id_old'] = vintage_id_old
    if target_series_id:
        data['target_series_id'] = target_series_id
    if target_period:
        data['target_period'] = target_period
    if github_run_id:
        data['github_run_id'] = github_run_id
    if github_workflow_run_url:
        data['github_workflow_run_url'] = github_workflow_run_url
    
    result = client.table(TABLES['forecast_runs']).insert(data).execute()
    return result.data[0] if result.data else None


@_ensure_client
def get_forecast_run(
    run_id: Optional[int] = None,
    github_run_id: Optional[str] = None,
    model_id: Optional[int] = None,
    status: Optional[str] = None,
    client: Optional[Client] = None
) -> Optional[Dict[str, Any]]:
    """
    Get forecast run by ID or filters.
    
    Parameters
    ----------
    run_id : int, optional
        Forecast run ID
    github_run_id : str, optional
        GitHub Actions run ID
    model_id : int, optional
        Model ID
    status : str, optional
        Filter by status
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    Dict[str, Any] or None
        Forecast run record, or None if not found
    """
    if run_id:
        result = (
            client.table(TABLES['forecast_runs'])
            .select('*')
            .eq('run_id', run_id)
            .execute()
        )
        return result.data[0] if result.data else None
    
    query = client.table(TABLES['forecast_runs']).select('*')
    
    if github_run_id:
        query = query.eq('github_run_id', github_run_id)
    if model_id:
        query = query.eq('model_id', model_id)
    if status:
        query = query.eq('status', status)
    
    query = query.order('created_at', desc=True).limit(1)
    result = query.execute()
    return result.data[0] if result.data else None


@_ensure_client
def update_forecast_run(
    run_id: int,
    status: Optional[str] = None,
    forecasts_generated: Optional[int] = None,
    error_message: Optional[str] = None,
    metadata_json: Optional[Dict[str, Any]] = None,
    client: Optional[Client] = None
) -> Dict[str, Any]:
    """
    Update a forecast run record.
    
    Parameters
    ----------
    run_id : int
        Forecast run ID
    status : str, optional
        New status: 'pending', 'running', 'completed', 'failed'
    forecasts_generated : int, optional
        Number of forecasts generated
    error_message : str, optional
        Error message if failed
    metadata_json : Dict[str, Any], optional
        Additional metadata
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    Dict[str, Any]
        Updated forecast run record
    """
    update_data = {}
    
    if status:
        update_data['status'] = status
        if status in ('completed', 'failed'):
            update_data['completed_at'] = datetime.now().isoformat()
    
    if forecasts_generated is not None:
        update_data['forecasts_generated'] = forecasts_generated
    if error_message:
        update_data['error_message'] = error_message
    if metadata_json:
        update_data['metadata_json'] = metadata_json
    
    if not update_data:
        return {}
    
    result = (
        client.table(TABLES['forecast_runs'])
        .update(update_data)
        .eq('run_id', run_id)
        .execute()
    )
    return result.data[0] if result.data else None


@_ensure_client
def save_forecasts_batch(
    run_id: int,
    forecasts: List[Dict[str, Any]],
    client: Optional[Client] = None
) -> int:
    """
    Save multiple forecasts in a batch.
    
    Parameters
    ----------
    run_id : int
        Forecast run ID
    forecasts : List[Dict[str, Any]]
        List of forecast dictionaries with keys:
        - model_id: int
        - series_id: str
        - forecast_date: date
        - forecast_value: float
        - lower_bound: float (optional)
        - upper_bound: float (optional)
        - confidence_level: float (optional, default 0.95)
        - metadata_json: Dict (optional)
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    int
        Number of forecasts saved
    """
    if not forecasts:
        return 0
    
    # Prepare data for batch insert
    data_list = []
    for forecast in forecasts:
        data = {
            'run_id': run_id,
            'model_id': forecast['model_id'],
            'series_id': forecast['series_id'],
            'forecast_date': forecast['forecast_date'].isoformat() if isinstance(forecast['forecast_date'], date) else forecast['forecast_date'],
            'forecast_value': forecast['forecast_value'],
            'lower_bound': forecast.get('lower_bound'),
            'upper_bound': forecast.get('upper_bound'),
            'confidence_level': forecast.get('confidence_level', 0.95),
        }
        if 'metadata_json' in forecast:
            data['metadata_json'] = forecast['metadata_json']
        data_list.append(data)
    
    # Batch insert (Supabase supports up to 1000 rows per insert)
    batch_size = 1000
    total_inserted = 0
    
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i + batch_size]
        result = client.table(TABLES['forecasts']).insert(batch).execute()
        total_inserted += len(result.data) if result.data else 0
    
    # Update forecast run with count
    update_forecast_run(
        run_id=run_id,
        forecasts_generated=total_inserted,
        client=client
    )
    
    return total_inserted



# ============================================================================
# Statistics Metadata Operations
# ============================================================================

@_ensure_client
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


@_ensure_client
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


@_ensure_client
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


@_ensure_client
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

@_ensure_client
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


@_ensure_client
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


@_ensure_client
def get_active_items_for_statistic(
    statistics_metadata_id: int,
    client: Optional[Client] = None,
    cycle: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get active items for data collection."""
    return get_statistics_items(statistics_metadata_id, client, cycle, is_active=True)
