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


@_ensure_client
def get_vintage_data(
    vintage_id: int,
    config_series_ids: Optional[List[str]] = None,
    client: Optional[Client] = None
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    Get all data for a vintage, formatted for DFM input.
    
    Returns
    -------
    tuple
        (Z, Time) where Z is (T x N) array and Time is DatetimeIndex
    """
    
    # Get observations
    df = get_observations(client=client, vintage_id=vintage_id)
    
    if df.empty:
        return pd.DataFrame(), pd.DatetimeIndex([])
    
    # Filter by config series if provided
    if config_series_ids:
        df = df[df['series_id'].isin(config_series_ids)]
    
    # Pivot to wide format: rows = dates, columns = series
    df_pivot = df.pivot(index='date', columns='series_id', values='value')
    
    # Sort by date
    df_pivot = df_pivot.sort_index()
    
    # Get Time index
    Time = df_pivot.index
    
    # Convert to numpy array (Z matrix)
    Z = df_pivot.values
    
    return Z, Time


# ============================================================================
# Model Operations
# ============================================================================

@_ensure_client
def save_model_config(
    config_name: str,
    config_json: Dict[str, Any],
    block_names: List[str],
    description: Optional[str] = None,
    country: str = 'KR',
    client: Optional[Client] = None
) -> Dict[str, Any]:
    """Save a model configuration."""
    
    data = {
        'config_name': config_name,
        'config_json': config_json,
        'block_names': block_names,
        'description': description,
        'country': country,
    }
    
    result = client.table(TABLES['model_configs']).upsert(data, on_conflict='config_name').execute()
    return result.data[0] if result.data else None


@_ensure_client
def load_model_config(config_name: str, client: Optional[Client] = None) -> Optional[Dict[str, Any]]:
    """Load a model configuration."""
    
    result = client.table(TABLES['model_configs']).select('*').eq('config_name', config_name).execute()
    return result.data[0] if result.data else None


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
