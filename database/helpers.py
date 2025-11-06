"""Helper functions for database operations.

This module provides common utilities and patterns used across database operations
to reduce duplication and improve consistency.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Callable, TypeVar
from functools import wraps
from supabase import Client

from .client import get_client

logger = logging.getLogger(__name__)

T = TypeVar('T')

# ============================================================================
# Client Management
# ============================================================================

def ensure_client(client: Optional[Client] = None) -> Client:
    """
    Ensure a client is available, creating one if needed.
    
    This helper reduces repetitive `if client is None: client = get_client()` patterns.
    
    Parameters
    ----------
    client : Optional[Client]
        Existing client instance or None
        
    Returns
    -------
    Client
        Client instance (existing or newly created)
    """
    if client is None:
        return get_client()
    return client


# ============================================================================
# Batch Operations
# ============================================================================

def batch_process(
    items: List[T],
    batch_size: int,
    processor: Callable[[List[T]], Any],
    error_handler: Optional[Callable[[Exception, List[T]], Any]] = None
) -> List[Any]:
    """
    Process items in batches with error handling.
    
    Parameters
    ----------
    items : List[T]
        Items to process
    batch_size : int
        Size of each batch
    processor : Callable[[List[T]], Any]
        Function to process each batch
    error_handler : Callable[[Exception, List[T]], Any], optional
        Function to handle errors (default: log and continue)
        
    Returns
    -------
    List[Any]
        Results from processing batches
    """
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        try:
            result = processor(batch)
            results.append(result)
        except Exception as e:
            if error_handler:
                error_handler(e, batch)
            else:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}", exc_info=True)
    
    return results


def batch_insert(
    client: Client,
    table_name: str,
    records: List[Dict[str, Any]],
    batch_size: int = 1000,
    on_conflict: Optional[str] = None
) -> int:
    """
    Insert records in batches with optional upsert.
    
    Parameters
    ----------
    client : Client
        Supabase client
    table_name : str
        Table name
    records : List[Dict[str, Any]]
        Records to insert
    batch_size : int
        Batch size (default: 1000)
    on_conflict : str, optional
        Conflict resolution strategy (for upsert)
        
    Returns
    -------
    int
        Total number of records inserted
    """
    if not records:
        return 0
    
    total_inserted = 0
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            query = client.table(table_name)
            
            if on_conflict:
                result = query.upsert(batch, on_conflict=on_conflict).execute()
            else:
                result = query.insert(batch).execute()
            
            if result.data:
                total_inserted += len(result.data)
                
        except Exception as e:
            logger.error(f"Error inserting batch {i//batch_size + 1}: {e}", exc_info=True)
            raise
    
    return total_inserted


def batch_query_in(
    client: Client,
    table_name: str,
    column: str,
    values: List[Any],
    batch_size: int = 100,
    select: str = '*',
    additional_filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Query records where column IN (values) using batching.
    
    Parameters
    ----------
    client : Client
        Supabase client
    table_name : str
        Table name
    column : str
        Column to filter on
    values : List[Any]
        Values to filter by
    batch_size : int
        Batch size for queries (default: 100)
    select : str
        Columns to select (default: '*')
    additional_filters : Dict[str, Any], optional
        Additional filters to apply
        
    Returns
    -------
    List[Dict[str, Any]]
        All matching records
    """
    all_results = []
    
    for i in range(0, len(values), batch_size):
        batch = values[i:i + batch_size]
        try:
            query = client.table(table_name).select(select)
            
            if len(batch) == 1:
                query = query.eq(column, batch[0])
            else:
                query = query.in_(column, batch)
            
            # Apply additional filters
            if additional_filters:
                for key, value in additional_filters.items():
                    query = query.eq(key, value)
            
            result = query.execute()
            if result.data:
                all_results.extend(result.data)
                
        except Exception as e:
            logger.error(f"Error querying batch {i//batch_size + 1}: {e}", exc_info=True)
            raise
    
    return all_results


# ============================================================================
# Query Builders
# ============================================================================

def build_query(
    client: Client,
    table_name: str,
    filters: Optional[Dict[str, Any]] = None,
    select: str = '*',
    order_by: Optional[str] = None,
    order_desc: bool = False,
    limit: Optional[int] = None
):
    """
    Build a Supabase query with common filters.
    
    Parameters
    ----------
    client : Client
        Supabase client
    table_name : str
        Table name
    filters : Dict[str, Any], optional
        Filters to apply (column: value)
    select : str
        Columns to select (default: '*')
    order_by : str, optional
        Column to order by
    order_desc : bool
        Order descending (default: False)
    limit : int, optional
        Limit results
        
    Returns
    -------
    QueryBuilder
        Configured query builder
    """
    query = client.table(table_name).select(select)
    
    if filters:
        for key, value in filters.items():
            if value is not None:
                query = query.eq(key, value)
    
    if order_by:
        query = query.order(order_by, desc=order_desc)
    
    if limit:
        query = query.limit(limit)
    
    return query


# ============================================================================
# Validation Helpers
# ============================================================================

def validate_required_fields(data: Dict[str, Any], required: List[str]) -> None:
    """
    Validate that required fields are present.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Data to validate
    required : List[str]
        Required field names
        
    Raises
    ------
    ValueError
        If any required field is missing
    """
    missing = [field for field in required if field not in data or data[field] is None]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")


def sanitize_for_db(value: Any) -> Any:
    """
    Sanitize value for database insertion.
    
    Parameters
    ----------
    value : Any
        Value to sanitize
        
    Returns
    -------
    Any
        Sanitized value
    """
    if isinstance(value, (dict, list)):
        # JSON serialization handled by Supabase client
        return value
    return value


# ============================================================================
# Error Handling
# ============================================================================

class DatabaseError(Exception):
    """Base exception for database operations."""
    pass


class NotFoundError(DatabaseError):
    """Raised when a record is not found."""
    pass


class ValidationError(DatabaseError):
    """Raised when data validation fails."""
    pass


# ============================================================================
# Serialization Utilities
# ============================================================================

def serialize_numpy_array(arr: np.ndarray) -> List:
    """Serialize numpy array to list for JSON storage."""
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    elif isinstance(arr, (list, tuple)):
        return [serialize_numpy_array(x) if isinstance(x, np.ndarray) else x for x in arr]
    return arr


def deserialize_numpy_array(data: List) -> np.ndarray:
    """Deserialize list to numpy array."""
    return np.array(data)


def serialize_dfm_result(result) -> Dict[str, Any]:
    """Serialize DFMResult to dictionary for database storage."""
    return {
        'parameters_json': {
            'C': serialize_numpy_array(result.C),
            'A': serialize_numpy_array(result.A),
            'Q': serialize_numpy_array(result.Q),
            'R': serialize_numpy_array(result.R),
        },
        'factors_json': {
            'Z': serialize_numpy_array(result.Z),
        },
        'standardization_json': {
            'Mx': serialize_numpy_array(result.Mx),
            'Wx': serialize_numpy_array(result.Wx),
        },
        'initial_conditions_json': {
            'Z_0': serialize_numpy_array(result.Z_0),
            'V_0': serialize_numpy_array(result.V_0),
        },
        'structure_json': {
            'r': serialize_numpy_array(result.r),
            'p': result.p,
        },
    }


def deserialize_dfm_result(data: Dict[str, Any]) -> Dict[str, Any]:
    """Deserialize database record to DFMResult-compatible dictionary."""
    result = {}
    
    if 'parameters_json' in data:
        params = data['parameters_json']
        result['C'] = deserialize_numpy_array(params.get('C', []))
        result['A'] = deserialize_numpy_array(params.get('A', []))
        result['Q'] = deserialize_numpy_array(params.get('Q', []))
        result['R'] = deserialize_numpy_array(params.get('R', []))
    
    if 'factors_json' in data:
        factors = data['factors_json']
        result['Z'] = deserialize_numpy_array(factors.get('Z', []))
    
    if 'standardization_json' in data:
        std = data['standardization_json']
        result['Mx'] = deserialize_numpy_array(std.get('Mx', []))
        result['Wx'] = deserialize_numpy_array(std.get('Wx', []))
    
    if 'initial_conditions_json' in data:
        init = data['initial_conditions_json']
        result['Z_0'] = deserialize_numpy_array(init.get('Z_0', []))
        result['V_0'] = deserialize_numpy_array(init.get('V_0', []))
    
    if 'structure_json' in data:
        struct = data['structure_json']
        result['r'] = deserialize_numpy_array(struct.get('r', []))
        result['p'] = struct.get('p', 1)
    
    return result


# ============================================================================
# Frequency Mapping
# ============================================================================

def map_frequency_to_code(frequency: str) -> str:
    """
    Map frequency code to internal DFM frequency code.
    
    Parameters
    ----------
    frequency : str
        Source frequency code (e.g., 'A', 'Q', 'M', 'D')
        
    Returns
    -------
    str
        Internal frequency code (d, w, m, q, sa, a)
    """
    mapping = {
        'D': 'd',   # Daily
        'SM': 'w',  # Semi-monthly → weekly
        'M': 'm',   # Monthly
        'Q': 'q',   # Quarterly
        'S': 'sa',  # Semi-annual
        'A': 'a'    # Annual
    }
    return mapping.get(frequency.upper(), frequency.lower())


# ============================================================================
# Blocks Table Helpers
# ============================================================================

def get_series_ids_for_config(
    config_name: str,
    client: Optional[Client] = None
) -> List[str]:
    """
    Get ordered list of series IDs for a configuration.
    
    Orders by blocks.series_order to maintain DFM block structure.
    
    Parameters
    ----------
    config_name : str
        Configuration name (e.g., '001-initial-spec')
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    List[str]
        Ordered list of series IDs
    """
    client = ensure_client(client)
    
    result = (
        client.table('blocks')
        .select('series_id')
        .eq('config_name', config_name)
        .order('series_order', desc=False)
        .execute()
    )
    
    if not result.data:
        return []
    
    # Return distinct series IDs in order
    seen = set()
    series_ids = []
    for row in result.data:
        series_id = row['series_id']
        if series_id not in seen:
            seen.add(series_id)
            series_ids.append(series_id)
    
    return series_ids


def get_block_assignments_for_config(
    config_name: str,
    client: Optional[Client] = None
) -> Dict[str, List[str]]:
    """
    Get block assignments for a configuration.
    
    Returns a dictionary mapping series_id to list of block names.
    
    Parameters
    ----------
    config_name : str
        Configuration name (e.g., '001-initial-spec')
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping series_id to list of block names
    """
    client = ensure_client(client)
    
    result = (
        client.table('blocks')
        .select('series_id, block_name')
        .eq('config_name', config_name)
        .execute()
    )
    
    assignments: Dict[str, List[str]] = {}
    for row in result.data:
        series_id = row['series_id']
        block_name = row['block_name']
        if series_id not in assignments:
            assignments[series_id] = []
        assignments[series_id].append(block_name)
    
    return assignments


def get_block_names_for_config(
    config_name: str,
    client: Optional[Client] = None
) -> List[str]:
    """
    Get unique block names for a configuration.
    
    Parameters
    ----------
    config_name : str
        Configuration name (e.g., '001-initial-spec')
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    List[str]
        Sorted list of unique block names
    """
    client = ensure_client(client)
    
    result = (
        client.table('blocks')
        .select('block_name')
        .eq('config_name', config_name)
        .execute()
    )
    
    block_names = sorted(set(row['block_name'] for row in result.data))
    return block_names


def resolve_config_name(
    config_id: Optional[int] = None,
    config_name: Optional[str] = None,
    client: Optional[Client] = None
) -> Optional[str]:
    """
    Resolve config_name from config_id if needed.
    
    If config_name is provided, returns it. If only config_id is provided,
    looks up the config_name from the database.
    
    Parameters
    ----------
    config_id : int, optional
        Configuration ID
    config_name : str, optional
        Configuration name
    client : Client, optional
        Supabase client instance
        
    Returns
    -------
    Optional[str]
        Configuration name, or None if not found
    """
    if config_name:
        return config_name
    
    if config_id is None:
        return None
    
    client = ensure_client(client)
    
    try:
        result = (
            client.table('model_configs')
            .select('config_name')
            .eq('config_id', config_id)
            .execute()
        )
        
        if result.data:
            return result.data[0]['config_name']
    except Exception as e:
        logger.warning(f"Could not resolve config_name from config_id {config_id}: {e}")
    
    return None

