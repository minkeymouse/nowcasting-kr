"""Helper functions for database operations.

This module provides common utilities and patterns used across database operations
to reduce duplication and improve consistency.
"""

import logging
from typing import List, Dict, Any, Optional, Callable, TypeVar
from functools import wraps
from supabase import Client

logger = logging.getLogger(__name__)

T = TypeVar('T')


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


def handle_db_error(func: Callable) -> Callable:
    """
    Decorator to handle common database errors.
    
    Parameters
    ----------
    func : Callable
        Function to wrap
        
    Returns
    -------
    Callable
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Database error in {func.__name__}: {e}", exc_info=True)
            raise DatabaseError(f"Error in {func.__name__}: {str(e)}") from e
    
    return wrapper

