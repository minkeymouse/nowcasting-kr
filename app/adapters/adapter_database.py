"""Database adapters for DFM module.

This module bridges the generic DFM module (src/nowcasting) with the application-specific
database layer. It provides functions to load data from database and save results back.

The DFM module itself remains generic and database-agnostic.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple, Union, Any, Dict
from datetime import date
import warnings
import logging
import pickle
import io
import os
import re

# Import generic DFM module functions
from dfm_python.config import DFMConfig
from dfm_python.data_loader import _transform_series

logger = logging.getLogger(__name__)


def _convert_to_dataframes(
    X: np.ndarray,
    Time: pd.DatetimeIndex,
    Z: Optional[np.ndarray],
    series_metadata_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DatetimeIndex, Optional[pd.DataFrame], pd.DataFrame]:
    """
    Convert numpy arrays to DataFrames for consistent return format.
    
    Parameters
    ----------
    X : np.ndarray
        Transformed data array
    Time : pd.DatetimeIndex
        Time index
    Z : np.ndarray, optional
        Raw data array
    series_metadata_df : pd.DataFrame
        Series metadata DataFrame
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DatetimeIndex, Optional[pd.DataFrame], pd.DataFrame]
        (data_df, Time, Z_df, series_metadata_df)
    """
    data_df = pd.DataFrame(X, index=Time, columns=None)
    Z_df = pd.DataFrame(Z, index=Time, columns=None) if Z is not None and len(Z) == len(Time) else None
    return data_df, Time, Z_df, series_metadata_df


def _get_db_client(client: Optional[object] = None):
    """Get database client, raising ImportError if database module unavailable."""
    if client is not None:
        return client
    try:
        from app.database import get_client
        return get_client()
    except ImportError as e:
        raise ImportError(
            "Database module not available. Install dependencies or use file-based loading."
        ) from e


def _normalize_date(date_input: Optional[Union[date, str, pd.Timestamp]]) -> Optional[date]:
    """Normalize date input to date object."""
    if date_input is None:
        return None
    if isinstance(date_input, date):
        return date_input
    if isinstance(date_input, str):
        return pd.to_datetime(date_input).date()
    if isinstance(date_input, pd.Timestamp):
        return date_input.date()
    return None


def _resolve_vintage_id(
    vintage_id: Optional[int] = None,
    vintage_date: Optional[Union[date, str, pd.Timestamp]] = None,
    client: Optional[object] = None
) -> Optional[int]:
    """Resolve vintage_id from vintage_date if needed."""
    if vintage_id is not None:
        return vintage_id
    
    if vintage_date is not None:
        try:
            from app.database import get_latest_vintage_id
            normalized_date = _normalize_date(vintage_date)
            resolved_id = get_latest_vintage_id(vintage_date=normalized_date, client=client)
            if resolved_id is None:
                logger.warning(f"No vintage found for date {normalized_date}")
            return resolved_id
        except ImportError:
            logger.error("Database module not available. Cannot resolve vintage_id from date.")
            return None
        except Exception as e:
            logger.error(f"Error resolving vintage_id from date {vintage_date}: {e}")
            return None
    
    return None


def _apply_transformations_from_metadata(
    Z: np.ndarray,
    Time: pd.DatetimeIndex,
    data_df: pd.DataFrame,
    series_metadata_df: pd.DataFrame,
    config: Optional[DFMConfig] = None
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Apply transformations using series metadata, falling back to config if needed.
    
    This function consolidates transformation logic to work with both database
        metadata and DFMConfig objects (from CSV or YAML).
    """
    T, N = Z.shape
    X = np.full((T, N), np.nan)
    
    # Build metadata lookup from database
    metadata_dict = {}
    if not series_metadata_df.empty and 'series_id' in series_metadata_df.columns:
        for _, row in series_metadata_df.iterrows():
            series_id = row['series_id']
            metadata_dict[series_id] = {
                'transformation': row.get('transformation', 'lin'),
                'frequency': row.get('frequency', 'm')
            }
    
    # Apply transformations using metadata or config
    if config is not None:
        # Use config (works with CSV or YAML configs)
        for i in range(N):
            if i < len(config.SeriesID):
                series_id = config.SeriesID[i]
                # Prefer metadata if available, otherwise use config
                if series_id in metadata_dict:
                    transform = metadata_dict[series_id]['transformation']
                    freq = metadata_dict[series_id]['frequency']
                else:
                    transform = config.Transformation[i]
                    freq = config.Frequency[i]
            else:
                # Fallback for series not in config
                transform = 'lin'
                freq = 'm'
            
            step = 3 if freq == 'q' else 1
            X[:, i] = _transform_series(Z[:, i], transform, freq, step)
    else:
        # Use metadata only (no config available)
        for i, series_id in enumerate(data_df.columns):
            if series_id in metadata_dict:
                transform = metadata_dict[series_id]['transformation']
                freq = metadata_dict[series_id]['frequency']
            else:
                transform = 'lin'
                freq = 'm'
            
            step = 3 if freq == 'q' else 1
            X[:, i] = _transform_series(Z[:, i], transform, freq, step)
    
    # Drop first 4 observations (transformations cause missing values)
    drop = 4
    if T > drop:
        return X[drop:], Time[drop:]
    return X, Time


def _handle_missing_series(
    data_df: pd.DataFrame,
    expected_series: List[str],
    vintage_id: int,
    strict_mode: bool = False
) -> pd.DataFrame:
    """Handle missing series in data, filling with NaN or raising error."""
    missing_series = set(expected_series) - set(data_df.columns)
    
    if missing_series:
        if strict_mode:
            raise ValueError(
                f"Missing series in vintage {vintage_id}: {sorted(missing_series)}"
            )
        logger.warning(
            f"Missing series in vintage {vintage_id}, filling with NaN: "
            f"{sorted(missing_series)}"
        )
        # Add NaN columns for missing series
        for series_id in missing_series:
            data_df[series_id] = np.nan
    
    # Reorder columns to match expected order
    ordered_series = [s for s in expected_series if s in data_df.columns]
    if len(ordered_series) != len(expected_series):
        logger.warning(
            f"Series order mismatch: expected {len(expected_series)}, got {len(ordered_series)}"
        )
    
    return data_df[ordered_series] if ordered_series else data_df


def _fetch_vintage_data(
    vintage_id: int,
    config_series_ids: List[str],
    config: Optional[DFMConfig] = None,
    config_name: Optional[str] = None,
    config_id: Optional[int] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    strict_mode: bool = False,
    client: Optional[object] = None
) -> Tuple[pd.DataFrame, pd.DatetimeIndex, pd.DataFrame, pd.DataFrame]:
    """Fetch vintage data and metadata from app.database.
    
    Uses config-specific function with blocks table ordering if available,
    otherwise falls back to general function.
    
    Parameters
    ----------
    config_name : str, optional
        Configuration name (e.g., '001-initial-spec') - uses blocks table for ordering
    config_id : int, optional, deprecated
        Configuration ID (deprecated: use config_name instead)
    """
    try:
        from app.database import (
            get_vintage_data,
            get_vintage_data_for_config,
            get_series_metadata_bulk
        )
    except ImportError:
        raise
    
    # Prefer config-specific function (uses blocks table for ordering)
    # Try config_name first, then config_id (deprecated)
    if config_name is not None:
        try:
            X, Time, Z, series_metadata_df = get_vintage_data_for_config(
                config_name=config_name,
                vintage_id=vintage_id,
                start_date=start_date,
                end_date=end_date,
                strict_mode=strict_mode,
                client=client
            )
            return _convert_to_dataframes(X, Time, Z, series_metadata_df)
        except (TypeError, AttributeError) as e:
            logger.warning(f"get_vintage_data_for_config failed ({e}), using general function")
    elif config_id is not None:
        # Deprecated: resolve config_name from config_id
        try:
            from app.database.helpers import resolve_config_name
            resolved_config_name = resolve_config_name(config_id=config_id, client=client)
            if resolved_config_name:
                logger.warning("config_id is deprecated. Use config_name instead.")
                X, Time, Z, series_metadata_df = get_vintage_data_for_config(
                    config_name=resolved_config_name,
                    vintage_id=vintage_id,
                    start_date=start_date,
                    end_date=end_date,
                    strict_mode=strict_mode,
                    client=client
                )
                return _convert_to_dataframes(X, Time, Z, series_metadata_df)
        except (TypeError, AttributeError, ImportError) as e:
            logger.warning(f"Could not resolve config_name from config_id ({e}), using general function")
    
    # Use general function with series IDs
    X, Time, Z, series_metadata_df = get_vintage_data(
        vintage_id=vintage_id,
        config_series_ids=config_series_ids,
        start_date=start_date,
        end_date=end_date,
        client=client
    )
    # Convert to DataFrame for consistency
    data_df = pd.DataFrame(X, index=Time, columns=config_series_ids)
    Z_df = pd.DataFrame(Z, index=Time, columns=config_series_ids) if Z is not None else None
    return data_df, Time, Z_df, series_metadata_df


def export_data_to_csv(
    output_path: Union[str, Path],
    vintage_id: Optional[int] = None,
    vintage_date: Optional[Union[date, str, pd.Timestamp]] = None,
    config: Optional[DFMConfig] = None,
    config_name: Optional[str] = None,
    config_id: Optional[int] = None,
    start_date: Optional[Union[date, str, pd.Timestamp]] = None,
    end_date: Optional[Union[date, str, pd.Timestamp]] = None,
    sample_start: Optional[Union[date, str, pd.Timestamp]] = None,
    strict_mode: bool = False,
    client: Optional[object] = None
) -> Path:
    """
    Export data from database to CSV file for DFM training/forecasting.
    
    This function queries the database for a specific vintage and exports
    the data to CSV format that can be used by the generic DFM module.
    
    Parameters
    ----------
    output_path : str or Path
        Path where CSV file will be saved
    vintage_id : int, optional
        Specific vintage ID to load
    vintage_date : date, str, or pd.Timestamp, optional
        Specific vintage date to load (used if vintage_id not provided)
    config : DFMConfig, optional
        Model configuration object
    config_name : str, optional
        Configuration name (preferred over config_id)
    config_id : int, optional
        Configuration ID (deprecated, use config_name)
    start_date : date, str, or pd.Timestamp, optional
        Start date for data filtering
    end_date : date, str, or pd.Timestamp, optional
        End date for data filtering
    sample_start : date, str, or pd.Timestamp, optional
        Sample start date for filtering
    strict_mode : bool, default False
        If True, raise error if series are missing
    client : object, optional
        Database client (auto-created if not provided)
    
    Returns
    -------
    Path
        Path to the created CSV file
    
    Examples
    --------
    >>> from app.adapters.adapter_database import export_data_to_csv
    >>> csv_path = export_data_to_csv(
    ...     output_path='data/vintage_1.csv',
    ...     vintage_id=1,
    ...     config_name='001-initial-spec'
    ... )
    """
    try:
        from app.database.operations import (
            get_latest_vintage_id,
            get_vintage
        )
        from app.database.helpers import (
            get_model_config_series_ids,
            resolve_config_name
        )
    except ImportError:
        raise ImportError("Database module not available. Cannot export data to CSV.")
    
    logger = logging.getLogger(__name__)
    
    # Resolve vintage
    if vintage_id is None:
        if vintage_date is not None:
            vintage_date_normalized = _normalize_date(vintage_date)
            vintage_info = get_vintage(vintage_date=vintage_date_normalized, client=client)
            if vintage_info:
                vintage_id = vintage_info['vintage_id']
            else:
                raise ValueError(f"Vintage not found for date: {vintage_date}")
        else:
            # Use latest vintage
            vintage_id = get_latest_vintage_id(client=client)
            if vintage_id is None:
                raise ValueError("No vintage found in database")
            logger.info(f"Using latest vintage_id: {vintage_id}")
    
    # Resolve config_name if needed
    if config_name is None and config_id is not None:
        try:
            config_name = resolve_config_name(config_id=config_id, client=client)
            if config_name:
                logger.warning("config_id is deprecated. Use config_name instead.")
        except ImportError:
            pass
    
    # Get series IDs from config
    if config_name:
        config_series_ids = get_model_config_series_ids(config_name, client=client)
    elif config and config.series:
        config_series_ids = [s.series_id for s in config.series]
    else:
        raise ValueError("Either config_name or config must be provided")
    
    if not config_series_ids:
        raise ValueError("No series found in configuration")
    
    # Load data from database (need DataFrames, not numpy arrays)
    # Use _fetch_vintage_data directly since we need DataFrames for CSV export
    data_df, Time, Z_df, series_metadata_df = _fetch_vintage_data(
        vintage_id=vintage_id,
        config_series_ids=config_series_ids,
        config=config,
        config_name=config_name,
        config_id=config_id,
        start_date=_normalize_date(start_date) if start_date else None,
        end_date=_normalize_date(end_date) if end_date else None,
        strict_mode=strict_mode,
        client=client
    )
    
    # Apply sample_start filter if needed
    if sample_start is not None:
        sample_start_dt = pd.to_datetime(sample_start) if isinstance(sample_start, str) else sample_start
        mask = Time >= sample_start_dt
        data_df = data_df[mask]
        Time = Time[mask]
        if Z_df is not None:
            Z_df = Z_df[mask]
    
    # Prepare CSV data: combine data with series metadata
    # Create DataFrame with series_id as index, dates as columns
    csv_data = data_df.T.copy()  # Transpose: series as rows, dates as columns
    csv_data.index.name = 'series_id'
    csv_data.reset_index(inplace=True)
    
    # Add series metadata columns
    if not series_metadata_df.empty:
        # Merge with series metadata
        csv_data = csv_data.merge(
            series_metadata_df[['series_id', 'series_name', 'frequency', 'transformation', 'category', 'units']],
            on='series_id',
            how='left'
        )
        # Reorder columns: metadata first, then date columns
        metadata_cols = ['series_id', 'series_name', 'frequency', 'transformation', 'category', 'units']
        date_cols = [col for col in csv_data.columns if col not in metadata_cols]
        csv_data = csv_data[metadata_cols + date_cols]
    else:
        # No metadata, just use series_id
        cols = ['series_id'] + [col for col in csv_data.columns if col != 'series_id']
        csv_data = csv_data[cols]
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_data.to_csv(output_path, index=False)
    
    logger.info(
        f"Exported data to CSV: {output_path}, "
        f"vintage_id={vintage_id}, series={len(config_series_ids)}, "
        f"observations={len(Time)}"
    )
    
    return output_path


def load_data_from_db(
    vintage_id: Optional[int] = None,
    vintage_date: Optional[Union[date, str, pd.Timestamp]] = None,
    config: Optional[DFMConfig] = None,
    config_name: Optional[str] = None,
    config_id: Optional[int] = None,
    config_series_ids: Optional[List[str]] = None,
    start_date: Optional[Union[date, str, pd.Timestamp]] = None,
    end_date: Optional[Union[date, str, pd.Timestamp]] = None,
    sample_start: Optional[Union[pd.Timestamp, str]] = None,
    strict_mode: bool = False,
    client: Optional[object] = None
) -> Tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Load vintage data from database and format for DFM.
    
    This function loads data from Supabase database and converts it to the format
    expected by DFM functions (numpy arrays). It applies transformations based on
    series metadata from the database.
    
    Parameters
    ----------
    vintage_id : int, optional
        Vintage ID (if None, uses vintage_date or latest vintage)
    vintage_date : date, str, or Timestamp, optional
        Vintage date (alternative to vintage_id)
    config : DFMConfig, optional
        Model configuration object (required for transformations)
    config_name : str, optional
        Model configuration name (e.g., '001-initial-spec') - uses blocks table for ordering
    config_id : int, optional, deprecated
        Model configuration ID (deprecated: use config_name instead)
    config_series_ids : List[str], optional
        List of series IDs in exact order (if None and config provided, uses config.SeriesID)
    start_date : date, str, or Timestamp, optional
        Start date for data filtering
    end_date : date, str, or Timestamp, optional
        End date for data filtering
    sample_start : Timestamp or str, optional
        Start date for estimation sample (data before this will be dropped)
    strict_mode : bool, default False
        If True, raise error for missing series. If False, fill with NaN and log warnings.
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
        
    Raises
    ------
    ValueError
        If config is required but not provided, or if strict_mode=True and series are missing
    ImportError
        If database module is not available
    """
    # Get database client
    client = _get_db_client(client)
    
    # Resolve vintage_id
    resolved_vintage_id = _resolve_vintage_id(vintage_id, vintage_date, client)
    if resolved_vintage_id is None:
        raise ValueError("Must provide either vintage_id or vintage_date")
    
    # Normalize date inputs
    normalized_start = _normalize_date(start_date)
    normalized_end = _normalize_date(end_date)
    
    # Determine series IDs
    if config_series_ids is None:
        if config is not None:
            config_series_ids = config.SeriesID
        else:
            raise ValueError("Must provide either config or config_series_ids")
    
    # Resolve config_name if only config_id provided (backward compatibility)
    if config_name is None and config_id is not None:
        try:
            from app.database.helpers import resolve_config_name
            config_name = resolve_config_name(config_id=config_id, client=client)
            if config_name:
                logger.warning("config_id is deprecated. Use config_name instead.")
        except ImportError:
            pass  # Will fall back to config_id handling in _fetch_vintage_data
    
    # Fetch data from database
    data_df, Time, Z_df, series_metadata_df = _fetch_vintage_data(
        vintage_id=resolved_vintage_id,
        config_series_ids=config_series_ids,
        config=config,
        config_name=config_name,
        config_id=config_id,
        start_date=normalized_start,
        end_date=normalized_end,
        strict_mode=strict_mode,
        client=client
    )
    
    # Handle missing series
    data_df = _handle_missing_series(
        data_df,
        expected_series=config_series_ids,
        vintage_id=resolved_vintage_id,
        strict_mode=strict_mode
    )
    
    # Convert to numpy array (raw data)
    Z = data_df.values.astype(float)
    
    # Apply sample_start filter BEFORE transformations (to avoid size mismatch)
    if sample_start is not None:
        if isinstance(sample_start, str):
            sample_start_dt = pd.to_datetime(sample_start)
        else:
            sample_start_dt = sample_start
        
        mask = Time >= sample_start_dt
        data_df = data_df[mask]
        Time = Time[mask]
        Z = data_df.values.astype(float)
    
    # Apply transformations
    X, Time = _apply_transformations_from_metadata(
        Z, Time, data_df, series_metadata_df, config
    )
    
    logger.info(
        f"Loaded data from database: vintage_id={resolved_vintage_id}, "
        f"series={len(config_series_ids)}, observations={len(Time)}"
    )
    
    # Ensure Z and Time have matching lengths
    if Z is not None and len(Z) != len(Time):
        Z = Z[:len(Time)] if len(Z) > len(Time) else Z
    
    # Convert to numpy arrays for DFM (return only X, Time, Z - 3 values)
    X_array = X.values if isinstance(X, pd.DataFrame) else X
    Z_array = Z.values if isinstance(Z, pd.DataFrame) else Z
    return X_array, Time, Z_array


def save_forecast_to_db(
    model_id: int,
    series_id: str,
    forecast_date: pd.Timestamp,
    forecast_value: float,
    vintage_id_new: Optional[int] = None,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    confidence_level: float = 0.95,
    client: Optional[object] = None
) -> None:
    """Save forward forecast to database.
    
    Parameters
    ----------
    model_id : int
        Model ID (required by schema, use 0 if models not in DB)
    series_id : str
        Series ID to forecast
    forecast_date : pd.Timestamp
        Target date for forecast (future period)
    forecast_value : float
        Forecast value
    vintage_id_new : int, optional
        Vintage ID used for forecasting (only new vintage, no old vintage)
    lower_bound : float, optional
        Lower confidence bound (not computed yet)
    upper_bound : float, optional
        Upper confidence bound (not computed yet)
    confidence_level : float, default=0.95
        Confidence level for bounds
    client : object, optional
        Supabase client. If None, will get from database module.
    
    Notes
    -----
    This saves forward forecasts (run_type='forecast') with:
    - vintage_id_new only (vintage_id_old = NULL)
    - Multiple rows per run (one per forecast period)
    """
    if not np.isfinite(forecast_value):
        logger.warning(
            f"Cannot save forecast to database: forecast_value is invalid "
            f"(NaN or Inf) for series {series_id}. Skipping database save."
        )
        return
    
    # Validate forecast_date
    try:
        if isinstance(forecast_date, str):
            forecast_date = pd.to_datetime(forecast_date)
        elif not isinstance(forecast_date, pd.Timestamp):
            forecast_date = pd.Timestamp(forecast_date)
    except Exception as e:
        logger.warning(
            f"Cannot save forecast to database: invalid forecast_date "
            f"for series {series_id}: {e}. Skipping database save."
        )
        return
    
    try:
        from app.database import get_client, save_forecast
        import os
        db_client = client or _get_db_client(client)
        
        # Get GitHub run ID if available
        github_run_id = os.getenv('GITHUB_RUN_ID')
        
        # Save forecast (run_type='forecast', vintage_id_old=NULL)
        save_forecast(
            model_id=model_id,
            series_id=series_id,
            forecast_date=forecast_date.date(),
            forecast_value=forecast_value,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=confidence_level,
            run_type='forecast',  # ← FORECAST (not nowcast)
            vintage_id_old=None,  # ← NULL for forecasts
            vintage_id_new=vintage_id_new,
            github_run_id=github_run_id,
            metadata_json=None,  # No news decomposition for forecasts
            client=db_client
        )
        
        logger.info(
            f"Saved forecast to database: series={series_id}, date={forecast_date.date()}, "
            f"value={forecast_value:.4f}, period={forecast_date.date()}"
        )
        
    except ImportError:
        logger.warning("Database module not available. Cannot save forecast to database.")
    except Exception as e:
        logger.warning(
            f"Failed to save forecast to database for series {series_id}: {e}. "
            f"Continuing without saving."
        )


def save_nowcast_to_db(
    model_id: Optional[int],
    series: str,
    forecast_date: pd.Timestamp,
    forecast_value: float,
    old_forecast_value: Optional[float] = None,
    impact_revisions: Optional[float] = None,
    impact_releases: Optional[float] = None,
    total_impact: Optional[float] = None,
    vintage_old: Optional[str] = None,
    vintage_new: Optional[str] = None,
    Res: Optional[Any] = None,
    confidence_level: float = 0.95,
    client: Optional[object] = None
) -> None:
    """Robustly save nowcast to database with comprehensive error handling.
    
    This function attempts to save nowcast values to the database but will not
    interrupt execution if saving fails. All errors are logged as warnings.
    
    Parameters
    ----------
    model_id : int, optional
        Model ID for database. If None, will attempt to extract from Res or skip.
    series : str
        Series ID for the nowcast
    forecast_date : pd.Timestamp
        Date of the forecast
    forecast_value : float
        New nowcast value
    old_forecast_value : float, optional
        Old nowcast value (for comparison)
    impact_revisions : float, optional
        Impact from data revisions
    impact_releases : float, optional
        Impact from data releases
    total_impact : float, optional
        Total impact (revisions + releases)
    vintage_old : str, optional
        Old vintage date
    vintage_new : str, optional
        New vintage date
    Res : DFMResult or dict, optional
        DFM results - may contain model_id
    confidence_level : float, default=0.95
        Confidence level for bounds (currently not computed)
    client : Client, optional
        Supabase client instance
    """
    # Note: model_id is required by forecasts table schema (NOT NULL)
    # But models are stored as pkl files, not in database
    # If model_id is None, we'll use 0 as placeholder
    # (This is acceptable since model_id has no FK constraint)
    
    if not np.isfinite(forecast_value):
        logger.warning(
            f"Cannot save nowcast to database: forecast_value is invalid "
            f"(NaN or Inf) for series {series}. Skipping database save."
        )
        return
    
    # Validate forecast_date
    try:
        if isinstance(forecast_date, str):
            forecast_date = pd.to_datetime(forecast_date)
        elif not isinstance(forecast_date, pd.Timestamp):
            forecast_date = pd.Timestamp(forecast_date)
    except Exception as e:
        logger.warning(
            f"Cannot save nowcast to database: invalid forecast_date "
            f"for series {series}: {e}. Skipping database save."
        )
        return
    
    # Attempt to save to database with comprehensive error handling
    try:
        from app.database import get_client, save_forecast, get_latest_vintage_id
        import os
        db_client = client or get_client()
        
        # Note: model_id is required by forecasts table schema, but models are pkl files
        # Use 0 as placeholder if model_id is None (models not in DB)
        effective_model_id = model_id if model_id is not None else 0
        
        # Resolve vintage IDs from vintage dates
        vintage_id_old_resolved = None
        vintage_id_new_resolved = None
        
        if vintage_old:
            try:
                vintage_id_old_resolved = get_latest_vintage_id(
                    vintage_date=_normalize_date(vintage_old),
                    client=db_client
                )
            except Exception as e:
                logger.debug(f"Could not resolve vintage_id_old from {vintage_old}: {e}")
        
        if vintage_new:
            try:
                vintage_id_new_resolved = get_latest_vintage_id(
                    vintage_date=_normalize_date(vintage_new),
                    client=db_client
                )
            except Exception as e:
                logger.debug(f"Could not resolve vintage_id_new from {vintage_new}: {e}")
        
        # Prepare metadata JSON with news decomposition info
        metadata = {}
        if old_forecast_value is not None:
            metadata['old_forecast_value'] = old_forecast_value
        if impact_revisions is not None:
            metadata['impact_revisions'] = impact_revisions
        if impact_releases is not None:
            metadata['impact_releases'] = impact_releases
        if total_impact is not None:
            metadata['total_impact'] = total_impact
        
        # Get GitHub run ID if available
        github_run_id = os.getenv('GITHUB_RUN_ID')
        
        # Save forecast with all metadata
        save_forecast(
            model_id=effective_model_id,
            series_id=series,
            forecast_date=forecast_date.date(),
            forecast_value=forecast_value,
            lower_bound=None,  # TODO: Compute confidence intervals
            upper_bound=None,
            confidence_level=confidence_level,
            run_type='nowcast',
            vintage_id_old=vintage_id_old_resolved,
            vintage_id_new=vintage_id_new_resolved,
            github_run_id=github_run_id,
            metadata_json=metadata if metadata else None,
            client=db_client
        )
        
        logger.info(
            f"Saved nowcast to database: series={series}, date={forecast_date.date()}, "
            f"value={forecast_value:.4f}"
        )
        
    except ImportError:
        logger.warning("Database module not available. Cannot save nowcast to database.")
    except Exception as e:
        logger.warning(
            f"Failed to save nowcast to database for series {series}: {e}. "
            f"Continuing execution."
        )



def save_blocks_to_db(
    config: DFMConfig,
    config_name: str,
    client: Optional[object] = None
) -> None:
    """Save block assignments from DFMConfig to database blocks table.
    
    This function extracts block information from a DFMConfig object and saves it
    to the blocks table. The config_name should be derived from the CSV filename
    (e.g., '001_initial_spec.csv' → '001-initial-spec').
    
    Parameters
    ----------
    config : DFMConfig
        DFM configuration with series and block information
    config_name : str
        Configuration name identifier (e.g., '001-initial-spec')
        Derived from CSV filename by replacing underscores with hyphens and removing .csv
    client : object, optional
        Database client. If None, will attempt to get from database module.
    
    Notes
    -----
    - Deletes existing blocks for this config_name before inserting new ones
    - Only saves blocks where the value is 1 (series loads on that block)
    - series_order is the index of the series in the config.series list
    """
    try:
        db_client = _get_db_client(client)
        
        if not config.block_names:
            logger.warning(f"No block_names in config. Skipping block save for {config_name}.")
            return
        
        # Prepare block records
        block_records = []
        for series_order, series_cfg in enumerate(config.series):
            # Get blocks array (list of 0/1 values)
            blocks = getattr(series_cfg, 'blocks', [])
            if not blocks:
                continue
            
            # Save each block where value is 1
            for block_idx, block_name in enumerate(config.block_names):
                if block_idx < len(blocks) and blocks[block_idx] == 1:
                    block_records.append({
                        'config_name': config_name,
                        'series_id': series_cfg.series_id,
                        'block_name': block_name,
                        'series_order': series_order
                    })
        
        if not block_records:
            logger.warning(f"No block assignments found in config. Skipping block save for {config_name}.")
            return
        
        # Delete existing blocks for this config_name
        try:
            db_client.table('blocks').delete().eq('config_name', config_name).execute()
            logger.debug(f"Deleted existing blocks for config_name={config_name}")
        except Exception as e:
            logger.warning(f"Could not delete existing blocks for {config_name}: {e}")
        
        # Insert new blocks in batches
        batch_size = 100
        total_inserted = 0
        for i in range(0, len(block_records), batch_size):
            batch = block_records[i:i + batch_size]
            try:
                db_client.table('blocks').insert(batch).execute()
                total_inserted += len(batch)
            except Exception as e:
                logger.error(f"Failed to insert block batch {i//batch_size + 1}: {e}")
                raise
        
        logger.info(
            f"Saved {total_inserted} block assignments to database for config_name={config_name}"
        )
        
    except ImportError:
        logger.warning("Database module not available. Cannot save blocks to database.")
    except Exception as e:
        logger.error(f"Failed to save blocks to database for config_name={config_name}: {e}")
        raise


# ============================================================================
# Supabase Storage Functions (Model Weights)
# ============================================================================

def upload_model_weights_to_storage(
    model_weights: Dict[str, Any],
    filename: str,
    bucket_name: str = "model-weights",
    client: Optional[object] = None
) -> str:
    """Upload model weights pickle file to Supabase storage.
    
    Parameters
    ----------
    model_weights : Dict[str, Any]
        Model weights dictionary to pickle and upload
    filename : str
        Filename in storage (e.g., "dfm_2025-11-07.pkl")
    bucket_name : str, default="model-weights"
        Supabase storage bucket name
    client : object, optional
        Supabase client. If None, will get from database module.
    
    Returns
    -------
    str
        Public URL of uploaded file (if public) or path
    
    Raises
    ------
    ImportError
        If database module not available
    Exception
        If upload fails
    """
    try:
        db_client = _get_db_client(client)
        
        # Serialize model weights to bytes
        pickled_data = pickle.dumps(model_weights)
        
        # Upload to Supabase storage
        # Supabase storage.upload expects file as bytes or file-like object
        response = db_client.storage.from_(bucket_name).upload(
            path=filename,
            file=pickled_data,
            file_options={"content-type": "application/octet-stream", "upsert": "true"}
        )
        
        logger.info(f"Uploaded model weights to storage: {bucket_name}/{filename}")
        
        # Get public URL
        try:
            public_url = db_client.storage.from_(bucket_name).get_public_url(filename)
            return public_url
        except Exception:
            # If public URL not available, return path
            return f"{bucket_name}/{filename}"
            
    except ImportError:
        raise ImportError("Database module not available. Cannot upload to storage.")
    except Exception as e:
        logger.error(f"Failed to upload model weights to storage: {e}")
        raise


def download_model_weights_from_storage(
    filename: str,
    bucket_name: str = "model-weights",
    client: Optional[object] = None
) -> Optional[Dict[str, Any]]:
    """Download model weights pickle file from Supabase storage.
    
    Parameters
    ----------
    filename : str
        Filename in storage (e.g., "dfm_2025-11-07.pkl")
    bucket_name : str, default="model-weights"
        Supabase storage bucket name
    client : object, optional
        Supabase client. If None, will get from database module.
    
    Returns
    -------
    Dict[str, Any] or None
        Model weights dictionary, or None if file not found
    
    Raises
    ------
    ImportError
        If database module not available
    Exception
        If download fails
    """
    try:
        db_client = _get_db_client(client)
        
        # Download from Supabase storage
        response = db_client.storage.from_(bucket_name).download(filename)
        
        if response is None:
            logger.warning(f"Model weights file not found in storage: {bucket_name}/{filename}")
            return None
        
        # Deserialize pickle data
        model_weights = pickle.loads(response)
        
        logger.info(f"Downloaded model weights from storage: {bucket_name}/{filename}")
        return model_weights
        
    except ImportError:
        raise ImportError("Database module not available. Cannot download from storage.")
    except FileNotFoundError:
        logger.warning(f"Model weights file not found: {bucket_name}/{filename}")
        return None
    except Exception as e:
        logger.error(f"Failed to download model weights from storage: {e}")
        raise


def cleanup_old_model_weights(
    keep_latest: int = 3,
    bucket_name: str = "model-weights",
    client: Optional[object] = None
) -> Dict[str, Any]:
    """Clean up old model weights, keeping only the latest N files.
    
    Parameters
    ----------
    keep_latest : int, default=3
        Number of latest model weight files to keep
    bucket_name : str, default="model-weights"
        Supabase storage bucket name
    client : object, optional
        Supabase client. If None, will get from database module.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with cleanup results:
        - total_files: Total number of files found
        - kept_files: List of files kept
        - deleted_files: List of files deleted
        - deleted_count: Number of files deleted
    
    Raises
    ------
    ImportError
        If database module not available
    Exception
        If cleanup fails
    """
    try:
        db_client = _get_db_client(client)
        
        # List all files in the bucket
        files = db_client.storage.from_(bucket_name).list()
        
        if not files:
            logger.debug(f"No files found in storage bucket: {bucket_name}")
            return {
                'total_files': 0,
                'kept_files': [],
                'deleted_files': [],
                'deleted_count': 0
            }
        
        # Filter for .pkl files only
        pkl_files = [f for f in files if f.get('name', '').endswith('.pkl')]
        
        if len(pkl_files) <= keep_latest:
            logger.debug(f"Only {len(pkl_files)} model weight files found, keeping all (limit: {keep_latest})")
            return {
                'total_files': len(pkl_files),
                'kept_files': [f['name'] for f in pkl_files],
                'deleted_files': [],
                'deleted_count': 0
            }
        
        # Sort files by creation time (newest first)
        # Use updated_at if available, otherwise use name (which contains date)
        def get_sort_key(file_info):
            # Try to get updated_at timestamp
            updated_at = file_info.get('updated_at')
            if updated_at:
                try:
                    from datetime import datetime
                    # Handle different types: string, datetime, or already parsed
                    if isinstance(updated_at, datetime):
                        return updated_at.timestamp()
                    elif isinstance(updated_at, str):
                        if updated_at.endswith('Z'):
                            return datetime.fromisoformat(updated_at.replace('Z', '+00:00')).timestamp()
                        else:
                            return datetime.fromisoformat(updated_at).timestamp()
                    else:
                        # Try to convert to datetime
                        return pd.to_datetime(updated_at).timestamp()
                except Exception:
                    pass
            
            # Fallback: extract date from filename (e.g., "dfm_2025-11-07.pkl" -> 2025-11-07)
            filename = file_info.get('name', '')
            import re
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
            if date_match:
                try:
                    from datetime import datetime
                    return datetime.strptime(date_match.group(1), '%Y-%m-%d').timestamp()
                except Exception:
                    pass
            
            # Last resort: use name for alphabetical sort
            return 0
        
        pkl_files.sort(key=get_sort_key, reverse=True)
        
        # Keep latest N files, delete the rest
        files_to_keep = pkl_files[:keep_latest]
        files_to_delete = pkl_files[keep_latest:]
        
        kept_filenames = [f['name'] for f in files_to_keep]
        deleted_filenames = [f['name'] for f in files_to_delete]
        
        # Delete old files
        deleted_count = 0
        for file_info in files_to_delete:
            filename = file_info['name']
            try:
                db_client.storage.from_(bucket_name).remove([filename])
                deleted_count += 1
                logger.debug(f"Deleted old model weight file: {filename}")
            except Exception as e:
                logger.warning(f"Failed to delete model weight file {filename}: {e}")
        
        logger.info(
            f"Cleaned up model weights: kept {len(kept_filenames)} latest files, "
            f"deleted {deleted_count} old files"
        )
        
        return {
            'total_files': len(pkl_files),
            'kept_files': kept_filenames,
            'deleted_files': deleted_filenames,
            'deleted_count': deleted_count
        }
        
    except ImportError:
        raise ImportError("Database module not available. Cannot cleanup storage.")
    except Exception as e:
        logger.error(f"Failed to cleanup old model weights: {e}")
        raise


def cleanup_old_models(
    keep_latest: int = 3,
    client: Optional[object] = None
) -> Dict[str, Any]:
    """Clean up old models, keeping only the latest N model_ids.
    
    This deletes old factors, which cascades to factor_values and
    factor_loadings via foreign key constraints.
    
    Parameters
    ----------
    keep_latest : int, default=3
        Number of latest models to keep
    client : object, optional
        Supabase client. If None, will get from database module.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with cleanup results:
        - total_models: Total number of models found
        - kept_models: List of model_ids kept
        - deleted_models: List of model_ids deleted
        - deleted_count: Number of models deleted
        - deleted_factors: Number of factors deleted (cascade)
        - deleted_factor_values: Estimated factor_values deleted
        - deleted_factor_loadings: Estimated factor_loadings deleted
    
    Raises
    ------
    ImportError
        If database module not available
    Exception
        If cleanup fails
    """
    try:
        db_client = _get_db_client(client)
        
        # Get all unique model_ids with their latest creation time
        # Use a subquery to get MAX(created_at) per model_id, or use id as fallback
        factors_result = db_client.table('factors').select('id, model_id, created_at').order('created_at', desc=True).execute()
        
        if not factors_result.data:
            logger.debug("No factors found in database")
            return {
                'total_models': 0,
                'kept_models': [],
                'deleted_models': [],
                'deleted_count': 0,
                'deleted_factors': 0,
                'deleted_factor_values': 0,
                'deleted_factor_loadings': 0
            }
        
        # Group by model_id and find latest created_at for each
        # Also track factor_id as fallback for sorting
        from collections import defaultdict
        from datetime import datetime
        
        model_ages = defaultdict(lambda: (datetime.min, -1))  # (created_at, max_factor_id)
        for row in factors_result.data:
            model_id = row['model_id']
            factor_id = row.get('id', -1)
            created_at_raw = row.get('created_at')
            
            created_at = datetime.min
            if created_at_raw:
                try:
                    # Handle different types: string, datetime, or already parsed
                    if isinstance(created_at_raw, datetime):
                        created_at = created_at_raw
                    elif isinstance(created_at_raw, str):
                        # Parse timestamp (handle both with and without timezone)
                        if 'T' in created_at_raw:
                            if created_at_raw.endswith('Z'):
                                created_at = datetime.fromisoformat(created_at_raw.replace('Z', '+00:00'))
                            else:
                                created_at = datetime.fromisoformat(created_at_raw)
                        else:
                            created_at = datetime.fromisoformat(created_at_raw)
                    else:
                        # Try to convert to datetime
                        created_at = pd.to_datetime(created_at_raw).to_pydatetime()
                except Exception as e:
                    logger.warning(f"Could not parse created_at for model_id {model_id}: {e}, type: {type(created_at_raw)}")
                    # Use current time as fallback
                    created_at = datetime.now()
            
            # Keep the latest timestamp and highest factor_id for each model_id
            current_max_time, current_max_id = model_ages[model_id]
            if created_at > current_max_time or (created_at == current_max_time and factor_id > current_max_id):
                model_ages[model_id] = (created_at, factor_id)
        
        # Convert to list of (model_id, latest_created_at, max_factor_id) tuples
        model_list = [(model_id, created_at, max_id) for model_id, (created_at, max_id) in model_ages.items()]
        
        if len(model_list) <= keep_latest:
            logger.debug(f"Only {len(model_list)} models found, keeping all (limit: {keep_latest})")
            return {
                'total_models': len(model_list),
                'kept_models': [model_id for model_id, _, _ in model_list],
                'deleted_models': [],
                'deleted_count': 0,
                'deleted_factors': 0,
                'deleted_factor_values': 0,
                'deleted_factor_loadings': 0
            }
        
        # Sort by created_at (newest first), then by factor_id as tiebreaker
        model_list.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        # Special case: if all factors have the same model_id, use factor_id-based cleanup instead
        unique_model_ids = set(model_id for model_id, _, _ in model_list)
        if len(unique_model_ids) == 1:
            # All factors belong to the same model_id - use factor_id-based cleanup
            logger.warning(f"All factors belong to the same model_id {list(unique_model_ids)[0]}. Using factor_id-based cleanup.")
            # Get all factors sorted by id (newest first, assuming id is auto-increment)
            all_factors = db_client.table('factors').select('id, model_id').order('id', desc=True).execute()
            if all_factors.data:
                total_factors = len(all_factors.data)
                if total_factors <= keep_latest:
                    logger.debug(f"Only {total_factors} factors found, keeping all (limit: {keep_latest})")
                    return {
                        'total_models': 1,
                        'kept_models': list(unique_model_ids),
                        'deleted_models': [],
                        'deleted_count': 0,
                        'deleted_factors': 0,
                        'deleted_factor_values': 0,
                        'deleted_factor_loadings': 0
                    }
                
                # Keep latest N factors by id
                factors_to_keep = all_factors.data[:keep_latest]
                factors_to_delete = all_factors.data[keep_latest:]
                
                kept_factor_ids = [f['id'] for f in factors_to_keep]
                deleted_factor_ids = [f['id'] for f in factors_to_delete]
                
                # Delete old factors by id (CASCADE will handle factor_values and factor_loadings)
                deleted_count = 0
                for factor_id in deleted_factor_ids:
                    try:
                        db_client.table('factors').delete().eq('id', factor_id).execute()
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete factor id={factor_id}: {e}")
                
                # Count factor_values and factor_loadings that will be deleted
                factor_values_count = 0
                factor_loadings_count = 0
                if deleted_factor_ids:
                    fv_result = db_client.table('factor_values').select('id', count='exact').in_('factor_id', deleted_factor_ids).execute()
                    factor_values_count = fv_result.count if hasattr(fv_result, 'count') else 0
                    
                    fl_result = db_client.table('factor_loadings').select('factor_id, series_id', count='exact').in_('factor_id', deleted_factor_ids).execute()
                    factor_loadings_count = fl_result.count if hasattr(fl_result, 'count') else 0
                
                logger.info(
                    f"Cleaned up factors by id: kept {len(kept_factor_ids)} latest factors, "
                    f"deleted {deleted_count} old factors (~{factor_values_count} factor_values, ~{factor_loadings_count} factor_loadings)"
                )
                
                return {
                    'total_models': 1,
                    'kept_models': list(unique_model_ids),
                    'deleted_models': [],
                    'deleted_count': deleted_count,
                    'deleted_factors': deleted_count,
                    'deleted_factor_values': factor_values_count,
                    'deleted_factor_loadings': factor_loadings_count
                }
        
        # Normal case: multiple model_ids - use model_id-based cleanup
        # Split into kept and deleted
        models_to_keep = model_list[:keep_latest]
        models_to_delete = model_list[keep_latest:]
        
        kept_model_ids = [model_id for model_id, _, _ in models_to_keep]
        deleted_model_ids = [model_id for model_id, _, _ in models_to_delete]
        
        # Count factors that will be deleted (for reporting)
        factors_to_delete = db_client.table('factors').select('id', count='exact').in_('model_id', deleted_model_ids).execute()
        factor_count = factors_to_delete.count if hasattr(factors_to_delete, 'count') else 0
        
        # Estimate factor_values and factor_loadings that will be deleted
        # (These will be deleted via CASCADE, so we estimate based on factors)
        factor_ids_result = db_client.table('factors').select('id').in_('model_id', deleted_model_ids).execute()
        factor_ids = [row['id'] for row in factor_ids_result.data] if factor_ids_result.data else []
        
        factor_values_count = 0
        factor_loadings_count = 0
        if factor_ids:
            # Count factor_values
            fv_result = db_client.table('factor_values').select('id', count='exact').in_('factor_id', factor_ids).execute()
            factor_values_count = fv_result.count if hasattr(fv_result, 'count') else 0
            
            # Count factor_loadings
            fl_result = db_client.table('factor_loadings').select('factor_id, series_id', count='exact').in_('factor_id', factor_ids).execute()
            factor_loadings_count = fl_result.count if hasattr(fl_result, 'count') else 0
        
        # Delete old models (this will CASCADE to factor_values and factor_loadings)
        deleted_count = 0
        for model_id in deleted_model_ids:
            try:
                # Delete factors for this model_id (CASCADE will handle factor_values and factor_loadings)
                db_client.table('factors').delete().eq('model_id', model_id).execute()
                deleted_count += 1
                logger.debug(f"Deleted factors for model_id={model_id} (cascade deleted factor_values and factor_loadings)")
            except Exception as e:
                logger.warning(f"Failed to delete factors for model_id={model_id}: {e}")
        
        logger.info(
            f"Cleaned up old models: kept {len(kept_model_ids)} latest models, "
            f"deleted {deleted_count} old models ({factor_count} factors, "
            f"~{factor_values_count} factor_values, ~{factor_loadings_count} factor_loadings)"
        )
        
        return {
            'total_models': len(model_list),
            'kept_models': kept_model_ids,
            'deleted_models': deleted_model_ids,
            'deleted_count': deleted_count,
            'deleted_factors': factor_count,
            'deleted_factor_values': factor_values_count,
            'deleted_factor_loadings': factor_loadings_count
        }
        
    except ImportError:
        raise ImportError("Database module not available. Cannot cleanup models.")
    except Exception as e:
        logger.error(f"Failed to cleanup old models: {e}")
        raise


def csv_spec_to_hydra_config(
    config: DFMConfig,
    dfm_config_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convert DFMConfig from CSV spec to Hydra-compatible config dictionary.
    
    This function translates a DFMConfig loaded from CSV into a format that can be
    merged with Hydra configuration. It extracts DFM-related parameters that can be
    configured via Hydra.
    
    Parameters
    ----------
    config : DFMConfig
        DFM configuration loaded from CSV spec file
    dfm_config_overrides : dict, optional
        Optional DFM config overrides (e.g., from CSV metadata or user preferences)
        Keys: ar_lag, factors_per_block, threshold, max_iter, nan_method, nan_k, clock,
        and numerical stability parameters (clip_ar_coefficients, clip_data_values,
        use_regularization, use_damped_updates, etc.) for dfm-python 0.1.5+
    
    Returns
    -------
    dict
        Dictionary with 'model' and 'dfm' keys containing Hydra-compatible config
        that can be merged with existing Hydra config via OmegaConf.merge()
    
    Examples
    --------
    >>> from dfm_python import load_config
    >>> from app.adapters.adapter_database import csv_spec_to_hydra_config
    >>> config = load_config('src/spec/001_initial_spec.csv')
    >>> hydra_config = csv_spec_to_hydra_config(config)
    >>> # Merge with existing Hydra config
    >>> from omegaconf import OmegaConf
    >>> merged = OmegaConf.merge(existing_config, hydra_config)
    """
    hydra_config = {
        'model': {},
        'dfm': {}
    }
    
    # Extract model config (factors_per_block from DFMConfig)
    if config.factors_per_block is not None:
        hydra_config['model']['factors_per_block'] = config.factors_per_block
    
    # Extract DFM config (can come from CSV metadata or overrides)
    dfm_defaults = {
        'ar_lag': 1,
        'factors_per_block': None,
        'threshold': 1e-5,
        'max_iter': 5000,
        'nan_method': 2,
        'nan_k': 3,
        'clock': 'm',  # Default to monthly clock
        # Numerical stability parameters (dfm-python 0.1.5+)
        'clip_ar_coefficients': True,
        'ar_clip_min': -0.99,
        'ar_clip_max': 0.99,
        'warn_on_ar_clip': True,
        'clip_data_values': True,
        'data_clip_threshold': 100.0,
        'warn_on_data_clip': True,
        'use_regularization': True,
        'regularization_scale': 1e-6,
        'min_eigenvalue': 1e-8,
        'max_eigenvalue': 1e6,
        'warn_on_regularization': True,
        'use_damped_updates': True,
        'damping_factor': 0.8,
        'warn_on_damped_update': True
    }
    
    # Start with defaults
    hydra_config['dfm'] = dfm_defaults.copy()
    
    # Apply overrides if provided
    if dfm_config_overrides:
        hydra_config['dfm'].update(dfm_config_overrides)
    
    # If factors_per_block is in model config but not in dfm config, use model's
    if config.factors_per_block is not None and hydra_config['dfm']['factors_per_block'] is None:
        hydra_config['dfm']['factors_per_block'] = config.factors_per_block
    
    return hydra_config


def list_spec_csv_files_from_storage(
    bucket_name: str = "spec",
    client: Optional[object] = None
) -> List[str]:
    """List all CSV files in the spec storage bucket.
    
    Parameters
    ----------
    bucket_name : str, default="spec"
        Supabase storage bucket name for spec files
    client : object, optional
        Supabase client. If None, will get from database module.
    
    Returns
    -------
    List[str]
        List of CSV filenames in the bucket, sorted by filename
    
    Raises
    ------
    ImportError
        If database module not available
    Exception
        If listing fails
    """
    try:
        db_client = _get_db_client(client)
        
        # List files in the bucket
        files = db_client.storage.from_(bucket_name).list()
        
        if not files:
            logger.debug(f"No files found in storage bucket: {bucket_name}")
            return []
        
        # Filter for CSV files only
        csv_files = [f['name'] for f in files if f.get('name', '').endswith('.csv')]
        
        logger.debug(f"Found {len(csv_files)} CSV files in storage bucket: {csv_files}")
        return sorted(csv_files)
        
    except ImportError:
        raise ImportError("Database module not available. Cannot list storage files.")
    except Exception as e:
        logger.debug(f"Failed to list files in storage bucket: {e}")
        return []


def get_latest_spec_csv_filename(
    bucket_name: str = "spec",
    client: Optional[object] = None
) -> Optional[str]:
    """Get the latest spec CSV filename (highest number prefix) from storage.
    
    Looks for files matching pattern: NNN_*.csv (e.g., 001_initial_spec.csv, 002_updated_spec.csv)
    Returns the one with the highest NNN number.
    
    Parameters
    ----------
    bucket_name : str, default="spec"
        Supabase storage bucket name for spec files
    client : object, optional
        Supabase client. If None, will get from database module.
    
    Returns
    -------
    str or None
        Latest spec CSV filename, or None if no CSV files found
    
    Raises
    ------
    ImportError
        If database module not available
    """
    try:
        csv_files = list_spec_csv_files_from_storage(bucket_name=bucket_name, client=client)
        
        if not csv_files:
            return None
        
        # Extract numbers from filenames (pattern: NNN_*.csv)
        file_numbers = []
        for filename in csv_files:
            match = re.match(r'^(\d+)_', filename)
            if match:
                number = int(match.group(1))
                file_numbers.append((number, filename))
        
        if not file_numbers:
            # No numbered files found, return the last one alphabetically
            logger.warning("No numbered spec files found (pattern: NNN_*.csv), using last file alphabetically")
            return csv_files[-1]
        
        # Sort by number (descending) and return the highest
        file_numbers.sort(key=lambda x: x[0], reverse=True)
        latest_filename = file_numbers[0][1]
        
        logger.info(f"Latest spec CSV file: {latest_filename} (number: {file_numbers[0][0]})")
        return latest_filename
        
    except ImportError:
        raise ImportError("Database module not available. Cannot get latest spec filename.")
    except Exception as e:
        logger.debug(f"Failed to get latest spec filename: {e}")
        return None


def upload_spec_csv_to_storage(
    file_path: str,
    filename: Optional[str] = None,
    bucket_name: str = "spec",
    client: Optional[object] = None
) -> str:
    """Upload spec CSV file to Supabase storage.
    
    Parameters
    ----------
    file_path : str
        Path to local CSV file to upload
    filename : str, optional
        Filename in storage (default: basename of file_path)
    bucket_name : str, default="spec"
        Supabase storage bucket name for spec files
    client : object, optional
        Supabase client. If None, will get from database module.
    
    Returns
    -------
    str
        Public URL of uploaded file (if public) or path
    
    Raises
    ------
    ImportError
        If database module not available
    Exception
        If upload fails
    """
    try:
        db_client = _get_db_client(client)
        
        # Read file content
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        # Use provided filename or basename
        if filename is None:
            filename = Path(file_path).name
        
        # Upload to Supabase storage
        response = db_client.storage.from_(bucket_name).upload(
            path=filename,
            file=file_content,
            file_options={"content-type": "text/csv", "upsert": "true"}
        )
        
        logger.info(f"Uploaded spec CSV to storage: {bucket_name}/{filename}")
        
        # Get public URL
        try:
            public_url = db_client.storage.from_(bucket_name).get_public_url(filename)
            return public_url
        except Exception:
            # If public URL not available, return path
            return f"{bucket_name}/{filename}"
            
    except ImportError:
        raise ImportError("Database module not available. Cannot upload to storage.")
    except Exception as e:
        logger.error(f"Failed to upload spec CSV to storage: {e}")
        raise


def download_spec_csv_from_storage(
    filename: str,
    bucket_name: str = "spec",
    client: Optional[object] = None
) -> Optional[bytes]:
    """Download spec CSV file from Supabase storage.
    
    Parameters
    ----------
    filename : str
        Filename in storage (e.g., "001_initial_spec.csv")
    bucket_name : str, default="spec"
        Supabase storage bucket name for spec files
    client : object, optional
        Supabase client. If None, will get from database module.
    
    Returns
    -------
    bytes or None
        CSV file content as bytes, or None if file not found
    
    Raises
    ------
    ImportError
        If database module not available
    Exception
        If download fails
    """
    try:
        db_client = _get_db_client(client)
        
        # Download from Supabase storage
        response = db_client.storage.from_(bucket_name).download(filename)
        
        if response is None:
            logger.debug(f"Spec CSV file not found in storage: {bucket_name}/{filename}")
            return None
        
        logger.info(f"Downloaded spec CSV from storage: {bucket_name}/{filename}")
        return response
        
    except ImportError:
        raise ImportError("Database module not available. Cannot download from storage.")
    except FileNotFoundError:
        logger.debug(f"Spec CSV file not found: {bucket_name}/{filename}")
        return None
    except Exception as e:
        logger.debug(f"Failed to download spec CSV from storage: {e}")
        return None


# ============================================================================
# Factor and Factor Loading Database Functions
# ============================================================================

def save_factors_to_db(
    Res: Any,  # DFMResult
    model_id: int,
    config: DFMConfig,
    vintage_id: int,
    Time: pd.DatetimeIndex,
    client: Optional[object] = None
) -> None:
    """Save factors, factor values, and factor loadings to database for frontend visualization.
    
    This function extracts factor information from DFMResult and saves it to:
    - factors: Factor metadata (name, description, index, block_name)
    - factor_values: Time series of factor estimates
    - factor_loadings: Factor-variable loading matrix
    
    Parameters
    ----------
    Res : DFMResult
        DFM estimation results containing factors (Z), loadings (C), etc.
    model_id : int
        Model identifier (used as model_id in factors table)
    config : DFMConfig
        Model configuration with SeriesID, block_names, etc.
    vintage_id : int
        Vintage ID for factor values
    Time : pd.DatetimeIndex
        Time index for factor values
    client : object, optional
        Database client. If None, will get from database module.
    
    Notes
    -----
    - Deletes existing factors for this model_id before inserting new ones
    - Factor names are derived from block_names if available
    - Factor index is 0-based (column index in Z matrix)
    """
    try:
        db_client = _get_db_client(client)
        
        # Get factor count
        n_factors = Res.Z.shape[1] if hasattr(Res, 'Z') and Res.Z is not None else 0
        if n_factors == 0:
            logger.warning("No factors found in DFMResult. Skipping factor save.")
            return
        
        # Get block names from config
        block_names = getattr(config, 'block_names', [])
        blocks = getattr(config, 'Blocks', None)
        
        # Delete existing factors for this model_id (cascade will delete factor_values and factor_loadings)
        try:
            db_client.table('factors').delete().eq('model_id', model_id).execute()
            logger.debug(f"Deleted existing factors for model_id={model_id}")
        except Exception as e:
            logger.warning(f"Could not delete existing factors for model_id={model_id}: {e}")
        
        # Prepare factor records
        factor_records = []
        factor_id_map = {}  # Map factor_index -> factor_id (will be set after insert)
        
        # Determine factor names based on block structure
        # First factors are typically global, then block-specific
        factor_idx = 0
        for block_idx, block_name in enumerate(block_names):
            # Get number of factors for this block (from r parameter)
            n_block_factors = int(Res.r[block_idx]) if hasattr(Res, 'r') and block_idx < len(Res.r) else 1
            
            for f in range(n_block_factors):
                if factor_idx >= n_factors:
                    break
                
                # Factor name
                if block_idx == 0 or block_name == 'Global':
                    factor_name = f"Global Factor {f+1}" if n_block_factors > 1 else "Global Factor"
                else:
                    factor_name = f"{block_name} Factor {f+1}" if n_block_factors > 1 else f"{block_name} Factor"
                
                factor_records.append({
                    'model_id': model_id,
                    'name': factor_name,
                    'description': f"Factor {factor_idx} from {block_name} block",
                    'factor_index': factor_idx,
                    'block_name': None if block_idx == 0 or block_name == 'Global' else block_name
                })
                factor_idx += 1
        
        # Fill remaining factors if any
        while factor_idx < n_factors:
            factor_records.append({
                'model_id': model_id,
                'name': f"Factor {factor_idx + 1}",
                'description': f"Factor {factor_idx}",
                'factor_index': factor_idx,
                'block_name': None
            })
            factor_idx += 1
        
        # Insert factors and get IDs
        if factor_records:
            result = db_client.table('factors').insert(factor_records).execute()
            if result.data:
                # Map factor_index to factor_id
                for factor_data in result.data:
                    factor_id_map[factor_data['factor_index']] = factor_data['id']
                logger.info(f"Inserted {len(result.data)} factors for model_id={model_id}")
            else:
                logger.warning("No factors inserted. Check database connection and permissions.")
                return
        
        # Prepare factor_values records (time series of factors)
        factor_value_records = []
        for factor_idx, factor_id in factor_id_map.items():
            if factor_idx >= Res.Z.shape[1]:
                continue
            
            # Extract factor time series
            factor_series = Res.Z[:, factor_idx]
            
            for t_idx, (date_val, value) in enumerate(zip(Time, factor_series)):
                if np.isfinite(value):
                    # Convert date to ISO format string for JSON serialization
                    if hasattr(date_val, 'date'):
                        date_str = date_val.date().isoformat()
                    else:
                        date_str = pd.to_datetime(date_val).date().isoformat()
                    
                    factor_value_records.append({
                        'factor_id': factor_id,
                        'vintage_id': vintage_id,
                        'date': date_str,
                        'value': float(value)
                    })
        
        # Insert factor values in batches
        if factor_value_records:
            batch_size = 1000
            total_inserted = 0
            for i in range(0, len(factor_value_records), batch_size):
                batch = factor_value_records[i:i + batch_size]
                try:
                    result = db_client.table('factor_values').upsert(
                        batch,
                        on_conflict='factor_id,vintage_id,date'
                    ).execute()
                    if result.data:
                        total_inserted += len(result.data)
                except Exception as e:
                    logger.error(f"Failed to insert factor_values batch {i//batch_size + 1}: {e}")
                    raise
            
            logger.info(f"Inserted {total_inserted} factor values for model_id={model_id}")
        
        # Prepare factor_loadings records (C matrix: N x m)
        factor_loading_records = []
        series_ids = config.SeriesID if hasattr(config, 'SeriesID') else []
        
        if hasattr(Res, 'C') and Res.C is not None:
            C = Res.C  # N x m matrix
            n_series, n_factors_actual = C.shape
            
            for series_idx, series_id in enumerate(series_ids):
                if series_idx >= n_series:
                    break
                
                for factor_idx, factor_id in factor_id_map.items():
                    if factor_idx >= n_factors_actual:
                        break
                    
                    loading_value = float(C[series_idx, factor_idx])
                    if np.isfinite(loading_value):
                        factor_loading_records.append({
                            'factor_id': factor_id,
                            'series_id': series_id,
                            'loading': loading_value
                        })
        
        # Insert factor loadings in batches
        if factor_loading_records:
            batch_size = 1000
            total_inserted = 0
            for i in range(0, len(factor_loading_records), batch_size):
                batch = factor_loading_records[i:i + batch_size]
                try:
                    result = db_client.table('factor_loadings').upsert(
                        batch,
                        on_conflict='factor_id,series_id'
                    ).execute()
                    if result.data:
                        total_inserted += len(result.data)
                except Exception as e:
                    logger.error(f"Failed to insert factor_loadings batch {i//batch_size + 1}: {e}")
                    raise
            
            logger.info(f"Inserted {total_inserted} factor loadings for model_id={model_id}")
        
        logger.info(
            f"Successfully saved factors, factor_values, and factor_loadings to database "
            f"for model_id={model_id}, vintage_id={vintage_id}"
        )
        
    except ImportError:
        logger.warning("Database module not available. Cannot save factors to database.")
    except Exception as e:
        logger.error(f"Failed to save factors to database: {e}", exc_info=True)
        raise
