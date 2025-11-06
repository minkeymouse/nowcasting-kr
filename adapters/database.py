"""Database adapters for DFM module.

This module bridges the generic DFM module (src/nowcasting) with the application-specific
database layer. It provides functions to load data from database and save results back.

The DFM module itself remains generic and database-agnostic.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple, Union, Any
from datetime import date
import warnings
import logging

# Import generic DFM module functions
from src.nowcasting.config import ModelConfig
from src.nowcasting.data_loader import _transform_series

logger = logging.getLogger(__name__)


def _get_db_client(client: Optional[object] = None):
    """Get database client, raising ImportError if database module unavailable."""
    if client is not None:
        return client
    try:
        from database import get_client
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
            from database import get_latest_vintage_id
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
    config: Optional[ModelConfig] = None
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Apply transformations using series metadata, falling back to config if needed.
    
    This function consolidates transformation logic to work with both database
    metadata and ModelConfig objects (from CSV or YAML).
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
    config: Optional[ModelConfig] = None,
    config_id: Optional[int] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    strict_mode: bool = False,
    client: Optional[object] = None
) -> Tuple[pd.DataFrame, pd.DatetimeIndex, pd.DataFrame]:
    """Fetch vintage data and metadata from database.
    
    Uses config-specific function if available, otherwise falls back to general function.
    """
    try:
        from database import (
            get_vintage_data,
            get_vintage_data_for_config,
            get_series_metadata_bulk
        )
    except ImportError:
        raise
    
    # Prefer config-specific function (uses model_block_assignments for ordering)
    if config_id is not None:
        try:
            return get_vintage_data_for_config(
                vintage_id=vintage_id,
                config_id=config_id,
                start_date=start_date,
                end_date=end_date,
                strict_mode=strict_mode,
                client=client
            )
        except (TypeError, AttributeError) as e:
            # Fallback if function signature differs or not available
            logger.warning(f"get_vintage_data_for_config failed ({e}), using general function")
    
    # Use general function with series IDs
    result = get_vintage_data(
        vintage_id=vintage_id,
        config_series_ids=config_series_ids,
        start_date=start_date,
        end_date=end_date,
        client=client
    )
    
    # Handle both return signatures (2-tuple or 3-tuple)
    if len(result) == 3:
        return result  # (data_df, time_index, series_metadata_df)
    else:
        data_df, time_index = result
        # Fetch metadata separately if needed
        series_metadata_df = (
            get_series_metadata_bulk(config_series_ids, client=client)
            if config_series_ids else pd.DataFrame()
        )
        return data_df, time_index, series_metadata_df


def load_data_from_db(
    vintage_id: Optional[int] = None,
    vintage_date: Optional[Union[date, str, pd.Timestamp]] = None,
    config: Optional[ModelConfig] = None,
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
    config : ModelConfig, optional
        Model configuration object (required for transformations)
    config_id : int, optional
        Model configuration ID in database (uses model_block_assignments for ordering)
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
    
    # Fetch data from database
    data_df, Time, series_metadata_df = _fetch_vintage_data(
        vintage_id=resolved_vintage_id,
        config_series_ids=config_series_ids,
        config=config,
            config_name=config_name,
            config_id=config_id,  # Deprecated but kept for backward compatibility
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
    
    # Apply transformations
    X, Time = _apply_transformations_from_metadata(
        Z, Time, data_df, series_metadata_df, config
    )
    
    # Apply sample_start filter if provided
    if sample_start is not None:
        if isinstance(sample_start, str):
            sample_start_dt = pd.to_datetime(sample_start)
        else:
            sample_start_dt = sample_start
        
        mask = Time >= sample_start_dt
        X = X[mask]
        Time = Time[mask]
        Z = Z[mask]
    
    logger.info(
        f"Loaded data from database: vintage_id={resolved_vintage_id}, "
        f"series={len(config_series_ids)}, observations={len(Time)}"
    )
    
    return X, Time, Z


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
        from database import get_client, save_forecast, get_latest_vintage_id
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

