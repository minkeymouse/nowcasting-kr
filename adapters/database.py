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
    config_name: Optional[str] = None,
    config_id: Optional[int] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    strict_mode: bool = False,
    client: Optional[object] = None
) -> Tuple[pd.DataFrame, pd.DatetimeIndex, pd.DataFrame]:
    """Fetch vintage data and metadata from database.
    
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
        from database import (
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
            # Convert to DataFrame for consistency
            data_df = pd.DataFrame(X, index=Time, columns=None)
            return data_df, Time, series_metadata_df
        except (TypeError, AttributeError) as e:
            logger.warning(f"get_vintage_data_for_config failed ({e}), using general function")
    elif config_id is not None:
        # Deprecated: resolve config_name from config_id
        try:
            from database.helpers import resolve_config_name
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
                # Convert to DataFrame for consistency
                data_df = pd.DataFrame(X, index=Time, columns=None)
                return data_df, Time, series_metadata_df
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
    return data_df, Time, series_metadata_df


def export_data_to_csv(
    output_path: Union[str, Path],
    vintage_id: Optional[int] = None,
    vintage_date: Optional[Union[date, str, pd.Timestamp]] = None,
    config: Optional[ModelConfig] = None,
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
    config : ModelConfig, optional
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
    >>> from adapters.database import export_data_to_csv
    >>> csv_path = export_data_to_csv(
    ...     output_path='data/vintage_1.csv',
    ...     vintage_id=1,
    ...     config_name='001-initial-spec'
    ... )
    """
    try:
        from database.operations import (
            get_latest_vintage_id,
            get_vintage,
            _normalize_date as normalize_date
        )
        from database.helpers import (
            get_model_config_series_ids,
            resolve_config_name
        )
    except ImportError:
        raise ImportError("Database module not available. Cannot export data to CSV.")
    
    logger = logging.getLogger(__name__)
    
    # Resolve vintage
    if vintage_id is None:
        if vintage_date is not None:
            vintage_date_normalized = normalize_date(vintage_date)
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
    
    # Load data from database
    data_df, Time, series_metadata_df = load_data_from_db(
        vintage_id=vintage_id,
        config=config,
        config_name=config_name,
        config_id=config_id,
        start_date=start_date,
        end_date=end_date,
        sample_start=sample_start,
        strict_mode=strict_mode,
        client=client
    )
    
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
    config: Optional[ModelConfig] = None,
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
    config : ModelConfig, optional
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
            from database.helpers import resolve_config_name
            config_name = resolve_config_name(config_id=config_id, client=client)
            if config_name:
                logger.warning("config_id is deprecated. Use config_name instead.")
        except ImportError:
            pass  # Will fall back to config_id handling in _fetch_vintage_data
    
    # Fetch data from database
    data_df, Time, series_metadata_df = _fetch_vintage_data(
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



def save_blocks_to_db(
    config: ModelConfig,
    config_name: str,
    client: Optional[object] = None
) -> None:
    """Save block assignments from ModelConfig to database blocks table.
    
    This function extracts block information from a ModelConfig object and saves it
    to the blocks table. The config_name should be derived from the CSV filename
    (e.g., '001_initial_spec.csv' → '001-initial-spec').
    
    Parameters
    ----------
    config : ModelConfig
        Model configuration with series and block information
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
