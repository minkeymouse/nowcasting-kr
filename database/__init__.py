"""Database module for Supabase integration with DFM nowcasting system."""

from .client import get_client, get_supabase_client
from .helpers import DatabaseError, NotFoundError, ValidationError, ensure_client
from .operations import (
    # Series operations
    get_series,
    upsert_series,
    list_series,
    # Vintage operations
    create_vintage,
    get_vintage,
    get_latest_vintage,
    update_vintage_status,
    # Vintage/job operations (integrated into data_vintages)
    ensure_vintage_and_job,
    finalize_ingestion_job,
    # Observation operations
    insert_observations_from_dataframe,
    get_observations,
    get_vintage_data,
    get_vintage_data_for_config,
    get_latest_vintage_id,
    get_series_metadata_bulk,
    get_model_config_series_ids,
    load_data_from_db,
    # Statistics metadata operations
    upsert_statistics_metadata,
    get_statistics_metadata,
    get_statistics_metadata_bulk,
    list_dfm_selected_statistics,
    update_statistics_metadata_status,
    # Statistics items operations
    upsert_statistics_items,
    get_statistics_items,
    get_active_items_for_statistic,
    # Model operations
    load_model_config,
    save_model_config,
    save_model_weights,
    load_model_weights,
    # Forecast operations
    save_forecast,
    get_forecast,
    get_latest_forecasts,
)
from .models import (
    SeriesModel,
    StatisticsMetadataModel,
    StatisticsItemModel,
    ObservationModel,
)
from .operations import generate_series_id, create_or_get_series, create_series_from_item, delete_old_vintages
from .spec_manager import (
    export_config_to_csv,
    get_latest_spec_from_db,
    sync_csv_from_db,
)

__all__ = [
    'get_client',
    'get_supabase_client',
    'ensure_client',
    'get_series',
    'upsert_series',
    'list_series',
    'create_vintage',
    'get_vintage',
    'get_latest_vintage',
    'update_vintage_status',
    'ensure_vintage_and_job',
    'finalize_ingestion_job',
    'insert_observations_from_dataframe',
    'get_observations',
    'get_latest_observation_date',
    'check_series_exists',
    'get_vintage_data',
    'get_vintage_data_for_config',
    'get_latest_vintage_id',
    'get_series_metadata_bulk',
    'get_model_config_series_ids',
    'load_data_from_db',
    'upsert_statistics_metadata',
    'get_statistics_metadata',
    'get_statistics_metadata_bulk',
    'list_dfm_selected_statistics',
    'update_statistics_metadata_status',
    'upsert_statistics_items',
    'get_statistics_items',
    'get_active_items_for_statistic',
    'load_model_config',
    'save_model_config',
    'save_model_weights',
    'load_model_weights',
    'save_forecast',
    'get_forecast',
    'get_latest_forecasts',
    'SeriesModel',
    'StatisticsMetadataModel',
    'StatisticsItemModel',
    'ObservationModel',
    'generate_series_id',
    'create_or_get_series',
    'create_series_from_item',
    'DatabaseError',
    'NotFoundError',
    'ValidationError',
    'export_config_to_csv',
    'get_latest_spec_from_db',
    'sync_csv_from_db',
    'delete_old_vintages',
]

