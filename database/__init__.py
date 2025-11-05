"""Database module for Supabase integration with DFM nowcasting system."""

from .client import get_client, get_supabase_client
from .operations import (
    # Data source operations
    get_source_id,
    get_source_code,
    # Series operations
    get_series,
    upsert_series,
    list_series,
    # Vintage operations
    create_vintage,
    get_vintage,
    get_latest_vintage,
    update_vintage_status,
    # Ingestion job operations
    create_ingestion_job,
    update_ingestion_job,
    get_ingestion_job,
    # Observation operations
    insert_observations_from_dataframe,
    get_observations,
    get_vintage_data,
    # Statistics metadata operations
    upsert_statistics_metadata,
    get_statistics_metadata,
    list_dfm_selected_statistics,
    update_statistics_metadata_status,
    # Statistics items operations
    upsert_statistics_items,
    get_statistics_items,
    get_active_items_for_statistic,
    # Model operations
    save_model_config,
    load_model_config,
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
from .series import SeriesManager

__all__ = [
    'get_client',
    'get_supabase_client',
    'get_source_id',
    'get_source_code',
    'get_series',
    'upsert_series',
    'list_series',
    'create_vintage',
    'get_vintage',
    'get_latest_vintage',
    'update_vintage_status',
    'create_ingestion_job',
    'update_ingestion_job',
    'get_ingestion_job',
    'insert_observations_from_dataframe',
    'get_observations',
    'get_vintage_data',
    'upsert_statistics_metadata',
    'get_statistics_metadata',
    'list_dfm_selected_statistics',
    'update_statistics_metadata_status',
    'upsert_statistics_items',
    'get_statistics_items',
    'get_active_items_for_statistic',
    'save_model_config',
    'load_model_config',
    'save_model_weights',
    'load_model_weights',
    'save_forecast',
    'get_forecast',
    'get_latest_forecasts',
    'SeriesModel',
    'StatisticsMetadataModel',
    'StatisticsItemModel',
    'ObservationModel',
    'SeriesManager',
]

