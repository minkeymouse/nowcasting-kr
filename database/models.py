"""Pydantic models for database schema validation."""

from typing import Optional, List, Dict, Any
from datetime import date, datetime
from pydantic import BaseModel, Field


class SeriesModel(BaseModel):
    """Series metadata model."""
    series_id: str = Field(..., description="Unique identifier for the series")
    series_name: str = Field(..., description="Human-readable name")
    frequency: str = Field(..., description="Frequency code: d, w, m, q, sa, a")
    units: Optional[str] = Field(None, description="Original units of measurement")
    transformation: Optional[str] = Field(None, description="Transformation code")
    category: Optional[str] = Field(None, description="Category classification")
    api_source: Optional[str] = Field(None, description="API source identifier (e.g., BOK, KOSIS, MANUAL)")
    data_code: Optional[str] = Field(None, description="API-specific series code (e.g., BOK: 200Y106, KOSIS: 101_DT_1DA7002S)")
    item_id: Optional[str] = Field(None, description="Item identifier (e.g., BOK: 1400, KOSIS: T80)")


class VintageModel(BaseModel):
    """Data vintage model."""
    vintage_id: Optional[int] = None
    vintage_date: date = Field(..., description="Vintage date (weekly snapshot)")
    country: str = Field("KR", description="Country code")
    description: Optional[str] = None
    fetch_status: str = Field("pending", description="Status: pending, in_progress, completed, failed")
    fetch_started_at: Optional[datetime] = None
    fetch_completed_at: Optional[datetime] = None
    github_run_id: Optional[str] = None
    github_workflow_run_url: Optional[str] = None
    error_message: Optional[str] = None


class ObservationModel(BaseModel):
    """Observation model for time-series data."""
    series_id: str = Field(..., description="Series identifier")
    vintage_id: int = Field(..., description="Vintage identifier")
    observation_date: date = Field(..., description="Observation date", alias="date")
    value: float = Field(..., description="Observation value")
    github_run_id: Optional[str] = Field(None, description="GitHub Actions run ID")
    is_forecast: bool = Field(False, description="Whether this is a forecast")
    api_source: Optional[str] = None
    item_code1: Optional[str] = Field(None, description="First level item code")
    item_code2: Optional[str] = Field(None, description="Second level item code")
    item_code3: Optional[str] = Field(None, description="Third level item code")
    item_code4: Optional[str] = Field(None, description="Fourth level item code")
    item_name1: Optional[str] = Field(None, description="First level item name")
    item_name2: Optional[str] = Field(None, description="Second level item name")
    item_name3: Optional[str] = Field(None, description="Third level item name")
    item_name4: Optional[str] = Field(None, description="Fourth level item name")
    weight: Optional[float] = Field(None, description="Weight value from source API")
    
    class Config:
        populate_by_name = True


class ForecastModel(BaseModel):
    """Forecast model."""
    model_id: int = Field(..., description="Trained model identifier")
    series_id: str = Field(..., description="Series identifier")
    forecast_date: date = Field(..., description="Forecast target date")
    forecast_value: float = Field(..., description="Forecast value")
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    confidence_level: float = Field(0.95, description="Confidence level for bounds")


class StatisticsMetadataModel(BaseModel):
    """Statistics metadata model."""
    id: Optional[int] = None
    source_id: int = Field(..., description="Reference to data_sources table")
    source_stat_code: str = Field(..., description="Source-specific statistic code (e.g., BOK: 200Y105, KOSIS: orgId_tblId)")
    source_stat_name: Optional[str] = Field(None, description="Original statistic name from source")
    source_stat_name_eng: Optional[str] = Field(None, description="English statistic name")
    cycle: Optional[str] = Field(None, description="Frequency code: A, S, Q, M, SM, D")
    frequency_code: Optional[str] = Field(None, description="Normalized frequency code")
    org_name: Optional[str] = Field(None, description="Organization name")
    is_searchable: bool = Field(False, description="Whether data can be fetched via API")
    parent_stat_code: Optional[str] = Field(None, description="Parent statistic code")
    parent_item_code: Optional[str] = Field(None, description="Parent item code")
    hierarchy_level: Optional[int] = Field(None, description="Depth in hierarchy")
    source_metadata: Optional[Dict[str, Any]] = Field(None, description="Source-specific fields as JSON")
    is_dfm_selected: bool = Field(False, description="Whether selected for DFM nowcasting")
    dfm_priority: Optional[int] = Field(None, description="Priority rank for DFM (1-50)")
    dfm_selected_at: Optional[datetime] = None
    dfm_selection_reason: Optional[str] = Field(None, description="Reason for DFM selection")
    is_active: bool = Field(True, description="Whether to collect data for this statistic")
    last_data_fetch_date: Optional[date] = None
    last_data_fetch_status: Optional[str] = Field(None, description="Status: success, failed, partial")
    data_start_date: Optional[date] = None
    data_end_date: Optional[date] = None
    total_observations: Optional[int] = None
    last_observation_date: Optional[date] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class StatisticsItemModel(BaseModel):
    """Statistics item model."""
    id: Optional[int] = None
    statistics_metadata_id: int = Field(..., description="Reference to statistics_metadata table")
    item_code: str = Field(..., description="Item code (e.g., '1101', '*AA')")
    item_name: Optional[str] = Field(None, description="Item name in Korean")
    item_name_eng: Optional[str] = Field(None, description="Item name in English")
    parent_item_code: Optional[str] = Field(None, description="Parent item code")
    parent_item_name: Optional[str] = Field(None, description="Parent item name")
    grp_code: Optional[str] = Field(None, description="Group code")
    grp_name: Optional[str] = Field(None, description="Group name")
    cycle: str = Field(..., description="Frequency code: A, S, Q, M, SM, D")
    start_time: Optional[str] = Field(None, description="Start time as string (e.g., '2024Q1')")
    end_time: Optional[str] = Field(None, description="End time as string")
    data_count: Optional[int] = Field(None, description="Number of available data points")
    unit_name: Optional[str] = Field(None, description="Unit name")
    weight: Optional[str] = Field(None, description="Weight value")
    is_active: bool = Field(True, description="Whether this item is active")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# Table names as constants
TABLES = {
    'series': 'series',
    'vintages': 'data_vintages',
    'observations': 'observations',
    'forecasts': 'forecasts',
    'statistics_metadata': 'statistics_metadata',
    'statistics_items': 'statistics_items',
    'factors': 'factors',
    'factor_values': 'factor_values',
    'factor_loadings': 'factor_loadings',
}

