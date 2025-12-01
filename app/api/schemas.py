"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class TrainRequest(BaseModel):
    """Request schema for training endpoint."""
    model_name: Optional[str] = None
    model_type: str  # "dfm" or "ddfm"
    experiment_id: str  # Experiment ID (e.g., "default", "exp1")
    data_path: str


class TrainResponse(BaseModel):
    """Response schema for training endpoint."""
    job_id: str
    status: str
    message: str


class TrainStatus(BaseModel):
    """Status schema for training job."""
    job_id: str
    status: str  # "running", "completed", "failed"
    progress: int  # 0-100
    message: str
    error: Optional[str] = None


class ModelInfo(BaseModel):
    """Schema for model information."""
    model_name: str
    timestamp: str
    config_path: str
    model_type: str


class InferenceRequest(BaseModel):
    """Request schema for inference endpoint."""
    model_name: str
    target_series: str
    view_date: str
    target_period: Optional[str] = None


class InferenceResponse(BaseModel):
    """Response schema for inference endpoint."""
    nowcast_value: float
    target_series: str
    target_period: str
    view_date: str
    data_availability: Optional[Dict[str, int]] = None
    factors_at_view: Optional[List[float]] = None


class ConfigResponse(BaseModel):
    """Response schema for config endpoint."""
    config_name: str
    content: str


class ConfigUpdateRequest(BaseModel):
    """Request schema for config update endpoint."""
    content: str


class UploadResponse(BaseModel):
    """Response schema for file upload endpoint."""
    filename: str
    path: str
    message: str


class ExperimentInfo(BaseModel):
    """Schema for experiment information."""
    experiment_id: str
    model_type: str
    created_at: Optional[str] = None


class ExperimentRequest(BaseModel):
    """Request schema for experiment creation/update."""
    model_type: str  # "dfm" or "ddfm"
    content: str  # YAML content


class ExperimentResponse(BaseModel):
    """Response schema for experiment endpoint."""
    experiment_id: str
    model_type: str
    content: str


class SeriesConfigResponse(BaseModel):
    """Response schema for series config endpoint."""
    series_name: str
    content: str


class BlockConfigResponse(BaseModel):
    """Response schema for block config endpoint."""
    block_name: str
    content: str


class DashboardStats(BaseModel):
    """Response schema for dashboard statistics."""
    total_models: int
    active_training_jobs: int
    total_experiments: int
    recent_success_rate: float
    models_by_type: Dict[str, int]

