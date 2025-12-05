"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class ModelInfo(BaseModel):
    """Schema for model information."""
    model_name: str
    model_type: str
    timestamp: Optional[str] = None
    config_path: Optional[str] = None
    experiment_id: Optional[str] = None


class ExperimentInfo(BaseModel):
    """Schema for experiment information."""
    experiment_id: str
    model_type: str
    created_at: Optional[str] = None


class ExperimentRunRequest(BaseModel):
    """Request schema for experiment run endpoint."""
    experiment_id: str
    data_path: Optional[str] = None
    model_name: Optional[str] = None
    config_overrides: Optional[List[str]] = None
    horizons: Optional[List[int]] = None


class ExperimentRunResponse(BaseModel):
    """Response schema for experiment run endpoint."""
    job_id: str
    status: str
    message: str


class ExperimentStatus(BaseModel):
    """Status schema for experiment job."""
    job_id: str
    status: str  # "running", "completed", "failed"
    experiment_id: str
    timestamp: str
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class UnifiedConfigResponse(BaseModel):
    """Response schema for unified config structure."""
    experiment_id: str
    model_type: str
    config: Dict[str, Any]


class UnifiedConfigUpdateRequest(BaseModel):
    """Request schema for unified config update."""
    config: Dict[str, Any]

