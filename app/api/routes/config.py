"""API endpoints for configuration, models, and file uploads."""

from fastapi import APIRouter, UploadFile, File
import shutil
import pandas as pd
from typing import List, Dict, Any

from api.dependencies import config_manager, model_registry
from api.error_handlers import handle_exceptions
from api.schemas import (
    ExperimentInfo,
    UnifiedConfigResponse, UnifiedConfigUpdateRequest,
    ModelInfo
)
from app.utils import (
    DATA_DIR,
    validate_csv_file, ensure_csv_extension, validate_csv_columns
)

router = APIRouter()


# Experiment endpoints
@router.get("/experiments", response_model=List[ExperimentInfo])
@handle_exceptions
async def list_experiments():
    """List all experiments."""
    experiment_ids = config_manager.list_experiments()
    experiments = []
    for exp_id in experiment_ids:
        try:
            exp_data = config_manager.get_experiment(exp_id)
            experiments.append(ExperimentInfo(
                experiment_id=exp_data["experiment_id"],
                model_type=exp_data["model_type"]
            ))
        except Exception:
            continue
    return experiments


@router.get("/experiment/{experiment_id}/unified", response_model=UnifiedConfigResponse)
@handle_exceptions
async def get_experiment_unified(experiment_id: str):
    """Get experiment config as unified structure (parsed dict)."""
    result = config_manager.get_experiment_unified(experiment_id)
    return UnifiedConfigResponse(**result)


@router.put("/experiment/{experiment_id}/unified")
@handle_exceptions
async def update_experiment_unified(experiment_id: str, request: UnifiedConfigUpdateRequest) -> Dict[str, str]:
    """Update experiment config from unified structure (dict)."""
    config_manager.update_experiment_unified(experiment_id, request.config)
    return {"message": "Experiment config updated successfully"}


# Series list endpoint
@router.get("/series", response_model=List[str])
@handle_exceptions
async def list_series():
    """List all series configs."""
    return config_manager.list_series_configs()


# Models endpoint
@router.get("/models", response_model=List[ModelInfo])
@handle_exceptions
async def list_models():
    """List all trained models."""
    models = model_registry.list_models()
    return [ModelInfo(**model) for model in models]


# File upload endpoints
@router.post("/data")
@handle_exceptions
async def upload_data(
    file: UploadFile = File(...),
    date_column: str = "date",
    date_format: str = "YYYY-MM-DD",
    filename: str = "sample_data.csv"
) -> Dict[str, Any]:
    """Upload CSV data file with date column specification."""
    validate_csv_file(file.filename)
    filename = ensure_csv_extension(filename)
    
    file_path = DATA_DIR / filename
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    
    validate_csv_columns(file_path, [date_column])
    
    return {
        "message": "Data uploaded successfully",
        "filename": filename,
        "path": str(file_path),
        "date_column": date_column,
        "date_format": date_format
    }


@router.post("/config")
@handle_exceptions
async def upload_config(
    file: UploadFile = File(...),
    filename: str = "metadata.csv"
) -> Dict[str, Any]:
    """Upload config CSV file with series metadata."""
    validate_csv_file(file.filename)
    filename = ensure_csv_extension(filename)
    
    file_path = DATA_DIR / filename
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    
    required_cols = ['series_name', 'series_description', 'frequency', 'release']
    validate_csv_columns(file_path, required_cols)
    
    df = pd.read_csv(file_path)
    return {
        "message": "Config uploaded successfully",
        "filename": filename,
        "path": str(file_path),
        "rows": len(df)
    }

