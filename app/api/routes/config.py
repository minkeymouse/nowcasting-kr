"""API endpoints for configuration, models, and file uploads."""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
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
from app.utils import DATA_DIR

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
        except:
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
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV file")
    
    if not filename.endswith('.csv'):
        filename = filename + '.csv'
    
    file_path = DATA_DIR / filename
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    
    try:
        df = pd.read_csv(file_path, nrows=1)
        if date_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Date column '{date_column}' not found in CSV. Available columns: {', '.join(df.columns)}"
            )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Failed to validate CSV: {str(e)}")
    
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
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV file")
    
    if not filename.endswith('.csv'):
        filename = filename + '.csv'
    
    file_path = DATA_DIR / filename
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    
    try:
        df = pd.read_csv(file_path)
        required_cols = ['series_name', 'series_description', 'frequency', 'release']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {', '.join(missing_cols)}. Found columns: {', '.join(df.columns)}"
            )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Failed to validate CSV: {str(e)}")
    
    return {
        "message": "Config uploaded successfully",
        "filename": filename,
        "path": str(file_path),
        "rows": len(df)
    }

