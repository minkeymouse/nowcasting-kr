"""Training-related API endpoints."""

from fastapi import APIRouter, HTTPException
from pathlib import Path

from api.dependencies import training_manager, config_manager
from api.error_handlers import handle_exceptions
from utils import generate_model_name, ConfigError
from api.schemas import TrainRequest, TrainResponse, TrainStatus

router = APIRouter()


@router.post("/train", response_model=TrainResponse)
@handle_exceptions
async def train_model(request: TrainRequest):
    """Start a training job."""
    # Generate model name if not provided
    model_name = request.model_name or generate_model_name()
    
    # Validate experiment exists
    experiment_info = config_manager.get_experiment(request.experiment_id)
    # Verify model type matches experiment
    if experiment_info["model_type"] != request.model_type:
        raise HTTPException(
            status_code=400,
            detail=f"Model type mismatch: experiment '{request.experiment_id}' is for '{experiment_info['model_type']}', but requested '{request.model_type}'"
        )
    
    # Validate data path exists
    data_path_obj = Path(request.data_path)
    if not data_path_obj.exists():
        raise HTTPException(status_code=400, detail=f"Data file not found: {request.data_path}")
    
    # Get config path from experiment
    config_path = config_manager.get_experiment_config_path(request.experiment_id)
    
    # Validate model type
    if request.model_type not in ["dfm", "ddfm"]:
        raise HTTPException(status_code=400, detail=f"Invalid model_type: {request.model_type}. Must be 'dfm' or 'ddfm'")
    
    # Start training (this runs synchronously)
    job_id = training_manager.start_training(
        model_name=model_name,
        model_type=request.model_type,
        config_path=config_path,
        data_path=request.data_path
    )
    
    return TrainResponse(
        job_id=job_id,
        status="started",
        message="Training job started"
    )


@router.get("/train/status/{job_id}", response_model=TrainStatus)
@handle_exceptions
async def get_training_status(job_id: str):
    """Get training job status."""
    status = training_manager.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return TrainStatus(
        job_id=job_id,
        status=status["status"],
        progress=status["progress"],
        message=status["message"],
        error=status.get("error")
    )

