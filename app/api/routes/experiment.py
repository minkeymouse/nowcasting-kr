"""API endpoints for experiment execution and training."""

from fastapi import APIRouter
from typing import List, Dict, Any, Optional

from app.api.dependencies import experiment_service, model_registry
from app.api.error_handlers import handle_exceptions
from app.api.schemas import (
    ExperimentRunRequest, ExperimentRunResponse, ExperimentStatus
)
from app.utils import JobStatus

router = APIRouter()


@router.post("/experiment/run", response_model=ExperimentRunResponse)
@handle_exceptions
async def run_experiment(request: ExperimentRunRequest) -> ExperimentRunResponse:
    """Run an experiment using src.training.train."""
    job_id = experiment_service.run_experiment(
        experiment_id=request.experiment_id,
        data_path=request.data_path,
        model_name=request.model_name,
        config_overrides=request.config_overrides,
        horizons=request.horizons
    )
    
    return ExperimentRunResponse(
        job_id=job_id,
        status=JobStatus.RUNNING,
        message="Experiment started"
    )


@router.get("/experiment/status/{job_id}", response_model=ExperimentStatus)
@handle_exceptions
async def get_experiment_status(job_id: str) -> ExperimentStatus:
    """Get experiment job status."""
    job = experiment_service.get_status(job_id)
    if not job:
        from app.utils import ModelNotFoundError
        raise ModelNotFoundError(f"Job {job_id} not found")
    
    return ExperimentStatus(
        job_id=job_id,
        status=job.get("status", JobStatus.RUNNING),
        experiment_id=job.get("experiment_id", ""),
        timestamp=job.get("timestamp", ""),
        error=job.get("error"),
        result=job.get("result")
    )


@router.get("/experiment/jobs", response_model=List[ExperimentStatus])
@handle_exceptions
async def get_recent_jobs(limit: int = 10) -> List[ExperimentStatus]:
    """Get recent experiment jobs."""
    jobs = experiment_service.get_recent_jobs(limit=limit)
    return [
        ExperimentStatus(
            job_id=job.get("job_id", ""),
            status=job.get("status", JobStatus.RUNNING),
            experiment_id=job.get("experiment_id", ""),
            timestamp=job.get("timestamp", ""),
            error=job.get("error")
        )
        for job in jobs
    ]

