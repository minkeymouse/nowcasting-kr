"""Dashboard-related API endpoints."""

from fastapi import APIRouter, HTTPException

from api.dependencies import training_manager, model_registry, config_manager
from api.error_handlers import handle_exceptions
from api.schemas import DashboardStats
from api.charts import (
    generate_model_distribution_chart,
    generate_experiment_usage_chart,
    generate_training_timeline_chart
)

router = APIRouter()


@router.get("/dashboard/training-jobs")
@handle_exceptions
async def get_training_jobs():
    """Get recent training jobs."""
    return training_manager.get_recent_jobs(limit=10)


@router.get("/dashboard/experiment-usage")
@handle_exceptions
async def get_experiment_usage():
    """Get experiment usage statistics."""
    return model_registry.get_experiment_usage()


@router.get("/dashboard/stats", response_model=DashboardStats)
@handle_exceptions
async def get_dashboard_stats():
    """Get dashboard statistics."""
    # Get all models
    models = model_registry.list_models()
    total_models = len(models)
    
    # Count models by type
    models_by_type = model_registry.get_model_counts_by_type()
    
    # Count active training jobs
    active_training_jobs = len([
        job_id for job_id, status in training_manager._jobs.items()
        if status.get("status") == "running"
    ])
    
    # Get experiments count
    try:
        experiments = config_manager.list_experiments()
        total_experiments = len(experiments)
    except:
        total_experiments = 0
    
    # Calculate recent success rate (last 10 jobs)
    recent_jobs = list(training_manager._jobs.values())[-10:]
    if recent_jobs:
        successful = sum(1 for job in recent_jobs if job.get("status") == "completed")
        recent_success_rate = (successful / len(recent_jobs)) * 100
    else:
        recent_success_rate = 0.0
    
    return DashboardStats(
        total_models=total_models,
        active_training_jobs=active_training_jobs,
        total_experiments=total_experiments,
        recent_success_rate=recent_success_rate,
        models_by_type=models_by_type
    )


@router.get("/dashboard/chart/model-distribution")
@handle_exceptions
async def get_model_distribution_chart():
    """Generate model distribution chart using seaborn."""
    return generate_model_distribution_chart(model_registry)


@router.get("/dashboard/chart/experiment-usage")
@handle_exceptions
async def get_experiment_usage_chart():
    """Generate experiment usage chart using seaborn."""
    return generate_experiment_usage_chart(model_registry)


@router.get("/dashboard/chart/training-timeline")
@handle_exceptions
async def get_training_timeline_chart():
    """Generate training timeline chart using seaborn."""
    return generate_training_timeline_chart(training_manager)

