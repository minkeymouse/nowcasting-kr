"""Configuration-related API endpoints."""

from fastapi import APIRouter
from typing import List

from api.dependencies import config_manager
from api.error_handlers import handle_exceptions
from api.schemas import (
    ConfigResponse, ConfigUpdateRequest, ExperimentInfo,
    ExperimentRequest, ExperimentResponse, SeriesConfigResponse,
    BlockConfigResponse
)

router = APIRouter()


# Legacy config endpoints (for backward compatibility)
@router.get("/configs", response_model=List[str])
@handle_exceptions
async def list_configs():
    """List available config files (legacy)."""
    return config_manager.list_configs()


@router.get("/config/{config_name}", response_model=ConfigResponse)
@handle_exceptions
async def get_config(config_name: str):
    """Get config file content (legacy)."""
    content = config_manager.get_config(config_name)
    return ConfigResponse(config_name=config_name, content=content)


@router.put("/config/{config_name}")
@handle_exceptions
async def update_config(config_name: str, request: ConfigUpdateRequest):
    """Update config file (legacy)."""
    config_manager.update_config(config_name, request.content)
    return {"message": "Config updated successfully"}


@router.post("/config/{config_name}")
@handle_exceptions
async def create_config(config_name: str, request: ConfigUpdateRequest):
    """Create a new config file (legacy)."""
    config_manager.update_config(config_name, request.content)
    return {"message": "Config created successfully"}


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
            # Skip experiments that can't be loaded
            continue
    return experiments


@router.get("/experiment/{experiment_id}", response_model=ExperimentResponse)
@handle_exceptions
async def get_experiment(experiment_id: str):
    """Get experiment config."""
    exp_data = config_manager.get_experiment(experiment_id)
    return ExperimentResponse(**exp_data)


@router.post("/experiment/{experiment_id}")
@handle_exceptions
async def create_experiment(experiment_id: str, request: ExperimentRequest):
    """Create a new experiment."""
    config_manager.create_experiment(experiment_id, request.model_type, request.content)
    return {"message": "Experiment created successfully"}


@router.put("/experiment/{experiment_id}")
@handle_exceptions
async def update_experiment(experiment_id: str, request: ExperimentRequest):
    """Update experiment config."""
    config_manager.update_experiment(experiment_id, request.content)
    return {"message": "Experiment updated successfully"}


# Series config endpoints
@router.get("/series-configs", response_model=List[str])
@handle_exceptions
async def list_series_configs():
    """List all series configs."""
    return config_manager.list_series_configs()


@router.get("/series-config/{series_name}", response_model=SeriesConfigResponse)
@handle_exceptions
async def get_series_config(series_name: str):
    """Get series config."""
    content = config_manager.get_series_config(series_name)
    return SeriesConfigResponse(series_name=series_name, content=content)


@router.put("/series-config/{series_name}")
@handle_exceptions
async def update_series_config(series_name: str, request: ConfigUpdateRequest):
    """Update series config."""
    config_manager.update_series_config(series_name, request.content)
    return {"message": "Series config updated successfully"}


# Block config endpoints
@router.get("/block-configs", response_model=List[str])
@handle_exceptions
async def list_block_configs():
    """List all block configs."""
    return config_manager.list_block_configs()


@router.get("/block-config/{block_name}", response_model=BlockConfigResponse)
@handle_exceptions
async def get_block_config(block_name: str):
    """Get block config."""
    content = config_manager.get_block_config(block_name)
    return BlockConfigResponse(block_name=block_name, content=content)


@router.put("/block-config/{block_name}")
@handle_exceptions
async def update_block_config(block_name: str, request: ConfigUpdateRequest):
    """Update block config."""
    config_manager.update_block_config(block_name, request.content)
    return {"message": "Block config updated successfully"}

