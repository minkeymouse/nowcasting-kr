"""Inference, models, and agent-related API endpoints."""

from fastapi import APIRouter, HTTPException
from typing import List

from api.dependencies import model_service, model_registry
from api.error_handlers import handle_exceptions
from api.schemas import InferenceRequest, InferenceResponse, ModelInfo

router = APIRouter()


@router.get("/models", response_model=List[ModelInfo])
@handle_exceptions
async def list_models():
    """List all trained models."""
    models = model_registry.list_models()
    return [ModelInfo(**model) for model in models]


@router.post("/inference", response_model=InferenceResponse)
@handle_exceptions
async def run_inference(request: InferenceRequest):
    """Run inference on a trained model."""
    result = model_service.run_inference(
        model_name=request.model_name,
        target_series=request.target_series,
        view_date=request.view_date,
        target_period=request.target_period
    )
    return InferenceResponse(**result)


@router.post("/agent/report")
async def generate_report():
    """Generate report using agent (placeholder)."""
    return {"message": "Report generation not yet implemented"}

