"""API routes module - combines all route routers."""

from fastapi import APIRouter
from .training import router as training_router
from .dashboard import router as dashboard_router
from .config import router as config_router
from .inference import router as inference_router
from .upload import router as upload_router

router = APIRouter()

# Include all sub-routers
router.include_router(training_router)
router.include_router(dashboard_router)
router.include_router(config_router)
router.include_router(inference_router)  # Includes models endpoints
router.include_router(upload_router)

