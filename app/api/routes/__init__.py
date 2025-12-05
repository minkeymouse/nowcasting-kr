"""API routes module."""

from fastapi import APIRouter
from .config import router as config_router
from .experiment import router as experiment_router

router = APIRouter()
router.include_router(config_router)
router.include_router(experiment_router)

