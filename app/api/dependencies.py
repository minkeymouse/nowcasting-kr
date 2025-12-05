"""Shared dependencies for API routes - singleton service instances."""

from app.services import ModelRegistry, ConfigManager, ModelService, ExperimentService

# Singleton service instances - shared across all routes
model_registry = ModelRegistry()
config_manager = ConfigManager()
model_service = ModelService()
experiment_service = ExperimentService()

