"""Shared dependencies for API routes - singleton service instances."""

from services import TrainingManager, ModelRegistry, ConfigManager, ModelService

# Singleton service instances - shared across all routes
training_manager = TrainingManager()
model_registry = ModelRegistry()
config_manager = ConfigManager()
model_service = ModelService()

