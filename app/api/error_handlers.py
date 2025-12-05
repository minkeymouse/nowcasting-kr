"""Error handling utilities for API routes."""

import logging
from functools import wraps
from fastapi import HTTPException
from typing import Callable, Any

from app.utils import ConfigError, ValidationError, ModelNotFoundError, TrainingError

# Set up logging
logger = logging.getLogger(__name__)


def handle_exceptions(func: Callable) -> Callable:
    """Decorator to handle common exceptions and convert to HTTPException.
    
    Logs all exceptions before converting to HTTPException for better debugging.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            # Re-raise HTTPExceptions as-is
            raise
        except ConfigError as e:
            logger.warning(f"ConfigError in {func.__name__}: {str(e)}")
            raise HTTPException(status_code=404, detail=f"Configuration error: {str(e)}")
        except ValidationError as e:
            logger.warning(f"ValidationError in {func.__name__}: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
        except ModelNotFoundError as e:
            logger.warning(f"ModelNotFoundError in {func.__name__}: {str(e)}")
            raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")
        except TrainingError as e:
            logger.error(f"TrainingError in {func.__name__}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    return wrapper


def to_http_exception(e: Exception) -> HTTPException:
    """Convert custom exceptions to HTTPException.
    
    Converts application-specific exceptions to appropriate HTTPException instances.
    This function is kept for backward compatibility but is not actively used
    since @handle_exceptions decorator handles this automatically.
    """
    if isinstance(e, HTTPException):
        return e
    elif isinstance(e, (ConfigError, ModelNotFoundError)):
        logger.warning(f"Resource not found: {str(e)}")
        return HTTPException(status_code=404, detail=str(e))
    elif isinstance(e, ValidationError):
        logger.warning(f"Validation error: {str(e)}")
        return HTTPException(status_code=400, detail=str(e))
    elif isinstance(e, TrainingError):
        logger.error(f"Training error: {str(e)}", exc_info=True)
        return HTTPException(status_code=500, detail=str(e))
    else:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return HTTPException(status_code=500, detail=str(e))

