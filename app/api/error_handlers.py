"""Error handling utilities for API routes."""

from functools import wraps
from fastapi import HTTPException
from typing import Callable, Any

from utils import ConfigError, ValidationError, ModelNotFoundError, TrainingError


def handle_exceptions(func: Callable) -> Callable:
    """Decorator to handle common exceptions and convert to HTTPException."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            raise
        except ConfigError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except ModelNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except TrainingError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return wrapper


def to_http_exception(e: Exception) -> HTTPException:
    """Convert custom exceptions to HTTPException."""
    if isinstance(e, HTTPException):
        return e
    elif isinstance(e, (ConfigError, ModelNotFoundError)):
        return HTTPException(status_code=404, detail=str(e))
    elif isinstance(e, ValidationError):
        return HTTPException(status_code=400, detail=str(e))
    elif isinstance(e, TrainingError):
        return HTTPException(status_code=500, detail=str(e))
    else:
        return HTTPException(status_code=500, detail=str(e))

