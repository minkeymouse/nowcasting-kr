"""Base classes and utilities for model training and forecasting."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import pickle
import logging

from src.utils import ValidationError

logger = logging.getLogger(__name__)


class BaseModelTrainer(ABC):
    """Base class for model trainers.
    
    Provides common interface for training and forecasting across all model types.
    """
    
    @abstractmethod
    def train(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any],
        **kwargs
    ) -> Any:
        """Train the model.
        
        Parameters
        ----------
        data : pd.DataFrame
            Training data
        config : Dict[str, Any]
            Model configuration
        **kwargs
            Additional model-specific parameters
            
        Returns
        -------
        Any
            Trained model object
        """
        pass
    
    @abstractmethod
    def forecast(
        self,
        model: Any,
        horizon: int,
        **kwargs
    ) -> pd.DataFrame:
        """Generate forecasts.
        
        Parameters
        ----------
        model : Any
            Trained model object
        horizon : int
            Forecast horizon
        **kwargs
            Additional forecast parameters
            
        Returns
        -------
        pd.DataFrame
            Forecasted values with DatetimeIndex
        """
        pass
    
    @abstractmethod
    def save_checkpoint(
        self,
        model: Any,
        checkpoint_path: Path,
        metadata: Dict[str, Any]
    ) -> None:
        """Save model checkpoint.
        
        Parameters
        ----------
        model : Any
            Trained model object
        checkpoint_path : Path
            Path to save checkpoint
        metadata : Dict[str, Any]
            Additional metadata to save
        """
        pass
    
    @abstractmethod
    def load_checkpoint(
        self,
        checkpoint_path: Path
    ) -> Tuple[Any, Dict[str, Any]]:
        """Load model checkpoint.
        
        Parameters
        ----------
        checkpoint_path : Path
            Path to checkpoint file
            
        Returns
        -------
        Tuple[Any, Dict[str, Any]]
            (model, metadata) tuple
        """
        pass


def save_model_checkpoint(
    model: Any,
    checkpoint_path: Path,
    metadata: Dict[str, Any]
) -> None:
    """Save model checkpoint with metadata."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint_data = {
        'model': model,
        'metadata': metadata
    }
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def load_model_checkpoint(checkpoint_path: Path) -> Tuple[Any, Dict[str, Any]]:
    """Load model checkpoint with metadata."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    model = checkpoint_data.get('model')
    metadata = checkpoint_data.get('metadata', {})
    
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return model, metadata

