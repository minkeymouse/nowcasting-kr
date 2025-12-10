"""Helper functions for DFM/DDFM forecasters."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _create_preprocessing_pipeline(model: Any, scaler_type: Optional[str] = None) -> Any:
    """Create preprocessing pipeline for DFM/DDFM model with scaler.
    
    Creates a pipeline with scaler (RobustScaler, StandardScaler, etc.) for
    extracting Mx/Wx statistics and enabling inverse transform during prediction.
    
    Parameters
    ----------
    model : Any
        DFM or DDFM model instance
    scaler_type : Optional[str]
        Scaler type ('robust', 'standard', etc.). If None, tries to get from model config.
        
    Returns
    -------
    Any
        Preprocessing pipeline with scaler transformer
    """
    try:
        from sktime.transformations.compose import TransformerPipeline
        from sklearn.preprocessing import RobustScaler, StandardScaler
        
        # Get scaler type from model config if not provided
        if scaler_type is None:
            if hasattr(model, 'config') and hasattr(model.config, 'scaler'):
                scaler_type = model.config.scaler
            else:
                scaler_type = 'robust'  # Default to robust as per config
        
        # Create scaler based on type
        scaler_type_lower = scaler_type.lower() if scaler_type else 'robust'
        if scaler_type_lower == 'robust':
            scaler = RobustScaler()
        elif scaler_type_lower == 'standard':
            scaler = StandardScaler()
        else:
            logger.warning(f"Unknown scaler type '{scaler_type}', using RobustScaler as fallback")
            scaler = RobustScaler()
        
        # Create pipeline with scaler (passthrough for data, but scaler for statistics extraction)
        # The scaler will be fitted during DataModule.setup() to extract Mx/Wx
        try:
            from dfm_python.lightning.data_module import create_passthrough_transformer
            passthrough = create_passthrough_transformer()
            # Create pipeline: passthrough -> scaler
            # Note: scaler is fitted during setup() to extract statistics
            return TransformerPipeline([
                ("passthrough", passthrough),
                ("scaler", scaler)
            ])
        except ImportError:
            # Fallback: just use scaler directly
            return scaler
    except ImportError as e:
        logger.warning(f"Could not create preprocessing pipeline with scaler: {e}")
        # Fallback: create simple passthrough
        try:
            from sktime.transformations.series.func_transform import FunctionTransformer
            return FunctionTransformer(func=None, inverse_func=None, validate=False)
        except ImportError:
            logger.warning("Could not create preprocessing pipeline, using None")
            return None

