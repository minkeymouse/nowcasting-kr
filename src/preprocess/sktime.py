"""Optional dependency handling for sktime transformers.

This module provides safe imports of sktime transformers with proper
error handling for when sktime is not installed.
"""

try:
    # sktime 0.40+ uses ColumnwiseTransformer instead of ColumnTransformer
    from sktime.transformations.compose import (
        TransformerPipeline,
        ColumnwiseTransformer
    )
    # Try to import ColumnTransformer - may not exist in newer versions
    try:
        from sktime.transformations.compose import ColumnTransformer
    except ImportError:
        # Fallback to sklearn's ColumnTransformer
        from sklearn.compose import ColumnTransformer
    
    from sktime.transformations.series.log import LogTransformer
    from sktime.transformations.series.difference import Differencer
    from sktime.transformations.series.func_transform import FunctionTransformer
    from sklearn.preprocessing import StandardScaler
    HAS_SKTIME = True
except ImportError:
    HAS_SKTIME = False
    ColumnTransformer = None
    TransformerPipeline = None
    ColumnwiseTransformer = None
    LogTransformer = None
    Differencer = None
    FunctionTransformer = None
    StandardScaler = None


def check_sktime_available():
    """Check if sktime is available and raise ImportError if not.
    
    Raises
    ------
    ImportError
        If sktime is not installed, with helpful installation message.
    """
    # Re-check at runtime in case import failed due to missing components
    try:
        import sktime
        # Check if essential components are available
        from sktime.transformations.compose import TransformerPipeline
        from sktime.transformations.series.func_transform import FunctionTransformer
        from sktime.transformations.series.difference import Differencer
        # If we get here, sktime is available
        return
    except ImportError as e:
        raise ImportError(
            f"sktime is required for sktime transformers. "
            f"Install it with: pip install sktime. "
            f"Original error: {e}"
        )

