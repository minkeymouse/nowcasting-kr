"""Unified preprocessing pipeline for Korean macroeconomic nowcasting data.

The NowcastingData class is the primary interface for all data preprocessing operations.
"""

import logging
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

# Import sktime components
try:
    from sktime.transformations.compose import TransformerPipeline, ColumnEnsembleTransformer
    from sktime.transformations.series.impute import Imputer
    from sktime.transformations.series.adapt import TabularToSeriesAdaptor
    from sktime.transformations.series.func_transform import FunctionTransformer
    from sktime.transformations.series.difference import Differencer
    from sktime.forecasting.naive import NaiveForecaster
    from sklearn.preprocessing import RobustScaler
    HAS_SKTIME = True
except ImportError:
    HAS_SKTIME = False
    logger.warning("sktime not available. Some features may be limited.")


def load_data(data_path: str = "data/data.csv") -> pd.DataFrame:
    """Load the main data file."""
    data = pd.read_csv(data_path)
    if 'date_w' in data.columns:
        data['date_w'] = pd.to_datetime(data['date_w'], errors='coerce')
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
    return data


def load_metadata(metadata_path: str = "data/metadata.csv") -> pd.DataFrame:
    """Load metadata file with series information."""
    metadata = pd.read_csv(metadata_path)
    if 'SeriesID' in metadata.columns:
        metadata = metadata.rename(columns={'SeriesID': 'Series_ID'})
    return metadata

class NowcastingData:
    """Unified data container with fixed preprocessing pipeline.
    
    Automatically loads data from data/data.csv and applies fixed preprocessing.
    Provides access to original, processed, and standardized data.
    
    Examples
    --------
    >>> from src.preprocess import NowcastingData
    >>> 
    >>> data = NowcastingData()
    >>> original = data.original      # Raw data
    >>> processed = data.processed    # Transformed and imputed
    >>> standardized = data.standardized  # Scaled
    """
    
    def __init__(self):
        """Initialize NowcastingData with fixed preprocessing pipeline."""
        self._raw_data = load_data()
        self._metadata = load_metadata()
        self._preprocessed_data = None
        self._standardized_data = None
        self._scaler = None
        self._pipeline = None
        
        self._preprocess()
        self._standardize()
    
    @property
    def original(self) -> pd.DataFrame:
        """Get original (raw) data."""
        return self._raw_data.copy()
    
    @property
    def processed(self) -> pd.DataFrame:
        """Get processed data (transformed and imputed)."""
        return self._preprocessed_data.copy()
    
    @property
    def standardized(self) -> pd.DataFrame:
        """Get standardized data (scaled)."""
        return self._standardized_data.copy()
    
    @property
    def scaler(self) -> Optional[Any]:
        """Get the fitted scaler used for standardization."""
        return self._scaler
    
    @property
    def transformerpipeline(self) -> Optional[TransformerPipeline]:
        """Get the fitted transformer pipeline used for preprocessing."""
        return self._pipeline
    
    def _preprocess(self) -> None:
        """Preprocess data using TransformerPipeline with sktime Imputer."""
        # Get numeric columns and set up index
        numeric_cols = list(self._raw_data.select_dtypes(include=[np.number]).columns)
        data = self._raw_data[numeric_cols].copy()
        data = self._setup_index(data)
        
        # Build and apply transformer pipeline (Imputer handles all imputation)
        pipeline = self._build_transformer_pipeline(data)
        self._pipeline = pipeline
        
        if pipeline:
            pipeline.fit(data)
            self._preprocessed_data = pipeline.transform(data).sort_index()
        else:
            self._preprocessed_data = data.sort_index()
    
    def _standardize(self) -> None:
        """Apply RobustScaler to processed data."""
        self._scaler = RobustScaler()
        self._scaler.fit(self._preprocessed_data.values)
        scaled_values = self._scaler.transform(self._preprocessed_data.values)
        self._standardized_data = pd.DataFrame(
            scaled_values, 
            index=self._preprocessed_data.index, 
            columns=self._preprocessed_data.columns
        )
    
    def _setup_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """Set up weekly index from date_w column."""
        if 'date_w' in data.columns:
            idx = pd.to_datetime(data['date_w'], errors='coerce')
            data.index = idx
            data = data[~data.index.isna()]
            if data.index.has_duplicates:
                data = data[~data.index.duplicated(keep='last')]
            for col in ['date', 'date_w']:
                if col in data.columns:
                    data = data.drop(columns=[col])
        return data
    
    def _build_transformer_pipeline(self, data: pd.DataFrame) -> Optional[TransformerPipeline]:
        """Build TransformerPipeline using ColumnEnsembleTransformer for per-column transformations."""
        if not HAS_SKTIME:
            return None
        
        steps = []
        
        # Imputation (applied to all columns)
        steps.extend([
            ('imputer_ffill', Imputer(method="ffill")),
            ('imputer_bfill', Imputer(method="bfill")),
            ('imputer_forecaster', Imputer(method="forecaster", forecaster=NaiveForecaster(strategy="last")))
        ])
        
        # Per-column transformations using ColumnEnsembleTransformer
        transformers = self._build_per_series_transformers(data)
        if transformers:
            steps.append(('transformers', ColumnEnsembleTransformer(transformers=transformers)))
        
        return TransformerPipeline(steps) if steps else None
    
    def _build_per_series_transformers(self, data: pd.DataFrame) -> List[Tuple[str, Any, int]]:
        """Build per-series transformers from metadata."""
        if self._metadata is None:
            return []
        
        series_col = 'Series_ID' if 'Series_ID' in self._metadata.columns else 'SeriesID'
        transformers = []
        FREQ_TO_LAG_YOY = {'m': 12, 'q': 4, 'sa': 2, 'a': 1, 'd': 365, 'w': 52}
        FREQ_TO_LAG_STEP = {'m': 1, 'q': 3, 'sa': 6, 'a': 12, 'd': 1, 'w': 1}
        
        for i, col_name in enumerate(data.columns):
            meta_row = self._metadata[self._metadata[series_col] == col_name]
            if len(meta_row) == 0:
                continue
            
            trans = str(meta_row.iloc[0].get('Transformation', 'lin')).lower()
            freq = str(meta_row.iloc[0].get('Frequency', 'm')).lower()
            
            transformer = self._create_series_transformer(trans, freq, FREQ_TO_LAG_YOY, FREQ_TO_LAG_STEP)
            if transformer:
                transformers.append((col_name, transformer, i))
        
        return transformers
    
    def _create_series_transformer(self, trans: str, freq: str, freq_yoy: Dict, freq_step: Dict) -> Optional[Any]:
        """Create transformer for a single series using FunctionTransformer for custom transforms."""
        if not HAS_SKTIME:
            return None
        
        # Use FunctionTransformer for all custom transformations
        if trans == 'lin':
            return FunctionTransformer(func=lambda X: X)
        
        elif trans == 'log':
            def log_transform(X):
                if isinstance(X, pd.Series):
                    vals = X.values
                    mask = vals > 0
                    result = vals.copy()
                    result[mask] = np.log(result[mask])
                    return pd.Series(result, index=X.index, name=X.name)
                return np.log(np.abs(X) + 1e-10)
            return FunctionTransformer(func=log_transform)
        
        elif trans == 'chg':
            # Use built-in Differencer
            return Differencer(lags=freq_step.get(freq, 1))
        
        elif trans == 'ch1':
            # Use built-in Differencer
            return Differencer(lags=freq_yoy.get(freq, 12))
        
        elif trans == 'pch':
            def pct_change_transform(X):
                if isinstance(X, pd.Series):
                    return X.pct_change(periods=1) * 100
                return np.diff(X) / (np.abs(X[:-1]) + 1e-10) * 100
            return FunctionTransformer(func=pct_change_transform)
        
        elif trans == 'pc1':
            lag = freq_yoy.get(freq, 12)
            def pct_change_yoy_transform(X):
                if isinstance(X, pd.Series):
                    return X.pct_change(periods=lag) * 100
                return (X[lag:] - X[:-lag]) / (np.abs(X[:-lag]) + 1e-10) * 100
            return FunctionTransformer(func=pct_change_yoy_transform)
        
        else:
            # Default: identity transform
            return FunctionTransformer(func=lambda X: X)
    
    def __repr__(self) -> str:
        """String representation."""
        n_rows = len(self._preprocessed_data) if self._preprocessed_data is not None else 0
        n_cols = len(self._preprocessed_data.columns) if self._preprocessed_data is not None else 0
        return f"NowcastingData(n_series={n_cols}, n_observations={n_rows})"
