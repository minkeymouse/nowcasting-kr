"""Unified preprocessing pipeline for Korean macroeconomic nowcasting data.

The BaseData class provides core preprocessing functionality.
ConsumptionData, InvestmentData, and ProductionData are model-specific subclasses.
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


def load_data(data_path: str) -> pd.DataFrame:
    """Load the main data file."""
    data = pd.read_csv(data_path)
    if 'date_w' in data.columns:
        data['date_w'] = pd.to_datetime(data['date_w'], errors='coerce')
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
    return data


def load_metadata(metadata_path: str) -> pd.DataFrame:
    """Load metadata file with series information."""
    metadata = pd.read_csv(metadata_path)
    # Handle both SeriesID and Series_ID column names
    if 'SeriesID' in metadata.columns and 'Series_ID' not in metadata.columns:
        metadata = metadata.rename(columns={'SeriesID': 'Series_ID'})
    # Filter out quarterly series if any exist
    if 'Frequency' in metadata.columns:
        quarterly = metadata[metadata['Frequency'].str.lower() == 'q']
        if len(quarterly) > 0:
            logger.warning(f"Found {len(quarterly)} quarterly series in metadata. Filtering them out.")
            metadata = metadata[metadata['Frequency'].str.lower() != 'q']
    return metadata


class BaseData:
    """Base class for data preprocessing with fixed pipeline.
    
    Provides core preprocessing functionality that can be inherited by
    model-specific data classes (ConsumptionData, InvestmentData, ProductionData).
    
    Examples
    --------
    >>> from src.preprocess import ConsumptionData
    >>> 
    >>> data = ConsumptionData()  # Uses default paths for consumption model
    >>> original = data.original      # Raw data
    >>> processed = data.processed    # Transformed and imputed
    >>> standardized = data.standardized  # Scaled
    >>> training_data = data.training_data  # Ready for model training
    """
    
    # Default paths - overridden by subclasses
    DEFAULT_DATA_PATH = "data/data.csv"
    DEFAULT_METADATA_PATH = "data/metadata.csv"
    
    def __init__(self, data_path: Optional[str] = None, metadata_path: Optional[str] = None):
        """Initialize BaseData with fixed preprocessing pipeline.
        
        Parameters
        ----------
        data_path : str, optional
            Path to the data CSV file. If None, uses class default.
        metadata_path : str, optional
            Path to the metadata CSV file. If None, uses class default.
        """
        self._data_path = data_path or self.DEFAULT_DATA_PATH
        self._metadata_path = metadata_path or self.DEFAULT_METADATA_PATH
        
        logger.info(f"Loading data from: {self._data_path}")
        logger.info(f"Loading metadata from: {self._metadata_path}")
        
        self._raw_data = load_data(self._data_path)
        self._metadata = load_metadata(self._metadata_path)
        
        # Filter data to only include series in metadata (and filter quarterly)
        self._filter_data_to_metadata()
        
        self._preprocessed_data = None
        self._standardized_data = None
        self._scaler = None
        self._pipeline = None
        
        self._preprocess()
        self._standardize()
    
    def _filter_data_to_metadata(self) -> None:
        """Filter raw data to only include series present in metadata."""
        series_col = 'Series_ID' if 'Series_ID' in self._metadata.columns else 'SeriesID'
        if series_col not in self._metadata.columns:
            logger.warning(f"Metadata does not contain '{series_col}' column. Skipping filtering.")
            return
        
        valid_series = set(self._metadata[series_col].values)
        date_cols = ['date', 'date_w']
        
        # Get columns that are either date columns or in metadata
        valid_cols = [col for col in self._raw_data.columns 
                     if col in date_cols or col in valid_series]
        
        # Also check for any quarterly series that might have slipped through
        if 'Frequency' in self._metadata.columns:
            quarterly_series = set(self._metadata[
                self._metadata['Frequency'].str.lower() == 'q'
            ][series_col].values)
            valid_cols = [col for col in valid_cols if col not in quarterly_series]
        
        missing_series = valid_series - set(valid_cols)
        if missing_series:
            logger.warning(f"Series in metadata but not in data: {list(missing_series)[:10]}")
        
        self._raw_data = self._raw_data[valid_cols].copy()
        logger.info(f"Filtered data to {len(valid_cols)} columns ({len(valid_cols) - len(date_cols)} data series)")
    
    @property
    def original(self) -> pd.DataFrame:
        """Get original (raw) data."""
        return self._raw_data.copy()
    
    @property
    def processed(self) -> pd.DataFrame:
        """Get processed data (transformed and imputed)."""
        df = self._preprocessed_data.copy()
        # Add 'date' column from index (which comes from date_w)
        df['date'] = df.index
        # Add year, month, day columns
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        return df
    
    @property
    def standardized(self) -> pd.DataFrame:
        """Get standardized data (scaled)."""
        df = self._standardized_data.copy()
        # Add 'date' column from index (which comes from date_w)
        df['date'] = df.index
        # Add year, month, day columns
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        return df
    
    @property
    def scaler(self) -> Optional[Any]:
        """Get the fitted scaler used for standardization."""
        return self._scaler
    
    @property
    def transformerpipeline(self) -> Optional[TransformerPipeline]:
        """Get the fitted transformer pipeline used for preprocessing."""
        return self._pipeline
    
    @property
    def training_data(self) -> pd.DataFrame:
        """Get training-ready data (standardized, without date columns).
        
        Returns
        -------
        pd.DataFrame
            Standardized data with date columns removed, ready for model training
        """
        return self.standardized.drop(columns=['date', 'year', 'month', 'day'], errors='ignore')
    
    @property
    def metadata(self) -> pd.DataFrame:
        """Get metadata for the loaded series."""
        return self._metadata.copy()
    
    def _preprocess(self) -> None:
        """Preprocess data using TransformerPipeline with sktime Imputer."""
        # Set up index first (needs date_w column)
        data = self._setup_index(self._raw_data.copy())
        
        # Then filter to numeric columns only
        numeric_cols = list(data.select_dtypes(include=[np.number]).columns)
        data = data[numeric_cols]
        
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
            # Sort index (required by sktime)
            data = data.sort_index()
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
            
            # Skip quarterly series
            if freq == 'q':
                logger.warning(f"Skipping quarterly series: {col_name}")
                continue
            
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
        
        elif trans == 'pca':
            # PCA transformation - treat as linear for now
            return FunctionTransformer(func=lambda X: X)
        
        else:
            # Default: identity transform
            return FunctionTransformer(func=lambda X: X)
    
    def __repr__(self) -> str:
        """String representation."""
        n_rows = len(self._preprocessed_data) if self._preprocessed_data is not None else 0
        n_cols = len(self._preprocessed_data.columns) if self._preprocessed_data is not None else 0
        return f"{self.__class__.__name__}(n_series={n_cols}, n_observations={n_rows})"


class ConsumptionData(BaseData):
    """Preprocessing for Consumption model data."""
    DEFAULT_DATA_PATH = "data/consumption.csv"
    DEFAULT_METADATA_PATH = "data/consumption_metadata.csv"


class InvestmentData(BaseData):
    """Preprocessing for Investment model data."""
    DEFAULT_DATA_PATH = "data/investment.csv"
    DEFAULT_METADATA_PATH = "data/investment_metadata.csv"


class ProductionData(BaseData):
    """Preprocessing for Production model data."""
    DEFAULT_DATA_PATH = "data/production.csv"
    DEFAULT_METADATA_PATH = "data/production_metadata.csv"


# Backward compatibility alias
NowcastingData = BaseData
