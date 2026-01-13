"""Unified preprocessing pipeline for Korean macroeconomic nowcasting data.

The BaseData class provides core preprocessing functionality.
InvestmentData and ProductionData are model-specific subclasses.
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
    from sklearn.preprocessing import StandardScaler
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
    model-specific data classes (InvestmentData, ProductionData).
    
    Examples
    --------
    >>> from src.preprocess import InvestmentData
    >>> 
    >>> data = InvestmentData()  # Uses default paths for investment model
    >>> original = data.original      # Raw data
    >>> processed = data.processed    # Transformed and imputed (differencing, log, etc.)
    >>> standardized = data.standardized  # Same as processed (for backward compatibility)
    >>> training_data = data.training_data  # Processed data without date columns
    >>> # Note: Standardization removed - models handle scaling internally
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
        self._scaler = None
        self._pipeline = None
        
        self._preprocess()
        # Standardization removed - models handle scaling internally
        # For backward compatibility, standardized property returns processed data
    
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
        """Get standardized data (for backward compatibility).
        
        Note: Standardization has been removed from preprocessing.
        This property now returns processed (transformed) data.
        Models are expected to handle scaling internally.
        """
        # Return processed data for backward compatibility
        return self.processed
    
    @property
    def scaler(self) -> Optional[Any]:
        """Get the fitted scaler (for backward compatibility).
        
        Note: Standardization has been removed. This property returns None.
        """
        return None  # No scaler - standardization removed
    
    @property
    def transformerpipeline(self) -> Optional[TransformerPipeline]:
        """Get the fitted transformer pipeline used for preprocessing."""
        return self._pipeline
    
    @property
    def training_data(self) -> pd.DataFrame:
        """Get training-ready data (processed/transformed, without date columns).
        
        Returns
        -------
        pd.DataFrame
            Processed (transformed) data with date columns removed, ready for model training.
            Note: Data is NOT standardized - models handle scaling internally.
        """
        return self.processed.drop(columns=['date', 'year', 'month', 'day'], errors='ignore')
    
    @property
    def metadata(self) -> pd.DataFrame:
        """Get metadata for the loaded series."""
        return self._metadata.copy()
    
    def _preprocess(self) -> None:
        """Preprocess data using a two-path pipeline for mixed-frequency data.
        
        Pipeline Overview:
        ==================
        For mixed-frequency data (weekly clock with monthly/slower series):
        
        1. CLOCK-FREQUENCY SERIES (weekly):
           - Impute missing values (forward fill, backward fill, forecast)
           - Apply transformations (differencing, log, etc.) at weekly frequency
           
        2. SLOWER-FREQUENCY SERIES (monthly/quarterly):
           - Aggregate to native frequency (e.g., monthly -> month-end values)
           - Apply transformations at native frequency (preserves values)
           - Re-expand to weekly format (values at period-end weeks, NaN elsewhere)
           
        This approach prevents data loss when differencing sparse series.
        
        Steps:
        ------
        1. Setup index from date_w column
        2. Identify slower-frequency vs clock-frequency series
        3. Process slower-frequency series: aggregate -> transform -> re-expand
        4. Process clock-frequency series: impute -> transform
        5. Combine all series
        """
        # Step 1: Set up weekly index from date_w column
        data = self._setup_index(self._raw_data.copy())
        
        # Step 2: Filter to numeric columns only
        numeric_cols = list(data.select_dtypes(include=[np.number]).columns)
        data = data[numeric_cols]
        
        # Step 3: Identify slower-frequency series (monthly, quarterly, etc.)
        clock_freq = 'w'  # Weekly clock frequency
        slower_freq_series = self._identify_slower_frequency_series(data, clock_freq)
        clock_freq_cols = [col for col in data.columns if col not in slower_freq_series]
        
        # Step 4: Process slower-frequency series (aggregate -> transform -> re-expand)
        if slower_freq_series:
            logger.info(f"Processing {len(slower_freq_series)} slower-frequency series "
                       f"(aggregate to native frequency, transform, then re-expand).")
            slower_data_transformed = self._transform_slower_frequency_series(
                data[slower_freq_series], clock_freq
            )
        
        # Step 5: Process clock-frequency series (impute -> transform)
        if clock_freq_cols and HAS_SKTIME:
            clock_data = data[clock_freq_cols].copy()
            
            # 5a. Impute missing values for clock-frequency series
            imputer_pipeline = TransformerPipeline([
                ('ffill', Imputer(method="ffill")),
                ('bfill', Imputer(method="bfill")),
                ('forecaster', Imputer(method="forecaster", forecaster=NaiveForecaster(strategy="last")))
            ])
            imputer_pipeline.fit(clock_data)
            clock_data_imputed = imputer_pipeline.transform(clock_data)
            
            # 5b. Apply transformations for clock-frequency series
            clock_pipeline = self._build_transformer_pipeline(clock_data_imputed)
            self._pipeline = clock_pipeline  # Store pipeline for consistency
            if clock_pipeline:
                clock_pipeline.fit(clock_data_imputed)
                clock_data_transformed = clock_pipeline.transform(clock_data_imputed)
            else:
                clock_data_transformed = clock_data_imputed
        elif clock_freq_cols:
            # No sktime available, keep clock-frequency data as-is
            clock_data_transformed = data[clock_freq_cols].copy()
            self._pipeline = None
        else:
            # No clock-frequency columns
            clock_data_transformed = pd.DataFrame(index=data.index)
        
        # Step 6: Combine all processed series
        if slower_freq_series and not clock_freq_cols:
            # Only slower-frequency series
            data = slower_data_transformed
        elif slower_freq_series:
            # Both types of series
            data = pd.concat([clock_data_transformed, slower_data_transformed], axis=1)
        else:
            # Only clock-frequency series - also need imputation if not done above
            if HAS_SKTIME:
                imputer_pipeline = TransformerPipeline([
                    ('ffill', Imputer(method="ffill")),
                    ('bfill', Imputer(method="bfill")),
                    ('forecaster', Imputer(method="forecaster", forecaster=NaiveForecaster(strategy="last")))
                ])
                imputer_pipeline.fit(data)
                data = imputer_pipeline.transform(data)
            
            # Build and apply transformation pipeline
            pipeline = self._build_transformer_pipeline(data)
            self._pipeline = pipeline
            
            if pipeline:
                pipeline.fit(data)
                data = pipeline.transform(data)
        
        self._preprocessed_data = data.sort_index()
    
    # Standardization removed - models handle scaling internally
    # def _standardize(self) -> None:
    #     """Apply StandardScaler to processed data."""
    #     # No longer used - standardization removed from preprocessing pipeline
    
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
        """Build TransformerPipeline using ColumnEnsembleTransformer for per-column transformations.
        
        Note: Imputation is now handled separately in _preprocess() to preserve NaN
        for slower-frequency series. This pipeline only handles transformations.
        """
        if not HAS_SKTIME:
            return None
        
        steps = []
        
        # Per-column transformations using ColumnEnsembleTransformer
        # (Imputation is handled separately in _preprocess() to preserve NaN for slower-frequency series)
        transformers = self._build_per_series_transformers(data)
        if transformers:
            steps.append(('transformers', ColumnEnsembleTransformer(transformers=transformers)))
        
        return TransformerPipeline(steps) if steps else None
    
    def _transform_slower_frequency_series(self, data: pd.DataFrame, clock_freq: str = 'w') -> pd.DataFrame:
        """Transform slower-frequency series using aggregation approach.
        
        Algorithm:
        =========
        For monthly (or other slower-frequency) series embedded in a weekly dataset:
        
        1. AGGREGATE: Resample to native frequency (e.g., weekly -> monthly)
           - Uses period-end aggregation (last value in each period)
           - Example: Weekly data -> Month-end values
        
        2. TRANSFORM: Apply transformations at native frequency
           - Differencing, log, percent change, etc. applied on aggregated data
           - Prevents data loss that occurs when differencing sparse data
        
        3. RE-EXPAND: Map transformed values back to weekly index
           - Place transformed values at the closest weekly date <= period-end
           - Fill remaining weeks with NaN (preserves sparsity structure)
        
        Why This Works:
        ===============
        Direct differencing on sparse monthly data in weekly format loses most values
        because consecutive weeks are NaN. By aggregating first, we have dense monthly
        data where differencing produces valid results.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with slower-frequency series as columns (weekly index)
        clock_freq : str
            Clock frequency of the dataset ('w' for weekly)
            
        Returns
        -------
        pd.DataFrame
            Transformed series in weekly format (same index as input)
            Values at period-end weeks, NaN elsewhere
        """
        if self._metadata is None:
            return data
        
        result = pd.DataFrame(index=data.index)
        series_col = 'Series_ID' if 'Series_ID' in self._metadata.columns else 'SeriesID'
        
        # Lag parameters for year-over-year transformations
        FREQ_TO_LAG_YOY = {'m': 12, 'q': 4, 'sa': 2, 'a': 1, 'd': 365, 'w': 52}
        FREQ_TO_LAG_STEP = {'m': 1, 'q': 3, 'sa': 6, 'a': 12, 'd': 1, 'w': 1}
        
        # Mapping to pandas resample frequency codes (modern format)
        # Uses 'ME' (Month End), 'QE' (Quarter End), 'YE' (Year End) instead of deprecated codes
        FREQ_TO_RESAMPLE = {
            'm': 'ME',    # Month-end (replaces deprecated 'M')
            'q': 'QE',    # Quarter-end (replaces deprecated 'Q')
            'sa': '6ME',  # Semi-annual (6 months, month-end)
            'a': 'YE',    # Year-end (replaces deprecated 'A')
            'w': 'W',     # Week
            'd': 'D'      # Day
        }
        
        for col_name in data.columns:
            # Get metadata for this series
            meta_row = self._metadata[self._metadata[series_col] == col_name]
            if len(meta_row) == 0:
                # No metadata found, keep series as-is
                result[col_name] = data[col_name]
                continue
            
            freq = str(meta_row.iloc[0].get('Frequency', clock_freq)).lower()
            trans = str(meta_row.iloc[0].get('Transformation', 'lin')).lower()
            series = data[col_name].copy()
            
            # Skip if frequency matches clock or is quarterly (not supported)
            if freq == 'q' or freq == clock_freq.lower():
                result[col_name] = series
                continue
            
            # ============================================================
            # STEP 1: Aggregate to native frequency (period-end values)
            # ============================================================
            resample_freq = FREQ_TO_RESAMPLE.get(freq, 'ME')
            try:
                aggregated = series.resample(resample_freq).last()  # Take last value in each period
            except Exception as e:
                logger.warning(f"Failed to resample {col_name} to {resample_freq}: {e}. Keeping original.")
                result[col_name] = series
                continue
            
            # ============================================================
            # STEP 2: Apply transformation at native frequency
            # ============================================================
            if trans == 'lin':
                transformed = aggregated  # No transformation
            elif trans == 'log':
                transformed = aggregated.copy()
                mask = transformed.notna() & (transformed > 0)
                transformed.loc[mask] = np.log(transformed.loc[mask])
            elif trans == 'chg':
                # First difference at native frequency
                transformed = aggregated.diff()
            elif trans == 'ch1':
                # Year-over-year difference
                lag = FREQ_TO_LAG_YOY.get(freq, 12)
                transformed = aggregated.diff(periods=lag)
            elif trans == 'pch':
                # Percent change (period-over-period)
                transformed = aggregated.pct_change(periods=1) * 100
            elif trans == 'pc1':
                # Year-over-year percent change
                lag = FREQ_TO_LAG_YOY.get(freq, 12)
                transformed = aggregated.pct_change(periods=lag) * 100
            elif trans in ['logchg', 'logdiff']:
                # Log-differencing: log(x_t) - log(x_{t-1})
                transformed = aggregated.copy()
                mask = transformed.notna() & (transformed > 0)
                transformed.loc[mask] = np.log(transformed.loc[mask])
                transformed = transformed.diff()
            else:
                # Unknown transformation, keep as-is
                transformed = aggregated
            
            # ============================================================
            # STEP 3: Re-expand to weekly format
            # ============================================================
            # Create empty series with weekly index
            expanded = pd.Series(index=data.index, dtype=float)
            
            # For each transformed period-end value, find closest weekly date <= period-end
            for period_end, value in transformed.dropna().items():
                # Find all weekly dates <= period-end, take the last one
                candidates = expanded.index[expanded.index <= period_end]
                if len(candidates) > 0:
                    expanded.loc[candidates[-1]] = value
            # Remaining weeks remain NaN (preserves sparsity)
            
            result[col_name] = expanded
        
        return result
    
    def _identify_slower_frequency_series(self, data: pd.DataFrame, clock_freq: str = 'w') -> List[str]:
        """Identify series with frequencies slower than the clock frequency.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with series as columns
        clock_freq : str
            Clock frequency ('w' for weekly, 'm' for monthly, etc.)
            
        Returns
        -------
        List[str]
            List of column names that are slower-frequency (should preserve NaN)
        """
        if self._metadata is None:
            return []
        
        # Frequency hierarchy: lower number = higher frequency (faster)
        # Weekly clock: w=52, m=12, q=4, sa=2, a=1
        FREQUENCY_HIERARCHY = {
            'w': 52,   # Weekly (fastest for our use case)
            'm': 12,   # Monthly
            'q': 4,    # Quarterly
            'sa': 2,   # Semi-annual
            'a': 1,    # Annual (slowest)
            'd': 365   # Daily (faster, but not typically used as clock)
        }
        
        clock_hierarchy = FREQUENCY_HIERARCHY.get(clock_freq.lower(), 999)
        slower_series = []
        
        series_col = 'Series_ID' if 'Series_ID' in self._metadata.columns else 'SeriesID'
        
        for col_name in data.columns:
            meta_row = self._metadata[self._metadata[series_col] == col_name]
            if len(meta_row) == 0:
                continue
            
            freq = str(meta_row.iloc[0].get('Frequency', clock_freq)).lower()
            freq_hierarchy = FREQUENCY_HIERARCHY.get(freq, clock_hierarchy)
            
            # Series with hierarchy < clock_hierarchy are slower (fewer observations per year)
            # For weekly clock: monthly (12) < weekly (52), so monthly is slower
            if freq_hierarchy < clock_hierarchy:
                slower_series.append(col_name)
        
        return slower_series
    
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
        
        elif trans == 'logchg' or trans == 'logdiff':
            # Log-differencing: log(x_t) - log(x_{t-1}) = log(x_t / x_{t-1})
            # Chain log transformation with differencing
            lag = freq_step.get(freq, 1)
            def log_transform(X):
                if isinstance(X, pd.Series):
                    vals = X.values
                    mask = vals > 0
                    result = vals.copy()
                    result[mask] = np.log(result[mask])
                    return pd.Series(result, index=X.index, name=X.name)
                return np.log(np.abs(X) + 1e-10)
            
            # Create a pipeline: log then differencing
            log_transformer = FunctionTransformer(func=log_transform)
            differencer = Differencer(lags=lag)
            # Use transformer composition (pipeline)
            return TransformerPipeline([log_transformer, differencer])
        
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


class InvestmentData(BaseData):
    """Preprocessing for Investment model data."""
    DEFAULT_DATA_PATH = "data/train_investment.csv"
    DEFAULT_METADATA_PATH = "data/investment_metadata.csv"


class ProductionData(BaseData):
    """Preprocessing for Production model data."""
    DEFAULT_DATA_PATH = "data/train_production.csv"
    DEFAULT_METADATA_PATH = "data/production_metadata.csv"


