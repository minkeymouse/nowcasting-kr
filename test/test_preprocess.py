"""Test preprocessing pipeline for all three model data types."""

import sys
from pathlib import Path
import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocess import (
    ConsumptionData, 
    InvestmentData, 
    ProductionData,
    BaseData,
    load_data, 
    load_metadata
)


# Test all three model data classes
MODEL_DATA_CLASSES = [
    ('consumption', ConsumptionData, 'data/consumption.csv', 'data/consumption_metadata.csv'),
    ('investment', InvestmentData, 'data/investment.csv', 'data/investment_metadata.csv'),
    ('production', ProductionData, 'data/production.csv', 'data/production_metadata.csv'),
]


@pytest.mark.parametrize("model_name,data_class,data_path,metadata_path", MODEL_DATA_CLASSES)
def test_data_files_exist(model_name, data_class, data_path, metadata_path):
    """Test that model-specific data files exist."""
    data_file = Path(data_path)
    metadata_file = Path(metadata_path)
    
    assert data_file.exists(), f"{model_name} data file not found: {data_file}"
    assert metadata_file.exists(), f"{model_name} metadata file not found: {metadata_file}"


@pytest.mark.parametrize("model_name,data_class,data_path,metadata_path", MODEL_DATA_CLASSES)
def test_load_data(model_name, data_class, data_path, metadata_path):
    """Test data loading for each model."""
    data = load_data(data_path)
    
    assert isinstance(data, pd.DataFrame), f"{model_name} data should be a DataFrame"
    assert len(data) > 0, f"{model_name} data should have rows"
    assert len(data.columns) > 0, f"{model_name} data should have columns"
    
    # Check for date columns
    assert 'date_w' in data.columns or 'date' in data.columns, f"{model_name} data should have date column"


@pytest.mark.parametrize("model_name,data_class,data_path,metadata_path", MODEL_DATA_CLASSES)
def test_load_metadata(model_name, data_class, data_path, metadata_path):
    """Test metadata loading for each model."""
    metadata = load_metadata(metadata_path)
    
    assert isinstance(metadata, pd.DataFrame), f"{model_name} metadata should be a DataFrame"
    assert len(metadata) > 0, f"{model_name} metadata should have rows"
    
    # Check for required columns
    assert 'Series_ID' in metadata.columns or 'SeriesID' in metadata.columns, f"{model_name} metadata should have SeriesID column"
    assert 'Transformation' in metadata.columns, f"{model_name} metadata should have Transformation column"
    assert 'Frequency' in metadata.columns, f"{model_name} metadata should have Frequency column"
    
    # Check no quarterly series
    quarterly = metadata[metadata['Frequency'].str.lower() == 'q']
    assert len(quarterly) == 0, f"{model_name} metadata should have no quarterly series"


@pytest.mark.parametrize("model_name,data_class,data_path,metadata_path", MODEL_DATA_CLASSES)
def test_data_class_initialization(model_name, data_class, data_path, metadata_path):
    """Test data class initialization for each model."""
    data = data_class()
    
    assert data is not None, f"{data_class.__name__} should initialize"
    assert isinstance(data, BaseData), f"{data_class.__name__} should inherit from BaseData"
    assert data.__class__.__name__ == data_class.__name__, f"Should be {data_class.__name__} instance"


@pytest.mark.parametrize("model_name,data_class,data_path,metadata_path", MODEL_DATA_CLASSES)
def test_original_data(model_name, data_class, data_path, metadata_path):
    """Test original data property for each model."""
    data = data_class()
    original = data.original
    
    assert isinstance(original, pd.DataFrame), f"{model_name} original should be a DataFrame"
    assert len(original) > 0, f"{model_name} original data should have rows"
    assert len(original.columns) > 0, f"{model_name} original data should have columns"


@pytest.mark.parametrize("model_name,data_class,data_path,metadata_path", MODEL_DATA_CLASSES)
def test_processed_data(model_name, data_class, data_path, metadata_path):
    """Test processed data property for each model."""
    data = data_class()
    processed = data.processed
    
    assert isinstance(processed, pd.DataFrame), f"{model_name} processed should be a DataFrame"
    assert len(processed) > 0, f"{model_name} processed data should have rows"
    assert len(processed.columns) > 0, f"{model_name} processed data should have columns"
    assert isinstance(processed.index, pd.DatetimeIndex), f"{model_name} processed data should have DatetimeIndex, got {type(processed.index)}"
    
    # Check for missing values (should be imputed)
    assert processed.isnull().sum().sum() == 0, f"{model_name} processed data should have no missing values"


@pytest.mark.parametrize("model_name,data_class,data_path,metadata_path", MODEL_DATA_CLASSES)
def test_standardized_data(model_name, data_class, data_path, metadata_path):
    """Test standardized data property for each model."""
    data = data_class()
    standardized = data.standardized
    
    assert isinstance(standardized, pd.DataFrame), f"{model_name} standardized should be a DataFrame"
    assert len(standardized) > 0, f"{model_name} standardized data should have rows"
    assert len(standardized.columns) > 0, f"{model_name} standardized data should have columns"
    assert isinstance(standardized.index, pd.DatetimeIndex), f"{model_name} standardized data should have DatetimeIndex, got {type(standardized.index)}"
    
    # Check that standardized data has no missing values
    assert standardized.isnull().sum().sum() == 0, f"{model_name} standardized data should have no missing values"


@pytest.mark.parametrize("model_name,data_class,data_path,metadata_path", MODEL_DATA_CLASSES)
def test_scaler(model_name, data_class, data_path, metadata_path):
    """Test scaler property for each model."""
    data = data_class()
    scaler = data.scaler
    
    assert scaler is not None, f"{model_name} scaler should be available"
    assert hasattr(scaler, 'transform'), f"{model_name} scaler should have transform method"
    assert hasattr(scaler, 'inverse_transform'), f"{model_name} scaler should have inverse_transform method"


@pytest.mark.parametrize("model_name,data_class,data_path,metadata_path", MODEL_DATA_CLASSES)
def test_transformer_pipeline(model_name, data_class, data_path, metadata_path):
    """Test transformer pipeline property for each model."""
    data = data_class()
    pipeline = data.transformerpipeline
    
    assert pipeline is not None, f"{model_name} transformer pipeline should be available"


@pytest.mark.parametrize("model_name,data_class,data_path,metadata_path", MODEL_DATA_CLASSES)
def test_training_data(model_name, data_class, data_path, metadata_path):
    """Test training_data property for each model."""
    data = data_class()
    training_data = data.training_data
    
    assert isinstance(training_data, pd.DataFrame), f"{model_name} training_data should be a DataFrame"
    assert len(training_data) > 0, f"{model_name} training_data should have rows"
    assert len(training_data.columns) > 0, f"{model_name} training_data should have columns"
    
    # Training data should not have date columns
    assert 'date' not in training_data.columns, f"{model_name} training_data should not have 'date' column"
    assert 'year' not in training_data.columns, f"{model_name} training_data should not have 'year' column"
    assert 'month' not in training_data.columns, f"{model_name} training_data should not have 'month' column"
    assert 'day' not in training_data.columns, f"{model_name} training_data should not have 'day' column"


@pytest.mark.parametrize("model_name,data_class,data_path,metadata_path", MODEL_DATA_CLASSES)
def test_data_consistency(model_name, data_class, data_path, metadata_path):
    """Test that original, processed, and standardized have consistent structure for each model."""
    data = data_class()
    
    original = data.original
    processed = data.processed
    standardized = data.standardized
    
    # All should have same number of numeric columns (after filtering)
    numeric_cols_original = list(original.select_dtypes(include=['number']).columns)
    # Processed should have numeric columns + date, year, month, day columns (4 total)
    assert len(processed.columns) == len(numeric_cols_original) + 4, f"{model_name} processed should have numeric columns + date, year, month, day columns"
    assert len(standardized.columns) == len(processed.columns), f"{model_name} standardized should have same columns as processed"
    
    # Check that date columns exist
    assert 'date' in processed.columns, f"{model_name} processed should have 'date' column"
    assert 'year' in processed.columns, f"{model_name} processed should have 'year' column"
    assert 'month' in processed.columns, f"{model_name} processed should have 'month' column"
    assert 'day' in processed.columns, f"{model_name} processed should have 'day' column"
    assert 'date' in standardized.columns, f"{model_name} standardized should have 'date' column"
    
    # Check date column is datetime
    assert pd.api.types.is_datetime64_any_dtype(processed['date']), f"{model_name} date column should be datetime"
    assert pd.api.types.is_datetime64_any_dtype(standardized['date']), f"{model_name} date column should be datetime"
    
    # Check date column equals index (from date_w)
    assert (processed['date'] == processed.index).all(), f"{model_name} date column should equal index (from date_w)"
    
    # All should have same index
    assert len(processed) == len(standardized), f"{model_name} processed and standardized should have same number of rows"
    assert processed.index.equals(standardized.index), f"{model_name} processed and standardized should have same index"


@pytest.mark.parametrize("model_name,data_class,data_path,metadata_path", MODEL_DATA_CLASSES)
def test_metadata_property(model_name, data_class, data_path, metadata_path):
    """Test metadata property for each model."""
    data = data_class()
    metadata = data.metadata
    
    assert isinstance(metadata, pd.DataFrame), f"{model_name} metadata property should return DataFrame"
    assert len(metadata) > 0, f"{model_name} metadata should have rows"
    
    # Verify no quarterly series
    quarterly = metadata[metadata['Frequency'].str.lower() == 'q']
    assert len(quarterly) == 0, f"{model_name} metadata should have no quarterly series"


def test_inheritance_structure():
    """Test that all data classes inherit from BaseData."""
    assert issubclass(ConsumptionData, BaseData), "ConsumptionData should inherit from BaseData"
    assert issubclass(InvestmentData, BaseData), "InvestmentData should inherit from BaseData"
    assert issubclass(ProductionData, BaseData), "ProductionData should inherit from BaseData"


def test_default_paths():
    """Test that each data class has correct default paths."""
    assert ConsumptionData.DEFAULT_DATA_PATH == "data/consumption.csv"
    assert ConsumptionData.DEFAULT_METADATA_PATH == "data/consumption_metadata.csv"
    
    assert InvestmentData.DEFAULT_DATA_PATH == "data/investment.csv"
    assert InvestmentData.DEFAULT_METADATA_PATH == "data/investment_metadata.csv"
    
    assert ProductionData.DEFAULT_DATA_PATH == "data/production.csv"
    assert ProductionData.DEFAULT_METADATA_PATH == "data/production_metadata.csv"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
