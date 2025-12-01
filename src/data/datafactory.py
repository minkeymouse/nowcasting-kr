"""Data preprocessing and transformation utilities."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any


def validate_csv(file_path: str) -> bool:
    """Validate CSV file format.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    try:
        df = pd.read_csv(file_path, nrows=1)
        
        # Check for Date column
        if 'Date' not in df.columns:
            raise ValueError("CSV must have a 'Date' column")
        
        # Check date format
        try:
            pd.to_datetime(df['Date'].iloc[0])
        except:
            raise ValueError("Date column must be in valid date format (YYYY-MM-DD)")
        
        return True
    except Exception as e:
        raise ValueError(f"CSV validation failed: {str(e)}") from e


def preprocess_data(
    file_path: str,
    transformations: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """Preprocess CSV data for DFM.
    
    Args:
        file_path: Path to CSV file
        transformations: Optional dict of series_id -> transformation type
        
    Returns:
        Preprocessed DataFrame
    """
    # Validate file
    validate_csv(file_path)
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Ensure Date column is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Apply transformations if specified
    if transformations:
        for series_id, transform_type in transformations.items():
            if series_id in df.columns:
                if transform_type == 'log':
                    df[series_id] = np.log(df[series_id] + 1e-10)
                elif transform_type == 'diff':
                    df[series_id] = df[series_id].diff()
                # Add more transformations as needed
    
    return df

