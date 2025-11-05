"""Series creation and management utilities."""

import logging
from typing import Optional, Dict, Any, Callable
from .client import get_client
from .operations import get_series, upsert_series
from .models import SeriesModel
from .helpers import map_frequency_to_code

logger = logging.getLogger(__name__)


class SeriesManager:
    """Manages series creation and retrieval."""
    
    def __init__(self, client=None):
        """Initialize series manager."""
        self.client = client or get_client()
    
    def generate_series_id(
        self,
        source_code: str,
        stat_code: str,
        item_code: Optional[str] = None
    ) -> str:
        """
        Generate consistent series_id.
        
        Format: {SOURCE}_{STAT_CODE} or {SOURCE}_{STAT_CODE}_{ITEM_CODE}
        Examples:
          - BOK_200Y105
          - BOK_200Y105_1101
        """
        if item_code:
            return f"{source_code}_{stat_code}_{item_code}"
        return f"{source_code}_{stat_code}"
    
    def create_or_get_series(
        self,
        source_code: str,
        stat_code: str,
        series_name: str,
        frequency: str,
        statistics_metadata_id: Optional[int] = None,
        item_code: Optional[str] = None,
        item_name: Optional[str] = None,
        units: Optional[str] = None,
        category: Optional[str] = None,
        api_source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create or get existing series.
        
        Parameters
        ----------
        source_code : str
            Source code (e.g., 'BOK', 'KOSIS')
        stat_code : str
            Statistic code (e.g., '200Y105')
        series_name : str
            Human-readable series name
        frequency : str
            Frequency code (d, w, m, q, sa, a)
        statistics_metadata_id : int, optional
            Reference to statistics_metadata
        item_code : str, optional
            Item code if this is an item-specific series
        item_name : str, optional
            Item name for display
        units : str, optional
            Units of measurement
        category : str, optional
            Category classification
        api_source : str, optional
            API source identifier (defaults to source_code if not provided)
        
        Returns
        -------
        Dict[str, Any]
            Series record from database
        """
        series_id = self.generate_series_id(source_code, stat_code, item_code)
        
        # Check if exists
        existing = get_series(series_id=series_id, client=self.client)
        if existing:
            return existing
        
        # Create new series
        # Use source_code as api_source if not provided
        api_source_value = api_source or source_code
        
        series_model = SeriesModel(
            series_id=series_id,
            series_name=series_name,
            frequency=frequency,
            api_source=api_source_value,
            api_code=stat_code,
            units=units,
            category=category
        )
        
        result = upsert_series(
            series=series_model,
            statistics_metadata_id=statistics_metadata_id,
            item_code=item_code,
            client=self.client
        )
        
        logger.debug(f"Created series: {series_id}")
        return result
    
    def create_series_from_item(
        self,
        source_code: str,
        stat_code: str,
        stat_name: str,
        item: Dict[str, Any],
        statistics_metadata_id: Optional[int] = None,
        frequency_mapper: Optional[Callable[[str], str]] = None
    ) -> Dict[str, Any]:
        """
        Create series from statistics item.
        
        Parameters
        ----------
        source_code : str
            Source code (e.g., 'BOK')
        stat_code : str
            Statistic code
        stat_name : str
            Statistic name
        item : Dict[str, Any]
            Item dictionary with item_code, item_name, cycle, unit_name
        statistics_metadata_id : int, optional
            Reference to statistics_metadata
        frequency_mapper : callable, optional
            Function to map cycle to frequency code
        
        Returns
        -------
        Dict[str, Any]
            Series record
        """
        item_code = item.get('item_code')
        item_name = item.get('item_name', '')
        cycle = item.get('cycle', '')
        unit_name = item.get('unit_name')
        
        # Map frequency
        frequency = (frequency_mapper or map_frequency_to_code)(cycle) if cycle else 'm'
        
        # Build series name
        if item_name:
            series_name = f"{stat_name} - {item_name}"
        elif item_code:
            series_name = f"{stat_name} - {item_code}"
        else:
            series_name = stat_name
        
        return self.create_or_get_series(
            source_code=source_code,
            stat_code=stat_code,
            series_name=series_name,
            frequency=frequency,
            statistics_metadata_id=statistics_metadata_id,
            item_code=item_code,
            item_name=item_name,
            units=unit_name,
            category=stat_name
        )

