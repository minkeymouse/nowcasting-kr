"""Base ingestion class for data sources.

This class combines API client, parsing, and metadata collection into a single interface.
Each data source (BOK, KOSIS) should implement this base class.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime, date
import pandas as pd

from database import (
    get_client,
    upsert_statistics_metadata,
    upsert_statistics_items,
    get_statistics_metadata,
    list_dfm_selected_statistics,
)
from database.models import StatisticsMetadataModel, StatisticsItemModel
from database.operations import create_series_from_item, create_or_get_series
from database.helpers import map_frequency_to_code
from services.api.base import BaseAPIClient

logger = logging.getLogger(__name__)


class BaseIngestion(ABC):
    """Base class for data source ingestion (API + parsing + metadata collection)."""
    
    def __init__(self, api_client: BaseAPIClient, create_series: bool = False):
        """
        Initialize ingestion handler.
        
        Parameters
        ----------
        api_client : BaseAPIClient
            API client for the data source
        create_series : bool
            Whether to create series during metadata collection
        """
        self.api_client = api_client
        self.client = get_client()
        self.create_series = create_series
        self._source_cache = {}  # Cache for source_id lookups
    
    @property
    def source_code(self) -> str:
        """Return the data source code."""
        return self.api_client.source_code
    
    @property
    def source_id(self) -> int:
        """Get source_id from database (cached)."""
        if self.source_code not in self._source_cache:
            # data_sources removed, source_id is just source_code
            self._source_cache[self.source_code] = self.source_code
        return self._source_cache[self.source_code]
    
    # Parsing methods (abstract)
    @abstractmethod
    def transform_statistics_list(self, response: Dict[str, Any]) -> pd.DataFrame:
        """Transform statistics list API response to DataFrame."""
        pass
    
    @abstractmethod
    def transform_statistic_items(
        self,
        response: Dict[str, Any],
        statistics_metadata_id: int
    ) -> pd.DataFrame:
        """Transform statistic items API response to DataFrame."""
        pass
    
    @abstractmethod
    def transform_statistic_data(
        self,
        response: Dict[str, Any],
        stat_code: str,
        frequency: str,
        item_code: Optional[str] = None
    ) -> pd.DataFrame:
        """Transform statistic data API response to DataFrame."""
        pass
    
    # Metadata collection methods
    def _load_dfm_selection(
        self,
        update_dfm_selection: bool,
        dfm_selection_csv: Optional[str]
    ) -> tuple[set, Dict[str, int]]:
        """Load DFM selection from CSV if provided."""
        dfm_selected_codes = set()
        dfm_priorities = {}
        if update_dfm_selection and dfm_selection_csv:
            try:
                df_dfm = pd.read_csv(dfm_selection_csv)
                dfm_selected_codes = set(df_dfm['stat_code'].tolist())
                for idx, row in df_dfm.iterrows():
                    dfm_priorities[row['stat_code']] = idx + 1
                logger.info(f"Loaded {len(dfm_selected_codes)} DFM-selected statistics from CSV")
            except Exception as e:
                logger.warning(f"Failed to load DFM selection CSV: {e}")
        return dfm_selected_codes, dfm_priorities
    
    def _build_metadata_from_row(
        self,
        row: Dict[str, Any],
        dfm_selected_codes: set,
        dfm_priorities: Dict[str, int],
        existing: Optional[Dict[str, Any]]
    ) -> StatisticsMetadataModel:
        """
        Build StatisticsMetadataModel from a row (source-specific).
        
        Subclasses should override this to handle source-specific row formats.
        """
        raise NotImplementedError("Subclasses must implement _build_metadata_from_row")
    
    def _fetch_statistics_list_pages(self) -> pd.DataFrame:
        """
        Fetch all statistics list pages and return as DataFrame.
        
        Subclasses should override this for source-specific pagination logic.
        """
        raise NotImplementedError("Subclasses must implement _fetch_statistics_list_pages")
    
    def collect_statistics_list(
        self,
        update_dfm_selection: bool = False,
        dfm_selection_csv: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch all statistics and save to statistics_metadata.
        
        Common implementation that handles DFM selection and metadata upsert.
        Source-specific logic is delegated to _fetch_statistics_list_pages and _build_metadata_from_row.
        """
        logger.info(f"Starting statistics list collection for {self.source_code}")
        
        stats = {
            'total': 0,
            'inserted': 0,
            'updated': 0,
            'errors': []
        }
        
        # Load DFM selection
        dfm_selected_codes, dfm_priorities = self._load_dfm_selection(
            update_dfm_selection, dfm_selection_csv
        )
        
        try:
            # Fetch statistics list (source-specific)
            df_stats = self._fetch_statistics_list_pages()
            
            if df_stats.empty:
                logger.warning("No statistics found")
                return stats
            
            stats['total'] = len(df_stats)
            
            # Process each statistic
            for _, row in df_stats.iterrows():
                try:
                    stat_code = self._extract_stat_code(row)
                    if not stat_code:
                        continue
                    
                    # Check if already exists
                    existing = get_statistics_metadata(
                        source_id=self.source_id,
                        source_stat_code=stat_code,
                        client=self.client
                    )
                    
                    # Build metadata model (source-specific)
                    metadata = self._build_metadata_from_row(
                        row, dfm_selected_codes, dfm_priorities, existing
                    )
                    
                    # Upsert
                    result = upsert_statistics_metadata(metadata, client=self.client)
                    
                    if existing:
                        stats['updated'] += 1
                    else:
                        stats['inserted'] += 1
                        
                except Exception as e:
                    error_msg = f"Error processing statistic: {str(e)}"
                    stats['errors'].append(error_msg)
                    logger.error(error_msg, exc_info=True)
            
            logger.info(f"Statistics list collection completed: {stats['total']} total, "
                       f"{stats['inserted']} inserted, {stats['updated']} updated")
            
        except Exception as e:
            error_msg = f"Failed to collect statistics list: {str(e)}"
            stats['errors'].append(error_msg)
            logger.error(error_msg, exc_info=True)
        
        return stats
    
    def _extract_stat_code(self, row: Dict[str, Any]) -> Optional[str]:
        """
        Extract statistic code from row (source-specific).
        
        Subclasses should override this.
        """
        raise NotImplementedError("Subclasses must implement _extract_stat_code")
    
    def collect_statistic_items(
        self,
        source_stat_code: str,
        statistics_metadata_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Collect items for a specific statistic.
        
        Common implementation that handles item parsing and upsert.
        Source-specific logic is in transform_statistic_items.
        """
        logger.info(f"Collecting items for statistic: {source_stat_code}")
        
        stats = {
            'total': 0,
            'inserted': 0,
            'updated': 0,
            'errors': []
        }
        
        try:
            # Get statistics_metadata_id if not provided
            if statistics_metadata_id is None:
                metadata = get_statistics_metadata(
                    source_id=self.source_id,
                    source_stat_code=source_stat_code,
                    client=self.client
                )
                if not metadata:
                    raise ValueError(f"Statistics metadata not found for {source_stat_code}")
                statistics_metadata_id = metadata['id']
            
            # Fetch item list (source-specific parameters)
            response = self.api_client.fetch_statistic_items(
                stat_code=source_stat_code,
                start_count=1,
                end_count=1000
            )
            
            # Parse to DataFrame (source-specific)
            df_items = self.transform_statistic_items(response, statistics_metadata_id)
            
            if df_items.empty:
                logger.warning(f"No items found for {source_stat_code}")
                return stats
            
            stats['total'] = len(df_items)
            
            # Convert to StatisticsItemModel list
            items = []
            for _, row in df_items.iterrows():
                try:
                    item = StatisticsItemModel(
                        statistics_metadata_id=statistics_metadata_id,
                        item_code=row['item_code'],
                        item_name=row.get('item_name'),
                        item_name_eng=row.get('item_name_eng'),
                        parent_item_code=row.get('parent_item_code'),
                        parent_item_name=row.get('parent_item_name'),
                        grp_code=row.get('grp_code'),
                        grp_name=row.get('grp_name'),
                        cycle=row.get('cycle', 'M'),
                        start_time=row.get('start_time'),
                        end_time=row.get('end_time'),
                        data_count=row.get('data_count'),
                        unit_name=row.get('unit_name'),
                        weight=row.get('weight'),
                    )
                    items.append(item)
                except Exception as e:
                    error_msg = f"Error processing item {row.get('item_code', 'unknown')}: {str(e)}"
                    stats['errors'].append(error_msg)
                    logger.error(error_msg, exc_info=True)
            
            # Batch upsert
            if items:
                results = upsert_statistics_items(items, client=self.client)
                stats['inserted'] = len(results)
                logger.info(f"Upserted {len(results)} items for {source_stat_code}")
                
                # Create series for each item if enabled
                if self.create_series:
                    metadata = get_statistics_metadata(
                        source_id=self.source_id,
                        source_stat_code=source_stat_code,
                        client=self.client
                    )
                    stat_name = metadata.get('source_stat_name', source_stat_code) if metadata else source_stat_code
                    for item_result in results:
                        try:
                            create_series_from_item(
                                source_code=self.source_code,
                                stat_code=source_stat_code,
                                stat_name=stat_name,
                                item=item_result,
                                statistics_metadata_id=statistics_metadata_id,
                                frequency_mapper=map_frequency_to_code,
                                client=self.client
                            )
                        except Exception as e:
                            logger.warning(f"Failed to create series for item {item_result.get('item_code')}: {e}")
            
        except Exception as e:
            error_msg = f"Failed to collect items for {source_stat_code}: {str(e)}"
            stats['errors'].append(error_msg)
            logger.error(error_msg, exc_info=True)
        
        return stats
    
    def update_all_metadata(
        self,
        update_dfm_selection: bool = False,
        dfm_selection_csv: Optional[str] = None,
        collect_items: bool = True,
        create_series: bool = True
    ) -> Dict[str, Any]:
        """
        Full metadata refresh workflow.
        
        Parameters
        ----------
        update_dfm_selection : bool
            Whether to update DFM selection from CSV
        dfm_selection_csv : str, optional
            Path to CSV file with DFM-selected statistics
        collect_items : bool
            Whether to collect items for each statistic
        create_series : bool
            Whether to create series during metadata collection
            
        Returns
        -------
        Dict[str, Any]
            Summary of all operations
        """
        # Series creation is handled via create_series flag
        
        summary = {
            'statistics_list': {},
            'items_collection': {
                'total_statistics': 0,
                'processed': 0,
                'failed': 0,
                'total_items': 0
            }
        }
        
        summary['statistics_list'] = self.collect_statistics_list(
            update_dfm_selection=update_dfm_selection,
            dfm_selection_csv=dfm_selection_csv
        )
        
        if collect_items:
            dfm_stats = list_dfm_selected_statistics(
                source_id=self.source_id,
                client=self.client
            )
            
            summary['items_collection']['total_statistics'] = len(dfm_stats)
            
            for stat in dfm_stats:
                stat_code = stat['source_stat_code']
                stat_id = stat['id']
                
                try:
                    item_stats = self.collect_statistic_items(
                        source_stat_code=stat_code,
                        statistics_metadata_id=stat_id
                    )
                    summary['items_collection']['processed'] += 1
                    summary['items_collection']['total_items'] += item_stats.get('total', 0)
                except Exception as e:
                    summary['items_collection']['failed'] += 1
                    logger.error(f"Failed to collect items for {stat_code}: {e}")
        
        return summary
