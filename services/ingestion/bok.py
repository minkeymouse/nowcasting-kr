"""BOK ingestion implementation.

This combines API client, parsing, and metadata collection for BOK data source.
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd

from database import (
    get_statistics_metadata,
    upsert_statistics_metadata,
    upsert_statistics_items,
)
from database.models import StatisticsMetadataModel, StatisticsItemModel
from database.settings import AppSettings
from database.helpers import map_frequency_to_code
from services.api.bok_client import BOKAPIClient
from services.ingestion.base import BaseIngestion

logger = logging.getLogger(__name__)


def _parse_bok_date(date_str: str, frequency: str) -> pd.Timestamp:
    """Parse BOK date string to pandas Timestamp."""
    if frequency == 'A':  # Annual: 2020
        return pd.Timestamp(year=int(date_str), month=12, day=31)
    elif frequency == 'Q':  # Quarterly: 2020Q1
        year, quarter = date_str.split('Q')
        month = int(quarter) * 3
        return pd.Timestamp(year=int(year), month=month, day=1)
    elif frequency == 'M':  # Monthly: 202001
        return pd.Timestamp(year=int(date_str[:4]), month=int(date_str[4:6]), day=1)
    elif frequency in ['SM', 'D']:  # Semi-monthly or Daily: 20200101
        return pd.Timestamp(year=int(date_str[:4]), month=int(date_str[4:6]), day=int(date_str[6:8]))
    elif frequency == 'S':  # Semi-annual: 2020S1, 2020S2
        year, half = date_str.split('S')
        month = 6 if half == '1' else 12
        return pd.Timestamp(year=int(year), month=month, day=30 if half == '1' else 31)
    else:
        return pd.Timestamp(date_str)


class BOKIngestion(BaseIngestion):
    """BOK data source ingestion (API + parsing + metadata collection)."""
    
    def __init__(self, create_series: bool = False):
        """Initialize BOK ingestion."""
        settings = AppSettings.load()
        api_client = BOKAPIClient(settings.bok_api_config)
        super().__init__(api_client, create_series)
    
    # Parsing methods
    def transform_statistics_list(self, response: Dict[str, Any]) -> pd.DataFrame:
        """Transform BOK StatisticTableList response to DataFrame."""
        # This is not used directly - BOK uses paginated API calls
        # But kept for interface compatibility
        if 'StatisticTableList' not in response:
            return pd.DataFrame()
        table_list = response['StatisticTableList']
        rows = table_list.get('row', [])
        return pd.DataFrame(rows)
    
    def transform_statistic_items(
        self,
        response: Dict[str, Any],
        statistics_metadata_id: int
    ) -> pd.DataFrame:
        """Transform BOK StatisticItemList response to DataFrame."""
        if 'StatisticItemList' not in response:
            raise ValueError("Invalid BOK API response format: StatisticItemList not found")
        
        item_list_data = response['StatisticItemList']
        rows = item_list_data.get('row', [])
        
        if not rows:
            return pd.DataFrame(columns=[
                'statistics_metadata_id', 'item_code', 'item_name', 'item_name_eng',
                'parent_item_code', 'parent_item_name', 'grp_code', 'grp_name',
                'cycle', 'start_time', 'end_time', 'data_count', 'unit_name', 'weight'
            ])
        
        data = []
        for row in rows:
            item_data = {
                'statistics_metadata_id': statistics_metadata_id,
                'item_code': row.get('ITEM_CODE', ''),
                'item_name': row.get('ITEM_NAME', ''),
                'item_name_eng': None,
                'parent_item_code': row.get('P_ITEM_CODE'),
                'parent_item_name': row.get('P_ITEM_NAME'),
                'grp_code': row.get('GRP_CODE'),
                'grp_name': row.get('GRP_NAME'),
                'cycle': row.get('CYCLE', ''),
                'start_time': row.get('START_TIME'),
                'end_time': row.get('END_TIME'),
                'data_count': row.get('DATA_CNT'),
                'unit_name': row.get('UNIT_NAME'),
                'weight': row.get('WEIGHT'),
            }
            data.append(item_data)
        
        return pd.DataFrame(data)
    
    def transform_statistic_data(
        self,
        response: Dict[str, Any],
        stat_code: str,
        frequency: str,
        item_code: Optional[str] = None
    ) -> pd.DataFrame:
        """Transform BOK StatisticSearch response to DataFrame."""
        if 'StatisticSearch' not in response:
            raise ValueError("Invalid BOK API response format")
        
        search_data = response['StatisticSearch']
        rows = search_data.get('row', [])
        
        if not rows:
            return pd.DataFrame(columns=[
                'date', 'value', 'series_id', 'stat_code', 'item_code',
                'item_code1', 'item_code2', 'item_code3', 'item_code4',
                'item_name1', 'item_name2', 'item_name3', 'item_name4',
                'weight', 'unit', 'item_name'
            ])
        
        data = []
        for row in rows:
            date_str = row.get('TIME')
            if not date_str:
                continue
            
            try:
                date = _parse_bok_date(str(date_str), frequency)
            except Exception:
                continue
            
            value_str = row.get('DATA_VALUE')
            if value_str is None or value_str == '':
                continue
            
            try:
                value = float(value_str)
            except (ValueError, TypeError):
                continue
            
            item_code1 = row.get('ITEM_CODE1')
            item_code2 = row.get('ITEM_CODE2')
            item_code3 = row.get('ITEM_CODE3')
            item_code4 = row.get('ITEM_CODE4')
            item_name1 = row.get('ITEM_NAME1')
            item_name2 = row.get('ITEM_NAME2')
            item_name3 = row.get('ITEM_NAME3')
            item_name4 = row.get('ITEM_NAME4')
            
            weight_str = row.get('WGT')
            weight = None
            if weight_str and weight_str != '':
                try:
                    weight = float(weight_str)
                except (ValueError, TypeError):
                    pass
            
            item_code_used = item_code or item_code1 or ''
            
            # Generate series_id
            source_code = self.source_code
            item_code_for_id = item_code1 or item_code_used or ''
            if item_code_for_id:
                series_id = f"{source_code}_{stat_code}_{item_code_for_id}"
            else:
                series_id = f"{source_code}_{stat_code}"
            
            data.append({
                'date': date,
                'value': value,
                'series_id': series_id,
                'stat_code': stat_code,
                'item_code': item_code_used,
                'item_code1': item_code1,
                'item_code2': item_code2,
                'item_code3': item_code3,
                'item_code4': item_code4,
                'item_name1': item_name1,
                'item_name2': item_name2,
                'item_name3': item_name3,
                'item_name4': item_name4,
                'weight': weight,
                'unit': row.get('UNIT_NAME', ''),
                'item_name': item_name1
            })
        
        df = pd.DataFrame(data)
        if df.empty:
            return df
        
        return df.sort_values('date').reset_index(drop=True)
    
    # Source-specific metadata collection helpers
    def _fetch_statistics_list_pages(self) -> pd.DataFrame:
        """Fetch all BOK statistics using paginated API."""
        all_rows = []
        page_size = 1000
        start_count = 1
        end_count = page_size
        
        while True:
            logger.debug(f"Fetching BOK statistics: {start_count} to {end_count}")
            
            response = self.api_client.fetch_statistics_list(
                start_count=start_count,
                end_count=end_count
            )
            
            if 'StatisticTableList' not in response:
                logger.error("Invalid response format")
                break
            
            table_list = response['StatisticTableList']
            rows = table_list.get('row', [])
            
            if not rows:
                break
            
            all_rows.extend(rows)
            
            # Check if more pages
            total_count = table_list.get('list_total_count', 0)
            if end_count >= total_count:
                break
            
            start_count = end_count + 1
            end_count = min(start_count + page_size - 1, total_count)
            
            # Rate limiting
            time.sleep(0.5)
        
        return pd.DataFrame(all_rows)
    
    def _extract_stat_code(self, row: Dict[str, Any]) -> Optional[str]:
        """Extract BOK stat code from row."""
        return row.get('STAT_CODE')
    
    def _build_metadata_from_row(
        self,
        row: Dict[str, Any],
        dfm_selected_codes: set,
        dfm_priorities: Dict[str, int],
        existing: Optional[Dict[str, Any]]
    ) -> StatisticsMetadataModel:
        """Build BOK StatisticsMetadataModel from row."""
        stat_code = row.get('STAT_CODE')
        is_dfm_selected = stat_code in dfm_selected_codes
        dfm_priority = dfm_priorities.get(stat_code)
        
        return StatisticsMetadataModel(
            source_id=self.source_id,
            source_stat_code=stat_code,
            source_stat_name=row.get('STAT_NAME'),
            source_stat_name_eng=row.get('STAT_NAME_ENG'),
            cycle=row.get('CYCLE'),
            frequency_code=map_frequency_to_code(row.get('CYCLE')) if row.get('CYCLE') else None,
            org_name=row.get('ORG_NAME'),
            is_searchable=(row.get('SRCH_YN') == 'Y'),
            parent_stat_code=row.get('P_STAT_CODE'),
            parent_item_code=row.get('P_ITEM_CODE'),
            source_metadata={
                'p_cycle': row.get('P_CYCLE'),
            },
            is_dfm_selected=is_dfm_selected,
            dfm_priority=dfm_priority if is_dfm_selected else None,
            dfm_selected_at=datetime.now() if is_dfm_selected and not existing else None,
        )
    


# Backward compatibility
def transform_bok_response(
    response: Dict[str, Any],
    stat_code: str,
    frequency: str,
    item_code: Optional[str] = None
) -> pd.DataFrame:
    """Backward compatibility function for BOK response parsing."""
    ingestion = BOKIngestion()
    return ingestion.transform_statistic_data(response, stat_code, frequency, item_code)


def transform_bok_item_list(
    response: Dict[str, Any],
    statistics_metadata_id: int
) -> pd.DataFrame:
    """Backward compatibility function for BOK item list parsing."""
    ingestion = BOKIngestion()
    return ingestion.transform_statistic_items(response, statistics_metadata_id)
