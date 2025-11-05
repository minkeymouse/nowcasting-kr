"""KOSIS ingestion implementation.

This combines API client, parsing, and metadata collection for KOSIS data source.
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
from services.api.kosis_client import KOSISAPIClient
from services.ingestion.base import BaseIngestion

logger = logging.getLogger(__name__)


def _parse_kosis_date(date_str: str, frequency: str) -> pd.Timestamp:
    """Parse KOSIS date string to pandas Timestamp.
    
    KOSIS date format: YYYYMMDD (8 digits) or YYYY (4 digits)
    """
    date_str = str(date_str).strip()
    
    if frequency == 'A':  # Annual: 2020
        if len(date_str) == 4:
            return pd.Timestamp(year=int(date_str), month=12, day=31)
        elif len(date_str) == 8:
            return pd.Timestamp(year=int(date_str[:4]), month=12, day=31)
    elif frequency == 'Q':  # Quarterly: 2020Q1 format stored as 20200301
        if len(date_str) == 8:
            return pd.Timestamp(year=int(date_str[:4]), month=int(date_str[4:6]), day=1)
    elif frequency == 'M':  # Monthly: 202001
        if len(date_str) == 6:
            return pd.Timestamp(year=int(date_str[:4]), month=int(date_str[4:6]), day=1)
        elif len(date_str) == 8:
            return pd.Timestamp(year=int(date_str[:4]), month=int(date_str[4:6]), day=1)
    elif frequency in ['SM', 'D']:  # Semi-monthly or Daily: 20200101
        if len(date_str) == 8:
            return pd.Timestamp(year=int(date_str[:4]), month=int(date_str[4:6]), day=int(date_str[6:8]))
    elif frequency == 'S':  # Semi-annual: 2020S1 format stored as 20200630
        if len(date_str) == 8:
            return pd.Timestamp(year=int(date_str[:4]), month=int(date_str[4:6]), day=int(date_str[6:8]))
    
    # Fallback: try to parse as-is
    try:
        return pd.Timestamp(date_str)
    except:
        return pd.NaT


class KOSISIngestion(BaseIngestion):
    """KOSIS data source ingestion (API + parsing + metadata collection)."""
    
    def __init__(self, create_series: bool = False):
        """Initialize KOSIS ingestion."""
        settings = AppSettings.load()
        kosis_config = settings.kosis_api_config
        if not kosis_config:
            raise ValueError("KOSIS API key not configured. Set KOSIS_API_KEY environment variable.")
        
        api_client = KOSISAPIClient(kosis_config)
        super().__init__(api_client, create_series)
    
    # Parsing methods
    def transform_statistics_list(self, response: Dict[str, Any]) -> pd.DataFrame:
        """Transform KOSIS statisticsList response to DataFrame."""
        data_list = response.get('data', response) if isinstance(response, dict) else response
        
        if not isinstance(data_list, list):
            return pd.DataFrame(columns=[
                'org_id', 'tbl_id', 'tbl_nm', 'tbl_nm_eng',
                'stat_id', 'list_id', 'list_nm', 'vw_cd', 'vw_nm',
                'send_de', 'rec_tbl_se'
            ])
        
        records = []
        for item in data_list:
            if not isinstance(item, dict):
                continue
            
            record = {
                'org_id': item.get('ORG_ID'),
                'tbl_id': item.get('TBL_ID'),
                'tbl_nm': item.get('TBL_NM'),
                'tbl_nm_eng': None,
                'stat_id': item.get('STAT_ID'),
                'list_id': item.get('LIST_ID'),
                'list_nm': item.get('LIST_NM'),
                'vw_cd': item.get('VW_CD'),
                'vw_nm': item.get('VW_NM'),
                'send_de': item.get('SEND_DE'),
                'rec_tbl_se': item.get('REC_TBL_SE'),
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    def transform_statistic_items(
        self,
        response: Dict[str, Any],
        statistics_metadata_id: int
    ) -> pd.DataFrame:
        """Transform KOSIS getMeta response to DataFrame."""
        data_list = response.get('data', response) if isinstance(response, dict) else response
        
        if not isinstance(data_list, list) or not data_list:
            return pd.DataFrame(columns=[
                'statistics_metadata_id', 'item_code', 'item_name', 'item_name_eng',
                'parent_item_code', 'parent_item_name', 'grp_code', 'grp_name',
                'cycle', 'start_time', 'end_time', 'data_count', 'unit_name', 'weight'
            ])
        
        items = []
        for item in data_list:
            if isinstance(item, dict):
                items.append({
                    'statistics_metadata_id': statistics_metadata_id,
                    'item_code': item.get('TBL_ID', ''),
                    'item_name': item.get('TBL_NM', ''),
                    'item_name_eng': item.get('TBL_NM_ENG'),
                    'parent_item_code': None,
                    'parent_item_name': None,
                    'grp_code': None,
                    'grp_name': None,
                    'cycle': None,
                    'start_time': None,
                    'end_time': None,
                    'data_count': None,
                    'unit_name': None,
                    'weight': None,
                })
        
        return pd.DataFrame(items)
    
    def transform_statistic_data(
        self,
        response: Dict[str, Any],
        stat_code: str,
        frequency: str,
        item_code: Optional[str] = None
    ) -> pd.DataFrame:
        """Transform KOSIS statisticsParameterData response to DataFrame."""
        data_list = response.get('data', response) if isinstance(response, dict) else response
        
        if not isinstance(data_list, list):
            return pd.DataFrame(columns=[
                'date', 'value', 'series_id', 'stat_code', 'item_code',
                'item_code1', 'item_code2', 'item_code3', 'item_code4',
                'item_name1', 'item_name2', 'item_name3', 'item_name4',
                'weight', 'unit', 'item_name'
            ])
        
        data = []
        for row in data_list:
            if not isinstance(row, dict):
                continue
            
            # Parse date
            prd_de = str(row.get('PRD_DE', '')).strip()
            if not prd_de:
                continue
            
            try:
                date = _parse_kosis_date(prd_de, frequency)
                if pd.isna(date):
                    continue
            except Exception:
                continue
            
            # Parse value
            dt_str = row.get('DT', '')
            if dt_str is None or dt_str == '':
                continue
            
            try:
                value = float(dt_str)
            except (ValueError, TypeError):
                continue
            
            # Extract classification codes
            item_code1 = row.get('C1')
            item_code2 = row.get('C2')
            item_code3 = row.get('C3')
            item_code4 = row.get('C4')
            
            # Extract classification names
            item_name1 = row.get('C1_NM')
            item_name2 = row.get('C2_NM')
            item_name3 = row.get('C3_NM')
            item_name4 = row.get('C4_NM')
            
            # Item ID and name
            itm_id = row.get('ITM_ID', '')
            itm_nm = row.get('ITM_NM', '')
            
            # Unit
            unit_nm = row.get('UNIT_NM', '')
            
            # Generate series_id
            org_id = row.get('ORG_ID', '')
            tbl_id = row.get('TBL_ID', stat_code)
            source_code = self.source_code
            
            parts = [source_code, org_id, tbl_id]
            if item_code1:
                parts.append(str(item_code1))
            if itm_id:
                parts.append(str(itm_id))
            
            series_id = '_'.join(parts)
            item_code_used = itm_id or item_code1 or ''
            
            data.append({
                'date': date,
                'value': value,
                'series_id': series_id,
                'stat_code': f"{org_id}_{tbl_id}" if org_id and tbl_id else stat_code,
                'item_code': item_code_used,
                'item_code1': item_code1,
                'item_code2': item_code2,
                'item_code3': item_code3,
                'item_code4': item_code4,
                'item_name1': item_name1 or itm_nm,
                'item_name2': item_name2,
                'item_name3': item_name3,
                'item_name4': item_name4,
                'weight': None,
                'unit': unit_nm,
                'item_name': itm_nm or item_name1
            })
        
        df = pd.DataFrame(data)
        if df.empty:
            return df
        
        return df.sort_values('date').reset_index(drop=True)
    
    # Metadata collection methods
    # Source-specific metadata collection helpers
    def _fetch_statistics_list_pages(self) -> pd.DataFrame:
        """Fetch KOSIS statistics list (single call, no pagination)."""
        response = self.api_client.fetch_statistics_list()
        return self.transform_statistics_list(response)
    
    def _extract_stat_code(self, row: Dict[str, Any]) -> Optional[str]:
        """Extract KOSIS stat code from row (orgId_tblId format)."""
        org_id = row.get('org_id')
        tbl_id = row.get('tbl_id')
        if not org_id or not tbl_id:
            return None
        return f"{org_id}_{tbl_id}"
    
    def _build_metadata_from_row(
        self,
        row: Dict[str, Any],
        dfm_selected_codes: set,
        dfm_priorities: Dict[str, int],
        existing: Optional[Dict[str, Any]]
    ) -> StatisticsMetadataModel:
        """Build KOSIS StatisticsMetadataModel from row."""
        stat_code = self._extract_stat_code(row)
        is_dfm_selected = stat_code in dfm_selected_codes
        dfm_priority = dfm_priorities.get(stat_code)
        
        return StatisticsMetadataModel(
            source_id=self.source_id,
            source_stat_code=stat_code,
            source_stat_name=row.get('tbl_nm'),
            source_stat_name_eng=row.get('tbl_nm_eng'),
            cycle=None,  # KOSIS: cycle not available in list
            frequency_code=None,
            org_name=None,
            is_searchable=True,
            parent_stat_code=row.get('stat_id'),
            parent_item_code=None,
            source_metadata={
                'org_id': row.get('org_id'),
                'tbl_id': row.get('tbl_id'),
                'stat_id': row.get('stat_id'),
                'list_id': row.get('list_id'),
                'list_nm': row.get('list_nm'),
                'vw_cd': row.get('vw_cd'),
                'vw_nm': row.get('vw_nm'),
            },
            is_dfm_selected=is_dfm_selected,
            dfm_priority=dfm_priority if is_dfm_selected else None,
            dfm_selected_at=datetime.now() if is_dfm_selected and not existing else None,
        )
    
