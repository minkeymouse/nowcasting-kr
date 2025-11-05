"""KOSIS (Statistics Korea) API client.

This is a concrete implementation of BaseAPIClient for KOSIS API.
"""

import time
from typing import Optional, Dict, Any, List
import httpx
from urllib.parse import urlencode
from database.settings import KOSISAPIConfig
from .base import BaseAPIClient, APIError


class KOSISAPIError(APIError):
    """KOSIS API specific error."""
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message, error_code, source="KOSIS")


class KOSISAPIClient(BaseAPIClient):
    """Client for KOSIS (Statistics Korea) API."""
    
    def __init__(self, config: KOSISAPIConfig):
        """
        Initialize KOSIS API client.
        
        Parameters
        ----------
        config : KOSISAPIConfig
            KOSIS API configuration from settings
        """
        self.config = config
        self.base_url = config.base_url.rstrip('/')
        self.api_key = config.api_key
        self.default_view_code = config.default_view_code
        self.timeout = config.timeout
        self.max_retries = config.max_retries
        self.retry_backoff = config.retry_backoff
        
        # Rate limiting
        self._last_request_time: Optional[float] = None
        self._rate_limit_delay = 60.0 / config.rate_limit_per_minute if config.rate_limit_per_minute else 0.0
    
    @property
    def source_code(self) -> str:
        """Return the data source code."""
        return 'KOSIS'
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        if self._rate_limit_delay > 0 and self._last_request_time:
            elapsed = time.time() - self._last_request_time
            if elapsed < self._rate_limit_delay:
                time.sleep(self._rate_limit_delay - elapsed)
        self._last_request_time = time.time()
    
    def _request_with_retry(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make API request with retry logic."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                self._rate_limit()
                
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.get(url, params=params)
                    response.raise_for_status()
                    
                    # KOSIS sometimes returns invalid JSON (unquoted keys)
                    # Try to parse as JSON, if it fails, fix unquoted keys
                    text = response.text.strip()
                    try:
                        data = response.json()
                    except ValueError as json_err:
                        # Try to fix unquoted keys (common KOSIS issue)
                        import re
                        import json as json_lib
                        
                        # Check if it's the KOSIS error format: {err:"20",errMsg:"..."}
                        if text.startswith('{') and 'err' in text.lower():
                            # Fix unquoted keys for error response
                            fixed_text = re.sub(r'(\w+):', r'"\1":', text)
                            try:
                                error_data = json_lib.loads(fixed_text)
                                err_code = error_data.get('err', '')
                                err_msg = error_data.get('errMsg', 'Unknown error')
                                raise KOSISAPIError(f"KOSIS API error {err_code}: {err_msg}")
                            except (json_lib.JSONDecodeError, KeyError):
                                raise KOSISAPIError(f"KOSIS API returned invalid response: {text[:200]}")
                        elif text.startswith('['):
                            # Fix unquoted keys for array response
                            # Pattern: {KEY:"value"} -> {"KEY":"value"}
                            fixed_text = re.sub(r'\{(\w+):', r'{"\1":', text)
                            fixed_text = re.sub(r',(\w+):', r',"\1":', fixed_text)
                            try:
                                data = json_lib.loads(fixed_text)
                            except json_lib.JSONDecodeError:
                                raise KOSISAPIError(f"KOSIS API returned invalid JSON: {text[:200]}")
                        else:
                            raise KOSISAPIError(f"Invalid JSON response: {str(json_err)}")
                    
                    # KOSIS returns list directly, not wrapped in dict
                    if isinstance(data, list):
                        return {'data': data}
                    
                    return data
                    
            except httpx.HTTPError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_backoff ** attempt
                    time.sleep(wait_time)
                    continue
                raise KOSISAPIError(f"HTTP error: {str(e)}")
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_backoff ** attempt
                    time.sleep(wait_time)
                    continue
                raise
        
        raise KOSISAPIError(f"Request failed after {self.max_retries} attempts: {last_error}")
    
    def fetch_statistics_list(
        self,
        start_count: int = 1,
        end_count: int = 1000,
        language: str = 'kr'
    ) -> Dict[str, Any]:
        """
        Fetch list of available statistics.
        
        Note: KOSIS API uses vwCd and parentId instead of pagination.
        This method fetches from default view (MT_ZTITLE) with root parentId.
        
        Parameters
        ----------
        start_count : int
            Not used for KOSIS (kept for interface compatibility)
        end_count : int
            Not used for KOSIS (kept for interface compatibility)
        language : str
            Not used for KOSIS (kept for interface compatibility)
            
        Returns
        -------
        Dict[str, Any]
            API response with statistics list
        """
        url = f"{self.base_url}statisticsList.do"
        params = {
            'method': 'getList',
            'apiKey': self.api_key,
            'vwCd': self.default_view_code,
            'parentId': 'MT_ZTITLE',  # Root level
            'format': 'json'
        }
        
        return self._request_with_retry(url, params)
    
    def fetch_statistic_items(
        self,
        stat_code: str,
        start_count: int = 1,
        end_count: int = 1000
    ) -> Dict[str, Any]:
        """
        Fetch items/parameters for a specific statistic.
        
        Note: KOSIS uses orgId + tblId format for stat_code.
        This method fetches metadata using getMeta endpoint.
        
        Parameters
        ----------
        stat_code : str
            Statistic identifier in format "orgId_tblId" or just "tblId" (orgId defaults)
        start_count : int
            Not used for KOSIS (kept for interface compatibility)
        end_count : int
            Not used for KOSIS (kept for interface compatibility)
            
        Returns
        -------
        Dict[str, Any]
            API response with items/parameters
        """
        # Parse orgId and tblId from stat_code
        if '_' in stat_code:
            org_id, tbl_id = stat_code.split('_', 1)
        else:
            # Default orgId if not provided
            org_id = '101'  # 통계청 기본 코드
            tbl_id = stat_code
        
        url = f"{self.base_url}statisticsData.do"
        params = {
            'method': 'getMeta',
            'type': 'TBL',
            'apiKey': self.api_key,
            'orgId': org_id,
            'tblId': tbl_id,
            'format': 'json'
        }
        
        return self._request_with_retry(url, params)
    
    def fetch_statistic_data(
        self,
        stat_code: str,
        frequency: str,
        start_date: str,
        end_date: str,
        item_code1: Optional[str] = None,
        item_code2: Optional[str] = None,
        item_code3: Optional[str] = None,
        item_code4: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch time-series data for a statistic.
        
        Parameters
        ----------
        stat_code : str
            Statistic identifier in format "orgId_tblId"
        frequency : str
            Frequency code (A, S, Q, M, SM, D)
        start_date : str
            Start date in format matching frequency
        end_date : str
            End date in format matching frequency
        item_code1-4 : str, optional
            For KOSIS: objL1~objL8 (classification codes) and itmId (item ID)
            item_code1 should be objL1, item_code2 should be itmId, etc.
            
        Returns
        -------
        Dict[str, Any]
            API response with time-series data
        """
        # Parse orgId and tblId from stat_code
        if '_' in stat_code:
            org_id, tbl_id = stat_code.split('_', 1)
        else:
            org_id = '101'  # Default
            tbl_id = stat_code
        
        # According to note_kosis.txt, use statisticsParameterData.do
        # This endpoint REQUIRES both objL1 (필수) and itmId (필수)
        # Note: base_url already includes '/openapi', so we need '/Param/statisticsParameterData.do'
        url = f"{self.base_url}/Param/statisticsParameterData.do"
        params = {
            'method': 'getList',
            'apiKey': self.api_key,
            'orgId': org_id,
            'tblId': tbl_id,
            'format': 'json',
            'prdSe': frequency
        }
        
        # objL1 is REQUIRED (필수) - use item_code1 or default to 'ALL' (works for many tables)
        # Note: 'ALL' works for DT_1J22003 (CPI), '00' or specific codes for others
        obj_l1 = item_code1 if item_code1 else 'ALL'
        params['objL1'] = obj_l1
        
        # itmId is REQUIRED (필수) - use item_code2 or default to 'T' (total/common item)
        # Note: 'T' works for DT_1J22003 (CPI total index), specific IDs for others
        if item_code2:
            params['itmId'] = item_code2
        else:
            # Default to 'T' - common for total/aggregate indices
            params['itmId'] = 'T'
        
        # Add optional classification codes (objL2~objL8)
        # Only add if provided and not '?' (KOSIS doesn't accept '?' as value)
        if item_code3 and item_code3 != '?':
            params['objL2'] = item_code3
        if item_code4 and item_code4 != '?':
            params['objL3'] = item_code4
        
        
        # Date range
        if start_date:
            params['startPrdDe'] = start_date
        if end_date:
            params['endPrdDe'] = end_date
        
        return self._request_with_retry(url, params)
    
    def get_statistics_list_by_view(
        self,
        view_code: str = 'MT_ZTITLE',
        parent_id: str = 'MT_ZTITLE'
    ) -> List[Dict[str, Any]]:
        """
        Get statistics list for a specific view and parent.
        
        Parameters
        ----------
        view_code : str
            View code (e.g., 'MT_ZTITLE', 'MT_OTITLE')
        parent_id : str
            Parent ID to start from
            
        Returns
        -------
        List[Dict[str, Any]]
            List of statistics
        """
        url = f"{self.base_url}statisticsList.do"
        params = {
            'method': 'getList',
            'apiKey': self.api_key,
            'vwCd': view_code,
            'parentId': parent_id,
            'format': 'json'
        }
        
        response = self._request_with_retry(url, params)
        return response.get('data', [])
    
    def get_statistics_data_by_user_stats_id(
        self,
        user_stats_id: str,
        frequency: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        new_est_prd_cnt: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get statistics data using userStatsId (user-registered table).
        
        Parameters
        ----------
        user_stats_id : str
            User-registered statistics ID
        frequency : str
            Frequency code
        start_date : str, optional
            Start date
        end_date : str, optional
            End date
        new_est_prd_cnt : int, optional
            Number of recent periods to fetch
            
        Returns
        -------
        List[Dict[str, Any]]
            List of data records
        """
        url = f"{self.base_url}statisticsData.do"
        params = {
            'method': 'getList',
            'apiKey': self.api_key,
            'userStatsId': user_stats_id,
            'prdSe': frequency,
            'format': 'json'
        }
        
        if start_date and end_date:
            params['startPrdDe'] = start_date
            params['endPrdDe'] = end_date
        elif new_est_prd_cnt:
            params['newEstPrdCnt'] = str(new_est_prd_cnt)
        
        response = self._request_with_retry(url, params)
        return response.get('data', [])

