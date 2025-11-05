"""BOK (Bank of Korea) API client.

This is a concrete implementation of BaseAPIClient for BOK Economic Statistics System.
Additional data sources should implement BaseAPIClient similarly.
"""

import time
from typing import Optional, Dict, Any
from datetime import datetime
import httpx
from database.settings import BOKAPIConfig, BOK_ERROR_CODES
from .base import BaseAPIClient, APIError


class BOKAPIError(APIError):
    """BOK API specific error."""
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message, error_code, source="BOK")


class BOKAPIClient(BaseAPIClient):
    """Client for BOK Economic Statistics System API."""
    
    def __init__(self, config: BOKAPIConfig):
        """
        Initialize BOK API client.
        
        Parameters
        ----------
        config : BOKAPIConfig
            BOK API configuration from settings
        """
        self.config = config
        self.base_url = config.base_url.rstrip('/')
        self.auth_key = config.auth_key
        self.timeout = config.timeout
        self.max_retries = config.max_retries
        self.retry_backoff = config.retry_backoff
        
        # Rate limiting
        self._last_request_time: Optional[float] = None
        self._rate_limit_delay = 60.0 / config.rate_limit_per_minute if config.rate_limit_per_minute else 0.0
    
    @property
    def source_code(self) -> str:
        """Return the data source code."""
        return 'BOK'
    
    def _build_url(self, service: str, **params) -> str:
        """Build API URL with parameters."""
        # URL format varies by service:
        # StatisticTableList: /api/StatisticTableList/{AuthKey}/{Format}/{Lang}/{Start}/{End}/{StatCode?}
        # StatisticItemList: /api/StatisticItemList/{AuthKey}/{Format}/{Lang}/{Start}/{End}/{StatCode}
        # StatisticSearch: /api/StatisticSearch/{AuthKey}/{Format}/{Lang}/{Start}/{End}/{StatCode}/{Frequency}/{StartDate}/{EndDate}/{ItemCode1}/{ItemCode2}/...
        
        parts = [
            self.base_url.rstrip('/'),
            service,
            self.auth_key,
            self.config.request_type,
            self.config.language,
            str(params.get('start_count', self.config.default_start_count)),
            str(params.get('end_count', self.config.default_end_count)),
        ]
        
        # Add optional parameters based on service
        if service == 'StatisticTableList':
            # StatCode is optional for StatisticTableList
            if 'stat_code' in params and params['stat_code']:
                parts.append(params['stat_code'])
        elif service == 'StatisticItemList':
            # StatCode is required for StatisticItemList
            if 'stat_code' in params:
                parts.append(params['stat_code'])
        elif service == 'StatisticSearch':
            # StatisticSearch has full parameter chain
            if 'stat_code' in params:
                parts.append(params['stat_code'])
                if 'frequency' in params:
                    parts.append(params['frequency'])
                    if 'start_date' in params:
                        parts.append(params['start_date'])
                        if 'end_date' in params:
                            parts.append(params['end_date'])
                            # Item codes (optional, use ? for empty)
                            for i in range(1, 5):
                                item_code = params.get(f'item_code{i}', '?')
                                parts.append(str(item_code))
        
        return '/'.join(parts)
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        if self._rate_limit_delay > 0 and self._last_request_time:
            elapsed = time.time() - self._last_request_time
            if elapsed < self._rate_limit_delay:
                time.sleep(self._rate_limit_delay - elapsed)
        self._last_request_time = time.time()
    
    def _request_with_retry(self, url: str) -> Dict[str, Any]:
        """Make API request with retry logic."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                self._rate_limit()
                
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.get(url)
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    # Check for BOK API errors
                    error_code = self._extract_error_code(data)
                    if error_code:
                        error_msg = BOK_ERROR_CODES.get(error_code, "Unknown error")
                        
                        # Retryable errors
                        if error_code in ['에러-400', '에러-600', '에러-602']:
                            if attempt < self.max_retries - 1:
                                wait_time = self.retry_backoff ** attempt
                                time.sleep(wait_time)
                                continue
                        
                        # Non-retryable or informational
                        raise BOKAPIError(f"{error_code}: {error_msg}", error_code)
                    
                    return data
                    
            except httpx.HTTPError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_backoff ** attempt
                    time.sleep(wait_time)
                    continue
                raise BOKAPIError(f"HTTP error: {str(e)}")
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_backoff ** attempt
                    time.sleep(wait_time)
                    continue
                raise
        
        raise BOKAPIError(f"Request failed after {self.max_retries} attempts: {last_error}")
    
    def _extract_error_code(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract error code from BOK API response."""
        # BOK API returns errors in various formats
        # Check common error patterns
        for key in ['RESULT', 'CODE', 'ERROR_CODE']:
            if key in data:
                code = data[key]
                if isinstance(code, str) and ('정보-' in code or '에러-' in code):
                    return code
        return None
    
    def fetch_statistics_list(
        self,
        start_count: int = 1,
        end_count: int = 1000,
        language: str = 'kr'
    ) -> Dict[str, Any]:
        """Fetch list of available statistics (implements BaseAPIClient)."""
        return self.get_statistic_table_list(start_count=start_count, end_count=end_count)
    
    def fetch_statistic_items(
        self,
        stat_code: str,
        start_count: int = 1,
        end_count: int = 1000
    ) -> Dict[str, Any]:
        """Fetch items for a specific statistic (implements BaseAPIClient)."""
        return self.get_statistic_item_list(stat_code, start_count, end_count)
    
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
        """Fetch time-series data (implements BaseAPIClient)."""
        return self.search_statistics(
            stat_code, frequency, start_date, end_date,
            item_code1, item_code2, item_code3, item_code4
        )
    
    def get_statistic_table_list(
        self,
        stat_code: Optional[str] = None,
        start_count: int = 1,
        end_count: int = 50
    ) -> Dict[str, Any]:
        """
        Get list of statistical tables.
        
        Parameters
        ----------
        stat_code : str, optional
            Specific statistic code to filter
        start_count : int
            Start count for pagination
        end_count : int
            End count for pagination
            
        Returns
        -------
        Dict[str, Any]
            API response with table list
        """
        url = self._build_url(
            'StatisticTableList',
            start_count=start_count,
            end_count=end_count,
            stat_code=stat_code
        ) if stat_code else self._build_url('StatisticTableList', start_count=start_count, end_count=end_count)
        
        return self._request_with_retry(url)
    
    def get_statistic_item_list(
        self,
        stat_code: str,
        start_count: int = 1,
        end_count: int = 100
    ) -> Dict[str, Any]:
        """
        Get list of items for a statistical table.
        
        Parameters
        ----------
        stat_code : str
            Statistic code (e.g., '200Y101')
        start_count : int
            Start count for pagination
        end_count : int
            End count for pagination
            
        Returns
        -------
        Dict[str, Any]
            API response with item list
        """
        url = self._build_url(
            'StatisticItemList',
            start_count=start_count,
            end_count=end_count,
            stat_code=stat_code
        )
        
        return self._request_with_retry(url)
    
    def search_statistics(
        self,
        stat_code: str,
        frequency: str,
        start_date: str,
        end_date: str,
        item_code1: Optional[str] = None,
        item_code2: Optional[str] = None,
        item_code3: Optional[str] = None,
        item_code4: Optional[str] = None,
        start_count: int = 1,
        end_count: int = 1000
    ) -> Dict[str, Any]:
        """
        Search statistical data.
        
        Parameters
        ----------
        stat_code : str
            Statistic code (e.g., '200Y101')
        frequency : str
            Frequency code: A, S, Q, M, SM, D
        start_date : str
            Start date in format matching frequency (e.g., '2020', '2020Q1', '202001')
        end_date : str
            End date in format matching frequency
        item_code1 : str, optional
            Statistic item code 1
        item_code2 : str, optional
            Statistic item code 2
        item_code3 : str, optional
            Statistic item code 3
        item_code4 : str, optional
            Statistic item code 4
        start_count : int
            Start count for pagination
        end_count : int
            End count for pagination (max 1000 per request)
            
        Returns
        -------
        Dict[str, Any]
            API response with statistical data
        """
        url = self._build_url(
            'StatisticSearch',
            start_count=start_count,
            end_count=end_count,
            stat_code=stat_code,
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            item_code1=item_code1 or '?',
            item_code2=item_code2 or '?',
            item_code3=item_code3 or '?',
            item_code4=item_code4 or '?'
        )
        
        return self._request_with_retry(url)

