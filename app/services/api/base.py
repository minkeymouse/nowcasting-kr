"""Base API client interface for data source integrations."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import date


class APIError(Exception):
    """Base API error for all data sources."""
    def __init__(self, message: str, error_code: Optional[str] = None, source: str = "unknown"):
        self.error_code = error_code
        self.source = source
        self.message = message
        super().__init__(self.message)


class BaseAPIClient(ABC):
    """Base interface for API clients from different data sources."""
    
    @abstractmethod
    def fetch_statistics_list(
        self,
        start_count: int = 1,
        end_count: int = 1000,
        language: str = 'kr'
    ) -> Dict[str, Any]:
        """
        Fetch list of available statistics.
        
        Returns
        -------
        Dict[str, Any]
            API response with statistics list
        """
        pass
    
    @abstractmethod
    def fetch_statistic_items(
        self,
        stat_code: str,
        start_count: int = 1,
        end_count: int = 1000
    ) -> Dict[str, Any]:
        """
        Fetch items for a specific statistic.
        
        Parameters
        ----------
        stat_code : str
            Statistic code
            
        Returns
        -------
        Dict[str, Any]
            API response with items list
        """
        pass
    
    @abstractmethod
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
            Statistic code
        frequency : str
            Frequency code (A, S, Q, M, SM, D, etc.)
        start_date : str
            Start date in format matching frequency
        end_date : str
            End date in format matching frequency
        item_code1-4 : str, optional
            Item codes for hierarchical statistics
            
        Returns
        -------
        Dict[str, Any]
            API response with time-series data
        """
        pass
    
    @property
    @abstractmethod
    def source_code(self) -> str:
        """Return the data source code (e.g., 'BOK', 'KOSIS')."""
        pass

