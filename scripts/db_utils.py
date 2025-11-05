"""Shared utilities for database initialization and updates.

This module provides common functions to eliminate code duplication between
initialization.py and update_database.py.
"""

import os
import time
import logging
from typing import Optional, Dict, Any
from datetime import date
from pathlib import Path

from database import (
    get_client,
    create_vintage,
    get_latest_vintage_id,
    create_ingestion_job,
    update_ingestion_job,
)
from database.settings import BOKAPIConfig, KOSISAPIConfig
from services.api.bok_client import BOKAPIClient
from services.api.kosis_client import KOSISAPIClient

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, bok_delay: float = 0.6, kosis_delay: float = 0.5):
        """
        Initialize rate limiter.
        
        Parameters
        ----------
        bok_delay : float
            Minimum delay between BOK API calls (seconds)
        kosis_delay : float
            Minimum delay between KOSIS API calls (seconds)
        """
        self.delays = {
            'BOK': bok_delay,
            'KOSIS': kosis_delay
        }
        self.last_call_time = {}
    
    def wait_if_needed(self, api_source: str) -> None:
        """
        Wait if needed to respect rate limits.
        
        Parameters
        ----------
        api_source : str
            API source ('BOK' or 'KOSIS')
        """
        if api_source not in self.delays:
            return
        
        delay = self.delays[api_source]
        current_time = time.time()
        
        if api_source in self.last_call_time:
            time_since_last = current_time - self.last_call_time[api_source]
            if time_since_last < delay:
                sleep_time = delay - time_since_last
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s before {api_source} API call")
                time.sleep(sleep_time)
        
        self.last_call_time[api_source] = time.time()


def initialize_api_clients() -> tuple[Optional[BOKAPIClient], Optional[KOSISAPIClient]]:
    """
    Initialize BOK and KOSIS API clients.
    
    Returns
    -------
    tuple[BOKAPIClient | None, KOSISAPIClient | None]
        Tuple of (bok_client, kosis_client)
    """
    bok_client = None
    bok_key = os.getenv('BOK_API_KEY')
    if bok_key:
        try:
            bok_config = BOKAPIConfig(auth_key=bok_key)
            bok_client = BOKAPIClient(bok_config)
            logger.info("✅ BOK API client initialized")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize BOK API client: {e}")
    else:
        logger.warning("⚠️  BOK_API_KEY not set")
    
    kosis_client = None
    kosis_key = os.getenv('KOSIS_API_KEY')
    if kosis_key:
        try:
            kosis_config = KOSISAPIConfig(api_key=kosis_key)
            kosis_client = KOSISAPIClient(kosis_config)
            logger.info("✅ KOSIS API client initialized")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize KOSIS API client: {e}")
    else:
        logger.warning("⚠️  KOSIS_API_KEY not set")
    
    return bok_client, kosis_client


def ensure_vintage_and_job(
    vintage_date: Optional[date] = None,
    client: Optional[Any] = None,
    dry_run: bool = False
) -> tuple[Optional[int], Optional[int]]:
    """
    Ensure a vintage and ingestion job exist for the given date.
    
    Parameters
    ----------
    vintage_date : date, optional
        Vintage date (default: today)
    client : Any, optional
        Supabase client (default: get_client())
    dry_run : bool
        If True, don't create (returns None)
        
    Returns
    -------
    tuple[int | None, int | None]
        Tuple of (vintage_id, job_id)
    """
    if dry_run:
        return None, None
    
    if client is None:
        client = get_client()
    
    if vintage_date is None:
        vintage_date = date.today()
    
    # Create or get vintage
    vintage_id = None
    try:
        vintage_result = create_vintage(vintage_date=vintage_date, country='KR', client=client)
        vintage_id = vintage_result['vintage_id']
        logger.info(f"✅ Created vintage {vintage_id} for {vintage_date}")
    except Exception as e:
        # Vintage might already exist
        if 'already exists' in str(e).lower() or 'duplicate' in str(e).lower():
            logger.info(f"Vintage for {vintage_date} already exists, retrieving...")
            latest_vintage = get_latest_vintage_id(vintage_date=vintage_date, client=client)
            if latest_vintage:
                vintage_id = latest_vintage
                logger.info(f"✅ Retrieved existing vintage {vintage_id}")
        else:
            logger.error(f"Failed to create vintage: {e}")
            raise
    
    # Create ingestion job
    job_id = None
    try:
        # Use a default run ID if not provided (for manual runs)
        run_id = os.getenv('GITHUB_RUN_ID', f'manual-{vintage_date.isoformat()}')
        job_result = create_ingestion_job(
            vintage_date=vintage_date,
            github_run_id=run_id,
            client=client
        )
        job_id = job_result['job_id']
        logger.info(f"✅ Created ingestion job {job_id}")
    except Exception as e:
        logger.warning(f"Could not create ingestion job: {e}")
    
    return vintage_id, job_id


def finalize_ingestion_job(
    job_id: Optional[int],
    status: str = 'completed',
    successful_series: Optional[int] = None,
    failed_series: Optional[int] = None,
    total_series: Optional[int] = None,
    client: Optional[Any] = None
) -> None:
    """
    Finalize an ingestion job by updating its status and statistics.
    
    Parameters
    ----------
    job_id : int, optional
        Job ID to finalize
    status : str
        Final status of the job (e.g., 'completed', 'failed'). Defaults to 'completed'.
    successful_series : int, optional
        Number of series successfully processed.
    failed_series : int, optional
        Number of series that failed to process.
    total_series : int, optional
        Total number of series attempted.
    client : Any, optional
        Supabase client (default: get_client())
    """
    if job_id is None:
        return
    
    if client is None:
        client = get_client()
    
    try:
        update_ingestion_job(
            job_id=job_id,
            status=status,
            successful_series=successful_series,
            failed_series=failed_series,
            total_series=total_series,
            client=client
        )
        logger.info(f"✅ Updated ingestion job {job_id} to {status}")
    except Exception as e:
        logger.warning(f"Could not update ingestion job {job_id}: {e}")


def get_next_period_date(latest_date: date, frequency: str) -> str:
    """
    Calculate the next period start date based on frequency.
    
    Parameters
    ----------
    latest_date : date
        Latest observation date
    frequency : str
        Frequency code ('q', 'm', 'd', 'a')
        
    Returns
    -------
    str
        Next period start date in API format
    """
    from datetime import timedelta
    
    if frequency == 'q':
        # Add one quarter
        q = (latest_date.month - 1) // 3 + 1
        if q == 4:
            return f"{latest_date.year + 1}Q1"
        else:
            return f"{latest_date.year}Q{q + 1}"
    elif frequency == 'm':
        # Add one month
        if latest_date.month == 12:
            start_date_obj = date(latest_date.year + 1, 1, 1)
        else:
            start_date_obj = date(latest_date.year, latest_date.month + 1, 1)
        return start_date_obj.strftime('%Y%m')
    elif frequency == 'd':
        # Add one day
        start_date_obj = latest_date + timedelta(days=1)
        return start_date_obj.strftime('%Y%m%d')
    else:
        # Default: add one year
        start_date_obj = date(latest_date.year + 1, 1, 1)
        return start_date_obj.strftime('%Y')

