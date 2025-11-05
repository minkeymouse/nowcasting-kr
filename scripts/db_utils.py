"""Shared utilities for database initialization and updates."""
import os
import time
import logging
from typing import Optional
from datetime import date
from database import get_client, create_vintage, get_latest_vintage_id, create_ingestion_job, update_ingestion_job
from database.settings import BOKAPIConfig, KOSISAPIConfig
from services.api.bok_client import BOKAPIClient
from services.api.kosis_client import KOSISAPIClient

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter for API calls."""
    def __init__(self, bok_delay: float = 0.6, kosis_delay: float = 0.5):
        self.delays = {'BOK': bok_delay, 'KOSIS': kosis_delay}
        self.last_call_time = {}
    
    def wait_if_needed(self, api_source: str) -> None:
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

def initialize_api_clients():
    """Initialize BOK and KOSIS API clients."""
    bok_client = None
    if bok_key := os.getenv('BOK_API_KEY'):
        try:
            bok_config = BOKAPIConfig(auth_key=bok_key)
            bok_client = BOKAPIClient(bok_config)
            logger.info("✅ BOK API client initialized")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize BOK API client: {e}")
    else:
        logger.warning("⚠️  BOK_API_KEY not set")
    
    kosis_client = None
    if kosis_key := os.getenv('KOSIS_API_KEY'):
        try:
            kosis_config = KOSISAPIConfig(api_key=kosis_key)
            kosis_client = KOSISAPIClient(kosis_config)
            logger.info("✅ KOSIS API client initialized")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize KOSIS API client: {e}")
    else:
        logger.warning("⚠️  KOSIS_API_KEY not set")
    
    return bok_client, kosis_client

def ensure_vintage_and_job(vintage_date: Optional[date] = None, client=None, dry_run: bool = False):
    """Ensure a vintage and ingestion job exist."""
    if dry_run:
        return None, None
    if client is None:
        client = get_client()
    if vintage_date is None:
        vintage_date = date.today()
    
    vintage_id = None
    try:
        vintage_result = create_vintage(vintage_date=vintage_date, country='KR', client=client)
        vintage_id = vintage_result['vintage_id']
        logger.info(f"✅ Created vintage {vintage_id} for {vintage_date}")
    except Exception as e:
        if 'already exists' in str(e).lower() or 'duplicate' in str(e).lower():
            logger.info(f"Vintage for {vintage_date} already exists, retrieving...")
            latest_vintage = get_latest_vintage_id(vintage_date=vintage_date, client=client)
            if latest_vintage:
                vintage_id = latest_vintage
                logger.info(f"✅ Retrieved existing vintage {vintage_id}")
        else:
            logger.error(f"Failed to create vintage: {e}")
            raise
    
    job_id = None
    try:
        job_result = create_ingestion_job(vintage_date=vintage_date, github_run_id=None, client=client)
        job_id = job_result['job_id']
        logger.info(f"✅ Created ingestion job {job_id}")
    except Exception as e:
        logger.warning(f"Could not create ingestion job: {e}")
    
    return vintage_id, job_id

def finalize_ingestion_job(job_id: Optional[int], client=None):
    """Finalize an ingestion job by updating its status."""
    if job_id is None or client is None:
        return
    try:
        update_ingestion_job(job_id, status='completed', client=client)
        logger.info(f"✅ Updated ingestion job {job_id} to completed")
    except Exception as e:
        logger.warning(f"Could not update ingestion job {job_id}: {e}")

def get_next_period_date(latest_date: date, frequency: str) -> str:
    """Calculate the next period start date based on frequency."""
    from datetime import timedelta
    if frequency == 'q':
        q = (latest_date.month - 1) // 3 + 1
        return f"{latest_date.year + 1}Q1" if q == 4 else f"{latest_date.year}Q{q + 1}"
    elif frequency == 'm':
        if latest_date.month == 12:
            start_date_obj = date(latest_date.year + 1, 1, 1)
        else:
            start_date_obj = date(latest_date.year, latest_date.month + 1, 1)
        return start_date_obj.strftime('%Y%m')
    elif frequency == 'd':
        return (latest_date + timedelta(days=1)).strftime('%Y%m%d')
    else:
        return date(latest_date.year + 1, 1, 1).strftime('%Y')
