"""Shared utilities for database initialization and updates.

This module provides common functions to eliminate code duplication between
initialization.py, test_initialization.py, and ingest_api.py.
"""

import os
import time
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd

from database import get_client, generate_series_id
from database.operations import (
    ensure_vintage_and_job,
    finalize_ingestion_job,
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


def fetch_series_data(
    series_id: str,
    api_code: str,
    api_client: Any,
    source: str,
    frequency: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch data for a single series from API.
    
    Parameters
    ----------
    series_id : str
        Series ID (e.g., BOK_200Y001)
    api_code : str
        API code for the statistic (e.g., 200Y001)
    api_client : Any
        API client instance (BOKAPIClient or KOSISAPIClient)
    source : str
        Source code (BOK or KOSIS)
    frequency : str
        Frequency code (q, m, d, etc.)
    start_date : str, optional
        Start date in API format
    end_date : str, optional
        End date in API format
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, value, series_id
    """
    
    # Map frequency to API format
    freq_map = {'q': 'Q', 'm': 'M', 'd': 'D', 'a': 'A', 'sa': 'S'}
    api_freq = freq_map.get(frequency.lower(), frequency.upper())
    
    # Set date range if not provided
    if not end_date:
        end_date_obj = datetime.now()
        if api_freq == 'Q':
            q = (end_date_obj.month - 1) // 3 + 1
            end_date = f"{end_date_obj.year}Q{q}"
        elif api_freq == 'M':
            end_date = end_date_obj.strftime('%Y%m')
        elif api_freq == 'D':
            end_date = end_date_obj.strftime('%Y%m%d')
        else:
            end_date = end_date_obj.strftime('%Y')
    
    if not start_date:
        start_date_obj = datetime.now() - timedelta(days=365*10)  # 10 years
        if api_freq == 'Q':
            q = (start_date_obj.month - 1) // 3 + 1
            start_date = f"{start_date_obj.year}Q{q}"
        elif api_freq == 'M':
            start_date = start_date_obj.strftime('%Y%m')
        elif api_freq == 'D':
            start_date = start_date_obj.strftime('%Y%m%d')
        else:
            start_date = start_date_obj.strftime('%Y')
    
    try:
        # Fetch data
        # For KOSIS: item_code1=objL1 (필수), item_code2=itmId (필수)
        # Different tables require different parameters:
        # - DT_1J22003 (CPI): objL1='ALL', itmId='T'
        # - DT_1JH20201 (Production Index): objL1='0 1', itmId='T1'
        # - DT_1DA7002S (Employment Rate): objL1='ALL', itmId='T90'
        # For BOK: these parameters are ignored
        if source == 'KOSIS':
            # Determine parameters based on table ID
            if 'DT_1JH20201' in api_code:
                # 전산업생산지수: objL1='0 1', itmId='T1'
                item_code1_val = '0 1'
                item_code2_val = 'T1'
            elif 'DT_1DA7002S' in api_code:
                # 고용율: objL1='ALL', itmId='T90'
                item_code1_val = 'ALL'
                item_code2_val = 'T90'
            else:
                # Default (CPI and others): objL1='ALL', itmId='T'
                item_code1_val = 'ALL'
                item_code2_val = 'T'
        else:
            item_code1_val = '?'
            item_code2_val = '?'
        
        data_result = api_client.fetch_statistic_data(
            stat_code=api_code,
            frequency=api_freq,
            start_date=start_date,
            end_date=end_date,
            item_code1=item_code1_val,
            item_code2=item_code2_val,
            item_code3=None,  # Don't pass '?' for optional params
            item_code4=None
        )
        
        # KOSIS returns list directly, BOK returns dict with 'StatisticSearch'
        if source == 'KOSIS':
            # KOSIS response is a list directly (wrapped in {'data': [...]} by client)
            if isinstance(data_result, dict) and 'data' in data_result:
                rows = data_result['data']
            elif isinstance(data_result, list):
                rows = data_result
            else:
                logger.warning(f"No data returned for {series_id}")
                return pd.DataFrame()
        else:
            # BOK response format - check for rate limiting errors
            if 'RESULT' in data_result:
                result = data_result['RESULT']
                if isinstance(result, dict) and 'CODE' in result:
                    error_code = result.get('CODE', '')
                    error_msg = result.get('MESSAGE', '')
                    if 'ERROR' in error_code or '602' in error_code:
                        logger.error(f"BOK API rate limit error for {series_id}: {error_msg}")
                        raise Exception(f"BOK API rate limit: {error_msg}")
                    else:
                        logger.warning(f"BOK API error for {series_id}: {error_code} - {error_msg}")
                        return pd.DataFrame()
            
            if 'StatisticSearch' not in data_result:
                logger.warning(f"No data returned for {series_id}")
                return pd.DataFrame()
            rows = data_result['StatisticSearch'].get('row', [])
        
        if not rows:
            logger.warning(f"Empty data for {series_id}")
            return pd.DataFrame()
        
        # Parse into DataFrame
        data_list = []
        for row in rows:
            # KOSIS uses PRD_DE and DT, BOK uses TIME and DATA_VALUE
            if source == 'KOSIS':
                time_str = row.get('PRD_DE', '')
                value = row.get('DT')
                # Filter by C1 value based on table:
                # - DT_1J22003 (CPI): C1='T10' (전국/whole country)
                # - DT_1JH20201 (Production Index): C1='0' (전산업생산지수)
                # - DT_1DA7002S (Employment Rate): Filter to overall rate (typically C1='T90' or similar)
                c1 = row.get('C1', '')
                
                if 'DT_1J22003' in series_id:
                    # CPI: Use C1='T10' (전국/whole country)
                    if c1 and c1 != 'T10':
                        continue
                elif 'DT_1JH20201' in series_id:
                    # Production Index: Use C1='0' (전산업생산지수)
                    if c1 and c1 != '0':
                        continue
                elif 'DT_1DA7002S' in series_id:
                    # Employment Rate: Filter to overall rate
                    # Based on URL, itmId='T90' suggests overall rate
                    # For now, accept all but could be refined if needed
                    pass  # Accept all for Employment Rate (can filter later if needed)
            else:
                time_str = row.get('TIME', '')
                value = row.get('DATA_VALUE')
            
            if value is None or value == '':
                continue
            
            # Parse date based on frequency
            try:
                if api_freq == 'Q':
                    # Format: 2023Q1 or YYYYMM (for monthly data that's quarterly)
                    if 'Q' in time_str:
                        year, q = time_str.split('Q')
                        month = int(q) * 3
                        date_obj = datetime(int(year), month, 1).date()
                    else:
                        # Try YYYYMM format
                        date_obj = datetime.strptime(time_str[:6], '%Y%m').date()
                elif api_freq == 'M':
                    # Format: YYYYMM (e.g., 202301)
                    date_obj = datetime.strptime(time_str[:6], '%Y%m').date()
                elif api_freq == 'D':
                    # Format: YYYYMMDD (e.g., 20230101)
                    date_obj = datetime.strptime(time_str[:8], '%Y%m%d').date()
                else:
                    # Annual or other
                    date_obj = datetime(int(time_str[:4]), 12, 31).date()
                
                data_list.append({
                    'date': date_obj,
                    'value': float(value),
                    'series_id': series_id
                })
            except Exception as e:
                logger.warning(f"Failed to parse date {time_str} for {series_id}: {e}")
                continue
        
        if not data_list:
            logger.warning(f"No valid data points after parsing for {series_id}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data_list)
        logger.info(f"Parsed {len(df)} valid observations for {series_id}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for {series_id}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return pd.DataFrame()


# ============================================================================
# CSV Loading and Parsing Utilities
# ============================================================================

def load_series_from_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """
    Load and parse series from CSV specification file.
    
    Parameters
    ----------
    csv_path : Path
        Path to CSV file
        
    Returns
    -------
    List[Dict[str, Any]]
        List of series dictionaries with all required fields
    """
    df = pd.read_csv(csv_path)
    series_list = []
    
    for _, row in df.iterrows():
        api_source = row.get('api_source', '')
        data_code = row.get('data_code', '')
        item_id = row.get('item_id', '')
        series_id = generate_series_id(api_source, data_code, item_id)
        
        # Extract block assignments
        block_cols = [col for col in df.columns if col.startswith('Block_')]
        blocks = {}
        for col in block_cols:
            block_name = col.replace('Block_', '')
            blocks[block_name] = int(row.get(col, 0))
        
        series_list.append({
            'series_id': series_id,
            'series_name': row['series_name'],
            'frequency': row['frequency'],
            'transformation': row['transformation'],
            'category': row.get('category'),
            'units': row.get('units'),
            'data_code': data_code,
            'item_id': item_id,
            'api_source': api_source,
            'blocks': blocks
        })
    
    return series_list


def get_block_names_from_csv(csv_path: Path) -> List[str]:
    """
    Extract block names from CSV columns.
    
    Parameters
    ----------
    csv_path : Path
        Path to CSV file
        
    Returns
    -------
    List[str]
        List of block names
    """
    df = pd.read_csv(csv_path)
    block_cols = [col for col in df.columns if col.startswith('Block_')]
    block_names = [col.replace('Block_', '') for col in block_cols]
    return block_names if block_names else ['Global', 'Consumption', 'Investment', 'External']


# ============================================================================
# Series Processing Utilities
# ============================================================================

def process_series(
    series_cfg: Dict[str, Any],
    bok_client: Optional[BOKAPIClient],
    kosis_client: Optional[KOSISAPIClient],
    rate_limiter: 'RateLimiter',
    vintage_id: Optional[int],
    client: Optional[Any] = None,
    dry_run: bool = False,
    github_run_id: Optional[str] = None
) -> Tuple[Optional[pd.DataFrame], bool, Optional[str]]:
    """
    Process a single series: fetch data and save metadata.
    
    Parameters
    ----------
    series_cfg : Dict[str, Any]
        Series configuration dictionary
    bok_client : Optional[BOKAPIClient]
        BOK API client
    kosis_client : Optional[KOSISAPIClient]
        KOSIS API client
    rate_limiter : RateLimiter
        Rate limiter instance
    vintage_id : Optional[int]
        Vintage ID for observations
    client : Optional[Any]
        Database client
    dry_run : bool
        If True, don't save to database
    github_run_id : Optional[str]
        GitHub run ID
        
    Returns
    -------
    Tuple[Optional[pd.DataFrame], bool, Optional[str]]
        (dataframe, success, error_message)
    """
    series_id = series_cfg['series_id']
    data_code = series_cfg.get('data_code')
    item_id = series_cfg.get('item_id')
    api_source = series_cfg.get('api_source')
    
    if not data_code or not api_source:
        return None, False, f"Missing data_code or api_source"
    
    # Get appropriate API client
    api_client = None
    if api_source == 'BOK' and bok_client:
        api_client = bok_client
        rate_limiter.wait_if_needed(api_source)
    elif api_source == 'KOSIS' and kosis_client:
        api_client = kosis_client
        rate_limiter.wait_if_needed(api_source)
    else:
        return None, False, f"No API client available for {api_source}"
    
    # Fetch data
    try:
        df_data = fetch_series_data(
            series_id=series_id,
            api_code=data_code,
            api_client=api_client,
            source=api_source,
            frequency=series_cfg['frequency']
        )
        
        if df_data.empty:
            return None, False, "No data fetched"
        
        # Save series metadata if not dry run
        if not dry_run and client:
            from database.models import SeriesModel
            from database.operations import get_series
            
            series_model = SeriesModel(
                series_id=series_id,
                series_name=series_cfg['series_name'],
                frequency=series_cfg['frequency'],
                transformation=series_cfg['transformation'],
                units=series_cfg.get('units'),
                category=series_cfg.get('category'),
                api_source=api_source,
                data_code=data_code,
                item_id=item_id
            )
            
            existing_series = get_series(series_id, client=client)
            if not existing_series:
                # Build data dict directly to ensure data_code and item_id are included
                # data_code is required (NOT NULL in DB), so it must be present
                if not data_code:
                    logger.error(f"data_code is None for {series_id}, cannot insert")
                    return None, False, f"data_code is required but was None"
                
                data = {
                    'series_id': series_id,
                    'series_name': series_cfg['series_name'],
                    'frequency': series_cfg['frequency'],
                    'transformation': series_cfg.get('transformation'),
                    'units': series_cfg.get('units'),
                    'category': series_cfg.get('category'),
                    'api_source': api_source,
                    'data_code': data_code,  # Required, should not be None
                    'item_id': item_id,  # Optional, can be None
                    'is_active': True,
                    'is_kpi': series_cfg.get('is_kpi', False)
                }
                # Remove None values for optional fields only, but ALWAYS keep data_code and item_id
                # Build filtered dict explicitly to ensure data_code is never removed
                filtered_data = {}
                for k, v in data.items():
                    if v is not None:
                        filtered_data[k] = v
                    elif k in ('data_code', 'item_id'):
                        # Keep data_code and item_id even if None (though data_code should never be None)
                        filtered_data[k] = v
                data = filtered_data
                
                # Log the data before insertion for debugging
                logger.info(f"About to insert series {series_id}: data_code={data.get('data_code')}, item_id={data.get('item_id')}")
                logger.info(f"Data keys: {list(data.keys())}")
                
                # Double-check data_code is present and not None
                if 'data_code' not in data:
                    logger.error(f"data_code key missing from data dict for {series_id}")
                    return None, False, f"data_code key missing from data dict"
                if data['data_code'] is None:
                    logger.error(f"data_code is None for {series_id}, cannot insert")
                    return None, False, f"data_code is None"
                
                # Debug: Log what we're about to insert
                logger.debug(f"Inserting series {series_id} with data_code={data.get('data_code')}, item_id={data.get('item_id')}")
                logger.debug(f"Full data dict keys: {list(data.keys())}")
                
                try:
                    client.table('series').insert(data).execute()
                except Exception as e:
                    logger.error(f"Failed to insert series {series_id}: {e}")
                    logger.error(f"Data dict was: {data}")
                    raise
        
        # Add vintage_id
        df_data['vintage_id'] = vintage_id
        return df_data, True, None
        
    except Exception as e:
        return None, False, str(e)


# ============================================================================
# Statistics and Reporting Utilities
# ============================================================================

def print_statistics_summary(stats: Dict[str, Any], vintage_id: Optional[int] = None) -> None:
    """
    Print formatted statistics summary.
    
    Parameters
    ----------
    stats : Dict[str, Any]
        Statistics dictionary
    vintage_id : Optional[int]
        Vintage ID to include in summary
    """
    print("\n" + "=" * 80)
    print("📊 STATISTICS SUMMARY")
    print("=" * 80)
    
    if vintage_id:
        print(f"📦 Vintage ID: {vintage_id}")
    
    print(f"\nProcessing Results:")
    print(f"   Total series: {stats['total']}")
    print(f"   ✅ Successful: {stats['successful']}")
    print(f"   ❌ Failed: {stats['failed']}")
    print(f"   ⏭️  Skipped: {stats['skipped']}")
    
    if 'observations_inserted' in stats:
        print(f"   📈 Observations inserted: {stats['observations_inserted']}")
    
    if stats.get('errors'):
        print(f"\n⚠️  Errors ({len(stats['errors'])}):")
        for error in stats['errors'][:10]:
            if isinstance(error, dict):
                print(f"   - {error.get('series_id', 'unknown')}: {error.get('error', 'unknown error')[:80]}")
            else:
                print(f"   - {error[:80]}")
        if len(stats['errors']) > 10:
            print(f"   ... and {len(stats['errors']) - 10} more")
    
    success_rate = (stats['successful'] / stats['total'] * 100) if stats['total'] > 0 else 0
    print(f"\n📈 Success rate: {success_rate:.1f}%")
    
    if stats['failed'] == 0 and stats['successful'] == stats['total']:
        print("\n🎉 All series processed successfully!")
    elif stats['failed'] > 0:
        print(f"\n⚠️  {stats['failed']} series failed to process")
    
    print("=" * 80)

