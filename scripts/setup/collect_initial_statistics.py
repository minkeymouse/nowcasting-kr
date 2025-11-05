"""Collect initial statistics list from BOK API and save to CSV.

This is a one-time setup script for initial data collection.
Run this before uploading metadata to the database.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import httpx
import time
import pandas as pd
from dotenv import load_dotenv
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables - check multiple locations
env_paths = [
    project_root / '.env.local',
    project_root.parent / '.env.local',
    Path.home() / 'Nowcasting' / '.env.local',
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        break

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import APIError from services (after path setup)
from services.api.bok_client import BOKAPIError


def parse_statistic_table_response(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse BOK StatisticTableList API response.
    
    Parameters
    ----------
    response : Dict[str, Any]
        API response dictionary
        
    Returns
    -------
    List[Dict[str, Any]]
        List of parsed statistics table records
    """
    records = []
    
    # Check for error in RESULT
    if 'RESULT' in response:
        result = response['RESULT']
        code = result.get('CODE', '')
        if code:
            if 'ERROR' in code or ('INFO' in code and code != '정보-200'):
                error_msg = result.get('MESSAGE', 'Unknown error')
                raise BOKAPIError(f"API error: {error_msg}", code)
            elif code == '정보-200':
                # 데이터 없음은 정상
                return records
    
    # Extract StatisticTableList data
    if 'StatisticTableList' in response:
        table_list = response['StatisticTableList']
        
        # Handle both single row and list of rows
        rows = table_list.get('row', [])
        if not isinstance(rows, list):
            rows = [rows] if rows else []
        
        for row in rows:
            # Handle nested structure (sometimes row is wrapped)
            if isinstance(row, dict):
                record = {
                    'stat_code': str(row.get('STAT_CODE', '') or row.get('stat_code', '') or ''),
                    'stat_name': str(row.get('STAT_NAME', '') or row.get('stat_name', '') or ''),
                    'stat_name_eng': str(row.get('STAT_NAME_ENG', '') or row.get('stat_name_eng', '') or ''),
                    'cycle': str(row.get('CYCLE', '') or row.get('cycle', '') or ''),  # 주기: A, S, Q, M, SM, D
                    'srch_yn': str(row.get('SRCH_YN', '') or row.get('srch_yn', '') or ''),  # 검색 가능 여부
                    'org_name': str(row.get('ORG_NAME', '') or row.get('org_name', '') or ''),
                    'p_cycle': str(row.get('P_CYCLE', '') or row.get('p_cycle', '') or ''),  # 주기 코드
                    'p_stat_code': str(row.get('P_STAT_CODE', '') or row.get('p_stat_code', '') or ''),  # 상위 통계 코드
                    'p_item_code': str(row.get('P_ITEM_CODE', '') or row.get('p_item_code', '') or ''),  # 상위 항목 코드
                }
                # Only add if stat_code exists
                if record['stat_code']:
                    records.append(record)
    
    return records


def save_statistics_list_to_csv(
    records: List[Dict[str, Any]],
    output_path: Optional[Path] = None
) -> Path:
    """
    Save statistics list to CSV file.
    
    Parameters
    ----------
    records : List[Dict[str, Any]]
        List of statistics table records
    output_path : Path, optional
        Output CSV file path. If None, defaults to data/bok_statistics_list.csv
        
    Returns
    -------
    Path
        Path to saved CSV file
    """
    if not records:
        logger.warning("No records to save")
        raise ValueError("No records to save")
    
    # Default output path
    if output_path is None:
        output_path = project_root / 'data' / 'bok_statistics_list.csv'
    
    # Ensure data directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    # Save to CSV
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    logger.info(f"Saved {len(records)} records to {output_path}")
    
    return output_path


def fetch_statistics_page(
    api_key: str,
    start_count: int = 1,
    end_count: int = 1000,
    language: str = 'kr'
) -> Dict[str, Any]:
    """
    Fetch one page of statistics tables from BOK API.
    
    Parameters
    ----------
    api_key : str
        BOK API authentication key
    start_count : int
        Start count for pagination
    end_count : int
        End count for pagination
    language : str
        Language code ('kr' or 'en')
        
    Returns
    -------
    Dict[str, Any]
        API response
    """
    # URL format: /api/StatisticTableList/{AuthKey}/{Format}/{Lang}/{Start}/{End}
    url = f"https://ecos.bok.or.kr/api/StatisticTableList/{api_key}/json/{language}/{start_count}/{end_count}"
    
    logger.info(f"Fetching: {url}")
    
    with httpx.Client(timeout=60.0) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json()


def fetch_all_statistics(
    api_key: str,
    max_pages: Optional[int] = None,
    page_size: int = 1000,
    language: str = 'kr'
) -> List[Dict[str, Any]]:
    """
    Fetch all statistics tables with pagination.
    
    Parameters
    ----------
    api_key : str
        BOK API authentication key
    max_pages : Optional[int]
        Maximum number of pages to fetch. If None, fetches all pages.
    page_size : int
        Number of records per page
    language : str
        Language code ('kr' or 'en')
        
    Returns
    -------
    List[Dict[str, Any]]
        All statistics table records
    """
    all_records = []
    start_count = 1
    page = 1
    
    if max_pages is None:
        logger.info(f"Starting to fetch statistics tables (all pages, {page_size} per page)...")
    else:
        logger.info(f"Starting to fetch statistics tables (max {max_pages} pages, {page_size} per page)...")
    
    while max_pages is None or page <= max_pages:
        try:
            end_count = start_count + page_size - 1
            logger.info(f"Fetching page {page} (records {start_count}-{end_count})...")
            
            response = fetch_statistics_page(
                api_key=api_key,
                start_count=start_count,
                end_count=end_count,
                language=language
            )
            
            # Parse response
            records = parse_statistic_table_response(response)
            
            if not records:
                logger.info("No more records found")
                break
            
            all_records.extend(records)
            logger.info(f"Fetched {len(records)} records from page {page}")
            
            # Check if we got fewer records than page size (last page)
            if len(records) < page_size:
                logger.info("Reached last page")
                break
            
            # Move to next page
            start_count = end_count + 1
            page += 1
            
            # Rate limiting (0.5 second delay between requests for faster fetching)
            time.sleep(0.5)
            
        except BOKAPIError as e:
            if e.error_code == '정보-200':  # No data
                logger.info("No more data available")
                break
            else:
                logger.error(f"BOK API error: {e.message}")
                raise
        except Exception as e:
            logger.error(f"Error fetching page {page}: {e}", exc_info=True)
            raise
    
    logger.info(f"Total fetched: {len(all_records)} statistics tables")
    return all_records


def main():
    """Main entry point for initial statistics collection."""
    try:
        # Get BOK API key from environment
        api_key = os.getenv('BOK_API_KEY')
        if not api_key:
            raise ValueError("BOK_API_KEY environment variable not set")
        
        logger.info("=" * 80)
        logger.info("Initial Statistics Collection from BOK API")
        logger.info("=" * 80)
        
        # Fetch all statistics
        all_records = fetch_all_statistics(
            api_key=api_key,
            max_pages=None,  # None = fetch all pages
            page_size=1000,  # BOK API max per request
            language='kr'
        )
        
        if not all_records:
            logger.warning("No statistics found")
            return
        
        # Save to CSV
        logger.info("Saving statistics to CSV...")
        output_path = save_statistics_list_to_csv(all_records)
        
        logger.info("=" * 80)
        logger.info("Collection Summary")
        logger.info("=" * 80)
        logger.info(f"Total fetched: {len(all_records)}")
        logger.info(f"Saved to: {output_path}")
        logger.info("=" * 80)
        
        # Print sample records
        if all_records:
            logger.info("\nSample records:")
            for i, record in enumerate(all_records[:5], 1):
                name = record.get('stat_name', 'N/A')
                logger.info(f"{i}. {record.get('stat_code')}: {name}")
        
        logger.info("\nNext step: Run scripts/setup/upload_initial_metadata.py to upload to database")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

