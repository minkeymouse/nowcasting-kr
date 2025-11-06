"""Main orchestrator for data ingestion workflow."""

import os
import logging
from typing import Optional, Dict, Any
from datetime import date, datetime

from database import (
    get_client,
    get_source_id,
    get_source_code,
    create_vintage,
    update_vintage_status,
    create_ingestion_job,
    update_ingestion_job,
    insert_observations_from_dataframe,
    get_statistics_metadata,
    get_active_items_for_statistic,
    update_statistics_metadata_status,
    list_dfm_selected_statistics,
)
from database.operations import create_series_from_item, create_or_get_series
from services.ingestion.bok import BOKIngestion
from services.ingestion.kosis import KOSISIngestion
from database.settings import AppSettings, DataSourceConfig
from database.helpers import map_frequency_to_code
from services.api.base import APIError

logger = logging.getLogger(__name__)


class DataIngestionOrchestrator:
    """Orchestrates the complete data ingestion workflow."""
    
    def __init__(self):
        """Initialize orchestrator with settings."""
        self.settings = AppSettings.load()
        self.api_sources = AppSettings.load_api_sources()
        self.client = get_client()
        
        # Initialize data sources
        self._sources = {}
        self._initialize_sources()
        
        # GitHub Actions context
        self.github_run_id = os.getenv('GITHUB_RUN_ID')
        self.github_run_url = os.getenv('GITHUB_RUN_URL') or (
            f"{os.getenv('GITHUB_SERVER_URL', '')}/{os.getenv('GITHUB_REPOSITORY', '')}/actions/runs/{self.github_run_id}"
            if self.github_run_id else None
        )
        
        # Source ID cache
        self._source_cache = {}
    
    def _initialize_sources(self):
        """Initialize data sources."""
        if self.settings.bok_api_auth_key:
            self._sources['BOK'] = {
                'ingestion': BOKIngestion()
            }
        
        if self.settings.kosis_api_config:
            self._sources['KOSIS'] = {
                'ingestion': KOSISIngestion()
            }
        
        if not self._sources:
            logger.warning("No data sources registered. Check API key configuration.")
    
    @property
    def metadata_collector(self):
        """Get default metadata collector."""
        if not self._sources:
            raise ValueError("No data sources registered. Check API key configuration.")
        return next(iter(s['ingestion'] for s in self._sources.values()))
    
    def run(
        self,
        vintage_date: Optional[date] = None,
        update_metadata: bool = False
    ) -> dict:
        """
        Run the complete ingestion workflow.
        
        Parameters
        ----------
        vintage_date : date, optional
            Vintage date (defaults to today)
        update_metadata : bool
            Whether to update metadata before data collection
            
        Returns
        -------
        dict
            Summary of ingestion results
        """
        if vintage_date is None:
            vintage_date = date.today()
        
        logger.info(f"Starting data ingestion for vintage: {vintage_date}")
        
        # Step 0: Update metadata if requested
        if update_metadata:
            logger.info("Updating metadata before data collection")
            try:
                self.metadata_collector.update_all_metadata(
                    collect_items=True,
                    create_series=True
                )
            except Exception as e:
                logger.warning(f"Metadata update failed: {e}")
        
        # Step 1: Create vintage
        vintage = create_vintage(
            vintage_date=vintage_date,
            github_run_id=self.github_run_id,
            github_workflow_run_url=self.github_run_url,
            client=self.client
        )
        vintage_id = vintage['vintage_id']
        
        # Step 2: Create ingestion job
        job = create_ingestion_job(
            github_run_id=self.github_run_id or '',
            vintage_date=vintage_date,
            client=self.client
        )
        job_id = job['job_id']
        
        # Step 3: Get DFM-selected statistics
        dfm_stats = []
        for source_code in self._sources.keys():
            try:
                source_id = self._get_source_id(source_code)
                stats = list_dfm_selected_statistics(source_id=source_id, client=self.client)
                dfm_stats.extend(stats)
            except Exception as e:
                logger.warning(f"Failed to get statistics for source {source_code}: {e}")
        
        if dfm_stats:
            # Use DFM-selected statistics from metadata
            logger.info(f"Processing {len(dfm_stats)} DFM-selected statistics")
            enabled_sources = None
            total_series = len(dfm_stats)
        else:
            # Fallback to config-based sources
            enabled_sources = [s for s in self.api_sources if s.enabled]
            total_series = len(enabled_sources)
            logger.info(f"Processing {total_series} enabled data sources from config")
        
        # Step 4: Process each source
        stats = {
            'total': total_series,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }
        
        if enabled_sources:
            # Process config-based sources (legacy)
            for idx, source in enumerate(enabled_sources, 1):
                logger.info(f"[{idx}/{total_series}] Processing: {source.name} ({source.api_code})")
                try:
                    success = self._process_source_legacy(source, vintage_id, job_id, vintage_date)
                    if success:
                        stats['successful'] += 1
                    else:
                        stats['skipped'] += 1
                except Exception as e:
                    stats['failed'] += 1
                    error_msg = f"Error processing {source.name}: {str(e)}"
                    stats['errors'].append(error_msg)
                    logger.error(error_msg, exc_info=True)
        else:
            # Process DFM-selected statistics
            for idx, stat in enumerate(dfm_stats, 1):
                stat_code = stat['source_stat_code']
                stat_name = stat.get('source_stat_name', stat_code)
                logger.info(f"[{idx}/{total_series}] Processing: {stat_name} ({stat_code})")
                try:
                    success = self._process_statistic(stat, vintage_id, job_id, vintage_date)
                    if success:
                        stats['successful'] += 1
                    else:
                        stats['skipped'] += 1
                except Exception as e:
                    stats['failed'] += 1
                    error_msg = f"Error processing {stat_code}: {str(e)}"
                    stats['errors'].append(error_msg)
                    logger.error(error_msg, exc_info=True)
                
                # Update job progress
                update_ingestion_job(
                    job_id=job_id,
                    total_series=total_series,
                    successful_series=stats['successful'],
                    failed_series=stats['failed'],
                    logs_json={
                        'current_series': idx,
                        'total_series': total_series,
                        'progress': f"{idx}/{total_series}",
                        'timestamp': datetime.now().isoformat()
                    },
                    client=self.client
                )
        
        # Step 5: Finalize
        update_ingestion_job(
            job_id=job_id,
            status='completed' if stats['failed'] == 0 else 'completed_with_errors',
            completed_at=datetime.now(),
            client=self.client
        )
        
        update_vintage_status(
            vintage_id=vintage_id,
            status='completed' if stats['failed'] == 0 else 'completed_with_errors',
            client=self.client
        )
        
        logger.info(f"Ingestion completed: {stats['successful']} successful, {stats['failed']} failed")
        
        return {
            'vintage_id': vintage_id,
            'job_id': job_id,
            'stats': stats
        }
    
    def _process_statistic(
        self,
        stat_metadata: Dict[str, Any],
        vintage_id: int,
        job_id: int,
        vintage_date: date
    ) -> bool:
        """
        Process a single statistic from metadata.
        
        Parameters
        ----------
        stat_metadata : Dict[str, Any]
            Statistics metadata dictionary
        vintage_id : int
            Vintage ID
        job_id : int
            Ingestion job ID
        vintage_date : date
            Vintage date
            
        Returns
        -------
        bool
            True if successful, False if skipped
        """
        stat_code = stat_metadata['source_stat_code']
        statistics_metadata_id = stat_metadata['id']
        stat_name = stat_metadata.get('source_stat_name', stat_code)
        
        try:
            # Get active items for this statistic
            items = get_active_items_for_statistic(
                statistics_metadata_id=statistics_metadata_id,
                cycle=stat_metadata.get('cycle'),
                client=self.client
            )
            
            if not items:
                logger.warning(f"No active items found for {stat_code}")
                return False
            
            # Get source code from metadata
            source_code = self._get_source_code_from_metadata(stat_metadata, statistics_metadata_id)
            
            # Process each item
            total_observations = 0
            for item in items:
                success = self._process_item(
                    stat_code=stat_code,
                    stat_name=stat_name,
                    statistics_metadata_id=statistics_metadata_id,
                    item=item,
                    vintage_id=vintage_id,
                    job_id=job_id,
                    source_code=source_code
                )
                if success:
                    total_observations += success
            
            # Update statistics metadata status
            if total_observations > 0:
                source_id = stat_metadata.get('source_id')
                update_statistics_metadata_status(
                    source_id=source_id,
                    source_stat_code=stat_code,
                    last_data_fetch_date=vintage_date,
                    last_data_fetch_status='success',
                    client=self.client
                )
            
            return total_observations > 0
            
        except Exception as e:
            logger.error(f"Error processing statistic {stat_code}: {e}", exc_info=True)
            return False
    
    def _process_item(
        self,
        stat_code: str,
        stat_name: str,
        statistics_metadata_id: int,
        item: Dict[str, Any],
        vintage_id: int,
        job_id: int,
        source_code: str
    ) -> int:
        """
        Process a single item: fetch data, ensure series exists, insert observations.
        
        Parameters
        ----------
        stat_code : str
            Statistic code
        stat_name : str
            Statistic name
        statistics_metadata_id : int
            Statistics metadata ID
        item : Dict[str, Any]
            Item dictionary
        vintage_id : int
            Vintage ID
        job_id : int
            Job ID
        source_code : str
            Source code (e.g., 'BOK')
            
        Returns
        -------
        int
            Number of observations inserted, 0 if failed
        """
        item_code = item['item_code']
        cycle = item['cycle']
        
        try:
            # Ensure series exists
            series = create_series_from_item(
                source_code=source_code,
                stat_code=stat_code,
                stat_name=stat_name,
                item=item,
                statistics_metadata_id=statistics_metadata_id,
                frequency_mapper=map_frequency_to_code,
                client=self.client
            )
            
            # Get appropriate ingestion handler
            ingestion = self._get_ingestion(source_code)
            
            # Fetch data
            response = ingestion.api_client.fetch_statistic_data(
                stat_code=stat_code,
                frequency=cycle,
                start_date=item.get('start_time') or '2020',
                end_date=item.get('end_time') or '2024',
                item_code1=item_code,
                item_code2=None,
                item_code3=None,
                item_code4=None
            )
            
            # Parse to DataFrame
            df = ingestion.transform_statistic_data(
                response=response,
                stat_code=stat_code,
                frequency=cycle,
                item_code=item_code
            )
            
            if df.empty:
                logger.debug(f"No data for {stat_code}/{item_code}")
                return 0
            
            # Insert observations
            inserted = insert_observations_from_dataframe(
                df=df,
                vintage_id=vintage_id,
                job_id=job_id,
                api_source=source_code,
                client=self.client
            )
            
            logger.info(f"Processed {item_code}: {inserted} observations")
            return inserted
            
        except Exception as e:
            logger.error(f"Error processing item {item_code} for {stat_code}: {e}", exc_info=True)
            return 0
    
    def _process_source_legacy(
        self,
        source: DataSourceConfig,
        vintage_id: int,
        job_id: int,
        vintage_date: date
    ) -> bool:
        """
        Legacy method for processing source without metadata (config-based).
        
        Returns
        -------
        bool
            True if successful, False if skipped
        """
        try:
            # Legacy config: use first available source
            if not self._sources:
                raise ValueError("No data sources registered. Cannot process legacy source.")
            source_code = next(iter(self._sources.keys()))
            
            ingestion = self._get_ingestion(source_code)
            
            # Fetch data from API
            response = ingestion.api_client.fetch_statistic_data(
                stat_code=source.api_code,
                frequency=source.frequency,
                start_date=source.start_date,
                end_date=source.end_date or '2024',
                item_code1=source.item_code1,
                item_code2=source.item_code2,
                item_code3=source.item_code3,
                item_code4=source.item_code4
            )
            
            # Parse to DataFrame
            df = ingestion.transform_statistic_data(
                response=response,
                stat_code=source.api_code,
                frequency=source.frequency,
                item_code=source.item_code1
            )
            
            if df.empty:
                logger.warning(f"No data returned for {source.name}")
                return False
            
            # Create series
            series = create_or_get_series(
                source_code=source_code,
                stat_code=source.api_code,
                series_name=source.name,
                frequency=map_frequency_to_code(source.frequency),
                item_code=source.item_code1,
                api_source=source_code,
                client=self.client
            )
            
            # Insert observations
            inserted = insert_observations_from_dataframe(
                df=df,
                vintage_id=vintage_id,
                job_id=job_id,
                api_source=source_code,
                client=self.client
            )
            
            logger.info(f"Inserted {inserted} observations for {source.name}")
            return inserted > 0
            
        except APIError as e:
            # Handle API errors (generic for all sources)
            # Check for "no data available" error codes (source-specific but handled generically)
            # Common patterns: '정보-200' (BOK), 'NO_DATA' (generic), etc.
            no_data_codes = ['정보-200', 'NO_DATA', 'NO_DATA_AVAILABLE']
            if hasattr(e, 'error_code') and e.error_code in no_data_codes:
                logger.info(f"No data available for {source.name}: {e.error_code}")
                return False
            else:
                logger.error(f"API error for {source.name}: {e.message}")
                raise
        except Exception as e:
            logger.error(f"Error processing {source.name}: {e}", exc_info=True)
            raise
    
    def _get_source_id(self, source_code: str) -> int:
        """Get source_id from database (cached)."""
        if source_code not in self._source_cache:
            self._source_cache[source_code] = get_source_id(source_code, client=self.client)
        return self._source_cache[source_code]
    
    def _get_source_code_from_metadata(
        self,
        stat_metadata: Dict[str, Any],
        statistics_metadata_id: int
    ) -> str:
        """Get source code from metadata with fallback logic."""
        # Try metadata dict first
        if 'source_code' in stat_metadata and stat_metadata.get('source_code'):
            return stat_metadata['source_code']
        
        # Fallback: query database
        result = self.client.table('statistics_metadata').select('source_id').eq('id', statistics_metadata_id).execute()
        if result.data:
            source_id = result.data[0]['source_id']
            return get_source_code(source_id, client=self.client)
        
        # Default fallback to first registered source
        if self._sources:
            fallback = next(iter(self._sources.keys()))
            logger.warning(
                f"Could not determine source_code for statistic {statistics_metadata_id}, "
                f"defaulting to {fallback}"
            )
            return fallback
        
        raise ValueError(
            f"Could not determine source_code for statistic {statistics_metadata_id} "
            "and no sources are registered"
        )
    
    def _get_ingestion(self, source_code: str):
        """
        Get ingestion handler for a source.
        
        Parameters
        ----------
        source_code : str
            Source code (e.g., 'BOK', 'KOSIS')
        
        Returns
        -------
        BaseIngestion
            Ingestion handler instance
        
        Raises
        ------
        ValueError
            If source is not registered
        """
        if source_code not in self._sources:
            raise ValueError(
                f"Unsupported data source: {source_code}. "
                f"Available sources: {list(self._sources.keys())}"
            )
        source = self._sources[source_code]
        return source['ingestion']
    
    

