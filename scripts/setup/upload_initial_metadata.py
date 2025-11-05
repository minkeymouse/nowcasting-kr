"""Upload initial statistics metadata and items to database.

This is a one-time setup script for uploading collected metadata to the database.
Run this after collecting statistics list using collect_initial_statistics.py.
"""

import sys
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

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

from services.ingestion.bok import BOKIngestion
from services.ingestion.kosis import KOSISIngestion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for initial metadata upload."""
    parser = argparse.ArgumentParser(
        description='Upload initial statistics metadata to database',
        epilog='This is a one-time setup script. For regular updates, use services/ingestion/orchestrator.py'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='BOK',
        help='Data source code (BOK, KOSIS, etc.)'
    )
    parser.add_argument(
        '--dfm-csv',
        type=str,
        default='data/bok_statistics_dfm_selected.csv',
        help='Path to CSV file with DFM-selected statistics'
    )
    parser.add_argument(
        '--update-dfm',
        action='store_true',
        help='Update DFM selection from CSV'
    )
    parser.add_argument(
        '--collect-items',
        action='store_true',
        default=True,
        help='Collect items for each statistic (default: True)'
    )
    parser.add_argument(
        '--statistics-only',
        action='store_true',
        help='Only update statistics list, skip items collection'
    )
    
    args = parser.parse_args()
    
    # Get ingestion handler for source
    if args.source == 'BOK':
        collector = BOKIngestion()
    elif args.source == 'KOSIS':
        collector = KOSISIngestion()
    else:
        logger.error(f"Unsupported data source: {args.source}")
        sys.exit(1)
    
    try:
        logger.info("=" * 80)
        logger.info("Initial Metadata Upload to Database")
        logger.info("=" * 80)
        
        if args.statistics_only:
            logger.info("Updating statistics list only")
            result = collector.collect_statistics_list(
                update_dfm_selection=args.update_dfm,
                dfm_selection_csv=args.dfm_csv if args.update_dfm else None
            )
        else:
            logger.info("Running full metadata update workflow")
            result = collector.update_all_metadata(
                update_dfm_selection=args.update_dfm,
                dfm_selection_csv=args.dfm_csv if args.update_dfm else None,
                collect_items=args.collect_items
            )
        
        logger.info("=" * 80)
        logger.info("Upload Summary")
        logger.info("=" * 80)
        
        if args.statistics_only:
            logger.info(f"Total statistics: {result.get('total', 0)}")
            logger.info(f"Inserted: {result.get('inserted', 0)}")
            logger.info(f"Updated: {result.get('updated', 0)}")
            if result.get('errors'):
                logger.warning(f"Errors: {len(result['errors'])}")
        else:
            stats_list = result.get('statistics_list', {})
            items_collection = result.get('items_collection', {})
            
            logger.info("Statistics List:")
            logger.info(f"  Total: {stats_list.get('total', 0)}")
            logger.info(f"  Inserted: {stats_list.get('inserted', 0)}")
            logger.info(f"  Updated: {stats_list.get('updated', 0)}")
            
            logger.info("Items Collection:")
            logger.info(f"  Statistics processed: {items_collection.get('processed', 0)}/{items_collection.get('total_statistics', 0)}")
            logger.info(f"  Total items: {items_collection.get('total_items', 0)}")
            logger.info(f"  Failed: {items_collection.get('failed', 0)}")
        
        logger.info("=" * 80)
        logger.info("\nInitial setup complete! Use scripts/ingest_data.py for regular data ingestion.")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

