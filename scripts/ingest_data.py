"""Main entry point for data ingestion (GitHub Actions)."""

import sys
import logging
from datetime import date
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.ingestion.orchestrator import DataIngestionOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for data ingestion."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run data ingestion')
    parser.add_argument(
        '--update-metadata',
        action='store_true',
        help='Update metadata before data collection'
    )
    
    args = parser.parse_args()
    
    try:
        orchestrator = DataIngestionOrchestrator()
        result = orchestrator.run(update_metadata=args.update_metadata)
        
        logger.info("=" * 80)
        logger.info("Ingestion Summary")
        logger.info("=" * 80)
        logger.info(f"Vintage ID: {result['vintage_id']}")
        logger.info(f"Job ID: {result['job_id']}")
        logger.info(f"Total: {result['stats']['total']}")
        logger.info(f"Successful: {result['stats']['successful']}")
        logger.info(f"Failed: {result['stats']['failed']}")
        logger.info(f"Skipped: {result['stats']['skipped']}")
        
        if result['stats']['errors']:
            logger.warning("Errors encountered:")
            for error in result['stats']['errors']:
                logger.warning(f"  - {error}")
        
        # Exit with error code if failures
        if result['stats']['failed'] > 0:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error in ingestion: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

