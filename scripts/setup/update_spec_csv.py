"""Update CSV specification file from database.

This script syncs the CSV specification file (migrations/001_initial_spec.csv)
with the latest model configuration from the database.

Usage:
    python scripts/setup/update_spec_csv.py
    python scripts/setup/update_spec_csv.py --config-name 001-initial-spec
    python scripts/setup/update_spec_csv.py --config-name 001-initial-spec --output custom_spec.csv
"""

import sys
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
env_paths = [
    project_root / '.env.local',
    project_root.parent / '.env.local',
    Path.home() / 'Nowcasting' / '.env.local',
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        break

from database import sync_csv_from_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Update CSV spec file from database."""
    parser = argparse.ArgumentParser(
        description='Update CSV specification file from database configuration'
    )
    parser.add_argument(
        '--config-name',
        type=str,
        default='001-initial-spec',
        help='Name of the model configuration in database (default: 001-initial-spec)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file path (default: migrations/001_initial_spec.csv)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be updated without writing to file'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Update CSV Specification from Database")
    logger.info("=" * 80)
    logger.info(f"Config name: {args.config_name}")
    logger.info("=" * 80)
    logger.info()
    
    try:
        if args.dry_run:
            # Just check if config exists and show summary
            from database import load_model_config, get_model_config_series_ids
            
            config = load_model_config(args.config_name)
            if not config:
                logger.error(f"Configuration '{args.config_name}' not found in database")
                sys.exit(1)
            
            series_ids = get_model_config_series_ids(config['config_id'])
            logger.info(f"Would export {len(series_ids)} series")
            logger.info(f"Block names: {config.get('block_names', [])}")
            logger.info("Dry run complete - no file written")
        else:
            # Actually sync the CSV
            output_path = Path(args.output) if args.output else None
            result_path = sync_csv_from_db(
                config_name=args.config_name,
                csv_path=output_path
            )
            
            logger.info()
            logger.info("=" * 80)
            logger.info("CSV Specification Updated Successfully")
            logger.info("=" * 80)
            logger.info(f"Output file: {result_path}")
            logger.info(f"Config: {args.config_name}")
            logger.info("=" * 80)
            
    except Exception as e:
        logger.error(f"Error updating CSV: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

