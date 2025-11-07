"""Main CLI entry point for Nowcasting KR application."""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Nowcasting KR - Korean macroeconomic nowcasting system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  ingest      Ingest data from BOK and KOSIS APIs
  train       Train DFM model
  nowcast     Generate nowcasts and forecasts
  validate    Validate spec file and API endpoints
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest data from APIs')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train DFM model')
    train_parser.add_argument('--config-name', default='default', help='Hydra config name')
    train_parser.add_argument('--series', default='default', help='Series config name')
    train_parser.add_argument('--max-iter', type=int, help='Maximum iterations')
    
    # Nowcast command
    nowcast_parser = subparsers.add_parser('nowcast', help='Generate nowcasts')
    nowcast_parser.add_argument('--config-name', default='default', help='Hydra config name')
    nowcast_parser.add_argument('--series', default='default', help='Series config name')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate spec file')
    validate_parser.add_argument('--spec-file', help='Path to spec CSV file')
    validate_parser.add_argument('--use-db', action='store_true', help='Load spec from database')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'ingest':
        from app.jobs.ingest import main
        main()
    elif args.command == 'train':
        import hydra
        from omegaconf import DictConfig
        from app.jobs.train import main as train_main
        
        overrides = []
        if args.max_iter:
            overrides.append(f'dfm.max_iter={args.max_iter}')
        
        with hydra.initialize(config_path="../../app/config", version_base=None):
            cfg = hydra.compose(
                config_name=args.config_name,
                overrides=[f'series={args.series}'] + overrides
            )
            train_main(cfg)
    elif args.command == 'nowcast':
        import hydra
        from omegaconf import DictConfig
        from app.jobs.nowcast import main as nowcast_main
        
        with hydra.initialize(config_path="../../app/config", version_base=None):
            cfg = hydra.compose(
                config_name=args.config_name,
                overrides=[f'series={args.series}']
            )
            nowcast_main(cfg)
    elif args.command == 'validate':
        from app.cli.validate_spec import main as validate_main
        validate_main()


if __name__ == '__main__':
    main()

