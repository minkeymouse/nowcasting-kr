# Nowcasting KR

Korean macroeconomic nowcasting system using Dynamic Factor Models (DFM).

This project provides a complete pipeline for:
- Data ingestion from BOK (Bank of Korea) and KOSIS (Korean Statistical Information Service) APIs
- DFM model training and estimation
- Nowcasting and forecasting of Korean macroeconomic indicators

## Installation

```bash
# Install dependencies
pip install uv
uv venv .venv
source .venv/bin/activate

# Install the application
uv pip install -e .[database,hydra]

# Or install all optional dependencies
uv pip install -e .[all]
```

## Dependencies

- **dfm-python**: Core DFM module (installed from PyPI as external dependency)
- **supabase**: Database backend
- **hydra-core**: Configuration management
- **pandas, numpy, scipy**: Data processing and modeling

Note: The DFM module is now a separate PyPI package (`dfm-python`). All DFM-related functionality is imported from `dfm_python`, not from local code.

## Project Structure

```
nowcasting-kr/
├── adapters/          # Database adapters for DFM module
├── config/            # Hydra configuration files
├── database/          # Database operations and models
├── scripts/           # Main application scripts
│   ├── ingest_api.py      # Data ingestion from APIs
│   ├── train_dfm.py       # DFM model training
│   └── nowcast_dfm.py     # Nowcasting and forecasting
└── services/          # API clients and ingestion services
```

## Usage

### Data Ingestion

```bash
python scripts/ingest_api.py
```

Ingests data from BOK and KOSIS APIs and stores it in the database.

### Model Training

```bash
python scripts/train_dfm.py --config-name=test series=test_series
```

Trains a DFM model using the specified configuration.

### Nowcasting

```bash
python scripts/nowcast_dfm.py --config-name=test series=test_series
```

Generates nowcasts and forecasts using the trained model.

## Configuration

Configuration is managed via Hydra. See `config/` directory for:
- `default.yaml`: Default configuration
- `test.yaml`: Test configuration (fast runs)
- `series/`: Series-specific configurations

## Database

The system uses Supabase (PostgreSQL) for:
- Storing time series data and observations
- Managing data vintages
- Storing model weights and configurations
- Storing forecasts and nowcasts

## GitHub Actions

The project includes automated workflows:
- `pipeline.yaml`: Full pipeline (ingest → train → nowcast)
- `database.yaml`: Data ingestion workflow
- `train.yaml`: Model training workflow
- `nowcast.yaml`: Nowcasting workflow

## License

MIT
