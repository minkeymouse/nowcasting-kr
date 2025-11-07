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
pip install -r requirements.txt

# Install with database support
pip install -r requirements.txt -r requirements-database.txt

# Install with Hydra configuration support
pip install -r requirements.txt -r requirements-hydra.txt

# Install with all optional dependencies
pip install -r requirements.txt -r requirements-database.txt -r requirements-hydra.txt
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
├── app/               # Main application package
│   ├── adapters/      # Database adapters for DFM module
│   ├── cli/           # CLI wrapper and validation tools
│   ├── config/        # Hydra configuration files
│   ├── database/      # Database operations and models
│   ├── jobs/          # GitHub Actions jobs (main entry points)
│   │   ├── ingest.py  # Job 1: Ingest data from APIs
│   │   ├── train.py   # Job 2: Train DFM model
│   │   └── nowcast.py # Job 3: Generate nowcasts/forecasts
│   ├── services/      # API clients and ingestion services
│   └── utils/         # Shared utility functions
├── .github/workflows/ # GitHub Actions workflows
│   ├── ingest.yaml    # Ingest job workflow
│   ├── train.yaml     # Train job workflow
│   ├── nowcast.yaml   # Nowcast job workflow
│   └── pipeline.yaml  # Full pipeline (all three jobs)
├── tests/             # Test suite
└── migrations/        # Database migrations
```

## Usage

### Job 1: Ingest Data

```bash
# Local execution
python -m app.jobs.ingest

# GitHub Actions
# Triggered automatically on push or manually via workflow_dispatch
```

**What it does:**
- Loads spec CSV from database storage bucket (or local fallback)
- Fetches data from BOK and KOSIS APIs
- Creates/updates data vintages in database
- Saves observations and series metadata

### Job 2: Train DFM Model

```bash
# Local execution
python -m app.jobs.train --config-name=test series=test_series

# GitHub Actions
# Triggered automatically on push or manually via workflow_dispatch
```

**What it does:**
- Queries database for latest data vintage
- Loads model configuration (from DB storage CSV or Hydra YAML)
- Trains DFM model using EM algorithm
- Saves model weights to Supabase storage bucket
- Saves factors to database

### Job 3: Generate Nowcasts/Forecasts

```bash
# Local execution
python -m app.jobs.nowcast --config-name=test series=test_series

# GitHub Actions
# Triggered automatically on push or manually via workflow_dispatch
```

**What it does:**
- Queries database for latest data vintage
- Downloads latest model weights from Supabase storage bucket
- Loads model configuration (from DB storage CSV or Hydra YAML)
- Generates nowcasts (current period) and forecasts (future periods)
- Saves nowcasts and forecasts to database with news decomposition

### Full Pipeline

The `pipeline.yaml` workflow runs all three jobs sequentially:
1. Ingest → 2. Train → 3. Nowcast

Triggered automatically on push to `main`/`master` branches.

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
