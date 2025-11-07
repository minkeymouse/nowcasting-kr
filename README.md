# dfm-python

A generic Python implementation of Dynamic Factor Models (DFM) for nowcasting and forecasting.

## Features

- **Generic DFM Estimation**: EM algorithm for estimating dynamic factor models
- **Mixed-Frequency Data**: Handle data with different frequencies (daily, monthly, quarterly)
- **Missing Data Handling**: Robust handling of missing observations
- **News Decomposition**: Compare forecasts between vintages and attribute changes
- **Flexible Configuration**: Support for YAML and CSV configuration files
- **Optional Hydra Integration**: Use Hydra for experiment management (optional dependency)

## Installation

```bash
pip install dfm-python
```

### Optional Dependencies

For Hydra configuration support:
```bash
pip install dfm-python[hydra]
```

For database integration (application-specific):
```bash
pip install dfm-python[database]
```

Install all optional dependencies:
```bash
pip install dfm-python[all]
```

## Quick Start

### Basic Usage

```python
from dfm_python import load_config, load_data, dfm
import pandas as pd

# Load configuration from YAML or CSV
config = load_config('config.yaml')

# Load data from CSV file
X, Time, Z = load_data('data.csv', config, sample_start=pd.Timestamp('2000-01-01'))

# Estimate DFM
Res = dfm(X, config, threshold=1e-4)

# Access results
factors = Res.Z  # Extracted factors
loadings = Res.C  # Factor loadings
```

### Configuration

The DFM module uses a configuration object (`DFMConfig`) that defines:

- **Series**: Each time series with frequency, transformation, and block membership
- **Blocks**: Factor blocks with number of factors per block
- **Estimation Parameters**: EM algorithm settings (threshold, max_iter, etc.)

#### YAML Configuration Example

```yaml
# config.yaml
model:
  series:
    series_1:
      frequency: "m"
      transformation: "pch"
      blocks: [Global]
    series_2:
      frequency: "q"
      transformation: "pca"
      blocks: [Global, Investment]
  
  blocks:
    Global:
      factors: 1
    Investment:
      factors: 1

dfm:
  ar_lag: 1
  threshold: 1e-5
  max_iter: 5000
```

#### CSV Configuration Example

```csv
SeriesID,Frequency,Transformation,Block_Global,Block_Investment
series_1,m,pch,1,0
series_2,q,pca,1,1
```

### Data Format

CSV data files should have:
- First column: Date (YYYY-MM-DD format)
- Subsequent columns: One per series (matching SeriesID from config)
- Column names: Match SeriesID from configuration

Example:
```csv
Date,series_1,series_2
2000-01-01,100.0,50.0
2000-02-01,101.0,51.0
...
```

## Core Components

### DFMConfig

Configuration dataclass that defines the model structure:

```python
from dfm_python import DFMConfig, SeriesConfig

config = DFMConfig(
    series=[
        SeriesConfig(
            series_id="gdp",
            frequency="q",
            transformation="pca",
            blocks=[1, 0]  # Binary: loads on block 0 (Global)
        )
    ],
    block_names=["Global", "Investment"],
    factors_per_block=[1, 1],
    ar_lag=1,
    threshold=1e-5,
    max_iter=5000
)
```

### DFM Estimation

```python
from dfm_python import dfm

# Estimate DFM model
result = dfm(X, config, threshold=1e-4, max_iter=1000)

# Access results
factors = result.Z          # Factor estimates (T x r)
loadings = result.C         # Factor loadings (N x r)
transition = result.A         # Transition matrix
covariance = result.Q         # Factor covariance
```

### News Decomposition

Compare forecasts between two vintages:

```python
from dfm_python import update_nowcast

# Compare old vs new vintage
news_df, forecast_df = update_nowcast(
    X_old, X_new, Time, config, result,
    series="gdp",
    vintage_old="2024-01-01",
    vintage_new="2024-02-01"
)
```

## API Reference

### Main Functions

- `load_config(file)`: Load configuration from YAML or CSV
- `load_data(file, config)`: Load and transform data from CSV
- `dfm(X, config)`: Estimate DFM model using EM algorithm
- `update_nowcast(...)`: Perform news decomposition between vintages

### Configuration

- `DFMConfig`: Main configuration class
- `SeriesConfig`: Individual series configuration

### Results

- `DFMResult`: Result object with factors, loadings, and model parameters

## Architecture

The DFM module is designed to be **generic and application-agnostic**:

- **Core Module** (`dfm_python`): Generic DFM estimation logic
- **Adapters** (application-specific): Database integration, API clients, etc.

This separation allows the core module to be used in any context while keeping application-specific code separate.

## Requirements

- Python >= 3.12
- numpy >= 1.24.0
- pandas >= 2.0.0
- scipy >= 1.10.0

## License

MIT License

## Contributing

Contributions are welcome! Please ensure that:
- Core module remains generic (no application-specific code)
- All optional dependencies are properly handled
- Tests pass for the core functionality

## Citation

If you use this package in your research, please cite:

```bibtex
@software{dfm-python,
  title = {dfm-python: Dynamic Factor Models for Nowcasting},
  author = {DFM Python Contributors},
  year = {2024},
  url = {https://github.com/yourusername/dfm-python}
}
```
