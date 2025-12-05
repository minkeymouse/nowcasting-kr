# sktime Documentation for Nowcasting Project

## Overview

sktime is a unified framework for machine learning with time series, compatible with scikit-learn. It provides:
- **Unified interface**: Same API for all time series learning tasks (forecasting, classification, regression, clustering)
- **Composability**: Combine transformers, forecasters, and other components
- **Compatibility**: Works with scikit-learn, pandas, numpy
- **Extensibility**: Easy to create custom forecasters and transformers

## Core Concepts

### Forecasting Workflow

The basic sktime forecasting workflow follows this pattern:

```python
from sktime.forecasting.base import ForecastingHorizon
from sktime.split import temporal_train_test_split

# 1. Split data temporally
y_train, y_test = temporal_train_test_split(y, test_size=0.2)

# 2. Create and fit forecaster
forecaster = SomeForecaster()
forecaster.fit(y_train, fh=[1, 2, 3, ...])  # fh = forecasting horizon

# 3. Predict
y_pred = forecaster.predict(fh=[1, 2, 3, ...])
```

### Data Container Formats

sktime supports multiple data container formats:

1. **pd.Series**: Univariate time series
   - Index: time index (datetime or integer)
   - Values: time series values

2. **pd.DataFrame**: Multivariate time series
   - Index: time index
   - Columns: different time series (variables)

3. **numpy.ndarray**: Array format
   - Shape: (n_timepoints, n_variables) for multivariate

4. **numpy3D**: Panel data (multiple time series)
   - Shape: (n_instances, n_timepoints, n_variables)

### Forecasting Horizon (fh)

The forecasting horizon specifies which time points to predict:

```python
from sktime.forecasting.base import ForecastingHorizon

# Absolute: specific time points
fh = ForecastingHorizon([1, 2, 3, 4, 5])  # Next 5 steps

# Relative: steps ahead
fh = ForecastingHorizon([1, 2, 3])  # 1, 2, 3 steps ahead

# Using numpy
import numpy as np
fh = np.arange(1, 13)  # Next 12 steps
```

## Basic Deployment Workflow

### 1. Fit-Predict Pattern

```python
# Fit on training data
forecaster.fit(y_train, fh=[1, 2, 3])

# Predict future values
y_pred = forecaster.predict(fh=[1, 2, 3])
```

### 2. Update-Predict Pattern

```python
# Initial fit
forecaster.fit(y_train, fh=[1, 2, 3])

# Update with new data (incremental learning)
forecaster.update(y_new)

# Predict with updated model
y_pred = forecaster.predict(fh=[1, 2, 3])
```

### 3. Fit-Predict-Update Pattern

```python
# Fit
forecaster.fit(y_train, fh=[1, 2, 3])

# Predict
y_pred = forecaster.predict(fh=[1, 2, 3])

# Update with actuals
forecaster.update(y_actual)

# Predict again
y_pred_new = forecaster.predict(fh=[1, 2, 3])
```

## Evaluation Workflow

### Temporal Cross-Validation

```python
from sktime.forecasting.model_evaluation import evaluate
from sktime.split import ExpandingWindowSplitter, SlidingWindowSplitter

# Expanding window: training set grows over time
cv = ExpandingWindowSplitter(
    fh=[1, 2, 3],
    initial_window=100,  # Initial training size
    step_length=12        # Step size between windows
)

# Sliding window: fixed training size
cv = SlidingWindowSplitter(
    fh=[1, 2, 3],
    window_length=100,   # Fixed training size
    step_length=12
)

# Evaluate
results = evaluate(
    forecaster=forecaster,
    y=y,
    cv=cv,
    scoring=["mse", "mae", "mape"]
)
```

### Single Train-Test Split

```python
from sktime.split import temporal_train_test_split
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanSquaredError,
    MeanAbsolutePercentageError
)

# Split
y_train, y_test = temporal_train_test_split(y, test_size=0.2)

# Fit
forecaster.fit(y_train, fh=[1, 2, 3])

# Predict
y_pred = forecaster.predict(fh=[1, 2, 3])

# Evaluate
mae = MeanAbsoluteError()
mae_score = mae(y_test, y_pred)
```

## Advanced Workflows

### 1. Exogenous Variables (X)

```python
# Fit with exogenous variables
forecaster.fit(y_train, X=X_train, fh=[1, 2, 3])

# Predict with future exogenous variables
y_pred = forecaster.predict(fh=[1, 2, 3], X=X_future)
```

### 2. Prediction Intervals

```python
# Predict with intervals
y_pred, y_pred_interval = forecaster.predict_interval(
    fh=[1, 2, 3],
    coverage=0.9  # 90% prediction interval
)
```

### 3. Quantile Forecasts

```python
# Predict quantiles
y_pred_quantiles = forecaster.predict_quantiles(
    fh=[1, 2, 3],
    alpha=[0.05, 0.5, 0.95]  # 5th, 50th, 95th percentiles
)
```

### 4. Variance Forecasts

```python
# Predict variance
y_pred_var = forecaster.predict_var(fh=[1, 2, 3])
```

## Forecaster Types and Tags

### Forecaster Tags

Forecasters have tags that describe their capabilities:

- `requires-fh-in-fit`: Whether forecasting horizon is required during fit
- `handles-missing-data`: Whether the forecaster can handle missing values
- `y_inner_mtype`: Internal data type for y (e.g., "pd.Series", "pd.DataFrame")
- `X_inner_mtype`: Internal data type for X (exogenous variables)
- `scitype:y`: Type of y ("univariate", "multivariate", "both")
- `capability:pred_int`: Whether prediction intervals are supported
- `capability:pred_var`: Whether variance prediction is supported

### Common Forecaster Types

1. **Direct Forecasters**: Predict all horizons directly
   - Example: `DirectTabularRegressionForecaster`

2. **Recursive Forecasters**: Predict one step ahead, then use predictions for next steps
   - Example: `ARIMA`, `ExponentialSmoothing`

3. **Reduced Forecasters**: Reduce forecasting to regression problem
   - Example: `ReducedRegressionForecaster`

## Composition Patterns

### 1. Transformer + Forecaster Pipeline

```python
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.transformations.series.detrend import Detrender
from sktime.forecasting.trend import PolynomialTrendForecaster

# Create pipeline: detrend → forecast
forecaster = TransformedTargetForecaster([
    ("detrend", Detrender(forecaster=PolynomialTrendForecaster(degree=1))),
    ("forecast", SomeForecaster())
])

forecaster.fit(y_train, fh=[1, 2, 3])
y_pred = forecaster.predict(fh=[1, 2, 3])
```

### 2. Ensemble Forecasters

```python
from sktime.forecasting.compose import EnsembleForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster

# Combine multiple forecasters
ensemble = EnsembleForecaster([
    ("naive", NaiveForecaster()),
    ("trend", PolynomialTrendForecaster(degree=2))
])

ensemble.fit(y_train, fh=[1, 2, 3])
y_pred = ensemble.predict(fh=[1, 2, 3])
```

### 3. Multiplexer (Model Selection)

```python
from sktime.forecasting.compose import MultiplexForecaster
from sktime.forecasting.model_selection import ForecastingGridSearchCV

# Multiple forecasters with parameter tuning
multiplexer = MultiplexForecaster(
    forecasters=[
        ("naive", NaiveForecaster()),
        ("trend", PolynomialTrendForecaster())
    ]
)

# Grid search for best forecaster
gscv = ForecastingGridSearchCV(
    forecaster=multiplexer,
    param_grid={"selected_forecaster": ["naive", "trend"]},
    cv=ExpandingWindowSplitter(fh=[1, 2, 3])
)

gscv.fit(y_train)
y_pred = gscv.predict(fh=[1, 2, 3])
```

## Integration with Nowcasting Project

### DFMForecaster and DDFMForecaster

The project implements sktime-compatible forecasters in `src/model/sktime_forecaster.py`:

1. **DFMForecaster**: Wraps DFM model for sktime API
   - Inherits from `BaseForecaster`
   - Implements `_fit()` and `_predict()` methods
   - Supports multivariate time series forecasting

2. **DDFMForecaster**: Wraps DDFM model for sktime API
   - Similar interface to DFMForecaster
   - Supports deep learning-based forecasting

### Usage Example

```python
from src.model.sktime_forecaster import DFMForecaster
from sktime.split import ExpandingWindowSplitter
from sktime.forecasting.model_evaluation import evaluate
import numpy as np

# Create forecaster
forecaster = DFMForecaster(
    config_path="config/experiment/my_experiment.yaml",
    max_iter=5000,
    threshold=1e-5
)

# Define forecasting horizon
fh = np.arange(1, 13)  # Next 12 steps

# Cross-validation
cv = ExpandingWindowSplitter(
    fh=fh,
    initial_window=100,
    step_length=12
)

# Evaluate
results = evaluate(
    forecaster=forecaster,
    y=y_data,
    cv=cv,
    scoring=["mse", "mae"]
)
```

### Training with sktime Integration

The `src/training.py` module provides `run_training_with_sktime()` function:

```python
from src.training import run_training_with_sktime
from sktime.split import ExpandingWindowSplitter
import numpy as np

# Define forecasting horizon
fh = np.arange(1, 13)

# Create splitter
cv = ExpandingWindowSplitter(
    fh=fh,
    initial_window=100,
    step_length=12
)

# Run training with cross-validation
results = run_training_with_sktime(
    experiment_id="my_experiment",
    data_path="data/sample_data.csv",
    fh=fh,
    cv_splitter=cv
)
```

## Key Methods and Attributes

### BaseForecaster Interface

All sktime forecasters implement:

- `fit(y, X=None, fh=None)`: Fit the forecaster
- `predict(fh=None, X=None)`: Generate predictions
- `update(y, X=None, update_params=True)`: Update with new data
- `predict_interval(fh=None, X=None, coverage=0.9)`: Prediction intervals
- `predict_quantiles(fh=None, X=None, alpha=None)`: Quantile predictions
- `predict_var(fh=None, X=None)`: Variance predictions
- `get_fitted_params()`: Get fitted parameters

### Tags

Access forecaster capabilities:

```python
# Check tags
forecaster.get_tags()

# Check specific tag
forecaster.get_tag("handles-missing-data")
forecaster.get_tag("capability:pred_int")
```

## Best Practices

1. **Always use temporal splits**: Use `temporal_train_test_split` or temporal cross-validation splitters
2. **Handle missing data**: Check `handles-missing-data` tag before using with missing values
3. **Specify forecasting horizon**: Always provide `fh` parameter explicitly
4. **Use appropriate data types**: Ensure data matches `y_inner_mtype` and `X_inner_mtype` tags
5. **Compose transformers**: Use `TransformedTargetForecaster` for preprocessing
6. **Evaluate properly**: Use temporal cross-validation for time series evaluation

## Common Patterns

### Pattern 1: Simple Forecasting

```python
forecaster = SomeForecaster()
forecaster.fit(y_train, fh=[1, 2, 3])
y_pred = forecaster.predict()
```

### Pattern 2: With Preprocessing

```python
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.transformations.series.detrend import Detrender

forecaster = TransformedTargetForecaster([
    ("detrend", Detrender()),
    ("forecast", SomeForecaster())
])
forecaster.fit(y_train, fh=[1, 2, 3])
y_pred = forecaster.predict()
```

### Pattern 3: With Cross-Validation

```python
from sktime.forecasting.model_evaluation import evaluate
from sktime.split import ExpandingWindowSplitter

cv = ExpandingWindowSplitter(fh=[1, 2, 3], initial_window=100)
results = evaluate(forecaster, y, cv=cv, scoring=["mse", "mae"])
```

### Pattern 4: Model Selection

```python
from sktime.forecasting.model_selection import ForecastingGridSearchCV

gscv = ForecastingGridSearchCV(
    forecaster=SomeForecaster(),
    param_grid={"param": [value1, value2]},
    cv=ExpandingWindowSplitter(fh=[1, 2, 3])
)
gscv.fit(y_train)
y_pred = gscv.predict()
```

## References

- sktime Documentation: https://www.sktime.net/
- Forecasting Tutorial: https://www.sktime.net/en/stable/examples/01_forecasting.html
- API Reference: https://www.sktime.net/en/stable/api_reference/forecasting.html
- Project Integration: `src/model/sktime_forecaster.py`, `src/training.py`

