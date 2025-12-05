# AGENT.md file for the project

## RULES

- NEVER DELETE files.
- DO NOT CREATE files except __init__.py or any occasion that you must create files. Try to work on existing files rather than creating many files.
- DO NOT let each file exceed 1000 lines.
- Make sure app/ contains maximum 35 files and src/ contains maximum 20 files. This is a strict limit that must be followed.
  - **File counting rules**: `__pycache__/` directories and `__init__.py` files do not count. Subdirectories are counted together with their parent. Test files count toward the limit.
- When working with dfm-python/ or nowcasting-report, NEVER CREATE OR DELETE any files.

## General Purpose

- This is project for nowcasting production and investment macro economic indicators using dfm(dynamic factor model) and ddfm(deep dynamic factor model) using the package at dfm-python/ directory.

## Project Structure

### dfm-python/
- This submodule is mainly used as a package that we will be using to train, predict, and nowcast the data in this project.
- It is dfm and ddfm implementation with pytorch lightning for better user experience.

### app/
- This is directory for service. User will use run.bat (to be implemented) or run script to run app/main.py which runs fastapi server with langgraph agent.
- The langgraph agent's job is to provide user with simple report based on the analysis.

### src/
- This is directory for research experiments, and inference.
- For research, we will write the results in nowcasting-report/ directory. Use this as a context while working on this project.
- For experiments, we run src/main.py with hydra-core using configs from config/ directory. 
- This directory should mainly wrap the dfm-python functionalities with the experiment specific utility, being thin wrapper for the library, mainly filling with functionalities for the app.
- **App must access dfm-python through src/ wrappers, not directly importing dfm-python.**
- **App can also call src/main.py for Hydra-based experiments if needed.**

**Current structure:**
```
src/
├── main.py              # Hydra-based experiment runner
├── model/               # Model wrappers (app uses these)
│   ├── __init__.py     # Exports DFM, DDFM
│   ├── dfm.py          # DFM wrapper with metadata & save_to_outputs()
│   └── ddfm.py         # DDFM wrapper with metadata & save_to_outputs()
└── utils/               # Utilities (to be added as needed)
    └── __init__.py
```

**Key features:**
- `model/dfm.py` and `model/ddfm.py` provide wrappers around dfm-python with:
  - Metadata tracking (creation time, training metrics, etc.)
  - `save_to_outputs()` method for saving to outputs/{model_name}/ structure
  - All original dfm-python functionality
- Additional utilities (evaluation, visualization, etc.) will be added to `src/utils/` as needed during development.
- **Note**: Current src/ structure needs refactoring to match AGENT.md specification. File count limit: 20 files (excluding __pycache__/ and __init__.py, subdirectories counted together, test files count)

### config/
- Centralized source of config for both src/ and app/

### dfm-python/
- This is a submodule for the project.
- The package should include metadata as a part of DFMresult class or any other necessary form for easier tracking.

### nowcasting-report/
- This is the report for this project.

### outputs/
- The directory should be structured accordingly with model registry. Here's example:
  ```
  outputs/
    {model_name}/
      model.pkl          # Pickled model, result, config, and time index
      config.yaml        # Copy of config used for training
      logs/              # Training progress, errors, metrics (to be implemented)
      plots/             # Visualization results (to be determined during app implementation)
      results/           # Model results (to be determined during app implementation)
  ```
- Structure is created by `model.save_to_outputs()` method in src/model/dfm.py and src/model/ddfm.py
- All subdirectories (logs/, plots/, results/) are created automatically

## Functionality

### Agent's role
- Langgraph agent uses the simple `create_agent` pattern from LangChain for report generation.
- Agent analyzes model results and writes simple reports based on the analysis.
- Result consolidation happens in a deterministic node (`consolidate_results`) before agent processing.
- The deterministic node consolidates training metrics, model results, statistics, and any other outputs for agent analysis.
- Agent is deterministic when it comes to inference. Agent only has tools for research (Tavily - planned) and report generation.
- Implementation should use `create_agent` from `langchain.agents` for simplicity.

### APP's access on DFM functionality
- App provides two key functionalities: Agent for report generation, tool for training the model.
- For training, user will be able to (1) upload their own data from csv file (2) Setup the configuration via UI (3) Run training for both dfm and ddfm (4) monitor the progress and results.
- For agent, user will be able to get the final pdf file from the results with tavily research aided sprinkle.
- **App must access dfm-python through src/ wrappers (src/model/dfm.py, src/model/ddfm.py), not directly importing dfm-python.**
  - Import example: `from model.dfm import DFM` or `from model.ddfm import DDFM`
  - These wrappers provide metadata tracking and `save_to_outputs()` method
- App should train models on demand, using pretrained models when user choose to. User should be able to see different results with different setup and configs.
- This is basically train, monitoring, analysis app.

### Report generation
- App's report from app should be generated via pandoc, converting markdown result to pdf file.
- This should use the custom.docx template by default
- Report should contain model's trained metrics, backtesting results, nowcast results, forecasting results, necessary plots, web search results (Tavily - planned), etc.
- **Current Priority**: Report generation is not a priority. Current focus is:
  1. Finalize dfm-python package (especially DDFM stability - this is a main goal)
  2. Fully functional src/ directory (needs refactoring)
  3. Partially functional app/ for training and monitoring progress
- Report generation will be updated when other tasks are stable.

### UI and Monitoring
- We separate the tab into two. One for uploading, training, monitoring the data, another for inference and agent report generation.
- **Training execution**: Training runs synchronously (user must wait for completion), but progress is streamed asynchronously via WebSocket. User should see a waiting screen during training.
- All results should be organized (including logs) and stored solely in outputs/ directory.
- **Logging**: During training, logs should be saved to outputs/{model_name}/logs/ including: training progress, errors, and metrics. Use PyTorch Lightning's built-in logging features from dfm-python package. Update dfm-python package to leverage Lightning loggers if needed.
- Frontend framework should be react, training progress should be streamed via websocket. This is local only (no internet for app) UI.
- User should be able to name the model. By default, use timestamp. There should be model registry/metadata file for stability.

## Config management
- UI should generate the YAML configs in config/ for hydra runs.
- UI should provide easier edit interface for user to change the config.
- We only use the YAML editing. Spec upload will not be supported.
- We use same config structure as dfm-python but need user-friendly interface for this. User can either directly modify the config in the source or use the UI to modify the config.

### Config UI
- Use a simple UI for configuration management.
- User should be able to choose experiments. When user chooses the experiment, there's sub tab where user can specify the model.
- For the case where experiment_id - model: dfm, user can specify dfm config, series config, block config. We will have general dfm config at the top and at the bottom, user can specify the series config and block config. For details, refer to dfm-python package's example configs.
- For the case where experiment_id - model: ddfm, user can specify ddfm config (like deep learning config learning rate, etc). **Note**: DDFM implementation is not yet stable - stabilizing DDFM is one of the main goals. For ddfm, we don't have block structure yet so only have series subconfig.

### Model registry
- Model registry should be using JSON format.
- **Required fields**: model_name, timestamp, config_path, model_type
- **Additional fields should include**: training metrics (converged, num_iter, loglik), model status, file paths (model.pkl path, config.yaml path), training duration, and any other relevant metadata for tracking and monitoring.

### API endpoint structure
- fastapi's endpoint should be as follows:
POST /api/train - Start training
GET /api/train/status/{job_id} - Training status
WebSocket /ws/train/{job_id} - Training progress stream
GET /api/models - List trained models
POST /api/inference - Run inference
POST /api/agent/report - Generate report
GET /api/config/{config_name} - Get config
PUT /api/config/{config_name} - Update config
- Port should always be 2020. Make sure when testing, DO NOT create new port forwardings

## Error Handling
- Error handling for training failures, report generation failures, invalid configs or data should all be handled properly.

## Data Transformations

The project supports various data transformations for time series preprocessing. These transformations are applied to each series based on the configuration in the spec file.

- **`lin`** - Levels (No Transformation)
  - No transformation applied, uses raw data
  - Formula: `X(:,i) = Z(:,i)`

- **`chg`** - Change (Difference)
  - First difference transformation
  - Formula: `X(t1:step:T,i) = [NaN; Z(t1+step:step:T,i) - Z(t1:step:T-t1,i)]`

- **`ch1`** - Year over Year Change (Difference)
  - Year-over-year difference (12 periods for monthly data)
  - Formula: `X(12+t1:step:T,i) = Z(12+t1:step:T,i) - Z(t1:step:T-12,i)`

- **`cha`** - Change (Annual Rate)
  - Annualized change rate
  - Formula: `X(t1:step:T,i) = 100*[NaN; (Z(t1+step:step:T,i) ./ Z(t1:step:T-step,i)).^(1/n) - 1]`

- **`pch`** - Percent Change
  - Percentage change from previous period
  - Formula: `X(t1:step:T,i) = 100*[NaN; Z(t1+step:step:T,i) ./ Z(t1:step:T-t1,i) - 1]`

- **`pc1`** - Year over Year Percent Change
  - Year-over-year percentage change
  - Formula: `X(12+t1:step:T,i) = 100*(Z(12+t1:step:T,i) ./ Z(t1:step:T-12,i) - 1)`

- **`pca`** - Percent Change (Annual Rate)
  - Annualized percentage change rate
  - Formula: `X(t1:step:T,i) = 100*[NaN; (Z(t1+step:step:T,i) ./ Z(t1:step:T-step,i)).^(1/n) - 1]`

- **`log`** - Natural Log
  - Natural logarithm transformation
  - Formula: `X(:,i) = log(Z(:,i))`
