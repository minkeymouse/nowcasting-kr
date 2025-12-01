# AGENT.md file for the project

## RULES

- NEVER DELETE files.
- DO NOT CREATE files except __init__.py or any occation that you must create files. Try to work on existing files rather than creating many files.
- DO NOT let each file exceed 1000 lines.
- Make sure app/ contains maximum 30 files and src/ contains maximum 15 files.
- When working with dfm-python/ or nowcasting-report, NEVER CREATE OR DELETE any files.

## General Purpose

- This is project for nowcasting production and investment macro economic indicators using dfm(dynamic factor model) and ddfm(deep dynamic factor model) using the package at dfm-python/ directory.

## Project Structure

### app/
- This is directory for service. User will use run.bat to run app/main.py which runs fastapi server with langgraph agent.
- The langgraph agent's job is to provide user with simple report based on the analysis.

### src/
- This is directory for research experiments, and inference.
- For research, we will write the results in nowcasting-report/ directory. Use this as a context while working on this project.
- For experiments, we run src/main.py with hydra-core using configs from config/ directory. 
- This directory should mainly wrap the dfm-python functionalities with the experiment specific utility, being thin wrapper for the library, mainly filling with functionalities for the app.

### config/
- Centralized source of config for both src/ and app/

### dfm-python/
- This is a submodule for the project.
- The package should include metadata as a part of DFMresult class or any other necessary form for easier tracking.

### nowcasting-report/
- This is the report for this project.

### outputs/
- The directory should be structured accordingly with model registry. Here's example:
  outputs/
    {model_name}/
      model.pkl (or similar)
      config.yaml
      logs/
      plots/
      results/

## Functionality

### Agent's role
- Langgraph agent is supposed to analyze the result and write simple report based on the results.
- Result will be generated and consolidated for agent to use from deterministc node before model call.
- It is just simple research agent given the model results.
- Agent is deterministic when it comes to inference. Agent only has tool for research and report generation.
- Deterministic node should consolidate training metrics, model results, statistics, and anything that came out. We will specify this later when things get more stable.

### APP's access on DFM functionality
- App provides two key functionalities: Agent for report generation, tool for training the model.
- For training, user will be able to (1) upload their own data from csv file (2) Setup the configuration via UI (3) Run training for both dfm and ddfm (4) monitor the progress and results.
- For agent, user will be able to get the final pdf file from the results with tavily research aided sprinkle.
- This means app should access the package with import, train models on demand, using pretrained models when user chooose to. User should be able to see different results with different setup and configs.
- This is basically train, monitoring, analysis app.

### Report generation
- App's report from app should be generated via pandoc, converting markdown result to pdf file.
- This should use the custom.docx template by default
- Report should contain model's trained metrics, backtesting results, nowcast results, forecasting results, necessary plots, web search resutls, etc.
- Report generation is not urgent task for now. This will be updated when other tasks are stable.

### UI and Monitoring
- We seperate the tab into two. One for uploading, training, monitoring the data, another for inference and agent report generation.
- This means training is sync jobs with user's click. All the results should be organized(including logs) and stored solely in outputs/ directory.
- Frontend framework should be react, training progress should be streamed via websocket. This is local only(no internet for app) UI.
- User should be able to name the model. By default, use timestamp. There should be model registry/metadata file for stability.
- We do not run async for training so user should wait for each process to finish. This means we need some screen for user waiting.

## Config management
- UI should generate the YAML configs in config/ for hydra runs.
- UI should provide easier edit interface for user to change the config.
- We only use the YAML editing. Spec upload will not be supported.
- We use same config structure as dfm-python but need user-friendly interface for this. User can either directly modify the config in the source or use the UI to modify the config.

### Config UI
- User should be able to choose experiments. When user choose the experiment, there's sub tab where user can specify the model.
- For the case where experiment_id - model: dfm, user can specify dfm config, series config, block config. We will have general dfm config at the top and at the bottom, user can specify the series config and block config. For details, refer to dfm-python package's example configs.
- For the case where experiment_id - model: ddfm, user can specify ddfm config(like deep learning config learning rate, etc which is not specified yet since ddfm implementation is not stable and ready). For ddfm, we don't have block structure yet so only have series subconfig.

### Model registry
- Model registry should be using the JSON.
- Fields should be model_name, timestamp, config_path, model_type

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

## Current priority
- dfm-python stability and UI construction with fast api.