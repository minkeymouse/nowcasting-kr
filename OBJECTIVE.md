# GOAL
- Complete the paper in nowcasting-report using src/ and dfm-python package.
- Forecast targets "KOGDP...D", "KOCNPER.D", and "KOGFCF..D" and compare the ressults.
- Compare the standardized MSE, MAE, RMSE for 1 day forecast, 7 days(1 week) horizon forecast, and 28 days(1 month) forecast results
- This is multivariate forecasting with models: "ARIMA(univariate benchmark)", "VAR", "VECM", "PytorchForecastingDeepAR", "PytorchForecastingTFT", "XGBoost", "LightGBM", "DFM" and "DDFM".
- Compare the nowcasting results with backtesting on masked data with simulated nowcasting. This should only be compared between DFM and DDFM since dfm-python package only support this.
- Ablation study with "DFM" and "DDFM"
- Write 20~30 pages report with detailed explanation, citation, with tables and plots. Write in korean

# RULES
- DO NOT CREATE FILES
- DO NOT DELETE FILES
- You can create config yaml files under config/model/*.yaml and config/experiment/*.yaml for experiments.
- Use sktime transformer, forecaster, evaluator pipelines to run experiments.
- Use config/ and hydra-core for configuration of model and series.
- Use data/sample_data.csv as a source
- Place images for plot in nowcasting-report/images/*.png
- Actively refer to neo4j mcp knowledgebase
- Always add the reference to nowcasting-report/references.bib when referenced. DO NOT add the reference if it doesn't exist in knowledgebase.
- DO NOT CREATE MARKDOWN FILES. Use the nowcasting-report as your context.
- If any broken codes or issues are identified in src/ or dfm-python, fix them.
- Try to work incremental. You don't have to do everything at once.

# REPORT