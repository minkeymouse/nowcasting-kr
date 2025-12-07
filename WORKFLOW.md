# RULES
- DO NOT CREATE FILES or MARKDOWNS
- DO NOT DELETE FILES
- ONLY WORK under modifying existing codes.
- NEVER HALLUCINATE. Only use the references at references.bib when writing the report.
- When need to add new information from knowledgebase, add the citation at references.bib
- src/ should contain maximum 15 files including __init__.py file. If necessary, restructure and consolidate.
- CONTEXT.md, STATUS.md, ISSUES.md MUST BE UNDER 1000 lines.
- Try to improve incrementally. Do not try to do everything at once. Try to prioritize the tasks and work on them one by one.
- Tables and Images specified by the user must be included in the paper.
- Don't worry about the backward compatibility. Focus on making clean code.

# GOAL
- Write the Complete report(20~30 pages) in @nowcasting-report/
- Finalize the package @dfm-python/ with clean code pattern, consistent and generic naming
- All tables and plots desired created and implemented.
- Clean and refactored codebase with consistent pattern and no unused, legacy codes, no redundancies and generic naming.

# RESOURCES
- CONTEXT.md: Use this file for context offloading for persistence if necessary.
- STATUS.md: Use this file to track the progress and leave the status for next iteration on updates.
- ISSUES.md: Track resolved issues and next steps. Keep file under 1000 lines. Mark resolved issues clearly.
- src/ : engine for running the experiment. This module provides wrapper for @sktime and @dfm-python packages with preprocessing - training - inference. Maximum 15 files including __init__.py.
- dfm-python/ : Core DFM/DDFM package - finalized with clean code patterns, consistent naming, legacy code cleaned up.
- nowcasting-report/code/plot.py : Code for creating plots used in the paper based on the results in outputs/ directory. Images should be created at nowcasting-report/images/*.png and used in the report properly.
- neo4j mcp : knowledgebase containing references. NEVER hallucinate.
- outputs/ : directory containing experiment results from @run_experiment.sh
- config/ : Hydra YAML configs in config/experiment/, config/model/, config/series/
- DDFM_COMPARISON.md : Comparison of original ddfm implementation and dfm-python
- @BENCHMARK_REPORT.md : Benchmark report that our report should focus on. This is report from another co-researcher that our report should consolidate and maintain the same experiment configuration. 

# Iteration Steps

## Step 1(Initial experiment run)
- Run the script @run_experiment.sh with bash.
- For incremental testing, use MODELS filter: `MODELS="dfm" bash run_experiment.sh` or `MODELS="ddfm" bash run_experiment.sh`
- Current status: ARIMA (9/9 complete), VAR (9/9 complete), DFM (1/9 tested, ready for full run), DDFM (1/9 tested, ready for full run)

## Step 2(cursor-agent, fresh new start)
- Inspect the @src/ @dfm-python/ and @nowcasting-report/ to understand the project.
- Check @STATUS.md and @ISSUES.md for current state and pending tasks.
- Study the experiment run output in outputs/ directory with latest run and plan how to update the @nowcasting-report with results.
- Current experiment status: 18/36 complete (ARIMA 9, VAR 9), 18 pending (DFM 8, DDFM 8)

## Step 3(cursor-agent resume)
- Work on the plan from step 2

## Step 4(cursor-agent resume)
- Analyze the results. If there's errors or issues, update them in the @STATUS.md and @ISSUES.md and inspect what happened.
- If there's something wrong with the numbers, also update them in the @STATUS.md and think about what happened.
- Mark resolved issues clearly in @ISSUES.md (use ✅ RESOLVED status).
- Update STATUS, ISSUES, CONTEXT if necessary. Keep files under 1000 lines.

## Step 5(cursor-agent resume)
- Plan how to improve the dfm-python package and nowcasting-report paper.
- If there are improvement points in the codes, such as numerical stability, convergence issues, theoretically wrong implementation(refer to kb and legacy clone repos if needed), include the improvements on them in the plan.
- If there are improvement points in the report, such as hallucination, lack of detail, redundancy, unnatural flow, include the improvements in the plan.
- If there are improvement points in the code quality such as redundancies, non-generic naming in dfm-python, inefficient logic, monkey patch, temporal fixes, include them in the plan.
- Note: Legacy code cleanup is completed. dfm-python is finalized with consistent naming and clean patterns.
- If there are any new experiments needed for the report or extensions, changes in experiment, include them in the plan.
- Do not make the plan too long. Leave the tasks at the @ISSUES.md and work incrementally. Plan with manageable tasks.

## Step 6(cursor-agent resume)
- Work on the plan

## Step 7(cursor-agent resume)
- Keep working on the plan with any unfinished tasks

## Step 8(cursor-agent resume)
- Identify the work done in this iteration. Identify what's done, what's not done. Update the @STATUS.md and @ISSUES.md for the next iteration.
- Mark resolved issues clearly in @ISSUES.md. Remove old resolved issues to keep file under 1000 lines.
- Update experiment status in @STATUS.md (completed/pending combinations).
- Next iteration will start fresh so you need to leave the proper context for next iteration.

## Step 9(cursor-agent resume)
- stage and commit the changes to keep track on them.
- Only in every 10 iterations, push the submodules to main.

# EXPERIMENT OUTPUT

## Tables
1. Table consisting of dataset details, arima, var, dfm and ddfm params(model and training)
2. Table consisting of averaged standardized MSE and standardized MAE for (target, model, horizon) pair. For instance, (dfm, 1), (dfm, 2), ..., (dfm, 7),... , (dfm, 30). We have 3 targets, 30 horizons averaged, and 4 models so the table will be 4 rows and 4 columns(model, target1, target2, target3 fields).
3. Table consisting of dfm and ddfm backtest results for year 2024~2025 each month. Train with data from 1985 to 2019, nowcast from Jan 2024 to Oct 2025(or any latest). By masking unavailable data, make sMSE, sMAE for each month by nowcasting.

## Images
1. Plot for 30 months forecasting and actual value in the test dataset for each target. This means 3 plots, each plot consist of original series line, arima, var, dfm, ddfm lines. This means 5 lines for each plot. Make sure x axis is monthly time stamp and y axis is target series. Make sure x axis has total 60 months which on the left, original series values so only one line and after the cutoff, 5 lines.
2. Plot for accuracy heatmap of 4 models and 3 targets.
3. Plot for performance trend with forecasting horizon.