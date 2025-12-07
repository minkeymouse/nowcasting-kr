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
- outputs/ : directory containing experiment results from @run_nowcast.sh
- config/ : Hydra YAML configs in config/experiment/, config/model/, config/series/
- DDFM_COMPARISON.md : Comparison of original ddfm implementation and dfm-python
- @BENCHMARK_REPORT.md : Benchmark report that our report should focus on. This is report from another co-researcher that our report should consolidate and maintain the same experiment configuration. 

# Iteration Steps

## Step 1(Initial experiment run)
- Run the script @run_nowcast.sh with bash.
- For incremental testing, use MODELS filter: `bash run_nowcast.sh --models 'arima var ddfm'`
- Experiments evaluate 22 monthly horizons (2024-01 to 2025-10) for each model-target combination.
- Total combinations: 3 targets × 4 models × 22 horizons = 264 combinations
- Nowcasting: 3 targets × 4 models × 22 months × 2 timepoints (4 weeks, 1 week) = 528 nowcast predictions

## Step 2(cursor-agent, fresh new start)
- Inspect the @src/ @dfm-python/ and @nowcasting-report/ to understand the project.
- Check @STATUS.md and @ISSUES.md for current state and pending tasks.
- Study the experiment run output in outputs/ directory with latest run and plan how to update the @nowcasting-report with results.
- Current experiment status: 22 monthly horizons (2024-01 to 2025-10). Track progress: 3 targets × 4 models × 22 horizons = 264 combinations total.

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
table1. Table consisting of dataset details, arima, var, dfm and ddfm params(model and training)

table2. Table consisting of standardized MSE and standardized MAE for forecasting results. 
- Experiments evaluate all horizons from 1 to 22 months (2024-01 to 2025-10), but table shows only selected horizons (1, 11, 22 months) for readability.
- Table structure: Rows = model-horizon combinations (4 models × 3 horizons = 12 rows: ARIMA-1, ARIMA-11, ARIMA-22, VAR-1, VAR-11, VAR-22, DFM-1, DFM-11, DFM-22, DDFM-1, DDFM-11, DDFM-22)
- Columns = target-metric combinations (3 targets × 2 metrics = 6 columns: KOIPALL.G_sMAE, KOIPALL.G_sMSE, KOEQUIPTE_sMAE, KOEQUIPTE_sMSE, KOWRCCNSE_sMAE, KOWRCCNSE_sMSE)
- Total: 12 rows × 7 columns (including model-horizon column)

table3. Table consisting of all models (ARIMA, VAR, DFM, DDFM) backtest results for year 2024~2025 each month. Train with data from 1985 to 2019, nowcast from Jan 2024 to Oct 2025 (22 months). For each target month, perform nowcasting at multiple time points (4 weeks before, 1 week before month end). By masking unavailable data based on release dates from series config, generate 1 horizon forecast at each time point. Calculate sMSE, sMAE for each month and time point. Table structure: Rows = model-timepoint combinations (4 models × 2 timepoints = 8 rows: ARIMA-4weeks, ARIMA-1week, VAR-4weeks, VAR-1week, DFM-4weeks, DFM-1week, DDFM-4weeks, DDFM-1week), Columns = target-metric combinations (3 targets × 2 metrics = 6 columns: KOIPALL.G_sMAE, KOIPALL.G_sMSE, KOEQUIPTE_sMAE, KOEQUIPTE_sMSE, KOWRCCNSE_sMAE, KOWRCCNSE_sMSE). Total: 8 rows × 7 columns (including model-timepoint column).

## Images
Plot1. Plot for 22 months forecasting and actual value for each target (2024-01 to 2025-10). This means 3 plots (one per target: KOIPALL.G, KOEQUIPTE, KOWRCCNSE), each plot consist of original series line, arima, var, dfm, ddfm lines. This means 5 lines for each plot. Make sure x axis is monthly time stamp and y axis is target series. Make sure x axis shows 22 months of forecasts (2024-01 to 2025-10) with actual values and 4 model predictions.

Plot2. Plot for accuracy heatmap of 4 models and 3 targets. Shows standardized RMSE values as a heatmap with models on one axis and targets on the other axis.

Plot3. Plot for performance trend with forecasting horizon. Shows sMSE values for all horizons from 1 to 22 months. X-axis: forecast horizon (1-22 months), Y-axis: sMSE value. Four lines representing four models (ARIMA, VAR, DFM, DDFM). This plot shows the complete performance trend across all evaluated horizons (2024-01 to 2025-10).

Plot4. Plot for nowcasting comparison at different time points. For each target (3 targets: KOIPALL.G, KOEQUIPTE, KOWRCCNSE), create side-by-side plots comparing "4 weeks before" vs "1 week before" nowcasting. Each plot shows 22 months (2024-01 to 2025-10) of predictions and actual values. X-axis: monthly time stamp (2024.01 ~ 2025.10), Y-axis: target series value (%). Each plot contains: actual value line (blue solid line) and model average prediction line (red dotted line with triangles). Total: 3 pairs of plots (6 plots total, one pair per target). This shows how prediction accuracy improves as we get closer to the month end (more data available).

# REPORT STRUCTURE

## 파일 구조
- `1_introduction.tex`: 서론 section
- `2_methodology.tex`: 결과 비교 section, 실험 설계 subsection
- `3_results.tex`: 결과 subsection (wrapper)
- `3_results_forecasting.tex`: Forecasting subsubsection
- `4_results_nowcasting.tex`: Nowcasting subsubsection
- `5_results_performance.tex`: Performance subsubsection
- `6_discussion.tex`: 논의 section
- `7_issues.tex`: 이슈 분석 section

## 서론 (section) - `1_introduction.tex`
### 선행연구 검토 (subsection)
    - DFM, DDFM 원본 논문 언급
    - Tent kernel 방법 설명
    - (subsubsection) 동적요인모형 (Dynamic Factor Model)
    - (subsubsection) 심층 동적요인모형 (Deep Dynamic Factor Model)
    - (subsubsection) 혼합주기 데이터 처리: 텐트 커널 방법

## 결과 비교 (section) - `2_methodology.tex`
### 실험 설계 (subsection)
    - (subsubsection) 실험 셋업
        - table 1: 실험 셋업 설명. 어떤 파라미터, 어떤 series 들어갔는지 왜 이런 설정으로 진행했는지
    - (subsubsection) 데이터 전처리
        - 데이터 전처리, imputation, scale 등 설명(series별 transformation도 설명)
    - (subsubsection) 데이터 품질 문제 및 시리즈 제거
    - (subsubsection) 모형 훈련 개선 사항
    - (subsubsection) 예측 모형
    - (subsubsection) Forecasting과 Nowcasting 차이 설명:
        - Forecasting: 1~22 horizon (월별, 2024-01부터 2025-10까지) 진행하고, 평균 내는 것. 하나의 모델을 훈련하고 모든 horizon에 대해 예측 수행.
        - Nowcasting: 각 목표 월(2024-01부터 2025-10까지, 22개월)에 대해 여러 시점(4주 전, 1주 전 등)에서 예측 수행. 각 시점에서 release 기준으로 nan masking하고, view_date = target_month_end - weeks_before로 계산. 각 시점에서 1 horizon forecast 생성. 시점별 예측 정확도 비교(시간이 지날수록 더 많은 데이터 사용 가능하여 정확도 향상).
    - (subsubsection) 평가 지표

### 결과 (subsection) - `3_results.tex` (wrapper)
    - (subsubsection) Forecasting - `3_results_forecasting.tex`
        - table 2: forecasting 결과 테이블. 실험은 1-22 horizon (월별) 모두 수행하지만, 표에는 1, 11, 22개월 값만 제시. Row는 model-horizon 조합(ARIMA-1, ARIMA-11, ARIMA-22, VAR-1, VAR-11, VAR-22, DFM-1, DFM-11, DFM-22, DDFM-1, DDFM-11, DDFM-22), Column은 target-metric 조합(KOIPALL.G_sMAE, KOIPALL.G_sMSE, KOEQUIPTE_sMAE, KOEQUIPTE_sMSE, KOWRCCNSE_sMAE, KOWRCCNSE_sMSE). 총 12 rows × 7 columns (model-horizon column 포함).
        - plot1: 각 대상 변수별 22개월 예측 및 실제 값 비교 플롯 (2024-01부터 2025-10까지). 3개 플롯(대상 변수별), 각 플롯은 원본 시계열, ARIMA, VAR, DFM, DDFM 예측선 포함(총 5개 선). X축은 월별 타임스탬프, Y축은 대상 변수 값. X축 22개월(2024-01 ~ 2025-10)의 실제값 + 4개 모형 예측값.
    
    - (subsubsection) Nowcasting - `4_results_nowcasting.tex`
        - Nowcasting 실험 구조 설명:
            - 모든 모형(ARIMA, VAR, DFM, DDFM)과 모든 대상 변수(3개)에 대해 수행.
            - 각 목표 월(2024-01 ~ 2025-10, 22개월)에 대해 여러 시점에서 예측:
                * 4주 전 시점: view_date = target_month_end - 4 weeks
                * 1주 전 시점: view_date = target_month_end - 1 week
            - 각 시점에서 release date 기반 데이터 마스킹 적용(시리즈별 발표 시차 사용).
            - 각 시점에서 1 horizon forecast 생성.
            - 시점별 예측 정확도 비교(4주 전 vs 1주 전).
        - table 3: 모든 모형(ARIMA, VAR, DFM, DDFM)의 2024-2025년 월별 백테스트 결과를 시점별로 제시. 훈련 기간 1985-2019, Nowcasting 기간 2024-01 ~ 2025-10 (22개월). Row는 model-timepoint 조합(8개 행: ARIMA-4weeks, ARIMA-1week, VAR-4weeks, VAR-1week, DFM-4weeks, DFM-1week, DDFM-4weeks, DDFM-1week), Column은 target-metric 조합(6개 열: KOIPALL.G_sMAE, KOIPALL.G_sMSE, KOEQUIPTE_sMAE, KOEQUIPTE_sMSE, KOWRCCNSE_sMAE, KOWRCCNSE_sMSE). 총 8 rows × 7 columns (model-timepoint column 포함). 각 셀은 해당 모형-시점-대상 조합에 대한 평균 sMSE 또는 sMAE를 나타냄.
        - plot4: Nowcasting 시점별 비교 플롯. 각 대상 변수별로 "4주 전 nowcasting"과 "1주 전 nowcasting"을 나란히 비교하는 플롯. 총 3쌍(6개 플롯, 대상 변수별 1쌍). 각 플롯은 22개월(2024-01 ~ 2025-10)의 예측값과 실제값을 시간 순서로 연결한 선 그래프. 파란선(실제값)과 빨간 점선(모형 평균 예측값) 비교. X축: 월별 타임스탬프(2024.01 ~ 2025.10), Y축: 대상 변수 값(%). 이 플롯은 시간이 지날수록(1주 전이 4주 전보다) 더 많은 데이터를 사용할 수 있어 예측 정확도가 향상됨을 보여줌.

    - (subsubsection) Performance - `5_results_performance.tex`
        - Training time: 각 모형의 훈련 시간 비교 및 분석
        - plot3: Horizon별 성능 추세 플롯. 모든 horizon(1-22개월)에 대한 sMSE 값을 플롯으로 제시. 가로축: 예측 수평선(1-22개월), 세로축: sMSE 값. 4개 모형(ARIMA, VAR, DFM, DDFM)에 대한 4개 선으로 표시. 이 플롯은 평가된 모든 수평선(2024-01부터 2025-10까지)에 걸친 완전한 성능 추세를 보여줌.
        - plot2: 정확도 히트맵. 모형 및 대상 변수별 표준화된 RMSE를 히트맵으로 시각화.

## 논의 (section) - `6_discussion.tex`
### 모델 비교 (subsection)
    - 네 가지 모형(ARIMA, VAR, DFM, DDFM)의 성능을 대상 변수와 예측 수평선에 걸쳐 비교 분석. Forecasting과 Nowcasting 결과를 종합적으로 비교.

### 원인 분석 (subsection)
    - (subsubsection) VAR의 수치적 불안정성
    - (subsubsection) DFM의 EM 알고리즘 수렴 문제
    - (subsubsection) DDFM의 비선형 패턴 포착 능력

### Nowcasting 시점별 분석 (subsection)
    - 4주 전 vs 1주 전 예측 정확도 비교. 시간이 지날수록 더 많은 데이터를 사용할 수 있어 예측 정확도가 향상되는 패턴 분석. 각 모형별로 시점별 성능 개선 정도 비교(예: DFM은 4주 전 0.8%p → 1주 전 0.6%p, DDFM은 4주 전 0.7%p → 1주 전 0.4%p 등). 벤치마크 리포트와의 비교 분석.
    - (subsubsection) 시점별 성능 개선 패턴
    - (subsubsection) 모형별 시점별 성능 차이
    - (subsubsection) Release date 기반 마스킹의 영향
    - (subsubsection) 벤치마크 리포트와의 비교
    - (subsubsection) 주요 발견 사항
    - (subsubsection) 실험 결과 업데이트 예정

## 이슈 분석 (section) - `7_issues.tex`
### 모형별 기술적 제한사항 (subsection)
    - (subsubsection) VAR의 긴 수평선에서의 불안정성
    - (subsubsection) DFM의 수치적 불안정성 경고
        - 모델별 성공/실패 현황 테이블
        - DFM 상세 결과
        - 훈련 로그 분석 결과
        - State Space 차원과 시리즈 개수의 관계
        - Kalman Filter의 F Matrix 불안정성
        - sum_EZZ 행렬의 Ill-conditioning
        - 연쇄 반응 (Cascade Failure)
        - 구현 차이로 인한 수치 불안정성
        - 해결 시도 실패
        - 적용된 해결 방안
        - 데이터 품질 분석
        - 단기/중기/장기 해결책
        - Clipping/Capping 접근법 상세 설명

### 실험 설계의 제한사항 (subsection)
    - (subsubsection) 테스트 데이터 부족
    - (subsubsection) 통계적 신뢰성 제한

### Nowcasting 실험의 제한사항 (subsection)
    - (subsubsection) Release date 정보의 정확성
    - (subsubsection) ARIMA/VAR 모형의 Release date 마스킹 구현
    - (subsubsection) 시점별 데이터 가용성 차이

## Appendix
### 전체 실험 결과 테이블(1~22 horizon, 모델별로 테이블 1개)
