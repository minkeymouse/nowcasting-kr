# 보고서에 필요한 실험 결과 목록

## 현재 보고서 상태 요약

### 완료된 결과
- ✅ 전체 모형 성능 비교 표 (tab:overall_metrics, tab:overall_metrics_by_target, tab:overall_metrics_by_horizon)
- ✅ 기본 시각화 4개 (model_comparison, horizon_trend, accuracy_heatmap, forecast_vs_actual)
- ✅ DFM vs DDFM 나우캐스팅 기본 비교 (tab:nowcasting_metrics)

### 미완성/누락된 결과 (우선순위 높음)
- ❌ 목표 변수별 상세 성능 표 (tab:gdp_performance, tab:consumption_performance, tab:investment_performance)
- ❌ 목표 변수별 예측값 vs 실제값 시계열 그림 (각 목표 변수별)
- ❌ 예측 기간별 상세 표 (tab:horizon_1day, tab:horizon_7day, tab:horizon_28day)
- ❌ 예측 기간별 성능 저하 분석 그림
- ❌ 나우캐스팅 목표 변수별/마스킹 기간별 표 (tab:nowcasting_by_target, tab:nowcasting_by_masking) - [결과] placeholder
- ❌ 나우캐스팅 시나리오별 분석 그림
- ❌ Ablation Study 모든 표 (모두 [결과] placeholder)
- ❌ Ablation Study 시각화

---

## 1. 전체 모형 성능 비교

### 1.1 표준화된 성능 지표 표
**필요한 데이터:**
- 9개 모형 (ARIMA, VAR, VECM, DeepAR, TFT, XGBoost, LightGBM, DFM, DDFM)
- 3개 목표 변수 (KOGDP___D, KOCNPER_D, KOGFCF__D)
- 3개 예측 기간 (1일, 7일, 28일)
- 3개 평가 지표 (sMSE, sMAE, sRMSE)

**출력 형식:**
- 표: `tab:overall_metrics` (모형별 전체 평균 성능) ✅ 완료
- 표: `tab:overall_metrics_by_target` (목표 변수별 성능) ✅ 완료
- 표: `tab:overall_metrics_by_horizon` (예측 기간별 성능) ✅ 완료

**파일 위치:**
- `outputs/experiments/overall_performance/`

**상태:** ✅ 완료

---

## 2. 목표 변수별 성능 분석

### 2.1 GDP (KOGDP___D) 예측 성능
**필요한 데이터:**
- 각 모형별 1일, 7일, 28일 예측 성능 (sMSE, sMAE, sRMSE)
- 모형별 예측값 vs 실제값 시계열 데이터
- 예측 오차 분석 (평균, 표준편차, 최대/최소 오차)

**출력 형식:**
- 표: `tab:gdp_performance` ❌ 누락 (보고서에 텍스트 설명만 있음)
- 그림: `fig:gdp_forecasts` (예측값 vs 실제값 시계열) ❌ 누락
- 그림: `fig:gdp_errors` (예측 오차 분포) ❌ 누락

**파일 위치:**
- `outputs/experiments/target_analysis/gdp/`

**상태:** ❌ 미완성 - 텍스트 설명만 있고 표/그림 없음

### 2.2 민간 소비 (KOCNPER_D) 예측 성능
**필요한 데이터:**
- 각 모형별 1일, 7일, 28일 예측 성능 (sMSE, sMAE, sRMSE)
- 모형별 예측값 vs 실제값 시계열 데이터
- 예측 오차 분석

**출력 형식:**
- 표: `tab:consumption_performance` ❌ 누락
- 그림: `fig:consumption_forecasts` ❌ 누락
- 그림: `fig:consumption_errors` ❌ 누락

**파일 위치:**
- `outputs/experiments/target_analysis/consumption/`

**상태:** ❌ 미완성 - 텍스트 설명만 있고 표/그림 없음

### 2.3 총고정자본형성 (KOGFCF__D) 예측 성능
**필요한 데이터:**
- 각 모형별 1일, 7일, 28일 예측 성능 (sMSE, sMAE, sRMSE)
- 모형별 예측값 vs 실제값 시계열 데이터
- 예측 오차 분석 (변동성이 큰 변수이므로 상세 분석 필요)

**출력 형식:**
- 표: `tab:investment_performance` ❌ 누락
- 그림: `fig:investment_forecasts` ❌ 누락
- 그림: `fig:investment_errors` ❌ 누락

**파일 위치:**
- `outputs/experiments/target_analysis/investment/`

**상태:** ❌ 미완성 - 텍스트 설명만 있고 표/그림 없음

---

## 3. 예측 기간별 성능 분석

### 3.1 1일 예측 성능
**필요한 데이터:**
- 모든 모형 × 모든 목표 변수에 대한 1일 예측 성능
- 모형별 성능 순위
- 고빈도 데이터 활용 모형의 상대적 성능

**출력 형식:**
- 표: `tab:horizon_1day` ❌ 누락 (보고서에 텍스트 설명만 있음)
- 그림: `fig:horizon_1day_comparison` (막대 그래프) ❌ 누락

**파일 위치:**
- `outputs/experiments/horizon_analysis/horizon_1day/`

**상태:** ❌ 미완성 - 텍스트 설명만 있고 표/그림 없음

### 3.2 7일 예측 성능
**필요한 데이터:**
- 모든 모형 × 모든 목표 변수에 대한 7일 예측 성능
- 1일 예측 대비 성능 저하 정도
- 시계열 의존성 학습 모형의 상대적 성능

**출력 형식:**
- 표: `tab:horizon_7day` ❌ 누락
- 그림: `fig:horizon_7day_comparison` ❌ 누락
- 그림: `fig:horizon_degradation_7day` (1일 대비 성능 저하) ❌ 누락

**파일 위치:**
- `outputs/experiments/horizon_analysis/horizon_7day/`

**상태:** ❌ 미완성 - 텍스트 설명만 있고 표/그림 없음

### 3.3 28일 예측 성능
**필요한 데이터:**
- 모든 모형 × 모든 목표 변수에 대한 28일 예측 성능
- 1일, 7일 예측 대비 성능 저하 정도
- 장기 의존성 학습 모형의 상대적 성능

**출력 형식:**
- 표: `tab:horizon_28day` ❌ 누락
- 그림: `fig:horizon_28day_comparison` ❌ 누락
- 그림: `fig:horizon_degradation_28day` (1일, 7일 대비 성능 저하) ❌ 누락
- 그림: `fig:horizon_trend` (예측 기간별 성능 추이) ✅ 완료

**파일 위치:**
- `outputs/experiments/horizon_analysis/horizon_28day/`

**상태:** ❌ 부분 완료 - horizon_trend만 있고 나머지 누락

---

## 4. DFM vs DDFM 나우캐스팅 비교

### 4.1 나우캐스팅 성능 지표
**필요한 데이터:**
- 마스킹된 데이터를 활용한 백테스팅 결과
- DFM과 DDFM의 나우캐스팅 성능 (sMSE, sMAE, sRMSE)
- 각 목표 변수별 나우캐스팅 성능
- 마스킹 기간별 성능 (예: 1주일 전, 2주일 전, 1개월 전)

**출력 형식:**
- 표: `tab:nowcasting_metrics` (전체 비교) ✅ 완료
- 표: `tab:nowcasting_by_target` (목표 변수별) ⚠️ [결과] placeholder
- 표: `tab:nowcasting_by_masking` (마스킹 기간별) ⚠️ [결과] placeholder
- 그림: `fig:nowcasting_comparison` (DFM vs DDFM 비교) ❌ 누락
- 그림: `fig:nowcasting_forecasts` (나우캐스팅 예측값 vs 실제값) ❌ 누락

**파일 위치:**
- `outputs/experiments/nowcasting/`

**상태:** ⚠️ 부분 완료 - 기본 표만 있고 목표 변수별/마스킹 기간별 표는 placeholder, 그림 누락

### 4.2 나우캐스팅 시나리오별 분석
**필요한 데이터:**
- 실제 나우캐스팅 시나리오 시뮬레이션 결과
- 각 시점에서 사용 가능한 데이터만을 활용한 예측
- 예측 정확도 시계열 (시간에 따른 나우캐스팅 성능 변화)

**출력 형식:**
- 그림: `fig:nowcasting_scenarios` (시나리오별 예측) ❌ 누락
- 그림: `fig:nowcasting_accuracy_timeline` (시간에 따른 정확도) ❌ 누락

**파일 위치:**
- `outputs/experiments/nowcasting/scenarios/`

**상태:** ❌ 미완성 - 완전히 누락

---

## 5. Ablation Study 결과

### 5.1 DFM 하이퍼파라미터 분석
**필요한 데이터:**
- 요인 개수 변화에 따른 성능 (1, 2, 3, 4, 5개)
- AR 차수 변화에 따른 성능 (1, 2차)
- 최대 반복 횟수 변화에 따른 성능 (100, 500, 1000, 5000)
- 수렴 임계값 변화에 따른 성능 (1e-3, 1e-4, 1e-5, 1e-6)
- 블록 구조 변화에 따른 성능

**출력 형식:**
- 표: `tab:dfm_ablation_factors` (요인 개수별) ⚠️ [결과] placeholder
- 표: `tab:dfm_ablation_ar` (AR 차수별) ❌ 누락
- 표: `tab:dfm_ablation_iterations` (반복 횟수별) ❌ 누락
- 그림: `fig:dfm_ablation_factors` (요인 개수별 성능 추이) ❌ 누락
- 그림: `fig:dfm_ablation_ar` (AR 차수별 성능 비교) ❌ 누락
- 그림: `fig:dfm_ablation_heatmap` (하이퍼파라미터 조합별 히트맵) ❌ 누락

**파일 위치:**
- `outputs/experiments/ablation/dfm/`

**상태:** ⚠️ 부분 완료 - 요인 개수별 표만 있고 모두 placeholder, 나머지 누락

### 5.2 DDFM 하이퍼파라미터 분석
**필요한 데이터:**
- 인코더 레이어 수 변화에 따른 성능 (1, 2, 3, 4 레이어)
- 인코더 레이어 구조 변화에 따른 성능 ([32], [64], [32, 32], [64, 32], [128, 64, 32])
- 요인 개수 변화에 따른 성능 (1, 2, 3, 4, 5개)
- 학습률 변화에 따른 성능 (0.0001, 0.001, 0.01, 0.1)
- 배치 크기 변화에 따른 성능 (16, 32, 64, 128)
- 에폭 수 변화에 따른 성능 (50, 100, 200, 500)
- 활성 함수 변화에 따른 성능 (ReLU, tanh, sigmoid)
- 동학 가중치(λ) 변화에 따른 성능

**출력 형식:**
- 표: `tab:ddfm_ablation_layers` (레이어 수별) ⚠️ [결과] placeholder
- 표: `tab:ddfm_ablation_architecture` (아키텍처별) ❌ 누락
- 표: `tab:ddfm_ablation_factors` (요인 개수별) ❌ 누락
- 표: `tab:ddfm_ablation_lr` (학습률별) ⚠️ [결과] placeholder
- 표: `tab:ddfm_ablation_batch` (배치 크기별) ❌ 누락
- 표: `tab:ddfm_ablation_epochs` (에폭 수별) ❌ 누락
- 표: `tab:ddfm_ablation_activation` (활성 함수별) ❌ 누락
- 그림: `fig:ddfm_ablation_layers` (레이어 수별 성능 추이) ❌ 누락
- 그림: `fig:ddfm_ablation_factors` (요인 개수별 성능 추이) ❌ 누락
- 그림: `fig:ddfm_ablation_lr` (학습률별 성능 추이) ❌ 누락
- 그림: `fig:ddfm_ablation_heatmap` (하이퍼파라미터 조합별 히트맵) ❌ 누락

**파일 위치:**
- `outputs/experiments/ablation/ddfm/`

**상태:** ⚠️ 부분 완료 - 레이어 수별, 학습률별 표만 있고 모두 placeholder, 나머지 대부분 누락

### 5.3 최적 하이퍼파라미터 조합
**필요한 데이터:**
- DFM 최적 하이퍼파라미터 조합 및 성능
- DDFM 최적 하이퍼파라미터 조합 및 성능
- 하이퍼파라미터별 중요도 분석

**출력 형식:**
- 표: `tab:optimal_hyperparameters` ❌ 누락
- 그림: `fig:hyperparameter_importance` (중요도 분석) ❌ 누락

**파일 위치:**
- `outputs/experiments/ablation/optimal/`

**상태:** ❌ 미완성 - 완전히 누락

---

## 6. 시각화 자료

### 6.1 모형별 성능 비교 시각화
**필요한 데이터:**
- 모든 모형의 성능 지표 (sMSE, sMAE, sRMSE)
- 목표 변수별, 예측 기간별 성능

**출력 형식:**
- 그림: `fig:model_comparison` (막대 그래프 - 모형별 전체 평균 성능) ✅ 완료
- 그림: `fig:model_comparison_by_target` (목표 변수별 모형 성능) ❌ 누락
- 그림: `fig:model_comparison_by_horizon` (예측 기간별 모형 성능) ❌ 누락
- 그림: `fig:model_ranking` (모형 순위) ❌ 누락

**파일 위치:**
- `nowcasting-report/images/` (서브디렉토리 없이 직접 저장)

**상태:** ⚠️ 부분 완료 - 기본 비교만 있고 나머지 누락

### 6.2 예측 기간별 성능 추이
**필요한 데이터:**
- 각 모형의 1일, 7일, 28일 예측 성능
- 예측 기간에 따른 성능 저하 패턴

**출력 형식:**
- 그림: `fig:horizon_trend` (선 그래프 - 예측 기간별 성능 추이) ✅ 완료
- 그림: `fig:horizon_degradation` (성능 저하율) ❌ 누락

**파일 위치:**
- `nowcasting-report/images/`

**상태:** ⚠️ 부분 완료 - 기본 추이만 있고 저하율 분석 누락

### 6.3 목표 변수별 예측 정확도 히트맵
**필요한 데이터:**
- 모형 × 목표 변수 × 예측 기간별 성능 지표

**출력 형식:**
- 그림: `fig:accuracy_heatmap` (히트맵 - 모형 × 목표 변수) ✅ 완료
- 그림: `fig:accuracy_heatmap_horizon` (히트맵 - 모형 × 예측 기간) ❌ 누락
- 그림: `fig:accuracy_heatmap_3d` (3D 히트맵 - 모형 × 목표 변수 × 예측 기간) ❌ 누락

**파일 위치:**
- `nowcasting-report/images/`

**상태:** ⚠️ 부분 완료 - 기본 히트맵만 있고 나머지 누락

### 6.4 예측값 vs 실제값 시계열
**필요한 데이터:**
- 각 모형의 예측값과 실제값 시계열 데이터
- 주요 시점별 예측 정확도

**출력 형식:**
- 그림: `fig:forecast_vs_actual_gdp` (GDP 예측값 vs 실제값) ❌ 누락 (현재는 통합 버전만 있음)
- 그림: `fig:forecast_vs_actual_consumption` (소비 예측값 vs 실제값) ❌ 누락
- 그림: `fig:forecast_vs_actual_investment` (투자 예측값 vs 실제값) ❌ 누락
- 그림: `fig:forecast_vs_actual_all` (모든 목표 변수 통합) ✅ 완료 (현재 forecast_vs_actual.png)

**파일 위치:**
- `nowcasting-report/images/`

**상태:** ⚠️ 부분 완료 - 통합 버전만 있고 목표 변수별 누락

### 6.5 예측 오차 분석
**필요한 데이터:**
- 각 모형의 예측 오차 분포
- 예측 오차의 시계열 패턴
- 예측 오차의 자기상관 분석

**출력 형식:**
- 그림: `fig:error_distribution` (오차 분포 히스토그램) ❌ 누락
- 그림: `fig:error_timeseries` (오차 시계열) ❌ 누락
- 그림: `fig:error_autocorr` (오차 자기상관) ❌ 누락

**파일 위치:**
- `nowcasting-report/images/`

**상태:** ❌ 미완성 - 완전히 누락

### 6.6 요인 분석 (DFM/DDFM)
**필요한 데이터:**
- 추출된 요인 시계열
- 요인 적재 행렬
- 요인과 목표 변수 간의 상관관계

**출력 형식:**
- 그림: `fig:factors_timeseries` (요인 시계열) ❌ 누락
- 그림: `fig:factor_loadings` (요인 적재 히트맵) ❌ 누락
- 그림: `fig:factor_correlation` (요인-목표 변수 상관관계) ❌ 누락

**파일 위치:**
- `nowcasting-report/images/`

**상태:** ❌ 미완성 - 완전히 누락

### 6.7 Ablation Study 시각화
**필요한 데이터:**
- 각 하이퍼파라미터 변화에 따른 성능 추이
- 하이퍼파라미터 조합별 성능

**출력 형식:**
- 그림: `fig:ablation_summary` (Ablation study 요약) ❌ 누락
- 그림: `fig:ablation_heatmap` (하이퍼파라미터 조합별 히트맵) ❌ 누락

**파일 위치:**
- `nowcasting-report/images/`

**상태:** ❌ 미완성 - 완전히 누락

---

## 7. 통계적 유의성 검정

### 7.1 모형 간 성능 차이 검정
**필요한 데이터:**
- 각 모형의 예측 오차
- 모형 간 성능 차이의 통계적 유의성

**출력 형식:**
- 표: `tab:statistical_tests` (t-검정, Wilcoxon 검정 결과) ❌ 누락
- 표: `tab:model_significance` (모형 간 유의한 차이) ❌ 누락

**파일 위치:**
- `outputs/experiments/statistical_tests/`

**상태:** ❌ 미완성 - 완전히 누락

---

## 8. 추가 분석

### 8.1 COVID-19 기간 성능 분석
**필요한 데이터:**
- 2020년 2-3분기 성능 분석
- 구조적 변화 시기 모형 성능 비교

**출력 형식:**
- 표: `tab:covid_performance` ❌ 누락
- 그림: `fig:covid_forecasts` (COVID-19 기간 예측값 vs 실제값) ❌ 누락

**파일 위치:**
- `outputs/experiments/covid_analysis/`

**상태:** ❌ 미완성 - 완전히 누락

### 8.2 고빈도 데이터 활용 효과 분석
**필요한 데이터:**
- 고빈도 데이터 사용/미사용 모형 성능 비교
- 고빈도 변수별 기여도 분석

**출력 형식:**
- 표: `tab:high_freq_contribution` ❌ 누락
- 그림: `fig:high_freq_importance` (고빈도 변수 중요도) ❌ 누락

**파일 위치:**
- `outputs/experiments/high_frequency_analysis/`

**상태:** ❌ 미완성 - 완전히 누락

---

## 데이터 형식 요구사항

### CSV 파일 형식
- 모든 결과는 CSV 파일로 저장
- 컬럼: `model`, `target`, `horizon`, `metric`, `value`, `timestamp`
- 예시: `outputs/experiments/overall_performance/results.csv`

### JSON 파일 형식
- 메타데이터 및 설정 정보는 JSON으로 저장
- 예시: `outputs/experiments/overall_performance/metadata.json`

### 이미지 파일 형식
- 모든 그림은 PNG 형식으로 저장 (해상도: 300 DPI 이상)
- 그림 크기: 최소 1200×800 픽셀
- 저장 위치: `nowcasting-report/images/*.png` (서브디렉토리 없이 직접 저장)

---

## 우선순위 (업데이트됨)

### 높은 우선순위 (필수) - 보고서 완성을 위해 반드시 필요
1. ✅ 전체 모형 성능 비교 표 (1.1) - 완료
2. ❌ 목표 변수별 성능 분석 표 및 그림 (2.1, 2.2, 2.3) - 텍스트만 있고 표/그림 누락
3. ❌ 예측 기간별 성능 분석 표 및 그림 (3.1, 3.2, 3.3) - 텍스트만 있고 표/그림 누락
4. ⚠️ DFM vs DDFM 나우캐스팅 비교 (4.1) - 기본 표만 있고 목표 변수별/마스킹 기간별 표는 placeholder, 그림 누락
5. ⚠️ 기본 시각화 (6.1, 6.2, 6.3) - 기본만 있고 세부 시각화 누락
6. ❌ 목표 변수별 예측값 vs 실제값 시계열 (6.4) - 통합 버전만 있고 목표 변수별 누락

### 중간 우선순위 (권장) - 보고서 품질 향상을 위해 필요
7. ⚠️ Ablation Study 결과 (5.1, 5.2) - 표는 있으나 모두 placeholder, 시각화 누락
8. ❌ 나우캐스팅 시나리오별 분석 (4.2) - 완전히 누락
9. ❌ 예측 오차 분석 (6.5) - 완전히 누락
10. ❌ 요인 분석 (6.6) - 완전히 누락
11. ❌ Ablation Study 시각화 (6.7) - 완전히 누락

### 낮은 우선순위 (선택) - 추가 분석
12. ❌ 통계적 유의성 검정 (7.1) - 완전히 누락
13. ❌ COVID-19 기간 분석 (8.1) - 완전히 누락
14. ❌ 고빈도 데이터 활용 효과 (8.2) - 완전히 누락

---

## 참고사항

- 모든 결과는 재현 가능하도록 실험 설정과 시드값을 기록해야 함
- 각 실험의 실행 시간과 리소스 사용량도 기록 권장
- 결과 해석을 위한 추가 통계량(신뢰구간, 분위수 등)도 포함 권장
- 이미지는 `nowcasting-report/images/*.png` 형식으로 저장 (서브디렉토리 없이)
- 보고서에 [결과] placeholder가 있는 표들은 실제 결과로 대체 필요
