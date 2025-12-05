# 보고서에 필요한 실험 결과 목록

## 보고서 구조 기반 필수 결과

본 문서는 `nowcasting-report/contents/3_result.tex`에 실제로 언급된 표와 그림만을 포함함.

---

## 1. 전체 모형 성능 비교

### 1.1 표준화된 성능 지표 표
**보고서 참조:** `tab:overall_metrics`, `tab:overall_metrics_by_target`, `tab:overall_metrics_by_horizon`

**필요한 데이터:**
- 9개 모형 (ARIMA, VAR, VECM, DeepAR, TFT, XGBoost, LightGBM, DFM, DDFM)
- 3개 목표 변수 (KOGDP___D, KOCNPER_D, KOGFCF__D)
- 3개 예측 기간 (1일, 7일, 28일)
- 3개 평가 지표 (sMSE, sMAE, sRMSE)

**출력 형식:**
- 표: `tab:overall_metrics` (모형별 전체 평균 성능)
- 표: `tab:overall_metrics_by_target` (목표 변수별 성능)
- 표: `tab:overall_metrics_by_horizon` (예측 기간별 성능)

**데이터 소스:**
- `outputs/comparisons/*/comparison_results.json`에서 추출
- `src/visualization/generate_tables.py`로 LaTeX 표 생성

**상태:** ✅ 완료 (generate_tables.py로 생성됨)

---

## 2. DFM과 DDFM의 성능 분석

### 2.1 예측 기간별 성능 분석
**보고서 참조:** 텍스트 설명만 있음 (표/그림 없음)

**필요한 데이터:**
- DFM과 DDFM의 1일, 7일, 28일 예측 성능 (3개 목표 변수별)
- 표준화된 RMSE 값

**출력 형식:**
- 표는 필요 없음 (텍스트 설명만 있음)
- 그림은 선택사항 (보고서에 언급 없음)

**데이터 소스:**
- `outputs/comparisons/*/comparison_results.json`에서 DFM, DDFM만 필터링

**상태:** ✅ 완료 (텍스트 설명만 있으면 됨)

---

## 3. 예측 기간별 성능 분석

### 3.1 예측 기간별 성능
**보고서 참조:** 텍스트 설명만 있음 (표/그림 없음)

**필요한 데이터:**
- 모든 모형의 1일, 7일, 28일 예측 성능

**출력 형식:**
- 표는 필요 없음 (텍스트 설명만 있음)
- 그림: `fig:horizon_trend` (이미 있음)

**데이터 소스:**
- `outputs/comparisons/*/comparison_results.json`
- `nowcasting-report/code/plot.py`의 `plot_horizon_trend()` 함수로 생성

**상태:** ✅ 완료 (horizon_trend.png 이미 생성됨)

---

## 4. DFM vs DDFM 나우캐스팅 비교

### 4.1 나우캐스팅 성능 지표
**보고서 참조:** `tab:nowcasting_metrics`, `tab:nowcasting_by_target`, `tab:nowcasting_by_masking`

**필요한 데이터:**
- 마스킹된 데이터를 활용한 백테스팅 결과
- DFM과 DDFM의 나우캐스팅 성능 (sMSE, sMAE, sRMSE)
- 각 목표 변수별 나우캐스팅 성능
- 마스킹 기간별 성능 (1주일 전, 2주일 전, 1개월 전)

**출력 형식:**
- 표: `tab:nowcasting_metrics` (전체 비교) ✅ 완료
- 표: `tab:nowcasting_by_target` (목표 변수별) ⚠️ placeholder
- 표: `tab:nowcasting_by_masking` (마스킹 기간별) ⚠️ placeholder

**데이터 소스:**
- `outputs/experiments/nowcasting/` 또는 `outputs/comparisons/*/comparison_results.json`에서 나우캐스팅 결과 추출
- `src/visualization/generate_tables.py`에 함수 추가 필요

**상태:** ⚠️ 부분 완료 - 기본 표만 있고 목표 변수별/마스킹 기간별 표는 placeholder

---

## 5. Ablation Study 결과

### 5.1 DFM 하이퍼파라미터 분석
**보고서 참조:** `tab:dfm_ablation_factors`

**필요한 데이터:**
- 요인 개수 변화에 따른 성능 (1, 2, 3, 4, 5개)
- 각 목표 변수별 성능 (GDP, 민간 소비, 총고정자본형성)

**출력 형식:**
- 표: `tab:dfm_ablation_factors` (요인 개수별) ⚠️ placeholder

**데이터 소스:**
- `outputs/experiments/ablation/dfm/` 또는 별도 실험 결과
- `src/visualization/generate_tables.py`에 함수 추가 필요

**상태:** ⚠️ placeholder만 있음

### 5.2 DDFM 하이퍼파라미터 분석
**보고서 참조:** `tab:ddfm_ablation_layers`, `tab:ddfm_ablation_lr`

**필요한 데이터:**
- 인코더 레이어 수 변화에 따른 성능 (1, 2, 3, 4 레이어)
- 학습률 변화에 따른 성능 (0.0001, 0.001, 0.01, 0.1)
- 각 목표 변수별 성능 (GDP, 민간 소비, 총고정자본형성)

**출력 형식:**
- 표: `tab:ddfm_ablation_layers` (레이어 수별) ⚠️ placeholder
- 표: `tab:ddfm_ablation_lr` (학습률별) ⚠️ placeholder

**데이터 소스:**
- `outputs/experiments/ablation/ddfm/` 또는 별도 실험 결과
- `src/visualization/generate_tables.py`에 함수 추가 필요

**상태:** ⚠️ placeholder만 있음

---

## 6. 시각화 자료

### 6.1 모형별 성능 비교
**보고서 참조:** `fig:model_comparison`

**필요한 데이터:**
- 모든 모형의 성능 지표 (sMSE, sMAE, sRMSE)
- 모형별 전체 평균 성능

**출력 형식:**
- 그림: `fig:model_comparison` (막대 그래프) ✅ 완료

**생성 코드:**
- `nowcasting-report/code/plot.py`의 `plot_model_comparison()` 함수

**상태:** ✅ 완료

### 6.2 예측 기간별 성능 추이
**보고서 참조:** `fig:horizon_trend`

**필요한 데이터:**
- 각 모형의 1일, 7일, 28일 예측 성능 (sRMSE)

**출력 형식:**
- 그림: `fig:horizon_trend` (선 그래프) ✅ 완료

**생성 코드:**
- `nowcasting-report/code/plot.py`의 `plot_horizon_trend()` 함수

**상태:** ✅ 완료

### 6.3 목표 변수별 예측 정확도 히트맵
**보고서 참조:** `fig:heatmap`

**필요한 데이터:**
- 모형 × 목표 변수별 성능 지표 (sRMSE, 평균)

**출력 형식:**
- 그림: `fig:accuracy_heatmap` (히트맵) ✅ 완료

**생성 코드:**
- `nowcasting-report/code/plot.py`의 `plot_accuracy_heatmap()` 함수

**상태:** ✅ 완료

### 6.4 예측값 vs 실제값 시계열
**보고서 참조:** `fig:forecast_vs_actual`

**필요한 데이터:**
- 주요 모형들의 예측값과 실제값 시계열 데이터
- GDP 목표 변수 (보고서에 명시됨)

**출력 형식:**
- 그림: `fig:forecast_vs_actual` (시계열 그래프) ⚠️ placeholder

**생성 코드:**
- `nowcasting-report/code/plot.py`의 `plot_forecast_vs_actual()` 함수
- 현재는 placeholder만 생성

**상태:** ⚠️ placeholder만 있음

---

## 데이터 형식 요구사항

### JSON 파일 형식 (comparison_results.json)
```json
{
  "target_series": "KOGDP...D",
  "comparison": {
    "metrics_table": [
      {
        "model": "DFM",
        "sMSE_h1": 0.123,
        "sMAE_h1": 0.456,
        "sRMSE_h1": 0.789,
        ...
      }
    ]
  }
}
```

### 이미지 파일 형식
- 모든 그림은 PNG 형식으로 저장 (해상도: 300 DPI)
- 그림 크기: 최소 1200×800 픽셀
- 저장 위치: `nowcasting-report/images/*.png` (서브디렉토리 없이 직접 저장)
- 영어 레이블만 사용 (한글 레이블은 LaTeX 컴파일 오류 발생)

---

## 우선순위

### 높은 우선순위 (필수) - 보고서 완성을 위해 반드시 필요
1. ✅ 전체 모형 성능 비교 표 (1.1) - 완료
2. ✅ 기본 시각화 4개 (6.1, 6.2, 6.3, 6.4) - 완료 (6.4는 placeholder)
3. ⚠️ 나우캐스팅 목표 변수별/마스킹 기간별 표 (4.1) - placeholder
4. ⚠️ Ablation Study 표 3개 (5.1, 5.2) - placeholder

### 중간 우선순위 (권장) - 보고서 품질 향상
5. ⚠️ 예측값 vs 실제값 시계열 (6.4) - placeholder를 실제 데이터로 대체

---

## 참고사항

- 모든 결과는 `outputs/comparisons/*/comparison_results.json`에서 추출 가능
- 표는 `src/visualization/generate_tables.py`로 생성
- 그림은 `nowcasting-report/code/plot.py`로 생성
- 이미지는 `nowcasting-report/images/*.png` 형식으로 저장 (서브디렉토리 없이)
- 보고서에 placeholder가 있는 표들은 실제 결과로 대체 필요
- 모든 숫자 결과는 `outputs/` 디렉토리에 생성됨
