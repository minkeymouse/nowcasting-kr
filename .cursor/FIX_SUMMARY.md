# DDFM 성능 저하 수정 요약

## 핵심 문제

**이전 구현 (잘 작동)**:
- `src/` 레벨에서 transformations 적용 (`prepare_multivariate_data`)
- 모델에 transformed 데이터 전달
- 예측 후 manually inverse transform

**현재 구현 (성능 저하)**:
- `prepare_multivariate_data`에서 DFM/DDFM에 대해 transformations 미적용
- Preprocessing pipeline에서도 transformations 제거
- 예측 후 inverse transform 없음

## 수정 사항

### 1. `prepare_multivariate_data`에서 DDFM에 대해 transformations 다시 적용
- **파일**: `src/train/preprocess.py`
- **변경**: DDFM에 대해서도 transformations 적용 (587줄, 625줄)
- **이유**: 이전 구현과 동일하게 유지

### 2. 예측 후 manually inverse transform 추가
- **파일**: `src/models/ddfm.py`
- **변경**: `forecast_ddfm`에서 예측 후 모든 시리즈에 대해 inverse transform 적용
- **이유**: 이전 구현과 동일하게 manually inverse transform

### 3. `_prepare_dfm_recent_data`에서 DFM/DDFM에 대해 transformations 제거
- **파일**: `src/train/train_common.py`
- **변경**: DFM/DDFM에 대해서는 transformations 적용 안 함
- **이유**: 학습 시와 일관성 유지

## 데이터 흐름

**학습 시**:
1. Raw data → `prepare_multivariate_data` → transformations 적용
2. Transformed data → preprocessing pipeline → standardization만 (transformations 없음)
3. Standardized data → 모델 학습

**예측 시**:
1. Raw data (y_recent) → `update()` → preprocessing pipeline 적용 (standardization만)
2. Factor forecast → `_transform_factors_to_observations` → transformed space
3. Manually inverse transform → original space

## 기대 효과

- 이전 구현과 동일한 데이터 흐름 유지
- Transformations가 적용되지만 double differencing 방지 (pipeline에서는 standardization만)
- 예측 후 manually inverse transform으로 원본 스케일 복원
