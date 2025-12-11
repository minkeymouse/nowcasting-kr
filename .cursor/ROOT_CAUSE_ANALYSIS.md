# DDFM 성능 저하 근본 원인 분석

## 핵심 발견

### 문제 상황
- **이전**: DDFM이 가장 좋은 성능
- **현재**: 성능 저하
- Double differencing 문제는 수정했지만 여전히 성능이 안 좋음

### 핵심 모순

1. **이전 구현 (잘 작동)**:
   - DDFM도 transformations 포함 (differencing 포함)
   - `WHY_IT_WORKED_BEFORE.md`: "예전: DDFM도 transformations 포함"
   - Mx, Wx는 differenced + standardized 데이터의 통계값
   - Predict()에서 differencing을 역변환하면서 원래 스케일로 복원

2. **현재 구현 (성능 저하)**:
   - DDFM은 transformations 제거 (표준화만)
   - `_create_preprocessing_pipeline`: DDFM은 표준화만 수행
   - Mx, Wx는 원본 데이터의 통계값
   - Predict()에서 inverse transform 없음

### 데이터 흐름 불일치 발견

**학습 시** (`prepare_multivariate_data`):
- 587줄: `if model_type.lower() not in ('dfm', 'ddfm'):` → DFM/DDFM은 transformations 적용 안 함
- Transformations는 preprocessing pipeline에서 처리됨

**예측 시** (`_prepare_dfm_recent_data`):
- 91줄: `selected_data = apply_transformations(...)` → transformations 적용함

**이것은 데이터 불일치를 야기할 수 있습니다!**

### 가능한 원인

#### 가설 1: 이전에는 `prepare_multivariate_data`에서 transformations를 적용했음
- 현재는 DFM/DDFM에 대해 transformations를 적용하지 않음
- 하지만 이전에는 적용했을 가능성
- 이 경우, preprocessing pipeline에서 transformations를 제거한 것이 문제

#### 가설 2: 예측 시 recent data 준비에서 불일치
- 학습 시: transformations 없음
- 예측 시: transformations 적용
- 이것은 데이터 불일치를 야기함

#### 가설 3: 이전 실험이 잘못되었을 수 있음
- 사용자가 의심하는 것
- 하지만 "original implementation과 비교하면서 확인까지 했었거든?" → 신뢰할 만함

## 해결 방안

### 옵션 1: DDFM에 transformations 다시 추가
- 이전과 동일하게 differencing 포함
- 하지만 이것은 "오리지널 DDFM 구현과 다름"이라는 문제가 있음

### 옵션 2: 예측 시 recent data 준비 수정
- `_prepare_dfm_recent_data`에서 transformations를 적용하지 않도록 수정
- 학습 시와 예측 시 일관성 유지

### 옵션 3: `prepare_multivariate_data`에서 transformations 다시 적용
- DFM/DDFM에 대해서도 transformations 적용
- 하지만 이것은 double differencing 문제를 다시 야기할 수 있음

## 권장 사항

**가장 가능성 높은 원인**: 예측 시 recent data 준비에서 transformations를 적용하는 것이 문제

**해결책**: `_prepare_dfm_recent_data`에서 DFM/DDFM에 대해서는 transformations를 적용하지 않도록 수정
