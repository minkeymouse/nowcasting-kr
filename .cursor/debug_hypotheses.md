# DDFM 성능 저하 원인 분석 가설

## 핵심 문제
- 이전: DDFM이 가장 좋은 성능
- 현재: 성능 저하
- Double differencing 문제는 수정했지만 여전히 성능이 안 좋음

## 핵심 가설

### 가설 1: 이전에는 Differencing이 포함되어 있었고, 그게 잘 작동했다
**증거:**
- `WHY_IT_WORKED_BEFORE.md`: "예전: DDFM도 transformations 포함"
- `_create_preprocessing_pipeline`: 현재는 DDFM에 transformations 제거 (184-203줄)
- 이전에는 `create_transformer_from_config(config)` 사용 → Differencer 포함

**검증 필요:**
- 이전 실험에서 실제로 differencing이 포함되어 있었는지 확인
- Differencing이 포함된 상태에서 성능이 좋았는지 확인

### 가설 2: Mx, Wx 추출 로직이 잘못되었을 수 있음
**현재 구현:**
- `data_module.py:708-709`: `_get_mean(scaler, X_df.values)`, `_get_scale(scaler, X_df.values)`
- `_get_scaler_attr`: `scaler.mean_`과 `scaler.scale_`을 직접 읽음
- Scaler는 `pipeline.fit_transform(X_df)`에서 원본 데이터에 fit됨

**검증 필요:**
- Scaler가 실제로 원본 데이터에 fit되었는지 확인
- `scaler.mean_`과 `scaler.scale_`이 실제로 원본 데이터의 통계값인지 확인
- 이전에는 `X_processed_np`에서 추출했는데, 지금은 `X_df.values`에서 추출 → 차이점 확인

### 가설 3: 예측 시 Inverse Transform이 누락되었을 수 있음
**현재 구현:**
- `ddfm.py:1483-1523`: `_transform_factors_to_observations` 사용
- `base.py:296-298`: `X_forecast = X_forecast_std * Wx + Mx`
- Transformations 없으므로 inverse transform 건너뛰기

**검증 필요:**
- `_transform_factors_to_observations`가 올바른 스케일로 변환하는지 확인
- Mx, Wx가 올바른 값인지 확인
- 예측값이 실제값과 같은 스케일인지 확인

### 가설 4: 이전에는 Preprocessing을 모델 밖에서 했고, Inverse Transform을 수동으로 했음
**사용자 설명:**
- "원래 preprocessing(imputation, transformation, scaling) 모델 넣기 전에 하고 모델에서 결과 가져다가 manually inverse transform 했던거거든?"
- "그걸 그냥 안으로 포함시킨거야"

**검증 필요:**
- 이전 구현에서 preprocessing이 어디서 수행되었는지 확인
- Inverse transform이 어디서 수행되었는지 확인
- 현재 구현과 차이점 확인

### 가설 5: Decoder Bias Term이 잘못 적용되었을 수 있음
**현재 구현:**
- `ddfm.py:1143-1144`: `Mx_adjusted = Mx_clean + bias * Wx_clean`
- `predict()`에서 `result.Mx` 사용 (이미 bias 조정됨)

**검증 필요:**
- Bias term이 올바르게 계산되는지 확인
- Bias term이 올바르게 적용되는지 확인

## 디버깅 계획

1. **학습 시 데이터 흐름 추적:**
   - `prepare_multivariate_data`: transformations 적용 여부 확인
   - `_create_preprocessing_pipeline`: DDFM pipeline 생성 확인
   - `DFMDataModule.setup()`: pipeline fit_transform 확인
   - Mx, Wx 추출 확인

2. **예측 시 데이터 흐름 추적:**
   - `update()`: preprocessing 적용 확인
   - `predict()`: factor forecast → observation transform 확인
   - Inverse transform 확인

3. **실제 값 비교:**
   - 예측값과 실제값의 스케일 비교
   - 예측값과 실제값의 통계값 비교
