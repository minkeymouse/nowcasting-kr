# DDFM 이슈 및 오리지널 구현 비교 분석

## 1. 발생한 이슈 요약

### 문제 현상
- **증상**: 모든 DDFM 실행에서 C matrix가 100% NaN 값
- **결과**: `n_valid=0`, `converged=False`, `loglik=None`
- **로그**: "Forward pass produced 2664 NaN and 0 Inf values" 반복 발생
- **로그**: "extract_decoder_params: C matrix contains 74/74 NaN values (100.0%)"

### 근본 원인 분석

1. **Forward Pass에서 NaN 발생**
   - Encoder forward pass 중에 NaN이 생성됨
   - NaN이 0으로 대체되지만 학습은 계속되어 decoder weights에 NaN이 전파됨

2. **Gradient Clipping 부재**
   - `DDFMTrainer`의 기본값이 `gradient_clip_val=None`
   - Gradient explosion이 발생해 NaN이 생성될 수 있음

3. **Learning Rate 과다**
   - 기본값 0.001이 데이터 스케일에 비해 높을 수 있음
   - 오리지널 구현은 0.005를 사용하지만, ExponentialDecay 스케줄러 사용

4. **입력 데이터의 Extreme Values**
   - 전처리 과정에서 extreme values가 생성될 수 있음
   - "invalid value encountered in power" 경고 발생

## 2. 오리지널 DDFM 구현 (Andreini et al., 2020)

### 구현 특징

**프레임워크**: TensorFlow/Keras
- 오리지널은 TensorFlow 1.x/2.x 기반
- 우리 구현은 PyTorch Lightning 기반

**학습 절차** (Algorithm 1 from paper):
1. **Pre-training**: Missing data 없는 부분으로 autoencoder 사전 학습
2. **MCMC 반복**:
   - Idiosyncratic component 분포 추정 (AR(1))
   - Monte Carlo 샘플링으로 noisy inputs 생성 (`epochs`개의 샘플)
   - 각 MC 샘플에 대해 autoencoder 학습 (epochs=1)
   - Factors를 MC 샘플들의 평균으로 계산
   - Convergence 체크

**주요 설정** (`DDFM/models/ddfm.py`):
- **Learning Rate**: 0.005 (기본값)
- **Learning Rate Decay**: ExponentialDecay 사용 (decay_rate=0.96, decay_steps=epochs, staircase=True)
- **Optimizer**: Adam 또는 SGD
- **Batch Size**: 100 (기본값)
- **Epochs per MCMC iteration**: 100 (기본값)
- **Activation**: ReLU (기본값, `link='relu'`)
- **Initialization**: GlorotNormal (Xavier 초기화)
- **Batch Normalization**: Encoder에 사용 (기본값 `batch_norm=True`)
- **Max Iterations**: 200 (기본값)
- **Tolerance**: 0.0005 (기본값)

**Loss Function**: `mse_missing` (`DDFM/tools/loss_tools.py`)
```python
@tf.function
def mse_missing(y_actual: tf.Tensor, y_predicted: tf.Tensor) -> tf.Tensor:
    mask = tf.where(tf.math.is_nan(y_actual), tf.zeros_like(y_actual), tf.ones_like(y_actual))
    y_actual_ = tf.where(tf.math.is_nan(y_actual), tf.zeros_like(y_actual), y_actual)
    y_predicted_ = tf.multiply(y_predicted, mask)
    return keras.losses.mean_squared_error(y_actual_, y_predicted_)
```
- Missing data를 mask하여 계산
- TensorFlow의 `@tf.function` 데코레이터 사용

**C Matrix 추출** (`DDFM/tools/getters_converters_tools.py`):
```python
def convert_decoder_to_numpy(decoder: keras.Model, has_bias: bool, factor_oder: int,
                             structure_decoder: tuple = None) -> Tuple[np.ndarray, np.ndarray]:
    if structure_decoder is None:
        if has_bias:
            ws, bs = decoder.get_layer(index=-1).get_weights()
        else:
            ws = decoder.get_layer(index=-1).get_weights()[0]
            bs = np.zeros(ws.shape[1])
        # observable equation
        if factor_oder == 2:
            emission = np.hstack((
                ws.T,  # weight term (m x N)
                np.zeros((ws.shape[1], ws.shape[0])),  # make zero lagged values
                np.identity(ws.shape[1])  # idio
            ))
        elif factor_oder == 1:
            emission = np.hstack((
                ws.T,  # weight term (m x N)
                np.identity(ws.shape[1])  # idio
            ))
    return bs, emission
```
- Decoder의 마지막 layer에서 weight 추출
- `ws.T`로 transpose하여 (m x N) 형태로 변환
- State-space emission matrix 구성: [C, zeros/identity, I]

**MCMC Training 구조** (`DDFM/models/ddfm.py:train()`):
```python
def train(self) -> None:
    self.autoencoder.compile(optimizer=self.optimizer, loss=mse_missing)
    self.build_inputs()
    prediction_iter = self.autoencoder.predict(self.data_tmp.values)
    self.data_mod_only_miss.values[self.lags_input:][self.bool_miss] = prediction_iter[self.bool_miss]
    self.eps = self.data_tmp[self.data.columns].values - prediction_iter
    
    iter = 0
    not_converged = True
    while not_converged and iter < self.max_iter:
        # 1. Get idio distr
        phi, mu_eps, std_eps = get_idio(self.eps, self.bool_no_miss)
        
        # 2. Subtract conditional AR-idio mean from x
        self.data_mod[self.lags_input + 1:] = self.data_mod_only_miss[self.lags_input + 1:] - self.eps[:-1, :] @ phi
        
        # 3. Build inputs
        self.build_inputs()
        
        # 4. Gen MC samples for idio (dims = Sim x T x D)
        eps_draws = self.rng.multivariate_normal(mu_eps, np.diag(std_eps), (self.epoch, self.data_tmp.shape[0]))
        
        # 5. Loop over MC samples
        x_sim_den = np.zeros((eps_draws.shape[0], eps_draws.shape[1], eps_draws.shape[2] * (self.lags_input + 1)))
        for i in range(self.epoch):
            x_sim_den[i, :, :] = self.data_tmp.copy()
            x_sim_den[i, :, :eps_draws[i, :, :].shape[1]] = x_sim_den[i, :, :eps_draws[i, :, :].shape[1]] - eps_draws[i, :, :]
            # Fit autoencoder for 1 epoch
            self.autoencoder.fit(x_sim_den[i, :, :], self.z_actual, epochs=1, batch_size=self.batch_size, verbose=0)
        
        # 6. Update factors: average over all predictions from the MC samples
        self.factors = np.array([self.encoder(x_sim_den[i, :, :]) for i in range(x_sim_den.shape[0])])
        
        # 7. Check convergence
        prediction_iter = np.mean(np.array([self.decoder(self.factors[i, :, :]) for i in range(self.factors.shape[0])]), axis=0)
        if iter > 1:
            delta, self.loss_now = convergence_checker(prediction_prev_iter, prediction_iter, self.z_actual)
            if delta < self.tolerance:
                not_converged = False
        
        # 8. Update missings and idio
        self.data_mod_only_miss.values[self.lags_input:][self.bool_miss] = prediction_iter[self.bool_miss]
        self.eps = self.data_mod_only_miss.values[self.lags_input:] - prediction_iter
        iter += 1
```

**Pre-training** (`DDFM/models/ddfm.py:pre_train()`):
```python
def pre_train(self, min_obs: int = 50, mult_epoch_pre: int = 1) -> None:
    self.build_inputs(interpolate=False)
    if len(self.data_tmp.dropna()) >= min_obs:
        inpt_pre_train = self.data_tmp.dropna().values
        self.autoencoder.compile(optimizer=self.optimizer, loss='mse')
    else:
        self.build_inputs()
        inpt_pre_train = self.data_tmp.dropna().values
        self.autoencoder.compile(optimizer=self.optimizer, loss=mse_missing)
    oupt_pre_train = self.data_tmp.dropna()[self.data.columns].values
    self.autoencoder.fit(inpt_pre_train, oupt_pre_train, epochs=self.epoch * mult_epoch_pre,
                         batch_size=self.batch_size, verbose=0)
```
- Missing 없는 데이터로 사전 학습
- 충분한 관측치가 있으면 `mse` 사용, 없으면 `mse_missing` 사용

**수치 안정성 조치**:
- **데이터 표준화**: `(data - mean) / std` (line 60)
- **Learning Rate Decay**: 기본적으로 활성화 (`decay_learning_rate=True`)
- **Batch Normalization**: Encoder에 기본적으로 사용
- **Gradient Clipping**: 명시적으로 설정되지 않음 (TensorFlow의 기본 동작에 의존)
- **NaN 처리**: TensorFlow가 내부적으로 처리 (`mse_missing`에서 mask 사용)

## 3. 우리 구현 (dfm-python)과의 차이점

### 아키텍처 차이

| 항목 | 오리지널 (TensorFlow) | 우리 구현 (PyTorch) |
|------|----------------------|-------------------|
| 프레임워크 | TensorFlow/Keras | PyTorch Lightning |
| Encoder 구조 | Dense layers + BatchNorm | Encoder class (nn.Module) |
| Decoder 구조 | Linear (기본) 또는 Dense | Decoder class (nn.Linear) |
| Activation | ReLU (기본) | ReLU (기본, 수정 후) |
| Initialization | GlorotNormal | PyTorch 기본 (Xavier/He) |
| Loss Function | `mse_missing` (TensorFlow) | MSE with missing mask (PyTorch) |

### 학습 절차 차이

| 항목 | 오리지널 | 우리 구현 |
|------|---------|----------|
| Pre-training | Missing 없는 데이터로 사전 학습 | ✅ 동일하게 구현됨 |
| MCMC 구조 | 명시적인 MCMC 루프 (`train()`) | `fit_mcmc()` 메서드 |
| MC 샘플링 | 각 iteration마다 `epochs`개의 샘플 생성 | ✅ 동일하게 구현됨 |
| 학습 방식 | 각 MC 샘플에 대해 1 epoch 학습 | ✅ 동일하게 구현됨 |
| Learning Rate | 0.005 + ExponentialDecay | 0.005 + ExponentialLR (수정 후) |
| Gradient Clipping | 없음 (TensorFlow 기본) | 없음 (기본) → 1.0 (수정 후) |
| Batch Size | 100 | 100 (수정 후) |
| Epochs per iteration | 100 | 100 |

### C Matrix 추출 차이

**오리지널** (`DDFM/tools/getters_converters_tools.py`):
```python
def convert_decoder_to_numpy(decoder: keras.Model, has_bias: bool, factor_oder: int,
                             structure_decoder: tuple = None) -> Tuple[np.ndarray, np.ndarray]:
    if structure_decoder is None:
        if has_bias:
            ws, bs = decoder.get_layer(index=-1).get_weights()
        else:
            ws = decoder.get_layer(index=-1).get_weights()[0]
            bs = np.zeros(ws.shape[1])
        # ws shape: (N x m) for decoder output
        # ws.T shape: (m x N) for emission matrix
        if factor_oder == 2:
            emission = np.hstack((
                ws.T,  # (m x N) weight term
                np.zeros((ws.shape[1], ws.shape[0])),  # (m x N) zero lagged values
                np.identity(ws.shape[1])  # (m x m) idio
            ))
        elif factor_oder == 1:
            emission = np.hstack((
                ws.T,  # (m x N) weight term
                np.identity(ws.shape[1])  # (m x m) idio
            ))
    return bs, emission
```

**우리 구현** (`dfm-python/src/dfm_python/encoder/vae.py`):
```python
def extract_decoder_params(decoder) -> Tuple[np.ndarray, np.ndarray]:
    decoder_layer = decoder.decoder
    weight = decoder_layer.weight.data.cpu().numpy()  # (N x m)
    if decoder_layer.bias is not None:
        bias = decoder_layer.bias.data.cpu().numpy()
    else:
        bias = np.zeros(weight.shape[0])
    C = weight  # (N x m) - no transpose needed
    return C, bias

def convert_decoder_to_numpy(decoder: Any, has_bias: bool = True,
                             factor_order: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    linear_layer = decoder.decoder if hasattr(decoder, 'decoder') else decoder
    weight = linear_layer.weight.data.cpu().numpy()  # (N x m)
    if has_bias and linear_layer.bias is not None:
        bias = linear_layer.bias.data.cpu().numpy()
    else:
        bias = np.zeros(weight.shape[0])
    
    N, m = weight.shape
    if factor_order == 2:
        emission = np.hstack([
            weight,  # (N x m) current factors
            np.zeros((N, m)),  # (N x m) lagged factors
            np.eye(N)  # (N x N) idiosyncratic
        ])
    elif factor_order == 1:
        emission = np.hstack([
            weight,  # (N x m) factors
            np.eye(N)  # (N x N) idiosyncratic
        ])
    return bias, emission
```

**차이점**:
- **오리지널**: `ws.T`로 transpose하여 (m x N) 형태로 사용
- **우리**: `weight`를 그대로 (N x m) 형태로 사용
- **Emission matrix 구조**: 오리지널은 (m x state_dim), 우리는 (N x state_dim)
  - 오리지널: `[C.T, zeros, I]` where C.T is (m x N)
  - 우리: `[C, zeros, I]` where C is (N x m)

### 수치 안정성 조치 비교

| 조치 | 오리지널 | 우리 구현 (수정 전) | 우리 구현 (수정 후) |
|------|---------|------------------|------------------|
| Learning Rate | 0.005 + Decay | 0.001 (고정) | ✅ 0.005 + ExponentialLR |
| Gradient Clipping | 없음 | 없음 | ✅ 1.0 |
| Input Clipping | 없음 | 없음 | ✅ [-10, 10] |
| NaN 처리 | TensorFlow 기본 | 0으로 대체 후 계속 | ✅ Batch 건너뛰기 |
| Batch Normalization | ✅ Encoder에 사용 | ✅ Encoder에 사용 | ✅ 동일 |
| Data Standardization | ✅ (data - mean) / std | ✅ 전처리 파이프라인 | ✅ 동일 |
| Pre-training | ✅ Missing 없는 데이터 | 없음 | ✅ 동일하게 구현됨 |
| Activation | ReLU | Tanh | ✅ ReLU |
| Batch Size | 100 | 32 | ✅ 100 |

## 4. 문제 발생 원인 분석

### 왜 오리지널은 문제가 없었을까?

1. **Learning Rate Decay**
   - 오리지널: ExponentialDecay로 점진적으로 감소 (0.96^step)
   - 우리 (수정 전): 고정 learning rate 0.001 → 너무 높을 수 있음
   - 우리 (수정 후): ✅ ExponentialLR (gamma=0.96) 추가

2. **TensorFlow의 내부 안정성**
   - TensorFlow는 내부적으로 일부 수치 안정성 조치를 포함할 수 있음
   - PyTorch는 더 명시적인 제어가 필요
   - 우리는 명시적으로 gradient clipping, input clipping 추가

3. **Pre-training**
   - 오리지널: Missing 없는 데이터로 사전 학습하여 안정적인 초기화
   - 우리 (수정 전): Pre-training 없이 바로 MCMC 시작 → 불안정할 수 있음
   - 우리 (수정 후): ✅ Pre-training 추가

4. **Batch Size**
   - 오리지널: 100 (더 큰 batch = 더 안정적인 gradient)
   - 우리 (수정 전): 32 (작은 batch = 더 불안정할 수 있음)
   - 우리 (수정 후): ✅ 100으로 증가

5. **Activation Function**
   - 오리지널: ReLU (더 안정적, gradient가 0 또는 1)
   - 우리 (수정 전): Tanh (extreme inputs에서 saturation 가능)
   - 우리 (수정 후): ✅ ReLU로 변경

6. **NaN 처리**
   - 오리지널: TensorFlow의 `mse_missing`이 mask를 사용하여 NaN을 건너뜀
   - 우리 (수정 전): NaN을 0으로 대체 후 계속 학습 → NaN 전파
   - 우리 (수정 후): ✅ NaN batch 건너뛰기

### 우리 구현에서 발생한 문제

1. **NaN 전파 메커니즘**:
   ```
   Forward pass → NaN 발생 → 0으로 대체 → Loss 계산 → Backward pass
   → Gradient에 NaN 포함 → Weight update → Decoder weights가 NaN
   → C matrix 추출 시 100% NaN
   ```

2. **Gradient Explosion**:
   - Gradient clipping이 없어서 extreme gradients가 발생
   - Learning rate가 높아서 weight update가 과도함

3. **Input Data Quality**:
   - 전처리 과정에서 extreme values 생성
   - 클리핑 없이 encoder에 입력 → NaN 발생

## 5. 적용한 수정사항

### 수정 1: Gradient Clipping 추가
```python
# src/model/sktime_forecaster.py line 915
trainer_kwargs["gradient_clip_val"] = 1.0  # Default: clip gradients at 1.0
```
- **효과**: Gradient explosion 방지
- **오리지널과 비교**: 오리지널은 명시적 clipping 없음 (TensorFlow 기본 동작)

### 수정 2: Learning Rate 및 Scheduler
```python
# dfm-python/src/dfm_python/models/ddfm.py line 565
learning_rate: float = 0.005  # Matches original DDFM default

# dfm-python/src/dfm_python/models/ddfm.py line 1507
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, gamma=0.96  # Matches original: decay_rate=0.96
)
```
- **효과**: 더 안정적인 학습, 오리지널과 동일한 learning rate decay
- **오리지널과 비교**: ✅ 동일 (0.005 + ExponentialDecay)

### 수정 3: Input Data Clipping
```python
# dfm-python/src/dfm_python/models/ddfm.py line 849
data_clipped = torch.clamp(data, min=-10.0, max=10.0)
```
- **효과**: Extreme values로 인한 NaN 방지
- **오리지널과 비교**: 오리지널은 명시적 clipping 없음 (표준화만 사용)

### 수정 4: NaN Batch 건너뛰기
```python
# dfm-python/src/dfm_python/models/ddfm.py lines 855-868
if torch.any(torch.isnan(reconstructed)):
    # Skip batch instead of replacing with zeros
    return large_loss_without_grad
```
- **효과**: NaN이 decoder weights에 전파되는 것 방지
- **오리지널과 비교**: 오리지널은 TensorFlow가 내부적으로 처리

### 수정 5: Pre-training 추가
```python
# dfm-python/src/dfm_python/models/ddfm.py line 1565
def pre_train(self, X: torch.Tensor, x_clean: torch.Tensor, missing_mask: np.ndarray, ...):
    # Pre-train autoencoder on data without missing values
    # Matches original DDFM implementation's pre-training step
```
- **효과**: 안정적인 초기화
- **오리지널과 비교**: ✅ 동일하게 구현됨

### 수정 6: Activation Function 변경
```python
# dfm-python/src/dfm_python/config/schema.py line 715
activation: str = 'relu'  # Changed from 'tanh' to match original
```
- **효과**: 더 안정적인 학습
- **오리지널과 비교**: ✅ 동일 (ReLU)

### 수정 7: Batch Size 증가
```python
# dfm-python/src/dfm_python/config/schema.py line 719
batch_size: int = 100  # Changed from 32 to match original
```
- **효과**: 더 안정적인 gradient estimation
- **오리지널과 비교**: ✅ 동일 (100)

### 수정 8: NaN 허용 (Kalman filter 활용)
```python
# dfm-python/src/dfm_python/lightning/data_module.py
# Removed validate_no_nan() - DFM/DDFM can handle NaN via Kalman filter
```
- **효과**: DFM/DDFM이 Kalman filter로 implicit하게 NaN 처리
- **오리지널과 비교**: ✅ 동일 (오리지널도 NaN 허용, `mse_missing` 사용)

## 6. 개선 제안

### 추가 개선 가능 사항

1. **Weight Initialization 명시**
   - GlorotNormal/Xavier 초기화 명시적 설정
   - 오리지널과 동일한 초기화 방법 사용

2. **C Matrix 추출 검증**
   - 오리지널과 우리 구현의 emission matrix 구조 차이 확인
   - State-space 모델에서 올바르게 사용되는지 검증

## 7. 결론

### 핵심 차이점
1. **프레임워크**: TensorFlow vs PyTorch - 내부 수치 안정성 처리 방식 차이
2. **Learning Rate**: ✅ 오리지널과 동일 (0.005 + ExponentialDecay)
3. **Gradient Clipping**: 오리지널은 없음, 우리는 명시적으로 추가 (안정성 향상)
4. **Pre-training**: ✅ 오리지널과 동일하게 구현됨
5. **NaN 처리**: 오리지널은 프레임워크가 처리, 우리는 명시적 처리 필요
6. **Activation**: ✅ 오리지널과 동일 (ReLU)
7. **Batch Size**: ✅ 오리지널과 동일 (100)
8. **NaN 허용**: ✅ 오리지널과 동일 (Kalman filter로 처리)

### 해결된 문제
- ✅ C matrix NaN 문제 해결 (gradient clipping + learning rate + input clipping)
- ✅ NaN 전파 방지 (batch 건너뛰기)
- ✅ Target series 예측 문제 해결
- ✅ Pre-training 추가 (오리지널과 동일)
- ✅ Learning rate scheduler 추가 (오리지널과 동일)
- ✅ Activation function 변경 (오리지널과 동일)
- ✅ Batch size 증가 (오리지널과 동일)
- ✅ NaN 허용 (Kalman filter 활용, 오리지널과 동일)

### 현재 상태
우리 구현은 오리지널 DDFM과 거의 동일한 설정과 절차를 따르고 있으며, PyTorch의 특성상 필요한 명시적 안정성 조치(gradient clipping, input clipping)를 추가했습니다. 주요 차이점은 프레임워크 차이와 C matrix 추출 방식(transpose 여부)이며, 이는 state-space 모델 구성 방식의 차이에서 기인합니다.

# C Matrix 추출 방식 이론적 분석

## 1. State-Space 모델 구조

### Measurement Equation
```
z_t = H x_t + v_t,  v_t ~ N(0, R)
```
- `z_t`: 관측치 벡터 (N x 1), N은 시리즈 개수
- `x_t`: State 벡터 (state_dim x 1)
- `H`: Observation matrix (N x state_dim)
- `R`: Observation noise covariance (N x N)

### Transition Equation
```
x_t = F x_{t-1} + w_t,  w_t ~ N(0, Q)
```
- `F`: Transition matrix (state_dim x state_dim)
- `Q`: Process noise covariance (state_dim x state_dim)

### State Vector 구조 (VAR(1))
```
x_t = [f_t, eps_t]
```
- `f_t`: Common factors (m x 1), m은 factor 개수
- `eps_t`: Idiosyncratic components (N x 1)
- `state_dim = m + N`

## 2. 오리지널 구현 분석

### Decoder Weight 추출
```python
# DDFM/tools/getters_converters_tools.py
ws, bs = decoder.get_layer(index=-1).get_weights()
# ws shape: (N x m) - Decoder output_dim x input_dim
```

### Emission Matrix 구성
```python
# 오리지널 구현 (factor_order=1)
emission = np.hstack((
    ws.T,  # (m x N) - transpose!
    np.identity(m)  # (m x m)
))
# Result: (m x (N + m))
```

### 문제점

**차원 불일치**:
- Measurement equation: `z_t = H x_t`
- `z_t` shape: (N,)
- `x_t` shape: (m + N,)
- 따라서 `H`는 **(N x (m + N))** 형태여야 함
- 하지만 오리지널 구현의 `emission`은 **(m x (N + m))** 형태

**이론적으로 불가능**:
```python
z_t = H @ x_t
(N,) = (m x (N+m)) @ (m+N,)  # ❌ 차원 불일치!
```

### 오리지널 구현이 작동하는 이유 추정

1. **PyKalman의 자동 처리**: PyKalman이 `observation_matrices`를 자동으로 transpose할 수 있음
2. **State vector 구조 차이**: 실제 state vector가 다를 수 있음
3. **실제 사용 방식**: Emission matrix가 직접 사용되지 않고 다른 방식으로 변환될 수 있음

## 3. 우리 구현 분석

### Decoder Weight 추출
```python
# dfm-python/src/dfm_python/encoder/vae.py
weight = decoder_layer.weight.data.cpu().numpy()  # (N x m)
```

### Emission Matrix 구성
```python
# 우리 구현 (factor_order=1)
emission = np.hstack([
    weight,  # (N x m) - transpose 없음!
    np.eye(N)  # (N x N)
])
# Result: (N x (m + N))
```

### 이론적 타당성

**차원 일치**:
- Measurement equation: `z_t = H x_t`
- `z_t` shape: (N,)
- `x_t` shape: (m + N,)
- `H` shape: **(N x (m + N))** ✓

**이론적으로 올바름**:
```python
z_t = H @ x_t
(N,) = (N x (m+N)) @ (m+N,)  # ✓ 차원 일치!
```

## 4. 수학적 검증

### Measurement Equation 구조

DDFM의 measurement equation은:
```
z_t = C f_t + eps_t
```

여기서:
- `z_t`: 관측치 (N x 1)
- `C`: Loading matrix (N x m)
- `f_t`: Factors (m x 1)
- `eps_t`: Idiosyncratic (N x 1)

State-space 형태로 변환하면:
```
z_t = [C, I] [f_t, eps_t]^T
     = H x_t
```

따라서:
- `H = [C, I]` where `C` is (N x m), `I` is (N x N)
- `H` shape: **(N x (m + N))** ✓

### Decoder의 역할

Decoder는 factor를 관측치로 매핑:
```
z_t = Decoder(f_t) = C f_t + bias
```

여기서:
- `C`: Decoder weight matrix (N x m)
- `bias`: Decoder bias (N,)

따라서 `C`는 이미 (N x m) 형태이므로, transpose할 필요가 없음.

## 5. 결론

### 우리 구현의 이론적 타당성

✅ **이론적으로 올바름**:
- Measurement equation의 차원이 일치함
- State-space 모델 구조와 일치함
- Decoder의 역할과 일치함

### 오리지널 구현의 문제

❌ **이론적으로 차원 불일치**:
- Emission matrix가 (m x (N+m)) 형태로 잘못 구성됨
- 하지만 실제로 작동한다면 PyKalman이 자동으로 처리하거나 다른 방식으로 사용됨

### 권장사항

1. **우리 구현 유지**: 현재 구현이 이론적으로 올바름
2. **오리지널 구현 검증**: 오리지널 코드가 실제로 어떻게 사용되는지 확인 필요
3. **테스트**: 두 방식의 결과를 비교하여 실제 차이 확인

## 6. 참고: PyKalman의 observation_matrices

PyKalman 문서에 따르면:
- `observation_matrices`: (n_dim_obs x n_dim_state) 형태
- `n_dim_obs`: 관측치 차원 (N)
- `n_dim_state`: State 차원 (m + N)

따라서 `observation_matrices`는 **(N x (m + N))** 형태여야 함.

우리 구현이 이론적으로 올바르며, 오리지널 구현은 차원 불일치가 있지만 PyKalman이 자동으로 처리하거나 다른 방식으로 사용될 수 있음.

