# DFM 수치적 불안정성 이슈

## 요약

DFM (Dynamic Factor Model) 훈련에서 시리즈 개수가 33개 이상인 경우 수치적 불안정성으로 실패합니다. KOEQUIPTE (32개 시리즈)는 성공했으나, KOIPALL.G (33개)와 KOWRCCNSE (39개)는 실패했습니다. DDFM, ARIMA, VAR은 모든 실험에서 성공합니다.

### 모델별 성공/실패 현황

| 모델 | KOEQUIPTE | KOIPALL.G | KOWRCCNSE |
|------|-----------|-----------|-----------|
| **DFM** | ✅ 성공 (32 시리즈) | ❌ 실패 (33 시리즈) | ❌ 실패 (39 시리즈) |
| **DDFM** | ✅ 성공 (32 시리즈) | ✅ 성공 (33 시리즈) | ✅ 성공 (39 시리즈) |
| **ARIMA** | ✅ 성공 (41 시리즈) | ✅ 성공 (40 시리즈) | ✅ 성공 (47 시리즈) |
| **VAR** | ✅ 성공 (41 시리즈) | ✅ 성공 (40 시리즈) | ✅ 성공 (47 시리즈) |

### 핵심 발견

- **임계값**: 시리즈 개수 ≥ 33개 또는 시리즈/state space 비율 > 2.13일 때 불안정
- **실패 메커니즘**: F matrix inversion 불안정 → Kalman filter 연쇄 실패 → sum_EZZ eigendecomposition 실패 → C matrix NaN
- **해결책**: 시리즈 개수 제한 (≤32) 또는 DDFM 사용
- **C Matrix 이슈 해결 방안**: 더 강한 사전 Regularization (Ridge Regression 방식) - eigendecomposition 실패 시에도 적용 가능

## 증상

### KOEQUIPTE (성공 사례)

로그 파일: `log/KOEQUIPTE_dfm_20251207_202911.log`

- EM algorithm이 112 iterations까지 실행
- Log-likelihood 변화:
  - Iteration 1: -3030.9163
  - Iteration 10: 88.9470 (최고값)
  - Iteration 112: 조기 중단 (log-likelihood 1014.2만큼 악화)
- NaN/Inf 경고 없음
- 모델이 정상적으로 저장됨

### KOIPALL.G (실패 사례)

로그 파일: `log/KOIPALL.G_dfm_20251207_203509.log`

- EM algorithm이 4 iterations만 실행 후 조기 수렴
- Iteration 1에서 즉시 실패:
  - Log-likelihood: 1429.6386 (초기값)
  - Kalman filter forward pass에서 t=13부터 NaN/Inf 발생
    - `Vu` matrix에 NaN/Inf (t=13부터)
    - `V` matrix에 Inf (t=14부터)
    - `F` matrix에 NaN/Inf (t=26부터)
- Iteration 1에서 `sum_EZZ` eigvalsh 수렴 실패:
  - Error code 49: "too many repeated eigenvalues"
  - Condition number 계산 불가
  - Adaptive regularization 적용 (base_reg_scale * 10)
- C matrix 업데이트 실패:
  - 1584/1584 (100.0%) NaN
  - `torch.linalg.solve` 실패
- Q matrix eigvalsh 수렴 실패 (error code 49)
- 최종 log-likelihood: -3102.04 (초기값에서 악화)

### KOWRCCNSE (실패 사례)

로그 파일: `log/KOWRCCNSE_dfm_20251207_203243.log`

- EM algorithm이 4 iterations만 실행 후 조기 수렴
- Iteration 1에서 즉시 실패:
  - Log-likelihood: -2471.9383 (초기값)
  - Kalman filter forward pass에서 t=25부터 NaN/Inf 발생
    - `F` matrix에 Inf (t=25부터)
    - `Vu` matrix에 NaN/Inf (t=25부터)
    - `V` matrix에 Inf (t=46부터)
- Iteration 1에서 `sum_EZZ` eigvalsh 수렴 실패:
  - Error code 55: "ill-conditioned matrix"
  - Condition number 계산 불가
  - Adaptive regularization 적용 (base_reg_scale * 10)
- C matrix 업데이트 실패:
  - 1998/2106 (94.9%) NaN
  - `torch.linalg.solve` 실패
- Q matrix eigvalsh 수렴 실패 (error code 55)
- 최종 log-likelihood: -2671.57 (초기값에서 악화)

## 데이터 품질 분석

### 시리즈 개수 (필터링 후)

로그에서 확인된 실제 사용 시리즈 개수:
- **KOEQUIPTE**: 32개 시리즈 (config: 41개, 주파수 불일치로 9개 제외)
- **KOIPALL.G**: 33개 시리즈 (config: 40개, 주파수 불일치로 7개 제외)
- **KOWRCCNSE**: 39개 시리즈 (config: 47개, 주파수 불일치로 8개 제외)

### Missing Data 비율

- **KOEQUIPTE**: 전체 79.9%, 최대 91.8% (A001)
- **KOIPALL.G**: 전체 80.7%, 최대 91.8% (A001)
- **KOWRCCNSE**: 전체 81.2%, 최대 93.7% (KOCSEOHDR)

### Effective Sample Size

- **KOEQUIPTE**: 평균 440.1개 관측치 (최소 179, 최대 1826)
- **KOIPALL.G**: 평균 420.6개 관측치 (최소 179, 최대 1826)
- **KOWRCCNSE**: 평균 410.6개 관측치 (최소 137, 최대 1826)

### 시리즈 상관관계

- **KOEQUIPTE**: 평균 절대 상관관계 0.138, 높은 상관관계 (>0.9) 없음
- **KOIPALL.G**: 평균 절대 상관관계 0.156, 높은 상관관계 1개
- **KOWRCCNSE**: 평균 절대 상관관계 0.148, 높은 상관관계 2개

### 분산 불균형

- **KOEQUIPTE**: 분산 CV 3.66
- **KOIPALL.G**: 분산 CV 3.15
- **KOWRCCNSE**: 분산 CV 5.83

## 근본 원인

### 1. State Space 차원

코드에서 확인된 state space 차원 계산:
- Factors: 3개
- AR lag: 1
- Quarterly-to-monthly aggregation: `ppC=5` (tent kernel)
- **실제 state space 차원: 15차원** (로그에서 확인: "Block 0: r_i=3, ppC=5, expected size=15")

시리즈 개수와 state space 차원의 비율:
- **KOEQUIPTE**: 32 시리즈 / 15 차원 = **2.13**
- **KOIPALL.G**: 33 시리즈 / 15 차원 = **2.20**
- **KOWRCCNSE**: 39 시리즈 / 15 차원 = **2.60**

### 2. Kalman Filter의 F Matrix 불안정성

코드 위치: `dfm-python/src/dfm_python/ssm/kalman.py:370-395`

**수학적 관계**:
- `F = C_t @ V @ C_t.T + R_t`
- `C_t`: (n_obs, 15) - t 시점에서 관측된 시리즈들의 loading matrix
- `V`: (15, 15) - factor covariance (prior)
- `R_t`: (n_obs, n_obs) - t 시점에서 관측된 시리즈들의 measurement error covariance
- `F`: (n_obs, n_obs) - innovation covariance

**F Matrix 크기와 시리즈 개수의 관계**:
- `n_obs`는 각 시간 t에서 실제로 관측된 시리즈 개수 (missing data 제외)
- 시리즈 개수가 많을수록 `n_obs`가 커질 가능성 증가
- 특히 missing data가 적을 때 `n_obs ≈ 시리즈 개수`
- `F` matrix 크기 = `n_obs × n_obs`

**문제 메커니즘**:
1. **F Matrix Inversion 불안정성**:
   - `iF = safe_inv(F, regularization=adaptive_reg)` (코드: `dfm-python/src/dfm_python/ssm/kalman.py:395`)
   - Matrix inversion은 O(n_obs³) 연산
   - `n_obs`가 클수록 수치적 불안정성 증가
   - 작은 수치 오차도 큰 행렬에서는 증폭됨

2. **연쇄 반응**:
   - F matrix inversion 실패 → `VCF` (Kalman gain) 손상
   - `VCF` 손상 → `Vu` (posterior covariance) 손상
   - `Vu` 손상 → 다음 시간의 `V` (prior covariance) 손상
   - `V` 손상 → 다음 시간의 `F` 손상

**로그에서 확인된 사실** (검증됨):
- **KOEQUIPTE**: 
  - 시리즈 개수: 32개 (로그: "Series: 32 (filtered from config)")
  - NaN/Inf 발생 없음
  - F matrix 실패 없음
- **KOIPALL.G**: 
  - 시리즈 개수: 33개 (로그: "Series: 33 (filtered from config)")
  - t=13: `Vu` matrix에 NaN/Inf 발생 (로그: "Vu at t=13 contains 24 NaN values and 82 Inf values")
  - t=14: `V` matrix에 Inf 발생 (로그: "V at t=14 contains 15 NaN values and 1106 Inf values"), `F` matrix에 Inf 발생 (로그: "F at t=14 contains 5 Inf values"), `missing_ratio=0.00` (로그: "missing_ratio=0.00")
  - `missing_ratio=0.00`이므로 `n_obs = 33` (모든 시리즈 관측)
  - F matrix 크기: (33, 33)
  - t=14 이후 연쇄적으로 NaN/Inf 전파
- **KOWRCCNSE**: 
  - 시리즈 개수: 39개 (로그: "Series: 39 (filtered from config)")
  - t=25: `F` matrix에 Inf 발생 (로그: "F at t=25 contains 73 Inf values"), `missing_ratio=0.00` (로그: "missing_ratio=0.00")
  - `missing_ratio=0.00`이므로 `n_obs = 39` (모든 시리즈 관측)
  - F matrix 크기: (39, 39)
  - t=25 이후 연쇄적으로 NaN/Inf 전파

**관찰된 패턴** (검증됨):
- KOEQUIPTE (32개 시리즈, F: ≤32×32): 안정적 ✅
- KOIPALL.G (33개 시리즈, F: 33×33): 불안정 ❌
- KOWRCCNSE (39개 시리즈, F: 39×39): 불안정 ❌

**주의**: 현재 3개 실험만 있으므로, "33×33 이상일 때 불안정"이라는 임계값은 관찰된 패턴이며, 더 많은 실험으로 검증이 필요합니다.

### 3. sum_EZZ 행렬의 Ill-conditioning

코드 위치: `dfm-python/src/dfm_python/ssm/em.py:244-286`

**수학적 관계**:
- `sum_EZZ = sum_t E[Z_t Z_t^T]`: (15, 15) - smoothed factors의 공분산 행렬 합
- `sum_yEZ = sum_t y_t E[Z_t^T]`: (N, 15) - 시리즈와 factors의 교차공분산 행렬 합
- `C_new = solve(sum_EZZ_reg.T, sum_yEZ.T).T`: (N, 15) - C matrix 업데이트

**문제 메커니즘**:
1. **Kalman filter 손상으로 인한 EZZ 손상**:
   - F matrix 불안정성으로 인해 Kalman filter forward pass에서 NaN/Inf 발생
   - 손상된 Kalman filter로 인해 smoothed factors (`EZ`, `EZZ`)가 손상됨
   - 손상된 `EZZ`로 인해 `sum_EZZ`가 ill-conditioned가 됨

2. **sum_EZZ Eigendecomposition 실패**:
   - KOIPALL.G: error code 49 (too many repeated eigenvalues)
   - KOWRCCNSE: error code 55 (ill-conditioned matrix)
   - Condition number 계산 불가

3. **Adaptive Regularization의 한계**:
   - Eigendecomposition 실패 시: `reg_scale = base_reg_scale * 10` (코드: `dfm-python/src/dfm_python/ssm/em.py:278`)
   - 이 regularization이 충분하지 않아 `torch.linalg.solve` 실패
   - C matrix가 NaN으로 채워짐

**코드에서 확인된 사실**:
- Adaptive regularization threshold: 1e8 (코드: `dfm-python/src/dfm_python/ssm/em.py:261`)
- Eigendecomposition 실패 시 fallback: `reg_scale = base_reg_scale * 10` (코드: `dfm-python/src/dfm_python/ssm/em.py:278`)
- 이 regularization이 충분하지 않아 `torch.linalg.solve(sum_EZZ_reg.T, sum_yEZ.T)` 실패

### 4. 연쇄 반응 (Cascade Failure)

**전체 실패 경로**:

1. **Kalman Filter Forward Pass** (코드: `dfm-python/src/dfm_python/ssm/kalman.py:350-450`)
   - `F = C_t @ V @ C_t.T + R_t` 계산
   - `F` matrix inversion 실패 (`iF = safe_inv(F)`)
   - `VCF = VC @ iF` (Kalman gain) 계산 시 NaN/Inf 발생
   - `Vu = V - VCF @ VC.T` (posterior covariance) 업데이트 시 NaN/Inf 전파
   - 다음 시간의 `V = A @ Vu @ A.T + Q` (prior covariance) 계산 시 NaN/Inf 전파

2. **Kalman Filter Backward Pass (Smoother)** (코드: `dfm-python/src/dfm_python/ssm/kalman.py:500-600`)
   - 손상된 forward pass 결과로 인해 `VmT` recursion에서 NaN/Inf 전파
   - Smoothed factors (`EZ`, `EZZ`)가 손상됨
   - 로그에서 확인: `smoother_backward: VmT[:, :, t] recursion produced NaN/Inf at t=13`

3. **EM M-step** (코드: `dfm-python/src/dfm_python/ssm/em.py:230-320`)
   - 손상된 `EZZ`로 인해 `sum_EZZ = sum_t EZZ` 계산 실패
   - `sum_EZZ` eigendecomposition 실패 (error code 49 또는 55)
   - `C_new = solve(sum_EZZ_reg.T, sum_yEZ.T).T` 실패
   - C matrix가 NaN으로 채워짐

4. **결과**
   - C matrix가 NaN으로 채워짐 (KOIPALL.G: 100%, KOWRCCNSE: 94.9%)
   - 다음 iteration에서 더 심각한 불안정성 발생
   - EM algorithm이 조기 수렴 (4 iterations)

**로그에서 확인된 연쇄 반응**:
- **KOIPALL.G**: t=13 (Vu 손상) → t=14 (V, F 손상) → t=26 (F 대량 손상) → iteration 1에서 sum_EZZ 실패
- **KOWRCCNSE**: t=25 (F 손상) → t=46 (V 손상) → iteration 1에서 sum_EZZ 실패

## 해결 방안

### 단기 해결책 (즉시 적용 가능, 검증됨) ✅

1. **시리즈 개수 제한**: 32개 이하로 제한 (시리즈/state space 비율 ≤ 2.13)
   - **검증**: KOEQUIPTE (32개) 성공, KOIPALL.G (33개) 실패

2. **DDFM 사용**: 시리즈 개수가 33개 이상인 경우 DDFM 사용
   - **검증**: KOIPALL.G (33개), KOWRCCNSE (39개) 모두 DDFM에서 성공

### C Matrix 이슈 해결 방안 (우선순위별)

**현재 문제**:
- `sum_EZZ` eigendecomposition 실패 (error code 49, 55) → condition number 계산 불가
- Eigendecomposition 실패 시 `reg_scale * 10`만 사용 (충분하지 않음)
- Pseudo-inverse fallback도 SVD convergence failure 발생
- C matrix에 66.7-97.0% NaN 발생

**원본 MATLAB 분석**:
- Line 470, 517, 711: 직접 `inv()` 사용 (MATLAB의 내부 수치적 안정성 처리 가능)
- Smoother에서만 `pinv()` 사용 (line 1039, 1064)
- Constraint 처리에서도 `inv()` 사용 (line 715)

**지식베이스 검색 결과**:
- Cholesky decomposition이 ridge regression에서 더 robust
- Regularization (`lambda * eye`)이 중요
- Pseudo-inverse는 일반적인 방법이지만 SVD 실패 시 문제

#### 1. 더 강한 사전 Regularization (Ridge Regression 방식) ⭐⭐⭐ **가장 적절한 대안**

**목적**: Eigendecomposition 실패 시에도 적용 가능한 regularization

**방법**:
- `sum_EZZ` 계산 직후, eigendecomposition 전에 더 큰 diagonal 추가
- Eigendecomposition 실패 시: `reg_scale = max(base_reg_scale * 10, 1e-3)`
- Ridge regression 방식: `sum_EZZ_reg = sum_EZZ + torch.eye(m) * reg_scale`

**장점**:
- Eigendecomposition 실패 시에도 적용 가능 (핵심!)
- 구현 간단
- Ridge regression 방식과 일치 (지식베이스 검색 결과)
- 수치적으로 안정적

**단점**:
- Bias 증가 가능 (하지만 NaN보다는 나음)

**구현 위치**: `em.py:283-292` (eigendecomposition 실패 시 fallback 개선)

#### 2. C Matrix NaN 처리 개선 ⭐⭐

**목적**: NaN 발생 시 이전 iteration의 정보 보존

**방법**:
- 현재: NaN columns를 0으로 설정
- 개선: NaN columns를 이전 iteration의 값으로 유지
- `C_new[nan_mask] = C_old[nan_mask]`

**장점**:
- 이전 iteration의 정보 보존
- 점진적 개선 가능
- EM algorithm의 점진적 수렴 특성 활용

**단점**:
- 근본 원인 해결은 아님 (하지만 실용적)

**구현 위치**: `em.py:307-320` (NaN 처리 로직 개선)

#### 3. Cholesky Decomposition 기반 Solver ⭐

**목적**: 더 robust한 linear solver

**방법**:
- `torch.linalg.cholesky()` + forward/backward substitution
- Regularization으로 positive definite 보장

**장점**:
- 수치적으로 더 안정적 (지식베이스 검색 결과)
- Ridge regression에서 검증된 방법

**단점**:
- Positive definite 필요 (regularization으로 보장 가능)
- 구현 복잡

**구현 위치**: `em.py:297-305` (solve 대신 cholesky 사용)

#### 4. SVD 기반 직접 계산

**목적**: Pseudo-inverse 대신 직접 SVD로 계산하고 작은 singular value 제거

**방법**:
- `torch.linalg.svd()` + thresholding
- 작은 singular value 제거 후 재구성

**장점**:
- 더 robust할 수 있음

**단점**:
- SVD도 convergence failure 가능
- 구현 복잡

**권장 구현 순서**:
1. 더 강한 사전 Regularization (즉시 적용) ⭐⭐⭐
2. C Matrix NaN 처리 개선 (즉시 적용) ⭐⭐
3. Cholesky Decomposition (필요시) ⭐

### 중기 해결책 (원본 MATLAB 구현 참고)

#### ✅ 구현 완료 (효과 제한적)

1. **Maximum Eigenvalue Capping**: F, V, Vu, sum_EZZ에 적용
   - **효과**: 일부 개선 (KOIPALL.G: 100% → 97%, KOWRCCNSE: 94.9% → 66.7%)
   - **한계**: sum_EZZ eigendecomposition 실패 시 적용 불가

2. **R Matrix 최소값 강화**: `1e-04` 적용 (원본 MATLAB 호환) ✅
3. **대칭성 강제**: 모든 covariance matrix에 적용 ✅

#### 🔄 우선 적용 권장 (원본 MATLAB 분석 기반)

1. **Pseudo-inverse 사용** (우선순위 1)
   - **원본 MATLAB**: Smoother에서 `pinv()` 사용 (line 1039, 1064)
   - **적용**: `torch.linalg.solve()` 실패 시 `torch.linalg.pinv()` fallback
   - **위치**: `em.py:293` (C matrix 업데이트)

2. **더 강한 사전 Regularization** (우선순위 2)
   - **목적**: sum_EZZ eigendecomposition 성공률 향상
   - **방법**: sum_EZZ 계산 직후, eigendecomposition 전에 더 강한 regularization 적용
   - **위치**: `em.py:252` (sum_EZZ 계산 후)

3. **Trace 기반 Scale Factor** (우선순위 3)
   - **목적**: Eigendecomposition 실패 시에도 capping 적용
   - **방법**: Trace를 기반으로 대략적인 scale factor 계산
   - **위치**: `em.py:252` (eigendecomposition 실패 시 fallback)

4. **F Matrix Regularization 강화** (n_obs 기반) ✅ **구현 완료**
   - **목적**: n_obs ≥ 30일 때 더 강한 regularization (generic threshold)
   - **방법**: 시리즈 개수에 비례하여 regularization 강화
   - **위치**: `kalman.py:369-376` (F matrix inversion 전)
   - **구현**: `n_obs_factor = 1.0 + (n_obs - 29) * 0.1` (30 → 1.1x, 33 → 1.4x, 39 → 2.0x)

#### 선택적 개선

5. **C Matrix Value Clipping**: 필요시에만 적용 (일반적으로 불필요)
6. **State Space 차원 조정**: 모델 구조 변경이므로 신중히 검토 필요

### Clipping/Capping 접근법 요약

**현재 적용된 Clipping/Capping** ✅:
- **A matrix**: `torch.clamp(A, min=-0.99, max=0.99)` - AR 계수 안정성
- **R matrix**: `torch.clamp(diag_R, min=1e-4, max=1e4)` - MATLAB 호환
- **Q matrix**: 최소 variance 강제
- **F, V, Vu, sum_EZZ**: Maximum eigenvalue capping (1e6)

**효과**: 일부 개선되었으나 eigendecomposition 실패 시 적용 불가

**장단점**:
- **장점**: 수치적 안정성 향상, NaN/Inf 전파 방지
- **단점**: Bias 증가 가능, eigendecomposition 실패 시 무효

### 장기 해결책

1. **Robust Kalman Filter**: F matrix inversion 실패 시 더 robust한 fallback
2. **Adaptive State Space Dimension**: 시리즈 개수에 따라 자동 조정
3. **Series Selection/Filtering**: 시리즈 개수가 많을 때 중요 시리즈만 선택
4. **Block-wise Processing**: 큰 block을 작은 sub-block으로 분할 처리

## 참고 자료

### 로그 파일

**DFM 로그**:
- `log/KOEQUIPTE_dfm_20251207_202911.log`: 성공 사례
- `log/KOIPALL.G_dfm_20251207_203509.log`: 실패 사례 (C matrix 100% NaN)
- `log/KOWRCCNSE_dfm_20251207_203243.log`: 실패 사례 (C matrix 94.9% NaN)

**DDFM 로그** (모두 성공):
- `log/KOEQUIPTE_ddfm_20251207_203136.log`: 성공 (MCMC 200 iterations)
- `log/KOIPALL.G_ddfm_20251207_203520.log`: 성공 (MCMC 8 iterations 수렴)
- `log/KOWRCCNSE_ddfm_20251207_203416.log`: 성공 (MCMC 200 iterations)

### 관련 코드

- `dfm-python/src/dfm_python/ssm/em.py:244-286`: `sum_EZZ` 계산 및 C matrix 업데이트
- `dfm-python/src/dfm_python/ssm/kalman.py:350-450`: Kalman filter forward pass, F matrix 계산
- `dfm-python/src/dfm_python/models/dfm.py`: DFM 모델 및 early stopping

### 설정 파일

- `config/experiment/investment_koequipte_report.yaml`: KOEQUIPTE 설정
- `config/experiment/production_koipallg_report.yaml`: KOIPALL.G 설정
- `config/experiment/consumption_kowrccnse_report.yaml`: KOWRCCNSE 설정
- `config/model/dfm.yaml`: DFM 모델 설정 (Block_Global, 3 factors)

### 데이터 품질 분석 결과

- `data_quality_analysis.json`: 각 실험의 missing data 비율 및 상관관계 분석 결과
- `detailed_analysis.json`: 시리즈/state space 비율, effective sample size 등 상세 분석 결과
- `series_characteristics.json`: 각 시리즈별 통계 (평균, 표준편차, 분산, 상관관계 등)

## 최신 실험 결과

### 실험 3: 더 강한 사전 Regularization 적용 (2025-12-07 21:34)

#### 구현 완료 ✅
- **더 강한 사전 Regularization (Ridge Regression 방식)**: `em.py:283-292`
  - Eigendecomposition 실패 시: `reg_scale = max(base_reg_scale * 10, 1e-3)`
  - All eigenvalues near-zero 시: `reg_scale = max(base_reg_scale * 100, 1e-3)`
  - 최소값 1e-3 보장 (base_reg_scale이 작은 경우에도 안정성 확보)

#### 실험 결과
- **KOIPALL.G**: 
  - 이전 (Pseudo-inverse): C matrix 97.0% NaN, 4 iterations, loglik: -3102.04
  - 최신 (더 강한 Regularization): C matrix 97.0% NaN, 4 iterations, loglik: -3102.04
  - **결과**: reg_scale = 1.00e-03 적용 확인 ✅, 하지만 C matrix NaN 비율 개선 없음 ❌
- **KOWRCCNSE**: 
  - 이전 (Pseudo-inverse): C matrix 66.7% NaN, 4 iterations, loglik: -2671.57
  - 최신 (더 강한 Regularization): C matrix 66.7% NaN, 4 iterations, loglik: -2671.57
  - **결과**: reg_scale = 1.00e-03 적용 확인 ✅, 하지만 C matrix NaN 비율 개선 없음 ❌

#### 분석
- ✅ **성공 사항**: 더 강한 regularization이 정상적으로 적용됨 (최소값 1e-3 보장)
- ❌ **여전한 문제**: 
  1. C matrix NaN 비율이 개선되지 않음 (KOIPALL.G: 97.0%, KOWRCCNSE: 66.7%)
  2. 1e-3 regularization도 충분하지 않음
  3. Kalman filter forward pass에서 NaN/Inf 전파가 근본 원인

#### 결론
더 강한 사전 Regularization만으로는 C matrix 이슈 해결 불가. Kalman filter forward pass NaN/Inf 전파가 근본 원인이므로, 추가 개선 방안 필요:
1. C Matrix NaN 처리 개선 (이전 값 유지) ⭐⭐
2. Kalman filter forward pass NaN/Inf 전파 근본 원인 해결
3. 더 강한 regularization (1e-3보다 더 큰 값, 예: 1e-2) 고려

### 실험 2: F Matrix Regularization 강화 + Pseudo-inverse 적용 (2025-12-07 21:26)

#### 구현 완료 ✅
- **F Matrix Regularization 강화 (n_obs 기반)**: `kalman.py:369-376`
  - n_obs ≥ 30일 때 자동으로 regularization 강화 (generic threshold)
  - n_obs_factor = 1.0 + (n_obs - 29) * 0.1 (30 → 1.1x, 33 → 1.4x, 39 → 2.0x)
- **Pseudo-inverse 사용 (solve 실패 시)**: `em.py:293-300`
  - `torch.linalg.solve()` 실패 시 `torch.linalg.pinv()` fallback
  - 원본 MATLAB 방식 (smoother에서 `pinv()` 사용)

#### 실험 결과
- **KOIPALL.G**: 
  - 이전: C matrix 100% NaN, 4 iterations, loglik: -3102.04
  - 최신: C matrix 97.0% NaN, 4 iterations, loglik: -3102.04
  - **결과**: 약간 개선 (NaN 3% 감소), Pseudo-inverse fallback 2회 사용 ✅
- **KOWRCCNSE**: 
  - 이전: C matrix 94.9% NaN, 4 iterations, loglik: -2671.57
  - 최신: C matrix 66.7% NaN, 4 iterations, loglik: -2671.57
  - **결과**: 개선 (NaN 28.2% 감소), Pseudo-inverse fallback 2회 사용 ✅

### 실험 1: Maximum Eigenvalue Capping 적용 (2025-12-07)

#### 구현 완료 ✅
- `cap_max_eigenval()` 함수 구현 및 F, V, Vu, sum_EZZ에 적용
- 코드 위치: `kalman.py:347,382,438`, `em.py:252`

#### 실험 결과
- **KOIPALL.G**: C matrix NaN 100% → 97.0% (약간 개선, 여전히 실패)
- **KOWRCCNSE**: C matrix NaN 94.9% → 66.7% (개선, 여전히 실패)

#### 문제점
1. **sum_EZZ eigendecomposition 실패**: 행렬이 너무 ill-conditioned여서 eigendecomposition 자체가 실패 (error code 49, 55). 이 경우 capping이 적용되지 않음.
2. **Kalman filter NaN/Inf 전파**: F, V, Vu에 capping 적용 후에도 forward pass에서 NaN/Inf 전파 발생.

## 이전 실험 결과 (2025-12-07, Maximum Eigenvalue Capping 적용 후)

### 구현 완료 ✅
- `cap_max_eigenval()` 함수 구현 및 F, V, Vu, sum_EZZ에 적용
- 코드 위치: `kalman.py:347,382,438`, `em.py:252`

### 실험 결과
- **KOIPALL.G**: C matrix NaN 100% → 97.0% (약간 개선, 여전히 실패)
- **KOWRCCNSE**: C matrix NaN 94.9% → 66.7% (개선, 여전히 실패)

### 문제점
1. **sum_EZZ eigendecomposition 실패**: 행렬이 너무 ill-conditioned여서 eigendecomposition 자체가 실패 (error code 49, 55). 이 경우 capping이 적용되지 않음.
2. **Kalman filter NaN/Inf 전파**: F, V, Vu에 capping 적용 후에도 forward pass에서 NaN/Inf 전파 발생.

### 결론
Maximum eigenvalue capping만으로는 문제 해결 불가. 추가 개선 방안 필요:
- Eigendecomposition 실패 시 대안 (trace 기반 scale factor, 더 강한 사전 regularization)
- Pseudo-inverse 사용 (원본 MATLAB: smoother에서 `pinv()` 사용)
- Kalman filter forward pass NaN/Inf 전파 근본 원인 해결

## 주요 발견 사항

### 관찰된 패턴 (3개 실험 기준)
- **시리즈 개수 임계값**: 32개 이하 안정적, 33개 이상 불안정
- **F Matrix 크기**: n_obs ≥ 33일 때 inversion 불안정
- **시리즈/state space 비율**: ≤ 2.13 안정적, > 2.13 불안정

**주의**: 3개 실험만 있으므로, 더 많은 실험으로 검증 필요

### 실패 메커니즘 (8단계 연쇄 반응)
1. F matrix inversion 불안정 (n_obs ≥ 33)
2. Kalman gain (VCF) 손상
3. Posterior covariance (Vu) 손상
4. Prior covariance (V) 손상
5. 다음 시간의 F matrix 손상
6. Smoothed factors (EZ, EZZ) 손상
7. sum_EZZ ill-conditioning
8. C matrix 업데이트 실패

### 모델별 특성
- **DDFM**: Neural network 기반으로 EM algorithm 행렬 연산 문제 회피 → 더 robust
- **ARIMA/VAR**: State space representation 없음 → 시리즈 개수에 무관하게 안정적
