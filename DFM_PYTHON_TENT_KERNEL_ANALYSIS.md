# dfm-python 패키지의 Tent Kernel 구현 분석

## 1. 설계 vs 실제 구현

### 1.1 설계상 지원하는 빈도 쌍

**`dfm-python/src/dfm_python/config/utils.py:200-209`**

```python
TENT_WEIGHTS_LOOKUP: Dict[Tuple[str, str], np.ndarray] = {
    # 저빈도 -> 고빈도 (올바른 방향)
    ('q', 'm'): np.array([1, 2, 3, 2, 1]),                    # 5 periods: 분기 -> 월간
    ('sa', 'm'): np.array([1, 2, 3, 4, 3, 2, 1]),             # 7 periods: 반기 -> 월간
    ('a', 'm'): np.array([1, 2, 3, 4, 5, 4, 3, 2, 1]),       # 9 periods: 연간 -> 월간
    ('sa', 'q'): np.array([1, 2, 1]),                         # 3 periods: 반기 -> 분기
    ('a', 'q'): np.array([1, 2, 3, 2, 1]),                    # 5 periods: 연간 -> 분기
    ('a', 'sa'): np.array([1, 2, 1]),                         # 3 periods: 연간 -> 반기
    
    # 고빈도 -> 저빈도 (⚠️ 설계 원칙과 맞지 않음)
    ('m', 'w'): np.array([1, 2, 3, 2, 1]),                    # 5 periods: 월간 -> 주간
    ('q', 'w'): np.array([1, 2, 3, 4, 5, 4, 3, 2, 1]),       # 9 periods: 분기 -> 주간
}
```

**설계상 지원:**
- ✅ 분기 → 월간 (q → m)
- ✅ 반기 → 월간 (sa → m)
- ✅ 연간 → 월간 (a → m)
- ✅ 반기 → 분기 (sa → q)
- ✅ 연간 → 분기 (a → q)
- ✅ 연간 → 반기 (a → sa)
- ⚠️ 월간 → 주간 (m → w) - 설계 원칙과 맞지 않음
- ⚠️ 분기 → 주간 (q → w) - 설계 원칙과 맞지 않음

### 1.2 실제 구현 (EM 알고리즘)

**`dfm-python/src/dfm_python/ssm/em.py:573-578`**

```python
# Determine tent kernel size (pC) for slower-frequency aggregation
pC = 5  # Default: quarterly to monthly uses 5 periods [1,2,3,2,1]
if R_mat is not None:
    pC = R_mat.shape[1]
elif tent_weights_dict is not None and 'q' in tent_weights_dict:
    pC = len(tent_weights_dict['q'])
```

**실제 구현:**
- ❌ **분기(quarterly)만 하드코딩됨**
- ❌ `nQ` (quarterly series count)만 사용
- ❌ `'q'` 키만 `tent_weights_dict`에서 확인
- ❌ 반기(sa), 연간(a)에 대한 처리가 없음

**`dfm-python/src/dfm_python/ssm/em.py:973-980`**

```python
# Add quarterly idiosyncratic chains (5-state: [1, 2, 3, 2, 1])
if nQ > 0:
    # Quarterly tent weights: [1, 2, 3, 2, 1]
    tent_q = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0], device=device, dtype=dtype)
    C_quarterly = torch.zeros(N, 5 * nQ, device=device, dtype=dtype)
    C_quarterly[nM:, :] = torch.kron(torch.eye(nQ, device=device, dtype=dtype), tent_q.unsqueeze(0))
    C = torch.cat([C, C_quarterly], dim=1)
```

**결론:**
- 실제 EM 알고리즘에서는 **분기(quarterly) → 월간(monthly)만 구현**되어 있음
- 다른 빈도 쌍(sa→m, a→m, sa→q, a→q, a→sa)은 **TENT_WEIGHTS_LOOKUP에 정의만 되어 있고 실제로 사용되지 않음**

---

## 2. 빈도 계층 구조

**`dfm-python/src/dfm_python/config/utils.py:112-119`**

```python
FREQUENCY_HIERARCHY: Dict[str, int] = {
    'd': 1,   # Daily (highest frequency)
    'w': 2,   # Weekly
    'm': 3,   # Monthly
    'q': 4,   # Quarterly
    'sa': 5,  # Semi-annual
    'a': 6    # Annual (lowest frequency)
}
```

**의미:**
- 숫자가 작을수록 높은 빈도 (더 자주 관측)
- 숫자가 클수록 낮은 빈도 (덜 자주 관측)

**설계 원칙:**
- Clock보다 **높은 빈도** (작은 숫자)는 **지원하지 않음**
- Clock보다 **낮은 빈도** (큰 숫자)는 tent kernel 사용

---

## 3. get_agg_structure 함수

**`dfm-python/src/dfm_python/config/utils.py:328-400`**

이 함수는 **설계상** 여러 빈도를 지원하도록 작성되었지만, **실제로는 EM 알고리즘에서 사용되지 않음**.

**함수 기능:**
- Config에서 모든 시리즈의 빈도를 확인
- Clock보다 낮은 빈도 시리즈에 대해 tent kernel 가중치 조회
- R_mat, q 제약 행렬 생성

**하지만:**
- EM 알고리즘의 `initialize_parameters`에서는 이 함수를 호출하지 않음
- 대신 `nQ` (quarterly count)만 사용하고 `'q'` 키만 확인

---

## 4. 월간 → 주간, 연간 → 분기 변환

### 4.1 월간 → 주간 (m → w)

**TENT_WEIGHTS_LOOKUP에 정의되어 있지만:**
- ⚠️ **설계 원칙 위반**: 고빈도 → 저빈도 변환
- ⚠️ **실제로 사용 불가**: Clock보다 높은 빈도는 에러 발생
- ⚠️ **코드에서도 사용 안 함**: EM 알고리즘에서 처리하지 않음

**결론:** 정의만 되어 있고 실제로는 사용할 수 없음

### 4.2 연간 → 분기 (a → q)

**TENT_WEIGHTS_LOOKUP에 정의되어 있지만:**
- ✅ **설계 원칙 준수**: 저빈도 → 고빈도 변환
- ❌ **실제로 구현 안 됨**: EM 알고리즘에서 `nQ`만 사용하고 연간 시리즈는 처리하지 않음

**결론:** 정의만 되어 있고 실제로는 사용할 수 없음

---

## 5. 실제 구현 요약

### 지원되는 빈도 변환

**실제로 작동하는 것:**
- ✅ **분기 → 월간 (q → m)**: 하드코딩되어 있음 (`pC=5`, `nQ` 사용)

**설계상 지원하지만 실제로 작동하지 않는 것:**
- ❌ 반기 → 월간 (sa → m): TENT_WEIGHTS_LOOKUP에만 정의
- ❌ 연간 → 월간 (a → m): TENT_WEIGHTS_LOOKUP에만 정의
- ❌ 반기 → 분기 (sa → q): TENT_WEIGHTS_LOOKUP에만 정의
- ❌ 연간 → 분기 (a → q): TENT_WEIGHTS_LOOKUP에만 정의
- ❌ 연간 → 반기 (a → sa): TENT_WEIGHTS_LOOKUP에만 정의

**설계 원칙 위반 (사용 불가):**
- ❌ 월간 → 주간 (m → w): Clock보다 높은 빈도는 에러 발생
- ❌ 분기 → 주간 (q → w): Clock보다 높은 빈도는 에러 발생

---

## 6. 왜 이런 상황인가?

### 가능한 원인

1. **MATLAB 코드 포팅 과정**
   - 원본 MATLAB 코드가 분기 데이터만 가정
   - 포팅 과정에서 분기만 하드코딩
   - 다른 빈도는 나중에 추가하려다 미완성

2. **점진적 개발**
   - 초기 버전은 분기만 지원
   - TENT_WEIGHTS_LOOKUP은 확장성을 위해 미리 정의
   - 하지만 실제 구현은 아직 완료되지 않음

3. **get_agg_structure 함수의 미사용**
   - 함수는 작성되었지만 EM 알고리즘에 통합되지 않음
   - `nQ` 기반의 단순한 로직만 사용

---

## 7. 결론

**현재 dfm-python 패키지의 tent kernel 구현:**

1. **실제로 작동하는 것:**
   - 분기 → 월간 (q → m)만 지원

2. **설계상 지원하지만 작동하지 않는 것:**
   - 반기 → 월간, 연간 → 월간, 반기 → 분기, 연간 → 분기, 연간 → 반기
   - TENT_WEIGHTS_LOOKUP에 정의만 되어 있고 실제로 사용되지 않음

3. **설계 원칙 위반:**
   - 월간 → 주간, 분기 → 주간은 정의되어 있지만 사용 불가 (clock보다 높은 빈도)

4. **현재 프로젝트의 문제:**
   - 모든 데이터가 월간인데도 `pC=5` 기본값이 적용됨
   - 분기 시리즈가 없어도 상태 공간이 5배로 확장됨

---

## 8. 개선 방향

### 옵션 1: 현재 구현 활용 (분기만)

- `tent_weights_dict: {q: [1.0]}` 설정으로 `pC=1`로 변경
- 월간 전용 데이터에 대해 상태 공간 확장 방지

### 옵션 2: get_agg_structure 함수 활용 (미완성)

- `get_agg_structure` 함수를 EM 알고리즘에 통합
- 여러 빈도를 지원하도록 확장
- 하지만 현재는 구현되지 않음

### 옵션 3: 라이브러리 수정 (근본적 해결)

- `nQ=0`일 때 `pC=p`로 자동 설정
- 또는 `get_agg_structure`를 실제로 사용하도록 수정

