# EM 알고리즘 구현 문제 및 설계 원칙 위반 원인 분석

## 1. 설계 원칙 위반이 발생한 이유

### 1.1 TENT_WEIGHTS_LOOKUP에 고빈도→저빈도가 정의된 이유

**코드 위치:** `dfm-python/src/dfm_python/config/utils.py:200-209`

```python
TENT_WEIGHTS_LOOKUP: Dict[Tuple[str, str], np.ndarray] = {
    # 저빈도 -> 고빈도 (올바른 방향)
    ('q', 'm'): np.array([1, 2, 3, 2, 1]),
    ('sa', 'm'): np.array([1, 2, 3, 4, 3, 2, 1]),
    ('a', 'm'): np.array([1, 2, 3, 4, 5, 4, 3, 2, 1]),
    ('sa', 'q'): np.array([1, 2, 1]),
    ('a', 'q'): np.array([1, 2, 3, 2, 1]),
    ('a', 'sa'): np.array([1, 2, 1]),
    
    # 고빈도 -> 저빈도 (⚠️ 설계 원칙 위반)
    ('m', 'w'): np.array([1, 2, 3, 2, 1]),      # 월간 -> 주간
    ('q', 'w'): np.array([1, 2, 3, 4, 5, 4, 3, 2, 1]),  # 분기 -> 주간
}
```

**왜 이런 정의가 있는가?**

1. **"완전성"을 위한 과도한 정의**
   - 초기 설계 단계에서 "모든 가능한 빈도 쌍"을 미리 정의하려는 의도
   - 나중에 필요할 수도 있다는 생각으로 포함
   - 하지만 실제 사용 여부를 검증하지 않음

2. **get_agg_structure 함수의 올바른 구현**
   - `get_agg_structure` 함수는 올바르게 구현되어 있음:
   ```python
   if FREQUENCY_HIERARCHY.get(freq, 999) > FREQUENCY_HIERARCHY.get(clock, 0):
       # clock보다 낮은 빈도만 처리
   ```
   - 이 조건문 때문에 `('m', 'w')`, `('q', 'w')`는 실제로 사용되지 않음
   - Clock이 'w'일 때, 'm'과 'q'는 더 낮은 빈도가 아니므로 조건을 통과하지 못함

3. **실제 사용 불가능**
   - `get_agg_structure`에서 걸러짐
   - EM 알고리즘에서도 사용되지 않음
   - 단순히 "정의만 되어 있는" 데드 코드

**결론:** 설계 원칙 위반 항목은 실제로 사용되지 않지만, 코드의 명확성을 해치고 혼란을 야기함.

---

## 2. EM 알고리즘이 "개판"인 이유

### 2.1 get_agg_structure 함수를 전혀 사용하지 않음

**문제점:**
- `get_agg_structure` 함수는 여러 빈도를 지원하도록 잘 작성되어 있음
- 하지만 EM 알고리즘의 `initialize_parameters`에서는 이 함수를 **전혀 호출하지 않음**
- 대신 `nQ` (quarterly count)만 하드코딩

**코드 위치:** `dfm-python/src/dfm_python/models/dfm.py:200-210`

```python
self.em.initialize_parameters(
    x=x_tensor,
    r=self.r,
    p=self.p,
    blocks=self.blocks,
    opt_nan=opt_nan,
    R_mat=None,  # ❌ get_agg_structure에서 생성한 R_mat을 사용하지 않음
    q=None,      # ❌ get_agg_structure에서 생성한 q를 사용하지 않음
    nQ=0,        # ❌ 하드코딩된 값만 사용
    ...
    tent_weights_dict=self.tent_weights_dict,  # ❌ 'q' 키만 확인
    ...
)
```

### 2.2 분기만 하드코딩

**코드 위치:** `dfm-python/src/dfm_python/ssm/em.py:573-578`

```python
# Determine tent kernel size (pC) for slower-frequency aggregation
pC = 5  # ❌ Default: quarterly to monthly uses 5 periods [1,2,3,2,1]
if R_mat is not None:
    pC = R_mat.shape[1]
elif tent_weights_dict is not None and 'q' in tent_weights_dict:  # ❌ 'q'만 확인
    pC = len(tent_weights_dict['q'])
```

**문제점:**
- `pC = 5`가 기본값으로 하드코딩됨
- `'q'` 키만 확인하고 다른 빈도(sa, a)는 무시
- `get_agg_structure`에서 계산한 `tent_weights`를 사용하지 않음

### 2.3 다른 빈도 완전히 무시

**코드 위치:** `dfm-python/src/dfm_python/ssm/em.py:973-980`

```python
# Add quarterly idiosyncratic chains (5-state: [1, 2, 3, 2, 1])
if nQ > 0:  # ❌ nQ만 확인, nSA, nA는 없음
    tent_q = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0], ...)  # ❌ 하드코딩
    C_quarterly = torch.zeros(N, 5 * nQ, ...)
    ...
```

**문제점:**
- 반기(sa), 연간(a) 시리즈에 대한 처리가 전혀 없음
- `nSA`, `nA` 같은 변수도 없음
- TENT_WEIGHTS_LOOKUP에 정의만 되어 있고 실제로 사용되지 않음

---

## 3. 왜 이런 일이 발생했나?

### 3.1 MATLAB 코드 포팅 과정

**추정 시나리오:**
1. 원본 MATLAB 코드가 분기 데이터만 가정
2. 포팅 과정에서 분기만 하드코딩
3. 다른 빈도는 "나중에 추가"하려다 미완성

### 3.2 점진적 개발의 부작용

**추정 시나리오:**
1. 초기 버전: 분기만 지원 (`nQ`, `pC=5` 하드코딩)
2. 확장성 고려: `TENT_WEIGHTS_LOOKUP`에 여러 빈도 쌍 정의
3. `get_agg_structure` 함수 작성 (여러 빈도 지원)
4. **하지만 EM 알고리즘에 통합하지 않음**
5. 결과: 설계와 구현의 불일치

### 3.3 설계와 구현의 분리

**문제점:**
- `get_agg_structure`는 "설계" 단계에서 작성됨
- EM 알고리즘은 "구현" 단계에서 기존 로직 유지
- 두 부분이 통합되지 않음

---

## 4. 실제 사용 흐름 분석

### 4.1 올바른 사용 흐름 (의도된 설계)

```
Config → get_agg_structure() → tent_weights, R_mat, q
                                    ↓
                          EM.initialize_parameters(R_mat, q, tent_weights_dict)
                                    ↓
                          여러 빈도 지원 (q, sa, a → m)
```

### 4.2 실제 사용 흐름 (현재 구현)

```
Config → nQ 계산 (quarterly만)
              ↓
    EM.initialize_parameters(nQ=..., tent_weights_dict={'q': ...})
              ↓
    분기만 지원 (q → m)
```

**문제:**
- `get_agg_structure`는 호출되지 않음
- `R_mat`, `q`는 `None`으로 전달됨
- `tent_weights_dict`는 `{'q': ...}`만 사용됨

---

## 5. 설계 원칙 위반 항목이 실제로 사용되는가?

### 5.1 get_agg_structure 함수에서

**코드:** `dfm-python/src/dfm_python/config/utils.py:381-393`

```python
for freq in frequencies:
    if FREQUENCY_HIERARCHY.get(freq, 999) > FREQUENCY_HIERARCHY.get(clock, 0):
        # clock보다 낮은 빈도만 처리
        tent_w = get_tent_weights(freq, clock)  # ('m', 'w')는 여기서 None 반환
```

**결과:**
- Clock이 'w'일 때: 'm'과 'q'는 더 낮은 빈도가 아니므로 조건을 통과하지 못함
- `get_tent_weights('m', 'w')`는 호출되지 않음
- 설계 원칙 위반 항목은 실제로 사용되지 않음

### 5.2 EM 알고리즘에서

- `get_agg_structure`를 호출하지 않으므로 설계 원칙 위반 항목은 접근할 수 없음

**결론:** 설계 원칙 위반 항목은 정의만 되어 있고 실제로는 사용되지 않는 데드 코드.

---

## 6. 왜 이런 혼란이 발생했나?

### 6.1 코드 일관성 부족

1. **TENT_WEIGHTS_LOOKUP**: 모든 빈도 쌍 정의 (일부는 사용 불가)
2. **get_agg_structure**: 올바른 필터링 (clock보다 낮은 빈도만)
3. **EM 알고리즘**: 분기만 하드코딩 (get_agg_structure 미사용)

**결과:** 세 부분이 서로 일관되지 않음

### 6.2 문서화 부족

- `TENT_WEIGHTS_LOOKUP`에 주석이 없어서 왜 고빈도→저빈도가 있는지 불명확
- `get_agg_structure`의 역할과 EM 알고리즘과의 관계가 명확하지 않음

### 6.3 테스트 부족

- 여러 빈도를 사용하는 테스트가 없어서 문제를 발견하지 못함
- `get_agg_structure`를 사용하는 테스트가 없어서 통합되지 않았는지 확인하지 못함

---

## 7. 결론

### 7.1 설계 원칙 위반 항목이 있는 이유

1. **과도한 "완전성" 추구**: 모든 가능한 빈도 쌍을 미리 정의하려는 의도
2. **실제 사용 여부 미검증**: 정의만 하고 실제로 사용되는지 확인하지 않음
3. **get_agg_structure의 올바른 필터링**: 조건문이 걸러내지만, 정의 자체는 남아있음

### 7.2 EM 알고리즘이 "개판"인 이유

1. **get_agg_structure 미사용**: 잘 작성된 함수를 전혀 사용하지 않음
2. **분기만 하드코딩**: `nQ`, `pC=5`, `'q'` 키만 확인
3. **다른 빈도 무시**: 반기, 연간 시리즈에 대한 처리가 전혀 없음
4. **설계와 구현의 분리**: 설계는 완료되었지만 구현에 통합되지 않음

### 7.3 근본 원인

1. **MATLAB 포팅의 한계**: 원본 코드가 분기만 가정
2. **점진적 개발의 부작용**: 설계는 확장했지만 구현은 유지
3. **통합 테스트 부족**: 여러 빈도를 사용하는 테스트가 없어서 문제를 발견하지 못함

### 7.4 개선 방향

1. **TENT_WEIGHTS_LOOKUP 정리**: 사용 불가능한 항목 제거 또는 명확한 주석 추가
2. **EM 알고리즘 수정**: `get_agg_structure`를 실제로 사용하도록 통합
3. **여러 빈도 지원**: 반기, 연간 시리즈에 대한 처리 추가
4. **테스트 추가**: 여러 빈도를 사용하는 통합 테스트 작성

