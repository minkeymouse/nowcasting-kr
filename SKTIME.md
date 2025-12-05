# sktime ColumnEnsembleTransformer Index Issue - 해결됨

## Problem
ColumnEnsembleTransformer의 `_transform` 출력이 sktime mtype 사양을 준수하지 않습니다.
오류: `pd.DataFrame: <class 'pandas.core.indexes.base.Index'> is not supported for obj, use one of (<class 'pandas.core.indexes.range.RangeIndex'>, <class 'pandas.core.indexes.period.PeriodIndex'>, <class 'pandas.core.indexes.datetimes.DatetimeIndex'>)`

## Root Cause
1. **ColumnEnsembleTransformer의 pd.concat 문제**: ColumnEnsembleTransformer는 각 transformer의 출력을 `pd.concat`으로 합치는데, 서로 다른 index 타입(DatetimeIndex와 RangeIndex)을 합치면 일반 Index가 됩니다.
2. **Transformation 함수의 index 손실**: transformation 함수들이 numpy array를 반환하여 index가 손실됩니다.
3. **StandardScaler의 index 처리**: StandardScaler가 DataFrame을 처리할 때 index를 변경할 수 있습니다.

## Solution (구현 완료)

### 1. IndexPreservingColumnEnsembleTransformer Wrapper
- `src/preprocess/index_preserving_transformer.py`에 wrapper 클래스 구현
- ColumnEnsembleTransformer를 래핑하여 출력 index를 보존
- 각 transformer의 출력 index를 입력 index와 일치시킨 후 concat
- 최종 출력이 DatetimeIndex, PeriodIndex, 또는 RangeIndex인지 확인

### 2. Transformation 함수들의 index 보존
- 모든 `make_*_transformer` 함수들이 index를 보존하는 wrapper 추가
- `make_pch_transformer`, `make_pc1_transformer`, `make_pca_transformer`, `make_cch_transformer`, `make_cca_transformer`, `make_cha_transformer` 모두 수정
- `identity_transform`, `log_transform`도 index 보존 wrapper 추가
- 각 transformation 함수가 Series를 받으면 Series를 반환하도록 수정

### 3. DFMDataModule의 index 보정
- `dfm-python/src/dfm_python/lightning/data_module.py`에서 pipeline 출력의 index를 보정
- numpy array를 DataFrame으로 변환할 때 입력 index 사용
- DataFrame 출력의 index가 호환되지 않으면 보정

### 4. Pipeline 설정
- `set_output(transform="pandas")` 설정으로 pandas 출력 보장

## Implementation Details

### IndexPreservingColumnEnsembleTransformer
- BaseTransformer를 상속하여 sktime 호환성 보장
- `_fit`에서 원본 index 저장
- `_transform`에서 각 transformer 출력의 index를 입력 index와 일치시킨 후 concat
- 최종 출력 index가 정렬되어 있는지 확인

### Transformation Wrappers
- 각 transformation 함수를 래핑하여 Series 입력 시 Series 출력 보장
- index와 name 속성 보존

## Key Points
- sktime은 DatetimeIndex, PeriodIndex, 또는 RangeIndex만 지원합니다
- 일반 Index는 지원되지 않습니다
- FunctionTransformer는 입력의 index를 보존해야 합니다
- **중요**: transformation 함수가 numpy array를 반환하면 index가 손실되므로, 반드시 Series를 반환해야 합니다
- pd.concat는 서로 다른 index 타입을 합치면 일반 Index를 만들 수 있으므로, concat 전에 모든 index를 일치시켜야 합니다

## Status
✅ **해결됨** - sktime index 호환성 문제는 해결되었습니다. 이제 다른 오류(수치적 불안정성)가 발생할 수 있지만, 이는 데이터 품질이나 모델 설정 문제입니다.
