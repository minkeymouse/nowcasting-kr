## 한국 실시간 경기진단(생산/투자) 보고서 딥러닝 예측 실험 코드

### 1) 환경 준비 (uv)

```bash
cd nowcasting-kr
uv venv
source .venv/bin/activate

# 루트 패키지 설치
uv sync

# 로컬 서브패키지(DFM/DDFM 라이브러리) 설치
uv pip install -e dfm-python

# (선택) Mamba 설치: 환경 의존성이 커서 옵션으로 분리 (아래 참고)
```

#### (선택) Mamba 설치/실행 관련 (환경 의존)

`mamba-ssm`은 C++/CUDA 확장 빌드가 필요할 수 있고, **로컬 CUDA toolkit 버전과 PyTorch가 컴파일된 CUDA 버전이 다르면 빌드가 실패**할 수 있음. Mamba까지 포함해 재현하려면 다음 중 하나가 필요합니다.\n  - 환경에 맞는 **사전 빌드 wheel**이 있는 경우 그것을 사용\n  - 또는 CUDA toolkit / torch CUDA 버전을 맞춘 뒤 로컬 설치

### 2) 데이터

실험 재현에 필요한 데이터/메타데이터는 `data/`에서 확인 가능.
- `train_investment.csv`, `test_investment.csv`
- `train_production.csv`, `test_production.csv`
- `investment_metadata.csv`, `production_metadata.csv`
- `raw_data.csv` (원본)

### 3) 실험 실행 (Hydra)

`src/main.py`을 엔트리 포인트로 활용.

#### 단기(재귀적 1-step) 실험

```bash
# 예: 투자 / Mamba
uv run python -m src.main data=investment model=mamba experiment=short_term train=true forecast=true

# 예: 생산 / iTransformer(itf)
uv run python -m src.main data=production model=itf experiment=short_term train=true forecast=true
```

#### 장기(다중 horizon) 실험

```bash
# 예: 투자 / PatchTST
uv run python -m src.main data=investment model=patchtst experiment=long_term train=true forecast=true
```

실험 결과는 기본적으로 아래에 저장됨.
- 단기: `outputs/short_term/{investment|production}/{model}/`
- 장기: `outputs/long_term/{investment|production}/{model}/horizon_{4w..40w}/`
- 체크포인트: `checkpoints/{investment|production}/{model}/`

학습 비용이 큰 모델(특히 DFM)은 재학습 없이 제공된 체크포인트로 예측/평가만 재현하는 것을 권장.

```bash
# 예: 단기 실험을 학습 없이 재실행(체크포인트 필요)
uv run python -m src.main data=investment model=mamba experiment=short_term train=false forecast=true
```

### 4) 보고서 산출물(표/그림) 생성

보고서 마크다운은 `nowcasting-report/`에 있고, 참조되는 주요 이미지는 `nowcasting-report/images/`에 생성.

> 배포(zip)에는 보고서 원문(`nowcasting-report/`)을 포함하지 않습니다. 다만 아래 스크립트는 실행 가능합니다.\n+
#### 결과 표(요약 CSV)

```bash
uv run python -m src.paper.table_results
# outputs/results_table.csv 생성/갱신
```

#### EDA 그림

```bash
uv run python -m src.paper.plot_eda
# nowcasting-report/images/combined_eda.png 생성/갱신
```

#### 예측 그림 (어텐션/SSM)

```bash
# 어텐션 계열(TFT/PatchTST/iTransformer)
uv run python -m src.paper.plot_forecast --model attention --experiment short_term
# nowcasting-report/images/combined_attention_forecast.png 생성/갱신

# 상태공간 계열(DFM/DDFM/Mamba)
uv run python -m src.paper.plot_forecast --model ssm --experiment short_term
# nowcasting-report/images/combined_ssm_forecast.png 생성/갱신
```

