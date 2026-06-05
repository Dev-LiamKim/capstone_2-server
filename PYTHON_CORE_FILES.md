# 핵심 Python 파일 설명

이 문서는 현재 프로젝트에서 학습과 실시간 추론 흐름에 직접 관련된 핵심 Python 파일 4개를 설명합니다. 특히 내가 새로 만들었거나 현재 구조로 크게 정리한 파일을 중심으로, 협업자가 어떤 파일을 언제 봐야 하는지 빠르게 파악할 수 있도록 정리했습니다.

대상 파일:

- `train.py`
- `models.py`
- `inference.py`
- `config.py`

## 전체 흐름

```text
config.py
  ├─ 공통 상수, 키 매핑, 실시간 추론 기본값 제공
  │
train.py
  ├─ new_dataset 또는 dataset CSV를 읽어 모델 학습
  ├─ best_emg_model.pt 생성
  └─ results/runs/<timestamp>/ 아래 학습 결과 저장
  │
models.py
  └─ inference.py에서 사용할 모델 구조 정의
  │
inference.py
  ├─ best_emg_model.pt 로드
  ├─ EMG 실시간 수신 또는 CSV replay 입력 처리
  └─ 키 예측 결과 출력/GUI 표시/로그 저장
```

## 파일별 작업 성격

```text
train.py      : 학습 파이프라인을 현재 실험 구조에 맞게 크게 정리한 파일
models.py     : 추론에서 모델 구조만 가볍게 import하기 위해 새로 분리한 파일
inference.py  : 실시간 PyTorch 추론, 로그, replay, calibration 기능을 크게 확장한 파일
config.py     : 공통 설정과 실시간 추론 파라미터를 한 곳에서 관리하도록 확장한 파일
```

## `config.py`

프로젝트 공통 설정 파일입니다. 학습 코드와 추론 코드가 함께 참조하는 값도 있고, 실시간 추론에서만 사용하는 값도 있습니다.

주요 역할:

- EMG 데이터 기본 구조 정의
  - `CHANNELS = 8`
  - `BATCH_SIZE = 32`
  - `SERVER_PORT = 5000`
- ESP32 TCP 패킷 크기 정의
- 오른손 키 매핑 정의
  - 예: `y`, `u`, `i`, `o`, `p`, `h`, `j`, `k` 등
- 실시간 추론 기본값 관리
  - 모델 경로
  - 필터 모드
  - window size
  - threshold 자동 보정
  - confidence/margin/vote/cooldown
  - GUI, 로그, replay, calibration 설정

구분해서 보면 다음과 같습니다.

```text
공통 설정:
  CHANNELS, BATCH_SIZE, SERVER_PORT, PACKET_SIZE, RIGHT_HAND_KEYS

수집/시각화 설정:
  WINDOW_SIZE, TIMER_INTERVAL_MS, HZ_CHECK_INTERVAL, HZ_THRESHOLD_LOW

실시간 추론 설정:
  INFERENCE_MODEL_PATH, INFERENCE_FILTER_MODE, INFERENCE_WINDOW_SIZE,
  INFERENCE_AUTO_THRESHOLD, INFERENCE_MIN_CONFIDENCE, INFERENCE_GUI 등
```

동료가 주로 수정할 가능성이 높은 값:

```python
INFERENCE_MODEL_PATH = "best_emg_model.pt"
INFERENCE_FILTER_MODE = "highpass_20"
INFERENCE_WINDOW_SIZE = 200
INFERENCE_AUTO_THRESHOLD = True
INFERENCE_THRESHOLD = 80000.0
INFERENCE_MIN_CONFIDENCE = 0.35
INFERENCE_MIN_MARGIN = 0.10
INFERENCE_GUI = False
```

주의할 점:

- `RIGHT_HAND_KEYS`를 바꾸면 학습 라벨 수와 추론 클래스 매핑에도 영향이 있습니다.
- `INFERENCE_WINDOW_SIZE`는 학습 때 사용한 window size와 맞추는 것이 좋습니다.
- `config.py`는 학습 CLI 옵션 전체를 대체하지 않습니다. 학습 실험 조건은 주로 `train.py` 실행 옵션으로 넘깁니다.
- CLI 옵션으로 넘긴 값은 `config.py` 기본값보다 우선 적용됩니다.

## `train.py`

EMG 키 분류 모델을 학습하는 메인 학습 스크립트입니다.

주요 역할:

- CSV 데이터셋 로드
- 키 이벤트 주변 window 추출
- 필터링 및 전처리
- train/validation/test 분리
- 데이터 증강
- CNN+LSTM 또는 ResNet1D 모델 학습
- best model 저장
- 학습 결과 리포트와 그래프 저장

대표 실행 예시:

```powershell
.\venv\Scripts\python.exe train.py --dataset-dir new_dataset --seed 42 --window-size 200 --pre-event 80 --max-epochs 80 --patience 8 --filter-mode highpass_20 --model cnn_lstm
```

참고로 `train.py`의 기본 dataset은 `dataset`이지만, 현재 추천 실험 조건은 `--dataset-dir new_dataset`을 명시해서 실행하는 방식입니다.

주요 출력:

- `best_emg_model.pt`
- `results/runs/<timestamp>/report.txt`
- `results/runs/<timestamp>/training_log.csv`
- `results/runs/<timestamp>/training_history.png`
- `results/runs/<timestamp>/confusion_matrix.png`

현재 기준 추천 학습 조건:

```text
Dataset: new_dataset
Model: cnn_lstm
Window size: 200
Pre-event: 80
Filter: highpass_20
```

주의할 점:

- `best_emg_model.pt`는 실시간 추론에서 사용하는 모델 파일입니다.
- 학습 조건을 바꾸면 `inference.py` 실행 시 window size, model type, filter mode도 맞춰야 합니다.
- 모델 구조를 `resnet1d`로 학습했다면 추론도 `--model resnet1d`로 실행해야 합니다.
- `train.py`의 필터 옵션은 학습용이라 `bandpass_20_150`, `bandpass_20_175`도 포함합니다. 현재 `inference.py`의 streaming filter는 `raw`, `notch`, `highpass_20`, `highpass_20_notch`를 지원합니다.

## `models.py`

실시간 추론에서 사용할 PyTorch 모델 구조를 정의한 파일입니다.

주요 역할:

- `CNNLSTM` 모델 정의
- `ResNet1D` 모델 정의
- `inference.py`가 `train.py` 전체를 import하지 않고도 모델을 로드할 수 있게 분리

이 파일을 분리한 이유:

- `train.py`는 학습용 코드, 데이터 로딩, 리포트 저장 로직까지 포함합니다.
- 추론에서는 모델 구조만 필요합니다.
- 그래서 `inference.py`는 `models.py`에서 모델 클래스만 가져와 가볍게 동작합니다.

사용 위치:

```python
from models import CNNLSTM, ResNet1D
```

주의할 점:

- `train.py`의 모델 구조를 수정했다면 `models.py`의 구조도 동일하게 맞춰야 합니다.
- 구조가 다르면 `best_emg_model.pt` 로드 시 `state_dict` 크기 불일치 오류가 발생할 수 있습니다.
- 현재 `models.py`는 추론에 필요한 `CNNLSTM`, `ResNet1D`만 담습니다. 학습 전용 유틸리티나 리포트 함수는 넣지 않습니다.

## `inference.py`

학습된 모델로 실시간 EMG 데이터를 추론하는 실행 스크립트입니다.

주요 역할:

- `best_emg_model.pt` 로드
- ESP32 TCP 데이터 수신
- streaming filter 적용
- RMS 기반 trigger threshold 판단
- 자동 threshold calibration
- 모델 추론
- confidence, margin, vote window, cooldown 기반 예측 안정화
- 터미널 또는 GUI에 예측 결과 표시
- 추론 로그 CSV 저장
- 센서 없이 CSV replay 추론 지원

기본 실행:

```powershell
.\venv\Scripts\python.exe inference.py
```

GUI 실행:

```powershell
.\venv\Scripts\python.exe inference.py --gui
```

CSV replay 실행:

```powershell
.\venv\Scripts\python.exe inference.py --replay-csv new_dataset\recording_20260529_193249.csv --no-replay-realtime
```

캘리브레이션 모드:

```powershell
.\venv\Scripts\python.exe inference.py --calibration-mode
```

센서 없이 TCP 수신 구조까지 테스트:

```powershell
.\venv\Scripts\python.exe inference.py
```

다른 터미널에서:

```powershell
.\venv\Scripts\python.exe esp32-simulator.py
```

주요 상태:

- `calibrating`: 안정 상태 RMS를 이용해 threshold 자동 보정 중
- `idle`: threshold 미만이라 입력 대기 중
- `pending`: 후보 예측은 있지만 vote 조건 대기 중
- `skip`: confidence 또는 margin 조건 미달
- `predict`: 최종 예측 출력
- `cooldown`: 중복 출력 방지 대기 중

주의할 점:

- 기본 모델 경로는 `config.py`의 `INFERENCE_MODEL_PATH`를 따릅니다.
- `best_emg_model.pt`가 없으면 실행할 수 없습니다.
- 학습 시 사용한 모델 구조와 추론 시 `--model` 옵션이 일치해야 합니다.
- 학습 시 사용한 window size와 추론 시 `--window-size` 또는 `INFERENCE_WINDOW_SIZE`가 맞아야 합니다.
- `--replay-csv`는 TCP 수신을 건너뛰고 CSV를 직접 읽습니다.
- TCP 수신 구조까지 테스트하려면 `esp32-simulator.py`를 별도로 실행해야 합니다.

## 자주 생길 수 있는 문제

```text
best_emg_model.pt 로드 실패:
  train.py와 models.py의 모델 구조가 다르거나 --model 옵션이 맞지 않는 경우가 많습니다.

추론이 거의 발생하지 않음:
  threshold가 너무 높거나 센서 접촉이 약할 수 있습니다.
  --print-rms 또는 --calibration-mode로 RMS 범위를 먼저 확인합니다.

예측이 너무 자주 중복 출력됨:
  INFERENCE_COOLDOWN_SAMPLES, INFERENCE_VOTE_WINDOW, INFERENCE_MIN_VOTES 값을 조정합니다.

replay는 되는데 실제 센서/TCP는 안 됨:
  replay는 network.py를 거치지 않습니다.
  esp32-simulator.py로 TCP 수신 구조를 따로 확인합니다.
```

## 수정 시 체크리스트

- `train.py`에서 모델 구조를 바꿨다면 `models.py`도 같이 수정했는가?
- `train.py`에서 window size를 바꿨다면 `config.py` 또는 `inference.py` 실행 옵션도 맞췄는가?
- `train.py`에서 filter mode를 바꿨다면 실시간 추론에서 같은 종류의 streaming filter를 지원하는가?
- `config.py`의 key mapping을 바꿨다면 기존 데이터셋 라벨과 호환되는가?
- `inference.py` 변경 후 `--help`, `--replay-csv`, `py_compile`을 확인했는가?

검증 명령:

```powershell
.\venv\Scripts\python.exe -m py_compile config.py train.py models.py inference.py
.\venv\Scripts\python.exe inference.py --help
```
