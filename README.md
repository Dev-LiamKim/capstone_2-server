# EMG 키보드 분류기

8채널 근전도(EMG) 신호를 ESP32에서 TCP로 수신하고, 오른손 키 입력을 분류하는 프로젝트입니다.

이 프로젝트는 다음 기능을 포함합니다.

- ESP32로부터 EMG 샘플 실시간 수신
- 원신호 및 전처리 신호 실시간 시각화
- 타이핑 연습 UI를 통한 키 라벨 CSV 데이터셋 기록
- PyTorch 기반 CNN+LSTM 모델 학습
- 데이터셋 이벤트 및 동기화 품질 분석
- 학습된 모델을 이용한 실시간 추론

## 프로젝트 구조

```text
.
├── main.py                 # 실시간 EMG 수집 앱
├── network.py              # ESP32 TCP 패킷 수신기
├── gui.py                  # PyQtGraph 신호 시각화
├── typing_practice.py      # 키 입력 연습 UI 및 이벤트 기록
├── train.py                # PyTorch 학습 파이프라인
├── models.py               # 추론/학습 공용 모델 정의
├── inference.py            # PyTorch 실시간 추론 스크립트
├── analyzer.py             # 데이터셋 이벤트 개수 분석
├── check_class_jitter.py   # 이벤트-피크 지터 분석
├── check_sync_pure.py      # 클래스별 물리 피크 리포트
├── esp32-simulator.py      # 센서 없이 TCP 수신 흐름을 테스트하는 ESP32 시뮬레이터
├── config.py               # 공통 설정 및 키 매핑
└── requirements.txt
```

## 데이터와 결과물

다음 디렉터리는 실험 재현을 위해 Git에 포함할 수 있습니다.

- `dataset/`
- `new_dataset/`
- `results/`

다음 파일은 로컬 실행 산출물이므로 기본적으로 Git에서 제외합니다.

- `*.pt`
- `*.keras`
- `*.npy`
- `*.db`
- `*.docx`

학습된 모델 파일은 실험 및 추론 실행을 위해 프로젝트 루트에 둘 수 있습니다. 단, 릴리즈 산출물로 공유할 목적이 아니라면 일반 커밋에는 포함하지 않는 것을 권장합니다.

## 설치

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt`로 CUDA 지원 PyTorch 설치가 실패하면, 사용 중인 CUDA 버전에 맞는 PyTorch를 공식 안내에 따라 먼저 설치한 뒤 나머지 패키지를 설치하면 됩니다.

## 데이터 수집

ESP32 클라이언트를 `config.py`의 `SERVER_PORT`로 연결한 뒤 수집 앱을 실행합니다.

```powershell
.\venv\Scripts\python.exe main.py
```

앱을 실행하면 다음 창이 열립니다.

- 실시간 EMG 신호 시각화 창
- 키 라벨 기록을 위한 타이핑 연습 창

수집이 완료된 세션은 CSV 파일로 `dataset/` 또는 지정된 데이터셋 디렉터리에 저장됩니다.

## 학습

현재 가장 좋은 결과를 낸 조건은 `new_dataset`, 200 샘플 윈도우, 이벤트 이전 80 샘플, 20 Hz 하이패스 필터, CNN+LSTM 모델입니다.

```powershell
.\venv\Scripts\python.exe train.py --dataset-dir new_dataset --seed 42 --window-size 200 --pre-event 80 --max-epochs 80 --patience 8 --filter-mode highpass_20 --model cnn_lstm
```

주요 옵션은 다음과 같습니다.

```text
--dataset-dir
--window-size
--pre-event
--filter-mode
--notch-filter
--align-peak
--model
--seed
--max-epochs
--patience
```

학습이 끝나면 다음 결과물이 생성됩니다.

- `best_emg_model.pt`
- `results/runs/<timestamp>/report.txt`
- `results/runs/<timestamp>/training_log.csv`
- `results/runs/<timestamp>/training_history.png`
- `results/runs/<timestamp>/confusion_matrix.png`

## 실시간 추론

근전도 센서와 ESP32를 연결한 상태에서 `inference.py`를 실행하면, TCP로 들어오는 EMG 데이터를 받아 실시간으로 키를 예측합니다.

기본 실행:

```powershell
.\venv\Scripts\python.exe inference.py
```

상태 GUI와 함께 실행:

```powershell
.\venv\Scripts\python.exe inference.py --gui
```

모델 경로, 윈도우 크기, 필터 모드를 직접 지정해서 실행할 수도 있습니다.

```powershell
.\venv\Scripts\python.exe inference.py --model-path best_emg_model.pt --window-size 200 --filter-mode highpass_20
```

`inference.py`는 다음 순서로 동작합니다.

1. `network.py`의 `EMGReceiver`를 통해 ESP32 TCP 클라이언트 연결을 대기합니다.
2. 수신한 8채널 EMG 데이터를 rolling buffer에 저장합니다.
3. 설정된 streaming filter를 적용합니다.
4. 초반 안정 상태 RMS를 이용해 trigger threshold를 자동 보정합니다.
5. threshold를 넘는 구간에서 최근 윈도우를 모델에 입력합니다.
6. confidence, margin, vote window, cooldown 조건을 거쳐 예측 결과를 출력합니다.

GUI를 사용하지 않으면 추론 결과는 터미널에 출력됩니다. `--gui` 옵션을 사용하면 별도 PyQt 상태 창에 최근 예측 결과, 신뢰도, RMS 상태가 표시됩니다.

추론을 실행하면 기본적으로 `inference_logs/` 아래에 CSV 로그가 저장됩니다. 로그에는 RMS, threshold, 상태, 예측 키, confidence, margin, top-k 후보, cooldown 정보가 기록됩니다. 로그 저장을 끄고 싶으면 다음처럼 실행합니다.

```powershell
.\venv\Scripts\python.exe inference.py --no-log
```

센서 없이 저장된 CSV를 재생하면서 추론 파이프라인을 테스트할 수도 있습니다.

```powershell
.\venv\Scripts\python.exe inference.py --replay-csv new_dataset\recording_20260529_193249.csv --no-replay-realtime
```

이 방식은 CSV를 `inference.py` 내부에서 직접 읽기 때문에 모델 추론, threshold, smoothing, GUI, 로그 기능을 빠르게 확인할 때 적합합니다.

실제 수신 속도에 맞춰 재생하려면 `--replay-realtime`을 사용하고, 반복 재생하려면 `--replay-loop`을 함께 사용합니다.

```powershell
.\venv\Scripts\python.exe inference.py --replay-csv new_dataset\recording_20260529_193249.csv --replay-realtime --replay-loop
```

센서 없이 실제 TCP 수신 구조까지 테스트하려면 `inference.py`를 먼저 실행한 뒤, 다른 터미널에서 `esp32-simulator.py`를 실행합니다.

```powershell
.\venv\Scripts\python.exe inference.py
```

```powershell
.\venv\Scripts\python.exe esp32-simulator.py
```

이 방식은 `esp32-simulator.py`가 CSV 데이터를 실제 ESP32처럼 TCP 바이너리 패킷으로 보내고, `inference.py`가 `network.py`를 통해 수신하므로 실제 센서 연결 흐름에 더 가깝습니다.

## 추론 설정

실시간 추론 파라미터는 `config.py`의 `INFERENCE_*` 항목에서 수정할 수 있습니다. CLI 옵션으로 넘긴 값은 `config.py` 값보다 우선 적용됩니다.

자주 조정하는 값은 다음과 같습니다.

```python
INFERENCE_GUI = False
INFERENCE_LOG_ENABLED = True
INFERENCE_LOG_DIR = "inference_logs"
INFERENCE_LOG_ALL_STATES = True
INFERENCE_MODEL_PATH = "best_emg_model.pt"
INFERENCE_MODEL_TYPE = "cnn_lstm"
INFERENCE_FILTER_MODE = "highpass_20"
INFERENCE_WINDOW_SIZE = 200
INFERENCE_AUTO_THRESHOLD = True
INFERENCE_THRESHOLD = 80000.0
INFERENCE_CALIBRATION_SECONDS = 3.0
INFERENCE_THRESHOLD_MULTIPLIER = 4.0
INFERENCE_MIN_CONFIDENCE = 0.35
INFERENCE_MIN_MARGIN = 0.10
INFERENCE_VOTE_WINDOW = 3
INFERENCE_MIN_VOTES = 2
INFERENCE_COOLDOWN_SAMPLES = 200
INFERENCE_CALIBRATION_MODE = False
INFERENCE_ACTIVE_CALIBRATION_SECONDS = 6.0
INFERENCE_REPLAY_CSV = ""
```

threshold는 기본적으로 센서 연결 직후의 안정 상태 RMS를 이용해 자동 설정됩니다.

```text
threshold = idle_mean + INFERENCE_THRESHOLD_MULTIPLIER * idle_std
```

`INFERENCE_THRESHOLD = 80000.0`은 자동 보정이 꺼졌거나 보정에 실패했을 때 사용하는 fallback 값입니다. 실제 센서 착용 상태, 전극 접촉, 사용자, 움직임 강도에 따라 적정 값이 달라질 수 있으므로 테스트 환경에서는 RMS를 확인한 뒤 조정하는 것이 좋습니다.

RMS 확인용 실행 예시는 다음과 같습니다.

```powershell
.\venv\Scripts\python.exe inference.py --print-rms --manual-threshold --threshold 999999999
```

이 상태에서 안정 상태 RMS와 실제 키 입력 시 RMS를 비교한 뒤, 두 범위 사이의 값을 수동 threshold로 지정할 수 있습니다.

캘리브레이션 모드를 사용하면 안정 상태 보정 이후 몇 초 동안 키를 눌러보며 active RMS, confidence, margin을 수집하고 추천 threshold를 출력합니다.

```powershell
.\venv\Scripts\python.exe inference.py --calibration-mode
```

캘리브레이션 리포트만 확인하고 종료하려면 다음처럼 실행합니다.

```powershell
.\venv\Scripts\python.exe inference.py --calibration-mode --calibration-only
```

## 현재 최고 성능

자세한 내용은 로컬 실험 보고서를 참고합니다.

```text
EMG_모델_개선_실험_보고서_20260601.md
```

요약:

- Dataset: `new_dataset`
- Model: CNN+LSTM
- Window: 200 samples
- Pre/Post event: 80 / 120 samples
- Filter: high-pass 20 Hz
- Test Accuracy: 76.84%

## 참고 사항

- `network.py`는 ESP32에서 전송되는 EMG 바이너리 패킷을 수신하는 역할을 합니다.
- `inference.py`는 `network.py`로 들어온 실시간 데이터를 모델 입력 형태로 변환하고 예측을 수행합니다.
- `esp32-simulator.py`는 센서 없이 `network.py`의 TCP 수신 구조를 검증할 때 사용하는 선택적 테스트 도구입니다.
- ESP32 데이터 형식은 8채널, batch size 32, little-endian int32 기준입니다.
- 실시간 추론에는 `train.py`로 학습한 PyTorch 모델 파일인 `best_emg_model.pt`가 필요합니다.
