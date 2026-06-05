# config.py
# 프로젝트 전반에서 공유하는 상수 모음입니다.
# 수집(main.py), 학습(train.py), 실시간 추론(inference.py)이 함께 참조하므로
# 키 매핑이나 채널 수처럼 데이터 형식에 직접 영향을 주는 값은 변경 시 주의가 필요합니다.
CHANNELS = 8
BATCH_SIZE = 32
WINDOW_SIZE = 2000 
SAMPLING_RATE_THEORETICAL = 350.0  
SERVER_PORT = 5000
# ESP32는 int32 샘플을 BATCH_SIZE * CHANNELS개씩 전송합니다.
PACKET_SIZE = (BATCH_SIZE * CHANNELS * 4)

# 수집/시각화 앱(main.py, gui.py)에서 사용하는 신호 처리 및 타이머 설정
NOTCH_F0 = 60.0                  # 노치 필터 중심 주파수
NOTCH_Q = 30.0                   # 노치 필터 Q-인자
TIMER_INTERVAL_MS = 10           # PyQt 타이머 주기 (ms)
HZ_CHECK_INTERVAL = 0.5          # 샘플링 레이트 계산 주기 (초)
HZ_HISTORY_DURATION = 5.0       # 이동 평균 계산 윈도우 크기 (초)
HZ_THRESHOLD_LOW = 345.0         # 입력 차단 기준 임계 속도 (Hz)

# 실시간 추론 기본 설정
# CLI 옵션을 주지 않았을 때 inference.py가 사용하는 기본값입니다.
# 학습 조건(window/model/filter)을 바꿨다면 이 값도 같이 맞춰야 합니다.
INFERENCE_MODEL_PATH = "best_emg_model.pt"
INFERENCE_MODEL_TYPE = "cnn_lstm"            # "cnn_lstm" 또는 "resnet1d"
INFERENCE_DEVICE = "auto"                    # "auto", "cpu", "cuda"
INFERENCE_FILTER_MODE = "highpass_20"        # "raw", "notch", "highpass_20", "highpass_20_notch"
INFERENCE_WINDOW_SIZE = 200
INFERENCE_BUFFER_SIZE = 2000

# 자동 threshold 보정: 연결 직후 안정 상태 RMS 기준으로 threshold 산출
INFERENCE_AUTO_THRESHOLD = True
INFERENCE_THRESHOLD = 80000.0                # 수동 threshold 또는 자동 보정 전 초기값
INFERENCE_CALIBRATION_SECONDS = 3.0
INFERENCE_THRESHOLD_MULTIPLIER = 4.0

# 예측 안정화 설정
INFERENCE_MIN_CONFIDENCE = 0.35
INFERENCE_MIN_MARGIN = 0.10                  # top-1 confidence - top-2 confidence
INFERENCE_VOTE_WINDOW = 3
INFERENCE_MIN_VOTES = 2
INFERENCE_COOLDOWN_SAMPLES = 200
INFERENCE_TOP_K = 3

# 로그 및 GUI 설정
INFERENCE_PRINT_RMS = False
INFERENCE_RMS_LOG_INTERVAL = 1.0
INFERENCE_GUI = False
INFERENCE_GUI_INTERVAL_MS = 10
INFERENCE_LOG_ENABLED = True
INFERENCE_LOG_DIR = "inference_logs"
INFERENCE_LOG_ALL_STATES = True

# 캘리브레이션/리플레이 설정
# calibration mode는 실제 착용 상태에서 RMS/confidence 범위를 확인할 때 사용합니다.
# replay는 센서 없이 CSV를 직접 읽어 inference.py 추론 로직만 빠르게 점검하는 기능입니다.
INFERENCE_CALIBRATION_MODE = False
INFERENCE_CALIBRATION_ONLY = False
INFERENCE_ACTIVE_CALIBRATION_SECONDS = 6.0
INFERENCE_REPLAY_CSV = ""
INFERENCE_REPLAY_LOOP = False
INFERENCE_REPLAY_REALTIME = True
INFERENCE_REPLAY_SPEED = 1.0

# 오른손 입력 가능 키 매핑
# CSV Event 값과 모델 클래스 인덱스의 기준이 되는 매핑입니다.
# 기존 데이터셋의 Event 라벨과 맞물려 있으므로 임의 변경하면 재학습이 필요합니다.
RIGHT_HAND_KEYS = {
    'y': 11, 'u': 12, 'i': 13, 'o': 14, 'p': 15, '[': 16, ']': 17, '\\': 18,
    'h': 21, 'j': 22, 'k': 23, 'l': 24, ';': 25, "'": 26,
    'n': 31, 'm': 32, ',': 33, '.': 34, '/': 35,
}
