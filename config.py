# config.py
CHANNELS = 8
BATCH_SIZE = 32
WINDOW_SIZE = 2000 
SAMPLING_RATE_THEORETICAL = 350.0  
SERVER_PORT = 5000
PACKET_SIZE = (BATCH_SIZE * CHANNELS * 4)

# [추가] main.py 이관 상수군 정의
NOTCH_F0 = 60.0                  # 노치 필터 중심 주파수
NOTCH_Q = 30.0                   # 노치 필터 Q-인자
TIMER_INTERVAL_MS = 10           # PyQt 타이머 주기 (ms)
HZ_CHECK_INTERVAL = 0.5          # 샘플링 레이트 계산 주기 (초)
HZ_HISTORY_DURATION = 5.0       # 이동 평균 계산 윈도우 크기 (초)
HZ_THRESHOLD_LOW = 345.0         # 입력 차단 기준 임계 속도 (Hz)

# 실시간 추론 기본 설정
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
INFERENCE_CALIBRATION_MODE = False
INFERENCE_CALIBRATION_ONLY = False
INFERENCE_ACTIVE_CALIBRATION_SECONDS = 6.0
INFERENCE_REPLAY_CSV = ""
INFERENCE_REPLAY_LOOP = False
INFERENCE_REPLAY_REALTIME = True
INFERENCE_REPLAY_SPEED = 1.0

# 오른손 입력 가능 키 매핑
RIGHT_HAND_KEYS = {
    'y': 11, 'u': 12, 'i': 13, 'o': 14, 'p': 15, '[': 16, ']': 17, '\\': 18,
    'h': 21, 'j': 22, 'k': 23, 'l': 24, ';': 25, "'": 26,
    'n': 31, 'm': 32, ',': 33, '.': 34, '/': 35,
}
