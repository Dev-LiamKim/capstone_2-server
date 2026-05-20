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

# 오른손 입력 가능 키 매핑
RIGHT_HAND_KEYS = {
    'y': 11, 'u': 12, 'i': 13, 'o': 14, 'p': 15, '[': 16, ']': 17, '\\': 18,
    'h': 21, 'j': 22, 'k': 23, 'l': 24, ';': 25, "'": 26,
    'n': 31, 'm': 32, ',': 33, '.': 34, '/': 35,
}