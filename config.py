# config.py
CHANNELS = 8
BATCH_SIZE = 32
WINDOW_SIZE = 2000 
SAMPLING_RATE_THEORETICAL = 546.75  # 8채널 세트 기준 Hz
PACKET_SIZE = BATCH_SIZE * CHANNELS * 4
SERVER_PORT = 5000

# [수정] 오른손 입력 가능 키 매핑 (ASCII 또는 고유 ID)
# 영문 소문자, 대문자 및 특수키 포함
RIGHT_HAND_KEYS = {
    # 문자열 (QWERTY 기준 오른손 영역)
    'y': 11, 'u': 12, 'i': 13, 'o': 14, 'p': 15, '[': 16, ']': 17, '\\': 18,
    'h': 21, 'j': 22, 'k': 23, 'l': 24, ';': 25, "'": 26,
    'n': 31, 'm': 32, ',': 33, '.': 34, '/': 35,
    
    # 특수 키 (pynput.keyboard.Key 명칭 대응)
    'space': 41,
    'enter': 42,
    'backspace': 43,
    'shift': 44,
    'shift_r': 44
}