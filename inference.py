# inference.py
import time
import numpy as np
import tensorflow as tf
from scipy import signal
from config import SERVER_PORT, CHANNELS, BATCH_SIZE, WINDOW_SIZE, RIGHT_HAND_KEYS
from network import EMGReceiver

class EMGRealTimeInference:
    def __init__(self):
        # 1. 네트워크 및 모델 로드
        self.receiver = EMGReceiver(SERVER_PORT)
        self.model = tf.keras.models.load_model('best_emg_model.keras')
        
        # 2. 레이블 역매핑 사전 구축 (학습 코드와 동기화)
        self.raw_labels = sorted(list(set(RIGHT_HAND_KEYS.values())))
        self.idx_to_label = {idx: raw_id for idx, raw_id in enumerate(self.raw_labels)}
        self.inv_keys = {v: k for k, v in RIGHT_HAND_KEYS.items()}
        
        # 3. 신호 처리 버퍼 및 필터 정의
        self.data_buffer = np.zeros((CHANNELS, WINDOW_SIZE)) 
        fs, f0, Q = 400.0, 60.0, 30.0
        self.b, self.a = signal.iirnotch(f0, Q, fs=fs)
        
        # 4. 실시간 트리거 제어 파라미터
        self.threshold = 800000  # 사용자 신호 강도에 따라 커스텀 조정 필요 (RMS 임계치)
        self.cooldown = 0        # 중복 예측 방지용 카운터

    def run(self):
        # ESP32 클라이언트 접속 대기
        self.receiver.wait_for_connection()
        print("\n[READY] 실시간 EMG 키보드 분류 시작...")
        
        while True:
            batch = self.receiver.receive_batch()
            if batch is not None:
                # 링 버퍼 형태로 실시간 데이터 축적
                self.data_buffer = np.roll(self.data_buffer, -BATCH_SIZE, axis=1)
                self.data_buffer[:, -BATCH_SIZE:] = batch
                
                # 60Hz 전원 잡음 제거 필터링
                # processed = np.array([signal.lfilter(self.b, self.a, ch - np.mean(ch)) for ch in self.data_buffer])
                processed = np.array([ch - np.mean(ch) for ch in self.data_buffer])
                
                # 쿨다운 타임 관리
                if self.cooldown > 0:
                    self.cooldown -= BATCH_SIZE
                    continue
                
                # 최근 수집된 배치의 평균 RMS 산출을 통한 타건 여부 감지
                recent_rms = np.mean([np.sqrt(np.mean(np.square(ch[-BATCH_SIZE:]))) for ch in processed])
                
                # 임계치를 초과하는 순간 특징 윈도우 추출 및 추론 가동
                if recent_rms > self.threshold:
                    # 최근 100 샘플(250ms) 슬라이싱 및 형상 변환 (8, 100) -> (100, 8)
                    input_window = processed[:, -100:].T 
                    
                    # 채널별 Z-Score 정규화 (학습 환경과 동일 스케일 적용)
                    X_mean = np.mean(input_window, axis=0, keepdims=True)
                    X_std = np.std(input_window, axis=0, keepdims=True)
                    X_scaled = (input_window - X_mean) / (X_std + 1e-8)
                    
                    # 모델 입력 차원 확장 (1, 100, 8)
                    X_input = np.expand_dims(X_scaled, axis=0)
                    
                    # 딥러닝 모델 예측 실행 및 후처리
                    prediction = self.model.predict(X_input, verbose=0)
                    pred_idx = np.argmax(prediction)
                    confidence = prediction[0][pred_idx]
                    
                    # 인덱스를 실제 키 문자로 복원
                    raw_id = self.idx_to_label[pred_idx]
                    key_char = self.inv_keys.get(raw_id, f"Unknown({raw_id})")
                    
                    # 결과 출력
                    print(f"[PREDICT] 인식된 키: '{key_char}' (신뢰도: {confidence * 100:.1f}%)")
                    
                    # 연속 중복 트리거 방지 목적 쿨다운 설정 (100 샘플 간 예측 멈춤)
                    self.cooldown = 100

if __name__ == "__main__":
    with tf.device('/CPU:0'): # 실시간 단일 추론 시 오버헤드 감소를 위해 CPU 강제 지정
        EMGRealTimeInference().run()