import time
import numpy as np
from scipy import signal
from network import EMGReceiver
from config import SERVER_PORT, CHANNELS, BATCH_SIZE

class RealtimeSpatialAvgDetector:
    def __init__(self, k_factor=3.5, ma_window=50, cooldown_time=0.2, fs=400.0):
        # 하이퍼파라미터 및 버퍼 초기화
        self.k_factor = k_factor
        self.ma_window = ma_window
        self.fs = fs
        self.cooldown_samples = int(cooldown_time * fs)
        
        self.current_cooldown = 0
        self.history_buffer = np.zeros((CHANNELS, self.ma_window))
        
        # 20Hz 고역통과 필터 설계 (인과적 필터용)
        nyquist = 0.5 * self.fs
        high_cut = 20.0 / nyquist
        self.b, self.a = signal.butter(4, high_cut, btype='high')
        
        # 필터 연속성 유지를 위한 상태 변수(zi) 초기화
        zi_1d = signal.lfilter_zi(self.b, self.a)
        self.filter_state = np.zeros((CHANNELS, len(zi_1d)))

    def process_batch(self, current_batch):
        """
        실시간 수신 패킷 처리 및 피크 감지 함수
        current_batch 형상: (CHANNELS, BATCH_SIZE)
        """
        # 1. 단방향 실시간 필터링 (Causal Filtering)
        filtered_batch, self.filter_state = signal.lfilter(
            self.b, self.a, current_batch, axis=1, zi=self.filter_state
        )
        
        # 2. 신호 정류화 (Rectification)
        rectified_batch = np.abs(filtered_batch)
        
        # 3. 다채널 신호 통합 (Spatial Averaging) -> 채널 축(Axis 0) 평균 연산
        spatial_avg_stream = np.mean(rectified_batch, axis=0)
        
        peak_detected = False
        
        # 4. 샘플 단위 실시간 순회 판별
        for i in range(len(spatial_avg_stream)):
            # 쿨다운 불응기 동작 제어
            if self.current_cooldown > 0:
                self.current_cooldown -= 1
                self.update_buffer(rectified_batch[:, i])
                continue
            
            # 동적 베이스라인 산출 (이동 평균 버퍼 전체 평균)
            current_baseline = np.mean(self.history_buffer)
            current_energy = spatial_avg_stream[i]
            
            # 임계치 돌파 조건 검증 (K-Factor 적용)
            if current_baseline > 0 and current_energy > (current_baseline * self.k_factor):
                peak_detected = True
                self.current_cooldown = self.cooldown_samples
            
            # 이동 버퍼 업데이트
            self.update_buffer(rectified_batch[:, i])
            
            if peak_detected:
                break  # 중복 트리거 방지
                
        return peak_detected

    def update_buffer(self, sample_data):
        """이동 평균 버퍼 갱신"""
        self.history_buffer = np.roll(self.history_buffer, -1, axis=1)
        self.history_buffer[:, -1] = sample_data

def main_execution():
    # 데이터 수신 및 객체 생성
    receiver = EMGReceiver(SERVER_PORT)
    receiver.wait_for_connection()
    
    # 디텍터 초기화
    detector = RealtimeSpatialAvgDetector(k_factor=3.5, ma_window=50, cooldown_time=0.2, fs=400.0)
    
    print("실시간 다채널 통합 피크 감지 루프 가동")
    while True:
        batch = receiver.receive_batch()
        
        if batch is not None:
            # 실시간 피크 판별 수행
            if detector.process_batch(batch):
                print("[TRIGGER] 근전도 피크 감지 완료")
        else:
            time.sleep(0.001)

if __name__ == "__main__":
    main_execution()