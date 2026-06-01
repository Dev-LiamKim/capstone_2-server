import time
import torch
import numpy as np
from scipy import signal
from network import EMGReceiver
from config import SERVER_PORT, CHANNELS, BATCH_SIZE, RIGHT_HAND_KEYS
from train1 import CNNLSTM # 학습 코드가 작성된 파일에서 모델 아키텍처 직접 임포트

class RealtimeEMGClassifier:
    def __init__(self, model_path='best_emg_model.pt', k_factor=3.5, ma_window=50, cooldown_time=0.2, fs=400.0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. 학습 환경 기반 자판 레이블 사전 구축 (train1.py 스키마 완전 동기화)
        self.raw_labels = sorted(list(set(RIGHT_HAND_KEYS.values())))
        self.idx_to_label = {idx: raw_id for idx, raw_id in enumerate(self.raw_labels)}
        self.inv_key_map = {v: k for k, v in RIGHT_HAND_KEYS.items()}
        self.num_classes = len(self.raw_labels)
        
        # 2. 파이토치 모델 로드 및 평가 모드 전환
        self.model = CNNLSTM(num_channels=CHANNELS, num_classes=self.num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        
        # 3. 비대칭 윈도우 파라미터 셋팅 (PRE_EVENT: 90, POST_EVENT: 60)
        self.window_size = 150
        self.pre_event = 90
        self.post_event = 60
        self.raw_data_buffer = np.zeros((CHANNELS, 300)) # 충분한 크기의 링 버퍼 생성
        self.post_peak_counter = -1
        
        # 4. 피크 감지용 하이퍼파라미터 및 실시간 인과적 필터 세팅
        self.k_factor = k_factor
        self.cooldown_samples = int(cooldown_time * fs)
        self.current_cooldown = 0
        self.history_buffer = np.zeros((CHANNELS, ma_window))
        
        nyquist = 0.5 * fs
        high_cut = 20.0 / nyquist
        self.b, self.a = signal.butter(4, high_cut, btype='high')
        zi_1d = signal.lfilter_zi(self.b, self.a)
        self.filter_state = np.zeros((CHANNELS, len(zi_1d)))

    def process_samples(self, current_batch):
        """배치 데이터를 받아 샘플 단위 인과적 필터링 및 비대칭 윈도우 기반 분류 수행"""
        filtered_batch, self.filter_state = signal.lfilter(
            self.b, self.a, current_batch, axis=1, zi=self.filter_state
        )
        rectified_batch = np.abs(filtered_batch)
        spatial_avg_stream = np.mean(rectified_batch, axis=0)
        
        for i in range(len(spatial_avg_stream)):
            # 원시 데이터 타임라인 동기화 업데이트
            self.raw_data_buffer = np.roll(self.raw_data_buffer, -1, axis=1)
            self.raw_data_buffer[:, -1] = current_batch[:, i]
            
            # 피크 트리거 발생 후 후속 60샘플 수집 타이머 체크
            if self.post_peak_counter > 0:
                self.post_peak_counter -= 1
                if self.post_peak_counter == 0:
                    # 전방 90샘플 + 피크 시점 + 후방 59샘플 총 150샘플 슬라이싱
                    target_window = self.raw_data_buffer[:, -self.window_size:] 
                    self.execute_inference(target_window)
                    self.post_peak_counter = -1
            
            # 쿨다운 제한 처리
            if self.current_cooldown > 0:
                self.current_cooldown -= 1
                self.update_history_buffer(rectified_batch[:, i])
                continue
                
            # 동적 베이스라인 기반 피크 검출
            current_baseline = np.mean(self.history_buffer)
            current_energy = spatial_avg_stream[i]
            
            if current_baseline > 0 and current_energy > (current_baseline * self.k_factor):
                self.current_cooldown = self.cooldown_samples
                self.post_peak_counter = self.post_event  # 후방 60샘플 수집 카운트다운 시작
                print("\n[TRIGGER] 피크 감지 완료 -> 후속 데이터 60샘플 수집 대기 중...")
                
            self.update_history_buffer(rectified_batch[:, i])

    def update_history_buffer(self, sample):
        self.history_buffer = np.roll(self.history_buffer, -1, axis=1)
        self.history_buffer[:, -1] = sample

    def execute_inference(self, window_data):
        """(8, 150) 원시 윈도우 데이터를 전처리 및 전치하여 딥러닝 모델 추론 진행"""
        # 1. 차원 전치: (8, 150) -> (150, 8) [Time, Channels] 매핑
        window_transposed = window_data.T
        
        # 2. 실시간 Z-score 표준화 수행 (각 채널별 시간축 기준 스케일링)
        win_mean = window_transposed.mean(axis=0, keepdims=True)
        win_std = window_transposed.std(axis=0, keepdims=True) + 1e-8
        normalized_window = (window_transposed - win_mean) / win_std
        
        # 3. 파이토치 입력 포맷 변환 및 배치 차원 할당 -> (1, 150, 8)
        model_input = torch.tensor(normalized_window, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 4. 모델 순방향 추론 (Gradient 연산 제외)
        with torch.no_grad():
            outputs = self.model(model_input)
            pred_idx = outputs.argmax(1).item()
            
        # 5. 2단계 역매핑을 통한 아스키 문자 자판 변환
        event_id = self.idx_to_label.get(pred_idx, None)
        target_key = self.inv_key_map.get(event_id, "Unknown")
        
        print(f"▶ [인프런스 완료] 입력된 키보드 자판: '{target_key}' (이벤트 클래스 코드: {event_id})")

def main_run():
    receiver = EMGReceiver(SERVER_PORT)
    receiver.wait_for_connection()
    
    classifier_system = RealtimeEMGClassifier()
    print("EMG 실시간 비대칭 윈도우 인공지능 분류 루프 시작")
    
    while True:
        batch = receiver.receive_batch()
        if batch is not None:
            classifier_system.process_samples(batch)
        else:
            time.sleep(0.001)

if __name__ == "__main__":
    main_run()