import time
import collections
import numpy as np
import torch
import torch.nn as nn
from network import EMGReceiver
from config import SERVER_PORT, CHANNELS, RIGHT_HAND_KEYS

# ==============================================================================
# 1. 하이브리드 모델 아키텍처 (독립형 구조)
# ==============================================================================
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        attn_weights = torch.softmax(self.attn(lstm_output), dim=1) 
        context = torch.sum(lstm_output * attn_weights, dim=1)     
        return context

class ResBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))

class HandcraftedFeatures(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1) + 1e-8)
        mav = torch.mean(torch.abs(x), dim=-1)
        wl = torch.sum(torch.abs(x[:, :, 1:] - x[:, :, :-1]), dim=-1)
        signs = torch.sign(x)
        zc = torch.sum(torch.abs(signs[:, :, 1:] - signs[:, :, :-1]) == 2, dim=-1).float()
        return torch.cat([rms, mav, wl, zc], dim=-1)

class HybridCNNLSTM(nn.Module):
    def __init__(self, num_channels, num_classes):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv1d(num_channels, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.res_block1 = ResBlock1D(128)
        self.pool1 = nn.MaxPool1d(2) 
        
        self.mid_conv = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.res_block2 = ResBlock1D(256) 

        self.lstm = nn.LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size=256)
        self.dropout_lstm = nn.Dropout(0.4)
        
        self.hc_extractor = HandcraftedFeatures()
        hc_dim = num_channels * 4
        combined_dim = 256 + hc_dim
        
        self.feature_bn = nn.BatchNorm1d(combined_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x_permuted = x.permute(0, 2, 1)
        hc_feats = self.hc_extractor(x_permuted)
        
        dl_x = self.init_conv(x_permuted)
        dl_x = self.res_block1(dl_x)
        dl_x = self.pool1(dl_x) 
        
        dl_x = self.mid_conv(dl_x)
        dl_x = self.res_block2(dl_x) 
        
        dl_x = dl_x.permute(0, 2, 1) 
        dl_x, _ = self.lstm(dl_x) 
        dl_feats = self.attention(dl_x)
        dl_feats = self.dropout_lstm(dl_feats)
        
        combined_feats = torch.cat([dl_feats, hc_feats], dim=-1)
        combined_feats = self.feature_bn(combined_feats)
        return self.classifier(combined_feats)


# ==============================================================================
# 2. 실시간 추론 엔진 클래스 (시뮬레이터 연동용 인라인 정규화 적용)
# ==============================================================================
WINDOW_SIZE = 150
PRE_EVENT = 90
BUFFER_SIZE = 500
PEAK_WAIT_FRAMES = 100
VOTING_SHIFTS = [-5, 0, 5]

class RealtimeInferenceEngine:
    def __init__(self, model_path='best_emg_model.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.raw_labels = sorted(list(set(RIGHT_HAND_KEYS.values())))
        self.idx_to_label = {idx: raw_id for idx, raw_id in enumerate(self.raw_labels)}
        self.inv_key_map = {v: k for k, v in RIGHT_HAND_KEYS.items()}
        self.num_classes = len(self.raw_labels)
        
        self.model = HybridCNNLSTM(num_channels=CHANNELS, num_classes=self.num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        
        self.buffer = collections.deque(maxlen=BUFFER_SIZE)
        self.state = 'IDLE'
        self.wait_counter = 0
        self.baseline_energy = 0.0
        self.alpha = 0.01
        
        # [추가] 수신 체크용 총 샘플(프레임) 카운터
        self.total_samples = 0

    def compute_tkeo(self, signal_2d):
        tkeo = np.zeros_like(signal_2d)
        tkeo[1:-1] = signal_2d[1:-1]**2 - signal_2d[:-2] * signal_2d[2:]
        tkeo = np.clip(tkeo, 0, None)
        return np.mean(tkeo, axis=1)

    def process_stream(self, new_data_chunk):
        # 1. 데이터를 루프 없이 한 번에 버퍼에 연장 (O(1) 처리)
        self.buffer.extend(new_data_chunk)
        self.total_samples += len(new_data_chunk)
        
        if self.total_samples % 175 < len(new_data_chunk):
            print(f"[수신 상태 확인] 현재 {self.total_samples}프레임 수신 중...")

        if len(self.buffer) < BUFFER_SIZE: return

        # 2. 배치 단위 1회 연산으로 CPU 과부하 방지
        current_raw_data = np.array(self.buffer)
        energy_profile = self.compute_tkeo(current_raw_data)
        current_energy = energy_profile[-1]

        if self.state == 'IDLE':
            self.baseline_energy = (1 - self.alpha) * self.baseline_energy + self.alpha * current_energy
            
            # 절대 상수를 제거하고 순수 비율(4.0배)로만 검증
            if current_energy > (self.baseline_energy * 4.0): 
                self.state = 'WAIT_PEAK'
                self.wait_counter = 0
                
        elif self.state == 'WAIT_PEAK':
            self.wait_counter += len(new_data_chunk)
            if self.wait_counter >= PEAK_WAIT_FRAMES:
                self.extract_and_infer(current_raw_data, energy_profile)
                self.state = 'COOLDOWN'
                self.wait_counter = 0
                
        elif self.state == 'COOLDOWN':
            self.wait_counter += len(new_data_chunk)
            if self.wait_counter >= 100: 
                self.state = 'IDLE'

    def extract_and_infer(self, current_raw_data, energy_profile):
        search_start = BUFFER_SIZE - PEAK_WAIT_FRAMES - 30
        peak_idx = search_start + np.argmax(energy_profile[search_start:BUFFER_SIZE])
        base_start = peak_idx - PRE_EVENT
        
        ensemble_probs = []
        
        with torch.no_grad():
            for shift in VOTING_SHIFTS:
                start_idx = base_start + shift
                end_idx = start_idx + WINDOW_SIZE
                if start_idx < 0 or end_idx > BUFFER_SIZE: continue
                
                # [핵심] 슬라이싱된 150프레임 추론 윈도우 내부에서만 독립적 Z-score 정규화 수행
                window = current_raw_data[start_idx:end_idx]
                w_mean = np.mean(window, axis=(0, 1), keepdims=True)
                w_std = np.std(window, axis=(0, 1), keepdims=True) + 1e-8
                normalized_window = (window - w_mean) / w_std
                
                tensor_input = torch.tensor(normalized_window, dtype=torch.float32).unsqueeze(0).to(self.device)
                probs = torch.softmax(self.model(tensor_input), dim=1).cpu().numpy()[0]
                ensemble_probs.append(probs)

        if ensemble_probs:
            avg_probs = np.mean(ensemble_probs, axis=0)
            pred_idx = np.argmax(avg_probs)
            confidence = avg_probs[pred_idx]
            
            if confidence > 0.5: # 판정 신뢰도 기준선
                event_id = self.idx_to_label.get(pred_idx, None)
                target_key = self.inv_key_map.get(event_id, "Unknown")
                print(f"▶ [실시간 감지 성공] 예측 키: '{target_key}' | 신뢰도: {confidence*100:.1f}%")

# ==============================================================================
# 3. 메인 실행 루프
# ==============================================================================
def main_run():
    receiver = EMGReceiver(SERVER_PORT)
    receiver.wait_for_connection()
    
    engine = RealtimeInferenceEngine()
    print("EMG 실시간 피크 동기화 추론 엔진 가동 (하이브리드 독립형)")
    print("※ 베이스라인 캘리브레이션을 위해 최초 1~2초간 손을 움직이지 마십시오.")
    
    try:
        while True:
            data_chunk = receiver.receive_batch() 
            if data_chunk is not None and len(data_chunk) > 0:
                engine.process_stream(data_chunk)
            else:
                time.sleep(0.001)
    except KeyboardInterrupt:
        print("\n추론 엔진 종료")

if __name__ == "__main__":
    main_run()