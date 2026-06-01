import time
import collections
import numpy as np
import torch
import torch.nn as nn
from network import EMGReceiver
from config import SERVER_PORT, CHANNELS, RIGHT_HAND_KEYS

# ==============================================================================
# [실시간 추론 제어 튜닝 파라미터]
# ==============================================================================
WINDOW_SIZE = 150          
STEP_SIZE = 10             
SHORT_TERM_LEN = 20        

THRESHOLD_MULTIPLIER = 2.5 
BASELINE_ALPHA = 0.005     
WARMUP_FRAMES = 800        # 실시간 글로벌 정규화 기준값 산출을 위한 초기 데이터 확보량 (400Hz 기준 2초)

COOLDOWN_FRAMES = 80       
MIN_CONFIDENCE = 0.60      
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


class ContinuousSlidingWindowEngine:
    def __init__(self, model_path='best_emg_model.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.raw_labels = sorted(list(set(RIGHT_HAND_KEYS.values())))
        self.idx_to_label = {idx: raw_id for idx, raw_id in enumerate(self.raw_labels)}
        self.inv_key_map = {v: k for k, v in RIGHT_HAND_KEYS.items()}
        self.num_classes = len(self.raw_labels)
        
        self.model = HybridCNNLSTM(num_channels=CHANNELS, num_classes=self.num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        
        self.frame_buffer = []
        self.cooldown_counter = 0
        self.total_samples = 0
        
        self.is_warmed_up = False
        self.session_mean = None
        self.session_std = None
        self.baseline_energy = 0.0
        
        self.last_energy = 0.0
        self.is_rising = False
        self.last_predicted_key = None 

    def process_stream(self, new_data_chunk):
        chunk_array = np.array(new_data_chunk).reshape(-1, CHANNELS)
        self.frame_buffer.extend(chunk_array.tolist())
        self.total_samples += chunk_array.shape[0]

        # 1. 예열 구간: 실시간 글로벌 정규화 기준 산출
        if not self.is_warmed_up:
            if len(self.frame_buffer) < WARMUP_FRAMES:
                return 
            
            warmup_arr = np.array(self.frame_buffer[:WARMUP_FRAMES])
            self.session_mean = np.mean(warmup_arr, axis=0, keepdims=True)
            self.session_std = np.std(warmup_arr, axis=0, keepdims=True) + 1e-8
            
            norm_warmup = (warmup_arr - self.session_mean) / self.session_std
            tkeo_w = np.zeros_like(norm_warmup)
            tkeo_w[1:-1] = norm_warmup[1:-1]**2 - norm_warmup[:-2] * norm_warmup[2:]
            tkeo_w = np.mean(np.clip(tkeo_w, 0, None), axis=1)
            self.baseline_energy = np.percentile(tkeo_w, 20) + 1e-6
            
            self.is_warmed_up = True
            self.frame_buffer = self.frame_buffer[WARMUP_FRAMES:] 
            print(f"▶ [캘리브레이션 완료] 통계적 기저선 확정: {self.baseline_energy:.4f}")
            return

        # 2. 정상 추론: 확정된 세션 기준값으로 실시간 정규화 처리
        while len(self.frame_buffer) >= WINDOW_SIZE:
            raw_window = np.array(self.frame_buffer[:WINDOW_SIZE])
            if self.cooldown_counter > 0:
                self.cooldown_counter = max(0, self.cooldown_counter - STEP_SIZE)
            
            normalized_window = (raw_window - self.session_mean) / self.session_std
            
            tkeo_array = np.zeros_like(normalized_window)
            tkeo_array[1:-1] = normalized_window[1:-1]**2 - normalized_window[:-2] * normalized_window[2:]
            tkeo_array = np.mean(np.clip(tkeo_array, 0, None), axis=1) 
            
            current_energy = np.mean(tkeo_array[-SHORT_TERM_LEN:])
            
            dynamic_threshold = self.baseline_energy * THRESHOLD_MULTIPLIER 

            if current_energy < dynamic_threshold:
                self.baseline_energy = (1 - BASELINE_ALPHA) * self.baseline_energy + BASELINE_ALPHA * current_energy
                self.is_rising = False
            else:
                if current_energy > self.last_energy:
                    self.is_rising = True
                elif current_energy < self.last_energy and self.is_rising:
                    if self.cooldown_counter == 0:
                        self.execute_inference(normalized_window)
                        self.cooldown_counter = COOLDOWN_FRAMES
                    self.is_rising = False

            self.last_energy = current_energy
            self.frame_buffer = self.frame_buffer[STEP_SIZE:]

    def execute_inference(self, normalized_window):
        with torch.no_grad():
            tensor_input = torch.tensor(normalized_window, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits = self.model(tensor_input)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
            
            if confidence >= MIN_CONFIDENCE:
                event_id = self.idx_to_label.get(pred_idx, None)
                target_key = self.inv_key_map.get(event_id, "Unknown")
                
                if target_key == self.last_predicted_key:
                    return
                
                self.last_predicted_key = target_key
                print(f"▶ [타건 감지] 예측 키: '{target_key}' | 신뢰도: {confidence*100:.1f}%")

def main_run():
    receiver = EMGReceiver(SERVER_PORT)
    receiver.wait_for_connection()
    
    engine = ContinuousSlidingWindowEngine()
    print("EMG 피크 기반 타건 추론 엔진 가동 시작")
    print("※ 캘리브레이션을 위해 최초 2초간 손을 움직이지 마십시오.")
    
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
    
    #  데이터셋 전체분석은 전체 데이터셋 값을 기준으로 정