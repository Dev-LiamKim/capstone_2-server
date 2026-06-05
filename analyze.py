import os
import glob
import torch
import numpy as np
import pandas as pd
from collections import deque
from config import CHANNELS, RIGHT_HAND_KEYS
from train1 import CNNLSTM  # 독립형 하이브리드 코드가 train1.py에 있다고 가정

# ==============================================================================
# [분석할 타겟 파라미터] - 이곳의 값을 바꿔가며 테스트해 보세요.
# ==============================================================================
WINDOW_SIZE = 150
STEP_SIZE = 10
THRESHOLD_MULTIPLIER = 1.5
BASELINE_ALPHA = 0.005
COOLDOWN_FRAMES = 130
MIN_CONFIDENCE = 0.60
# ==============================================================================

def compute_tkeo_energy(window_data):
    tkeo = np.zeros_like(window_data)
    tkeo[1:-1] = window_data[1:-1]**2 - window_data[:-2] * window_data[2:]
    return np.mean(np.clip(tkeo, 0, None))

def run_offline_tuning_simulation(dataset_dir="new_dataset", model_path="best_emg_model.pt"):
    print(f"=== EMG 파라미터 튜닝 진단 시작 ===")
    print(f"적용 파라미터: Multiplier={THRESHOLD_MULTIPLIER}, Cooldown={COOLDOWN_FRAMES}, MinConf={MIN_CONFIDENCE}\n")

    if not os.path.exists(dataset_dir):
        print(f"[오류] '{dataset_dir}' 폴더가 없습니다.")
        return

    csv_files = glob.glob(os.path.join(dataset_dir, "*.csv"))
    if not csv_files:
        print("[오류] 분석할 데이터셋 파일이 없습니다.")
        return

    # 모델 세팅
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw_labels = sorted(list(set(RIGHT_HAND_KEYS.values())))
    idx_to_label = {idx: raw_id for idx, raw_id in enumerate(raw_labels)}
    inv_key_map = {v: k for k, v in RIGHT_HAND_KEYS.items()}
    num_classes = len(raw_labels)

    # train1.py 내부에 HybridCNNLSTM 이 선언되어 있다고 가정합니다.
    # 만약 에러가 난다면 RealtimeMonitor.py 내부의 독립 모델 클래스를 복사해 넣으셔도 무방합니다.
    try:
        from RealtimeMonitor import HybridCNNLSTM
        model = HybridCNNLSTM(num_channels=CHANNELS, num_classes=num_classes).to(device)
    except ImportError:
        print("HybridCNNLSTM 임포트 실패. 구조를 직접 코드 안에 넣으십시오.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 분석용 통계 변수
    total_events_in_dataset = 0
    total_triggers = 0
    confidence_drops = 0
    detected_keys_counter = {k: 0 for k in RIGHT_HAND_KEYS.keys()}

    for file_path in csv_files:
        print(f"-> 시뮬레이션 중: {os.path.basename(file_path)}")
        try:
            df = pd.read_csv(file_path, engine='python', on_bad_lines='skip')
        except Exception:
            continue
            
        ch_cols = [col for col in df.columns if col.startswith('CH')]
        if not ch_cols: continue
        
        signal = df[ch_cols].values.astype(np.float32)
        
        # Ground Truth 이벤트 개수 파악
        if 'Event' in df.columns:
            # 연속된 이벤트 구간을 1개로 계산하기 위한 간단한 로직
            events = df['Event'].values
            event_changes = np.where(np.diff(events) != 0)[0]
            for idx in event_changes:
                if events[idx+1] != 0:
                    total_events_in_dataset += 1

        # 상태 변수 초기화 (파일 단위)
        frame_buffer = []
        cooldown_counter = 0
        baseline_energy = 0.0
        last_energy = 0.0
        is_rising = False

        # 스트림 시뮬레이션
        for chunk_idx in range(0, len(signal), STEP_SIZE):
            chunk = signal[chunk_idx:chunk_idx+STEP_SIZE]
            frame_buffer.extend(chunk.tolist())

            while len(frame_buffer) >= WINDOW_SIZE:
                window = np.array(frame_buffer[:WINDOW_SIZE])
                if cooldown_counter > 0:
                    cooldown_counter = max(0, cooldown_counter - STEP_SIZE)
                
                current_energy = compute_tkeo_energy(window)
                
                if baseline_energy == 0:
                    baseline_energy = current_energy + 1e-6
                else:
                    if current_energy < (baseline_energy * 1.5):
                        baseline_energy = 0.90 * baseline_energy + 0.10 * current_energy
                    else:
                        baseline_energy = (1 - BASELINE_ALPHA) * baseline_energy + BASELINE_ALPHA * current_energy

                dynamic_threshold = baseline_energy * THRESHOLD_MULTIPLIER

                # 피크 감지
                if current_energy > dynamic_threshold:
                    if current_energy > last_energy:
                        is_rising = True
                    elif current_energy < last_energy and is_rising:
                        if cooldown_counter == 0:
                            # --- 추론 실행 ---
                            with torch.no_grad():
                                w_mean = np.mean(window, axis=0, keepdims=True)
                                w_std = np.std(window, axis=0, keepdims=True) + 1e-8
                                norm_window = (window - w_mean) / w_std
                                tensor_input = torch.tensor(norm_window, dtype=torch.float32).unsqueeze(0).to(device)
                                probs = torch.softmax(model(tensor_input), dim=1).cpu().numpy()[0]
                                pred_idx = np.argmax(probs)
                                confidence = probs[pred_idx]
                                
                                if confidence >= MIN_CONFIDENCE:
                                    total_triggers += 1
                                    event_id = idx_to_label.get(pred_idx, None)
                                    target_key = inv_key_map.get(event_id, "Unknown")
                                    if target_key in detected_keys_counter:
                                        detected_keys_counter[target_key] += 1
                                else:
                                    confidence_drops += 1 # 에너지는 피크를 쳤으나 신뢰도가 낮아 버려짐
                                    
                            cooldown_counter = COOLDOWN_FRAMES
                        is_rising = False
                else:
                    is_rising = False

                last_energy = current_energy
                frame_buffer = frame_buffer[STEP_SIZE:]

    # ==============================================================================
    # 진단 리포트 출력
    # ==============================================================================
    print("\n============================================================")
    print("   [분석 리포트] 피크 감지 및 추론 파라미터 시뮬레이션 결과")
    print("============================================================")
    print(f" 1. 데이터셋 내 실제 타건 수 (Ground Truth) : 약 {total_events_in_dataset} 회")
    print(f" 2. 시뮬레이터가 감지한 총 타건 횟수 (Trigger) : {total_triggers} 회")
    print(f" 3. 신뢰도(Confidence) 미달로 폐기된 피크 수 : {confidence_drops} 회")
    print("------------------------------------------------------------")
    
    # 평가 코멘트 생성
    if total_events_in_dataset > 0:
        ratio = total_triggers / total_events_in_dataset
        print(f" [진단] 실제 타건 대비 감지율 : {ratio*100:.1f}%")
        
        if ratio > 1.2:
            print(" ⚠️ 중복 감지가 심합니다. COOLDOWN_FRAMES를 늘리거나 THRESHOLD_MULTIPLIER를 높이세요.")
        elif ratio < 0.8:
            print(" ⚠️ 감지 누락이 심합니다. THRESHOLD_MULTIPLIER를 낮추세요.")
        else:
            print(" ✅ 감지 빈도가 실제 타건 수와 매우 유사합니다. 안정적인 설정입니다.")
            
    print("\n [클래스별 감지 분포]")
    for k, v in detected_keys_counter.items():
        if v > 0:
            print(f" - Key '{k}' : {v} 회 감지")

if __name__ == "__main__":
    run_offline_tuning_simulation()