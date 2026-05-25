# check_class_jitter.py
import os
import glob
import pandas as pd
import numpy as np
from config import RIGHT_HAND_KEYS, CHANNELS

def analyze_class_jitter(dataset_dir="for_analyze"):
    """
    데이터셋 내 모든 CSV 파일을 스캔하여 
    각 자판 클래스별 평균 싱크 오차 및 지터 변동폭(표준편차)을 정밀 계산
    """
    csv_files = glob.glob(os.path.join(dataset_dir, "*.csv"))
    if not csv_files:
        print(f"[ERROR] '{dataset_dir}' 폴더에 CSV 파일이 존재하지 않습니다.")
        return

    # 마커 ID 오프셋 역산 매핑
    inv_keys = {v: k for k, v in RIGHT_HAND_KEYS.items()}
    
    # 클래스별 오차 컨테이너 초기화
    class_errors = {k: [] for k in RIGHT_HAND_KEYS.keys()}
    class_counts = {k: 0 for k in RIGHT_HAND_KEYS.keys()}
    class_peaks = {k: 0 for k in RIGHT_HAND_KEYS.keys()}
    
    ch_columns = [f"CH{i}" for i in range(CHANNELS)]
    
    # 350Hz 하드웨어 사양 반영 (샘플당 약 2.857ms)
    sampling_rate = 350.0
    ms_per_sample = 1000.0 / sampling_rate
    half_window_samples = int(sampling_rate * 0.5) # 전후 500ms 구간 (총 1000ms 윈도우)

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            event_indices = df[df['Event'] != 0].index.tolist()
            
            for idx in event_indices:
                event_id = df.loc[idx, 'Event']
                key_name = inv_keys.get(event_id, None)
                if key_name is None:
                    continue
                
                class_counts[key_name] += 1
                
                # 1000ms 탐색 윈도우 슬라이싱
                search_start = max(0, idx - half_window_samples)
                search_end = min(len(df), idx + half_window_samples)
                
                window_df = df.iloc[search_start:search_end]
                emg_signals = window_df[ch_columns].values
                row_energies = np.sum(np.square(emg_signals), axis=1)
                
                # Z-Score 기반 통계적 피크 검증 (노이즈와 실제 수축 분리)
                mean_energy = np.mean(row_energies)
                std_energy = np.std(row_energies)
                max_energy = np.max(row_energies)
                
                z_score = (max_energy - mean_energy) / std_energy if std_energy > 0 else 0
                
                # 유효 피크 조건 충족 시 시차 저장 ($Z > 3.5$)
                if z_score > 3.5:
                    class_peaks[key_name] += 1
                    local_peak_idx = search_start + np.argmax(row_energies)
                    sample_error = local_peak_idx - idx
                    ms_error = sample_error * ms_per_sample
                    class_errors[key_name].append(ms_error)
                    
        except Exception as e:
            print(f"[파일 처리 실패] {os.path.basename(file_path)}: {e}")

    # ==============================================================================
    # 클래스별 독립 지터 변동폭 결과 리포트 출력
    # ==============================================================================
    print("\n" + "="*95)
    print(f" {'각 자판 클래스별 개별 지터(Jitter) 변동폭 분석 리포트':^75}")
    print("="*95)
    print(f"{'키 명칭':^8} | {'총 마커 수':^10} | {'피크 확인 수':^12} | {'피크 존재율':^12} | {'평균 싱크 오차':^14} | {'★개별 지터 변동폭 (표준편차)'}")
    print("-"*95)
    
    for key_name in sorted(class_errors.keys()):
        total = class_counts[key_name]
        detected = class_peaks[key_name]
        errors = class_errors[key_name]
        
        if total == 0:
            print(f"  {key_name:^6} | {0:^14} | {0:^16} | {'0.0%':^16} | {'N/A':^16} | 데이터 없음")
            continue
            
        exist_rate = (detected / total) * 100
        avg_error = np.mean(errors) if errors else 0.0
        jitter_val = np.std(errors) if errors else 0.0
        
        avg_err_str = f"{avg_error:>11.2f} ms" if errors else f"{'N/A':^14}"
        jitter_str = f"{jitter_val:>15.2f} ms" if errors else f"{'N/A':^22}"
        
        print(f"  {key_name:^6} | {total:^14} | {detected:^16} | {exist_rate:>14.1f}% | {avg_err_str} | {jitter_str}")
        
    print("="*95)

if __name__ == "__main__":
    analyze_class_jitter()