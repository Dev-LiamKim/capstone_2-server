# check_sync_class_report.py
import os
import glob
import pandas as pd
import numpy as np
from config import RIGHT_HAND_KEYS, CHANNELS

def generate_class_peak_report(dataset_dir="for_analyze"):
    """
    전체 데이터셋의 마커를 클래스(키 명칭)별로 분류하여
    물리적 피크 존재율, 평균 싱크 오차, 지터 변동 폭(표준편차)을 전수조사 리포트로 출력
    """
    csv_files = glob.glob(os.path.join(dataset_dir, "*.csv"))
    if not csv_files:
        print(f"[ERROR] '{dataset_dir}' 폴더에 CSV 파일이 존재하지 않습니다.")
        return

    # 마커 ID 오프셋 역산 매핑
    inv_keys = {v: k for k, v in RIGHT_HAND_KEYS.items()}
    
    # 클래스별 통계 데이터 구조 초기화
    class_stats = {k: {"errors": [], "total_events": 0, "detected_peaks": 0} for k in RIGHT_HAND_KEYS.keys()}
    ch_columns = [f"CH{i}" for i in range(CHANNELS)]
    
    # 250Hz 하드웨어 타이머 고정 사양 반영 (샘플당 4ms)
    sampling_rate = 250.0
    ms_per_sample = 1000.0 / sampling_rate
    half_window_samples = int(sampling_rate * 0.5)

    print(f"[INFO] 총 {len(csv_files)}개 파일 스캔 및 클래스별 피크 전수조사 분석 시작...\n")

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            event_indices = df[df['Event'] != 0].index.tolist()
            
            for idx in event_indices:
                event_id = df.loc[idx, 'Event']
                key_name = inv_keys.get(event_id, None)
                if key_name is None:
                    continue
                
                class_stats[key_name]["total_events"] += 1
                
                # 마커 전후 500ms 구간 (총 1000ms 윈도우) 추출
                search_start = max(0, idx - half_window_samples)
                search_end = min(len(df), idx + half_window_samples)
                
                window_df = df.iloc[search_start:search_end]
                emg_signals = window_df[ch_columns].values
                row_energies = np.sum(np.square(emg_signals), axis=1)
                
                # Z-Score 통계적 피크 검증 연산
                mean_energy = np.mean(row_energies)
                std_energy = np.std(row_energies)
                max_energy = np.max(row_energies)
                
                z_score = (max_energy - mean_energy) / std_energy if std_energy > 0 else 0
                
                # Z-Score 가 3.5 표준편차를 초과할 경우 유효 물리 피크로 인정
                if z_score > 3.5:
                    class_stats[key_name]["detected_peaks"] += 1
                    
                    # 정점 인덱스 추적 및 오차 역산 (ms)
                    local_peak_idx = search_start + np.argmax(row_energies)
                    sample_error = local_peak_idx - idx
                    ms_error = sample_error * ms_per_sample
                    class_stats[key_name]["errors"].append(ms_error)
                    
        except Exception as e:
            print(f"파일 처리 실패 ({os.path.basename(file_path)}): {e}")

    # ==============================================================================
    # 클래스별 상세 전수조사 리포트 출력 구역
    # ==============================================================================
    print("="*105)
    print(f" {'클래스별 근전도 물리적 피크 전수조사 보고서 (1000ms 윈도우 기준)':^85}")
    print("="*105)
    print(f"{'키 명칭':^8} | {'총 마커 발생':^10} | {'피크 검출 횟수':^12} | {'신호 유효 탐지율':^14} | {'평균 싱크 오차':^14} | {'지터 변동 폭(표준편차)'}")
    print("-"*105)
    
    global_total = 0
    global_detected = 0
    all_errors = []
    
    # 가독성을 위해 키 명칭 알파벳 순 정렬 출력
    for key_name in sorted(class_stats.keys()):
        stats = class_stats[key_name]
        total = stats["total_events"]
        detected = stats["detected_peaks"]
        errors = stats["errors"]
        
        global_total += total
        global_detected += detected
        all_errors.extend(errors)
        
        if total == 0:
            print(f"  {key_name:^6} | {0:^14} | {0:^16} | {'0.0%':^18} | {'N/A':^16} | 데이터 없음")
            continue
            
        detection_rate = (detected / total) * 100
        avg_err_str = f"{np.mean(errors):>11.2f} ms" if errors else f"{'N/A':^14}"
        std_err_str = f"{np.std(errors):>15.2f} ms" if errors else f"{'N/A':^18}"
        
        print(f"  {key_name:^6} | {total:^14} | {detected:^16} | {detection_rate:>16.1f}% | {avg_err_str} | {std_err_str}")
        
    print("="*105)
    
    # ==============================================================================
    # 전체 종합 요약 리포트 출력 구역
    # ==============================================================================
    if global_total > 0:
        global_rate = (global_detected / global_total) * 100
        print(f" [종합 요약]")
        print(f"  - 총 타건 마커 발생 횟수: {global_total}회")
        print(f"  - 실제 근수축 피크 확인된 횟수: {global_detected}회")
        print(f"  - 전체 신호 유효 탐지율: {global_rate:.1f}%")
        if all_errors:
            print(f"  - 유효 피크들의 종합 평균 싱크 오차: {np.mean(all_errors):.2f} ms")
            print(f"  - 종합 지터 변동 폭 (오차 표준편차): {np.std(all_errors):.2f} ms")
    print("="*105)

if __name__ == "__main__":
    generate_class_peak_report()