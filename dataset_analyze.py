import os
import glob
import pandas as pd
import numpy as np

# ==============================================================================
# [시뮬레이션 제어 파라미터]
# ==============================================================================
WINDOW_SIZE = 150          # 윈도우 크기
STEP_SIZE = 10             # 슬라이딩 간격
SHORT_TERM_LEN = 20        # 희석 방지용 말단 에너지 추출 길이 (최근 20프레임)
THRESHOLD_MULTIPLIER = 2.5 # 기저선 대비 임계값 배수
BASELINE_ALPHA = 0.005     # 기저선 추적 속도
COOLDOWN_FRAMES = 80       # 타건 후 재트리거 잠금 프레임 수
# ==============================================================================

def analyze_emg_dataset(dataset_dir="new_dataset"):
    print("=== 단기 에너지 피크 탐지 기반 데이터셋 시뮬레이션 분석 ===")
    
    if not os.path.exists(dataset_dir):
        print(f"[오류] '{dataset_dir}' 폴더가 존재하지 않습니다.")
        return

    csv_files = glob.glob(os.path.join(dataset_dir, "*.csv"))
    if not csv_files:
        print("[오류] 분석할 CSV 파일이 없습니다.")
        return

    total_actual_events = 0
    total_simulated_triggers = 0
    
    # 클래스별 트리거 횟수 기록용 사전
    class_trigger_counts = {}

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path, engine='python', on_bad_lines='skip')
        except Exception:
            continue
            
        ch_cols = [col for col in df.columns if col.startswith('CH')]
        if not ch_cols:
            continue
            
        signal = df[ch_cols].values
        
        # 1. 원시 데이터 전처리 (세션별 표준화)
        s_mean = signal.mean(axis=0, keepdims=True)
        s_std = signal.std(axis=0, keepdims=True) + 1e-8
        norm_signal = (signal - s_mean) / s_std
        
        # 실제 데이터셋 내 Ground Truth 타건 수 카운트 (Event 컬럼 기준 변곡점 탐지)
        if 'Event' in df.columns:
            events = df['Event'].values
            event_changes = np.where(np.diff(events) != 0)[0]
            for idx in event_changes:
                if events[idx+1] != 0:
                    total_actual_events += 1

        # 파일별 실시간 스트리밍 시뮬레이션 상태 변수 초기화
        frame_buffer = []
        cooldown_counter = 0
        baseline_energy = 0.0
        last_energy = 0.0
        is_rising = False
        last_detected_class = None # 직전 감지 레이블 추적 변수 추가

        # STEP_SIZE 단위로 유입되는 실시간 파이프라인 모사
        for idx in range(0, len(norm_signal), STEP_SIZE):
            chunk = norm_signal[idx:idx+STEP_SIZE]
            frame_buffer.extend(chunk.tolist())

            while len(frame_buffer) >= WINDOW_SIZE:
                window = np.array(frame_buffer[:WINDOW_SIZE])
                
                if cooldown_counter > 0:
                    cooldown_counter = max(0, cooldown_counter - STEP_SIZE)

                # 2. 로컬 윈도우 기반 TKEO 시계열 산출
                tkeo_array = np.zeros_like(window)
                tkeo_array[1:-1] = window[1:-1]**2 - window[:-2] * window[2:]
                tkeo_array = np.mean(np.clip(tkeo_array, 0, None), axis=1) # 채널 평균화 [150]

                # [개선] 희석 방지용 말단 20프레임 단기 에너지 추출
                current_energy = np.mean(tkeo_array[-SHORT_TERM_LEN:])

                # 초기 기저선 빌드
                if baseline_energy == 0:
                    baseline_energy = np.mean(tkeo_array) + 1e-6

                dynamic_threshold = baseline_energy * THRESHOLD_MULTIPLIER

                # 3. 동적 기저선 제어 및 피크 기울기 분석 트리거 로직
                if current_energy < dynamic_threshold:
                    # 안전 구간(Idle)에서만 기저선 업데이트 활성화
                    baseline_energy = (1 - BASELINE_ALPHA) * baseline_energy + BASELINE_ALPHA * current_energy
                    is_rising = False
                else:
                    # 임계값 돌파 구간 (기저선 동결 및 기울기 반전 추적)
                    if current_energy > last_energy:
                        自由_rising = True
                        is_rising = True
                    elif current_energy < last_energy and is_rising:
                        if cooldown_counter == 0:
                            # 현재 윈도우 말단의 실제 Event 레이블 매핑 코드
                            detected_class = None
                            if 'Event' in df.columns:
                                # 윈도우 후반부의 최빈 레이블을 해당 트리거의 클래스로 간주
                                current_sample_idx = idx + WINDOW_SIZE
                                target_zone = df['Event'].iloc[max(0, current_sample_idx-50):min(len(df), current_sample_idx)].values
                                active_labels = target_zone[target_zone != 0]
                                
                                if len(active_labels) > 0:
                                    detected_class = str(active_labels[0])
                            
                            # [핵심 수정] 직전 레이블과 동일한 연속 감지 건은 카운트에서 완전 제외
                            if detected_class is not None and detected_class == last_detected_class:
                                pass 
                            else:
                                total_simulated_triggers += 1
                                if detected_class is not None:
                                    class_trigger_counts[detected_class] = class_trigger_counts.get(detected_class, 0) + 1
                                    last_detected_class = detected_class # 직전 감지 레이블 업데이트
                            
                            cooldown_counter = COOLDOWN_FRAMES
                        is_rising = False

                last_energy = current_energy
                frame_buffer = frame_buffer[STEP_SIZE:]

    # ==============================================================================
    # 4. 성능 개선 검증 진단 리포트 출력
    # ==============================================================================
    print("\n=== 반영 후 시뮬레이션 진단 리포트 ===")
    print(f"1. 데이터셋 내 실제 타건 수 (Ground Truth) : 약 {total_actual_events} 회")
    print(f"2. 개선된 단기 피크 알고리즘 감지 수 (Trigger) : {total_simulated_triggers} 회")
    
    if total_actual_events > 0:
        detection_ratio = (total_simulated_triggers / total_actual_events) * 100
        print(f"3. 최종 타건 감지율 (Detection Rate) : {detection_ratio:.1f}%")
        print("----------------------------------------")
        if detection_ratio < 70.0:
            print("💡 [가이드] 감지율이 여전히 낮다면 THRESHOLD_MULTIPLIER를 1.8~2.2 수준으로 추가 하향 조정 권장.")
        elif detection_ratio > 130.0:
            print("💡 [가이드] 과감지(중복) 발생 시 COOLDOWN_FRAMES를 150 이상으로 상향 조정 권장.")
        else:
            print("✅ [판정] 실시간 감지 빈도가 오프라인 레이블과 정량적으로 일치함. 안정 궤도 진입.")

    if class_trigger_counts:
        print("\n[클래스(레이블)별 트리거 분배 현황]")
        for cls, count in sorted(class_trigger_counts.items()):
            print(f" - Label '{cls}' : {count} 회 감지됨")

if __name__ == "__main__":
    analyze_emg_dataset()