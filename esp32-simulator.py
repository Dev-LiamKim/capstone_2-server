import os
import socket
import time
import struct
import pandas as pd
import numpy as np
from config import SERVER_PORT, CHANNELS, BATCH_SIZE

def run_esp32_simulator(dataset_dir="new_dataset", fs=400.0):
    print("=== 가상 ESP32 송신 시뮬레이터 구동 ===")
    
    # 1. 수집 폴더 내 CSV 파일 탐색 및 데이터 로드
    if not os.path.exists(dataset_dir):
        print(f"[오류] '{dataset_dir}' 폴더가 없습니다.")
        return

    csv_files = [os.path.join(root, f) for root, dirs, files in os.walk(dataset_dir) for f in files if f.endswith('.csv')]
    if not csv_files:
        print("[오류] 전송할 CSV 파일이 존재하지 않습니다.")
        return

    print(f"로드 완료된 가상 데이터셋 파일 수: {len(csv_files)}개")

    # 2. 메인 분류기 서버(EMGReceiver)에 연결 시도
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    while True:
        try:
            client_sock.connect(('127.0.0.1', SERVER_PORT))
            print("▶ 실시간 분류기 서버 접속 성공. 데이터 스트리밍을 시작합니다.")
            break
        except ConnectionRefusedError:
            print("분류기 서버가 켜지지 않음. 1초 후 재시도...")
            time.sleep(1)

    # 3. CSV 데이터 스트리밍 루프
    # 주기 제어용 타임 인터벌 연산 (BATCH_SIZE 단위 송신 딜레이)
    send_interval = BATCH_SIZE / fs 

    try:
        for file_path in csv_files:
            print(f"현재 전송 중인 파일: {os.path.basename(file_path)}")
            
            # --- 파싱 에러(ParserError) 및 빈 파일 예외 처리 블록 ---
            if os.path.getsize(file_path) == 0:
                print(f"[경고] 빈 파일 건너뜀: {os.path.basename(file_path)}\n")
                continue
                
            try:
                # engine='python' 적용 및 에러 발생 라인 무시(on_bad_lines='skip')
                df = pd.read_csv(file_path, engine='python', encoding='utf-8-sig', on_bad_lines='skip')
            except Exception as e:
                print(f"[경고] 파일 읽기 실패 건너뜀: {os.path.basename(file_path)} - {e}\n")
                continue
            
            if df.empty:
                print(f"[경고] 유효한 데이터가 없음 건너뜀: {os.path.basename(file_path)}\n")
                continue
            # --------------------------------------------------------
            
            # Event 열 제외 후 순수 채널 데이터만 추출
            if 'Event' in df.columns:
                emg_raw = df.drop(columns=['Event']).values
            else:
                emg_raw = df.values

            total_rows = emg_raw.shape[0]
            
            # BATCH_SIZE 크기만큼 잘라서 바이너리 패킹 후 전송
            for start in range(0, total_rows - BATCH_SIZE + 1, BATCH_SIZE):
                t_start = time.perf_counter()
                
                chunk = emg_raw[start:start+BATCH_SIZE] # 구조: (BATCH_SIZE, CHANNELS)
                
                # network.py 수신 규격(<{BATCH_SIZE * CHANNELS}i) 맞춤형 1차원 플래터닝
                flattened_data = chunk.astype(np.int32).flatten().tolist()
                
                # 바이너리 패킹 데이터 스트림 생성
                packet_bytes = struct.pack(f'<{BATCH_SIZE * CHANNELS}i', *flattened_data)
                
                # TCP 소켓 전송
                client_sock.sendall(packet_bytes)
                
                # 400Hz 샘플링 속도 모사를 위한 정밀 타임 슬립 제어
                elapsed = time.perf_counter() - t_start
                sleep_time = max(0, send_interval - elapsed)
                time.sleep(sleep_time)
                
            print(f"파일 전송 완료: {os.path.basename(file_path)}\n")
            time.sleep(0.5) # 파일 간 공백 패러다임 마진

    except ConnectionResetError:
        print("[경고] 실시간 분류기(서버)와의 연결이 중단되었습니다.")
    finally:
        client_sock.close()
        print("=== 시뮬레이터 종료 ===")

if __name__ == "__main__":
    run_esp32_simulator()