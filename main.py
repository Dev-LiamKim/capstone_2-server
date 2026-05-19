# main.py
import sys
import time
import csv
import numpy as np
import os
from pyqtgraph.Qt import QtWidgets, QtCore
from scipy import signal
from config import SERVER_PORT, CHANNELS, WINDOW_SIZE, BATCH_SIZE
from network import EMGReceiver
from gui import EMGVisualizer
from typing_practice import TypingWindow

class EMGApp:
    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)
        self.receiver = EMGReceiver(SERVER_PORT)
        self.visualizer = EMGVisualizer()
        self.typing_win = TypingWindow(self)
        
        self.data_buffer = np.zeros((CHANNELS, WINDOW_SIZE))
        self.is_recording = False
        self.pending_event = 0
        self.csv_file = None
        self.csv_writer = None

        # [수정] 카운터 기반 정밀 동기화 제어 변수군 정의
        self.last_packet_counter = 0
        self.marker_history = [] 

        self.hz_history = []
        self.last_hz_update = time.time()
        self.sample_counter = 0

        fs, f0, Q = 400.0, 60.0, 30.0
        self.b, self.a = signal.iirnotch(f0, Q, fs=fs)

        self.visualizer.show()
        self.typing_win.show()
        self.receiver.wait_for_connection()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.process)
        self.timer.start(10)

    def start_full_recording(self, filename):
        self.is_recording = True
        self.csv_file = open(filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([f"CH{i}" for i in range(CHANNELS)] + ["Event"])
        print(f"[START] 자동 기록 시작: {filename}")

    def stop_full_recording(self):
        self.is_recording = False
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
        print("[STOP] 자동 기록 종료 및 데이터셋 파일 저장 완료")

    def process(self):
        # [수정] network.py측에서 데이터와 카운터를 튜플 형태로 리턴하도록 연동 가정
        result = self.receiver.receive_batch() 
        
        if result is not None:
            batch, current_packet_counter = result
            
            # 1. 샘플 유실 유무 실시간 모니터링 검증
            if self.last_packet_counter != 0:
                lost_samples = (current_packet_counter - self.last_packet_counter) - BATCH_SIZE
                if lost_samples > 0:
                    print(f"[WARNING] 하드웨어 패킷 유실 감지됨: {lost_samples} Samples 드롭")

            self.sample_counter += BATCH_SIZE
            current_time = time.time()
            elapsed_interval = current_time - self.last_hz_update
            
            if elapsed_interval >= 0.5:
                self.hz_history.append((self.sample_counter, elapsed_interval, current_time))
                self.sample_counter = 0
                self.last_hz_update = current_time
                self.hz_history = [record for record in self.hz_history if current_time - record[2] <= 10.0]
                
                if self.hz_history:
                    total_samples = sum(record[0] for record in self.hz_history)
                    total_time = sum(record[1] for record in self.hz_history)
                    if total_time > 0:
                        moving_avg_hz = total_samples / total_time
                        self.visualizer.lbl_fps.setText(f"평균 Sampling Rate (최근 10초): {moving_avg_hz:.2f} Hz")

            self.data_buffer = np.roll(self.data_buffer, -BATCH_SIZE, axis=1)
            self.data_buffer[:, -BATCH_SIZE:] = batch
            processed = np.array([signal.lfilter(self.b, self.a, ch - np.mean(ch)) for ch in self.data_buffer])
            
            # [수정] 키보드 입력 발생 시 현재 하드웨어 카운터 값을 기준으로 매핑할 정확한 내부 인덱스를 산출
            if self.pending_event != 0:
                # 32개 윈도우 스케일 내부의 물리적 상대 오프셋 정합 연산 수행
                start_counter = current_packet_counter - BATCH_SIZE
                target_offset = self.pending_event_counter - start_counter # 상대 샘플 간격 역산
                
                # 가용 범주 검증 필터링
                if 0 <= target_offset < BATCH_SIZE:
                    self.marker_history.append((target_offset, self.pending_event))
                else:
                    # 네트워크 레이턴시로 인해 오차가 발생했을 경우 근사치 강제 정렬 보정
                    self.marker_history.append((0, self.pending_event))
                self.pending_event = 0

            if self.is_recording:
                events = np.zeros((BATCH_SIZE, 1), dtype=int)
                
                # 대기열에 저장된 정밀 인덱스를 매핑 구역에 주입하여 지터 원천 배제
                while self.marker_history:
                    offset, event_id = self.marker_history.pop(0)
                    events[offset] = event_id
                    
                self.csv_writer.writerows(np.hstack((batch.T, events)))
                
            self.visualizer.update_plots(self.data_buffer, processed)
            self.last_packet_counter = current_packet_counter # 최종 카운터 백업 업데이트