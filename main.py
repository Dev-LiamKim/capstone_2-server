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
        
        # 메모리 버퍼 및 타겟 경로 보존 변수
        self.recorded_chunks = []
        self.target_filename = ""

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
        self.target_filename = filename
        self.recorded_chunks = []
        print(f"[START] 기록 세션 개시 (메모리 버퍼링 활성화): {filename}")

    def stop_full_recording(self, success=False):
        self.is_recording = False
        if success and self.recorded_chunks:
            try:
                with open(self.target_filename, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([f"CH{i}" for i in range(CHANNELS)] + ["Event"])
                    writer.writerows(self.recorded_chunks)
                print(f"[SUCCESS] 50사이클 완주 성공. 데이터셋 파일 생성 완료: {self.target_filename}")
            except Exception as e:
                print(f"[ERROR] 파일 저장 실패: {e}")
        else:
            print("[CANCEL] 측정 미완료 또는 오타/중도 종료 발생. 데이터셋 폐기 완료.")
        
        self.recorded_chunks = []
        self.target_filename = ""

    def process(self):
        batch = self.receiver.receive_batch()
        if batch is not None:
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
            
            # 디스크 쓰기 연산을 전면 배제하고 파이썬 리스트 메모리 버퍼에 임시 적재
            if self.is_recording:
                events = np.zeros((BATCH_SIZE, 1), dtype=int)
                if self.pending_event != 0:
                    events[0] = self.pending_event
                    self.pending_event = 0
                
                combined = np.hstack((batch.T, events))
                self.recorded_chunks.extend(combined.tolist())
                
            self.visualizer.update_plots(self.data_buffer, processed)

if __name__ == "__main__":
    app = EMGApp()
    sys.exit(app.app.exec_())