# main.py
import sys
import time
import csv
import numpy as np
import os
from pyqtgraph.Qt import QtWidgets, QtCore
from scipy import signal
from config import (
    SERVER_PORT, CHANNELS, WINDOW_SIZE, BATCH_SIZE,
    NOTCH_F0, NOTCH_Q, TIMER_INTERVAL_MS, HZ_CHECK_INTERVAL,
    HZ_HISTORY_DURATION, HZ_THRESHOLD_LOW
)
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
        
        self.recorded_chunks = []
        self.target_filename = ""

        self.hz_history = []          
        self.last_hz_update = time.time()
        self.sample_counter = 0

        fs = 400.0  
        self.b, self.a = signal.iirnotch(NOTCH_F0, NOTCH_Q, fs=fs)

        self.visualizer.show()
        self.typing_win.show() 
        self.receiver.wait_for_connection()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.process)
        self.timer.start(TIMER_INTERVAL_MS)

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
            
            if elapsed_interval >= HZ_CHECK_INTERVAL:
                self.hz_history.append((self.sample_counter, elapsed_interval, current_time))
                self.sample_counter = 0
                self.last_hz_update = current_time
                self.hz_history = [record for record in self.hz_history if current_time - record[2] <= HZ_HISTORY_DURATION]
                
                if self.hz_history:
                    total_samples = sum(record[0] for record in self.hz_history)
                    total_time = sum(record[1] for record in self.hz_history)
                    if total_time > 0:
                        moving_avg_hz = total_samples / total_time
                        self.visualizer.lbl_fps.setText(f"평균 Sampling Rate (최근 10초): {moving_avg_hz:.2f} Hz")
                        
                        if moving_avg_hz < HZ_THRESHOLD_LOW:
                            self.typing_win.entry.setDisabled(True)
                            self.typing_win.lbl_status.setText("네트워크 지연 발생: 수신 속도 저하로 입력을 제한합니다.")
                            self.typing_win.lbl_status.setStyleSheet("color: orange; font-weight: bold;")
                        else:
                            if self.typing_win.current_cycle <= self.typing_win.max_cycles:
                                self.typing_win.entry.setDisabled(False)
                                self.typing_win.entry.setFocus()  
                                # 속도 복구 시 하단 레이블 메시지 및 가독성 색상 상태 업데이트 추가
                                self.typing_win.lbl_status.setText("정상 속도 복구: 입력 가능 상태입니다.")
                                self.typing_win.lbl_status.setStyleSheet("color: green; font-weight: normal;")
                                
            self.data_buffer = np.roll(self.data_buffer, -BATCH_SIZE, axis=1)
            self.data_buffer[:, -BATCH_SIZE:] = batch
            
            processed = np.array([signal.lfilter(self.b, self.a, ch - np.mean(ch)) for ch in self.data_buffer])
            
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