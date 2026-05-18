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

        # 노치 필터 계수 (60Hz 제거)
        fs, f0, Q = 400.0, 60.0, 30.0
        self.b, self.a = signal.iirnotch(f0, Q, fs=fs)

        self.visualizer.show()
        self.typing_win.show() 
        self.receiver.wait_for_connection()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.process)
        self.timer.start(10)

    def start_full_recording(self):
        """데이터 기록 시작 및 파일 생성"""
        dir_name = "dataset"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        self.is_recording = True
        filename = os.path.join(dir_name, f"emg_session_{int(time.time())}.csv")
        self.csv_file = open(filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([f"CH{i}" for i in range(CHANNELS)] + ["Event"])
        print(f"[START] 자동 기록 시작: {filename}")

    def stop_full_recording(self):
        """데이터 기록 중단 및 파일 저장"""
        self.is_recording = False
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
        print("[STOP] 자동 기록 종료 및 데이터셋 파일 저장 완료")

    def process(self):
        batch = self.receiver.receive_batch()
        if batch is not None:
            self.data_buffer = np.roll(self.data_buffer, -BATCH_SIZE, axis=1)
            self.data_buffer[:, -BATCH_SIZE:] = batch
            
            # 시각화용 필터링
            processed = np.array([signal.lfilter(self.b, self.a, ch - np.mean(ch)) for ch in self.data_buffer])
            
            if self.is_recording:
                events = np.zeros((BATCH_SIZE, 1), dtype=int)
                if self.pending_event != 0:
                    events[0] = self.pending_event
                    self.pending_event = 0
                self.csv_writer.writerows(np.hstack((batch.T, events)))

            self.visualizer.update_charts(self.data_buffer, processed)

    def run(self):
        try:
            sys.exit(self.app.exec_())
        finally:
            self.stop_full_recording()

if __name__ == "__main__":
    EMGApp().run()