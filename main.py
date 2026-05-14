# main.py (수정본)
import sys
import time
import csv
import numpy as np
from pyqtgraph.Qt import QtWidgets, QtCore
from config import SERVER_PORT, CHANNELS, WINDOW_SIZE, BATCH_SIZE
from network import EMGReceiver
from gui import EMGVisualizer

class EMGApp:
    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)
        self.receiver = EMGReceiver(SERVER_PORT)
        self.visualizer = EMGVisualizer()
        
        # 데이터 관리 변수
        self.data_buffer = np.zeros((CHANNELS, WINDOW_SIZE))
        self.is_recording = False
        self.recording_data = [] # 기록용 리스트
        
        # 성능 측정 변수
        self.sample_count = 0
        self.last_time = time.time()

        # GUI 키 이벤트 연결
        self.visualizer.win.keyPressEvent = self.handle_key_press
        
        self.visualizer.show()
        self.receiver.wait_for_connection()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.process)
        self.timer.start(10)

    def handle_key_press(self, event):
        # Space 바 클릭 시 기록 토글
        if event.key() == QtCore.Qt.Key_Space:
            if not self.is_recording:
                self.start_recording()
            else:
                self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        self.recording_data = []
        print("[RECORD] 기록 시작... (중단: Space)")

    def stop_recording(self):
        self.is_recording = False
        filename = f"emg_data_{int(time.time())}.csv"
        self.save_to_csv(filename)
        print(f"[RECORD] 기록 종료 및 저장 완료: {filename}")

    def save_to_csv(self, filename):
        if not self.recording_data:
            return
        
        # 수집된 데이터를 (Total_Samples, Channels) 형태로 변환
        full_data = np.vstack(self.recording_data)
        
        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            # 헤더 작성
            writer.writerow([f"CH{i}" for i in range(CHANNELS)])
            # 데이터 작성
            writer.writerows(full_data)

    def process(self):
        batch = self.receiver.receive_batch()
        if batch is not None:
            # 시각화 버퍼 업데이트 (8, BATCH_SIZE) -> (BATCH_SIZE, 8) 전치 후 사용
            batch_t = batch.T
            self.data_buffer = np.roll(self.data_buffer, -BATCH_SIZE, axis=1)
            self.data_buffer[:, -BATCH_SIZE:] = batch
            
            # 기록 중인 경우 데이터 저장
            if self.is_recording:
                self.recording_data.append(batch_t)
            
            # Hz 측정 및 UI 업데이트 로직 (생략)
            self.sample_count += BATCH_SIZE
            now = time.time()
            if now - self.last_time >= 1.0:
                hz = self.sample_count / (now - self.last_time)
                self.visualizer.win.setWindowTitle(f"EMG Monitor - {hz:.2f} Hz {'[REC]' if self.is_recording else ''}")
                self.sample_count = 0
                self.last_time = now

            self.visualizer.update_charts(self.data_buffer)

    def run(self):
        sys.exit(self.app.exec_())

if __name__ == "__main__":
    EMGApp().run()