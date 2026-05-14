# main.py
import sys
import time
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
        
        self.data_buffer = np.zeros((CHANNELS, WINDOW_SIZE))
        self.sample_count = 0
        self.last_time = time.time()

        self.visualizer.show()
        self.receiver.wait_for_connection()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.process)
        self.timer.start(10)

    def process(self):
        batch = self.receiver.receive_batch()
        if batch is not None:
            # 버퍼 업데이트
            self.data_buffer = np.roll(self.data_buffer, -BATCH_SIZE, axis=1)
            self.data_buffer[:, -BATCH_SIZE:] = batch
            
            # Hz 측정
            self.sample_count += BATCH_SIZE
            now = time.time()
            if now - self.last_time >= 1.0:
                print(f"[STATUS] 수신 속도: {self.sample_count / (now - self.last_time):.2f} Hz")
                self.sample_count = 0
                self.last_time = now

            # UI 갱신
            self.visualizer.update_charts(self.data_buffer)

    def run(self):
        sys.exit(self.app.exec_())

if __name__ == "__main__":
    EMGApp().run()