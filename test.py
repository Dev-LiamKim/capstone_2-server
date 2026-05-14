import socket
import struct
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import time  # 주파수 측정을 위해 추가

# [설정] 클라이언트(ESP32)와 일치 필수
CHANNELS = 8
BATCH_SIZE = 32
WINDOW_SIZE = 2000  # X축 고정 길이 (샘플 개수)
PACKET_SIZE = BATCH_SIZE * CHANNELS * 4 # 1024 bytes

class EMGMonitor:
    def __init__(self):
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title="Integrated High-Speed EMG Monitor")
        self.win.resize(1000, 800)
        self.win.show()

        self.curves = []
        self.plots = []
        # 데이터 버퍼 초기화 (8채널 x 2000샘플)
        self.data_buffer = np.zeros((CHANNELS, WINDOW_SIZE))

        # 주파수 측정용 변수 초기화
        self.last_time = time.time()
        self.sample_count = 0

        for i in range(CHANNELS):
            p = self.win.addPlot(row=i, col=0)
            p.setXRange(0, WINDOW_SIZE, padding=0)
            p.setYRange(-2500000, 2500000)

            # 1. 축 별 마우스 제어 및 확대/축소 완전 차단
            p.getViewBox().setMouseEnabled(x=False, y=False) 
            p.setMenuEnabled(False)  # 우클릭 메뉴 제거
            p.hideButtons()  # 자동 범위 버튼 제거
            
            # 2. 모든 마우스 버튼 입력 무시 (AttributeError 해결 로직)
            p.getViewBox().setAcceptedMouseButtons(QtCore.Qt.NoButton)
    
            c = p.plot(pen=pg.mkPen(color=pg.intColor(i), width=1), connect="all")
            self.curves.append(c)

        # 소켓 설정
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) # 지연 방지
        self.server_socket.bind(('0.0.0.0', 5000))
        self.server_socket.listen(1)
        
        print("ESP32 접속 대기 중 (Port: 5000)...")
        self.conn, addr = self.server_socket.accept()
        self.conn.setblocking(False)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(10) # 10ms 주기로 화면 갱신

    def update_plot(self):
        try:
            while True:
                # 패킷 데이터 수신
                data = b''
                while len(data) < PACKET_SIZE:
                    packet = self.conn.recv(PACKET_SIZE - len(data))
                    if not packet: return
                    data += packet
                
                # 수신된 샘플 수 누적 (배치 사이즈만큼 증가)
                self.sample_count += BATCH_SIZE

                # 1초마다 주파수(Hz) 계산 및 출력
                current_time = time.time()
                elapsed = current_time - self.last_time
                if elapsed >= 1.0:
                    hz = self.sample_count / elapsed
                    print(f"[STATUS] 실시간 수신 속도: {hz:.2f} Hz")
                    self.last_time = current_time
                    self.sample_count = 0

                # 데이터 해석 (Little Endian)
                unpacked = struct.unpack(f'<{BATCH_SIZE * CHANNELS}i', data)
                reshaped = np.array(unpacked).reshape(BATCH_SIZE, CHANNELS).T
                
                # 버퍼 업데이트 (Roll 방식)
                self.data_buffer = np.roll(self.data_buffer, -BATCH_SIZE, axis=1)
                self.data_buffer[:, -BATCH_SIZE:] = reshaped
                
                # 화면 갱신
                for i in range(CHANNELS):
                    # DC Offset 제거 (0점 조정)
                    clean_data = self.data_buffer[i] - np.mean(self.data_buffer[i])
                    self.curves[i].setData(clean_data)
                    
        except BlockingIOError:
            pass
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    monitor = EMGMonitor()
    monitor.app.exec_()