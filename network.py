import socket
import struct
import numpy as np
from config import PACKET_SIZE, BATCH_SIZE, CHANNELS

class EMGReceiver:
    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.bind(('0.0.0.0', port))
        self.sock.listen(1)
        self.conn = None
        self.buffer = b''  # 데이터 보존용 누적 버퍼 추가

    def wait_for_connection(self):
        print("ESP32 접속 대기 중...")
        self.conn, addr = self.sock.accept()
        self.conn.setblocking(False)
        return addr

    def receive_batch(self):
        try:
            # 논블로킹으로 수신 가능한 모든 데이터 누적
            packet = self.conn.recv(4096)
            if not packet: 
                return None
            self.buffer += packet
        except BlockingIOError:
            pass  # 현재 수신 큐가 비어있을 경우 예외 무시 후 버퍼 길이 확인 단계로 이동

        # 버퍼에 하나의 완전한 패킷 이상이 모였을 때만 파싱 처리
        if len(self.buffer) >= PACKET_SIZE:
            data = self.buffer[:PACKET_SIZE]
            self.buffer = self.buffer[PACKET_SIZE:]  # 처리 완료된 데이터는 버퍼에서 제거
            
            unpacked = struct.unpack(f'<{BATCH_SIZE * CHANNELS}i', data)
            return np.array(unpacked).reshape(BATCH_SIZE, CHANNELS).T
            
        return None