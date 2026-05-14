# network.py
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

    def wait_for_connection(self):
        print("ESP32 접속 대기 중...")
        self.conn, addr = self.sock.accept()
        self.conn.setblocking(False)
        return addr

    def receive_batch(self):
        try:
            data = b''
            while len(data) < PACKET_SIZE:
                packet = self.conn.recv(PACKET_SIZE - len(data))
                if not packet: return None
                data += packet
            
            unpacked = struct.unpack(f'<{BATCH_SIZE * CHANNELS}i', data)
            return np.array(unpacked).reshape(BATCH_SIZE, CHANNELS).T
        except BlockingIOError:
            return None