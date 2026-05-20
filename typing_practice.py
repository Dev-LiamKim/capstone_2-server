import random 
import os
import time
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui 
from config import RIGHT_HAND_KEYS

class TypingWindow(QtWidgets.QWidget):
    def __init__(self, app_reference=None):
        super().__init__()
        self.app = app_reference  
        self.setWindowTitle("EMG Key Position Practice")
        self.resize(550, 250)
        self.setWindowFlags(QtCore.Qt.WindowType.WindowStaysOnTopHint)

        self.char_pool = [k for k in RIGHT_HAND_KEYS.keys() if len(k) == 1]
        self.pool_size = len(self.char_pool)
        self.practice_queue = []
        
        self.current_cycle = 1
        self.max_cycles = 50
        self.keys_in_cycle = 0
        
        layout = QtWidgets.QVBoxLayout()
        
        self.lbl_instr = QtWidgets.QLabel(f"제시된 글자를 누르세요 (진행도: {self.current_cycle}/{self.max_cycles} 사이클 | {self.keys_in_cycle}/{self.pool_size})")
        self.lbl_instr.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        target_layout = QtWidgets.QHBoxLayout()
        target_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        target_layout.setSpacing(25)
        
        self.lbl_current = QtWidgets.QLabel("")
        self.lbl_current.setFont(QtGui.QFont("Arial", 38, QtGui.QFont.Weight.Bold))
        self.lbl_current.setStyleSheet("color: #0055ff;")
        
        self.lbl_next = QtWidgets.QLabel("")
        self.lbl_next.setFont(QtGui.QFont("Arial", 26, QtGui.QFont.Weight.Normal))
        self.lbl_next.setStyleSheet("color: #888888;")
        
        self.lbl_next_next = QtWidgets.QLabel("")
        self.lbl_next_next.setFont(QtGui.QFont("Arial", 18, QtGui.QFont.Weight.Normal))
        self.lbl_next_next.setStyleSheet("color: #d3d3d3;")
        
        target_layout.addWidget(self.lbl_current)
        target_layout.addWidget(self.lbl_next)
        target_layout.addWidget(self.lbl_next_next)
        
        self.entry = QtWidgets.QLineEdit()
        self.entry.setFont(QtGui.QFont("Arial", 18))
        self.entry.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.entry.textChanged.connect(self.check_result)

        self.lbl_status = QtWidgets.QLabel("대기 중...")
        self.lbl_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.lbl_instr)
        layout.addLayout(target_layout)
        layout.addWidget(self.entry)
        layout.addWidget(self.lbl_status)
        self.setLayout(layout)

        self.update_display()

    def fill_queue_if_needed(self):
        while len(self.practice_queue) < 3:
            new_batch = list(self.char_pool)
            random.shuffle(new_batch)
            self.practice_queue.extend(new_batch)

    def update_display(self):
        if self.current_cycle > self.max_cycles:
            self.lbl_current.setText("종료")
            self.lbl_next.setText("")
            self.lbl_next_next.setText("")
            return

        self.fill_queue_if_needed()
        
        self.target_text = self.practice_queue.pop(0)
        self.lbl_current.setText(self.target_text)
        
        remaining_total = ((self.max_cycles - self.current_cycle) * self.pool_size) + (self.pool_size - self.keys_in_cycle)
        
        if remaining_total >= 2:
            self.lbl_next.setText(self.practice_queue[0])
        else:
            self.lbl_next.setText("")
            
        if remaining_total >= 3:
            self.lbl_next_next.setText(self.practice_queue[1])
        else:
            self.lbl_next_next.setText("")

    def check_result(self, text):
        if not text:
            return

        if self.current_cycle > self.max_cycles:
            self.entry.blockSignals(True)
            self.entry.clear()
            self.entry.setDisabled(True)
            self.entry.blockSignals(False)
            return

        if self.app and not self.app.is_recording and self.current_cycle == 1 and self.keys_in_cycle == 0:
            os.makedirs("dataset", exist_ok=True)
            current_timestamp = time.strftime("%Y%m%d_%H%M%S")
            generated_filename = f"dataset/recording_{current_timestamp}.csv"
            
            # 파일 생성 없이 기록 플래그 활성화 및 파일명만 전달
            self.app.start_full_recording(generated_filename)

        if text == self.target_text:
            if self.app:
                self.app.pending_event = RIGHT_HAND_KEYS.get(text, 0)

            self.keys_in_cycle += 1
            
            if self.keys_in_cycle >= self.pool_size:
                self.keys_in_cycle = 0
                self.current_cycle += 1
            
            if self.current_cycle > self.max_cycles:
                self.lbl_instr.setText("제시된 글자를 누르세요 (진행도: 완료)")
                self.lbl_status.setText("50 사이클 연습 전체 완료! 데이터셋 저장 완료.")
                self.lbl_status.setStyleSheet("color: blue; font-weight: bold;")
                self.lbl_current.setText("종료")
                self.lbl_next.setText("")
                self.lbl_next_next.setText("")
                self.entry.setDisabled(True)
                
                # 성공 종료 플래그(True) 전달
                if self.app and self.app.is_recording:
                    self.app.stop_full_recording(success=True)
            else:
                self.lbl_instr.setText(f"제시된 글자를 누르세요 (진행도: {self.current_cycle}/{self.max_cycles} 사이클 | {self.keys_in_cycle}/{self.pool_size})")
                self.lbl_status.setText("정확함!")
                self.lbl_status.setStyleSheet("color: green;")
                self.update_display()
            
            self.entry.blockSignals(True)
            self.entry.clear()
            self.entry.blockSignals(False)
        else:
            if self.app:
                self.app.pending_event = 0

            self.lbl_status.setText(f"오타! '{self.target_text}'를 누르세요")
            self.lbl_status.setStyleSheet("color: red;")
            
            self.entry.blockSignals(True)
            self.entry.clear()
            self.entry.blockSignals(False)

    # 창을 강제로 닫을 때 실패 처리 매핑
    def closeEvent(self, event):
        if self.app and self.app.is_recording:
            self.app.stop_full_recording(success=False)
        event.accept()