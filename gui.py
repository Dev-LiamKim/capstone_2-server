import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import numpy as np
from config import CHANNELS, WINDOW_SIZE

class EMGVisualizer:
    def __init__(self):
        # 메인 윈도우 및 레이아웃 설정
        self.main_widget = QtWidgets.QWidget()
        self.main_widget.setWindowTitle("EMG Background Data Collector")
        self.main_widget.resize(1200, 850)
        
        layout = QtWidgets.QVBoxLayout(self.main_widget)
        
        # [추가] 실시간 상단 정보 표시 바 (샘플링 레이트 표시용)
        self.info_layout = QtWidgets.QHBoxLayout()
        self.lbl_fps = QtWidgets.QLabel("실시간 Sampling Rate: 계산 중... Hz")
        self.lbl_fps.setStyleSheet("font-size: 14px; font-weight: bold; color: #00FF00; background-color: #1E1E1E; padding: 5px; border-radius: 3px;")
        self.info_layout.addWidget(self.lbl_fps)
        self.info_layout.addStretch()
        layout.addLayout(self.info_layout)

        # 기존 GraphicsLayoutWidget 생성 및 배치
        self.win = pg.GraphicsLayoutWidget()
        layout.addWidget(self.win)
        
        self.curves = []
        for i in range(CHANNELS):
            p = self.win.addPlot(row=i, col=0)
            p.setYRange(-4000000, 4000000)
            p.getViewBox().setMouseEnabled(x=False, y=False)
            c = p.plot(pen=pg.mkPen(color=pg.intColor(i), width=1))
            self.curves.append(c)

        self.bar_plot = self.win.addPlot(row=0, col=1, rowspan=4)
        self.bar_plot.setYRange(0, 1000000)
        self.bar_item = pg.BarGraphItem(x=range(CHANNELS), height=[0]*CHANNELS, width=0.6, brush='y')
        self.bar_plot.addItem(self.bar_item)

        self.fft_plot = self.win.addPlot(row=4, col=1, rowspan=4)
        self.fft_plot.setYRange(0, 1e8)
        self.fft_curves = []
        for i in range(CHANNELS):
            c = self.fft_plot.plot(pen=pg.mkPen(color=pg.intColor(i), width=1))
            self.fft_curves.append(c)

    def show(self):
        self.main_widget.show()

    def update_plots(self, data, processed):
        for i in range(CHANNELS):
            self.curves[i].setData(data[i])
            
        rms_values = np.sqrt(np.mean(np.square(processed[:, -100:]), axis=1))
        self.bar_item.setOpts(height=rms_values)
        
        for i in range(CHANNELS):
            ch_data = processed[i, -256:]
            fft_vals = np.abs(np.fft.rfft(ch_data - np.mean(ch_data)))
            self.fft_curves[i].setData(fft_vals)