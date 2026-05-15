import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import numpy as np
from config import CHANNELS, WINDOW_SIZE

class EMGVisualizer:
    def __init__(self):
        self.win = pg.GraphicsLayoutWidget(title="EMG Background Data Collector")
        self.win.resize(1200, 800)
        
        self.curves = []
        for i in range(CHANNELS):
            p = self.win.addPlot(row=i, col=0)
            p.setYRange(-2500000, 2500000)
            p.getViewBox().setMouseEnabled(x=False, y=False)
            c = p.plot(pen=pg.mkPen(color=pg.intColor(i), width=1))
            self.curves.append(c)

        # RMS 활성도 표시
        self.bar_plot = self.win.addPlot(row=0, col=1, rowspan=4)
        self.bar_item = pg.BarGraphItem(x=range(CHANNELS), height=[0]*CHANNELS, width=0.6, brush='y')
        self.bar_plot.addItem(self.bar_item)

        # FFT 주파수 분석
        self.fft_plot = self.win.addPlot(row=4, col=1, rowspan=4)
        self.fft_curves = []
        for i in range(CHANNELS):
            c = self.fft_plot.plot(pen=pg.mkPen(color=pg.intColor(i), width=1))
            self.fft_curves.append(c)

    def update_charts(self, buffer, processed):
        for i, curve in enumerate(self.curves):
            curve.setData(processed[i])
        
        rms_values = [np.sqrt(np.mean(np.square(processed[i, -100:]))) for i in range(CHANNELS)]
        self.bar_item.setOpts(height=rms_values)

        n, fs = 512, 400
        for i in range(CHANNELS):
            data = processed[i, -n:]
            fft_data = np.abs(np.fft.rfft(data))
            freqs = np.fft.rfftfreq(n, d=1/fs)
            self.fft_curves[i].setData(freqs, fft_data + 1)

    def show(self):
        self.win.show()