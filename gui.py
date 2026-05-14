# gui.py
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from config import CHANNELS, WINDOW_SIZE

class EMGVisualizer:
    def __init__(self):
        self.win = pg.GraphicsLayoutWidget(title="Modular EMG Monitor")
        self.win.resize(1000, 800)
        self.curves = []

        for i in range(CHANNELS):
            p = self.win.addPlot(row=i, col=0)
            p.setXRange(0, WINDOW_SIZE, padding=0)
            p.setYRange(-2500000, 2500000)
            
            # 인터랙션 완전 차단
            p.getViewBox().setMouseEnabled(x=False, y=False)
            p.setMenuEnabled(False)
            p.hideButtons()
            p.getViewBox().setAcceptedMouseButtons(QtCore.Qt.NoButton)
            
            c = p.plot(pen=pg.mkPen(color=pg.intColor(i), width=1), connect="all")
            self.curves.append(c)

    def update_charts(self, buffer):
        for i, curve in enumerate(self.curves):
            # DC Offset 제거 후 갱신
            curve.setData(buffer[i] - buffer[i].mean())

    def show(self):
        self.win.show()