import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QIcon


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Robot Calibration'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        button=QPushButton('Calibrate',self)
        button.move(100,70)
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())