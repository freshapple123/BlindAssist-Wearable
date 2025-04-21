from PyQt5.QtWidgets import QLabel, QVBoxLayout, QApplication, QWidget, QSlider
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtCore import QThread, Qt
from picamera2 import Picamera2
import RPi.GPIO as gp
import time
import os
import numpy as np

width = 320
height = 240
images = {"A": None, "B": None, "C": None}
offset = 0

adapter_info = {
    "A": {"i2c_cmd": "i2cset -y 10 0x70 0x00 0x04", "gpio_sta": [0, 0, 1]},
    "B": {"i2c_cmd": "i2cset -y 10 0x70 0x00 0x05", "gpio_sta": [1, 0, 1]},
    "C": {"i2c_cmd": "i2cset -y 10 0x70 0x00 0x06", "gpio_sta": [0, 1, 0]}
}

class WorkThread(QThread):
    def __init__(self):
        super(WorkThread, self).__init__()
        gp.setwarnings(False)
        gp.setmode(gp.BOARD)
        for pin in [7, 11, 12]:
            gp.setup(pin, gp.OUT)

    def select_channel(self, index):
        channel_info = adapter_info.get(index)
        if channel_info:
            for pin, state in zip([7, 11, 12], channel_info["gpio_sta"]):
                gp.output(pin, state)

    def init_i2c(self, index):
        os.system(adapter_info[index]["i2c_cmd"])

    def run(self):
        global picam2
        flag = False
        for cam in {"A", "B", "C"}:
            try:
                self.select_channel(cam)
                self.init_i2c(cam)
                time.sleep(0.5)
                if flag:
                    picam2.close()
                else:
                    flag = True
                picam2 = Picamera2()
                picam2.configure(
                    picam2.create_still_configuration(
                        main={"size": (width, height), "format": "BGR888"},
                        buffer_count=2
                    )
                )
                picam2.start()
                time.sleep(2)
                picam2.capture_array(wait=False)
                time.sleep(0.1)
            except Exception as e:
                print("init error", cam, e)

        while True:
            for cam in {"A", "B", "C"}:
                try:
                    self.select_channel(cam)
                    time.sleep(0.02)
                    buf = picam2.capture_array()
                    images[cam] = buf.copy()
                except Exception as e:
                    print(f"capture error {cam}: {e}")
            self.msleep(50)

class PanoramaDisplay(QLabel):
    def paintEvent(self, event):
        if images["C"] is None:
            return

        qp = QPainter(self)

        # 중앙 고정 이미지
        center_img = QImage(images["C"].data, width, height, QImage.Format_RGB888)
        qp.drawImage(0, 0, center_img)

        # 왼쪽 이미지 offset만큼 왼쪽에서 오른쪽으로 밀기
        if images["A"] is not None:
            left_img = QImage(images["A"].data, width, height, QImage.Format_RGB888)
            qp.drawImage(-offset, 0, left_img)

        # 오른쪽 이미지 offset만큼 오른쪽에서 왼쪽으로 밀기
        if images["B"] is not None:
            right_img = QImage(images["B"].data, width, height, QImage.Format_RGB888)
            qp.drawImage(offset, 0, right_img)

        qp.end()

app = QApplication([])
window = QWidget()

layout = QVBoxLayout()
slider = QSlider(Qt.Horizontal)
slider.setRange(0, 320)
slider.setValue(0)

display = PanoramaDisplay()
display.setFixedSize(320, 240)

def on_slider_changed(val):
    global offset
    offset = val
    display.update()

slider.valueChanged.connect(on_slider_changed)

layout.addWidget(display)
layout.addWidget(slider)

window.setLayout(layout)
window.setWindowTitle("Panorama Offset Viewer")
window.resize(340, 300)

work = WorkThread()
work.start()

window.show()
app.exec()
work.quit()
picam2.close()
