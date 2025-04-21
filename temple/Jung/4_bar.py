from PyQt5.QtWidgets import QLabel, QHBoxLayout, QVBoxLayout, QApplication, QWidget, QSlider
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, Qt
import RPi.GPIO as gp
from picamera2 import Picamera2
import time
import os
import numpy as np

width = 320
height = 240

adapter_info = {
    "A": {"i2c_cmd": "i2cset -y 10 0x70 0x00 0x04", "gpio_sta": [0, 0, 1]},
    "B": {"i2c_cmd": "i2cset -y 10 0x70 0x00 0x05", "gpio_sta": [1, 0, 1]},
    "C": {"i2c_cmd": "i2cset -y 10 0x70 0x00 0x06", "gpio_sta": [0, 1, 0]},
}

class WorkThread(QThread):
    def __init__(self, sliders):
        super(WorkThread, self).__init__()
        self.sliders = sliders
        gp.setwarnings(False)
        gp.setmode(gp.BOARD)
        gp.setup(7, gp.OUT)
        gp.setup(11, gp.OUT)
        gp.setup(12, gp.OUT)

    def select_channel(self, index):
        channel_info = adapter_info.get(index)
        if channel_info is None:
            print("Can't get this info")
            return
        gpio_sta = channel_info["gpio_sta"]
        gp.output(7, gpio_sta[0])
        gp.output(11, gpio_sta[1])
        gp.output(12, gpio_sta[2])

    def init_i2c(self, index):
        os.system(adapter_info[index]["i2c_cmd"])

    def run(self):
        global picam2
        flag = False

        for item in {"A", "B", "C"}:
            try:
                self.select_channel(item)
                self.init_i2c(item)
                time.sleep(0.5)
                if flag:
                    picam2.close()
                else:
                    flag = True
                print("init1 " + item)
                picam2 = Picamera2()
                picam2.configure(
                    picam2.create_still_configuration(
                        main={"size": (width, height), "format": "BGR888"},
                        buffer_count=2,
                    )
                )
                picam2.start()
                time.sleep(2)
                picam2.capture_array(wait=False)
                time.sleep(0.1)
            except Exception as e:
                print("except: " + str(e))

        while True:
            for item in {"A", "B", "C"}:
                self.select_channel(item)
                time.sleep(0.02)
                try:
                    buf = picam2.capture_array()

                    # crop logic
                    crop_val = self.sliders[item].value()
                    crop_px = int(width * (crop_val / 100.0))

                    if item == "A":
                        # 오른쪽에서 crop
                        buf = buf[:, : width - crop_px]
                    elif item == "B":
                        # 좌우 대칭 crop
                        buf = buf[:, crop_px // 2 : width - crop_px // 2]
                    elif item == "C":
                        # 왼쪽에서 crop
                        buf = buf[:, crop_px:]

                    cvimg = QImage(buf, buf.shape[1], buf.shape[0], QImage.Format_RGB888)
                    pixmap = QPixmap(cvimg)

                    if item == "A":
                        image_label.setPixmap(pixmap)
                    elif item == "B":
                        image_label2.setPixmap(pixmap)
                    elif item == "C":
                        image_label3.setPixmap(pixmap)
                except Exception as e:
                    print("capture_buffer: " + str(e))

app = QApplication([])
window = QWidget()

layout_main = QVBoxLayout()
layout_h = QHBoxLayout()
layout_s = QHBoxLayout()

image_label = QLabel()
image_label2 = QLabel()
image_label3 = QLabel()

# 슬라이더 3개 (0~100%)
slider_left = QSlider(Qt.Horizontal)
slider_center = QSlider(Qt.Horizontal)
slider_right = QSlider(Qt.Horizontal)
for s in (slider_left, slider_center, slider_right):
    s.setMinimum(0)
    s.setMaximum(100)
    s.setValue(0)

# 딕셔너리로 슬라이더 전달
sliders = {"A": slider_left, "B": slider_center, "C": slider_right}
work = WorkThread(sliders)

if __name__ == "__main__":
    image_label.setFixedSize(320, 240)
    image_label2.setFixedSize(320, 240)
    image_label3.setFixedSize(320, 240)

    window.setWindowTitle("Qt Picamera2 Arducam Multi Camera with Crop Slider")

    layout_h.addWidget(image_label)
    layout_h.addWidget(image_label2)
    layout_h.addWidget(image_label3)

    layout_s.addWidget(slider_left)
    layout_s.addWidget(slider_center)
    layout_s.addWidget(slider_right)

    layout_main.addLayout(layout_h)
    layout_main.addLayout(layout_s)

    window.setLayout(layout_main)
    window.resize(960, 300)

    work.start()
    window.show()
    app.exec()
    work.quit()
    picam2.close()
