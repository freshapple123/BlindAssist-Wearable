from PyQt5.QtWidgets import QLabel, QHBoxLayout, QApplication, QWidget
from picamera2 import Picamera2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread
import RPi.GPIO as gp
import time
import os

width = 320
height = 240 

adapter_info = {  
    "A": {   
        "i2c_cmd": "i2cset -y 1 0x70 0x00 0x04",
        "gpio_sta": [0, 0, 1],
    }, 
    "B": {
        "i2c_cmd": "i2cset -y 1 0x70 0x00 0x05",
        "gpio_sta": [1, 0, 1],
    }, 
    "C": {
        "i2c_cmd": "i2cset -y 1 0x70 0x00 0x06",
        "gpio_sta": [0, 1, 0],
    }
}

class WorkThread(QThread):
    def __init__(self):
        super().__init__()
        gp.setwarnings(False)
        gp.setmode(gp.BOARD)
        gp.setup(7, gp.OUT)
        gp.setup(11, gp.OUT)
        gp.setup(12, gp.OUT)

    def select_channel(self, channel):
        gpio_sta = adapter_info[channel]["gpio_sta"]
        gp.output(7, gpio_sta[0])
        gp.output(11, gpio_sta[1])
        gp.output(12, gpio_sta[2])
        os.system(adapter_info[channel]["i2c_cmd"])

    def init_i2c(self, index):
        channel_info = adapter_info.get(index)
        os.system(channel_info["i2c_cmd"])

    def run(self):
        global picam2
        flag = False

        # previewOpencv.py 방식의 카메라 초기화
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
                picam2.configure(picam2.create_still_configuration(
                    main={"size": (width, height), "format": "BGR888"},
                    buffer_count=2
                ))
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
                    buf = picam2.capture_array()
                    cvimg = QImage(buf, width, height, QImage.Format_RGB888)
                    pixmap = QPixmap(cvimg)
                    if item == 'A':
                        label_A.setPixmap(pixmap)
                    elif item == 'B':
                        label_B.setPixmap(pixmap)
                    elif item == 'C':
                        label_C.setPixmap(pixmap)
                except Exception as e:
                    print(f"capture_buffer: {e}")

app = QApplication([])
window = QWidget()
layout = QHBoxLayout()
layout.setSpacing(0)  # 레이블 간 간격 제거
layout.setContentsMargins(0, 0, 0, 0)  # 마진 제거

# 세 개의 레이블을 가로로 배치
label_A = QLabel()
label_B = QLabel()
label_C = QLabel()
for label in (label_A, label_B, label_C):
    label.setFixedSize(width, height)
    label.setStyleSheet("QLabel { margin: 0px; padding: 0px; }")  # 레이블 자체의 마진/패딩 제거
    layout.addWidget(label)

window.setLayout(layout)
window.setWindowTitle("Three Camera View")
work = WorkThread()

if __name__ == "__main__":
    try:
        work.start()
        window.show()
        app.exec()
    finally:
        work.quit()
        picam2.close()
        gp.cleanup()
