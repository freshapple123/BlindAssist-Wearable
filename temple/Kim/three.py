from PyQt5.QtWidgets import QLabel, QVBoxLayout, QApplication, QWidget
from picamera2 import Picamera2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread
import RPi.GPIO as gp
import time
import os
import cv2
import numpy as np

width = 320
height = 240 

adapter_info = {  
    "A": {   
        "i2c_cmd": "i2cset -y 10 0x70 0x00 0x04",
        "gpio_sta": [0, 0, 1],
    }, 
    "B": {
        "i2c_cmd": "i2cset -y 10 0x70 0x00 0x05",
        "gpio_sta": [1, 0, 1],
    }, 
    "C": {
        "i2c_cmd": "i2cset -y 10 0x70 0x00 0x06",
        "gpio_sta": [0, 1, 0],
    }
}

class WorkThread(QThread):
    def __init__(self):
        super().__init__()
        self.setup_gpio()
        
    def setup_gpio(self):
        gp.setwarnings(False)
        gp.setmode(gp.BOARD)
        gp.setup(7, gp.OUT)
        gp.setup(11, gp.OUT)
        gp.setup(12, gp.OUT)

    def select_channel(self, index):
        if index not in adapter_info:
            return
        gpio_sta = adapter_info[index]["gpio_sta"]
        for pin, state in zip((7, 11, 12), gpio_sta):
            gp.output(pin, state)

    def init_i2c(self, index):
        os.system(adapter_info[index]["i2c_cmd"])

    def run(self):
        global picam2
        flag = False

        # 카메라 초기화
        for item in ("A", "B", "C"):
            try:
                self.select_channel(item)
                self.init_i2c(item)
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
                picam2.capture_array()
                time.sleep(0.1)
            except Exception as e:
                print(f"초기화 오류 {item}: {e}")

        while True:
            try:
                frames = []
                # 각 카메라에서 이미지 캡처
                for item in ("A", "B", "C"):
                    self.select_channel(item)
                    time.sleep(0.02)
                    frame = picam2.capture_array()
                    frames.append(frame)
                
                if len(frames) == 3:
                    # 세 이미지를 가로로 연결
                    combined = np.hstack(frames)
                    # Qt 이미지로 변환
                    h, w, _ = combined.shape
                    qimg = QImage(combined.data, w, h, combined.strides[0], QImage.Format_RGB888)
                    combined_label.setPixmap(QPixmap.fromImage(qimg))
                    
            except Exception as e:
                print(f"캡처 오류: {e}")

app = QApplication([])
window = QWidget()
layout = QVBoxLayout()
combined_label = QLabel()
combined_label.setFixedSize(width*3, height)
layout.addWidget(combined_label)
window.setLayout(layout)
window.setWindowTitle("Three Camera View")

work = WorkThread()

if __name__ == "__main__":
    work.start()
    window.show()
    app.exec()
    work.quit()
    picam2.close()
    gp.cleanup()
