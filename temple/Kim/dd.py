from PyQt5.QtWidgets import QLabel, QStackedLayout, QApplication, QWidget
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
        self.running = True

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

        # 카메라 초기화
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

        while self.running:
            for item in {"A", "B", "C"}:
                self.select_channel(item)
                time.sleep(0.02)
                try:
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

    def cleanup(self):
        self.running = False

app = QApplication([])
window = QWidget()
layout = QStackedLayout()
layout.setStackingMode(QStackedLayout.StackAll)  # 모든 위젯을 겹쳐서 표시

# 레이블 설정
label_A = QLabel()
label_B = QLabel()
label_C = QLabel()

# 레이블 스타일 및 투명도 설정
label_A.setStyleSheet("QLabel { background-color: transparent; }")
label_B.setStyleSheet("QLabel { background-color: transparent; opacity: 0.7; }")
label_C.setStyleSheet("QLabel { background-color: transparent; opacity: 0.5; }")

for label in (label_A, label_B, label_C):
    label.setFixedSize(width, height)
    layout.addWidget(label)

window.setLayout(layout)
window.setWindowTitle("Overlapped Camera View")
work = WorkThread()

if __name__ == "__main__":
    try:
        work.start()
        window.show()
        app.exec()
    finally:
        work.cleanup()
        work.quit()
        picam2.close()
        gp.cleanup()