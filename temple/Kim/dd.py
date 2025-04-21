from PyQt5.QtWidgets import QLabel, QStackedLayout, QApplication, QWidget, QGraphicsOpacityEffect
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

        # 카메라 초기화 - three.py와 동일한 방식
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
                time.sleep(0.1)  # 0.02에서 0.1로 증가
                try:
                    # three.py와 동일하게 두 번 캡처
                    buf = picam2.capture_array()
                    buf = picam2.capture_array()
                    cvimg = QImage(buf, width, height, QImage.Format_RGB888)
                    pixmap = QPixmap(cvimg)
                    
                    # 이전 이미지 해제
                    if item == 'A':
                        if label_A.pixmap():
                            label_A.pixmap().detach()
                        label_A.setPixmap(pixmap)
                    elif item == 'B':
                        if label_B.pixmap():
                            label_B.pixmap().detach()
                        label_B.setPixmap(pixmap)
                    elif item == 'C':
                        if label_C.pixmap():
                            label_C.pixmap().detach()
                        label_C.setPixmap(pixmap)
                except Exception as e:
                    print(f"capture_buffer: {e}")

    def cleanup(self):
        self.running = False

# GUI 초기화 부분 수정
app = QApplication([])
window = QWidget()
layout = QStackedLayout()
layout.setStackingMode(QStackedLayout.StackAll)

# 레이블 설정
label_A = QLabel()
label_B = QLabel()
label_C = QLabel()

# 투명도 효과 설정
opacity_effect_B = QGraphicsOpacityEffect()
opacity_effect_B.setOpacity(0.6)
label_B.setGraphicsEffect(opacity_effect_B)

opacity_effect_C = QGraphicsOpacityEffect()
opacity_effect_C.setOpacity(0.3)
label_C.setGraphicsEffect(opacity_effect_C)

# 배경 투명 설정
for label in (label_A, label_B, label_C):
    label.setFixedSize(width, height)
    label.setStyleSheet("QLabel { background-color: transparent; }")
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