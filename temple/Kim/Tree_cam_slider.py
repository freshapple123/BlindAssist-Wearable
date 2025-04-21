from PyQt5.QtWidgets import (QLabel, QVBoxLayout, QApplication, QWidget, 
                           QSlider, QHBoxLayout, QGroupBox, QGraphicsOpacityEffect)
from picamera2 import Picamera2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, Qt
import RPi.GPIO as gp
import time
import os

width, height = 320, 240

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

    def run(self):
        global picam2
        flag = False

        # 카메라 초기화
        for item in {"A", "B", "C"}:
            try:
                self.select_channel(item)
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
                time.sleep(0.1)
                try:
                    buf = picam2.capture_array()
                    buf = picam2.capture_array()
                    cvimg = QImage(buf, width, height, QImage.Format_RGB888)
                    pixmap = QPixmap(cvimg)
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

# UI 부분 수정
app = QApplication([])
window = QWidget()
main_layout = QVBoxLayout()

# 이미지 컨테이너 위젯 (더 큰 크기로 설정)
image_container = QWidget()
image_container.setFixedSize(960, 240)  # 최대 너비는 320*3
main_layout.addWidget(image_container)

# 레이블 설정 (스택 순서 중요: 뒤에서 앞으로 C -> B -> A)
label_C = QLabel(image_container)
label_B = QLabel(image_container)
label_A = QLabel(image_container)  # A가 마지막에 생성되어 가장 앞에 위치

# 기본 레이블 설정
for label in (label_C, label_B, label_A):  # 순서 변경
    label.setFixedSize(width, height)
    label.setStyleSheet("QLabel { background-color: transparent; }")

# 초기 위치 설정
label_B.move(320, 0)  # 중앙
label_A.move(0, 0)    # 왼쪽
label_C.move(640, 0)  # 오른쪽

# 명시적으로 스택 순서 설정
label_A.raise_()  # A를 최상위로

# 슬라이더 컨트롤
controls = QGroupBox("위치 조절")
slider_layout = QHBoxLayout()

# A 카메라 위치 슬라이더
a_layout = QVBoxLayout()
a_label = QLabel("A 카메라 위치")
a_slider = QSlider(Qt.Horizontal)
a_slider.setRange(0, 320)  # 0 ~ 320 (완전 분리 ~ 완전 겹침)
a_slider.setValue(0)
a_layout.addWidget(a_label)
a_layout.addWidget(a_slider)

# C 카메라 위치 슬라이더
c_layout = QVBoxLayout()
c_label = QLabel("C 카메라 위치")
c_slider = QSlider(Qt.Horizontal)
c_slider.setRange(0, 320)
c_slider.setValue(0)
c_layout.addWidget(c_label)
c_layout.addWidget(c_slider)

slider_layout.addLayout(a_layout)
slider_layout.addLayout(c_layout)
controls.setLayout(slider_layout)
main_layout.addWidget(controls)

# 슬라이더 이벤트 핸들러
def update_position_A(value):
    new_x = value  # 0에서 시작해서 오른쪽으로
    label_A.move(new_x, 0)

def update_position_C(value):
    new_x = 640 - value  # 640에서 시작해서 왼쪽으로
    label_C.move(new_x, 0)

a_slider.valueChanged.connect(update_position_A)
c_slider.valueChanged.connect(update_position_C)

window.setLayout(main_layout)
window.setWindowTitle("카메라 뷰 위치 조절")

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