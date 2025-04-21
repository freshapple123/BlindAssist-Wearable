from PyQt5.QtWidgets import (QLabel, QVBoxLayout, QApplication, QWidget, 
                           QSlider, QHBoxLayout, QGroupBox, QStackedLayout, 
                           QGraphicsOpacityEffect)
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

# UI 초기화
app = QApplication([])
window = QWidget()
main_layout = QVBoxLayout()

# 이미지 표시 영역
image_widget = QWidget()
image_layout = QStackedLayout()
image_layout.setStackingMode(QStackedLayout.StackAll)

# 레이블 설정
label_A = QLabel()
label_B = QLabel()
label_C = QLabel()

# 기본 레이블 설정
for label in (label_A, label_B, label_C):
    label.setFixedSize(width, height)
    label.setStyleSheet("QLabel { background-color: transparent; }")
    image_layout.addWidget(label)

image_widget.setLayout(image_layout)
main_layout.addWidget(image_widget)

# 슬라이더 컨트롤
controls = QGroupBox("투명도 조절")
slider_layout = QHBoxLayout()

# B 카메라 슬라이더
b_layout = QVBoxLayout()
b_label = QLabel("B 카메라")
b_slider = QSlider(Qt.Horizontal)
b_slider.setRange(0, 100)
b_slider.setValue(60)
b_layout.addWidget(b_label)
b_layout.addWidget(b_slider)

# C 카메라 슬라이더
c_layout = QVBoxLayout()
c_label = QLabel("C 카메라")
c_slider = QSlider(Qt.Horizontal)
c_slider.setRange(0, 100)
c_slider.setValue(30)
c_layout.addWidget(c_label)
c_layout.addWidget(c_slider)

slider_layout.addLayout(b_layout)
slider_layout.addLayout(c_layout)
controls.setLayout(slider_layout)
main_layout.addWidget(controls)

# 슬라이더 이벤트 핸들러
def update_opacity(label, value):
    opacity = value / 100.0
    effect = label.graphicsEffect()
    if not effect:
        effect = QGraphicsOpacityEffect()
        label.setGraphicsEffect(effect)
    effect.setOpacity(opacity)

b_slider.valueChanged.connect(lambda v: update_opacity(label_B, v))
c_slider.valueChanged.connect(lambda v: update_opacity(label_C, v))

# 초기 투명도 설정
update_opacity(label_B, 60)
update_opacity(label_C, 30)

window.setLayout(main_layout)
window.setWindowTitle("카메라 뷰 투명도 조절")

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