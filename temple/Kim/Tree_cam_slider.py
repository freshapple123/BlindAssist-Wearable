from PyQt5.QtWidgets import QLabel, QVBoxLayout, QApplication, QWidget
from picamera2 import Picamera2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread
import RPi.GPIO as gp
import time
import os
import numpy as np
import cv2

# 전역 변수 최적화
width, height = 320, 240
STITCH_INTERVAL = 2  # 스티칭 간격 상수화

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
        self.image_buffer = []
        self.last_stitched_image = None
        self.last_stitch_time = time.time()
        self._setup_gpio()
        
    def _setup_gpio(self):
        gp.setwarnings(False)
        gp.setmode(gp.BOARD)
        for pin in (7, 11, 12):
            gp.setup(pin, gp.OUT)
    
    def select_channel(self, index):
        if index not in adapter_info:
            return
        gpio_sta = adapter_info[index]["gpio_sta"]
        for pin, state in zip((7, 11, 12), gpio_sta):
            gp.output(pin, state)

    def init_i2c(self, index):
        os.system(adapter_info[index]["i2c_cmd"])

    def stitch_images(self, images):
        try:
            # 이미지 크기 축소하여 처리 속도 향상
            resized_images = [cv2.resize(img, (width//2, height//2)) for img in images]
            merged = cv2.hconcat(resized_images)
            return cv2.resize(merged, (width*3, height))  # 원본 크기로 복원
        except Exception as e:
            print(f"Stitching error: {e}")
            return None

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
                        buffer_count=2
                    )
                )
                picam2.start()
                time.sleep(2)
                picam2.capture_array(wait=False)
                time.sleep(0.1)
            except Exception as e:
                print("except: " + str(e))

        while True:
            images = []
            for item in ("A", "B", "C"):  # set 대신 tuple 사용
                self.select_channel(item)
                try:
                    images.append(picam2.capture_array())
                except Exception as e:
                    print(f"capture_buffer: {e}")

            current_time = time.time()
            if current_time - self.last_stitch_time >= STITCH_INTERVAL and images:
                self.last_stitch_time = current_time
                stitched_image = self.stitch_images(images)
                if stitched_image is not None:
                    # 이미지가 실제로 변경된 경우에만 UI 업데이트
                    if (self.last_stitched_image is None or 
                        not np.array_equal(stitched_image[-10:,-10:], self.last_stitched_image[-10:,-10:])):
                        self.last_stitched_image = stitched_image
                        qimage = QImage(stitched_image.data, stitched_image.shape[1],
                                      stitched_image.shape[0], stitched_image.strides[0],
                                      QImage.Format_RGB888)
                        stitched_label.setPixmap(QPixmap.fromImage(qimage))

# UI 부분 간소화
app = QApplication([])
window = QWidget()
layout = QVBoxLayout()
stitched_label = QLabel()
stitched_label.setFixedSize(width*3, height)
layout.addWidget(stitched_label)
window.setLayout(layout)
window.setWindowTitle("Arducam Multi Camera Demo")

work = WorkThread()

if __name__ == "__main__":
    work.start()
    window.show()
    app.exec()
    work.quit()
    picam2.close()