from PyQt5.QtWidgets import QLabel, QHBoxLayout, QVBoxLayout, QApplication, QWidget, QSlider
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, Qt
from picamera2 import Picamera2
import RPi.GPIO as gp
import numpy as np
import time
import os

width = 280
height = 200 

adapter_info = {
    "A": {"i2c_cmd": "i2cset -y 10 0x70 0x00 0x04", "gpio_sta": [0, 0, 1]},
    "B": {"i2c_cmd": "i2cset -y 10 0x70 0x00 0x05", "gpio_sta": [1, 0, 1]},
    "C": {"i2c_cmd": "i2cset -y 10 0x70 0x00 0x06", "gpio_sta": [0, 1, 0]},
}

class WorkThread(QThread):
    def __init__(self, update_callback):
        super().__init__()
        self.update_callback = update_callback
        self.left_crop = 0
        self.right_crop = 0
        self.running = True
        gp.setwarnings(False)
        gp.setmode(gp.BOARD)
        gp.setup(7, gp.OUT)
        gp.setup(11, gp.OUT)
        gp.setup(12, gp.OUT)

    def set_crop_values(self, left, right):
        self.left_crop = left
        self.right_crop = right

    def select_channel(self, index):
        gpio_sta = adapter_info[index]["gpio_sta"]
        gp.output(7, gpio_sta[0])
        gp.output(11, gpio_sta[1])
        gp.output(12, gpio_sta[2])

    def init_i2c(self, index):
        os.system(adapter_info[index]["i2c_cmd"])

    def get_frame(self, cam_id):
        try:
            self.select_channel(cam_id)
            self.init_i2c(cam_id)
            time.sleep(0.1)
            cam = Picamera2()
            cam.configure(cam.create_still_configuration(main={"size": (width, height), "format": "BGR888"}))
            cam.start()
            time.sleep(1)
            frame = cam.capture_array()
            cam.close()
            return frame
        except Exception as e:
            print(f"{cam_id} error: {e}")
            return np.zeros((height, width, 3), dtype=np.uint8)

    def run(self):
        while self.running:
            frame_left = self.get_frame("A")
            frame_center = self.get_frame("B")
            frame_right = self.get_frame("C")

            # Crop 비율 계산
            l_crop = int(width * self.left_crop / 100)
            r_crop = int(width * self.right_crop / 100)

            left_cropped = frame_left[:, -l_crop:] if l_crop > 0 else np.zeros((height, 0, 3), dtype=np.uint8)
            right_cropped = frame_right[:, :r_crop] if r_crop > 0 else np.zeros((height, 0, 3), dtype=np.uint8)

            # 합치기
            merged = np.concatenate((left_cropped, frame_center, right_cropped), axis=1)
            self.update_callback(merged)

            time.sleep(1)

class PanoramaApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manual Panorama")

        self.image_label = QLabel()
        self.image_label.setFixedSize(960, 240)

        self.slider_left = QSlider(Qt.Horizontal)
        self.slider_left.setRange(0, 100)
        self.slider_left.setValue(0)

        self.slider_right = QSlider(Qt.Horizontal)
        self.slider_right.setRange(0, 100)
        self.slider_right.setValue(0)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        layout.addWidget(QLabel("왼쪽 카메라 합성 비율"))
        layout.addWidget(self.slider_left)
        layout.addWidget(QLabel("오른쪽 카메라 합성 비율"))
        layout.addWidget(self.slider_right)

        self.setLayout(layout)

        self.worker = WorkThread(self.update_image)
        self.slider_left.valueChanged.connect(self.update_crop)
        self.slider_right.valueChanged.connect(self.update_crop)

        self.worker.start()

    def update_crop(self):
        self.worker.set_crop_values(self.slider_left.value(), self.slider_right.value())

    def update_image(self, frame):
        h, w, ch = frame.shape
        img = QImage(frame, w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(img))

    def closeEvent(self, event):
        self.worker.running = False
        self.worker.quit()
        self.worker.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication([])
    win = PanoramaApp()
    win.resize(1000, 400)
    win.show()
    app.exec()
