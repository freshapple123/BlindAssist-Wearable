from PyQt5.QtWidgets import QLabel, QVBoxLayout, QApplication, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread
import RPi.GPIO as gp
import time
import os
import numpy as np
import subprocess

width = 320
height = 240 

class WorkThread(QThread):
    def __init__(self):
        super().__init__()
        self.setup_gpio()
        self.running = True
        
    def setup_gpio(self):
        gp.setwarnings(False)
        gp.setmode(gp.BOARD)
        gp.setup(7, gp.OUT)
        gp.setup(11, gp.OUT)
        gp.setup(12, gp.OUT)

    def capture_image(self, camera_id):
        # 카메라 설정
        if camera_id == 'A':
            i2c = "i2cset -y 1 0x70 0x00 0x04"
            gp.output(7, False)
            gp.output(11, False)
            gp.output(12, True)
        elif camera_id == 'B':
            i2c = "i2cset -y 1 0x70 0x00 0x05"
            gp.output(7, True)
            gp.output(11, False)
            gp.output(12, True)
        elif camera_id == 'C':
            i2c = "i2cset -y 1 0x70 0x00 0x06"
            gp.output(7, False)
            gp.output(11, True)
            gp.output(12, False)
            
        os.system(i2c)
        time.sleep(0.1)
        
        # 이미지 캡처
        temp_file = f"temp_{camera_id}.jpg"
        cmd = f"libcamera-still -n -o {temp_file} --width {width} --height {height}"
        subprocess.run(cmd.split(), capture_output=True)
        
        # 이미지 읽기
        if os.path.exists(temp_file):
            img = np.fromfile(temp_file, np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            os.remove(temp_file)
            return img
        return None

    def run(self):
        while self.running:
            try:
                frames = []
                # 각 카메라에서 순차적으로 이미지 캡처
                for camera in ['A', 'B', 'C']:
                    frame = self.capture_image(camera)
                    if frame is not None:
                        frames.append(frame)
                
                if len(frames) == 3:
                    # 세 이미지를 가로로 연결
                    combined = np.hstack(frames)
                    # Qt 이미지로 변환
                    h, w, _ = combined.shape
                    bytes_per_line = 3 * w
                    qimg = QImage(combined.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    combined_label.setPixmap(QPixmap.fromImage(qimg))
                    
            except Exception as e:
                print(f"캡처 오류: {e}")
            time.sleep(0.1)

    def cleanup(self):
        self.running = False
        gp.output(7, False)
        gp.output(11, False)
        gp.output(12, True)

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
    try:
        work.start()
        window.show()
        app.exec()
    finally:
        work.cleanup()
        work.quit()
        gp.cleanup()
