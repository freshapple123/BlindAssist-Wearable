from PyQt5.QtWidgets import QLabel, QVBoxLayout, QApplication, QWidget
from picamera2 import Picamera2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread
import RPi.GPIO as gp
import time
import os
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading

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
        self.image_buffer = Queue(maxsize=3)  # 이미지 버퍼 큐
        self.last_stitched_image = None
        self.last_stitch_time = time.time()
        self._setup_gpio()
        self.thread_pool = ThreadPoolExecutor(max_workers=3)  # 스레드 풀 생성
        self.running = True
        
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

    def capture_image(self, item):
        self.select_channel(item)
        try:
            return cv2.resize(picam2.capture_array(), (width//2, height//2))  # 캡처 시점에서 리사이즈
        except Exception as e:
            print(f"capture_buffer: {e}")
            return None

    def stitch_images(self, images):
        try:
            if len(images) < 2:
                return None
                
            # SIFT 파라미터 최적화
            sift = cv2.SIFT_create(nfeatures=500)  # 특징점 개수 제한
            
            # FLANN 매처 최적화
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=3)
            search_params = dict(checks=30)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            result = images[0]
            for i in range(1, len(images)):
                if images[i] is None:
                    continue
                    
                # ROI 설정으로 매칭 영역 제한
                roi_width = width // 4
                img1_roi = result[:, -roi_width:]
                img2_roi = images[i][:, :roi_width]
                
                # ROI 영역에서만 특징점 검출
                kp1, des1 = sift.detectAndCompute(img1_roi, None)
                kp2, des2 = sift.detectAndCompute(img2_roi, None)
                
                if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
                    continue

                matches = flann.knnMatch(des1, des2, k=2)
                good = [m for m, n in matches if m.distance < 0.7 * n.distance][:20]  # 상위 20개만 사용
                
                if len(good) >= 4:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    
                    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 3.0)
                    
                    if H is not None:
                        result = cv2.hconcat([result, images[i]])
            
            return cv2.resize(result, (width*3, height))
            
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

        while self.running:
            # 병렬로 이미지 캡처
            futures = [self.thread_pool.submit(self.capture_image, item) 
                      for item in ("A", "B", "C")]
            images = [f.result() for f in futures]
            
            current_time = time.time()
            if current_time - self.last_stitch_time >= STITCH_INTERVAL and any(images):
                self.last_stitch_time = current_time
                # 스티칭 처리를 별도 스레드에서 수행
                def process_stitch():
                    stitched_image = self.stitch_images(images)
                    if stitched_image is not None:
                        if (self.last_stitched_image is None or 
                            not np.array_equal(stitched_image[-10:,-10:], 
                                             self.last_stitched_image[-10:,-10:])):
                            self.last_stitched_image = stitched_image
                            qimage = QImage(stitched_image.data, stitched_image.shape[1],
                                          stitched_image.shape[0], stitched_image.strides[0],
                                          QImage.Format_RGB888)
                            stitched_label.setPixmap(QPixmap.fromImage(qimage))
                
                threading.Thread(target=process_stitch).start()

    def cleanup(self):
        self.running = False
        self.thread_pool.shutdown()

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
    work.cleanup()  # 종료 시 리소스 정리
    work.quit()
    picam2.close()