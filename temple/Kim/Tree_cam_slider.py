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
                
            # ORB 특징점 검출기 사용 (SIFT보다 빠름)
            orb = cv2.ORB_create(nfeatures=1000)
            
            result = images[0]
            for i in range(1, len(images)):
                if images[i] is None:
                    continue
                    
                # 오버랩 영역 설정 (20%)
                overlap_width = int(width * 0.2)
                img1_roi = result[:, -overlap_width:]
                img2_roi = images[i][:, :overlap_width]
                
                # 특징점 검출 및 매칭
                kp1, des1 = orb.detectAndCompute(img1_roi, None)
                kp2, des2 = orb.detectAndCompute(img2_roi, None)
                
                if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
                    # 매칭 실패시 단순 연결
                    result = cv2.hconcat([result, images[i]])
                    continue

                # Brute Force 매처 사용
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                
                # 상위 10개 매칭점만 사용
                matches = sorted(matches, key=lambda x: x.distance)[:10]
                
                if len(matches) >= 4:
                    # 매칭점 좌표 추출
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    src_pts[:,:,0] += result.shape[1] - overlap_width  # 전체 이미지 기준 좌표로 변환
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    
                    # 호모그래피 계산
                    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                    
                    if H is not None:
                        # 블렌딩을 위한 마스크 생성
                        mask = np.zeros((height, overlap_width), dtype=np.float32)
                        for x in range(overlap_width):
                            mask[:,x] = x / overlap_width
                            
                        # 이미지 와핑 및 블렌딩
                        warped = cv2.warpPerspective(images[i], H, (result.shape[1] + width//2, height))
                        # 오버랩 영역 블렌딩
                        overlap_region = result[:, -overlap_width:]
                        warped_region = warped[:, :overlap_width]
                        blended = cv2.addWeighted(overlap_region, 1-mask, warped_region, mask, 0)
                        
                        # 결과 이미지 생성
                        result = cv2.hconcat([result[:, :-overlap_width], blended, warped[:, overlap_width:]])
                    else:
                        result = cv2.hconcat([result, images[i]])
                else:
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
                
                # PDAF 에러 해결을 위한 설정 수정
                config = picam2.create_still_configuration(
                    main={"size": (width, height), 
                          "format": "BGR888"},
                    buffer_count=2,
                    controls={"NoiseReductionMode": 0,  # 노이즈 감소 비활성화
                             "AfMode": 0,               # 자동초점 비활성화
                             "AfTrigger": 0}           # AF 트리거 비활성화
                )
                picam2.configure(config)
                picam2.start()
                time.sleep(1)  # 대기 시간 축소
                picam2.capture_array(wait=True)  # wait=True로 변경
                time.sleep(0.1)
            except Exception as e:
                print("except: " + str(e))

        while self.running:
            # 병렬로 이미지 캡처
            futures = [self.thread_pool.submit(self.capture_image, item) 
                      for item in ("A", "B", "C")]
            images = [f.result() for f in futures]
            
            current_time = time.time()
            # 이미지 유효성 검사 수정
            valid_images = [img for img in images if img is not None]
            
            if current_time - self.last_stitch_time >= STITCH_INTERVAL and len(valid_images) > 0:
                self.last_stitch_time = current_time
                # 스티칭 처리를 별도 스레드에서 수행
                def process_stitch():
                    stitched_image = self.stitch_images(valid_images)
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