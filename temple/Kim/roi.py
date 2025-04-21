from picamera2 import Picamera2
import RPi.GPIO as gp
import cv2
import numpy as np
import time
import os

width, height = 160, 120  # 이미지 크기 축소

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

def setup_camera():
    gp.setwarnings(False)
    gp.setmode(gp.BOARD)
    for pin in (7, 11, 12):
        gp.setup(pin, gp.OUT)
    picam2 = Picamera2()
    
    # PDAF 에러 해결을 위한 상세 설정
    config = picam2.create_still_configuration(
        main={"size": (width, height), 
              "format": "RGB888"},
        controls={
            "NoiseReductionMode": 1,        # Fast 노이즈 감소
            "AfMode": 0,                    # 수동 포커스
            "AfTrigger": 0,                 # AF 트리거 비활성화
            "LensPosition": 1.5,            # 고정 포커스 위치
            "FrameDurationLimits": (33333, 33333),  # 프레임 속도 제한
        }
    )
    picam2.configure(config)
    return picam2

def select_channel(index):
    gpio_sta = adapter_info[index]["gpio_sta"]
    for pin, state in zip((7, 11, 12), gpio_sta):
        gp.output(pin, state)
    os.system(adapter_info[index]["i2c_cmd"])

def capture_image(picam2, channel):
    select_channel(channel)
    time.sleep(0.5)  # 안정화를 위한 대기 시간 증가
    frame = picam2.capture_array()
    time.sleep(0.1)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def find_roi_between_images(img1, img2, visualize=True):
    # 이미지 전처리 추가
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 노이즈 제거
    gray1 = cv2.GaussianBlur(gray1, (5,5), 0)
    gray2 = cv2.GaussianBlur(gray2, (5,5), 0)
    
    # CLAHE 적용으로 대비 향상
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray1 = clahe.apply(gray1)
    gray2 = clahe.apply(gray2)
    
    # ORB 파라미터 조정
    orb = cv2.ORB_create(
        nfeatures=1000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        patchSize=31
    )
    
    # 특징점 검출
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None:
        return None
    
    # 매칭 - 상위 10개만
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)[:10]
    
    # ROI 계산
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    roi = {
        'img1': {'start': int(np.min(src_pts[:,0])), 'end': int(np.max(src_pts[:,0]))},
        'img2': {'start': int(np.min(dst_pts[:,0])), 'end': int(np.max(dst_pts[:,0]))}
    }
    
    if visualize:
        # matplotlib 대신 OpenCV 사용
        vis_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('ROI Detection', vis_img)
        cv2.waitKey(1)  # non-blocking
    
    return roi

if __name__ == "__main__":
    picam2 = setup_camera()
    picam2.start()
    time.sleep(2)  # 초기 안정화 대기
    
    try:
        while True:
            img1 = capture_image(picam2, "A")
            img2 = capture_image(picam2, "B")
            
            if img1 is None or img2 is None:
                print("이미지 캡처 실패")
                continue
                
            roi_info = find_roi_between_images(img1, img2)
            if roi_info:
                print("ROI:", roi_info)
            
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
                
    finally:
        cv2.destroyAllWindows()
        picam2.close()
        gp.cleanup()
