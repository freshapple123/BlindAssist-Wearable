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
    return Picamera2()

def select_channel(index):
    gpio_sta = adapter_info[index]["gpio_sta"]
    for pin, state in zip((7, 11, 12), gpio_sta):
        gp.output(pin, state)
    os.system(adapter_info[index]["i2c_cmd"])

def capture_image(picam2, channel):
    select_channel(channel)
    time.sleep(0.1)
    return cv2.resize(picam2.capture_array(), (width//2, height//2))

def find_roi_between_images(img1, img2, visualize=True):
    # ORB 특징점 개수 축소
    orb = cv2.ORB_create(nfeatures=200)
    
    # 이미지 그레이스케일 변환으로 처리 속도 향상
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
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
    config = picam2.create_still_configuration(
        main={"size": (width, height), "format": "BGR888"},
        buffer_count=2,
        controls={"NoiseReductionMode": 0, "AfMode": 0, "AfTrigger": 0}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    
    try:
        while True:
            img1 = capture_image(picam2, "A")
            img2 = capture_image(picam2, "B")
            roi_info = find_roi_between_images(img1, img2)
            print("ROI:", roi_info)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        picam2.close()
        gp.cleanup()
