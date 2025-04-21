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
        gp.output(pin, False)  # 초기 상태 초기화
    
    picam2 = Picamera2()
    config = picam2.create_still_configuration(
        main={"size": (width, height), 
              "format": "RGB888"},
        buffer_count=4,  # 버퍼 수 증가
        controls={
            "NoiseReductionMode": 0,
            "AfMode": 0,
            "AfTrigger": 0,
            "FrameTimeout": 1000,  # 타임아웃 설정
            "FrameDurationLimits": (33333, 33333)
        }
    )
    picam2.configure(config)
    return picam2

def select_channel(index):
    gpio_sta = adapter_info[index]["gpio_sta"]
    for pin, state in zip((7, 11, 12), gpio_sta):
        gp.output(pin, state)
    os.system(adapter_info[index]["i2c_cmd"])

def capture_image(picam2, channel, retry_count=3):
    for attempt in range(retry_count):
        try:
            select_channel(channel)
            time.sleep(0.5)  # 채널 전환 후 안정화 대기
            
            # 버퍼 클리어
            for _ in range(2):
                picam2.capture_array(wait=False)
                time.sleep(0.1)
            
            frame = picam2.capture_array(wait=True)
            if frame is not None:
                return cv2.resize(frame, (width//2, height//2))
        except Exception as e:
            print(f"Capture error on attempt {attempt + 1}: {e}")
            if attempt < retry_count - 1:
                time.sleep(1)  # 재시도 전 대기
                try:
                    # 카메라 재초기화 시도
                    picam2.stop()
                    time.sleep(0.5)
                    picam2.start()
                    time.sleep(1)
                except:
                    pass
    return None

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
    picam2 = None
    try:
        picam2 = setup_camera()
        picam2.start()
        time.sleep(2)  # 충분한 초기화 대기 시간
        
        while True:
            img1 = capture_image(picam2, "A")
            if img1 is None:
                print("Camera A capture failed")
                time.sleep(1)
                continue
                
            img2 = capture_image(picam2, "B")
            if img2 is None:
                print("Camera B capture failed")
                time.sleep(1)
                continue
            
            roi_info = find_roi_between_images(img1, img2)
            print("ROI:", roi_info)
            
            if cv2.waitKey(1000) & 0xFF == ord('q'):  # 대기 시간 증가
                break
                
    except KeyboardInterrupt:
        print("Program terminated by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if picam2:
            picam2.close()
        cv2.destroyAllWindows()
        gp.cleanup()
