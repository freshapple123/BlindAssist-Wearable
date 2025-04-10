import cv2
import numpy as np
import subprocess
import shlex
import threading

# libcamera-vid 명령어 정의
def start_camera(camera_id, frame_holder):
    cmd = f'libcamera-vid --inline --nopreview -t 0 --codec mjpeg --width 320 --height 240 --framerate 30 -o - --camera {camera_id}'
    process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    buffer = b""
    while True:
        data = process.stdout.read(4096)
        if not data:
            break
        buffer += data

        a = buffer.find(b'\xff\xd8')
        b = buffer.find(b'\xff\xd9')

        if a != -1 and b != -1:
            jpg = buffer[a:b+2]
            buffer = buffer[b+2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                frame = cv2.flip(frame, 0)
                frame_holder[0] = frame

    process.terminate()

# 카메라 프레임 저장용 변수
frameL = [None]
frameR = [None]

# 카메라 쓰레드 시작
threadL = threading.Thread(target=start_camera, args=(0, frameL))
threadR = threading.Thread(target=start_camera, args=(1, frameR))
threadL.start()
threadR.start()

# 슬라이더 창 생성
cv2.namedWindow("Panorama")
cv2.createTrackbar("Left Cut(px)", "Panorama", 247, 300, lambda x: None)
cv2.createTrackbar("Left Scale(%)", "Panorama", 150, 200, lambda x: None)
cv2.createTrackbar("Left Vertical Shift(px)", "Panorama", 100, 200, lambda x: None)

try:
    while True:
        if frameL[0] is None or frameR[0] is None:
            continue

        left = frameL[0]
        right = frameR[0]

        cut_px = cv2.getTrackbarPos("Left Cut(px)", "Panorama")
        scale_percent = cv2.getTrackbarPos("Left Scale(%)", "Panorama")
        shift_val = cv2.getTrackbarPos("Left Vertical Shift(px)", "Panorama") - 100

        hL, wL = left.shape[:2]
        cut_px = min(cut_px, wL - 1)
        left_crop = left[:, :wL - cut_px]

        # 스케일 조정
        if scale_percent != 100:
            new_w = int(left_crop.shape[1] * scale_percent / 100)
            new_h = int(left_crop.shape[0] * scale_percent / 100)
            left_crop = cv2.resize(left_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 수직 이동
        h, w = left_crop.shape[:2]
        left_shifted = np.zeros_like(left_crop)
        if shift_val >= 0:
            left_shifted[shift_val:, :] = left_crop[: h - shift_val, :]
        else:
            shift_val = abs(shift_val)
            left_shifted[: h - shift_val, :] = left_crop[shift_val:, :]

        # 높이 맞추기
        min_h = min(left_shifted.shape[0], right.shape[0])
        left_shifted = left_shifted[:min_h, :]
        right_crop = right[:min_h, :]

        panorama = cv2.hconcat([left_shifted, right_crop])
        cv2.imshow("Panorama", panorama)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cv2.destroyAllWindows()
