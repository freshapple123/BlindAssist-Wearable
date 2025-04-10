import cv2
import numpy as np
import subprocess
import shlex
import threading

# libcamera-vid 명령어 정의
def start_camera(camera_id, frame_holder):
    cmd = f'libcamera-vid --inline --nopreview -t 0 --codec mjpeg --width 640 --height 480 --framerate 30 -o - --camera {camera_id}'
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

try:
    while True:
        if frameL[0] is None or frameR[0] is None:
            continue

        left = frameL[0]
        right = frameR[0]

        cut_px = cv2.getTrackbarPos("Left Cut(px)", "Panorama")

        hL, wL = left.shape[:2]
        cut_px = min(cut_px, wL - 1)
        left_crop = left[:, :wL - cut_px]

        # 높이 맞추기
        min_h = min(left_crop.shape[0], right.shape[0])
        left_crop = left_crop[:min_h, :]
        right_crop = right[:min_h, :]

        panorama = cv2.hconcat([left_crop, right_crop])
        cv2.imshow("Panorama", panorama)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cv2.destroyAllWindows()
