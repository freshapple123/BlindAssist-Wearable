import cv2
import numpy as np
import subprocess
import shlex
import threading

def start_camera(camera_id, frame_holder):
    cmd = f"libcamera-vid --inline --nopreview -t 0 --codec mjpeg --width 400 --height 240 --framerate 30 -o - --camera {camera_id}"
    process = subprocess.Popen(
        shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )

    buffer = b""
    while True:
        data = process.stdout.read(4096)
        if not data:
            break
        buffer += data

        a = buffer.find(b"\xff\xd8")
        b = buffer.find(b"\xff\xd9")

        if a != -1 and b != -1:
            jpg = buffer[a : b + 2]
            buffer = buffer[b + 2 :]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                frame = cv2.flip(frame, 0)
                frame_holder[0] = frame

    process.terminate()

# 프레임 저장 변수
frameL = [None]
frameR = [None]

# 카메라 쓰레드 시작
threadL = threading.Thread(target=start_camera, args=(0, frameL))
threadR = threading.Thread(target=start_camera, args=(1, frameR))
threadL.start()
threadR.start()

try:
    while True:
        if frameL[0] is None or frameR[0] is None:
            continue

        left = frameL[0]
        right = frameR[0]

        # 고정된 파라미터 조절 (필요시 수동 변경)
        cut_px = 247      # 왼쪽 영상 자르는 부분
        align_px = -10    # 오른쪽 영상 수직 정렬 (음수면 위로)

        hL, wL = left.shape[:2]
        cut_px = min(cut_px, wL - 1)
        left_crop = left[:, : wL - cut_px]

        # 높이 맞추기
        min_h = min(left_crop.shape[0], right.shape[0])
        left_crop = left_crop[:min_h, :]
        right_crop = right[:min_h, :]

        # 수직 정렬
        right_crop = np.roll(right_crop, align_px, axis=0)

        # 파노라마 스티칭
        panorama = cv2.hconcat([left_crop, right_crop])
        cv2.imshow("Live Stitching", panorama)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cv2.destroyAllWindows()
