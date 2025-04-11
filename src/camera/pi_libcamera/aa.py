import cv2
import numpy as np
import subprocess
import shlex
import threading


# libcamera-vid 명령어 정의
def start_camera(camera_id, frame_holder):
    cmd = f"libcamera-vid --inline --nopreview -t 0 --codec mjpeg --width 400 --height 240 --framerate 30 -o - --camera {camera_id}"
    process = subprocess.Popen(
        shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE
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
cv2.createTrackbar("Right Align(px)", "Panorama", 100, 200, lambda x: None)

# N개의 슬라이더로 수평 보정값 조절
num_sliders = 12
for i in range(num_sliders):
    cv2.createTrackbar(f"Offset {i}", "Panorama", 100, 200, lambda x: None)

try:
    while True:
        if frameL[0] is None or frameR[0] is None:
            continue

        left = frameL[0]
        right = frameR[0]

        cut_px = cv2.getTrackbarPos("Left Cut(px)", "Panorama")
        align_px = cv2.getTrackbarPos("Right Align(px)", "Panorama") - 100  # 수직 이동

        # 왼쪽 이미지 자르기
        hL, wL = left.shape[:2]
        cut_px = min(cut_px, wL - 1)
        left_crop = left[:, : wL - cut_px]

        # 높이 맞추기
        min_h = min(left_crop.shape[0], right.shape[0])
        left_crop = left_crop[:min_h, :]
        right = right[:min_h, :]

        # 보간할 offset 값 구하기
        slider_values = [
            cv2.getTrackbarPos(f"Offset {i}", "Panorama") - 100
            for i in range(num_sliders)
        ]
        slider_y = np.linspace(0, min_h - 1, num=num_sliders)
        all_y = np.arange(min_h)
        interpolated_offsets = np.interp(all_y, slider_y, slider_values).astype(int)

        # 오른쪽 이미지 각 행별로 수평 이동
        right_aligned = np.zeros_like(right)
        for y in range(min_h):
            shift = interpolated_offsets[y]
            if shift > 0:
                right_aligned[y, shift:] = right[y, :-shift]
            elif shift < 0:
                right_aligned[y, :shift] = right[y, -shift:]
            else:
                right_aligned[y] = right[y]

        # 오른쪽 이미지 수직 이동
        right_aligned = np.roll(right_aligned, align_px, axis=0)

        # 파노라마 생성
        panorama = cv2.hconcat([left_crop, right_aligned])
        cv2.imshow("Panorama", panorama)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cv2.destroyAllWindows()
