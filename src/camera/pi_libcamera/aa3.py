import cv2
import numpy as np
import subprocess
import shlex
import threading
import time


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


# 프레임 저장
frameL = [None]
frameR = [None]

# 카메라 쓰레드 시작
threading.Thread(target=start_camera, args=(0, frameL)).start()
threading.Thread(target=start_camera, args=(1, frameR)).start()

# 특징점 추출기
orb = cv2.ORB_create(1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

homography = None
last_update_time = 0
update_interval = 1.0  # 3초마다 갱신

try:
    while True:
        if frameL[0] is None or frameR[0] is None:
            continue

        left = frameL[0]
        right = frameR[0]

        current_time = time.time()
        should_update = (homography is None) or (
            (current_time - last_update_time) > update_interval
        )

        if should_update:
            kp1, des1 = orb.detectAndCompute(left, None)
            kp2, des2 = orb.detectAndCompute(right, None)

            if des1 is not None and des2 is not None:
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)

                if len(matches) > 10:
                    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(
                        -1, 1, 2
                    )
                    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(
                        -1, 1, 2
                    )

                    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
                    if H is not None:
                        homography = H
                        last_update_time = current_time  # 업데이트 시간 갱신

        if homography is not None:
            warped = cv2.warpPerspective(
                right, homography, (left.shape[1] * 2, left.shape[0])
            )
            warped[0 : left.shape[0], 0 : left.shape[1]] = left
            cv2.imshow("Live Panorama", warped)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cv2.destroyAllWindows()
