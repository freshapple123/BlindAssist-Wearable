import cv2
import numpy as np
import subprocess
import shlex
import threading

# 카메라 프레임을 받아오는 함수
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

# 프레임 저장 변수
frameL = [None]
frameR = [None]

# 카메라 쓰레드 시작
threadL = threading.Thread(target=start_camera, args=(0, frameL))
threadR = threading.Thread(target=start_camera, args=(1, frameR))
threadL.start()
threadR.start()

# 이전 Homography 행렬
prev_H = None
frame_count = 0
detect_interval = 10  # 몇 프레임마다 새로 계산할지

orb = cv2.ORB_create(nfeatures=500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

cv2.namedWindow("Live Stitching")

try:
    while True:
        if frameL[0] is None or frameR[0] is None:
            continue

        left = frameL[0]
        right = frameR[0]

        H = prev_H
        if frame_count % detect_interval == 0:
            kp1, des1 = orb.detectAndCompute(left, None)
            kp2, des2 = orb.detectAndCompute(right, None)

            if des1 is not None and des2 is not None:
                matches = bf.knnMatch(des1, des2, k=2)

                good = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append(m)

                if len(good) > 10:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        if H is not None:
            prev_H = H
            warped_right = cv2.warpPerspective(right, H, (left.shape[1] + right.shape[1], left.shape[0]))
            stitched = warped_right.copy()
            stitched[0:left.shape[0], 0:left.shape[1]] = left
        else:
            # fallback: 그냥 좌우로 붙이기
            stitched = np.hstack((left, right))

        cv2.imshow("Live Stitching", stitched)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

finally:
    cv2.destroyAllWindows()
