import cv2
import numpy as np
import subprocess
import shlex
import threading

# 카메라 스트림 가져오기 함수
def read_camera(cmd, buffer, lock):
    process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    temp_buffer = b""

    try:
        while True:
            temp_buffer += process.stdout.read(4096)
            a = temp_buffer.find(b'\xff\xd8')
            b = temp_buffer.find(b'\xff\xd9')

            if a != -1 and b != -1:
                jpg = temp_buffer[a:b+2]
                temp_buffer = temp_buffer[b+2:]

                with lock:
                    buffer[0] = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
    finally:
        process.terminate()

# 특징점 검출기 설정 (SIFT)
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

H = None  # 변환 행렬 저장

# 변환 행렬 찾기 함수
def find_homography(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        return H
    return None

# 카메라 명령어 설정 (libcamera-vid)
cmd_1 = 'libcamera-vid --inline --nopreview -t 0 --codec mjpeg --width 640 --height 480 --framerate 30 -o - --camera 0'
cmd_2 = 'libcamera-vid --inline --nopreview -t 0 --codec mjpeg --width 640 --height 480 --framerate 30 -o - --camera 1'

# 카메라 버퍼
frame1 = [None]
frame2 = [None]

# 스레드 동기화용 Lock
lock1 = threading.Lock()
lock2 = threading.Lock()

# 두 카메라를 각각의 스레드로 실행
thread1 = threading.Thread(target=read_camera, args=(cmd_1, frame1, lock1))
thread2 = threading.Thread(target=read_camera, args=(cmd_2, frame2, lock2))

thread1.start()
thread2.start()

# 초기 변환 행렬 계산
H = None
while True:
    with lock1:
        img1 = frame1[0]
    with lock2:
        img2 = frame2[0]

    if img1 is not None and img2 is not None:
        H = find_homography(img1, img2)
        if H is not None:
            print("변환 행렬 계산 완료")
            break

# 출력 크기 설정
output_width = 1280  # 스티칭 결과의 너비 (카메라 두 개를 나란히 배치하기 위해서 두 배로 설정)
output_height = 480  # 스티칭 결과의 높이 (카메라 해상도와 동일)

# 실시간 영상 스티칭
while True:
    with lock1:
        img1 = frame1[0]
    with lock2:
        img2 = frame2[0]

    if img1 is None or img2 is None:
        continue

    # 두 번째 카메라의 영상 변환
    warped_img2 = cv2.warpPerspective(img2, H, (output_width, output_height))

    # 결과 이미지를 담을 빈 이미지 생성
    stitched_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    # 첫 번째 카메라 영상 넣기
    stitched_image[0:480, 0:640] = img1

    # 두 번째 카메라 영상 넣기
    mask = (warped_img2 > 0)
    stitched_image[mask] = warped_img2[mask]

    # 결과 출력
    cv2.imshow("Stitched Video", stitched_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
