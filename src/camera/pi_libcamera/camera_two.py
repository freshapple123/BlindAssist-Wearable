import cv2
import numpy as np
import subprocess
import shlex
import threading

# 카메라 데이터를 받는 함수 정의
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
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    buffer[0] = frame
    finally:
        process.terminate()

# 라즈베리파이 카메라 명령어 설정 (카메라 0, 카메라 1)
cmd_1 = 'libcamera-vid --inline --nopreview -t 0 --codec mjpeg --width 640 --height 480 --framerate 30 -o - --camera 0'
cmd_2 = 'libcamera-vid --inline --nopreview -t 0 --codec mjpeg --width 640 --height 480 --framerate 30 -o - --camera 1'

# 카메라 데이터를 저장할 버퍼 (공유 메모리)
frame1 = [None]
frame2 = [None]

# 스레드 동기화를 위한 Lock
lock1 = threading.Lock()
lock2 = threading.Lock()

# 각 카메라를 독립적으로 실행하는 스레드 생성
thread1 = threading.Thread(target=read_camera, args=(cmd_1, frame1, lock1))
thread2 = threading.Thread(target=read_camera, args=(cmd_2, frame2, lock2))

# 스레드 실행
thread1.start()
thread2.start()

# 출력 창의 크기 설정
output_width = 640 * 2  # 두 카메라를 가로로 배치할 것이므로 두 배로 설정
output_height = 480

while True:
    with lock1:
        img1 = frame1[0].copy() if frame1[0] is not None else np.zeros((480, 640, 3), dtype=np.uint8)
    with lock2:
        img2 = frame2[0].copy() if frame2[0] is not None else np.zeros((480, 640, 3), dtype=np.uint8)

    # 두 카메라 영상을 하나의 창에 붙이기 (좌우로 연결)
    combined_image = np.hstack((img1, img2))

    # 결과 출력
    cv2.imshow("Combined Camera Stream", combined_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
