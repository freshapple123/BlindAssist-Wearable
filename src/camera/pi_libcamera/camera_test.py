# 📌 라즈베리파이5 카메라 테스트 코드 (libcamera)

import cv2
import numpy as np
import subprocess
import shlex


# 1. 카메라 정보 확인하기
result = subprocess.run(['libcamera', 'list'], capture_output=True, text=True)
print('📸 연결된 카메라 정보:')
print(result.stdout)


# 2. 카메라 영상 스트리밍 명령어 설정
cmd = 'libcamera-vid --inline --nopreview -t 0 --codec mjpeg --width 640 --height 480 --framerate 30 -o - --camera 0'

# 3. 카메라 스트리밍 프로세스 시작
process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)


try:
    buffer = b""
    while True:
        buffer += process.stdout.read(4096)
        a = buffer.find(b'\xff\xd8')  # JPEG 데이터의 시작
        b = buffer.find(b'\xff\xd9')  # JPEG 데이터의 끝

        if a != -1 and b != -1:
            jpg = buffer[a:b+2]
            buffer = buffer[b+2:]

            # JPEG 이미지를 OpenCV에서 사용할 수 있도록 변환하기
            bgr_frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

            if bgr_frame is not None:
                cv2.imshow('Camera Stream', bgr_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

finally:
    process.terminate()
    cv2.destroyAllWindows()
