import cv2
import numpy as np
import subprocess
import shlex
import datetime
import time

cmd = 'libcamera-vid --inline --nopreview -t 0 --codec mjpeg --width 640 --height 480 --framerate 30 -o - --camera 0'

process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

capture_interval = 3
last_capture_time = time.time()

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 640, 360)

try:
    buffer = b""
    while True:
        current_time = time.time()

        buffer += process.stdout.read(4096)
        a = buffer.find(b'\xff\xd8')
        b = buffer.find(b'\xff\xd9')

        if a != -1 and b != -1:
            jpg = buffer[a:b+2]
            buffer = buffer[b+2:]

            bgr_frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

            if bgr_frame is not None:
                cv2.imshow('frame', bgr_frame)

                if current_time - last_capture_time >= capture_interval:
                    now = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
                    print("take picture :",now)
                    cv2.imwrite("./data/image/" + now + ".jpg", bgr_frame)  # save frame
                    last_capture_time = current_time

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

finally:
    process.terminate()
    cv2.destroyAllWindows()
# -----------------------
# 본 코드는 mkdir -p data/image로 파일을 생성하여 저장함.
# -----------------------