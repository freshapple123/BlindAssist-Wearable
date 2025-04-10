import cv2
import numpy as np
import subprocess
import shlex

# 예시 파라미터. (파일로 가져오는 코드 구현하도록.)
camera_matrix = np.array([[797.06780848,   0.00000000, 302.24268281],
                          [  0.00000000, 795.42029565, 202.36518816],
                          [  0.00000000,   0.00000000,   1.00000000]])

dist_coeffs = np.array([[-0.37404558, 0.22824453, -0.00193706, -0.00100882, -0.10711871]])

cmd = 'libcamera-vid --inline --nopreview -t 0 --codec mjpeg --width 640 --height 480 --framerate 30 -o - --camera 0'

process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 640, 360)

try:
    buffer = b""
    while True:
        buffer += process.stdout.read(4096)
        a = buffer.find(b'\xff\xd8')
        b = buffer.find(b'\xff\xd9')

        if a != -1 and b != -1:
            jpg = buffer[a:b+2]
            buffer = buffer[b+2:]

            bgr_frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

            if bgr_frame is not None:
                h, w = bgr_frame.shape[:2]
                new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
                calibrated_frame = cv2.undistort(bgr_frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

                x, y, w, h = roi
                calibrated_frame = calibrated_frame[y:y+h, x:x+w]

                cv2.imshow('frame', calibrated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
finally:
    process.terminate()
    cv2.destroyAllWindows()