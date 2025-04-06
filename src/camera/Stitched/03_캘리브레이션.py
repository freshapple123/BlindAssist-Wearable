import cv2
import numpy as np

# 카메라 파라미터 불러오기
camera_matrix = np.load("camera_matrix_0.npy")
dist_coeffs = np.load("dist_coeffs_0.npy")

# 웹캠 설정
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("왜곡 보정된 영상을 확인하려면 웹캠을 비춰주세요.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("웹캠에서 이미지를 읽을 수 없습니다.")
        break

    # 왜곡 보정 (Undistort)
    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    # 왜곡 보정된 이미지 (중앙 영역만 잘라냄)
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # ROI에 해당하는 부분만 잘라내기 (중앙 영역)
    x, y, w, h = roi
    undistorted_frame = undistorted_frame[y:y+h, x:x+w]

    # 왜곡 전 이미지와 왜곡 후 이미지를 비교하기 위해 각각 출력
    cv2.imshow("Original Image", frame)           # 왜곡 전 이미지
    cv2.imshow("Undistorted Image", undistorted_frame)  # 왜곡 보정된 이미지

    # 'q'를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
