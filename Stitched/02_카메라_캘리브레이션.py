import cv2
import numpy as np
import glob

# 바둑판 패턴의 크기 (내가 예로 든 것은 9x6)
CHECKERBOARD = (9, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 바둑판 모서리 좌표 준비
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# 객체 포인트와 이미지 포인트 저장할 리스트
objpoints = []  # 실제 바둑판 모서리 좌표 (3D)
imgpoints = []  # 이미지에서 찾은 모서리 좌표 (2D)

# 바둑판 이미지를 저장한 폴더 경로 (캘리브레이션용 이미지들)
images = glob.glob('calibration_images/*.jpg')  # 바둑판 이미지들이 있는 경로

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 바둑판 모서리 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # 찾은 코너 그리기 (확인용)
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Calibration', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# 카메라 캘리브레이션 수행
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 결과 출력
print("카메라 매트릭스 (camera_matrix):\n", camera_matrix)
print("\n왜곡 계수 (dist_coeffs):\n", dist_coeffs)

# 결과를 파일로 저장하기 (두 웹캠 각각에 대해 따로 저장)
np.save("camera_matrix_1.npy", camera_matrix)
np.save("dist_coeffs_1.npy", dist_coeffs)
