import numpy as np
import cv2
import glob

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objectpoints = np.zeros((8*6,3), np.float32)
objectpoints[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

worldpoints = [] # 3d point in real world space
imagepoints = [] # 2d points in image plane.

images = glob.glob('./data/image/*.jpg')

for idx, fname in enumerate(images):
    print("Processing image", idx+1, ":", fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

    if ret == True:
        print("Chessboard corners found in image:", fname)
        worldpoints.append(objectpoints)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imagepoints.append(corners2)

        img = cv2.drawChessboardCorners(img, (8,6), corners2, ret)

    else:
        print("Chessboard corners not found in image:", fname)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(worldpoints, imagepoints, gray.shape[::-1], None, None)

# .npy 파일로 저장해서 불러와서 쓸 수 있도록 구현하도록. 
print("Camera matrix:")
print(mtx)
print("\nDistortion coefficients:")
print(dist)

cv2.destroyAllWindows()

# ---------------------
# 캘리브레이션을 위한 이미지 찾는 경로 확인.
# ---------------------
