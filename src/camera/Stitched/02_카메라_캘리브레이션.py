import cv2
import cv2.aruco as aruco
import numpy as np

# =======================
# 설정
# =======================
CAMERA_INDEX = 0  # 사용할 웹캠 번호 (0 또는 1)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# Charuco 보드 설정
squaresX = 9  # 가로 사각형 수
squaresY = 6  # 세로 사각형 수
squareLength = 40  # 사각형 한 변의 크기 (픽셀)
markerLength = 20  # ArUco 마커의 한 변 크기 (픽셀)
charuco_board = aruco.CharucoBoard((squaresX, squaresY), squareLength, markerLength, aruco_dict)

# 캘리브레이션을 위한 포인트 저장 리스트
all_corners = []
all_ids = []
image_size = None

# 웹캠 설정
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Charuco 보드를 웹캠에 비춰주세요. 스페이스바를 눌러 데이터를 저장하고 'q'를 눌러 종료합니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("웹캠에서 이미지를 읽을 수 없습니다.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if image_size is None:
        image_size = gray.shape[::-1]  # 이미지 크기 저장

    # ArUco 마커 검출
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)

    if len(corners) > 0:
        # 검출된 마커들을 영상에 표시
        aruco.drawDetectedMarkers(frame, corners, ids)

        # Charuco 보드의 코너 검출
        ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray, charuco_board)

        if ret > 0 and len(charuco_corners) >= 4:  # 검출된 코너가 4개 이상인 경우만 저장
            # 영상에 Charuco 코너 표시
            aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)

            # 스페이스바를 눌렀을 때만 데이터를 저장
            if cv2.waitKey(1) & 0xFF == ord(' '):  # 스페이스바를 눌렀을 때
                print("데이터 저장")
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)

    # 영상 출력
    cv2.imshow("Calibration", frame)

    # 'q'를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# =======================
# 카메라 캘리브레이션
# =======================
if len(all_corners) > 0:
    print(f"총 {len(all_corners)} 프레임에서 Charuco 보드를 검출했습니다.")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=charuco_board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    # 결과 출력
    print("카메라 매트릭스 (camera_matrix):\n", camera_matrix)
    print("\n왜곡 계수 (dist_coeffs):\n", dist_coeffs)

    # 결과 저장
    np.save(f"camera_matrix_{CAMERA_INDEX}.npy", camera_matrix)
    np.save(f"dist_coeffs_{CAMERA_INDEX}.npy", dist_coeffs)

else:
    print("캘리브레이션을 위한 Charuco 보드를 충분히 인식하지 못했습니다.")
