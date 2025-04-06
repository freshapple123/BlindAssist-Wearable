import cv2

# ArUco 사전 정의
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Charuco 보드 생성
board = cv2.aruco.CharucoBoard((9, 6), 0.04, 0.02, aruco_dict)

# 보드 이미지 생성 (generateImage 사용)
board_image = board.generateImage((800, 600))

# 보드 이미지 출력
cv2.imshow('Charuco Board', board_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 보드 이미지 저장
cv2.imwrite("charuco_board.png", board_image)
