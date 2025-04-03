import cv2
import numpy as np

# 웹캠 설정
width, height = 640, 480  # 원하는 해상도
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

cap1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# 특징점 검출기 생성 (SIFT)
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

H = None  # 변환 행렬 저장

def find_homography(img1, img2):
    """초기 한 번만 호출하여 변환 행렬을 찾음"""
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

# 초기 변환 행렬 계산
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("웹캠 오류")
        break

    H = find_homography(frame1, frame2)
    if H is not None:
        print("변환 행렬 계산 완료")
        break

# 출력 크기 설정
output_width = width * 2
output_height = height

# 실시간 영상 스티칭
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("웹캠 오류")
        break

    # 두 번째 카메라 영상 변환
    warped_frame2 = cv2.warpPerspective(frame2, H, (output_width, output_height))

    # 최종 스티칭 결과를 위한 블랭크 이미지 생성
    stitched_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    # 왼쪽에는 첫 번째 카메라 영상 배치
    stitched_frame[:, :width] = frame1

    # 오른쪽에는 변환된 두 번째 카메라 영상 배치
    mask = (warped_frame2 > 0)  # 유효한 영역만 합치기 위한 마스크
    stitched_frame[mask] = warped_frame2[mask]

    # 화면 출력
    cv2.imshow("Stitched Video", stitched_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
