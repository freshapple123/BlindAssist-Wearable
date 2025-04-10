# 프로그래스 바 를 통해 뎁스에 따라 연결하는거

import cv2
import numpy as np

capL = cv2.VideoCapture(1)
capR = cv2.VideoCapture(0)

if not capL.isOpened() or not capR.isOpened():
    print("카메라 열기 실패")
    exit()

cv2.namedWindow("Panorama")
cv2.createTrackbar("Left Cut(px)", "Panorama", 247, 300, lambda x: None)
cv2.createTrackbar("Left Scale(%)", "Panorama", 150, 200, lambda x: None)
cv2.createTrackbar("Left Vertical Shift(px)", "Panorama", 1, 200, lambda x: None)


while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    if not retL or not retR:
        break

    cut_px = cv2.getTrackbarPos("Left Cut(px)", "Panorama")
    scale_percent = cv2.getTrackbarPos("Left Scale(%)", "Panorama")
    shift_val = (
        cv2.getTrackbarPos("Left Vertical Shift(px)", "Panorama") - 100
    )  # 기준점 0

    hL, wL = frameL.shape[:2]
    cut_px = min(cut_px, wL - 1)
    frameL_crop = frameL[:, : wL - cut_px]

    # 축소
    if scale_percent != 100:
        new_w = int(frameL_crop.shape[1] * scale_percent / 100)
        new_h = int(frameL_crop.shape[0] * scale_percent / 100)
        frameL_crop = cv2.resize(
            frameL_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )

    # 수직 이동 (위/아래 모두 가능)
    h, w = frameL_crop.shape[:2]
    frameL_shifted = np.zeros_like(frameL_crop)

    if shift_val >= 0:
        if shift_val < h:
            frameL_shifted[shift_val:, :] = frameL_crop[: h - shift_val, :]
    else:
        shift_val = abs(shift_val)
        if shift_val < h:
            frameL_shifted[: h - shift_val, :] = frameL_crop[shift_val:, :]

    # 오른쪽 영상 그대로 사용
    frameR_crop = frameR

    # 높이 맞추기
    min_h = min(frameL_shifted.shape[0], frameR_crop.shape[0])
    frameL_shifted = frameL_shifted[:min_h, :]
    frameR_crop = frameR_crop[:min_h, :]

    # 이어붙이기
    panorama = cv2.hconcat([frameL_shifted, frameR_crop])
    cv2.imshow("Panorama", panorama)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
