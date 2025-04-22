import cv2
from ultralytics import YOLO

# 모델 로드 (YOLOv8n이 가장 가벼움, CPU에서도 가능)
model = YOLO('yolov8n.pt')

# 라즈베리파이 카메라 연결 (보통 0번, USB면 1번)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 객체 탐지
    results = model.predict(source=frame, imgsz=640, verbose=False)
    
    # 결과에서 박스 그리기
    annotated_frame = results[0].plot()

    # 출력
    cv2.imshow('YOLO Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
