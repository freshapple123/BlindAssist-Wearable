import cv2
import numpy as np
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QApplication, QWidget, QSlider
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from picamera2 import Picamera2

class ImageMerger(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Merger with Overlap Control")
        self.setGeometry(100, 100, 1000, 500)

        self.layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)  # 최대 겹침 비율 (0% ~ 100%)
        self.slider.setValue(20)  # 초기 겹침 비율
        self.slider.valueChanged.connect(self.update_image)
        self.layout.addWidget(self.slider)

        self.setLayout(self.layout)

        # Picamera2 초기화
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_still_configuration(main={"size": (320, 240), "format": "BGR888"}))
        self.picam2.start()

        # 카메라에서 이미지를 가져옴
        self.images = [self.capture_image() for _ in range(3)]
        self.update_image()

    def capture_image(self):
        """Picamera2에서 이미지를 캡처."""
        frame = self.picam2.capture_array()
        return frame

    def merge_images(self, images, overlap):
        """이미지를 겹침 비율에 따라 병합."""
        if len(images) < 2:
            return images[0] if images else None

        merged = images[0]
        for img in images[1:]:
            overlap_px = int(overlap * merged.shape[1] / 100)  # 겹침 픽셀 계산
            merged = cv2.hconcat([merged[:, :-overlap_px], img])  # 겹침 적용하여 병합
        return merged

    def update_image(self):
        """슬라이더 값에 따라 이미지를 병합하고 업데이트."""
        overlap = self.slider.value()  # 겹침 비율 (%)
        self.images = [self.capture_image() for _ in range(3)]  # 새 이미지 캡처
        merged_image = self.merge_images(self.images, overlap)

        if merged_image is not None:
            # OpenCV 이미지를 QImage로 변환
            height, width, channel = merged_image.shape
            bytes_per_line = channel * width
            qimage = QImage(merged_image.data, width, height, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap(qimage)
            self.image_label.setPixmap(pixmap)

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = ImageMerger()
    window.show()
    sys.exit(app.exec_())
