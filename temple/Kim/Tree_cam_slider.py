import cv2
import numpy as np
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QApplication, QWidget, QSlider
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from picamera2 import Picamera2

class CameraThread(QThread):
    images_captured = pyqtSignal(list)

    def __init__(self, picam2):
        super().__init__()
        self.picam2 = picam2
        self.running = True

    def run(self):
        while self.running:
            images = [self.picam2.capture_array() for _ in range(3)]
            self.images_captured.emit(images)

    def stop(self):
        self.running = False
        self.wait()

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
        self.slider.setMaximum(100)
        self.slider.setValue(20)
        self.slider.valueChanged.connect(self.update_image)
        self.layout.addWidget(self.slider)

        self.setLayout(self.layout)

        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_still_configuration(main={"size": (320, 240), "format": "BGR888"}))
        self.picam2.start()

        self.camera_thread = CameraThread(self.picam2)
        self.camera_thread.images_captured.connect(self.update_images)
        self.camera_thread.start()

        self.images = []
        self.last_overlap = None

    def update_images(self, images):
        self.images = images
        self.update_image()

    def merge_images(self, images, overlap):
        if len(images) < 2:
            return images[0] if images else None

        merged = images[0]
        for img in images[1:]:
            overlap_px = int(overlap * merged.shape[1] / 100)
            merged = cv2.hconcat([merged[:, :-overlap_px], img])
        return merged

    def update_image(self):
        overlap = self.slider.value()
        if overlap == self.last_overlap:
            return
        self.last_overlap = overlap

        if self.images:
            merged_image = self.merge_images(self.images, overlap)
            if merged_image is not None:
                height, width, channel = merged_image.shape
                bytes_per_line = channel * width
                qimage = QImage(merged_image.data, width, height, bytes_per_line, QImage.Format_BGR888)
                pixmap = QPixmap(qimage)
                self.image_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.camera_thread.stop()
        self.picam2.stop()
        super().closeEvent(event)

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = ImageMerger()
    window.show()
    sys.exit(app.exec_())
