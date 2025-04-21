from PyQt5.QtWidgets import (QLabel, QVBoxLayout, QApplication, QWidget, 
                           QSlider, QHBoxLayout, QGroupBox)
from picamera2 import Picamera2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, Qt, pyqtSignal
import RPi.GPIO as gp
import numpy as np
import time
import os

width, height = 320, 240
overlap_width = 80  # 겹치는 영역의 너비

class CameraThread(QThread):
    frame_ready = pyqtSignal(str, QPixmap)  # 카메라 ID와 이미지를 전달하는 시그널

    def __init__(self, camera_id):
        super().__init__()
        self.camera_id = camera_id
        self.running = True
        
    def run(self):
        while self.running:
            try:
                self.select_channel(self.camera_id)
                time.sleep(0.1)
                buf = picam2.capture_array()
                buf = picam2.capture_array()
                cvimg = QImage(buf, width, height, QImage.Format_RGB888)
                pixmap = QPixmap(cvimg)
                self.frame_ready.emit(self.camera_id, pixmap)
            except Exception as e:
                print(f"Camera {self.camera_id} error: {e}")

class OverlapThread(QThread):
    overlap_ready = pyqtSignal(str, QPixmap)  # 겹침 영역 ID와 이미지를 전달하는 시그널

    def __init__(self, side):
        super().__init__()
        self.side = side  # 'left' or 'right'
        self.running = True
        self.current_frame = None
        self.main_frame = None

    def update_frames(self, frame1, frame2):
        self.current_frame = frame1
        self.main_frame = frame2
        
    def run(self):
        while self.running:
            if self.current_frame is not None and self.main_frame is not None:
                try:
                    # 겹치는 영역 계산
                    overlap = self.compute_overlap()
                    if overlap is not None:
                        self.overlap_ready.emit(self.side, overlap)
                    time.sleep(0.03)
                except Exception as e:
                    print(f"Overlap {self.side} error: {e}")

    def compute_overlap(self):
        if self.current_frame is None or self.main_frame is None:
            return None
            
        try:
            # QPixmap을 QImage로 변환
            current_img = self.current_frame.toImage()
            main_img = self.main_frame.toImage()
            
            # QImage를 numpy 배열로 변환
            current_array = self.qimage_to_numpy(current_img)
            main_array = self.qimage_to_numpy(main_img)
            
            # 겹치는 영역 추출
            if self.side == 'left':
                current_roi = current_array[:, -overlap_width:]  # A의 오른쪽 부분
                main_roi = main_array[:, :overlap_width]        # B의 왼쪽 부분
            else:  # right
                current_roi = current_array[:, :overlap_width]   # C의 왼쪽 부분
                main_roi = main_array[:, -overlap_width:]       # B의 오른쪽 부분
            
            # 이미지 블렌딩
            blended = self.blend_images(current_roi, main_roi)
            
            # numpy 배열을 QPixmap으로 변환
            height, width = blended.shape[:2]
            bytes_per_line = 3 * width
            qimg = QImage(blended.data, width, height, bytes_per_line, QImage.Format_RGB888)
            return QPixmap.fromImage(qimg)
            
        except Exception as e:
            print(f"Compute overlap error: {e}")
            return None

    def blend_images(self, img1, img2):
        try:
            # 가중치 배열 생성 (선형 그라데이션)
            weight = np.linspace(0, 1, overlap_width)
            weight = weight.reshape(1, -1, 1)  # (1, overlap_width, 1) 형태로 변경
            
            # 이미지 블렌딩
            if self.side == 'left':
                # A와 B의 겹침 영역
                blended = img1 * (1 - weight) + img2 * weight
            else:
                # B와 C의 겹침 영역
                blended = img1 * weight + img2 * (1 - weight)
            
            return blended.astype(np.uint8)
            
        except Exception as e:
            print(f"Blend images error: {e}")
            return None

    def qimage_to_numpy(self, qimage):
        """QImage를 numpy 배열로 변환"""
        width = qimage.width()
        height = qimage.height()
        
        # RGB 형식으로 변환
        ptr = qimage.bits()
        ptr.setsize(height * width * 3)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
        return arr

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_threads()

    def init_ui(self):
        self.layout = QVBoxLayout()
        
        # 이미지 컨테이너
        self.image_container = QWidget()
        self.image_container.setFixedSize(960, 240)
        
        # 메인 카메라 레이블들
        self.label_A = QLabel(self.image_container)
        self.label_B = QLabel(self.image_container)
        self.label_C = QLabel(self.image_container)
        
        # 겹침 영역 레이블들
        self.label_overlap_left = QLabel(self.image_container)
        self.label_overlap_right = QLabel(self.image_container)

        # 레이블 초기 위치 설정
        self.label_B.move(320, 0)
        self.label_A.move(0, 0)
        self.label_C.move(640, 0)
        
        self.label_overlap_left.move(320-overlap_width, 0)
        self.label_overlap_right.move(640, 0)

        # 크기 설정
        for label in (self.label_A, self.label_B, self.label_C):
            label.setFixedSize(width, height)
        
        for label in (self.label_overlap_left, self.label_overlap_right):
            label.setFixedSize(overlap_width, height)

    def init_threads(self):
        # 카메라 쓰레드
        self.thread_A = CameraThread('A')
        self.thread_B = CameraThread('B')
        self.thread_C = CameraThread('C')
        
        # 겹침 영역 쓰레드
        self.thread_overlap_left = OverlapThread('left')
        self.thread_overlap_right = OverlapThread('right')

        # 시그널 연결
        self.thread_A.frame_ready.connect(self.update_frame)
        self.thread_B.frame_ready.connect(self.update_frame)
        self.thread_C.frame_ready.connect(self.update_frame)
        
        self.thread_overlap_left.overlap_ready.connect(self.update_overlap)
        self.thread_overlap_right.overlap_ready.connect(self.update_overlap)

    @pyqtSlot(str, QPixmap)
    def update_frame(self, camera_id, pixmap):
        if camera_id == 'A':
            self.label_A.setPixmap(pixmap)
            self.thread_overlap_left.update_frames(pixmap, self.label_B.pixmap())
        elif camera_id == 'B':
            self.label_B.setPixmap(pixmap)
        elif camera_id == 'C':
            self.label_C.setPixmap(pixmap)
            self.thread_overlap_right.update_frames(pixmap, self.label_B.pixmap())

    @pyqtSlot(str, QPixmap)
    def update_overlap(self, side, pixmap):
        if side == 'left':
            self.label_overlap_left.setPixmap(pixmap)
        else:
            self.label_overlap_right.setPixmap(pixmap)