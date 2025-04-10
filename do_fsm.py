import cv2
import torch
import numpy as np
import time
from threading import Thread
import queue

# 앞서 구현한 FSM 모델 임포트 (이전 코드에서 정의한 클래스들 사용)
from fsm_implementation import FSMDualCameraSystem, warp_image

class CameraCapture:
    """스레드로 카메라 캡처를 실행하는 클래스"""
    def __init__(self, camera_id, width=640, height=480):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.frame_queue = queue.Queue(maxsize=2)  # 최신 프레임 2개만 유지
        self.stopped = False
        
    def start(self):
        # 카메라 캡처 스레드 시작
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        if not self.cap.isOpened():
            print(f"카메라 {self.camera_id} 열기 실패")
            return False
        
        # 스레드 시작
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return True
    
    def update(self):
        # 프레임 지속적으로 업데이트
        while True:
            if self.stopped:
                break
            
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            
            # 이전 프레임 버리고 새 프레임 추가
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            else:
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put(frame)
                except queue.Empty:
                    pass
    
    def read(self):
        # 최신 프레임 반환
        if self.frame_queue.empty():
            return None
        return self.frame_queue.get()
    
    def stop(self):
        # 스레드 및 카메라 정지
        self.stopped = True
        if hasattr(self, 'thread'):
            self.thread.join()
        if hasattr(self, 'cap'):
            self.cap.release()


class RealtimeFSMSystem:
    """실시간 FSM 기반 듀얼 카메라 시스템"""
    def __init__(self, fsm_model, cam1_id=0, cam2_id=1, img_width=640, img_height=480):
        self.fsm_model = fsm_model
        self.img_width = img_width
        self.img_height = img_height
        
        # 카메라 캡처 객체 초기화
        self.cam1 = CameraCapture(cam1_id, img_width, img_height)
        self.cam2 = CameraCapture(cam2_id, img_width, img_height)
        
        # 70도 FOV를 가진 카메라의 내부 파라미터 계산 (나중에 실제 값으로 대체)
        fx = img_width / (2 * np.tan(np.radians(70) / 2))
        fy = fx
        cx = img_width / 2
        cy = img_height / 2
        
        self.K1 = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        self.K2 = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        
        # 디스플레이 창 초기화
        self.window_name = 'FSM Dual Camera System'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # 처리 시간 추적용
        self.last_time = time.time()
        self.fps = 0
        self.fps_alpha = 0.1  # FPS 업데이트 비율
        
    def start(self):
        """시스템 시작"""
        if not self.cam1.start():
            print("카메라 1 시작 실패")
            return False
        
        if not self.cam2.start():
            print("카메라 2 시작 실패")
            self.cam1.stop()
            return False
        
        print("두 카메라 캡처 시작 성공")
        return True
    
    def stop(self):
        """시스템 정지"""
        self.cam1.stop()
        self.cam2.stop()
        cv2.destroyAllWindows()
        print("시스템 정지")
    
    def process_frame(self):
        """한 프레임 처리"""
        # 카메라에서 프레임 읽기
        frame1 = self.cam1.read()
        frame2 = self.cam2.read()
        
        if frame1 is None or frame2 is None:
            return False, None
        
        # 프레임 전처리
        img1 = cv2.resize(frame1, (self.img_width, self.img_height))
        img2 = cv2.resize(frame2, (self.img_width, self.img_height))
        
        # FSM 모델로 프레임 처리
        try:
            results = self.fsm_model.process_dual_camera_images(img1, img2, self.K1, self.K2)
            
            # FPS 계산
            current_time = time.time()
            elapsed = current_time - self.last_time
            self.last_time = current_time
            
            # 지수 이동 평균으로 FPS 스무딩
            current_fps = 1.0 / elapsed if elapsed > 0 else 0
            self.fps = self.fps_alpha * current_fps + (1 - self.fps_alpha) * self.fps
            
            # 결과 이미지 준비
            output_img = self.prepare_visualization(results, img1, img2)
            
            return True, output_img
            
        except Exception as e:
            print(f"프레임 처리 오류: {e}")
            return False, None
    
    def prepare_visualization(self, results, img1, img2):
        """처리 결과 시각화"""
        # BGR -> RGB 변환 (Matplotlib 형식에서 OpenCV 형식으로)
        combined_view = results['combined_view']
        if combined_view.shape[2] == 3:
            combined_view = combined_view[:, :, ::-1]  # RGB -> BGR
        
        # 값 범위 [0,1] -> [0,255]로 변환
        combined_view = (combined_view * 255).astype(np.uint8)
        
        # 작은 영상으로 depth map 표시
        depth1 = results['depth1']
        depth2 = results['depth2']
        
        # depth map을 시각화 (0-1 범위로 정규화 후 heatmap 적용)
        depth1_vis = cv2.normalize(depth1, None, 0, 255, cv2.NORM_MINMAX)
        depth2_vis = cv2.normalize(depth2, None, 0, 255, cv2.NORM_MINMAX)
        
        depth1_vis = cv2.applyColorMap(depth1_vis.astype(np.uint8), cv2.COLORMAP_MAGMA)
        depth2_vis = cv2.applyColorMap(depth2_vis.astype(np.uint8), cv2.COLORMAP_MAGMA)
        
        # 입력 이미지와 depth 맵을 작은 크기로 조정
        small_h, small_w = 120, 160
        img1_small = cv2.resize(img1, (small_w, small_h))
        img2_small = cv2.resize(img2, (small_w, small_h))
        depth1_small = cv2.resize(depth1_vis, (small_w, small_h))
        depth2_small = cv2.resize(depth2_vis, (small_w, small_h))
        
        # FPS 정보 추가
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(combined_view, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 작은 영상들을 하단에 배치
        h, w = combined_view.shape[:2]
        small_images_row = np.hstack([img1_small, depth1_small, img2_small, depth2_small])
        padding = np.zeros((h - small_h, small_w * 4, 3), dtype=np.uint8)
        small_images_col = np.vstack([padding, small_images_row])
        
        # 메인 뷰와 작은 영상들 합치기
        output = np.hstack([combined_view, small_images_col])
        
        return output
    
    def run(self):
        """실시간 처리 루프 실행"""
        print("실시간 처리 시작. 종료하려면 'q'를 누르세요.")
        
        try:
            while True:
                success, output_img = self.process_frame()
                
                if success and output_img is not None:
                    cv2.imshow(self.window_name, output_img)
                
                # 키 입력 확인 (q: 종료)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("사용자에 의해 중단됨")
        finally:
            self.stop()


def load_calibration_data(calib_file):
    """카메라 캘리브레이션 데이터 로드"""
    try:
        # 캘리브레이션 파일에서 내부 파라미터 로드
        calib_data = np.load(calib_file, allow_pickle=True).item()
        K1 = calib_data.get('K1')
        K2 = calib_data.get('K2')
        return K1, K2
    except Exception as e:
        print(f"캘리브레이션 데이터 로드 실패: {e}")
        return None, None


def main():
    """메인 함수"""
    # 설정
    IMG_WIDTH = 640
    IMG_HEIGHT = 480
    CAM1_ID = 0  # 첫 번째 카메라 ID
    CAM2_ID = 1  # 두 번째 카메라 ID
    
    # 모델 로드
    fsm_system = FSMDualCameraSystem(img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
    
    try:
        # 사전 학습된 모델 로드
        fsm_system.load_models('depth_net.pth', 'pose_net.pth')
        print("사전 학습된 모델 로드 성공")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        print("사전 학습 없이 초기 모델로 진행합니다.")
    
    # 캘리브레이션 데이터 로드 시도
    try:
        K1, K2 = load_calibration_data('camera_calib.npy')
        if K1 is not None and K2 is not None:
            print("캘리브레이션 데이터 로드 성공")
        else:
            print("캘리브레이션 데이터 없음, 기본값 사용")
    except:
        print("캘리브레이션 파일 로드 실패, 기본값 사용")
        K1, K2 = None, None
    
    # 실시간 시스템 초기화
    realtime_system = RealtimeFSMSystem(
        fsm_system, 
        cam1_id=CAM1_ID, 
        cam2_id=CAM2_ID,
        img_width=IMG_WIDTH, 
        img_height=IMG_HEIGHT
    )
    
    # 캘리브레이션 데이터가 있으면 적용
    if K1 is not None and K2 is not None:
        realtime_system.K1 = K1
        realtime_system.K2 = K2
    
    # 시스템 시작
    if realtime_system.start():
        realtime_system.run()
    else:
        print("실시간 시스템 시작 실패")


if __name__ == "__main__":
    main()