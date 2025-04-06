# Detection 폴더 안내

본 폴더는 BlindAssist-Wearable 프로젝트에서 객체 인식을 수행하고, YOLO 모델을 활용하여 특징점 매칭을 구현하는 코드가 포함된 디렉토리다.

---

## 📌 폴더 구성
```
detection/
├── matching/               # 특징점 매칭 알고리즘 코드 (예: SIFT, ORB 사용)
├── yolo/                   # YOLO 모델을 활용한 객체 탐지 코드
├── README.md               # Detection 폴더 설명 파일
```

---

## 🔍 폴더 설명

### 1. `matching/`
- 두 카메라에서 얻어진 영상의 특징점을 추출하고 비교하는 기능을 담당한다.
- SIFT, ORB 등 다양한 알고리즘을 활용하여 특징점 매칭을 수행한다.

### 2. `yolo/`
- YOLO 모델을 사용하여 객체를 실시간으로 탐지하고, 그 결과를 이용하여 특징점 매칭과 결합하는 코드가 포함된다.
- YOLO 모델을 실행하기 위한 인터페이스와 모델 변환 코드가 포함되어 있다.

---

## 📜 예제 실행 방법 (YOLO 모델 실행 - 외부 리포지토리 사용)

1. **코드 다운로드**  
```bash
$ git clone --depth 1 https://github.com/raspberrypi/rpicam-apps.git ~/rpicam-apps
```

2. **예제 실행**  
```bash
$ rpicam-hello -t 0 --post-process-file ~/rpicam-apps/assets/hailo_yolov6_inference.json --lores-width 640 --lores-height 640
```
- 해당 명령어를 통해 YOLO 모델을 실시간으로 구동한다. 이 예제는 BlindAssist-Wearable 프로젝트와는 별도로, 외부 리포지토리인 **`rpicam-apps`** 에서 가져온다. 따라서 프로젝트 내에서 사용할 때는 이 리포지토리를 별도로 설치하고 설정해야 한다.
- 모델은 Hailo-8L AI 가속기를 활용하여 실행되며, 설정 파일(`hailo_yolov6_inference.json`)을 사용한다.
- 해상도는 640x640으로 설정되어 있다.

---

## 💡 참고 사항
- YOLO 모델의 정확도를 개선하기 위해 모델을 재학습하거나 설정 파일을 수정할 수 있다.
- `matching/` 폴더의 특징점 매칭 코드와 결합하여 더 정확한 객체 인식을 수행할 수 있다.
- 최적화 작업은 계속해서 개선하며, 프레임 드랍 문제를 해결하기 위해 다양한 방법을 시도하고 있다.

---

본 폴더는 객체 인식 및 특징점 매칭 기능을 구현하기 위해 지속적으로 개선될 예정이다.

