## 📌 라즈베리파이5 카메라 설정 및 출력 테스트 가이드

### 1. 라즈베리파이5 설정 및 카메라 연결

#### 1-1. 운영체제 설정
라즈베리파이5는 **Bookworm OS**만 지원하며, 이전 버전의 OS는 호환되지 않는다. 기존에 사용되던 **Picamera 라이브러리**는 Bookworm OS에서 더 이상 지원되지 않으므로, 대신 **libcamera 라이브러리**를 사용해야 한다.

#### 1-2. config.txt 파일 설정

라즈베리파이5는 최대 2개의 카메라를 연결할 수 있다. **config.txt 파일 수정은 연결된 카메라의 칩셋을 인식하고 활성화하기 위해 반드시 필요하다.**

사용 중인 카메라의 칩셋 정보를 확인한 후, 해당 칩셋에 맞게 `dtoverlay` 설정을 추가해야 한다. 예를 들어, imx219 칩셋을 사용하는 경우 다음과 같은 설정을 추가한다:

```bash
sudo nano /boot/firmware/config.txt
```

```bash
dtoverlay=imx219,cam0
dtoverlay=imx219,cam1
```

다른 칩셋을 사용하는 경우에도 같은 방식으로 설정을 추가하면 된다. 예를 들어 OV5647 칩셋의 경우:

```bash
dtoverlay=ov5647,cam0
```

**이 설정은 칩셋 종류에 따라 달라지므로, 사용자가 사용하는 카메라의 칩셋을 확인하는 것이 중요하다.**

---

### 2. 카메라 사용 테스트 (libcamera 라이브러리)

#### 2-1. 카메라 칩셋 정보 확인

카메라가 사용하는 칩셋을 확인하려면 다음 명령어를 사용한다:
```bash
libcamera-hello --list-cameras
```
또는
```bash
libcamera-vid --list-cameras
```

이 명령어를 사용하면 현재 연결된 카메라의 칩셋 정보가 출력된다. 이를 통해 사용 중인 카메라가 어떤 칩셋을 사용하는지 확인하고, config.txt 파일을 올바르게 설정할 수 있다.

---

#### 2-2. 카메라 화면 출력 코드

이 코드의 목적은 연결된 카메라의 영상을 스트리밍하여 화면에 출력하는 것이다. 각 코드의 기능을 하나하나 설명한다.

1. **필요한 라이브러리 불러오기:**
```python
import cv2
import numpy as np
import subprocess
import shlex
```
- `cv2`: OpenCV 라이브러리로, 영상 처리를 담당한다.
- `numpy`: 이미지 데이터를 배열로 변환하고 처리하는 데 사용된다.
- `subprocess`: 터미널 명령어를 실행하기 위해 사용된다.
- `shlex`: 명령어를 안전하게 분리하기 위해 사용된다.

2. **카메라 칩셋 정보 확인하기:**
```python
result = subprocess.run(['libcamera', 'list'], capture_output=True, text=True)
print('카메라 정보:')
print(result.stdout)
```
- `libcamera list` 명령어를 사용하여 연결된 카메라의 정보를 출력한다.
- `result.stdout`에는 카메라 칩셋 정보가 텍스트 형태로 저장된다.

3. **카메라 영상 스트리밍 명령어 설정:**
```python
cmd = 'libcamera-vid --inline --nopreview -t 0 --codec mjpeg --width 640 --height 480 --framerate 30 -o - --camera 0'
```
- 카메라 스트리밍을 위한 명령어를 정의한다.
- `--width`와 `--height`는 영상의 해상도를 설정한다.
- `--framerate`는 초당 프레임 수를 의미한다.
- `--camera 0`은 첫 번째 카메라를 사용하도록 지정한다.

4. **카메라 스트리밍 프로세스 시작:**
```python
process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
```
- `subprocess.Popen`을 이용하여 설정된 명령어를 실행한다.
- 출력 영상 데이터를 파이프 형태로 전달받는다.

5. **영상 버퍼 처리 및 출력:**
```python
try:
    buffer = b""
    while True:
        buffer += process.stdout.read(4096)
        a = buffer.find(b'\xff\xd8')
        b = buffer.find(b'\xff\xd9')

        if a != -1 and b != -1:
            jpg = buffer[a:b+2]
            buffer = buffer[b+2:]

            bgr_frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

            if bgr_frame is not None:
                cv2.imshow('Camera Stream', bgr_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
```
- `buffer` 변수에 스트리밍된 데이터를 계속 추가한다.
- JPEG 데이터의 시작(`\xff\xd8`)과 끝(`\xff\xd9`)을 찾아서 영상으로 변환한다.
- `cv2.imshow`를 사용하여 영상을 화면에 출력한다.
- 키보드에서 `q`를 누르면 종료된다.

6. **프로세스 종료 및 창 닫기:**
```python
finally:
    process.terminate()
    cv2.destroyAllWindows()
```
- 스트리밍 프로세스를 안전하게 종료하고 모든 OpenCV 창을 닫는다.

---

### 2-3. 코드 실행 방법
코드를 파이썬 파일로 저장한 후, 아래 명령어로 실행한다:

(현재 위치를 docs라 가정)
```bash
python3 ../src/camera/pi_libcamera/camera-test.py
```

카메라 스트리밍 창이 정상적으로 출력되면 설정이 완료된 것이다.



---
