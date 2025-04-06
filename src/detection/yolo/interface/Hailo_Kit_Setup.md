# AI 키트 및 예제 실행

이 문서에서는 라즈베리파이 5와 Hailo-8L AI 가속기를 사용하여 예제를 실행하기 위한 과정을 설명한다.

---

## 📌 펌웨어 업데이트

### 1. 라즈베리파이 업데이트

```bash
$ sudo apt update && sudo apt full-upgrade
```

- 소프트웨어를 최신 버전으로 업데이트한다.

### 2. 라즈베리파이 펌웨어 업데이트

```bash
$ sudo rpi-eeprom-update
```

- 현재 펌웨어 버전을 확인한다.
- 2023년 12월 6일 이후의 펌웨어를 사용 중이라면 업데이트 생략이 가능하다.
- 업데이트가 필요하면 아래 명령어로 진행한다.

```bash
$ sudo raspi-config
```

- 설정 창에서 `[Advanced Options] >> [Bootloader Version] >> [Latest] >> [Yes] >> [Ok]` 순으로 선택하여 최신 부트로더를 사용하도록 변경한다.

```bash
$ sudo rpi-eeprom-update -a
```

- 펌웨어를 업데이트한 후 재부팅한다.

---

## 📌 라즈베리파이 Connect 설치 및 설정

### 1. 설치하기

```bash
$ sudo apt install rpi-connect
$ sudo reboot
```

- 라즈베리파이 Connect를 설치하고 재부팅한다.

### 2. 라즈베리파이 계정 연결하기

- 공식 홈페이지에서 계정을 생성한다: [Raspberry Pi Connect](https://connect.raspberrypi.com/sign-in)

```bash
$ rpi-connect signin
```

- 명령어를 입력하면 표시되는 URL을 PC에서 접속하여 장치를 등록한다.

---

## 📌 PCIe 3.0 설정하기

```bash
$ sudo raspi-config
```

- 설정 창에서 `[Advanced Options] >> [PCIe Speed] >> [Yes] >> [Ok]` 순으로 설정을 진행한다.
- 설정 완료 후 재부팅한다.

---

## 📌 AI 키트 종속성 설치하기

```bash
$ sudo apt install hailo-all
$ sudo reboot
```

- AI 키트 종속성을 설치하고 재부팅한다.

```bash
$ hailortcli fw-control identify
```

- 성공적으로 설치되었는지 확인한다.

---

## 📌 예제 실행하기

```bash
$ git clone --depth 1 https://github.com/raspberrypi/rpicam-apps.git ~/rpicam-apps
$ rpicam-hello -t 0 --post-process-file ~/rpicam-apps/assets/hailo_yolov6_inference.json --lores-width 640 --lores-height 640
```

- 예제를 다운로드하고 실행한다. 예제가 정상적으로 실행되면 설정이 완료된 것이다.

---

