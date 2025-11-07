# **😴 DL 기반 실시간 운전자 졸음 방지 시스템 (Driver Drowsiness Detection System)**

## **🌟 프로젝트 개요**

이 프로젝트는 $\\text{OpenCV}$와 $\\text{Dlib}$ 라이브러리를 활용하여 **실시간**으로 운전자의 얼굴을 분석하고, **눈 감김, 하품, 머리 기울임**의 세 가지 복합적인 징후를 동시에 감지하여 졸음 운전을 예방하는 캡스톤 프로젝트입니다. $\\text{AI}$ 기반의 $\\text{Computer}$ $\\text{Vision}$ 기술을 적용하여 높은 정확도의 실시간 경고 시스템을 구현하는 데 중점을 두었습니다.

## **⚙️ 주요 기술 스택 (Tech Stack)**

| 분류 | 기술 | 역할 |
| :---- | :---- | :---- |
| **핵심 언어** | $\\text{Python 3.x}$ | 전체 시스템 개발 및 알고리즘 구현 |
| **컴퓨터 비전** | $\\text{OpenCV}$, $\\text{imutils}$ | 실시간 비디오 스트림 처리 및 화면 출력 |
| **얼굴 랜드마크** | $\\text{Dlib}$ (shape\_predictor\_68\_face\_landmarks.dat) | 얼굴 영역 및 $\\text{68}$개 랜드마크 추출 |
| **핵심 알고리즘** | $\\text{EAR}$, $\\text{MAR}$, $\\text{PnP}$ | 졸음 징후 감지 및 머리 포즈 추정 |
| **데이터 처리** | $\\text{NumPy}$, $\\text{SciPy}$ | 행렬 연산 및 유클리디안 거리 계산 |

## **💡 핵심 기능 및 알고리즘**

### **1\. 3가지 복합 졸음 감지 알고리즘**

단일 지표의 한계를 극복하기 위해 3가지 지표를 통합하여 졸음 감지 정확도를 높였습니다.

| 지표 | 로직 설명 | 구현 파일 | 임계값 |
| :---- | :---- | :---- | :---- |
| **Eye Aspect Ratio (EAR)** | 눈의 수직 거리와 수평 거리의 비율을 계산하여 눈 감김 상태를 감지합니다. **연속된 프레임 (**$\\text{3}$ **프레임 이상)** 동안 임계값 이하일 경우 경고를 발생시킵니다. | EAR.py | $\\text{0.25}$ |
| **Mouth Aspect Ratio (MAR)** | 입의 수직 거리와 수평 거리의 비율을 계산하여 하품(입 벌림) 상태를 감지합니다. | MAR.py | $\\text{0.79}$ |
| **Head Pose Estimation** | $\\text{OpenCV}$의 $\\text{solvePnP}$ 함수를 사용하여 얼굴의 $\\text{3D}$ 포즈를 추정하고, 오일러 각도를 변환하여 **머리 기울임 각도**를 계산해 주시 태만을 감지합니다. | HeadPose.py | (로직 내부 정의) |

### **2\. Dlib 얼굴 랜드마크 활용**

* **사용 데이터:** $\\text{Dlib}$에서 제공하는 **shape\_predictor\_68\_face\_landmarks.dat** 파일을 사용하여 얼굴의 $\\text{68}$개 핵심 포인트를 정확하게 추출합니다.  
* **활용:** 이 랜드마크 좌표를 기반으로 눈 영역과 입 영역을 추출하여 $\\text{EAR}$ 및 $\\text{MAR}$ 계산의 입력값으로 사용합니다.

## **🛠️ 실행 방법 (How to Run)**

### **1\. 환경 설정 및 필수 라이브러리 설치**

프로젝트 실행을 위해 다음 $\\text{Python}$ 패키지들이 필요합니다. 특히, $\\text{dlib}$의 경우 설치에 다소 시간이 걸리거나 $\\text{Visual Studio C++ Build Tools}$ (Windows 환경)이 요구될 수 있습니다.

| 라이브러리 | 용도 |
| :---- | :---- |
| opencv-python | 비디오 스트림 처리 및 화면 출력 |
| dlib | 얼굴 검출 및 $\\text{68}$개 얼굴 랜드마크 예측 |
| imutils | $\\text{OpenCV}$ 기본 기능 간소화 및 비디오 스트림 처리 |
| scipy | $\\text{EAR/MAR}$ 계산을 위한 유클리디안 거리 함수 (dist.euclidean) 제공 |
| numpy | 배열 및 행렬 연산 (특히 $\\text{Head}$ $\\text{Pose}$ $\\text{Estimation}$에 사용) |

**설치 명령어:**

pip install opencv-python dlib imutils scipy numpy

### **2\. dlib 랜드마크 데이터 다운로드**

시스템이 얼굴 랜드마크를 정확하게 예측하기 위해 $\\text{Dlib}$의 랜드마크 예측 모델 파일이 필요합니다.

1. [shape\_predictor\_68\_face\_landmarks.dat](https://www.google.com/search?q=http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) 파일을 다운로드합니다.  
2. 압축을 해제한 후, 해당 파일을 저장소 내의 **dlib\_shape\_predictor** 폴더 안에 위치시킵니다.

### **3\. 애플리케이션 실행**

메인 실행 파일을 $\\text{Python}$으로 실행합니다. (카메라 장치가 연결되어 있어야 합니다.)

python "Driver Drowsiness Detection.py"

* **결과:** $\\text{Webcam}$이 활성화되며 실시간으로 운전자의 눈, 입, 고개 상태를 감지하고 화면에 경고 메시지를 출력합니다.

## **🤝 기여 및 참고 자료**

* **프로젝트** $\\text{URL}$**:** https://github.com/zcx1119son/DL-Drowsiness-Detection-Capstone