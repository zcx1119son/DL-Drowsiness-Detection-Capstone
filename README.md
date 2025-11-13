# **😴 DL 기반 실시간 운전자 졸음 방지 시스템 (Driver Drowsiness Detection System)**

| 구분 | 내용 |
| --- | --- |
| **팀 이름** | 졸문철TV |
| **팀원** | 홍석우 (팀장), 최윤성, 엄정민 |
| **개발 기간** | 2022년 (발표일: 2022년 12월 2일) |
| **깃허브 기여자** | @zcx1119son 외 2인 |

### 🎯 나의 핵심 기여 (My Core Contributions)

이 프로젝트에서 저는 **핵심 졸음 판별 알고리즘 설계 및 구현**을 담당했습니다.

1. **기술 기반 기준 확립:** 기존 연구 논문([4] EAR 기반 논문 등)을 심층적으로 분석하여, 단순 타이머 방식이 아닌 운전자의 생체적 반응을 정량화하는 **Eye Aspect Ratio (EAR)** 지표를 최종 졸음 판단 기준으로 확립했습니다.
2. **3가지 지표 통합 및 강건성 확보:** 눈 깜빡임(EAR), 하품(MAR), 머리 기울임(Head Pose) 세 가지 지표의 상호 보완적인 로직을 통합하여 단일 지표의 오판율을 낮추고, 시스템의 강건성(Robustness)을 크게 향상시켰습니다.
3. **임계값 최적화:** 실제 운전자 환경 시뮬레이션을 통해 EAR **0.25** 및 MAR **0.79**의 임계값을 결정하고, 알람이 발생하기 위한 최소 연속 프레임(3 프레임)을 튜닝하여 오경보를 최소화했습니다.

## **🌟 프로젝트 개요**

이 프로젝트는 **OpenCV**와 **Dlib** 라이브러리를 활용하여 **실시간**으로 운전자의 얼굴을 분석하고, **눈 감김, 하품, 머리 기울임**의 세 가지 복합적인 징후를 동시에 감지하여 졸음 운전을 예방하는 캡스톤 프로젝트입니다. **AI** 기반의 **Computer Vision** 기술을 적용하여 높은 정확도의 실시간 경고 시스템을 구현하는 데 중점을 두었습니다.

## **⚙️ 주요 기술 스택 (Tech Stack)**

| 분류 | 기술 | 역할 |
| --- | --- | --- |
| **핵심 언어** | **Python 3.10** | 전체 시스템 개발 및 알고리즘 구현 |
| **컴퓨터 비전** | **OpenCV**, **imutils** | 실시간 비디오 스트림 처리 및 화면 출력 |
| **얼굴 랜드마크** | **Dlib** (`shape_predictor_68_face_landmarks.dat`) | 얼굴 영역 및 **68**개 랜드마크 추출 |
| **핵심 알고리즘** | **EAR**, **MAR**, **PnP** | 졸음 징후 감지 및 머리 포즈 추정 |
| **데이터 처리** | **NumPy**, **SciPy** | 행렬 연산 및 유클리디안 거리 계산 |

## **💡 핵심 기능 및 알고리즘**

### **1. 3가지 복합 졸음 감지 알고리즘**

단일 지표의 한계를 극복하기 위해 3가지 지표를 통합하여 졸음 감지 정확도를 높였습니다.

| 지표 | 로직 설명 | 구현 파일 | 임계값 |
| --- | --- | --- | --- |
| **Eye Aspect Ratio (EAR)** | 눈의 수직 거리와 수평 거리의 비율을 계산하여 눈 감김 상태를 감지합니다. **연속된 프레임 (3 프레임 이상)** 동안 임계값 이하일 경우 경고를 발생시킵니다. | `EAR.py` | **0.25** |
| **Mouth Aspect Ratio (MAR)** | 입의 수직 거리와 수평 거리의 비율을 계산하여 하품(입 벌림) 상태를 감지합니다. | `MAR.py` | **0.79** |
| **Head Pose Estimation** | **OpenCV**의 `solvePnP` 함수를 사용하여 얼굴의 **3D** 포즈를 추정하고, 오일러 각도를 변환하여 **머리 기울임 각도**를 계산해 주시 태만을 감지합니다. | `HeadPose.py` | (로직 내부 정의) |

## **💻 핵심 알고리즘 코드 스니펫 (Core Algorithm Code Snippets)**

### **1. Eye Aspect Ratio (EAR) 계산 로직**

`EAR.py` 파일에 정의된 눈 종횡비 계산 함수입니다. 눈의 6개 랜드마크를 활용하여 눈 깜빡임을 정량화합니다.

```python
from scipy.spatial import distance as dist

def eye_aspect_ratio(eye):
    # 수직 눈 좌표 거리: (p2, p6)와 (p3, p5)의 유클리디안 거리
    # [Dlib의 68개 랜드마크 중 눈의 좌표 번호가 eye 배열의 인덱스로 매핑됨]
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # 수평 눈 좌표 거리: (p1, p4)의 유클리디안 거리 (눈꼬리와 눈 앞머리)
    C = dist.euclidean(eye[0], eye[3])

    # 눈 종횡비 계산: EAR = (수직거리_A + 수직거리_B) / (2.0 * 수평거리_C)
    ear = (A + B) / (2.0 * C)
    return ear

```

### **2. Mouth Aspect Ratio (MAR) 계산 로직**

`Mar.py` 파일에 정의된 입 종횡비 계산 함수입니다. 입의 12개 랜드마크 중 6개를 활용하여 하품 상태를 감지합니다.

```python
from scipy.spatial import distance as dist

def mouth_aspect_ratio(mouth):
    # 수직 입의 좌표 거리: (p51, p59)와 (p53, p57)
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])

    # 수평 입의 좌표 거리: (p49, p55)
    C = dist.euclidean(mouth[0], mouth[6])

    # 입 비율 계산: MAR = (수직거리_A + 수직거리_B) / (2.0 * 수평거리_C)
    mar = (A + B) / (2.0 * C)
    return mar

```

### **3. 머리 포즈 추정 (Head Pose Estimation) 핵심**

`HeadPose.py`는 **3D 모델 좌표**와 **2D 이미지 좌표**를 이용해 카메라 행렬을 구성하고, **OpenCV**의 **solvePnP** 함수를 사용해 실시간으로 머리의 회전 벡터를 추정합니다.

```python
import cv2
import numpy as np

def getHeadTiltAndCoords(size, image_points, frame_height):
    # 카메라 행렬 구성
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)

    # 3x3 카메라 행렬 (임시 값, 실제 구현에 맞게 조정 필요)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    # PnP (Perspective-n-Point) 알고리즘을 사용해 머리 자세 추정
    (_, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    # 회전 벡터를 행렬로 변환 후 오일러 각도를 추출하여 'head_tilt_degree' 계산
    # ... (rotationMatrixToEulerAngles 호출)
    return rotation_vector, translation_vector  # 예시 반환

```

## **🛠️ 실행 방법 (How to Run)**

### **1. 환경 설정 및 필수 라이브러리 설치**

프로젝트에 필요한 모든 라이브러리 목록과 정확한 버전은 **requirements.txt** 파일에 명시되어 있습니다. dlib 라이브러리의 특성상, 설치에 시간이 걸릴 수 있습니다.

**설치 명령어:**

pip install -r requirements.txt

### **2. dlib 랜드마크 데이터 다운로드**

시스템이 얼굴 랜드마크를 정확하게 예측하기 위해 **Dlib**의 랜드마크 예측 모델 파일이 필요합니다.

1. [shape_predictor_68_face_landmarks.dat](https://www.google.com/search?q=%5Bhttp://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2%5D%5C(http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2%5C)) 파일을 다운로드합니다.
2. 압축을 해제한 후, 해당 파일을 저장소 내의 **dlib_shape_predictor** 폴더 안에 위치시킵니다.

### **3. 애플리케이션 실행**

메인 실행 파일을 **Python**으로 실행합니다. (카메라 장치가 연결되어 있어야 합니다.)

python "Driver Drowsiness [Detection.py](http://detection.py/)"

- **결과: Webcam**이 활성화되며 실시간으로 운전자의 눈, 입, 고개 상태를 감지하고 화면에 경고 메시지를 출력합니다.

## **🤝 기여 및 참고 자료 (Contributions & References)**

이 프로젝트는 기존의 **Computer Vision** 분야에서 확립된 학술 연구 및 공개된 기술을 기반으로 구현되었습니다.

### **1. 프로젝트 관련 문서 및 자료**

| 자료 | 설명 | 링크 |
| --- | --- | --- |
| **졸업 작품 전체 자료** | 발표 **PPT**, 최종 보고서 **PDF**, 등 모든 프로젝트 문서 자료 | [팀 프로젝트](https://www.notion.so/2a72e77d543b80a09c49dfe5a4492bb9?source=copy_link) |
| **소스 코드 저장소** | 프로젝트 **Python** 소스 코드 및 **README** | [소스 링크]https://github.com/zcx1119son/DL-Drowsiness-Detection-Capstone |

### **2. 핵심 알고리즘 및 학술적 참고 문헌**

| 번호 | 참고 문헌 / 자료 | 분류 |
| --- | --- | --- |
| [1] | 이승학. “졸음운전과 교통사고” | 국내 연구 (교통사고) |
| [2] | 이대연, “졸음방지를 위한 안면검출 해석과 서비스에 관한 연구” | 국내 연구 (졸음방지) |
| [3] | Vahid Kazemi and Josephine Sullivan, “One Millisecond Face Alignment with an Ensembl[($\\text{EAR}$ 기반) | **핵심 기술 논문** |
| [5] | 오미연, “얼굴 특징점 기반의 졸음운전 감지 알고리즘” | 국내 연구 (감지 알고리즘) |
| [6] | Philipp P. Cafﬁer, Udo Erdmann ,Peter Ullsperger, “The spontaneous eye-blink as sleepiness indicator in patients with obstructive sleep apnoea syndrome-a pilot study” | 학술 논문 (졸음 지표) |
| [7] | dohyeon2’s log, “ **영상처리** 2D영상에서 물체까지 3D 거리 구하기” **PnP** 구현 참고) | 기술 블로그/구현 가이드 |
