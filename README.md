# Canon Product Quality Inspection System

YOLO11 기반 세그멘테이션을 활용한 Canon 제품 검수 자동화 시스템

## 📋 프로젝트 개요

이 프로젝트는 Canon 제품의 화면 레이아웃 및 버튼 배치를 자동으로 검증하는 AI 검수 시스템입니다. YOLO11 세그멘테이션 모델을 사용하여 28개의 클래스(버튼, 창, 텍스트 등)를 탐지하고, 통계 기반 검증 로직으로 제품의 합격/불합격을 판정합니다.

### 주요 기능

- **다국어 지원**: 영어, 한국어, 일본어, 중국어(간체/번체) 화면 검증
- **8가지 Case 분류**: 언어 및 버튼 조합에 따른 케이스 자동 분류
- **통계 기반 검증**: PASS 데이터 기반 레이아웃 통계를 활용한 정밀 검수
- **실시간 UI**: Gradio 기반 웹 인터페이스 제공
- **상세 로깅**: 검수 결과 및 실패 원인 상세 분석

## 🗂️ 프로젝트 구조

```
canon/
├── UI/
│   └── ui.py                    # Gradio 기반 웹 인터페이스
├── yolo/
│   ├── configs/
│   │   └── canon_seg.yml        # YOLO 학습 설정 파일
│   ├── data/
│   │   ├── images/              # 학습/검증 이미지
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── labels/              # YOLO 세그멘테이션 라벨
│   │       ├── train/
│   │       └── val/
│   ├── scripts/
│   │   ├── train.py             # 모델 학습 스크립트
│   │   └── inference.py         # 검증 및 추론 스크립트
│   ├── runs/
│   │   └── segment/             # 학습 결과 저장
│   ├── pass_data_statitics/
│   │   └── layout_stats.json    # PASS 데이터 통계 (레이아웃 검증용)
│   └── val_inference_results/
│       └── result_logs/         # 검증 결과 로그
└── README.md
```

## 🏷️ 클래스 정의

총 28개 클래스로 구성:

| ID | 클래스명 | 설명 |
|----|---------|------|
| 0-4 | home_eng, home_kor, home_jap, home_chi_1, home_chi_2 | Home 버튼 텍스트 (언어별) |
| 5-6 | home_window, home_button | Home 버튼 윈도우 및 버튼 |
| 7-12 | back_eng/kor/jap/chi, back_window, back_button | Back 버튼 관련 |
| 13-18 | id_eng/kor/jap/chi, id_window, id_button | ID 버튼 관련 |
| 19-25 | monitor_eng/kor/jap/chi_1/chi_2, monitor_window, monitor_button | Monitor 버튼 관련 |
| 26 | screen | 스크린 영역 |
| 27 | hand | 손 영역 (가림 감지용) |

## 🎯 Case 분류

8가지 케이스로 화면 레이아웃 분류:

- **Case 1**: 영어 (home, monitor, back)
- **Case 2**: 영어 + ID (home, monitor, id)
- **Case 3**: 중국어 간체 (home_chi_1, monitor_chi_1, back_chi)
- **Case 4**: 중국어 간체 + ID
- **Case 5**: 중국어 번체 (home_chi_2, monitor_chi_2, back_chi)
- **Case 6**: 일본어 (home_jap, monitor_jap, back_jap)
- **Case 7**: 한국어 (home_kor, monitor_kor, back_kor)
- **Case 8**: 텍스트 없음 (window/button만)

## 🚀 설치 및 실행

### 요구사항

```bash
Python 3.8+
CUDA 지원 GPU (권장)
```

### 패키지 설치

```bash
pip install ultralytics torch torchvision gradio pillow
```

## ⚠️ 데이터 준비

**주의**: 본 저장소에는 기업 데이터가 포함되어 있지 않습니다. 학습 및 검증을 위해서는 자체 데이터를 준비해야 합니다.

### 데이터 구조

다음 경로에 이미지와 라벨 파일을 추가하세요:

```
yolo/data/
├── images/
│   ├── train/        # 학습용 이미지 (.jpg, .jpeg, .png 등)
│   └── val/          # 검증용 이미지
└── labels/
    ├── train/        # 학습용 라벨 (.txt, YOLO 세그멘테이션 형식)
    └── val/          # 검증용 라벨
```

### 라벨 형식

YOLO 세그멘테이션 형식 (.txt):
```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```
- `class_id`: 0-27 (클래스 ID)
- `x, y`: 정규화된 폴리곤 좌표 (0.0 ~ 1.0)

### 모델 학습

```bash
cd yolo/scripts
python train.py
```

학습 설정:
- 모델: YOLO11m-seg
- 이미지 크기: 1024x1024
- Epochs: 100
- Batch size: 8
- **결과 저장 경로**: `yolo/runs/segment/canon_yolo11m_1203/`
  - **학습 가중치**: `yolo/runs/segment/canon_yolo11m_1203/weights/best.pt`
  - **학습 로그**: `yolo/runs/segment/canon_yolo11m_1203/results.csv`
  - **검증 이미지**: `yolo/runs/segment/canon_yolo11m_1203/val_batch*.jpg`

### 검증/추론 실행

#### 1. 경로 설정

`yolo/scripts/inference.py` 파일을 열고 다음 경로들을 수정하세요:

```python
# 모델 가중치 경로 (학습된 best.pt 파일)
MODEL_PATH = "/home/injaejung/canon/yolo/runs/segment/canon_yolo11m_1203/weights/best.pt"

# 추론할 이미지 폴더 경로 (검증 데이터 또는 테스트 데이터)
SOURCE = "/home/injaejung/canon/yolo/data/images/val"

# 결과 저장 폴더
PROJECT = "/home/injaejung/canon/yolo/val_inference_results"

# 실험 이름 (결과 폴더명)
NAME = "canon_yolo11m_1213_val"

# PASS 데이터 통계 파일 경로 (레이아웃 검증용)
LAYOUT_STATS_PATH = "/home/injaejung/canon/yolo/pass_data_statitics/layout_stats.json"
```

#### 2. 추론 실행

```bash
cd yolo/scripts
python inference.py
```

#### 3. 결과 확인

- **추론 결과 이미지**: `yolo/val_inference_results/{NAME}/`
- **검수 로그**: `yolo/val_inference_results/result_logs/{NAME}.txt`
- **Case 실패 로그**: `yolo/val_inference_results/result_logs/{NAME}_case_only_fail.txt`

### UI 실행

#### 1. 경로 설정

`UI/ui.py` 파일을 열고 다음 경로들을 수정하세요:

```python
# 모델 가중치 경로
MODEL_PATH = "/home/injaejung/canon/yolo/runs/segment/canon_yolo11m_1203/weights/best.pt"

# PASS 데이터 통계 파일 경로
LAYOUT_STATS_PATH = "/home/injaejung/canon/yolo/pass_data_statitics/layout_stats.json"
```

#### 2. UI 실행

```bash
cd UI
python ui.py
```

웹 브라우저에서 `http://localhost:7860` 접속

## 📊 검증 로직

### 1단계: 세그멘테이션 수행
- YOLO11 모델을 통해 28개 클래스 탐지
- Confidence threshold 적용

### 2단계: Case 분류
- 탐지된 라벨 조합으로 Case 자동 분류
- 필수 라벨 누락 시 None 반환

### 3단계: 레이아웃 검증
- PASS 데이터 통계 기반 위치/크기 검증
- 상대 좌표 및 면적 비율 비교
- 표준편차 기반 허용 범위 체크

### 4단계: 손 가림 검증
- hand 마스크와 다른 라벨 마스크 겹침 검출
- IoU 임계값 기반 가림 판정

### 5단계: 최종 판정
- **PASS**: 모든 검증 통과
- **FAIL**: 하나 이상의 검증 실패

## 📈 결과 분석

### 검증 결과 로그

`yolo/val_inference_results/result_logs/` 디렉토리에 저장:

- **전체 결과**: `{name}.txt`
- **Case 실패만**: `{name}_case_only_fail.txt`

로그 포맷:
```
[이미지명] 검수 결과
  GT case : {Ground Truth Case}
  Pred case : {예측 Case}
  라벨별 mask 면적 (cur_ratio vs PASS mean, ratio=cur/mean):
    - {label_name}: cur={현재 비율}, mean={PASS 평균}, ratio={비율}
```

## 🛠️ 커스터마이징

### 통계 파일 생성

PASS 데이터로부터 통계 생성:

```python
# inference.py의 compute_layout_stats 함수 활용
stats = compute_layout_stats(pass_image_folder)
```

### 검증 임계값 조정

`inference.py` 또는 `ui.py`의 다음 파라미터 수정:

- `POS_TOLERANCE`: 위치 허용 오차 (기본: 3.0 std)
- `SIZE_TOLERANCE`: 크기 허용 오차 (기본: 3.0 std)
- `AREA_RATIO_MIN/MAX`: 면적 비율 허용 범위

## 📝 라이센스

이 프로젝트는 내부 사용을 목적으로 제작되었습니다.

## 👤 작성자

정인재, 오연우, 황윤성, 안지수, 설승진

## 🤝 기여

버그 리포트 및 기능 제안은 이슈로 등록해주세요.

---

**Note**: 본 프로젝트는 Canon 제품 검수 자동화를 위한 연구 프로젝트입니다. 실제 프로덕션 환경에서 사용 시 추가적인 테스트와 검증이 필요합니다.
