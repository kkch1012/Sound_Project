# 📊 모델링 구조 문서 (Modeling Structure Documentation)

이 문서는 Sound Project의 모든 노트북 파일과 모델링 구조를 설명합니다.

---

## 📁 노트북 파일 목록 및 설명

### 1. `01_EDA.ipynb` - 탐색적 데이터 분석

**목적**: 데이터셋의 기본 통계, 분포, 특성을 분석

**주요 내용**:
- 데이터 로드 및 구조 확인
- 상태별(State) 샘플 수 분포
- 문제별(Problem) 샘플 수 분포
- 오디오 파일 길이, 샘플링 레이트 분석
- 시각화 (히스토그램, 박스 플롯 등)

**출력**: 
- 데이터셋 통계 요약
- 클래스 불균형 분석
- 데이터 품질 검증

---

### 2. `02_Data_Augmentation.ipynb` - 데이터 증강

**목적**: 오디오 데이터 증강 기법 적용 및 증강된 데이터셋 생성

**주요 내용**:
- Time Stretching (시간 늘리기/줄이기)
- Pitch Shifting (음높이 변경)
- Time Shifting (시간 이동)
- Noise Injection (노이즈 추가)
- SpecAugment (스펙트로그램 증강)
  - Frequency Masking
  - Time Masking

**출력**:
- 증강된 데이터 저장 (`data/augmented/`)
- 증강 기법별 성능 비교

---

### 3. `03_Model_Training.ipynb` - 기본 모델 학습

**목적**: 다양한 딥러닝 모델 학습 및 성능 비교

**주요 내용**:
- **CNN 모델**: 기본 합성곱 신경망
- **CRNN 모델**: CNN + LSTM + Attention
- **Transformer 모델**: Self-attention 기반
- 피처 추출: Mel Spectrogram, MFCC
- 학습 설정: AdamW optimizer, Cosine Annealing scheduler
- 평가: Accuracy, Precision, Recall, F1-Score

**출력**:
- 학습된 모델 체크포인트 (`checkpoints/`)
- 성능 비교 리포트
- Confusion Matrix

---

### 4. `04_Hierarchical_Classification.ipynb` - 계층적 분류

**목적**: 2단계 계층적 분류 접근법 구현

**주요 내용**:
- **1단계**: 상태(State) 분류 - braking/idle/startup
- **2단계**: 각 상태별 세부 문제(Problem) 분류
  - Braking State → normal_brakes / worn_out_brakes
  - Idle State → normal_engine_idle / low_oil / power_steering / serpentine_belt
  - Startup State → normal_engine_startup / bad_ignition / dead_battery

**모델 구조**:
- State Classifier (CRNN)
- Problem Classifier (각 상태별 CRNN)

**특징**:
- 계층적 구조로 분류 정확도 향상
- 각 상태별 문제에 특화된 모델 학습

**출력**:
- 1단계 모델 체크포인트
- 2단계 모델 체크포인트 (각 상태별)
- 계층적 분류 성능 리포트

---

### 5. `04_Feature_Clustering_CRNN.ipynb` - 피처 군집 분석 및 CRNN

**목적**: 다양한 오디오 피처의 군집 분석 및 CRNN 모델 학습

**주요 내용**:
- **피처 추출**: 
  - Mel Spectrogram
  - MFCC
  - MFCC Delta
  - Chroma
  - Spectral Contrast
  - Spectral Features
  - 통계적 특징 (mean, std, max, min)
- **군집 분석**:
  - PCA (2D, 3D) 시각화
  - K-means 클러스터링
  - 성능 평가 지표 (Silhouette Score, ARI, NMI)
- **CRNN 모델 학습**:
  - Mel Spectrogram + MFCC (2채널) 입력
  - Chroma 피처 입력 (최고 성능 피처)
  - 성능 비교

**데이터 처리**:
- `combined` 폴더 제외 (더 깔끔한 군집 분석)

**출력**:
- 피처별 클러스터링 성능 비교
- 최고 성능 피처 식별 (Chroma)
- CRNN 모델 체크포인트 (멀티채널, Chroma)

---

### 6. `05_Combined_Feature_Clustering.ipynb` - 결합 피처 군집 분석

**목적**: 성능이 좋은 여러 피처들을 결합하여 고차원 특징벡터 생성 및 군집 분석

**주요 내용**:
- **피처 추출**: 다양한 오디오 피처 추출
- **피처 성능 평가**: 각 피처별 클러스터링 성능 평가
  - Silhouette Score
  - Adjusted Rand Index (ARI)
  - Normalized Mutual Information (NMI)
  - Davies-Bouldin Score
- **고차원 특징벡터 생성**: 
  - 상위 N개 (기본 5개) 피처 선택
  - 선택된 피처들을 결합하여 고차원 벡터 생성
- **결합 피처 군집 분석**:
  - PCA (2D, 3D) 시각화
  - K-means 클러스터링
- **성능 비교**: 단일 피처 vs 결합 피처

**데이터 처리**:
- `combined` 폴더 제외
- 샘플링 (1000개)으로 계산 효율성 확보

**출력**:
- 피처별 성능 순위
- 결합 피처 클러스터링 결과
- 성능 비교 리포트

---

### 7. `06_Spectrogram_CNN_Classification.ipynb` - 스펙트로그램 기반 CNN 분류

**목적**: 소리 신호를 2차원 이미지로 표현하여 CNN으로 분류하는 다양한 방법 비교

**주요 내용**:

#### 방법 1: 로그 멜 스펙트로그램 + 미분값 채널
- **입력**: Log Mel Spectrogram + Delta (2채널)
- **모델**: `SimpleCNN2Layer` (2개 합성곱 계층 + 2개 전연결 계층)
- **특징**: 미분값을 추가 채널로 사용하여 시간적 변화 정보 활용

#### 방법 2: 데이터 증강 + 로그 멜 스펙트로그램
- **입력**: Log Mel Spectrogram (1채널)
- **데이터 증강**: SpecAugment (Frequency Masking, Time Masking)
- **모델**: `SimpleCNN3Layer` (3개 합성곱 계층 + 2개 전연결 계층)
- **특징**: 데이터 증강으로 모델 일반화 성능 향상

#### 방법 3: 다양한 2D 이미지 표현 비교
- **입력**: 
  - Spectrogram (Mel Spectrogram)
  - MFCC
  - Chroma (CRP)
- **모델**: `SimpleCNN3Layer` (3개 합성곱 계층 + 2개 전연결 계층)
- **특징**: 스펙트로그램, MFCC, Chroma를 각각 이미지로 표현하여 CNN 모델 비교
- **참고**: 문헌 연구에 따르면 스펙트로그램 입력이 가장 좋은 성능을 보임

**출력**:
- 각 방법별 모델 성능 비교
- Confusion Matrix
- 전체 성능 비교 리포트

---

## 🔄 모델링 워크플로우

```
1. 데이터 분석 (01_EDA.ipynb)
   ↓
2. 데이터 증강 (02_Data_Augmentation.ipynb)
   ↓
3. 기본 모델 학습 (03_Model_Training.ipynb)
   ↓
4. 계층적 분류 실험 (04_Hierarchical_Classification.ipynb)
   ↓
5. 피처 군집 분석 및 CRNN (04_Feature_Clustering_CRNN.ipynb)
   ↓
6. 결합 피처 군집 분석 (05_Combined_Feature_Clustering.ipynb)
   ↓
7. 스펙트로그램 기반 CNN 분류 (06_Spectrogram_CNN_Classification.ipynb)
```

---

## 📊 주요 모델 아키텍처

### 1. CNN (Convolutional Neural Network)
- **용도**: 기본 이미지 분류
- **입력**: Mel Spectrogram (128 x 시간)
- **구조**: Conv2D → Pooling → FC → Softmax

### 2. CRNN (Convolutional Recurrent Neural Network)
- **용도**: 시계열 특성을 고려한 분류
- **입력**: Mel Spectrogram + MFCC (2채널)
- **구조**: 
  - CNN 부분: 공간적 특징 추출
  - LSTM 부분: 시간적 패턴 학습
  - Attention: 중요한 시간 구간에 집중

### 3. SimpleCNN2Layer
- **용도**: 방법 1 (멜 스펙트로그램 + Delta)
- **구조**: 2개 Conv Block + 2개 FC Layer

### 4. SimpleCNN3Layer
- **용도**: 방법 2, 3 (데이터 증강, 다양한 피처 비교)
- **구조**: 3개 Conv Block + 2개 FC Layer

---

## 🎯 피처 추출 방법

### 1. Mel Spectrogram
- **설명**: 멜 주파수 스케일로 변환된 스펙트로그램
- **차원**: (128, 시간)
- **용도**: CNN, CRNN 입력

### 2. MFCC (Mel-Frequency Cepstral Coefficients)
- **설명**: 음성 인식에 널리 사용되는 특징
- **차원**: (13, 시간)
- **용도**: CRNN 채널 추가, CNN 입력

### 3. Chroma
- **설명**: 음악 분석에 사용되는 특징 (12개 음계)
- **차원**: (12, 시간)
- **용도**: 군집 분석 최고 성능 피처

### 4. Spectral Contrast
- **설명**: 스펙트럼 대비 특징
- **용도**: 군집 분석

### 5. 통계적 특징
- **설명**: Mean, Std, Max, Min 등
- **용도**: 군집 분석, 고차원 특징벡터

---

## 📈 평가 지표

### 분류 성능
- **Accuracy**: 전체 정확도
- **Precision**: 정밀도
- **Recall**: 재현율
- **F1-Score**: F1 점수
- **Confusion Matrix**: 혼동 행렬

### 군집 분석 성능
- **Silhouette Score**: 군집 분리도 (-1 ~ 1)
- **Adjusted Rand Index (ARI)**: 실제 레이블과의 일치도 (0 ~ 1)
- **Normalized Mutual Information (NMI)**: 정보 공유도 (0 ~ 1)
- **Davies-Bouldin Score**: 군집 내/간 거리 비율 (낮을수록 좋음)

---

## 🗂️ 데이터 구조

### 원본 데이터
```
data/
├── braking state/
│   ├── normal_brakes/
│   └── worn_out_brakes/
├── idle state/
│   ├── normal_engine_idle/
│   ├── low_oil/
│   ├── power_steering/
│   ├── serpentine_belt/
│   └── combined/  # 일부 노트북에서 제외
└── startup state/
    ├── normal_engine_startup/
    ├── bad_ignition/
    └── dead_battery/
```

### 증강된 데이터
```
data/
└── augmented/
    ├── braking state/
    ├── idle state/
    └── startup state/
```

---

## 📝 코드 통일성 규칙

### 1. 파라미터 통일
- **KMeans**: `n_init='auto'` (기존: `n_init=10`)
- **classification_report**: `zero_division='warn'` (기존: `zero_division=0`)
- **matplotlib colormap**: `plt.cm.get_cmap('RdYlGn')` (기존: `plt.cm.RdYlGn`)

### 2. 데이터 로드 규칙
- **combined 폴더 제외**: 군집 분석 관련 노트북에서 일관되게 제외
- **상태 매핑**: `{'braking state': 0, 'idle state': 1, 'startup state': 2}`

### 3. 시각화 설정
- **색상**: `['#FF6B6B', '#4ECDC4', '#45B7D1']` (빨강, 청록, 파랑)
- **마커**: `['o', 's', '^']` (원, 사각, 삼각)
- **폰트**: 'Malgun Gothic' (한글 지원)

---

## 🎓 참고 문헌 및 연구

- **스펙트로그램 기반 CNN**: 스펙트로그램을 입력으로 사용하는 것이 MFCC, Chroma보다 더 좋은 성능을 보임
- **SpecAugment**: 음성 인식에서 널리 사용되는 데이터 증강 기법
- **계층적 분류**: 복잡한 다중 클래스 분류 문제를 단순화하는 효과적인 접근법

---

**문서 작성일**: 2025년 1월
**최종 수정일**: 2025년 1월

