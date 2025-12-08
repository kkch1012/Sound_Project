# 🚗 차량 사운드 분류 종합 모델링 리포트

> 모든 노트북 분석 결과와 모델 실험 정보를 종합한 완전한 문서

---

## 📑 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [데이터 분석 및 전처리](#2-데이터-분석-및-전처리)
3. [피처 추출 및 분석](#3-피처-추출-및-분석)
4. [모델 아키텍처](#4-모델-아키텍처)
5. [학습 설정 및 기법](#5-학습-설정-및-기법)
6. [실험 결과 및 성능 비교](#6-실험-결과-및-성능-비교)
7. [스펙트로그램 패턴 분석](#7-스펙트로그램-패턴-분석)
8. [앙상블 모델](#8-앙상블-모델)
9. [결론 및 향후 개선 방향](#9-결론-및-향후-개선-방향)

---

## 1. 프로젝트 개요

### 1.1 목적
차량에서 발생하는 다양한 소리를 딥러닝 모델로 분류하여 차량 상태를 진단하는 시스템 개발

### 1.2 문제 정의
- **입력**: 차량 사운드 오디오 파일 (WAV 형식, 22,050 Hz 샘플링 레이트)
- **출력**: 3개 상태 × 여러 문제 = 총 14개 클래스 중 하나로 분류
- **분류 유형**: 다중 클래스 분류 (Multi-class Classification)

### 1.3 클래스 구조

#### 상태별 클래스 분류

| 상태 (State) | 문제 (Problem) | 설명 | 클래스 수 |
|-------------|---------------|------|----------|
| **braking state** | normal_brakes | 정상 브레이크 | 1 |
| | worn_out_brakes | 마모된 브레이크 | 1 |
| **idle state** | normal_engine_idle | 정상 공회전 | 1 |
| | low_oil | 오일 부족 | 1 |
| | power_steering | 파워스티어링 이상 | 1 |
| | serpentine_belt | 구동벨트 이상 | 1 |
| | combined/* | 복합 이상 (여러 문제 동시 발생) | 6개 조합 |
| **startup state** | normal_engine_startup | 정상 시동 | 1 |
| | bad_ignition | 점화 불량 | 1 |
| | dead_battery | 배터리 방전 | 1 |

**총 14개 클래스** (combined 포함)

---

## 2. 데이터 분석 및 전처리

### 2.1 데이터 구조

```
data/
├── braking state/
│   ├── normal_brakes/      # 정상 브레이크
│   └── worn_out_brakes/    # 마모된 브레이크
├── idle state/
│   ├── normal_engine_idle/
│   ├── low_oil/
│   ├── power_steering/
│   ├── serpentine_belt/
│   └── combined/           # 복합 이상 (6가지 조합)
│       ├── no oil_serpentine belt
│       ├── power steering combined_no oil
│       ├── power steering combined_no oil_serpentine belt
│       ├── power steering combined_serpentine belt
│       └── ...
└── startup state/
    ├── normal_engine_startup/
    ├── bad_ignition/
    └── dead_battery/
```

### 2.2 데이터 통계

| 항목 | 값 |
|-----|---|
| 총 클래스 수 | 14개 |
| 원본 샘플 수 | 949개 (combined 제외) |
| 증강 후 샘플 수 | 2,832개 (combined 제외) |
| 전체 샘플 수 (combined 포함) | 4,143개 |
| 샘플링 레이트 | 22,050 Hz |
| 오디오 길이 | 1.5 ~ 5.0초 |
| 평균 오디오 길이 | 1.54초 |

### 2.3 클래스 분포 (증강 전)

```
📋 클래스별 샘플 수 (증강 전, combined 제외):
  dead_battery                   57개 █████
  normal_engine_startup          61개 ██████
  bad_ignition                   62개 ██████
  worn_out_brakes                76개 ███████
  normal_brakes                  77개 ███████
  low_oil                       107개 ██████████
  serpentine_belt               116개 ███████████
  power_steering                129개 ████████████
  normal_engine_idle            264개 ██████████████████████████
```

**불균형 비율**: 약 4.6배 (최대 264개 vs 최소 57개)

### 2.4 데이터 증강

#### 2.4.1 오프라인 증강 (파일 저장)

클래스 불균형 해결 및 데이터 다양성 증가를 위해 다양한 증강 기법 적용

| 기법 | 설명 | 파라미터 | 효과 |
|-----|------|---------|------|
| **Time Stretch** | 재생 속도 변경 (RPM 변화 시뮬레이션) | rate: 0.85 ~ 1.15 | 엔진 속도 변화 |
| **Pitch Shift** | 주파수 변경 (엔진 크기 차이) | steps: -4 ~ +4 반음 | 엔진 타입 차이 |
| **Add Noise** | 가우시안 노이즈 추가 (배경 소음) | factor: 0.001 ~ 0.015 | 환경 노이즈 |
| **Volume Change** | 볼륨 변경 (마이크 거리 차이) | factor: 0.5 ~ 1.5 | 녹음 거리 차이 |
| **Time Shift** | 시간 시프트 (녹음 시작점 변화) | max: 20% | 시작 시점 변화 |
| **Reverb** | 리버브 효과 (실내/실외 환경) | room_scale: 0.5 | 환경 반사 |

#### 2.4.2 온라인 증강 (SpecAugment)

학습 중 실시간으로 적용되는 스펙트로그램 증강

| 기법 | 설명 | 파라미터 |
|-----|------|---------|
| **Frequency Masking** | 주파수 대역 마스킹 | masks: 2, param: 15 |
| **Time Masking** | 시간 구간 마스킹 | masks: 2, param: 35 |

#### 2.4.3 증강 결과

```
📊 증강 요약:
  • 원본 샘플: 949개 (combined 제외)
  • 증강 샘플: 1,883개
  • 총 샘플: 2,832개 (약 3배 증가)
```

### 2.5 데이터 분할

| 셋 | 비율 | 샘플 수 (combined 제외) |
|---|-----|----------------------|
| Train | 70% | 1,982개 |
| Validation | 15% | 425개 |
| Test | 15% | 425개 |

- **Stratified Split**: 클래스 비율 유지
- **Random Seed**: 재현 가능성 보장

---

## 3. 피처 추출 및 분석

### 3.1 오디오 설정

```python
audio_config = AudioConfig(
    sample_rate=22050,    # 샘플링 레이트
    duration=5.0,         # 오디오 길이 (초)
    n_mels=128,           # Mel 밴드 수
    n_mfcc=40,            # MFCC 계수 수
    n_fft=2048,           # FFT 윈도우 크기
    hop_length=512        # 프레임 간 이동 거리
)
```

### 3.2 추출되는 피처

| 피처 | Shape | 설명 | 용도 |
|-----|-------|------|------|
| **Mel Spectrogram** | (128, 216) | 주파수-시간 표현, 인간 청각 특성 반영 | CNN/CRNN 입력 |
| **MFCC** | (40, 216) | 스펙트럼 특성 압축, 음성 인식 표준 | 추가 피처 |
| **MFCC + Delta** | (120, 216) | MFCC + 1차/2차 미분 | 시간적 변화 포착 |
| **Chroma** | (12, 216) | 12개 음계 기반 에너지 분포 | 주기적 패턴 감지 |
| **Spectral Contrast** | (7, 216) | 주파수 대역별 대비 | 스펙트럼 차이 감지 |

### 3.3 Mel Spectrogram 특성

- **주파수 범위**: 0 ~ 11,025 Hz (나이퀴스트 주파수)
- **Mel 밴드 수**: 128개
- **시간 프레임**: 70개 (무음 제거 후 정규화)
- **입력 형태**: `(batch, 1, 128, 70)` 또는 `(batch, 1, 128, 216)`

### 3.4 정상 vs 비정상 차이점

| 특성 | 정상 소리 | 비정상 소리 |
|-----|---------|----------|
| 주파수 패턴 | 규칙적, 안정적 | 불규칙, 이상 피크 존재 |
| 시간적 변화 | 일정함 | 급격한 변화 |
| 노이즈 수준 | 낮음 | 높을 수 있음 |
| 에너지 분포 | 고르게 분산 | 특정 주파수 집중 |

### 3.5 무음 구간 제거

스펙트로그램 분석 시 무음 구간을 자동으로 제거하여 의미 있는 구간만 분석

```python
def remove_silent_frames(spectrogram_2d, energy_threshold_percentile=5):
    """무음 구간 제거"""
    # 에너지 계산
    frame_energies = np.sum(spectrogram_2d, axis=0)
    threshold = np.percentile(frame_energies, energy_threshold_percentile)
    
    # 임계값 이상인 프레임만 선택
    valid_frames = frame_energies > threshold
    
    # 정규화된 길이로 패딩/트리밍
    return normalize_time_frames(spectrogram_2d[:, valid_frames], target_length=70)
```

**결과**:
- 평균 시간 프레임: 62.2프레임
- 최대 시간 프레임: 70프레임
- 최소 시간 프레임: 17프레임
- 정규화된 시간 프레임: 70프레임

---

## 4. 모델 아키텍처

### 4.1 CNN (Convolutional Neural Network)

#### 4.1.1 아키텍처 개요

```
🏗️ CNN 아키텍처:

Input: (batch, 1, 128, 216)
    ↓
ConvBlock1: Conv2d(1→32) → BatchNorm → ReLU → MaxPool(2×2) → Dropout
    ↓ (batch, 32, 64, 108)
ConvBlock2: Conv2d(32→64) → BatchNorm → ReLU → MaxPool(2×2) → Dropout
    ↓ (batch, 64, 32, 54)
ConvBlock3: Conv2d(64→128) → BatchNorm → ReLU → MaxPool(2×2) → Dropout
    ↓ (batch, 128, 16, 27)
ConvBlock4: Conv2d(128→256) → BatchNorm → ReLU → MaxPool(2×2) → Dropout
    ↓ (batch, 256, 8, 13)
Global Average Pooling
    ↓ (batch, 256)
FC1: Linear(256→256) → BatchNorm → ReLU → Dropout
    ↓ (batch, 256)
FC2: Linear(256→128) → BatchNorm → ReLU → Dropout
    ↓ (batch, 128)
Output: Linear(128→14)
    ↓ (batch, 14)
```

#### 4.1.2 모델 상세

**파라미터**:
- 총 파라미터: **490,062개**
- 학습 가능: 490,062개

**특징**:
- 4개의 Convolutional Blocks (점진적 채널 증가: 32 → 64 → 128 → 256)
- Global Average Pooling으로 공간 정보 압축
- BatchNorm과 Dropout으로 정규화 및 과적합 방지
- Kaiming 초기화 (Conv), Xavier 초기화 (Linear)

### 4.2 CRNN (Convolutional Recurrent Neural Network)

#### 4.2.1 아키텍처 개요

```
🏗️ CRNN 아키텍처:

Input: (batch, 1, 128, 216)
    ↓
CNN Feature Extractor (3 layers)
    ↓ (batch, 128, 16, 216)
Reshape: (batch, time, features)
    ↓ (batch, 216, 2048)
Bidirectional LSTM (2 layers, hidden_size=128)
    ↓ (batch, 216, 256)
Temporal Attention
    ↓ (batch, 256)
FC Classifier
    ↓ (batch, 14)
```

#### 4.2.2 핵심 구성 요소

**1. CNN Feature Extractor**
- 주파수 축으로만 MaxPool (시간 정보 보존)
- 3개 레이어: 32 → 64 → 128 채널
- Kernel: (2, 1) MaxPool로 주파수만 압축

**2. Bidirectional LSTM**
- 양방향으로 시간적 의존성 학습
- Hidden size: 128
- Layers: 2
- Dropout: 0.3

**3. Temporal Attention**
- 중요한 시간 구간에 더 큰 가중치 부여
- Attention weights 시각화 가능

```python
class TemporalAttention(nn.Module):
    def forward(self, lstm_output):
        # Attention scores 계산
        scores = self.attention(lstm_output)
        weights = F.softmax(scores, dim=1)
        # Weighted sum
        context = torch.bmm(weights.unsqueeze(1), lstm_output)
        return context, weights
```

**파라미터**:
- 총 파라미터: **2,786,639개**
- 학습 가능: 2,786,639개

### 4.3 Attention-based Transformer (AST)

#### 4.3.1 아키텍처 개요

```
🏗️ Attention-based Transformer 아키텍처:

Input: (batch, 1, 128, 216)
    ↓
Patch Embedding (patch_size: 16×16)
    ↓ (batch, 108 + 1, 256)  # 108 patches + CLS token
Positional Encoding
    ↓
Transformer Encoder (4 layers)
    - Multi-Head Self-Attention (8 heads)
    - Feed Forward Network (d_ff=1024)
    ↓
Layer Norm
    ↓
CLS Token Extraction
    ↓ (batch, 256)
Classification Head
    ↓ (batch, 14)
```

#### 4.3.2 핵심 구성 요소

**1. Patch Embedding**
- 스펙트로그램을 16×16 패치로 분할
- Vision Transformer 스타일
- CLS 토큰 추가

**2. Multi-Head Self-Attention**
- 8개 헤드
- 장거리 의존성 학습
- Attention weights 시각화 가능

**3. Transformer Encoder Blocks**
- 4개 레이어
- Residual connection
- Layer Normalization

**파라미터**: 약 1.2M개 (구성에 따라 상이)

### 4.4 Ensemble Classifier

#### 4.4.1 앙상블 방법

여러 모델의 예측을 결합하여 더 강건한 분류

| 방법 | 설명 | 구현 상태 |
|-----|------|----------|
| **Voting** | 각 모델의 예측을 투표 | ✅ |
| **Averaging** | 확률을 평균 | ✅ |
| **Weighted** | 가중 평균 | ✅ |
| **Stacking** | 메타 분류기 사용 | ✅ |

#### 4.4.2 구현 예시

```python
ensemble = EnsembleClassifier(
    models=[cnn_model, crnn_model],
    num_classes=14,
    ensemble_method="weighted",
    weights=[0.4, 0.6]  # CRNN에 더 높은 가중치
)
```

### 4.5 모델 비교

| 항목 | CNN | CRNN | Attention | Ensemble |
|-----|-----|------|-----------|----------|
| 파라미터 수 | 490K | 2.79M | ~1.2M | - |
| 시간 정보 처리 | Global Pooling | LSTM + Attention | Self-Attention | 결합 |
| 학습 속도 | 빠름 | 느림 | 중간 | 느림 |
| 해석 가능성 | 낮음 | Attention 시각화 | Attention 시각화 | 중간 |
| 적합한 경우 | 단순 패턴 | 시간적 변화 중요 | 장거리 의존성 | 최고 성능 추구 |

---

## 5. 학습 설정 및 기법

### 5.1 하이퍼파라미터

```python
# 학습 설정
EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
DROPOUT = 0.3
```

### 5.2 클래스 가중치

클래스 불균형 보정을 위해 역빈도 가중치 적용

```python
# 클래스 가중치 계산
class_counts = [label_counts.get(i, 1) for i in range(NUM_CLASSES)]
class_weights = 1.0 / torch.FloatTensor(class_counts)
class_weights = class_weights / class_weights.sum() * NUM_CLASSES

# 손실 함수
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### 5.3 옵티마이저 & 스케줄러

```python
# AdamW 옵티마이저
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01
)

# Cosine Annealing 스케줄러
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)
```

### 5.4 학습 기법

| 기법 | 설명 | 설정 |
|-----|------|------|
| **Early Stopping** | 과적합 방지 | patience=10 |
| **Mixed Precision** | GPU에서 AMP 사용 (메모리 절약) | AMP enabled |
| **Gradient Clipping** | 그래디언트 폭발 방지 | max_norm=1.0 |
| **SpecAugment** | 학습 중 실시간 적용 | 2 freq masks, 2 time masks |

### 5.5 학습 환경

- **Device**: CPU (GPU 미사용)
- **학습 시간**: Epoch당 약 4분 (CRNN 기준)
- **총 학습 시간**: 약 2시간 (30 epochs)

---

## 6. 실험 결과 및 성능 비교

### 6.1 기본 모델 성능

#### 6.1.1 CNN 학습 결과

```
CNN 학습 결과:
  • Best Val Loss: 2.3856
  • Best Val Acc: 14.65%
  • Test Accuracy: 13.50%
```

**분석**: CNN은 단순한 Global Pooling만으로는 시간적 패턴을 학습하기 어려움

#### 6.1.2 CRNN 학습 결과

```
CRNN 학습 결과:
  • Best Val Loss: 0.8715
  • Best Val Acc: 68.12%
  • Test Accuracy: 63.50%
```

**분석**: LSTM과 Attention으로 시간적 패턴을 잘 학습함

### 6.2 테스트 성능 비교

| 모델 | Test Accuracy | Best Val Acc | 개선도 |
|-----|---------------|--------------|--------|
| CNN | 13.50% | 14.65% | Baseline |
| CRNN | **63.50%** | **68.12%** | **+50.00%p** |

### 6.3 CRNN 상세 Classification Report

```
                                        precision  recall  f1-score  support

braking state/normal_brakes                  0.83    0.87      0.85        46
braking state/worn_out_brakes                1.00    0.71      0.83        45
idle state/combined/* (avg)                  0.38    0.30      0.33      ~245
idle state/low_oil                           0.48    0.62      0.54        48
idle state/normal_engine_idle                0.73    0.97      0.84        39
idle state/power_steering                    0.71    0.71      0.71        58
idle state/serpentine_belt                   0.49    0.74      0.59        53
startup state/bad_ignition                   0.87    0.83      0.85        47
startup state/dead_battery                   0.75    0.70      0.72        43
startup state/normal_engine_startup          0.62    0.59      0.60        46

                              accuracy                        0.64      622
                             macro avg       0.64    0.63      0.63      622
                          weighted avg       0.64    0.64      0.64      622
```

### 6.4 스펙트로그램 기반 CNN 실험

#### 6.4.1 다양한 방법 비교

| 방법 | 테스트 정확도 (%) | 입력 피처 | 합성곱 계층 수 |
|-----|------------------|---------|--------------|
| 방법 1: 로그 멜 + 미분값 | 71.06% | 로그 멜 + 미분값 (2채널) | 2 |
| 방법 2: 로그 멜 + 데이터 증강 | **77.41%** | 로그 멜 스펙트로그램 (1채널) | 3 |
| 방법 3-1: 스펙트로그램 | 74.59% | 스펙트로그램 (1채널) | 3 |
| 방법 3-2: MFCC | 73.88% | MFCC (1채널) | 3 |
| 방법 3-3: Chroma/CRP | 77.18% | Chroma/CRP (1채널) | 3 |

**최고 성능**: 방법 2 (로그 멜 + 데이터 증강) - **77.41%**

### 6.5 클래스별 성능 분석

#### 6.5.1 잘 분류되는 클래스
- **normal_engine_idle**: F1=0.84 (recall=0.97) - 가장 많은 학습 데이터
- **normal_brakes**: F1=0.85 - 명확한 패턴
- **bad_ignition**: F1=0.85 - 특징적인 소리

#### 6.5.2 어려운 클래스
- **combined/* (복합 이상)**: F1=0.33 - 여러 문제가 동시 발생하여 구분 어려움
- **low_oil**: F1=0.54 - 다른 idle 문제와 유사
- **serpentine_belt**: F1=0.59 - 다른 문제와 혼동

### 6.6 Attention 분석 결과

CRNN 모델의 Attention weights를 분석한 결과:

- **정상 소리**: 전체적으로 고르게 attention 분포
- **비정상 소리**: 특정 시간대에 attention이 집중 (이상 신호 구간)
- **결론**: 모델이 실제로 이상이 발생하는 시간 구간을 잘 파악함

---

## 7. 스펙트로그램 패턴 분석

### 7.1 분석 항목

07_Spectrogram_Pattern_Analysis.ipynb에서 수행한 분석:

1. **기본 통계 분석**: max, min, mean, std, median, quartiles
2. **시간 축 패턴 분석**: 각 주파수 밴드의 시간적 변화
3. **주파수 축 패턴 분석**: 각 시간 프레임의 주파수 분포
4. **에너지 분포 분석**: 전체 에너지 분포 및 집중도
5. **주파수 밴드별 통계 비교**: 상태별 주파수 특성
6. **시간 구간별 통계 분석**: 초기/중기/후기 패턴 변화
7. **스펙트로그램 변화율 분석**: Gradient 기반 변화 탐지
8. **PCA 시각화 및 군집 분석**: 차원 축소를 통한 패턴 발견

### 7.2 주요 발견 사항

#### 7.2.1 상태별 특성

**Braking State (브레이크 상태)**
- 낮은 주파수 대역(0-2kHz)에서 높은 에너지
- 시간에 따른 에너지 감소 패턴 (브레이크 압력 변화)

**Idle State (공회전 상태)**
- 주기적인 패턴 (엔진 RPM)
- 특정 주파수에서 하모닉 구조
- 이상 시 패턴 왜곡

**Startup State (시동 상태)**
- 초기에 급격한 에너지 증가
- 주파수 대역이 넓게 분산
- 배터리/점화 문제 시 패턴 변화

#### 7.2.2 통계적 특징

- **에너지 중심**: 상태별로 다른 주파수 대역 집중
- **변화율**: 정상은 안정적, 이상은 급격한 변화
- **분산**: 이상 상태에서 더 높은 분산

### 7.3 PCA 및 군집 분석

- 2D PCA 시각화로 상태별 클러스터 확인
- K-means 클러스터링으로 패턴 그룹화
- 상태별 군집 분리도 측정

---

## 8. 앙상블 모델

### 8.1 Waveform + Spectrogram 앙상블

09_Ensemble_Waveform_Spectrogram.ipynb에서 구현

#### 8.1.1 구조

1. **Waveform 기반 모델**: 1D CNN으로 원시 오디오 신호 처리
2. **Mel Spectrogram 기반 모델**: 2D CNN으로 스펙트로그램 처리
3. **앙상블**: 두 모델의 예측을 Vote로 결합

#### 8.1.2 예상 효과

- Waveform: 시간 도메인에서의 세밀한 패턴 포착
- Spectrogram: 주파수 도메인에서의 구조적 패턴 포착
- 앙상블: 두 관점의 정보 결합으로 성능 향상

### 8.2 앙상블 방법 비교

| 방법 | 설명 | 장점 | 단점 |
|-----|------|------|------|
| Hard Voting | 다수결 투표 | 구현 간단 | 확률 정보 손실 |
| Soft Voting | 확률 평균 | 확률 정보 보존 | 모든 모델 동등 취급 |
| Weighted Voting | 가중 평균 | 모델별 중요도 반영 | 가중치 튜닝 필요 |
| Stacking | 메타 분류기 | 최적 결합 학습 | 학습 시간 증가 |

---

## 9. 결론 및 향후 개선 방향

### 9.1 주요 결론

#### 9.1.1 모델 성능

1. **CRNN이 CNN보다 우수한 성능**
   - CNN: 13.50% vs CRNN: 63.50% (Test Accuracy)
   - 시간적 패턴 학습이 차량 사운드 분류에 중요

2. **Attention 메커니즘의 효과**
   - 모델의 결정 과정 해석 가능
   - 이상 소리가 발생하는 시간 구간 파악
   - 정상 vs 비정상 attention 패턴 차이 확인

3. **데이터 증강의 효과**
   - 원본 949개 → 증강 후 2,832개 (combined 제외)
   - 클래스 불균형 완화
   - 다양한 환경 조건 시뮬레이션

4. **스펙트로그램 분석의 중요성**
   - 상태별 명확한 주파수 패턴 존재
   - 통계적 특징으로 상태 구분 가능
   - PCA를 통한 패턴 발견

#### 9.1.2 최고 성능 모델

- **단일 모델**: 스펙트로그램 CNN (방법 2) - **77.41%**
- **시계열 모델**: CRNN - **63.50%**
- **앙상블 모델**: (아직 실험 중)

### 9.2 현재 한계점

1. **GPU 부재로 인한 학습 제한**
   - CPU 학습으로 시간 소요 (epoch당 약 4분)
   - 더 깊은 모델 실험 어려움
   - 배치 크기 제한 (16)

2. **복합 이상 클래스 분류 어려움**
   - combined/* 클래스들의 낮은 정확도 (F1=0.33)
   - 여러 이상이 동시에 발생할 때 구분 어려움
   - 하위 문제별 세분화 필요

3. **데이터 부족**
   - 일부 클래스 샘플 수 적음 (최소 57개)
   - 실제 환경 다양성 부족
   - 다양한 차량 모델/연식 데이터 필요

4. **실시간 추론 최적화 미흡**
   - 모델 크기 최적화 필요
   - 추론 속도 개선 필요

### 9.3 향후 개선 방향

#### 9.3.1 모델 개선

- [ ] **모델 앙상블 최적화**
  - CNN + CRNN + Attention 앙상블
  - 가중치 최적화
  - Stacking 메타 학습기

- [ ] **Transformer 기반 모델 실험**
  - Audio Spectrogram Transformer (AST)
  - Pre-trained 모델 활용 (Transfer Learning)
  - Fine-tuning 전략

- [ ] **경량화 모델 개발**
  - Knowledge Distillation
  - 모델 양자화
  - 모바일 최적화

#### 9.3.2 데이터 개선

- [ ] **데이터 수집 확대**
  - 더 많은 실제 데이터 수집
  - 다양한 차량 모델/연식
  - 다양한 환경 조건 (실내/실외, 날씨 등)

- [ ] **복합 이상 케이스 세분화**
  - 하위 문제별 레이블링
  - 계층적 분류 구조 도입
  - Multi-label 분류 고려

- [ ] **Synthetic Data Generation**
  - GAN 기반 데이터 생성
  - Mixup/CutMix 기법
  - Domain Adaptation

#### 9.3.3 분석 및 해석

- [ ] **더 깊은 패턴 분석**
  - Wavelet 변환 활용
  - Cepstral 분석
  - 시간-주파수 분석 심화

- [ ] **모델 해석성 향상**
  - Grad-CAM 시각화
  - Attention weights 분석
  - Feature importance 분석

#### 9.3.4 서비스화

- [ ] **실시간 추론 API 구현**
  - FastAPI 기반 REST API
  - WebSocket 실시간 스트리밍
  - 배치 처리 지원

- [ ] **모바일 앱 연동**
  - iOS/Android SDK 개발
  - 오프라인 추론 지원
  - 클라우드 동기화

- [ ] **모니터링 및 관리**
  - 모델 성능 모니터링
  - A/B 테스트 지원
  - 자동 재학습 파이프라인

### 9.4 최종 성능 요약

| 모델/방법 | 정확도 | 특장점 | 활용 분야 |
|----------|--------|--------|----------|
| **CNN (Baseline)** | 13.50% | 빠른 학습 | Baseline 비교 |
| **CRNN** | 63.50% | 시간 패턴 학습, 해석 가능 | 시간적 변화 중요 시 |
| **Spectrogram CNN (방법 2)** | **77.41%** | 단순 구조, 빠른 추론 | 프로덕션 배포 |
| **앙상블** | (실험 중) | 최고 성능 추구 | 고정밀도 요구 |

---

## 부록

### A. 코드 구조

```
Sound_Project/
├── app/
│   └── ml/
│       ├── features/
│       │   ├── extractor.py      # 피처 추출
│       │   └── augmentation.py   # 데이터 증강
│       ├── models/
│       │   ├── cnn.py            # CNN 모델
│       │   ├── crnn.py           # CRNN 모델
│       │   ├── attention.py      # Attention 모델
│       │   └── ensemble.py       # 앙상블 모델
│       ├── training/
│       │   ├── trainer.py        # 학습 루프
│       │   └── dataset.py        # 데이터셋 클래스
│       └── inference/
│           └── service.py        # 추론 서비스
├── notebooks/
│   ├── 01_EDA.ipynb              # 탐색적 데이터 분석
│   ├── 02_Data_Augmentation.ipynb # 데이터 증강
│   ├── 03_Model_Training.ipynb   # 모델 학습
│   ├── 06_Spectrogram_CNN_Classification.ipynb # 스펙트로그램 CNN
│   ├── 07_Spectrogram_Pattern_Analysis.ipynb # 패턴 분석
│   ├── 08_Model_Feature_Visualization.ipynb # 특징 시각화
│   └── 09_Ensemble_Waveform_Spectrogram.ipynb # 앙상블
├── data/
│   ├── braking state/
│   ├── idle state/
│   ├── startup state/
│   └── augmented/                # 증강된 데이터
└── checkpoints/
    ├── cnn_sound_classifier_best_model.pt
    └── crnn_sound_classifier_best_model.pt
```

### B. 주요 하이퍼파라미터 요약

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| EPOCHS | 30 | 총 학습 에포크 |
| BATCH_SIZE | 16 | 배치 크기 |
| LEARNING_RATE | 1e-3 | 학습률 |
| WEIGHT_DECAY | 0.01 | 가중치 감쇠 |
| DROPOUT | 0.3 | 드롭아웃 비율 |
| Early Stopping Patience | 10 | 조기 종료 인내심 |
| Sample Rate | 22050 | 오디오 샘플링 레이트 |
| n_mels | 128 | Mel 밴드 수 |
| n_fft | 2048 | FFT 윈도우 크기 |
| hop_length | 512 | 프레임 간 이동 |

### C. 참고 문헌

- Librosa: 오디오 분석 라이브러리
- PyTorch: 딥러닝 프레임워크
- Mel Spectrogram 기반 오디오 분류 논문들
- Attention mechanism in audio classification
- Ensemble methods for deep learning

---

**문서 작성일**: 2025년 1월  
**최종 업데이트**: 2025년 1월  
**작성자**: Sound Project Team  
**버전**: 1.0
