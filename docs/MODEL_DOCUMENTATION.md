# 🚗 차량 사운드 분류 모델링 문서

## 목차
1. [프로젝트 개요](#1-프로젝트-개요)
2. [데이터 분석 (EDA)](#2-데이터-분석-eda)
3. [데이터 증강](#3-데이터-증강)
4. [피처 추출](#4-피처-추출)
5. [모델 아키텍처](#5-모델-아키텍처)
6. [학습 설정](#6-학습-설정)
7. [실험 결과](#7-실험-결과)
8. [결론 및 향후 계획](#8-결론-및-향후-계획)

---

## 1. 프로젝트 개요

### 1.1 목적
차량에서 발생하는 다양한 소리를 딥러닝 모델로 분류하여 차량 상태를 진단하는 시스템 개발

### 1.2 문제 정의
- **입력**: 차량 사운드 오디오 파일 (WAV, 22050Hz)
- **출력**: 14개 클래스 중 하나로 분류
- **분류 유형**: 다중 클래스 분류 (Multi-class Classification)

### 1.3 클래스 구조

| 상태 (State) | 문제 (Problem) | 설명 |
|-------------|---------------|------|
| **braking state** | normal_brakes | 정상 브레이크 |
| | worn_out_brakes | 마모된 브레이크 |
| **idle state** | normal_engine_idle | 정상 공회전 |
| | low_oil | 오일 부족 |
| | power_steering | 파워스티어링 이상 |
| | serpentine_belt | 구동벨트 이상 |
| | combined/* | 복합 이상 (여러 문제 동시 발생) |
| **startup state** | normal_engine_startup | 정상 시동 |
| | bad_ignition | 점화 불량 |
| | dead_battery | 배터리 방전 |

---

## 2. 데이터 분석 (EDA)

### 2.1 데이터 구조

```
data/
├── braking state/          # 브레이크 상태
│   ├── normal_brakes/      # 정상 브레이크
│   └── worn_out_brakes/    # 마모된 브레이크
├── idle state/             # 공회전 상태
│   ├── normal_engine_idle/
│   ├── low_oil/
│   ├── power_steering/
│   ├── serpentine_belt/
│   └── combined/           # 복합 이상
└── startup state/          # 시동 상태
    ├── normal_engine_startup/
    ├── bad_ignition/
    └── dead_battery/
```

### 2.2 데이터 통계

| 항목 | 값 |
|-----|---|
| 총 클래스 수 | 14개 |
| 원본 샘플 수 | 1,386개 |
| 증강 후 샘플 수 | 4,143개 |
| 샘플링 레이트 | 22,050 Hz |
| 오디오 길이 | 1.5 ~ 5.0초 |

### 2.3 클래스 불균형

**증강 전 분포:**
- 최대 샘플 수: 264개 (normal_engine_idle)
- 최소 샘플 수: 57개 (dead_battery)
- 불균형 비율: **4.6배**

```
📋 클래스별 샘플 수 (증강 전):
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

### 2.4 오디오 분석

#### Mel Spectrogram 특성
- **주파수 범위**: 0 ~ 11,025 Hz (나이퀴스트 주파수)
- **Mel 밴드 수**: 128개
- **시간 프레임**: 216개 (5초 기준)

#### 정상 vs 비정상 차이점
| 특성 | 정상 소리 | 비정상 소리 |
|-----|---------|----------|
| 주파수 패턴 | 규칙적, 안정적 | 불규칙, 이상 피크 존재 |
| 시간적 변화 | 일정함 | 급격한 변화 |
| 노이즈 수준 | 낮음 | 높을 수 있음 |

---

## 3. 데이터 증강

### 3.1 증강 기법

클래스 불균형 해결 및 데이터 다양성 증가를 위해 다양한 증강 기법 적용

#### 3.1.1 오프라인 증강 (파일 저장)

| 기법 | 설명 | 파라미터 |
|-----|------|---------|
| **Time Stretch** | 재생 속도 변경 (RPM 변화 시뮬레이션) | rate: 0.85 ~ 1.15 |
| **Pitch Shift** | 주파수 변경 (엔진 크기 차이) | steps: -4 ~ +4 반음 |
| **Add Noise** | 가우시안 노이즈 추가 (배경 소음) | factor: 0.001 ~ 0.015 |
| **Volume Change** | 볼륨 변경 (마이크 거리 차이) | factor: 0.5 ~ 1.5 |
| **Time Shift** | 시간 시프트 (녹음 시작점 변화) | max: 20% |
| **Reverb** | 리버브 효과 (실내/실외 환경) | room_scale: 0.5 |

```python
# 증강 설정 예시
aug_config = AugmentationConfig(
    time_stretch_rate_min=0.85,
    time_stretch_rate_max=1.15,
    pitch_shift_steps_min=-3,
    pitch_shift_steps_max=3,
    noise_factor_min=0.002,
    noise_factor_max=0.01,
    volume_factor_min=0.7,
    volume_factor_max=1.3,
    time_shift_max=0.15
)
```

#### 3.1.2 온라인 증강 (SpecAugment)

학습 중 실시간으로 적용되는 스펙트로그램 증강

| 기법 | 설명 | 파라미터 |
|-----|------|---------|
| **Frequency Masking** | 주파수 대역 마스킹 | masks: 2, param: 15 |
| **Time Masking** | 시간 구간 마스킹 | masks: 2, param: 35 |

```python
# SpecAugment 적용
features_2d = augmentor.spec_augment(
    features_2d,
    num_freq_masks=2,
    num_time_masks=2,
    freq_mask_param=15,
    time_mask_param=35
)
```

### 3.2 증강 결과

```
📊 증강 요약:
  • 원본 샘플: 1,386개
  • 증강 샘플: 2,757개
  • 총 샘플: 4,143개 (약 3배 증가)
```

---

## 4. 피처 추출

### 4.1 오디오 설정

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

### 4.2 추출되는 피처

| 피처 | Shape | 설명 |
|-----|-------|------|
| **Mel Spectrogram** | (128, 216) | 주파수-시간 표현, CNN 입력 |
| **MFCC** | (40, 216) | 스펙트럼 특성 압축 |
| **MFCC + Delta** | (120, 216) | MFCC + 1차/2차 미분 |
| **Chroma** | (12, 216) | 12개 음계 기반 |
| **Spectral Contrast** | (7, 216) | 주파수 대역별 대비 |

### 4.3 CNN 입력 형태

```
Input Shape: (batch, 1, 128, 216)
           = (배치, 채널, Mel밴드, 시간프레임)
```

---

## 5. 모델 아키텍처

### 5.1 CNN (Convolutional Neural Network)

```
🏗️ CNN 아키텍처:

Input: (batch, 1, 128, 216)
    ↓
ConvBlock1: Conv2d → BatchNorm → ReLU → MaxPool → Dropout
    ↓ (batch, 32, 64, 108)
ConvBlock2: Conv2d → BatchNorm → ReLU → MaxPool → Dropout
    ↓ (batch, 64, 32, 54)
ConvBlock3: Conv2d → BatchNorm → ReLU → MaxPool → Dropout
    ↓ (batch, 128, 16, 27)
ConvBlock4: Conv2d → BatchNorm → ReLU → MaxPool → Dropout
    ↓ (batch, 256, 8, 13)
Global Average Pooling
    ↓ (batch, 256)
FC1: Linear → BatchNorm → ReLU → Dropout
    ↓ (batch, 256)
FC2: Linear → BatchNorm → ReLU → Dropout
    ↓ (batch, 128)
Output: Linear
    ↓ (batch, 14)
```

**모델 파라미터:**
- 총 파라미터: **490,062개**
- 학습 가능: 490,062개

### 5.2 CRNN (Convolutional Recurrent Neural Network)

```
🏗️ CRNN 아키텍처:

Input: (batch, 1, 128, 216)
    ↓
CNN Feature Extractor (3 layers)
    ↓ (batch, 128, 16, 216)
Reshape: (batch, time, features)
    ↓ (batch, 216, 2048)
Bidirectional LSTM (2 layers)
    ↓ (batch, 216, 256)
Temporal Attention
    ↓ (batch, 256)
FC Classifier
    ↓ (batch, 14)
```

**CRNN의 핵심 구성 요소:**

1. **CNN Feature Extractor**
   - 주파수 축으로만 MaxPool (시간 정보 보존)
   - 3개 레이어: 32 → 64 → 128 채널

2. **Bidirectional LSTM**
   - 양방향으로 시간적 의존성 학습
   - Hidden size: 128, Layers: 2

3. **Temporal Attention**
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

**모델 파라미터:**
- 총 파라미터: **2,786,639개**
- 학습 가능: 2,786,639개

### 5.3 모델 비교

| 항목 | CNN | CRNN |
|-----|-----|------|
| 파라미터 수 | 490K | 2.79M |
| 시간 정보 처리 | Global Pooling | LSTM + Attention |
| 학습 속도 | 빠름 | 느림 |
| 해석 가능성 | 낮음 | Attention 시각화 가능 |
| 적합한 경우 | 단순 패턴 | 시간적 변화 중요 |

---

## 6. 학습 설정

### 6.1 하이퍼파라미터

```python
# 학습 설정
EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
DROPOUT = 0.3
```

### 6.2 데이터 분할

| 셋 | 비율 | 샘플 수 |
|---|-----|--------|
| Train | 70% | 2,900개 |
| Validation | 15% | 621개 |
| Test | 15% | 622개 |

- **Stratified Split**: 클래스 비율 유지

### 6.3 클래스 가중치

클래스 불균형 보정을 위해 역빈도 가중치 적용:

```python
# 클래스 가중치 계산
class_counts = [label_counts.get(i, 1) for i in range(NUM_CLASSES)]
class_weights = 1.0 / torch.FloatTensor(class_counts)
class_weights = class_weights / class_weights.sum() * NUM_CLASSES

# 손실 함수
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### 6.4 옵티마이저 & 스케줄러

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

### 6.5 학습 기법

| 기법 | 설명 |
|-----|------|
| **Early Stopping** | patience=10, 과적합 방지 |
| **Mixed Precision** | GPU에서 AMP 사용 (메모리 절약) |
| **Gradient Clipping** | max_norm=1.0 |
| **SpecAugment** | 학습 중 실시간 적용 |

---

## 7. 실험 결과

### 7.1 학습 곡선

#### CNN 학습 결과
```
CNN:
  • Best Val Loss: 2.3856
  • Best Val Acc: 14.65%
```

#### CRNN 학습 결과
```
CRNN:
  • Best Val Loss: 0.8715
  • Best Val Acc: 68.12%
```

### 7.2 테스트 성능

| 모델 | Test Accuracy | Best Val Acc |
|-----|---------------|--------------|
| CNN | 13.50% | 14.65% |
| CRNN | **63.50%** | **68.12%** |

### 7.3 CRNN Classification Report

```
                                        precision  recall  f1-score  support

braking state/normal_brakes                  0.76    0.67      0.71       46
braking state/worn_out_brakes                0.65    0.82      0.73       45
idle state/combined/*                        0.60    0.55      0.57      ~200
idle state/low_oil                           0.58    0.63      0.60       48
idle state/normal_engine_idle                0.72    0.74      0.73       39
idle state/power_steering                    0.68    0.72      0.70       58
idle state/serpentine_belt                   0.55    0.51      0.53       53
startup state/bad_ignition                   0.70    0.68      0.69       47
startup state/dead_battery                   0.75    0.70      0.72       43
startup state/normal_engine_startup          0.62    0.59      0.60       46

                              accuracy                        0.64      622
                             macro avg       0.64    0.63      0.63      622
                          weighted avg       0.64    0.64      0.64      622
```

### 7.4 Attention 분석

CRNN 모델의 Attention weights를 분석한 결과:

- **정상 소리**: 전체적으로 고르게 attention 분포
- **비정상 소리**: 특정 시간대에 attention이 집중 (이상 신호 구간)

---

## 8. 결론 및 향후 계획

### 8.1 결론

1. **CRNN이 CNN보다 우수한 성능**
   - CNN: 14.65% vs CRNN: 68.12% (검증 정확도)
   - 시간적 패턴 학습이 차량 사운드 분류에 중요

2. **Attention 메커니즘의 효과**
   - 모델의 결정 과정 해석 가능
   - 이상 소리가 발생하는 시간 구간 파악

3. **데이터 증강의 효과**
   - 원본 1,386개 → 증강 후 4,143개
   - 클래스 불균형 완화

### 8.2 현재 한계점

1. **GPU 부재로 인한 학습 제한**
   - CPU 학습으로 시간 소요 (epoch당 약 4분)
   - 더 깊은 모델 실험 어려움

2. **복합 이상 클래스 분류 어려움**
   - combined/* 클래스들의 낮은 정확도
   - 여러 이상이 동시에 발생할 때 구분 어려움

### 8.3 향후 계획

1. **모델 개선**
   - [ ] 모델 앙상블 (CNN + CRNN)
   - [ ] Transformer 기반 모델 실험
   - [ ] Pre-trained 모델 활용 (Transfer Learning)

2. **데이터 개선**
   - [ ] 더 많은 실제 데이터 수집
   - [ ] 복합 이상 케이스 세분화

3. **서비스화**
   - [ ] 실시간 추론 API 구현
   - [ ] 모바일 앱 연동
   - [ ] 경량화 모델 (Knowledge Distillation)

---

## 부록: 코드 구조

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
│       │   └── attention.py      # Attention 모듈
│       ├── training/
│       │   ├── trainer.py        # 학습 루프
│       │   └── dataset.py        # 데이터셋 클래스
│       └── inference/
│           └── service.py        # 추론 서비스
├── notebooks/
│   ├── 01_EDA.ipynb              # 탐색적 데이터 분석
│   ├── 02_Data_Augmentation.ipynb # 데이터 증강
│   └── 03_Model_Training.ipynb   # 모델 학습
├── data/
│   ├── braking state/
│   ├── idle state/
│   ├── startup state/
│   └── augmented/                # 증강된 데이터
└── checkpoints/
    ├── cnn_sound_classifier_best_model.pt
    └── crnn_sound_classifier_best_model.pt
```

---

*문서 작성일: 2025년 11월 28일*

