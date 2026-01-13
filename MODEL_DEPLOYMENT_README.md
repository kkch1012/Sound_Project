# 모델 배포 가이드

다른 컴퓨터에서 학습된 모델들을 사용하기 위한 가이드입니다.

## 📦 모델 종류

이 프로젝트에는 두 가지 모델이 있습니다:

1. **상태 분류 모델** (`inference_state_model.py`) - 3가지 상태 분류 (Braking, Idle, Startup)
2. **컬럼 분류 모델** (`inference_model.py`) - Idle 상태 내 4가지 컬럼 분류 (low_oil, normal_engine_idle, power_steering, serpentine_belt)

---

## 🚗 상태 분류 모델 (State Classification)

### 필요한 파일

1. **`best_ensemble_state_model.pth`** - 모델 가중치 파일 (필수)
2. **`inference_state_model.py`** - 추론 스크립트 (필수)
3. **`importance_mask.npy`** - 중요 영역 마스크 파일 (선택사항)

### 사용 방법

```bash
# 기본 사용
python inference_state_model.py --audio_path "path/to/audio.wav"

# GPU 사용
python inference_state_model.py --audio_path "audio.wav" --device cuda

# 모델 경로 지정
python inference_state_model.py --audio_path "audio.wav" --model_path "checkpoints/best_ensemble_state_model.pth"
```

### 출력 예시

```
📂 오디오 파일 로드: audio.wav
🤖 모델 로드: checkpoints/best_ensemble_state_model.pth

✅ 예측 결과:
  예측된 상태: idle state

📊 각 상태별 확률:
  idle state: 92.45%
  startup state: 5.23%
  braking state: 2.32%
```

### 모델 설정

- **입력 오디오 길이**: 5초
- **Sample Rate**: 22050 Hz
- **Mel Spectrogram**: 128 mel bins, 216 time frames
- **분류 클래스**: 3개 (braking state, idle state, startup state)

---

## 🔧 컬럼 분류 모델 (Column Classification)

### 필요한 파일

1. **`best_ensemble_model.pth`** - 모델 가중치 파일 (필수)
2. **`inference_model.py`** - 추론 스크립트 (필수)
3. **`importance_mask.npy`** - 중요 영역 마스크 파일 (선택사항)

### 사용 방법

```bash
# 기본 사용
python inference_model.py --audio_path "path/to/audio.wav"

# GPU 사용
python inference_model.py --audio_path "audio.wav" --device cuda

# 모델 경로 지정
python inference_model.py --audio_path "audio.wav" --model_path "best_ensemble_model.pth"
```

### 출력 예시

```
📂 오디오 파일 로드: audio.wav
🤖 모델 로드: best_ensemble_model.pth

✅ 예측 결과:
  예측된 클래스: power_steering

📊 각 클래스별 확률:
  power_steering: 85.23%
  normal_engine_idle: 8.45%
  serpentine_belt: 4.12%
  low_oil: 2.20%
```

### 모델 설정

- **입력 오디오 길이**: 2초
- **Sample Rate**: 22050 Hz
- **Waveform**: 1D CNN (시간적 패턴 학습)
- **Mel Spectrogram**: CRNN (CNN + LSTM, 주파수-시간 패턴 학습) 🌟
- **MFCC**: 2D CNN (주파수-시간 패턴 학습)
- **분류 클래스**: 4개 (low_oil, normal_engine_idle, power_steering, serpentine_belt)
- **모델 구조**: 3-경로 앙상블 (Waveform + Mel Spectrogram CRNN + MFCC)

---

## 🔧 환경 설정

### 1. Python 패키지 설치

```bash
pip install torch torchvision torchaudio
pip install librosa numpy scipy
```

### 2. 파일 배치

상태 분류 모델:
```
프로젝트 폴더/
├── checkpoints/
│   └── best_ensemble_state_model.pth
├── inference_state_model.py
└── importance_mask.npy (선택사항)
```

컬럼 분류 모델:
```
프로젝트 폴더/
├── best_ensemble_model.pth
├── inference_model.py
└── importance_mask.npy (선택사항)
```

---

## 📝 Python 코드에서 직접 사용

### 상태 분류 모델

```python
from inference_state_model import predict

predicted_state, probabilities = predict(
    audio_path="path/to/audio.wav",
    model_path="checkpoints/best_ensemble_state_model.pth",
    device="cpu"  # 또는 "cuda"
)

print(f"예측된 상태: {predicted_state}")
print(f"확률: {probabilities}")
```

### 컬럼 분류 모델

```python
from inference_model import predict

predicted_label, probabilities = predict(
    audio_path="path/to/audio.wav",
    model_path="best_ensemble_model.pth",
    device="cpu"  # 또는 "cuda"
)

print(f"예측된 클래스: {predicted_label}")
print(f"확률: {probabilities}")
```

---

## ⚠️ 주의사항

### 공통 주의사항

1. **모델 구조 일치**: 추론 스크립트의 모델 클래스 정의는 학습 시 사용한 것과 **정확히 동일**해야 합니다.

2. **전처리 일치**: 오디오 전처리 방식도 학습 시와 동일해야 합니다.

3. **중요 영역 마스크**: `importance_mask.npy` 파일이 없어도 동작하지만, 있으면 더 정확한 예측이 가능합니다.

### 상태 분류 모델 주의사항

- 오디오는 **5초 길이**로 정규화됩니다 (짧으면 패딩, 길면 자름)
- 레이블 순서: 0=braking state, 1=idle state, 2=startup state

### 컬럼 분류 모델 주의사항

- 오디오는 **2초 길이**로 정규화됩니다 (짧으면 패딩, 길면 자름)
- **Idle 상태의 오디오**만 입력해야 합니다
- 레이블 순서: 0=low_oil, 1=normal_engine_idle, 2=power_steering, 3=serpentine_belt

---

## 🔍 문제 해결

### 모델 로드 오류
- 모델 파일 경로가 올바른지 확인
- 모델 구조가 학습 시와 동일한지 확인
- PyTorch 버전 호환성 확인

### 전처리 오류
- librosa 버전 확인 (권장: 0.10.0 이상)
- 오디오 파일 형식 확인 (WAV, MP3 등 지원)
- 오디오 길이 확인 (너무 짧거나 길면 문제 발생 가능)

### CUDA 오류
- CUDA가 설치되어 있는지 확인
- `--device cpu` 옵션으로 CPU 사용 가능

### 메모리 부족
- 배치 크기를 줄이거나 CPU 사용
- 오디오 길이 확인 (너무 길면 메모리 부족 가능)

---

## 📋 전송할 파일 목록

### 상태 분류 모델
- `checkpoints/best_ensemble_state_model.pth`
- `inference_state_model.py`
- `importance_mask.npy` (선택사항)

### 컬럼 분류 모델
- `best_ensemble_model.pth`
- `inference_model.py`
- `importance_mask.npy` (선택사항)

### 공통 파일
- `MODEL_DEPLOYMENT_README.md` (이 파일)

---

## 💡 사용 예시

### 2단계 분류 파이프라인

```python
from inference_state_model import predict as predict_state
from inference_model import predict as predict_column

# 1단계: 상태 분류
state, state_probs = predict_state(
    "audio.wav",
    "checkpoints/best_ensemble_state_model.pth"
)

# 2단계: Idle 상태인 경우에만 컬럼 분류
if state == "idle state":
    column, column_probs = predict_column(
        "audio.wav",
        "best_ensemble_model.pth"
    )
    print(f"상태: {state}, 컬럼: {column}")
else:
    print(f"상태: {state} (컬럼 분류 불필요)")
```

---

## 📚 추가 정보

- 두 모델 모두 앙상블 구조를 사용합니다
- 상태 분류 모델: Waveform CNN + Mel Spectrogram CNN
- 컬럼 분류 모델: Waveform CNN + Mel Spectrogram CRNN + MFCC CNN 🌟
- 모든 모델은 PyTorch로 구현되었습니다
- 컬럼 분류 모델의 Mel Spectrogram은 CRNN(CNN + LSTM)을 사용하여 시간적 패턴을 더 잘 학습합니다
