# Sound Project 🚗🔊

**차량 사운드 기반 진단 시스템** - 딥러닝을 활용한 차량 상태 분류 및 문제 진단

FastAPI 기반의 API 서버로, 차량 소리를 분석하여 브레이크, 엔진 공회전, 시동 상태의 이상 여부를 진단합니다.

## 주요 기능

- 🎵 **사운드 분석**: Mel-Spectrogram, MFCC 등 다양한 오디오 피처 추출
- 🧠 **딥러닝 모델**: CNN, CRNN, Transformer 기반 분류 모델
- 🔧 **차량 진단**: 브레이크 마모, 오일 부족, 배터리 방전 등 감지
- 📊 **API 서비스**: RESTful API를 통한 실시간 진단
- ☁️ **클라우드 저장**: AWS S3를 통한 오디오 파일 관리

## 기술 스택

- **Backend**: FastAPI (Python 3.11+)
- **ML/DL**: PyTorch, Librosa, Scikit-learn
- **Database**: PostgreSQL 15
- **Cache**: Redis 7
- **Storage**: AWS S3
- **Authentication**: JWT (OAuth2)
- **Container**: Docker & Docker Compose

## 프로젝트 구조

```
Sound_Project/
├── app/
│   ├── api/
│   │   ├── endpoints/
│   │   │   ├── health.py
│   │   │   ├── sounds.py
│   │   │   └── diagnosis.py      # 차량 진단 API
│   │   └── router.py
│   ├── core/
│   │   └── config.py
│   ├── ml/                        # 머신러닝 모듈
│   │   ├── features/              # 피처 엔지니어링
│   │   │   ├── extractor.py       # MFCC, Mel-Spectrogram 등
│   │   │   └── augmentation.py    # 데이터 증강
│   │   ├── models/                # 딥러닝 모델
│   │   │   ├── cnn.py             # CNN 분류기
│   │   │   ├── crnn.py            # CRNN (CNN+LSTM)
│   │   │   ├── attention.py       # Transformer
│   │   │   └── ensemble.py        # 앙상블
│   │   ├── training/              # 학습 파이프라인
│   │   │   ├── dataset.py
│   │   │   └── trainer.py
│   │   └── inference/             # 추론 서비스
│   │       └── service.py
│   ├── crud/
│   ├── db/
│   ├── models/
│   ├── schemas/
│   ├── services/
│   └── main.py
├── scripts/                       # 학습/평가 스크립트
│   ├── train.py
│   └── evaluate.py
├── data/                          # 학습 데이터
│   ├── braking state/
│   ├── idle state/
│   └── startup state/
├── checkpoints/                   # 학습된 모델
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.gpu                 # GPU 학습용
├── requirements.txt
└── README.md
```

## 시작하기

### 1. 환경 변수 설정

**⚠️ 중요: `.env` 파일은 절대 GitHub에 커밋하지 마세요!**

`env.example` 파일을 복사하여 `.env` 파일을 생성하고, 필요한 설정을 입력합니다.

```bash
# Windows
copy env.example .env

# Linux/Mac
cp env.example .env
```

`.env` 파일을 열어 다음 설정을 입력합니다:

```env
# PostgreSQL (Docker Compose 사용 시 기본값 그대로 사용 가능)
POSTGRES_SERVER=db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=sound_project

# Redis (Docker Compose 사용 시 기본값 그대로 사용 가능)
REDIS_HOST=redis
REDIS_PORT=6379

# AWS S3 (필수 - 음성 파일 저장용)
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_REGION=ap-northeast-2
S3_BUCKET_NAME=your_bucket_name

# JWT 인증 (필수 - 프로덕션에서는 반드시 강력한 비밀키 사용!)
SECRET_KEY=your-very-secure-secret-key-here
```

**보안 주의사항:**
- `.env` 파일은 이미 `.gitignore`에 포함되어 있어 GitHub에 업로드되지 않습니다
- 프로덕션 환경에서는 `SECRET_KEY`를 반드시 강력한 랜덤 문자열로 변경하세요
- AWS 자격 증명은 최소 권한 원칙에 따라 필요한 권한만 부여하세요

### 2. Docker로 실행

```bash
# 빌드 및 실행
docker-compose up --build

# 백그라운드 실행
docker-compose up -d --build

# 로그 확인
docker-compose logs -f

# 종료
docker-compose down
```

### 3. API 접속

- **API 서버**: http://localhost:8000
- **API 문서 (Swagger)**: http://localhost:8000/docs
- **API 문서 (ReDoc)**: http://localhost:8000/redoc

## API 엔드포인트

### Health Check
- `GET /api/v1/health` - 서버 상태 확인

### Authentication (인증) 🔐
- `POST /api/v1/auth/register` - 회원가입
- `POST /api/v1/auth/login` - 로그인 (JWT 토큰 발급)
- `POST /api/v1/auth/logout` - 로그아웃
- `GET /api/v1/auth/me` - 현재 사용자 정보 조회

### Sounds (파일 관리)
- `POST /api/v1/sounds/upload` - 사운드 파일 업로드 (S3 저장)
- `GET /api/v1/sounds/` - 사운드 목록 조회
- `GET /api/v1/sounds/{sound_id}` - 특정 사운드 조회
- `DELETE /api/v1/sounds/{sound_id}` - 사운드 삭제 (S3에서도 삭제)

### Diagnosis (차량 진단) 🆕
- `POST /api/v1/diagnosis/analyze` - 단일 파일 진단
- `POST /api/v1/diagnosis/analyze/batch` - 여러 파일 일괄 진단
- `GET /api/v1/diagnosis/model/info` - 모델 정보 조회
- `POST /api/v1/diagnosis/model/load` - 모델 로드/교체
- `GET /api/v1/diagnosis/labels` - 지원 레이블 조회

## AWS S3 설정

1. AWS 콘솔에서 S3 버킷을 생성합니다.
2. IAM 사용자를 생성하고 S3 접근 권한을 부여합니다.
3. Access Key와 Secret Key를 `.env` 파일에 입력합니다.

### 필요한 IAM 정책

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:DeleteObject"
            ],
            "Resource": "arn:aws:s3:::your-bucket-name/*"
        }
    ]
}
```

## 개발 환경 (Docker 없이)

### 1. 가상환경 생성 및 활성화

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Windows)
venv\Scripts\activate

# 가상환경 활성화 (Linux/Mac)
source venv/bin/activate
```

### 2. 의존성 설치

```bash
# 기본 패키지 설치
pip install -r requirements.txt

# Jupyter 노트북 사용 시 (선택사항)
pip install -r requirements_ml.txt
```

### 3. 서버 실행

```bash
uvicorn app.main:app --reload
```

## 모델 학습

### 1. 학습 실행

```bash
# CPU 학습
python scripts/train.py --data_dir data --model_type cnn --epochs 100

# GPU 학습 (Docker)
docker-compose --profile training up trainer

# 다양한 옵션
python scripts/train.py \
    --data_dir data \
    --model_type crnn \
    --epochs 150 \
    --batch_size 64 \
    --lr 0.0005 \
    --scheduler warmup_cosine \
    --use_class_weights \
    --experiment_name vehicle_sound_crnn
```

### 2. 모델 평가

```bash
python scripts/evaluate.py \
    --model_path checkpoints/sound_classifier_best_model.pt \
    --config_path checkpoints/sound_classifier_config.json \
    --data_dir data
```

### 3. 지원 모델

| 모델 | 설명 | 파라미터 |
|------|------|----------|
| CNN | 기본 CNN 분류기 | ~2.5M |
| CRNN | CNN + Bidirectional LSTM | ~3.5M |
| Attention | Audio Spectrogram Transformer | ~4M |

## 데이터 구조

```
data/
├── braking state/           # 브레이크 상태
│   ├── normal_brakes/       # 정상
│   └── worn_out_brakes/     # 마모
├── idle state/              # 공회전 상태
│   ├── normal_engine_idle/  # 정상
│   ├── low_oil/             # 오일 부족
│   ├── power_steering/      # 파워 스티어링 이상
│   ├── serpentine_belt/     # 벨트 이상
│   └── combined/            # 복합 문제
└── startup state/           # 시동 상태
    ├── normal_engine_startup/  # 정상
    ├── bad_ignition/           # 점화 불량
    └── dead_battery/           # 배터리 방전
```

## 진단 결과 예시

```json
{
  "state": "braking state",
  "problem": "worn_out_brakes",
  "confidence": 0.92,
  "severity": "위험",
  "recommendations": [
    "브레이크 패드 점검이 필요합니다.",
    "가능한 빨리 정비소를 방문하세요.",
    "브레이크 디스크 마모 상태도 함께 확인하세요."
  ]
}
```

## 라이선스

MIT License

