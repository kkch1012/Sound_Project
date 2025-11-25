from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl


class Settings(BaseSettings):
    PROJECT_NAME: str = "Sound Project"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # CORS
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []
    
    # PostgreSQL 데이터베이스 설정
    POSTGRES_SERVER: str = "db"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "sound_project"
    POSTGRES_PORT: str = "5432"
    
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # AWS S3 설정
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "ap-northeast-2"
    S3_BUCKET_NAME: str = ""
    
    # ML 모델 설정
    MODEL_PATH: Optional[str] = "checkpoints/sound_classifier_best_model.pt"
    MODEL_TYPE: str = "cnn"  # cnn, crnn, attention
    MODEL_CONFIG_PATH: Optional[str] = "checkpoints/sound_classifier_config.json"
    
    # 오디오 처리 설정
    AUDIO_SAMPLE_RATE: int = 22050
    AUDIO_DURATION: float = 5.0
    SPECTROGRAM_HEIGHT: int = 128
    SPECTROGRAM_WIDTH: int = 216
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
