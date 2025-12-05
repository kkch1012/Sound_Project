from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.api.router import api_router
from app.core.config import settings
from app.db.session import engine
from app.db.base import Base
from app.services.redis_service import get_redis_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    # PostgreSQL 연결 및 테이블 생성
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ PostgreSQL 연결 성공 및 테이블 생성 완료")
    except Exception as e:
        logger.warning(f"⚠️ PostgreSQL 연결 실패: {e}. 데이터베이스 없이 계속 진행합니다.")
        logger.warning("데이터베이스 기능을 사용하려면 PostgreSQL을 실행하고 연결 정보를 확인하세요.")
    
    # Redis 연결 확인
    try:
        redis_service = get_redis_service()
        if redis_service.ping():
            logger.info("✅ Redis 연결 성공")
        else:
            logger.warning("⚠️ Redis 연결 실패: ping 실패")
    except Exception as e:
        logger.warning(f"⚠️ Redis 연결 실패: {e}. Redis 없이 계속 진행합니다.")
        logger.warning("캐싱 기능을 사용하려면 Redis를 실행하고 연결 정보를 확인하세요.")
    
    yield
    
    # Shutdown
    try:
        await engine.dispose()
    except Exception:
        pass


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Sound Project API",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS] if settings.BACKEND_CORS_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    return {"message": "Welcome to Sound Project API"}


