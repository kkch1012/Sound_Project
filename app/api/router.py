from fastapi import APIRouter

from app.api.endpoints import sounds, health, diagnosis

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(sounds.router, prefix="/sounds", tags=["sounds"])
api_router.include_router(diagnosis.router, prefix="/diagnosis", tags=["diagnosis"])

