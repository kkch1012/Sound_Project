from fastapi import APIRouter

router = APIRouter()


@router.get("", summary="서버 상태 확인", description="API 서버의 상태를 확인합니다.")
async def health_check():
    """서버 상태 확인"""
    return {"status": "ok"}

