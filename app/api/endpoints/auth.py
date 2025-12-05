from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import timedelta

from app.db.session import get_db
from app.schemas.user import UserCreate, UserResponse, Token
from app.crud import user as user_crud
from app.core.security import create_access_token
from app.core.config import settings
from app.core.deps import get_current_active_user
from app.models.user import User

router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    회원가입
    
    - **email**: 사용자 이메일 (고유해야 함)
    - **username**: 사용자명 (고유해야 함)
    - **password**: 비밀번호
    """
    try:
        user = await user_crud.create_user(db, user_data)
        return user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    """
    로그인 (OAuth2 호환)
    
    **주의**: username 필드에 **이메일 주소**를 입력하세요.
    
    - **username**: 이메일 주소 (예: user@example.com)
    - **password**: 비밀번호
    
    성공 시 JWT 액세스 토큰을 반환합니다.
    Swagger UI 우측 상단의 "Authorize" 버튼을 클릭하여 토큰을 입력할 수 있습니다.
    """
    user = await user_crud.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="이메일 또는 비밀번호가 올바르지 않습니다",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="비활성화된 사용자입니다"
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_active_user),
):
    """
    로그아웃
    
    현재 로그인한 사용자를 로그아웃합니다.
    JWT는 stateless이므로 클라이언트에서 토큰을 삭제하면 됩니다.
    """
    # JWT는 stateless이므로 클라이언트에서 토큰을 삭제하면 됩니다.
    # 필요하다면 토큰 블랙리스트를 구현할 수 있습니다.
    return {"message": "로그아웃되었습니다"}


@router.get("/me", response_model=UserResponse)
async def read_users_me(
    current_user: User = Depends(get_current_active_user),
):
    """
    현재 로그인한 사용자 정보 조회
    
    인증이 필요한 엔드포인트입니다.
    먼저 로그인하여 토큰을 받은 후, Swagger UI 우측 상단의 "Authorize" 버튼을 클릭하여 토큰을 입력하세요.
    """
    return current_user

