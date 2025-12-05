from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional

from app.models.user import User
from app.schemas.user import UserCreate
from app.core.security import get_password_hash, verify_password


async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """이메일로 사용자 조회"""
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def get_user_by_username(db: AsyncSession, username: str) -> Optional[User]:
    """사용자명으로 사용자 조회"""
    result = await db.execute(select(User).where(User.username == username))
    return result.scalar_one_or_none()


async def get_user_by_id(db: AsyncSession, user_id: int) -> Optional[User]:
    """ID로 사용자 조회"""
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def create_user(db: AsyncSession, user_data: UserCreate) -> User:
    """새로운 사용자 생성"""
    # 이메일 중복 확인
    existing_user = await get_user_by_email(db, user_data.email)
    if existing_user:
        raise ValueError("이미 등록된 이메일입니다.")
    
    # 사용자명 중복 확인
    existing_username = await get_user_by_username(db, user_data.username)
    if existing_username:
        raise ValueError("이미 사용 중인 사용자명입니다.")
    
    # 비밀번호 해싱
    hashed_password = get_password_hash(user_data.password)
    
    # 사용자 생성
    db_user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed_password,
    )
    db.add(db_user)
    await db.flush()
    await db.refresh(db_user)
    return db_user


async def authenticate_user(db: AsyncSession, email: str, password: str) -> Optional[User]:
    """사용자 인증"""
    user = await get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

