from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional

from app.models.sound import Sound
from app.schemas.sound import SoundCreate


async def create_sound(db: AsyncSession, sound_data: SoundCreate) -> Sound:
    """새로운 사운드 레코드 생성"""
    db_sound = Sound(**sound_data.model_dump())
    db.add(db_sound)
    await db.flush()
    await db.refresh(db_sound)
    return db_sound


async def get_sound(db: AsyncSession, sound_id: int) -> Optional[Sound]:
    """ID로 사운드 조회"""
    result = await db.execute(select(Sound).where(Sound.id == sound_id))
    return result.scalar_one_or_none()


async def get_sounds(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[Sound]:
    """사운드 목록 조회"""
    result = await db.execute(
        select(Sound).offset(skip).limit(limit).order_by(Sound.created_at.desc())
    )
    return result.scalars().all()


async def delete_sound(db: AsyncSession, sound_id: int) -> bool:
    """사운드 삭제"""
    sound = await get_sound(db, sound_id)
    if sound:
        await db.delete(sound)
        await db.flush()
        return True
    return False

