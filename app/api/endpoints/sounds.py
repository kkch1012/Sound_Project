from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.db.session import get_db
from app.services.s3_service import S3Service
from app.schemas.sound import SoundCreate, SoundResponse
from app.crud import sound as sound_crud

router = APIRouter()


@router.post("/upload", response_model=SoundResponse)
async def upload_sound(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """사운드 파일을 S3에 업로드하고 메타데이터를 DB에 저장"""
    # 파일 확장자 검증
    allowed_extensions = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}
    file_ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(allowed_extensions)}"
        )
    
    # S3에 업로드
    s3_service = S3Service()
    s3_url = await s3_service.upload_file(file)
    
    # DB에 메타데이터 저장
    sound_data = SoundCreate(
        filename=file.filename,
        s3_url=s3_url,
        content_type=file.content_type,
        file_size=file.size,
    )
    sound = await sound_crud.create_sound(db, sound_data)
    
    return sound


@router.get("/", response_model=List[SoundResponse])
async def get_sounds(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    """저장된 모든 사운드 목록 조회"""
    sounds = await sound_crud.get_sounds(db, skip=skip, limit=limit)
    return sounds


@router.get("/{sound_id}", response_model=SoundResponse)
async def get_sound(
    sound_id: int,
    db: AsyncSession = Depends(get_db),
):
    """특정 사운드 조회"""
    sound = await sound_crud.get_sound(db, sound_id)
    if not sound:
        raise HTTPException(status_code=404, detail="사운드를 찾을 수 없습니다")
    return sound


@router.delete("/{sound_id}")
async def delete_sound(
    sound_id: int,
    db: AsyncSession = Depends(get_db),
):
    """사운드 삭제 (S3 파일 및 DB 레코드)"""
    sound = await sound_crud.get_sound(db, sound_id)
    if not sound:
        raise HTTPException(status_code=404, detail="사운드를 찾을 수 없습니다")
    
    # S3에서 파일 삭제
    s3_service = S3Service()
    await s3_service.delete_file(sound.s3_url)
    
    # DB에서 레코드 삭제
    await sound_crud.delete_sound(db, sound_id)
    
    return {"message": "사운드가 삭제되었습니다"}

