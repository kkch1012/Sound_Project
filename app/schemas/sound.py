from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class SoundBase(BaseModel):
    filename: str
    s3_url: str
    content_type: Optional[str] = None
    file_size: Optional[int] = None


class SoundCreate(SoundBase):
    pass


class SoundUpdate(BaseModel):
    filename: Optional[str] = None


class SoundResponse(SoundBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

