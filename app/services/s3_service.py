import boto3
from botocore.exceptions import ClientError
from fastapi import UploadFile, HTTPException
import uuid
from typing import Optional

from app.core.config import settings


class S3Service:
    def __init__(self):
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )
        self.bucket_name = settings.S3_BUCKET_NAME
    
    async def upload_file(self, file: UploadFile) -> str:
        """파일을 S3에 업로드하고 URL 반환"""
        try:
            # 고유한 파일명 생성
            file_extension = file.filename.split(".")[-1] if "." in file.filename else ""
            unique_filename = f"sounds/{uuid.uuid4()}.{file_extension}"
            
            # 파일 내용 읽기
            file_content = await file.read()
            
            # S3에 업로드
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=unique_filename,
                Body=file_content,
                ContentType=file.content_type,
            )
            
            # 파일 포인터 리셋 (필요한 경우를 위해)
            await file.seek(0)
            
            # S3 URL 반환
            s3_url = f"https://{self.bucket_name}.s3.{settings.AWS_REGION}.amazonaws.com/{unique_filename}"
            return s3_url
            
        except ClientError as e:
            raise HTTPException(
                status_code=500,
                detail=f"S3 업로드 실패: {str(e)}"
            )
    
    async def delete_file(self, s3_url: str) -> bool:
        """S3에서 파일 삭제"""
        try:
            # URL에서 키 추출
            key = self._extract_key_from_url(s3_url)
            if not key:
                return False
            
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=key,
            )
            return True
            
        except ClientError as e:
            raise HTTPException(
                status_code=500,
                detail=f"S3 삭제 실패: {str(e)}"
            )
    
    def _extract_key_from_url(self, s3_url: str) -> Optional[str]:
        """S3 URL에서 객체 키 추출"""
        try:
            # https://bucket-name.s3.region.amazonaws.com/key 형식
            parts = s3_url.split(f"{self.bucket_name}.s3.{settings.AWS_REGION}.amazonaws.com/")
            if len(parts) > 1:
                return parts[1]
            return None
        except Exception:
            return None
    
    async def get_presigned_url(self, s3_url: str, expiration: int = 3600) -> str:
        """사전 서명된 URL 생성 (비공개 파일 접근용)"""
        try:
            key = self._extract_key_from_url(s3_url)
            if not key:
                raise HTTPException(status_code=400, detail="유효하지 않은 S3 URL")
            
            presigned_url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": key},
                ExpiresIn=expiration,
            )
            return presigned_url
            
        except ClientError as e:
            raise HTTPException(
                status_code=500,
                detail=f"사전 서명 URL 생성 실패: {str(e)}"
            )

