import redis
from typing import Optional, Any
import json
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)


class RedisService:
    """Redis 서비스 클래스 - 캐싱, 세션 관리 등에 사용"""
    
    def __init__(self):
        """Redis 클라이언트 초기화"""
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                db=settings.REDIS_DB,
                decode_responses=settings.REDIS_DECODE_RESPONSES,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
            )
            # 연결 테스트
            if self.ping():
                logger.info(f"Redis 연결 성공: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
            else:
                logger.warning("Redis 연결 실패: ping 실패")
        except Exception as e:
            logger.error(f"Redis 초기화 실패: {e}")
            raise
    
    def ping(self) -> bool:
        """Redis 연결 확인"""
        try:
            return self.redis_client.ping()
        except Exception:
            return False
    
    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """
        키-값 저장
        
        Args:
            key: 저장할 키
            value: 저장할 값 (문자열, 숫자, 딕셔너리 등)
            ex: 만료 시간 (초 단위)
        
        Returns:
            성공 여부
        """
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            return self.redis_client.set(key, value, ex=ex)
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        키로 값 조회
        
        Args:
            key: 조회할 키
        
        Returns:
            저장된 값 또는 None
        """
        try:
            value = self.redis_client.get(key)
            if value is None:
                return None
            
            # JSON 파싱 시도
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """키 삭제"""
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """키 존재 여부 확인"""
        try:
            return bool(self.redis_client.exists(key))
        except Exception:
            return False
    
    def expire(self, key: str, time: int) -> bool:
        """키에 만료 시간 설정"""
        try:
            return self.redis_client.expire(key, time)
        except Exception:
            return False
    
    def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """값 증가 (카운터용)"""
        try:
            return self.redis_client.incrby(key, amount)
        except Exception:
            return None
    
    def decrement(self, key: str, amount: int = 1) -> Optional[int]:
        """값 감소"""
        try:
            return self.redis_client.decrby(key, amount)
        except Exception:
            return None
    
    def set_hash(self, name: str, mapping: dict, ex: Optional[int] = None) -> bool:
        """해시 저장"""
        try:
            result = self.redis_client.hset(name, mapping=mapping)
            if ex:
                self.redis_client.expire(name, ex)
            return bool(result)
        except Exception:
            return False
    
    def get_hash(self, name: str, key: Optional[str] = None) -> Optional[Any]:
        """해시 조회"""
        try:
            if key:
                return self.redis_client.hget(name, key)
            return self.redis_client.hgetall(name)
        except Exception:
            return None
    
    def delete_hash(self, name: str, *keys: str) -> int:
        """해시 필드 삭제"""
        try:
            return self.redis_client.hdel(name, *keys)
        except Exception:
            return 0
    
    def get_client(self) -> redis.Redis:
        """Redis 클라이언트 직접 반환 (고급 사용)"""
        return self.redis_client


# 싱글톤 인스턴스
_redis_service: Optional[RedisService] = None


def get_redis_service() -> RedisService:
    """Redis 서비스 싱글톤 인스턴스 반환"""
    global _redis_service
    if _redis_service is None:
        _redis_service = RedisService()
    return _redis_service

