"""
차량 사운드 진단 API 엔드포인트
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
from enum import Enum
import tempfile
from pathlib import Path

from app.ml.inference.service import SoundDiagnosticService, VehicleDiagnosis, DiagnosisResult
from app.core.config import settings


router = APIRouter()

# 전역 진단 서비스 (싱글톤)
_diagnostic_service: Optional[SoundDiagnosticService] = None


def get_diagnostic_service() -> SoundDiagnosticService:
    """진단 서비스 의존성 주입"""
    global _diagnostic_service
    if _diagnostic_service is None:
        _diagnostic_service = SoundDiagnosticService(
            model_path=settings.MODEL_PATH if hasattr(settings, 'MODEL_PATH') else None,
            model_type=settings.MODEL_TYPE if hasattr(settings, 'MODEL_TYPE') else "cnn"
        )
    return _diagnostic_service


# Response 스키마
class DiagnosisResponse(BaseModel):
    """진단 결과 응답"""
    state: str
    problem: str
    confidence: float
    severity: str
    recommendations: List[str]
    all_predictions: Optional[Dict[str, float]] = None


class ModelInfoResponse(BaseModel):
    """모델 정보 응답"""
    status: str
    model_type: Optional[str] = None
    device: Optional[str] = None
    num_classes: Optional[int] = None
    total_parameters: Optional[int] = None


class BatchDiagnosisResponse(BaseModel):
    """배치 진단 결과 응답"""
    results: List[DiagnosisResponse]
    total_count: int
    normal_count: int
    warning_count: int
    critical_count: int


@router.post("/analyze", response_model=DiagnosisResponse)
async def analyze_sound(
    file: UploadFile = File(...),
    return_all_probs: bool = False,
    service: SoundDiagnosticService = Depends(get_diagnostic_service)
):
    """
    차량 사운드 파일 분석
    
    - **file**: WAV, MP3 등의 오디오 파일
    - **return_all_probs**: 모든 클래스의 확률 반환 여부
    
    Returns:
        진단 결과 (상태, 문제, 신뢰도, 심각도, 권장 조치)
    """
    # 파일 확장자 검증
    allowed_extensions = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
    file_ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(allowed_extensions)}"
        )
    
    try:
        # 파일 내용 읽기
        content = await file.read()
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        # 진단 수행
        diagnosis = service.diagnose(tmp_path, return_all_probs=return_all_probs)
        
        # 임시 파일 삭제
        Path(tmp_path).unlink()
        
        return DiagnosisResponse(
            state=diagnosis.state,
            problem=diagnosis.problem,
            confidence=diagnosis.confidence,
            severity=diagnosis.severity.value,
            recommendations=diagnosis.recommendations,
            all_predictions=diagnosis.all_predictions if return_all_probs else None
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"진단 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/analyze/batch", response_model=BatchDiagnosisResponse)
async def analyze_sounds_batch(
    files: List[UploadFile] = File(...),
    service: SoundDiagnosticService = Depends(get_diagnostic_service)
):
    """
    여러 사운드 파일 일괄 분석
    
    - **files**: 오디오 파일 목록
    
    Returns:
        각 파일의 진단 결과 및 요약 통계
    """
    results = []
    normal_count = 0
    warning_count = 0
    critical_count = 0
    
    for file in files:
        try:
            content = await file.read()
            file_ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ".wav"
            
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            diagnosis = service.diagnose(tmp_path)
            Path(tmp_path).unlink()
            
            result = DiagnosisResponse(
                state=diagnosis.state,
                problem=diagnosis.problem,
                confidence=diagnosis.confidence,
                severity=diagnosis.severity.value,
                recommendations=diagnosis.recommendations
            )
            results.append(result)
            
            # 통계 집계
            if diagnosis.severity == DiagnosisResult.NORMAL:
                normal_count += 1
            elif diagnosis.severity == DiagnosisResult.WARNING:
                warning_count += 1
            else:
                critical_count += 1
                
        except Exception as e:
            # 개별 파일 오류는 기록하고 계속 진행
            results.append(DiagnosisResponse(
                state="error",
                problem=str(e),
                confidence=0.0,
                severity="error",
                recommendations=["파일 처리 중 오류가 발생했습니다."]
            ))
    
    return BatchDiagnosisResponse(
        results=results,
        total_count=len(results),
        normal_count=normal_count,
        warning_count=warning_count,
        critical_count=critical_count
    )


@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info(
    service: SoundDiagnosticService = Depends(get_diagnostic_service)
):
    """
    현재 로드된 모델 정보 조회
    """
    info = service.get_model_info()
    return ModelInfoResponse(**info)


@router.post("/model/load")
async def load_model(
    model_path: str,
    model_type: str = "cnn",
    config_path: Optional[str] = None,
    service: SoundDiagnosticService = Depends(get_diagnostic_service)
):
    """
    모델 로드/교체
    
    - **model_path**: 모델 파일 경로 (.pt)
    - **model_type**: 모델 타입 (cnn, crnn, attention)
    - **config_path**: 설정 파일 경로 (선택사항)
    """
    try:
        service.model_type = model_type
        service.load_model(model_path, config_path)
        return {"status": "success", "message": f"모델 로드 완료: {model_path}"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"모델 로드 실패: {str(e)}"
        )


@router.get("/labels")
async def get_labels(
    service: SoundDiagnosticService = Depends(get_diagnostic_service)
):
    """
    지원하는 분류 레이블 목록 조회
    """
    return {
        "labels": list(service.idx_to_label.values()) if service.idx_to_label else [],
        "label_to_idx": service.label_to_idx,
        "severity_map": {
            "normal_brakes": "정상",
            "normal_engine_idle": "정상",
            "normal_engine_startup": "정상",
            "worn_out_brakes": "위험",
            "low_oil": "주의",
            "power_steering": "주의",
            "serpentine_belt": "주의",
            "bad_ignition": "위험",
            "dead_battery": "위험"
        }
    }

