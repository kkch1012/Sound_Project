"""
추론 서비스 모듈
학습된 모델을 사용한 차량 사운드 진단
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import tempfile
from dataclasses import dataclass
from enum import Enum

from app.ml.features.extractor import AudioFeatureExtractor, AudioConfig
from app.ml.models import SoundClassifierCNN, SoundClassifierCRNN, SoundClassifierAttention


class DiagnosisResult(str, Enum):
    """진단 결과 등급"""
    NORMAL = "정상"
    WARNING = "주의"
    CRITICAL = "위험"


@dataclass
class VehicleDiagnosis:
    """차량 진단 결과"""
    state: str  # 차량 상태 (braking, idle, startup)
    problem: str  # 감지된 문제
    confidence: float  # 신뢰도
    severity: DiagnosisResult  # 심각도
    recommendations: List[str]  # 권장 조치
    all_predictions: Dict[str, float]  # 전체 예측 확률


class SoundDiagnosticService:
    """
    차량 사운드 진단 서비스
    
    Features:
    - 여러 모델 앙상블 지원
    - 신뢰도 기반 진단
    - 상세 진단 보고서 생성
    """
    
    # 문제별 권장 조치 매핑
    RECOMMENDATIONS = {
        "worn_out_brakes": [
            "브레이크 패드 점검이 필요합니다.",
            "가능한 빨리 정비소를 방문하세요.",
            "브레이크 디스크 마모 상태도 함께 확인하세요."
        ],
        "low_oil": [
            "엔진 오일 레벨을 확인하세요.",
            "오일 누유 여부를 점검하세요.",
            "정기 오일 교환 주기를 확인하세요."
        ],
        "power_steering": [
            "파워 스티어링 오일 레벨을 확인하세요.",
            "스티어링 시스템 점검이 필요합니다.",
            "이상 소음이 지속되면 정비소 방문을 권장합니다."
        ],
        "serpentine_belt": [
            "서펜타인 벨트 상태를 점검하세요.",
            "벨트 장력과 마모 상태를 확인하세요.",
            "교체 시기가 지났다면 즉시 교체하세요."
        ],
        "bad_ignition": [
            "점화 플러그 상태를 점검하세요.",
            "점화 코일 및 배선을 확인하세요.",
            "연료 시스템 점검도 권장됩니다."
        ],
        "dead_battery": [
            "배터리 전압을 측정하세요.",
            "배터리 단자 부식 여부를 확인하세요.",
            "배터리 수명(보통 3-5년)을 확인하세요."
        ],
        "normal": [
            "현재 차량 상태가 정상입니다.",
            "정기적인 점검을 계속 유지하세요."
        ]
    }
    
    # 문제별 심각도 매핑
    SEVERITY_MAP = {
        "normal_brakes": DiagnosisResult.NORMAL,
        "normal_engine_idle": DiagnosisResult.NORMAL,
        "normal_engine_startup": DiagnosisResult.NORMAL,
        "worn_out_brakes": DiagnosisResult.CRITICAL,
        "low_oil": DiagnosisResult.WARNING,
        "power_steering": DiagnosisResult.WARNING,
        "serpentine_belt": DiagnosisResult.WARNING,
        "bad_ignition": DiagnosisResult.CRITICAL,
        "dead_battery": DiagnosisResult.CRITICAL,
        "combined": DiagnosisResult.CRITICAL
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "cnn",
        config_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.model_type = model_type
        self.model = None
        self.label_to_idx = {}
        self.idx_to_label = {}
        
        # 피처 추출기
        self.feature_extractor = AudioFeatureExtractor(AudioConfig())
        
        # 모델 및 설정 로드
        if model_path and Path(model_path).exists():
            self.load_model(model_path, config_path)
    
    def load_model(
        self, 
        model_path: str, 
        config_path: Optional[str] = None
    ):
        """학습된 모델 로드"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 설정 로드
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                config = json.load(f)
            self.label_to_idx = config.get("label_to_idx", {})
            self.idx_to_label = config.get("idx_to_label", {})
            num_classes = config.get("num_classes", len(self.label_to_idx))
        else:
            # 체크포인트에서 클래스 수 추론
            num_classes = checkpoint.get("num_classes", 10)
        
        # 모델 생성
        self.model = self._create_model(num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Number of classes: {num_classes}")
    
    def _create_model(self, num_classes: int):
        """모델 타입에 따른 모델 생성"""
        if self.model_type == "cnn":
            return SoundClassifierCNN(num_classes=num_classes)
        elif self.model_type == "crnn":
            return SoundClassifierCRNN(num_classes=num_classes)
        elif self.model_type == "attention":
            return SoundClassifierAttention(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    @torch.no_grad()
    def diagnose(
        self, 
        audio_input: Union[str, bytes, np.ndarray],
        return_all_probs: bool = False
    ) -> VehicleDiagnosis:
        """
        차량 사운드 진단
        
        Args:
            audio_input: 오디오 파일 경로, 바이트, 또는 numpy 배열
            return_all_probs: 모든 클래스의 확률 반환 여부
        
        Returns:
            VehicleDiagnosis 객체
        """
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")
        
        # 피처 추출
        features = self._extract_features(audio_input)
        features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        
        # 예측
        outputs = self.model(features_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        
        # 최고 확률 클래스
        max_prob, predicted_idx = probabilities.max(0)
        predicted_idx = predicted_idx.item()
        confidence = max_prob.item()
        
        # 레이블 매핑
        if self.idx_to_label:
            predicted_label = self.idx_to_label.get(str(predicted_idx), f"class_{predicted_idx}")
        else:
            predicted_label = f"class_{predicted_idx}"
        
        # 상태와 문제 파싱
        state, problem = self._parse_label(predicted_label)
        
        # 심각도 결정
        severity = self._determine_severity(problem, confidence)
        
        # 권장 조치
        recommendations = self._get_recommendations(problem)
        
        # 전체 확률 (옵션)
        all_predictions = {}
        if return_all_probs:
            for idx, prob in enumerate(probabilities.cpu().numpy()):
                label = self.idx_to_label.get(str(idx), f"class_{idx}")
                all_predictions[label] = float(prob)
        
        return VehicleDiagnosis(
            state=state,
            problem=problem,
            confidence=confidence,
            severity=severity,
            recommendations=recommendations,
            all_predictions=all_predictions
        )
    
    def _extract_features(self, audio_input: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """오디오 입력에서 피처 추출"""
        if isinstance(audio_input, str):
            # 파일 경로
            return self.feature_extractor.extract_for_cnn(audio_input)
        elif isinstance(audio_input, bytes):
            # 바이트 데이터 -> 임시 파일로 처리
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(audio_input)
                    temp_path = f.name
                features = self.feature_extractor.extract_for_cnn(temp_path)
                return features
            finally:
                if temp_path and Path(temp_path).exists():
                    Path(temp_path).unlink()
        elif isinstance(audio_input, np.ndarray):
            # 이미 로드된 오디오
            # 피처 추출기 직접 사용
            mel_spec = self.feature_extractor.extract_mel_spectrogram(
                audio_input, 
                self.feature_extractor.config.sample_rate
            )
            mel_spec = self.feature_extractor._resize_spectrogram(mel_spec, (128, 216))
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
            return mel_spec[np.newaxis, :, :]
        else:
            raise ValueError(f"Unsupported input type: {type(audio_input)}")
    
    def _parse_label(self, label: str) -> Tuple[str, str]:
        """레이블에서 상태와 문제 파싱"""
        parts = label.split("/")
        if len(parts) >= 2:
            state = parts[0]
            problem = parts[1] if len(parts) == 2 else "/".join(parts[1:])
        else:
            state = "unknown"
            problem = label
        return state, problem
    
    def _determine_severity(self, problem: str, confidence: float) -> DiagnosisResult:
        """심각도 결정"""
        # 기본 심각도
        base_severity = self.SEVERITY_MAP.get(problem.split("/")[0], DiagnosisResult.WARNING)
        
        # 신뢰도가 낮으면 한 단계 낮춤
        if confidence < 0.5:
            if base_severity == DiagnosisResult.CRITICAL:
                return DiagnosisResult.WARNING
            elif base_severity == DiagnosisResult.WARNING:
                return DiagnosisResult.NORMAL
        
        return base_severity
    
    def _get_recommendations(self, problem: str) -> List[str]:
        """권장 조치 반환"""
        # 문제 키워드 매칭
        problem_lower = problem.lower()
        
        for key, recommendations in self.RECOMMENDATIONS.items():
            if key in problem_lower:
                return recommendations
        
        # 기본 권장 조치
        return ["정기 점검을 권장합니다.", "이상 증상이 지속되면 정비소를 방문하세요."]
    
    @torch.no_grad()
    def diagnose_batch(
        self, 
        audio_inputs: List[Union[str, bytes, np.ndarray]]
    ) -> List[VehicleDiagnosis]:
        """배치 진단"""
        results = []
        for audio_input in audio_inputs:
            result = self.diagnose(audio_input)
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        if self.model is None:
            return {"status": "not_loaded"}
        
        num_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "status": "loaded",
            "model_type": self.model_type,
            "device": self.device,
            "num_classes": len(self.idx_to_label) if self.idx_to_label else "unknown",
            "total_parameters": num_params,
            "trainable_parameters": trainable_params,
            "labels": list(self.idx_to_label.values()) if self.idx_to_label else []
        }


class EnsembleDiagnosticService:
    """
    앙상블 진단 서비스
    여러 모델의 예측을 결합하여 더 정확한 진단
    """
    
    def __init__(
        self,
        model_paths: List[str],
        model_types: List[str],
        config_path: str,
        weights: Optional[List[float]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.services = []
        
        for model_path, model_type in zip(model_paths, model_types):
            service = SoundDiagnosticService(
                model_path=model_path,
                model_type=model_type,
                config_path=config_path,
                device=device
            )
            self.services.append(service)
        
        if weights is None:
            weights = [1.0 / len(self.services)] * len(self.services)
        self.weights = weights
    
    def diagnose(
        self, 
        audio_input: Union[str, bytes, np.ndarray]
    ) -> VehicleDiagnosis:
        """앙상블 진단"""
        all_probs = []
        
        for service in self.services:
            result = service.diagnose(audio_input, return_all_probs=True)
            all_probs.append(result.all_predictions)
        
        # 가중 평균 계산
        combined_probs = {}
        for probs, weight in zip(all_probs, self.weights):
            for label, prob in probs.items():
                combined_probs[label] = combined_probs.get(label, 0) + prob * weight
        
        # 최고 확률 클래스
        best_label = max(combined_probs, key=combined_probs.get)
        confidence = combined_probs[best_label]
        
        # 결과 생성 (첫 번째 서비스의 메서드 사용)
        state, problem = self.services[0]._parse_label(best_label)
        severity = self.services[0]._determine_severity(problem, confidence)
        recommendations = self.services[0]._get_recommendations(problem)
        
        return VehicleDiagnosis(
            state=state,
            problem=problem,
            confidence=confidence,
            severity=severity,
            recommendations=recommendations,
            all_predictions=combined_probs
        )

