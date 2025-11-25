"""
오디오 피처 추출 모듈
MFCC, Mel-Spectrogram, Chroma, Spectral Features 등 다양한 피처 추출
"""
import numpy as np
import librosa
import cv2
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class FeatureType(str, Enum):
    MFCC = "mfcc"
    MEL_SPECTROGRAM = "mel_spectrogram"
    CHROMA = "chroma"
    SPECTRAL_CONTRAST = "spectral_contrast"
    TONNETZ = "tonnetz"
    COMBINED = "combined"


@dataclass
class AudioConfig:
    """오디오 처리 설정"""
    sample_rate: int = 22050
    duration: float = 5.0  # 초
    n_mfcc: int = 40
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    n_chroma: int = 12
    

class AudioFeatureExtractor:
    """
    종합적인 오디오 피처 추출기
    차량 사운드 분석에 최적화된 다양한 피처 추출 기능 제공
    """
    
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        
    def load_audio(
        self, 
        file_path: str, 
        duration: Optional[float] = None
    ) -> Tuple[np.ndarray, int]:
        """오디오 파일 로드 및 전처리"""
        duration = duration or self.config.duration
        
        # 오디오 로드
        y, sr = librosa.load(
            file_path, 
            sr=self.config.sample_rate,
            duration=duration
        )
        
        # 길이 정규화 (패딩 또는 트리밍)
        target_length = int(self.config.sample_rate * duration)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            y = y[:target_length]
            
        return y, sr
    
    def extract_mfcc(
        self, 
        y: np.ndarray, 
        sr: int,
        include_delta: bool = True
    ) -> np.ndarray:
        """
        MFCC (Mel-frequency cepstral coefficients) 추출
        - 음성/사운드의 단기 파워 스펙트럼 표현
        - Delta 및 Delta-Delta 포함 시 더 풍부한 시간적 정보
        """
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=self.config.n_mfcc,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        )
        
        if include_delta:
            # Delta (1차 미분) - 시간에 따른 변화율
            mfcc_delta = librosa.feature.delta(mfcc)
            # Delta-Delta (2차 미분) - 가속도
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            mfcc = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
            
        return mfcc
    
    def extract_mel_spectrogram(
        self, 
        y: np.ndarray, 
        sr: int,
        to_db: bool = True
    ) -> np.ndarray:
        """
        Mel Spectrogram 추출
        - 인간의 청각 특성을 반영한 주파수 스케일
        - 차량 소리의 주파수 패턴 분석에 효과적
        """
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        )
        
        if to_db:
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
        return mel_spec
    
    def extract_chroma(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Chroma Features 추출
        - 12개의 피치 클래스로 에너지 분포
        - 주기적인 엔진 사운드 패턴 감지에 유용
        """
        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=sr,
            n_chroma=self.config.n_chroma,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        )
        return chroma
    
    def extract_spectral_contrast(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Spectral Contrast 추출
        - 주파수 대역별 피크와 밸리의 차이
        - 다양한 엔진 상태의 스펙트럼 차이 감지
        """
        contrast = librosa.feature.spectral_contrast(
            y=y,
            sr=sr,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        )
        return contrast
    
    def extract_tonnetz(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Tonnetz (Tonal Centroid) 추출
        - 하모닉 관계 표현
        - 엔진의 하모닉 특성 분석
        """
        # Harmonic component 추출
        y_harmonic = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(
            y=y_harmonic,
            sr=sr
        )
        return tonnetz
    
    def extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        추가 스펙트럼 피처 추출
        - Spectral Centroid: 스펙트럼의 "무게 중심"
        - Spectral Bandwidth: 스펙트럼 폭
        - Spectral Rolloff: 에너지가 집중된 주파수
        - Zero Crossing Rate: 신호가 0을 지나는 비율
        """
        features = {}
        
        # Spectral Centroid
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=self.config.hop_length
        )
        
        # Spectral Bandwidth
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, hop_length=self.config.hop_length
        )
        
        # Spectral Rolloff
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            y=y, sr=sr, hop_length=self.config.hop_length
        )
        
        # Zero Crossing Rate
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(
            y, hop_length=self.config.hop_length
        )
        
        # RMS Energy
        features['rms'] = librosa.feature.rms(
            y=y, hop_length=self.config.hop_length
        )
        
        return features
    
    def extract_statistical_features(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        피처 매트릭스에서 통계적 특징 추출
        시간 축을 따라 요약 통계량 계산
        """
        stats = []
        
        for feature_row in feature_matrix:
            row_stats = [
                np.mean(feature_row),
                np.std(feature_row),
                np.min(feature_row),
                np.max(feature_row),
                np.median(feature_row),
                # Skewness (비대칭도)
                self._skewness(feature_row),
                # Kurtosis (첨도)
                self._kurtosis(feature_row),
            ]
            stats.extend(row_stats)
            
        return np.array(stats)
    
    def _skewness(self, x: np.ndarray) -> float:
        """비대칭도 계산"""
        n = len(x)
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0.0
        return (np.sum((x - mean) ** 3) / n) / (std ** 3)
    
    def _kurtosis(self, x: np.ndarray) -> float:
        """첨도 계산"""
        n = len(x)
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0.0
        return (np.sum((x - mean) ** 4) / n) / (std ** 4) - 3
    
    def extract_all_features(
        self, 
        file_path: str,
        feature_type: FeatureType = FeatureType.COMBINED
    ) -> Dict[str, np.ndarray]:
        """
        지정된 피처 타입에 따라 피처 추출
        COMBINED: 모든 피처를 추출하여 종합
        """
        y, sr = self.load_audio(file_path)
        
        features = {}
        
        if feature_type in [FeatureType.MFCC, FeatureType.COMBINED]:
            features['mfcc'] = self.extract_mfcc(y, sr)
            
        if feature_type in [FeatureType.MEL_SPECTROGRAM, FeatureType.COMBINED]:
            features['mel_spectrogram'] = self.extract_mel_spectrogram(y, sr)
            
        if feature_type in [FeatureType.CHROMA, FeatureType.COMBINED]:
            features['chroma'] = self.extract_chroma(y, sr)
            
        if feature_type in [FeatureType.SPECTRAL_CONTRAST, FeatureType.COMBINED]:
            features['spectral_contrast'] = self.extract_spectral_contrast(y, sr)
            
        if feature_type in [FeatureType.TONNETZ, FeatureType.COMBINED]:
            features['tonnetz'] = self.extract_tonnetz(y, sr)
            
        if feature_type == FeatureType.COMBINED:
            spectral_features = self.extract_spectral_features(y, sr)
            features.update(spectral_features)
            
        return features
    
    def extract_for_cnn(
        self, 
        file_path: str,
        target_shape: Tuple[int, int] = (128, 216)
    ) -> np.ndarray:
        """
        CNN 입력용 Mel Spectrogram 추출
        고정된 크기로 리사이즈
        """
        y, sr = self.load_audio(file_path)
        mel_spec = self.extract_mel_spectrogram(y, sr, to_db=True)
        
        # 크기 조정
        mel_spec = self._resize_spectrogram(mel_spec, target_shape)
        
        # 정규화
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        # 채널 차원 추가 (1, H, W) for CNN
        return mel_spec[np.newaxis, :, :]
    
    def extract_for_crnn(
        self, 
        file_path: str,
        target_shape: Tuple[int, int] = (128, 216)
    ) -> np.ndarray:
        """
        CRNN 입력용 피처 추출
        Mel Spectrogram + MFCC를 멀티채널로 결합
        """
        y, sr = self.load_audio(file_path)
        
        # Mel Spectrogram
        mel_spec = self.extract_mel_spectrogram(y, sr, to_db=True)
        mel_spec = self._resize_spectrogram(mel_spec, target_shape)
        
        # MFCC (n_mfcc x time) -> 리사이즈
        mfcc = self.extract_mfcc(y, sr, include_delta=False)
        mfcc = self._resize_spectrogram(mfcc, (self.config.n_mfcc, target_shape[1]))
        
        # 정규화
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
        
        # MFCC를 mel_spec 크기에 맞게 패딩
        mfcc_padded = np.zeros(target_shape)
        mfcc_padded[:self.config.n_mfcc, :] = mfcc
        
        # 2채널로 스택
        return np.stack([mel_spec, mfcc_padded], axis=0)
    
    def _resize_spectrogram(
        self, 
        spec: np.ndarray, 
        target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """스펙트로그램 크기 조정 (간단한 보간법)"""
        # OpenCV는 (width, height) 순서
        resized = cv2.resize(
            spec, 
            (target_shape[1], target_shape[0]), 
            interpolation=cv2.INTER_LINEAR
        )
        return resized

