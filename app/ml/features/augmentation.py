"""
오디오 데이터 증강 모듈
학습 데이터 다양성 증가를 위한 증강 기법
"""
import numpy as np
import librosa
from typing import Optional, Tuple, List
from dataclasses import dataclass
import random


@dataclass
class AugmentationConfig:
    """증강 설정"""
    # Time Stretch
    time_stretch_rate_min: float = 0.8
    time_stretch_rate_max: float = 1.2
    
    # Pitch Shift
    pitch_shift_steps_min: int = -4
    pitch_shift_steps_max: int = 4
    
    # Noise
    noise_factor_min: float = 0.001
    noise_factor_max: float = 0.015
    
    # Volume
    volume_factor_min: float = 0.5
    volume_factor_max: float = 1.5
    
    # Time Shift
    time_shift_max: float = 0.2  # 최대 20% 시프트


class AudioAugmentor:
    """
    오디오 데이터 증강기
    차량 사운드 데이터의 다양성을 증가시키기 위한 증강 기법
    """
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        
    def time_stretch(
        self, 
        y: np.ndarray, 
        rate: Optional[float] = None
    ) -> np.ndarray:
        """
        시간 스트레칭
        재생 속도를 변경하여 다양한 RPM 상황 시뮬레이션
        """
        if rate is None:
            rate = random.uniform(
                self.config.time_stretch_rate_min,
                self.config.time_stretch_rate_max
            )
        
        # rate 검증: 1.0이면 변환 불필요, 0 이하면 오류
        if rate <= 0:
            raise ValueError(f"time_stretch rate must be positive, got {rate}")
        if abs(rate - 1.0) < 1e-6:
            return y.copy()
        
        return librosa.effects.time_stretch(y, rate=rate)
    
    def pitch_shift(
        self, 
        y: np.ndarray, 
        sr: int,
        n_steps: Optional[int] = None
    ) -> np.ndarray:
        """
        피치 시프트
        주파수를 변경하여 다양한 엔진 크기/타입 시뮬레이션
        """
        if n_steps is None:
            n_steps = random.randint(
                self.config.pitch_shift_steps_min,
                self.config.pitch_shift_steps_max
            )
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    
    def add_noise(
        self, 
        y: np.ndarray, 
        noise_factor: Optional[float] = None
    ) -> np.ndarray:
        """
        가우시안 노이즈 추가
        실제 환경의 배경 소음 시뮬레이션
        """
        if noise_factor is None:
            noise_factor = random.uniform(
                self.config.noise_factor_min,
                self.config.noise_factor_max
            )
        noise = np.random.randn(len(y))
        return y + noise_factor * noise
    
    def change_volume(
        self, 
        y: np.ndarray, 
        factor: Optional[float] = None
    ) -> np.ndarray:
        """
        볼륨 변경
        마이크 거리/감도 차이 시뮬레이션
        """
        if factor is None:
            factor = random.uniform(
                self.config.volume_factor_min,
                self.config.volume_factor_max
            )
        return y * factor
    
    def time_shift(
        self, 
        y: np.ndarray, 
        shift_max: Optional[float] = None
    ) -> np.ndarray:
        """
        시간 시프트
        녹음 시작점 변화 시뮬레이션
        """
        if shift_max is None:
            shift_max = self.config.time_shift_max
            
        shift = int(len(y) * random.uniform(-shift_max, shift_max))
        return np.roll(y, shift)
    
    def add_reverb(
        self, 
        y: np.ndarray, 
        sr: int,
        room_scale: float = 0.5
    ) -> np.ndarray:
        """
        간단한 리버브 효과
        실내/실외 환경 차이 시뮬레이션
        """
        # 간단한 컨볼루션 리버브 (임펄스 응답 시뮬레이션)
        impulse_length = int(sr * room_scale * 0.1)
        impulse = np.exp(-np.linspace(0, 5, impulse_length))
        impulse = impulse / np.sum(impulse)
        
        reverb = np.convolve(y, impulse, mode='same')
        return 0.7 * y + 0.3 * reverb
    
    def frequency_mask(
        self, 
        spec: np.ndarray, 
        num_masks: int = 1,
        freq_mask_param: int = 20
    ) -> np.ndarray:
        """
        주파수 마스킹 (SpecAugment)
        특정 주파수 대역을 마스킹하여 모델 견고성 향상
        """
        spec = spec.copy()
        num_freq_bins = spec.shape[0]
        
        for _ in range(num_masks):
            f = random.randint(0, freq_mask_param)
            f0 = random.randint(0, num_freq_bins - f)
            spec[f0:f0 + f, :] = 0
            
        return spec
    
    def time_mask(
        self, 
        spec: np.ndarray, 
        num_masks: int = 1,
        time_mask_param: int = 30
    ) -> np.ndarray:
        """
        시간 마스킹 (SpecAugment)
        특정 시간 구간을 마스킹하여 모델 견고성 향상
        """
        spec = spec.copy()
        num_time_steps = spec.shape[1]
        
        for _ in range(num_masks):
            t = random.randint(0, time_mask_param)
            t0 = random.randint(0, num_time_steps - t)
            spec[:, t0:t0 + t] = 0
            
        return spec
    
    def spec_augment(
        self, 
        spec: np.ndarray,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        freq_mask_param: int = 15,
        time_mask_param: int = 35
    ) -> np.ndarray:
        """
        SpecAugment 적용
        주파수 및 시간 마스킹 결합
        """
        spec = self.frequency_mask(spec, num_freq_masks, freq_mask_param)
        spec = self.time_mask(spec, num_time_masks, time_mask_param)
        return spec
    
    def augment(
        self, 
        y: np.ndarray, 
        sr: int,
        augmentations: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        랜덤 증강 적용
        여러 증강 기법을 확률적으로 적용
        """
        if augmentations is None:
            augmentations = [
                'time_stretch', 'pitch_shift', 'add_noise', 
                'change_volume', 'time_shift'
            ]
        
        augmented = y.copy()
        
        for aug_name in augmentations:
            # 각 증강을 50% 확률로 적용
            if random.random() > 0.5:
                continue
                
            if aug_name == 'time_stretch':
                augmented = self.time_stretch(augmented)
            elif aug_name == 'pitch_shift':
                augmented = self.pitch_shift(augmented, sr)
            elif aug_name == 'add_noise':
                augmented = self.add_noise(augmented)
            elif aug_name == 'change_volume':
                augmented = self.change_volume(augmented)
            elif aug_name == 'time_shift':
                augmented = self.time_shift(augmented)
            elif aug_name == 'add_reverb':
                augmented = self.add_reverb(augmented, sr)
                
        return augmented
    
    def generate_augmented_samples(
        self, 
        y: np.ndarray, 
        sr: int,
        num_augmented: int = 5
    ) -> List[np.ndarray]:
        """
        여러 개의 증강된 샘플 생성
        """
        samples = [y]  # 원본 포함
        
        for _ in range(num_augmented):
            augmented = self.augment(y, sr)
            samples.append(augmented)
            
        return samples

