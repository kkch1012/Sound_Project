"""
데이터셋 모듈
차량 사운드 데이터 로딩 및 전처리

Note: Windows에서 num_workers > 0 사용 시 
      if __name__ == '__main__' 블록 내에서 실행해야 합니다.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple, List, Dict
from pathlib import Path
import random
import platform

from app.ml.features.extractor import AudioFeatureExtractor, AudioConfig
from app.ml.features.augmentation import AudioAugmentor


class SoundDataset(Dataset):
    """
    차량 사운드 데이터셋
    
    디렉토리 구조:
    data/
    ├── braking state/
    │   ├── normal_brakes/
    │   └── worn_out_brakes/
    ├── idle state/
    │   ├── normal_engine_idle/
    │   ├── low_oil/
    │   └── ...
    └── startup state/
        ├── normal_engine_startup/
        ├── bad_ignition/
        └── ...
    """
    
    def __init__(
        self,
        data_dir: str,
        feature_extractor: Optional[AudioFeatureExtractor] = None,
        augmentor: Optional[AudioAugmentor] = None,
        target_shape: Tuple[int, int] = (128, 216),
        augment: bool = False,
        cache_features: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.feature_extractor = feature_extractor or AudioFeatureExtractor()
        self.augmentor = augmentor or AudioAugmentor()
        self.target_shape = target_shape
        self.augment = augment
        self.cache_features = cache_features
        
        # 파일 목록 및 레이블 로드
        self.samples = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        
        # 상태별 레이블 매핑 (계층적)
        self.state_labels = {}  # 상위 카테고리 (braking, idle, startup)
        self.problem_labels = {}  # 하위 카테고리 (각 세부 문제)
        
        self._load_data()
        
        # 피처 캐시
        self.feature_cache = {} if cache_features else None
        
    def _load_data(self):
        """데이터 디렉토리 탐색 및 샘플 목록 생성"""
        label_idx = 0
        state_idx = 0
        
        # 상태별 디렉토리 탐색
        for state_dir in sorted(self.data_dir.iterdir()):
            if not state_dir.is_dir():
                continue
                
            state_name = state_dir.name  # e.g., "braking state"
            
            if state_name not in self.state_labels:
                self.state_labels[state_name] = state_idx
                state_idx += 1
            
            # 문제별 디렉토리 탐색
            for problem_dir in sorted(state_dir.iterdir()):
                if not problem_dir.is_dir():
                    continue
                    
                problem_name = problem_dir.name  # e.g., "normal_brakes"
                full_label = f"{state_name}/{problem_name}"
                
                if full_label not in self.label_to_idx:
                    self.label_to_idx[full_label] = label_idx
                    self.idx_to_label[label_idx] = full_label
                    self.problem_labels[full_label] = {
                        "state": state_name,
                        "state_idx": self.state_labels[state_name],
                        "problem": problem_name,
                        "problem_idx": label_idx
                    }
                    label_idx += 1
                
                # WAV 파일 수집
                for audio_file in problem_dir.glob("*.wav"):
                    self.samples.append({
                        "path": str(audio_file),
                        "label": full_label,
                        "label_idx": self.label_to_idx[full_label],
                        "state": state_name,
                        "state_idx": self.state_labels[state_name],
                        "problem": problem_name
                    })
                    
                # combined 폴더 내부도 탐색
                for sub_dir in problem_dir.iterdir():
                    if sub_dir.is_dir():
                        sub_label = f"{state_name}/{problem_name}/{sub_dir.name}"
                        if sub_label not in self.label_to_idx:
                            self.label_to_idx[sub_label] = label_idx
                            self.idx_to_label[label_idx] = sub_label
                            label_idx += 1
                            
                        for audio_file in sub_dir.glob("*.wav"):
                            self.samples.append({
                                "path": str(audio_file),
                                "label": sub_label,
                                "label_idx": self.label_to_idx[sub_label],
                                "state": state_name,
                                "state_idx": self.state_labels[state_name],
                                "problem": sub_dir.name
                            })
        
        print(f"Loaded {len(self.samples)} samples")
        print(f"Number of classes: {len(self.label_to_idx)}")
        print(f"Number of states: {len(self.state_labels)}")
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        file_path = sample["path"]
        label_idx = sample["label_idx"]
        
        # 캐시 확인
        if self.feature_cache is not None and file_path in self.feature_cache:
            features = self.feature_cache[file_path]
        else:
            # 피처 추출
            features = self.feature_extractor.extract_for_cnn(
                file_path, 
                target_shape=self.target_shape
            )
            
            if self.feature_cache is not None:
                self.feature_cache[file_path] = features
        
        # 증강 적용 (학습 시에만)
        if self.augment:
            features = self._apply_augmentation(features)
        
        # Tensor로 변환
        features_tensor = torch.from_numpy(features).float()
        
        return features_tensor, label_idx
    
    def _apply_augmentation(self, features: np.ndarray) -> np.ndarray:
        """스펙트로그램 기반 증강 적용"""
        if random.random() > 0.5:
            # SpecAugment
            features = features.squeeze(0)  # (H, W)
            features = self.augmentor.spec_augment(
                features,
                num_freq_masks=2,
                num_time_masks=2,
                freq_mask_param=15,
                time_mask_param=35
            )
            features = features[np.newaxis, :, :]  # (1, H, W)
        return features
    
    def get_sample_info(self, idx: int) -> Dict:
        """샘플 정보 반환"""
        return self.samples[idx]
    
    def get_class_weights(self) -> torch.Tensor:
        """클래스 불균형 해결을 위한 가중치 계산"""
        class_counts = np.zeros(len(self.label_to_idx))
        for sample in self.samples:
            class_counts[sample["label_idx"]] += 1
        
        # Inverse frequency weighting
        weights = 1.0 / (class_counts + 1e-6)
        weights = weights / weights.sum() * len(weights)
        
        return torch.from_numpy(weights).float()


class SoundDatasetMultiChannel(SoundDataset):
    """
    멀티채널 입력 데이터셋
    Mel Spectrogram + MFCC를 채널로 결합
    """
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        file_path = sample["path"]
        label_idx = sample["label_idx"]
        
        # 멀티채널 피처 추출
        features = self.feature_extractor.extract_for_crnn(
            file_path,
            target_shape=self.target_shape
        )
        
        if self.augment:
            # 각 채널에 증강 적용
            for c in range(features.shape[0]):
                if random.random() > 0.5:
                    features[c] = self.augmentor.spec_augment(features[c])
        
        features_tensor = torch.from_numpy(features).float()
        
        return features_tensor, label_idx


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    test_split: float = 0.1,
    num_workers: int = 4,
    target_shape: Tuple[int, int] = (128, 216),
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    학습/검증/테스트 데이터로더 생성
    
    Returns:
        train_loader, val_loader, test_loader, dataset_info
    
    Note:
        Windows에서는 멀티프로세싱 이슈로 num_workers가 자동으로 0으로 설정됩니다.
    """
    # Windows 호환성: num_workers 조정
    if platform.system() == 'Windows' and num_workers > 0:
        print("Warning: Windows에서 num_workers=0으로 설정됩니다.")
        num_workers = 0
    
    # 전체 데이터셋 로드
    full_dataset = SoundDataset(
        data_dir=data_dir,
        target_shape=target_shape,
        augment=False
    )
    
    # 데이터 분할
    total_size = len(full_dataset)
    indices = list(range(total_size))
    
    random.seed(seed)
    random.shuffle(indices)
    
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - test_size - val_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # 서브셋 데이터셋 생성
    train_dataset = SoundDatasetSubset(full_dataset, train_indices, augment=True)
    val_dataset = SoundDatasetSubset(full_dataset, val_indices, augment=False)
    test_dataset = SoundDatasetSubset(full_dataset, test_indices, augment=False)
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    dataset_info = {
        "num_classes": len(full_dataset.label_to_idx),
        "num_states": len(full_dataset.state_labels),
        "label_to_idx": full_dataset.label_to_idx,
        "idx_to_label": full_dataset.idx_to_label,
        "state_labels": full_dataset.state_labels,
        "class_weights": full_dataset.get_class_weights(),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset)
    }
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, dataset_info


class SoundDatasetSubset(Dataset):
    """데이터셋 서브셋 (증강 옵션 포함)"""
    
    def __init__(
        self, 
        dataset: SoundDataset, 
        indices: List[int],
        augment: bool = False
    ):
        self.dataset = dataset
        self.indices = indices
        self.augment = augment
        self.augmentor = AudioAugmentor()
        
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        original_idx = self.indices[idx]
        features, label = self.dataset[original_idx]
        
        if self.augment:
            # 추가 증강
            features_np = features.numpy()
            if random.random() > 0.5:
                features_np = features_np.squeeze(0)
                features_np = self.augmentor.spec_augment(features_np)
                features_np = features_np[np.newaxis, :, :]
            features = torch.from_numpy(features_np).float()
        
        return features, label

