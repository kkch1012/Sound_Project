"""
다른 컴퓨터에서 상태 분류 모델을 사용하기 위한 추론 스크립트

필요한 파일:
1. best_ensemble_state_model.pth (모델 가중치 파일)
2. 이 스크립트 (inference_state_model.py)
3. importance_mask.npy (선택사항, 없으면 기본값 사용)

사용 방법:
    python inference_state_model.py --audio_path "path/to/audio.wav"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from pathlib import Path
import argparse


# ============================================================
# 모델 클래스 정의 (학습 시와 동일해야 함)
# ============================================================

class WaveformCNN1D(nn.Module):
    """Waveform을 입력으로 받는 1D CNN 모델"""
    
    def __init__(
        self,
        num_classes: int,
        input_length: int = 110250,  # 5초 @ 22050 Hz
        base_channels: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # 1D Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout / 2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout / 2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout / 2)
        )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(base_channels * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.fc_out = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x: (batch, 1, length)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc_out(x)
        return x


class MaskedSpatialAttention(nn.Module):
    """마스킹 기반 Spatial Attention"""
    
    def __init__(self, importance_mask: torch.Tensor, learnable: bool = True):
        super().__init__()
        self.importance_mask = nn.Parameter(importance_mask, requires_grad=learnable)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, freq, time)
        # 마스크를 입력 shape에 맞게 조정
        mask = self.importance_mask.expand(x.size(0), -1, -1)
        mask = mask.unsqueeze(1)  # (batch, 1, freq, time)
        
        # 마스크 적용 (중요 영역 강조)
        x = x * (1 + mask)
        return x


class MaskedCNN(nn.Module):
    """마스킹 기반 Mel Spectrogram CNN 모델"""
    
    def __init__(self, num_classes: int, importance_mask: torch.Tensor, 
                 in_channels: int = 1, base_channels: int = 32, dropout: float = 0.3):
        super().__init__()
        
        self.num_classes = num_classes
        
        # 마스킹 기반 Spatial Attention
        self.masked_attention = MaskedSpatialAttention(importance_mask, learnable=True)
        
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout)
        )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(base_channels * 8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.fc_out = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 마스킹 기반 Attention 적용
        x = self.masked_attention(x)
        
        # Convolutional blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Global Average Pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc_out(x)
        
        return x


class EnsembleVoteModel(nn.Module):
    """
    Waveform과 Mel Spectrogram 모델을 Vote 방식으로 결합
    """
    
    def __init__(
        self,
        waveform_model: nn.Module,
        spectrogram_model: nn.Module,
        vote_method: str = 'soft'  # 'hard' or 'soft'
    ):
        super().__init__()
        
        self.waveform_model = waveform_model
        self.spectrogram_model = spectrogram_model
        self.vote_method = vote_method
    
    def forward(self, waveform_input, spectrogram_input):
        """
        두 모델의 예측을 결합
        
        Args:
            waveform_input: Waveform 텐서 (batch, 1, length)
            spectrogram_input: Mel Spectrogram 텐서 (batch, 1, freq, time)
        
        Returns:
            ensemble_output: 앙상블 예측 결과 (batch, num_classes) - logits
        """
        # Waveform 모델 예측
        waveform_output = self.waveform_model(waveform_input)
        waveform_probs = F.softmax(waveform_output, dim=1)
        
        # Mel Spectrogram 모델 예측
        spec_output = self.spectrogram_model(spectrogram_input)
        spec_probs = F.softmax(spec_output, dim=1)
        
        # Vote 방식에 따라 결합
        if self.vote_method == 'hard':
            # Hard Voting: 다수결
            waveform_pred = waveform_output.argmax(dim=1)
            spec_pred = spec_output.argmax(dim=1)
            
            ensemble_pred = torch.stack([waveform_pred, spec_pred], dim=1)
            ensemble_pred = torch.mode(ensemble_pred, dim=1)[0]
            
            # One-hot으로 변환 후 logits로 변환
            ensemble_output = F.one_hot(ensemble_pred, num_classes=waveform_output.size(1)).float()
            # logits로 변환 (큰 값 사용)
            ensemble_output = ensemble_output * 10.0 - 5.0
        
        else:  # soft voting
            # Soft Voting: logits 평균 (학습 시 사용)
            ensemble_output = (waveform_output + spec_output) / 2
        
        return ensemble_output


# ============================================================
# 전처리 함수
# ============================================================

def preprocess_audio(audio_path, sample_rate=22050, duration=5.0, n_mels=128, n_fft=2048, hop_length=512):
    """
    오디오 파일을 전처리하여 모델 입력 형태로 변환
    
    Returns:
        waveform: (1, 1, length) 형태의 텐서
        mel_spec: (1, 1, freq, time) 형태의 텐서
    """
    # 오디오 로드
    y, sr = librosa.load(str(audio_path), sr=sample_rate)
    
    # Waveform 길이 정규화
    target_length = int(sample_rate * duration)
    if len(y) < target_length:
        y_padded = np.pad(y, (0, target_length - len(y)), mode='constant')
    elif len(y) > target_length:
        y_padded = y[:target_length]
    else:
        y_padded = y
    
    # Mel Spectrogram 추출
    mel_spec = librosa.feature.melspectrogram(
        y=y_padded,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # dB로 변환
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 텐서로 변환
    waveform = torch.FloatTensor(y_padded).unsqueeze(0).unsqueeze(0)  # (1, 1, length)
    mel_spec = torch.FloatTensor(mel_spec_db).unsqueeze(0)  # (1, freq, time)
    
    return waveform, mel_spec


# ============================================================
# 추론 함수
# ============================================================

def load_model(model_path, device='cpu', num_classes=3, importance_mask=None):
    """모델 로드"""
    # 중요 영역 마스크 설정
    if importance_mask is None:
        # 기본값: 모든 영역 동일한 중요도 (128, 216)
        importance_mask = torch.zeros(128, 216)
    else:
        importance_mask = torch.FloatTensor(importance_mask)
    
    importance_mask = importance_mask.to(device)
    
    # 개별 모델 생성
    waveform_model = WaveformCNN1D(
        num_classes=num_classes,
        input_length=110250,  # 5초 @ 22050 Hz
        base_channels=64,
        dropout=0.3
    )
    
    mel_model = MaskedCNN(
        num_classes=num_classes,
        importance_mask=importance_mask,
        in_channels=1,
        base_channels=32,
        dropout=0.3
    )
    
    # 앙상블 모델 생성
    model = EnsembleVoteModel(
        waveform_model=waveform_model,
        spectrogram_model=mel_model,
        vote_method='soft'
    )
    
    # 가중치 로드
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(audio_path, model_path, device='cpu', importance_mask_path=None):
    """
    오디오 파일에 대한 예측 수행
    
    Args:
        audio_path: 오디오 파일 경로
        model_path: 모델 가중치 파일 경로 (.pth)
        device: 사용할 디바이스 ('cpu' 또는 'cuda')
        importance_mask_path: 중요 영역 마스크 파일 경로 (선택사항)
    
    Returns:
        predicted_label: 예측된 상태 이름
        probabilities: 각 상태에 대한 확률
    """
    # 레이블 매핑 (학습 시와 동일한 순서)
    idx_to_state = {
        0: 'braking state',
        1: 'idle state',
        2: 'startup state'
    }
    
    # 중요 영역 마스크 로드
    if importance_mask_path and Path(importance_mask_path).exists():
        importance_mask = np.load(importance_mask_path)
    else:
        importance_mask = None
    
    # 모델 로드
    model = load_model(model_path, device, num_classes=len(idx_to_state), importance_mask=importance_mask)
    
    # 전처리
    waveform, mel_spec = preprocess_audio(audio_path)
    waveform = waveform.to(device)
    mel_spec = mel_spec.to(device)
    
    # 예측
    with torch.no_grad():
        outputs = model(waveform, mel_spec)
        
        probabilities = F.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        predicted_state = idx_to_state[predicted_idx]
        
        prob_dict = {idx_to_state[i]: prob.item() for i, prob in enumerate(probabilities[0])}
    
    return predicted_state, prob_dict


# ============================================================
# 메인 함수
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='상태 분류 모델 추론 (Braking, Idle, Startup)')
    parser.add_argument('--audio_path', type=str, required=True, help='오디오 파일 경로')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_ensemble_state_model.pth', 
                       help='모델 가중치 파일 경로')
    parser.add_argument('--importance_mask', type=str, default=None, 
                       help='중요 영역 마스크 파일 경로 (선택사항)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], 
                       help='사용할 디바이스')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA를 사용할 수 없습니다. CPU로 전환합니다.")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    
    # 예측 수행
    print(f"📂 오디오 파일 로드: {args.audio_path}")
    print(f"🤖 모델 로드: {args.model_path}")
    
    try:
        predicted_state, probabilities = predict(
            args.audio_path,
            args.model_path,
            device=device,
            importance_mask_path=args.importance_mask
        )
        
        print(f"\n✅ 예측 결과:")
        print(f"  예측된 상태: {predicted_state}")
        print(f"\n📊 각 상태별 확률:")
        for state, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            print(f"  {state}: {prob*100:.2f}%")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

