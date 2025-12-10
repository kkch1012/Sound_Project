"""
CRNN (Convolutional Recurrent Neural Network) 모델
멜 스펙트로그램의 시간적 패턴을 학습하기 위한 모델

구조:
1. CNN: 주파수 축의 공간적 특징 추출
2. RNN (LSTM/GRU): 시간 축의 시퀀스 패턴 학습
3. Fully Connected: 최종 분류
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNNMelSpectrogram(nn.Module):
    """
    멜 스펙트로그램을 위한 CRNN 모델
    
    Args:
        num_classes: 분류할 클래스 수
        n_mels: 멜 빈 개수 (기본 128)
        time_frames: 시간 프레임 개수 (기본 87)
        base_channels: CNN 기본 채널 수
        hidden_size: RNN hidden size
        num_layers: RNN 레이어 수
        dropout: Dropout 비율
        rnn_type: 'LSTM' 또는 'GRU'
        bidirectional: 양방향 RNN 사용 여부
    """
    
    def __init__(
        self,
        num_classes: int,
        n_mels: int = 128,
        time_frames: int = 87,
        base_channels: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        rnn_type: str = 'LSTM',
        bidirectional: bool = True
    ):
        super(CRNNMelSpectrogram, self).__init__()
        
        self.num_classes = num_classes
        self.n_mels = n_mels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        
        # ============================================================
        # CNN 부분: 주파수 축의 공간적 특징 추출
        # ============================================================
        
        # Conv Block 1: (n_mels, time_frames) -> (n_mels/2, time_frames/2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout2d(dropout / 3)
        )
        
        # Conv Block 2: (n_mels/2, time_frames/2) -> (n_mels/4, time_frames/4)
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout2d(dropout / 3)
        )
        
        # Conv Block 3: (n_mels/4, time_frames/4) -> (n_mels/8, time_frames/8)
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout2d(dropout / 3)
        )
        
        # CNN 출력 차원 계산
        # 입력: (batch, 1, n_mels, time_frames)
        # After conv1: (batch, base_channels, n_mels/2, time_frames/2)
        # After conv2: (batch, base_channels*2, n_mels/4, time_frames/4)
        # After conv3: (batch, base_channels*4, n_mels/8, time_frames/8)
        
        # RNN 입력 차원: (batch, time_steps, features)
        # time_steps = time_frames / 8
        # features = (base_channels * 4) * (n_mels / 8)
        self.cnn_output_features = base_channels * 4
        self.cnn_output_time = time_frames // 8  # 3번의 MaxPool2d(2,2) 적용
        self.cnn_output_freq = n_mels // 8
        
        # ============================================================
        # RNN 부분: 시간적 패턴 학습
        # ============================================================
        
        # CNN 출력을 RNN 입력 형태로 변환
        # (batch, channels, freq, time) -> (batch, time, channels * freq)
        rnn_input_size = self.cnn_output_features * self.cnn_output_freq
        
        if rnn_type.upper() == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=rnn_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        elif rnn_type.upper() == 'GRU':
            self.rnn = nn.GRU(
                input_size=rnn_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        else:
            raise ValueError(f"rnn_type must be 'LSTM' or 'GRU', got {rnn_type}")
        
        # RNN 출력 차원
        rnn_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # ============================================================
        # Fully Connected 부분: 최종 분류
        # ============================================================
        
        self.fc1 = nn.Sequential(
            nn.Linear(rnn_output_size, 256),
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
    
    def forward(self, x, mask=None):
        """
        Forward pass
        
        Args:
            x: 입력 텐서 (batch, 1, n_mels, time_frames)
            mask: 중요 영역 마스크 (batch, n_mels, time_frames) - 선택사항
        
        Returns:
            output: 분류 로짓 (batch, num_classes)
        """
        # 마스크 적용 (선택사항)
        if mask is not None:
            # 마스크를 (batch, 1, n_mels, time_frames) 형태로 확장
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            x = x * (1 + mask)
        
        # ============================================================
        # CNN: 공간적 특징 추출
        # ============================================================
        x = self.conv1(x)  # (batch, base_channels, n_mels/2, time_frames/2)
        x = self.conv2(x)  # (batch, base_channels*2, n_mels/4, time_frames/4)
        x = self.conv3(x)  # (batch, base_channels*4, n_mels/8, time_frames/8)
        
        # ============================================================
        # CNN 출력을 RNN 입력 형태로 변환
        # ============================================================
        # (batch, channels, freq, time) -> (batch, time, channels * freq)
        batch_size = x.size(0)
        x = x.permute(0, 3, 1, 2)  # (batch, time, channels, freq)
        x = x.contiguous().view(batch_size, x.size(1), -1)  # (batch, time, channels * freq)
        
        # ============================================================
        # RNN: 시간적 패턴 학습
        # ============================================================
        rnn_out, _ = self.rnn(x)  # (batch, time, hidden_size * directions)
        
        # 마지막 시간 스텝의 출력 사용 (또는 평균/최대 풀링)
        # Option 1: 마지막 스텝만 사용
        rnn_out = rnn_out[:, -1, :]  # (batch, hidden_size * directions)
        
        # Option 2: 평균 풀링 (주석 해제하여 사용 가능)
        # rnn_out = torch.mean(rnn_out, dim=1)  # (batch, hidden_size * directions)
        
        # ============================================================
        # Fully Connected: 최종 분류
        # ============================================================
        x = self.fc1(rnn_out)
        x = self.fc2(x)
        output = self.fc_out(x)
        
        return output


class CRNNMelSpectrogramWithAttention(nn.Module):
    """
    Attention 메커니즘을 추가한 CRNN 모델
    시간 스텝별 중요도를 학습하여 더 나은 성능 기대
    """
    
    def __init__(
        self,
        num_classes: int,
        n_mels: int = 128,
        time_frames: int = 87,
        base_channels: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        rnn_type: str = 'LSTM',
        bidirectional: bool = True
    ):
        super(CRNNMelSpectrogramWithAttention, self).__init__()
        
        # 기본 CRNN 구조 (위와 동일)
        self.crnn = CRNNMelSpectrogram(
            num_classes=num_classes,
            n_mels=n_mels,
            time_frames=time_frames,
            base_channels=base_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            rnn_type=rnn_type,
            bidirectional=bidirectional
        )
        
        # Attention 메커니즘
        rnn_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = nn.Sequential(
            nn.Linear(rnn_output_size, rnn_output_size // 2),
            nn.Tanh(),
            nn.Linear(rnn_output_size // 2, 1)
        )
    
    def forward(self, x, mask=None):
        # CNN 부분은 동일하게 처리
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            x = x * (1 + mask)
        
        x = self.crnn.conv1(x)
        x = self.crnn.conv2(x)
        x = self.crnn.conv3(x)
        
        # RNN 입력 형태로 변환
        batch_size = x.size(0)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(batch_size, x.size(1), -1)
        
        # RNN: 모든 시간 스텝의 출력 사용
        rnn_out, _ = self.crnn.rnn(x)  # (batch, time, hidden_size * directions)
        
        # Attention 가중치 계산
        attention_weights = self.attention(rnn_out)  # (batch, time, 1)
        attention_weights = F.softmax(attention_weights, dim=1)  # (batch, time, 1)
        
        # Attention 가중치를 적용한 가중 평균
        attended_output = torch.sum(rnn_out * attention_weights, dim=1)  # (batch, hidden_size * directions)
        
        # Fully Connected
        x = self.crnn.fc1(attended_output)
        x = self.crnn.fc2(x)
        output = self.crnn.fc_out(x)
        
        return output


# ============================================================
# 사용 예시
# ============================================================

if __name__ == '__main__':
    # 모델 생성
    model = CRNNMelSpectrogram(
        num_classes=4,
        n_mels=128,
        time_frames=87,
        base_channels=64,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        rnn_type='LSTM',
        bidirectional=True
    )
    
    # 더미 입력으로 테스트
    dummy_input = torch.randn(2, 1, 128, 87)  # (batch, channels, freq, time)
    output = model(dummy_input)
    
    print(f"입력 shape: {dummy_input.shape}")
    print(f"출력 shape: {output.shape}")
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # Attention 버전 테스트
    model_attn = CRNNMelSpectrogramWithAttention(
        num_classes=4,
        n_mels=128,
        time_frames=87
    )
    output_attn = model_attn(dummy_input)
    print(f"\nAttention 모델 출력 shape: {output_attn.shape}")
    print(f"Attention 모델 파라미터 수: {sum(p.numel() for p in model_attn.parameters()):,}")

