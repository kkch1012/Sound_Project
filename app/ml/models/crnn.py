"""
CRNN (Convolutional Recurrent Neural Network) 기반 사운드 분류 모델
CNN으로 공간적 특징 추출 + RNN으로 시간적 패턴 학습
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class SoundClassifierCRNN(nn.Module):
    """
    CRNN 기반 차량 사운드 분류기
    
    Architecture:
    - CNN: 스펙트로그램의 주파수 패턴 추출
    - Bidirectional LSTM: 시간적 의존성 학습
    - Attention: 중요한 시간 구간에 집중
    - FC: 최종 분류
    """
    
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        cnn_channels: Optional[List[int]] = None,
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        
        if cnn_channels is None:
            cnn_channels = [32, 64, 128]
        
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.rnn_hidden_size = rnn_hidden_size
        
        # CNN Feature Extractor
        cnn_layers = []
        in_ch = in_channels
        for out_ch in cnn_channels:
            cnn_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1)),  # 주파수만 풀링
                nn.Dropout2d(dropout / 2)
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        
        # 출력 주파수 차원 계산 (3번 pooling으로 128 -> 16)
        self.cnn_output_freq = 128 // (2 ** len(cnn_channels))
        rnn_input_size = cnn_channels[-1] * self.cnn_output_freq
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if rnn_num_layers > 1 else 0
        )
        
        # Attention mechanism
        rnn_output_size = rnn_hidden_size * 2 if bidirectional else rnn_hidden_size
        self.attention = TemporalAttention(rnn_output_size)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(rnn_output_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, channels, freq, time)
               Expected shape: (batch, 1, 128, 216)
        
        Returns:
            Output tensor of shape (batch, num_classes)
        """
        batch_size = x.size(0)
        
        # CNN feature extraction
        # (batch, 1, 128, 216) -> (batch, 128, 16, 216)
        x = self.cnn(x)
        
        # Reshape for RNN: (batch, time, features)
        # (batch, channels, freq, time) -> (batch, time, channels * freq)
        x = x.permute(0, 3, 1, 2)  # (batch, time, channels, freq)
        x = x.reshape(batch_size, x.size(1), -1)  # (batch, time, channels * freq)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, time, rnn_hidden * 2)
        
        # Attention
        context, attention_weights = self.attention(lstm_out)  # (batch, rnn_hidden * 2)
        
        # Classification
        out = self.fc(context)
        
        return out
    
    def forward_with_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attention weights도 반환하는 forward"""
        batch_size = x.size(0)
        
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, x.size(1), -1)
        
        lstm_out, _ = self.lstm(x)
        context, attention_weights = self.attention(lstm_out)
        out = self.fc(context)
        
        return out, attention_weights


class TemporalAttention(nn.Module):
    """
    시간축 Attention 메커니즘
    중요한 시간 구간에 더 큰 가중치 부여
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: (batch, time, hidden_size)
        
        Returns:
            context: (batch, hidden_size) - Attention 적용된 context vector
            weights: (batch, time) - Attention weights
        """
        # Attention scores
        scores = self.attention(lstm_output)  # (batch, time, 1)
        scores = scores.squeeze(-1)  # (batch, time)
        
        # Attention weights (softmax)
        weights = F.softmax(scores, dim=1)  # (batch, time)
        
        # Weighted sum
        context = torch.bmm(
            weights.unsqueeze(1),  # (batch, 1, time)
            lstm_output  # (batch, time, hidden_size)
        ).squeeze(1)  # (batch, hidden_size)
        
        return context, weights


class SoundClassifierGRU(nn.Module):
    """
    GRU 버전 (더 빠른 학습, 더 적은 파라미터)
    """
    
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        cnn_channels: Optional[List[int]] = None,
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        if cnn_channels is None:
            cnn_channels = [32, 64, 128]
        
        # CNN Feature Extractor
        cnn_layers = []
        in_ch = in_channels
        for out_ch in cnn_channels:
            cnn_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1)),
                nn.Dropout2d(dropout / 2)
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        
        cnn_output_freq = 128 // (2 ** len(cnn_channels))
        rnn_input_size = cnn_channels[-1] * cnn_output_freq
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_num_layers > 1 else 0
        )
        
        # Classifier (last hidden state 사용)
        self.fc = nn.Sequential(
            nn.Linear(rnn_hidden_size * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, x.size(1), -1)
        
        _, hidden = self.gru(x)
        # Concatenate forward and backward hidden states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        out = self.fc(hidden)
        return out

