"""
CNN 기반 사운드 분류 모델
Mel Spectrogram을 이미지처럼 처리하여 분류
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ConvBlock(nn.Module):
    """Convolutional Block with BatchNorm and Dropout"""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool_size: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(pool_size)
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class SoundClassifierCNN(nn.Module):
    """
    CNN 기반 차량 사운드 분류기
    
    Architecture:
    - 4개의 Convolutional Blocks (점진적 채널 증가)
    - Global Average Pooling
    - Fully Connected Layers with Dropout
    - Multi-head output for hierarchical classification
    """
    
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        base_channels: int = 32,
        input_shape: Tuple[int, int] = (128, 216),
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_shape = input_shape
        
        # Convolutional layers
        self.conv1 = ConvBlock(in_channels, base_channels, dropout=dropout)
        self.conv2 = ConvBlock(base_channels, base_channels * 2, dropout=dropout)
        self.conv3 = ConvBlock(base_channels * 2, base_channels * 4, dropout=dropout)
        self.conv4 = ConvBlock(base_channels * 4, base_channels * 8, dropout=dropout)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        fc_input_size = base_channels * 8
        self.fc1 = nn.Linear(fc_input_size, 256)
        self.fc_bn1 = nn.BatchNorm1d(256)
        self.fc_dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(256, 128)
        self.fc_bn2 = nn.BatchNorm1d(128)
        self.fc_dropout2 = nn.Dropout(dropout)
        
        # Output layer
        self.fc_out = nn.Linear(128, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """가중치 초기화 (Conv: Kaiming, Linear: Xavier)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
               Expected shape: (batch, 1, 128, 216) for mel spectrogram
        
        Returns:
            Output tensor of shape (batch, num_classes)
        """
        # Convolutional blocks
        x = self.conv1(x)  # -> (batch, 32, 64, 108)
        x = self.conv2(x)  # -> (batch, 64, 32, 54)
        x = self.conv3(x)  # -> (batch, 128, 16, 27)
        x = self.conv4(x)  # -> (batch, 256, 8, 13)
        
        # Global Average Pooling
        x = self.global_pool(x)  # -> (batch, 256, 1, 1)
        x = x.view(x.size(0), -1)  # -> (batch, 256)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.fc_bn1(x)
        x = F.relu(x)
        x = self.fc_dropout1(x)
        
        x = self.fc2(x)
        x = self.fc_bn2(x)
        x = F.relu(x)
        x = self.fc_dropout2(x)
        
        # Output
        x = self.fc_out(x)
        
        return x
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """피처 추출 (마지막 FC 레이어 전까지)"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.fc_bn1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = self.fc_bn2(x)
        x = F.relu(x)
        
        return x


class SoundClassifierCNNDeep(nn.Module):
    """
    더 깊은 CNN 아키텍처 (ResNet 스타일 skip connections)
    """
    
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        base_channels: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Initial conv
        self.conv_init = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual blocks
        self.res_block1 = self._make_res_block(base_channels, base_channels, 2)
        self.res_block2 = self._make_res_block(base_channels, base_channels * 2, 2, stride=2)
        self.res_block3 = self._make_res_block(base_channels * 2, base_channels * 4, 2, stride=2)
        self.res_block4 = self._make_res_block(base_channels * 4, base_channels * 8, 2, stride=2)
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_channels * 8, num_classes)
        
    def _make_res_block(
        self, 
        in_channels: int, 
        out_channels: int, 
        num_blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """Residual block 생성"""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_init(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class ResidualBlock(nn.Module):
    """Residual Block with skip connection"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = F.relu(out)
        
        return out

