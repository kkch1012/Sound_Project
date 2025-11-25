"""
Self-Attention / Transformer 기반 사운드 분류 모델
Audio Spectrogram Transformer (AST) 스타일
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding
    시퀀스의 위치 정보를 인코딩
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self Attention
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int = 8, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final linear projection
        output = self.W_o(context)
        
        return output, attention_weights


class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block
    Self-Attention + Feed Forward Network
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int = 8, 
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        # Self-attention with residual
        attn_out, attention_weights = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x, attention_weights


class PatchEmbedding(nn.Module):
    """
    스펙트로그램을 패치로 분할하고 임베딩
    Vision Transformer 스타일
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        patch_size: Tuple[int, int] = (16, 16),
        d_model: int = 256,
        input_shape: Tuple[int, int] = (128, 216)
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.d_model = d_model
        
        # 패치 수 계산
        self.num_patches_h = input_shape[0] // patch_size[0]
        self.num_patches_w = input_shape[1] // patch_size[1]
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # 패치 임베딩 (Conv2d로 구현)
        self.patch_embed = nn.Conv2d(
            in_channels, 
            d_model, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # CLS 토큰
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, d_model)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, height, width)
        
        Returns:
            (batch, num_patches + 1, d_model)
        """
        batch_size = x.size(0)
        
        # 패치 임베딩
        x = self.patch_embed(x)  # (batch, d_model, h_patches, w_patches)
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, d_model)
        
        # CLS 토큰 추가
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, num_patches + 1, d_model)
        
        # Positional embedding
        x = x + self.pos_embed
        
        return x


class SoundClassifierAttention(nn.Module):
    """
    Audio Spectrogram Transformer (AST) 스타일 분류기
    
    Architecture:
    - Patch Embedding: 스펙트로그램을 패치로 분할
    - Transformer Encoder: Self-attention으로 패턴 학습
    - Classification Head: CLS 토큰으로 분류
    """
    
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        input_shape: Tuple[int, int] = (128, 216),
        patch_size: Tuple[int, int] = (16, 16),
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            d_model=d_model,
            input_shape=input_shape
        )
        
        # Positional Encoding (추가적인 시간 정보)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer Norm
        self.norm = nn.LayerNorm(d_model)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input spectrogram (batch, channels, freq, time)
        
        Returns:
            Class logits (batch, num_classes)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (batch, num_patches + 1, d_model)
        
        # Transformer encoder
        attention_weights_list = []
        for encoder_layer in self.encoder_layers:
            x, attn_weights = encoder_layer(x)
            attention_weights_list.append(attn_weights)
        
        x = self.norm(x)
        
        # CLS 토큰으로 분류
        cls_output = x[:, 0]  # (batch, d_model)
        
        # Classification
        out = self.classifier(cls_output)
        
        return out
    
    def forward_with_attention(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, list]:
        """Attention weights도 반환"""
        x = self.patch_embed(x)
        
        attention_weights_list = []
        for encoder_layer in self.encoder_layers:
            x, attn_weights = encoder_layer(x)
            attention_weights_list.append(attn_weights)
        
        x = self.norm(x)
        cls_output = x[:, 0]
        out = self.classifier(cls_output)
        
        return out, attention_weights_list


class SoundClassifierConformer(nn.Module):
    """
    Conformer 스타일 모델
    Convolution + Transformer 결합
    """
    
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 입력 프로젝션
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 1))  # Freq 축만 다운샘플
        )
        
        # Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(d_model, num_heads, conv_kernel_size, dropout)
            for _ in range(num_layers)
        ])
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, 1, 128, 216)
        batch_size = x.size(0)
        
        # Conv feature extraction
        x = self.input_proj(x)  # (batch, d_model, 32, 216)
        
        # Reshape: (batch, d_model, freq, time) -> (batch, time, d_model * freq)
        x = x.permute(0, 3, 1, 2)  # (batch, time, d_model, freq)
        x = x.reshape(batch_size, x.size(1), -1)  # (batch, time, d_model * freq)
        
        # Project to d_model
        # 여기서는 간단하게 평균을 사용
        x = x.view(batch_size, x.size(1), -1, 32).mean(dim=-1)  # (batch, time, d_model)
        
        # Conformer blocks
        for block in self.conformer_blocks:
            x = block(x)
        
        # Global pooling
        x = x.transpose(1, 2)  # (batch, d_model, time)
        x = self.global_pool(x).squeeze(-1)  # (batch, d_model)
        
        # Classification
        out = self.classifier(x)
        
        return out


class ConformerBlock(nn.Module):
    """
    Conformer Block: FFN + Self-Attention + Conv + FFN
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int,
        conv_kernel_size: int = 31,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # First FFN (half)
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        # Multi-head self-attention
        self.attn_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_dropout = nn.Dropout(dropout)
        
        # Convolution module
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(
                d_model, d_model, 
                kernel_size=conv_kernel_size, 
                padding=conv_kernel_size // 2,
                groups=d_model
            ),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )
        
        # Second FFN (half)
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        # First FFN (half step)
        x = x + 0.5 * self.ffn1(x)
        
        # Self-attention
        residual = x
        x = self.attn_norm(x)
        x, _ = self.attention(x, x, x)
        x = residual + self.attn_dropout(x)
        
        # Convolution
        residual = x
        x = self.conv_norm(x)
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (batch, seq_len, d_model)
        x = residual + x
        
        # Second FFN (half step)
        x = x + 0.5 * self.ffn2(x)
        
        x = self.final_norm(x)
        
        return x

