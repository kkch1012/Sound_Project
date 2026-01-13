"""
다른 컴퓨터에서 베스트 모델을 사용하기 위한 추론 스크립트

필요한 파일:
1. best_ensemble_model.pth (모델 가중치 파일)
2. 이 스크립트 (inference_model.py)
3. importance_mask.npy (선택사항, 없으면 기본값 사용)

사용 방법:
    python inference_model.py --audio_path "path/to/audio.wav"
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
    def __init__(self, num_classes: int, input_length: int = 110250, base_channels: int = 32, dropout: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        
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
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
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
    
    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1) 
            x = x * (1 + mask)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class CRNNMelSpectrogram(nn.Module):
    """
    멜 스펙트로그램을 위한 CRNN 모델
    CNN으로 주파수 축의 공간적 특징 추출 + RNN으로 시간 축의 시퀀스 패턴 학습
    """
    
    def __init__(
        self,
        num_classes: int,
        n_mels: int = 128,
        time_frames: int = 87,
        base_channels: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.4,
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
        self.cnn_output_features = base_channels * 4
        self.cnn_output_time = time_frames // 8  # 3번의 MaxPool2d(2,2) 적용
        self.cnn_output_freq = n_mels // 8
        
        # ============================================================
        # RNN 부분: 시간적 패턴 학습
        # ============================================================
        
        # CNN 출력을 RNN 입력 형태로 변환
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
        # Fully Connected 부분: 중간 특징 추출 (256차원 출력)
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
            output: 256차원 특징 벡터 (EnsembleVoteModel과 호환)
        """
        # 마스크 적용 (선택사항)
        if mask is not None:
            # 마스크 shape 처리:
            # - (1, n_mels, time_frames) -> (batch, 1, n_mels, time_frames)
            # - (n_mels, time_frames) -> (batch, 1, n_mels, time_frames)
            # - (batch, n_mels, time_frames) -> (batch, 1, n_mels, time_frames)
            if mask.dim() == 2:  # (n_mels, time_frames)
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, time_frames)
            elif mask.dim() == 3:
                if mask.size(0) == 1:  # (1, n_mels, time_frames) - importance_mask_tensor
                    mask = mask.unsqueeze(1)  # (1, 1, n_mels, time_frames)
                else:  # (batch, n_mels, time_frames)
                    mask = mask.unsqueeze(1)  # (batch, 1, n_mels, time_frames)
            
            # 배치 차원 확장 (브로드캐스팅)
            if mask.size(0) == 1 and x.size(0) > 1:
                mask = mask.expand(x.size(0), -1, -1, -1)
            
            # 마스크 적용: 중요 영역 강조
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
        
        # 마지막 시간 스텝의 출력 사용
        rnn_out = rnn_out[:, -1, :]  # (batch, hidden_size * directions)
        
        # ============================================================
        # Fully Connected: 중간 특징 추출 (256차원 출력)
        # ============================================================
        x = self.fc1(rnn_out)  # (batch, 256)
        # fc2와 fc_out은 EnsembleVoteModel의 final_fc에서 처리
        
        return x  # 256차원 출력 반환 (EnsembleVoteModel과 호환)


class MFCCCCNN(nn.Module):
    def __init__(self, num_classes: int = 4, num_mfcc_coeffs: int = 20, time_frames: int = 87, base_channels: int = 64, dropout: float = 0.4):
        super(MFCCCCNN, self).__init__()
        self.output_dim = 256
        
        self.conv1 = self._conv_block(1, base_channels, dropout / 3)
        self.conv2 = self._conv_block(base_channels, base_channels * 2, dropout / 3)
        self.conv3 = self._conv_block(base_channels * 2, base_channels * 4, dropout / 3)
        
        final_dim = self._calculate_final_dim(base_channels, num_mfcc_coeffs, time_frames)
        
        self.fc1 = nn.Sequential(
            nn.Linear(final_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
    def _conv_block(self, in_c, out_c, dropout):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout)
        )
        
    def _calculate_final_dim(self, base_channels, num_mfcc_coeffs, time_frames):
        try:
            with torch.no_grad():
                dummy_input = torch.randn(1, 1, num_mfcc_coeffs, time_frames)
                x = self.conv1(dummy_input)
                x = self.conv2(x)
                x = self.conv3(x)
                x = F.adaptive_avg_pool2d(x, 1)
                final_dim = x.numel() 
                return final_dim if final_dim > 0 else base_channels * 4 * 2 * 6
        except Exception as e:
            print(f"⚠️ MFCCCCNN 최종 차원 계산 실패: {e}. 기본값 사용.")
            return base_channels * 4 * 2 * 6

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.adaptive_avg_pool2d(x, 1) 
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        return x


class EnsembleVoteModel(nn.Module):
    def __init__(self, num_classes: int = 4, base_channels_wf: int = 32, base_channels_mel: int = 64, base_channels_mfcc: int = 64, num_mfcc_coeffs: int = 20, time_frames: int = 87):
        super(EnsembleVoteModel, self).__init__()
        
        wf_out_dim = 256
        mel_out_dim = 256
        mfcc_out_dim = 256
        
        self.wf_model = WaveformCNN1D(num_classes=wf_out_dim, base_channels=base_channels_wf) 
        # 🌟 CRNN 모델 사용 (기존 MaskedCNN 대신)
        self.mel_model = CRNNMelSpectrogram(
            num_classes=mel_out_dim,
            n_mels=128,
            time_frames=time_frames,
            base_channels=base_channels_mel,
            hidden_size=128,
            num_layers=2,
            dropout=0.4,
            rnn_type='LSTM',
            bidirectional=True
        )
        self.mfcc_model = MFCCCCNN(num_classes=mfcc_out_dim, num_mfcc_coeffs=num_mfcc_coeffs, time_frames=time_frames, base_channels=base_channels_mfcc) 
        
        total_input_dim = wf_out_dim + mel_out_dim + mfcc_out_dim
        self.final_fc = nn.Linear(total_input_dim, num_classes)
        
    def forward(self, wf, mel, mfcc, wf_mask, importance_mask=None): 
        wf_out = self.wf_model(wf, wf_mask)
        mel_out = self.mel_model(mel, importance_mask)
        mfcc_out = self.mfcc_model(mfcc)
        combined = torch.cat((wf_out, mel_out, mfcc_out), dim=1) 
        output = self.final_fc(combined)
        return output


# ============================================================
# 전처리 함수
# ============================================================

def extract_mfcc(y, sr, n_fft=2048, hop_length=512, n_mels=128):
    """MFCC 추출"""
    n_mfcc = 20
    mfccs = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        n_mfcc=n_mfcc
    )
    mfccs_db = librosa.power_to_db(mfccs, ref=np.max)
    return mfccs_db


def preprocess_audio(audio_path, sample_rate=22050, duration=2.0, n_mels=128, n_fft=2048, hop_length=512):
    """
    오디오 파일을 전처리하여 모델 입력 형태로 변환
    
    Returns:
        waveform: (1, length) 형태의 텐서
        mel_spec: (1, n_mels, time_frames) 형태의 텐서
        mfcc_spec: (1, n_mfcc, time_frames) 형태의 텐서
        waveform_mask: (1, length) 형태의 텐서
    """
    # 오디오 로드
    y, sr = librosa.load(str(audio_path), sr=sample_rate)
    
    # 1. Trimming (선행/후행 무음 구간 제거)
    y_trimmed, _ = librosa.effects.trim(y, top_db=60)
    
    # 2. Normalization (피크 정규화)
    y_normalized = librosa.util.normalize(y_trimmed)
    
    # 3. Waveform 길이 정규화
    target_length = int(sample_rate * duration)
    if len(y_normalized) < target_length:
        y_padded = np.pad(y_normalized, (0, target_length - len(y_normalized)), mode='constant')
    elif len(y_normalized) > target_length:
        y_padded = y_normalized[:target_length]
    else:
        y_padded = y_normalized
    
    # 4. Mel Spectrogram 추출
    mel_spec = librosa.feature.melspectrogram(
        y=y_normalized,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 5. Mel Spectrogram 시간 프레임 정규화
    target_time_frames = int(np.ceil(target_length / hop_length))
    current_time_frames = mel_spec_db.shape[1]
    
    if current_time_frames < target_time_frames:
        padding = np.zeros((mel_spec_db.shape[0], target_time_frames - current_time_frames))
        padding.fill(mel_spec_db.min())
        mel_spec_db = np.concatenate([mel_spec_db, padding], axis=1)
    elif current_time_frames > target_time_frames:
        mel_spec_db = mel_spec_db[:, :target_time_frames]
    
    # 6. MFCC 추출
    mfcc_spec_db = extract_mfcc(y_normalized, sr, n_fft, hop_length, n_mels)
    
    current_time_frames = mfcc_spec_db.shape[1]
    if current_time_frames < target_time_frames:
        padding = np.zeros((mfcc_spec_db.shape[0], target_time_frames - current_time_frames))
        padding.fill(mfcc_spec_db.min())
        mfcc_spec_db = np.concatenate([mfcc_spec_db, padding], axis=1)
    elif current_time_frames > target_time_frames:
        mfcc_spec_db = mfcc_spec_db[:, :target_time_frames]
    
    # 7. Waveform 어텐션 마스크 생성 (RMS 에너지 기반)
    rms = librosa.feature.rms(y=y_padded)[0]
    if rms.max() > rms.min():
        attention = (rms - rms.min()) / (rms.max() - rms.min())
    else:
        attention = np.ones_like(rms) * 0.5
    
    # 원본 waveform 길이에 맞추기 위해 보간
    target_length_wf = len(y_padded)
    if len(attention) != target_length_wf:
        from scipy.interpolate import interp1d
        x_old = np.linspace(0, 1, len(attention))
        x_new = np.linspace(0, 1, target_length_wf)
        f = interp1d(x_old, attention, kind='linear')
        attention = f(x_new)
    
    # 텐서로 변환
    waveform = torch.FloatTensor(y_padded).unsqueeze(0).unsqueeze(0)  # (1, 1, length)
    mel_spec = torch.FloatTensor(mel_spec_db).unsqueeze(0)  # (1, n_mels, time_frames)
    mfcc_spec = torch.FloatTensor(mfcc_spec_db).unsqueeze(0)  # (1, n_mfcc, time_frames)
    waveform_mask = torch.FloatTensor(attention).unsqueeze(0)  # (1, length)
    
    return waveform, mel_spec, mfcc_spec, waveform_mask


# ============================================================
# 추론 함수
# ============================================================

def load_model(model_path, device='cpu', num_classes=4):
    """모델 로드"""
    model = EnsembleVoteModel(
        num_classes=num_classes,
        base_channels_wf=32,
        base_channels_mel=64,
        base_channels_mfcc=64
    )
    
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
        predicted_label: 예측된 레이블 이름
        probabilities: 각 클래스에 대한 확률
    """
    # 레이블 매핑 (학습 시와 동일한 순서)
    idx_to_label = {
        0: 'low_oil',
        1: 'normal_engine_idle',
        2: 'power_steering',
        3: 'serpentine_belt'
    }
    
    # 모델 로드
    model = load_model(model_path, device, num_classes=len(idx_to_label))
    
    # 전처리
    waveform, mel_spec, mfcc_spec, waveform_mask = preprocess_audio(audio_path)
    waveform = waveform.to(device)
    mel_spec = mel_spec.to(device)
    mfcc_spec = mfcc_spec.to(device)
    waveform_mask = waveform_mask.to(device)
    
    # 중요 영역 마스크 로드 (없으면 None)
    if importance_mask_path and Path(importance_mask_path).exists():
        importance_mask = torch.FloatTensor(np.load(importance_mask_path)).unsqueeze(0).to(device)
    else:
        # 기본값: 모든 영역 동일한 중요도
        importance_mask = None
    
    # 예측
    with torch.no_grad():
        outputs = model(
            wf=waveform,
            mel=mel_spec,
            mfcc=mfcc_spec,
            wf_mask=waveform_mask,
            importance_mask=importance_mask
        )
        
        probabilities = F.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        predicted_label = idx_to_label[predicted_idx]
        
        prob_dict = {idx_to_label[i]: prob.item() for i, prob in enumerate(probabilities[0])}
    
    return predicted_label, prob_dict


# ============================================================
# 메인 함수
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Idle 상태 컬럼 분류 모델 추론')
    parser.add_argument('--audio_path', type=str, required=True, help='오디오 파일 경로')
    parser.add_argument('--model_path', type=str, default='best_ensemble_model.pth', help='모델 가중치 파일 경로')
    parser.add_argument('--importance_mask', type=str, default=None, help='중요 영역 마스크 파일 경로 (선택사항)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='사용할 디바이스')
    
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
        predicted_label, probabilities = predict(
            args.audio_path,
            args.model_path,
            device=device,
            importance_mask_path=args.importance_mask
        )
        
        print(f"\n✅ 예측 결과:")
        print(f"  예측된 클래스: {predicted_label}")
        print(f"\n📊 각 클래스별 확률:")
        for label, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            print(f"  {label}: {prob*100:.2f}%")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

