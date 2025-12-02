"""
스펙트로그램을 사용한 PCA 시각화 및 군집 분석 예제

이 스크립트는 Mel Spectrogram을 flatten하여 PCA와 K-means 클러스터링을 수행하는 방법을 보여줍니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, 
    normalized_mutual_info_score
)
import sys
sys.path.insert(0, '..')

from app.ml.features.extractor import AudioFeatureExtractor, AudioConfig

# ============================================================
# 1. 데이터 로드 및 스펙트로그램 추출
# ============================================================

# 피처 추출기 초기화
config = AudioConfig()
feature_extractor = AudioFeatureExtractor(config)

# 데이터 경로
data_dir = Path('../data')
all_files = []
all_states = []

# 파일 수집 (예시)
for state_dir in sorted(data_dir.iterdir()):
    if not state_dir.is_dir() or state_dir.name == 'augmented':
        continue
    
    state_name = state_dir.name
    state_idx = {'braking state': 0, 'idle state': 1, 'startup state': 2}.get(state_name, -1)
    
    if state_idx == -1:
        continue
    
    for problem_dir in state_dir.iterdir():
        if not problem_dir.is_dir() or problem_dir.name == 'combined':
            continue
        
        for file_path in problem_dir.glob('*.wav'):
            all_files.append(file_path)
            all_states.append(state_idx)

print(f"✅ 총 {len(all_files)}개 파일 수집 완료!")

# ============================================================
# 2. 스펙트로그램 추출 및 Flatten
# ============================================================

print("\n🔄 스펙트로그램 추출 중...")

all_spectrograms = []

for file_path in all_files:
    try:
        # Mel Spectrogram 추출 (2D 배열: frequency × time)
        mel_spec = feature_extractor.extract_mel_spectrogram(
            *feature_extractor.load_audio(str(file_path))
        )
        
        # Flatten: 2D → 1D 벡터로 변환
        # 예: (128, 216) → (27648,)
        mel_spec_flat = mel_spec.flatten()
        
        all_spectrograms.append(mel_spec_flat)
        
    except Exception as e:
        print(f"⚠️  오류: {file_path} - {e}")
        # 오류 시 0으로 채운 벡터 추가
        all_spectrograms.append(np.zeros(128 * 216))

# NumPy 배열로 변환: (samples, features)
X = np.array(all_spectrograms)
y_states = np.array(all_states)

print(f"✅ 스펙트로그램 추출 완료!")
print(f"   X shape: {X.shape}")  # 예: (2832, 27648)
print(f"   y_states shape: {y_states.shape}")  # 예: (2832,)

# ============================================================
# 3. 데이터 정규화
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n✅ 데이터 정규화 완료!")

# ============================================================
# 4. PCA 수행 (차원 축소)
# ============================================================

print("\n🔄 PCA 수행 중...")

# 2D PCA (시각화용)
pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_scaled)

# 3D PCA (시각화용)
pca_3d = PCA(n_components=3, random_state=42)
X_pca_3d = pca_3d.fit_transform(X_scaled)

print(f"✅ PCA 완료!")
print(f"   2D 설명 분산: {pca_2d.explained_variance_ratio_.sum():.4f}")
print(f"   3D 설명 분산: {pca_3d.explained_variance_ratio_.sum():.4f}")

# ============================================================
# 5. PCA 2D 시각화
# ============================================================

state_names = ['braking', 'idle', 'startup']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
markers = ['o', 's', '^']

fig, ax = plt.subplots(figsize=(12, 10))

for state_idx, (name, color, marker) in enumerate(zip(state_names, colors, markers)):
    mask = y_states == state_idx
    ax.scatter(
        X_pca_2d[mask, 0], 
        X_pca_2d[mask, 1], 
        c=color, 
        marker=marker,
        label=f'{name} (n={mask.sum()})',
        alpha=0.6,
        s=50
    )

ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%})', fontsize=12)
ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%})', fontsize=12)
ax.set_title('🎯 스펙트로그램 기반 PCA 2D 시각화', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spectrogram_pca_2d.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# 6. K-means 클러스터링
# ============================================================

print("\n🔄 K-Means 클러스터링 수행 중...")

# K=3으로 클러스터링 (3개 상태에 맞춤)
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(X_scaled)

print(f"✅ K-Means 클러스터링 완료!")

# ============================================================
# 7. 클러스터링 성능 평가
# ============================================================

silhouette = silhouette_score(X_scaled, cluster_labels)
ari = adjusted_rand_score(y_states, cluster_labels)
nmi = normalized_mutual_info_score(y_states, cluster_labels)

print("\n" + "=" * 60)
print("📊 클러스터링 성능 평가")
print("=" * 60)
print(f"\n1️⃣ Silhouette Score: {silhouette:.4f}")
print("   (-1 ~ 1, 높을수록 군집이 잘 분리됨)")
print(f"\n2️⃣ Adjusted Rand Index (ARI): {ari:.4f}")
print("   (0 ~ 1, 높을수록 실제 레이블과 일치)")
print(f"\n3️⃣ Normalized Mutual Information (NMI): {nmi:.4f}")
print("   (0 ~ 1, 높을수록 실제 레이블과 일치)")

# ============================================================
# 8. 클러스터링 결과 시각화
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 실제 레이블
for state_idx, (name, color, marker) in enumerate(zip(state_names, colors, markers)):
    mask = y_states == state_idx
    axes[0].scatter(
        X_pca_2d[mask, 0], X_pca_2d[mask, 1],
        c=color, marker=marker,
        label=name, alpha=0.6, s=50
    )

axes[0].set_xlabel('PC1', fontsize=12)
axes[0].set_ylabel('PC2', fontsize=12)
axes[0].set_title('🔵 실제 레이블', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 클러스터링 결과
cluster_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
for cluster_id in range(3):
    mask = cluster_labels == cluster_id
    axes[1].scatter(
        X_pca_2d[mask, 0], X_pca_2d[mask, 1],
        c=cluster_colors[cluster_id], marker='o',
        label=f'Cluster {cluster_id}', alpha=0.6, s=50
    )

# 클러스터 중심점 표시
centers_pca = pca_2d.transform(kmeans.cluster_centers_)
axes[1].scatter(
    centers_pca[:, 0], centers_pca[:, 1],
    c='black', marker='X', s=200, edgecolors='white', linewidths=2,
    label='Centroids'
)

axes[1].set_xlabel('PC1', fontsize=12)
axes[1].set_ylabel('PC2', fontsize=12)
axes[1].set_title('🔵 K-Means 클러스터링 결과', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('📊 실제 레이블 vs K-Means 클러스터링', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('spectrogram_clustering_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ 스펙트로그램 기반 PCA 및 군집 분석 완료!")
print("   시각화 이미지 저장: spectrogram_pca_2d.png, spectrogram_clustering_comparison.png")

