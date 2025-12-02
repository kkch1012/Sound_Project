# Notebooks 폴더 구조

이 폴더에는 차량 사운드 분석 및 모델 학습을 위한 Jupyter 노트북들이 포함되어 있습니다.

## 📁 파일 구조

```
notebooks/
├── utils.py                          # 공통 유틸리티 모듈
├── 01_EDA.ipynb                      # 탐색적 데이터 분석
├── 02_Data_Augmentation.ipynb       # 데이터 증강
├── 03_Model_Training.ipynb          # 모델 학습
├── 04_Feature_Clustering_CRNN.ipynb # CRNN 피처 군집 분석
├── 05_Combined_Feature_Clustering.ipynb # 결합 피처 군집 분석
├── 06_Spectrogram_CNN_Classification.ipynb # 스펙트로그램 CNN 분류
├── 07_Spectrogram_Pattern_Analysis.ipynb # 스펙트로그램 패턴 분석
├── 08_Hierarchical_Classification.ipynb # 계층적 분류
└── example_spectrogram_clustering.py # 스펙트로그램 클러스터링 예제 스크립트
```

## 🔧 공통 유틸리티 사용

모든 노트북에서 공통으로 사용하는 기능은 `utils.py`에 정의되어 있습니다.

### 사용 예시

```python
# 노트북 시작 부분에 추가
from utils import (
    setup_plotting,
    translate_state,
    translate_problem,
    get_data_dir,
    get_state_mapping,
    get_state_names
)

# 시각화 설정
setup_plotting()

# 데이터 경로 가져오기
data_dir = get_data_dir()

# 번역 함수 사용
korean_state = translate_state('braking state')  # '브레이크 상태'
```

## 📋 노트북 실행 순서

1. **01_EDA.ipynb**: 데이터 구조 및 분포 확인
2. **02_Data_Augmentation.ipynb**: 클래스 불균형 해결을 위한 데이터 증강
3. **03_Model_Training.ipynb**: 모델 학습 및 평가
4. **04_Feature_Clustering_CRNN.ipynb**: CRNN 기반 피처 군집 분석
5. **05_Combined_Feature_Clustering.ipynb**: 결합 피처 군집 분석
6. **06_Spectrogram_CNN_Classification.ipynb**: 스펙트로그램 기반 CNN 분류
7. **07_Spectrogram_Pattern_Analysis.ipynb**: 스펙트로그램 패턴 분석
8. **08_Hierarchical_Classification.ipynb**: 계층적 분류 모델

## 📝 참고사항

- 모든 노트북은 `../data` 디렉토리의 데이터를 사용합니다.
- `utils.py`의 공통 함수를 사용하면 코드 중복을 줄일 수 있습니다.
- `example_spectrogram_clustering.py`는 스펙트로그램 클러스터링의 예제 스크립트입니다.

