"""
Notebooks 공통 유틸리티 모듈
모든 노트북에서 공통으로 사용하는 함수와 설정
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import warnings

# 상위 디렉토리를 path에 추가 (app 모듈 사용을 위해)
# 노트북이 notebooks 폴더에서 실행될 때만 추가
notebooks_dir = Path(__file__).parent
project_root = notebooks_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

warnings.filterwarnings('ignore')  # 경고 메시지 숨김

# ============================================================
# 시각화 스타일 설정
# ============================================================
def setup_plotting():
    """시각화 스타일 설정"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 영어 → 한글 번역 딕셔너리
# ============================================================

# 상태(State)
STATE_KO = {
    'braking state': '브레이크 상태',
    'idle state': '공회전 상태',
    'startup state': '시동 상태',
}

# 문제(Problem)
PROBLEM_KO = {
    # 브레이크 관련
    'normal_brakes': '정상 브레이크',
    'worn_out_brakes': '마모된 브레이크',
    
    # 공회전 관련
    'normal_engine_idle': '정상 공회전',
    'low_oil': '오일 부족',
    'no oil': '오일 없음',
    'exhaust_leak': '배기 누출',
    'misfire': '점화 실패',
    'vacuum_leak': '진공 누출',
    'power_steering': '파워스티어링 이상',
    'power steering combined': '파워스티어링 복합',
    'serpentine_belt': '구동벨트 이상',
    'no oil_serpentine belt': '오일없음+구동벨트',
    'power steering combined_serpentine belt': '파워스티어링+구동벨트',
    'power steering combined_no oil': '파워스티어링+오일없음',
    'power steering combined_no oil_serpentine belt': '파워스티어링+오일없음+구동벨트',
    'combined': '복합 이상',
    
    # 시동 관련
    'normal_engine_startup': '정상 시동',
    'bad_ignition': '점화 불량',
    'dead_battery': '배터리 방전',
    'bad_starter': '스타터 불량',
    'fuel_pump_issue': '연료펌프 문제',
}

def translate_state(eng_name: str) -> str:
    """영어 상태명을 한글로 번역"""
    return STATE_KO.get(eng_name, eng_name)

def translate_problem(eng_name: str) -> str:
    """영어 문제명을 한글로 번역"""
    # 슬래시가 있는 경우 (예: "combined/normal")
    if '/' in eng_name:
        parts = eng_name.split('/')
        translated = [PROBLEM_KO.get(p, p) for p in parts]
        return '/'.join(str(t) for t in translated)
    return PROBLEM_KO.get(eng_name, eng_name)

# ============================================================
# 데이터 경로 설정
# ============================================================
def get_data_dir() -> Path:
    """데이터 디렉토리 경로 반환"""
    return Path('../data')

def get_state_mapping() -> dict:
    """상태 매핑 딕셔너리 반환"""
    return {
        'braking state': 0,
        'idle state': 1,
        'startup state': 2
    }

def get_state_names() -> list:
    """상태 이름 리스트 반환"""
    return ['braking', 'idle', 'startup']

