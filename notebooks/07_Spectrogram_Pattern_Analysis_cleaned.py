"""
노트북 정리 스크립트: UnifiedMultiHeadModel 관련 코드 제거
"""
import json
from pathlib import Path

def clean_notebook():
    """UnifiedMultiHeadModel 관련 셀 제거"""
    input_path = Path("notebooks/07_Spectrogram_Pattern_Analysis.ipynb")
    output_path = Path("notebooks/07_Spectrogram_Pattern_Analysis_cleaned.ipynb")
    
    # 노트북 로드
    with open(input_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # 제거할 셀 인덱스 찾기
    cells_to_remove = []
    
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            
            # UnifiedMultiHeadModel 관련 셀 찾기 (WithMasking 제외)
            if 'class UnifiedMultiHeadModel(' in source and 'UnifiedMultiHeadModelWithMasking' not in source:
                cells_to_remove.append(i)
                print(f"제거할 셀 {i}: 클래스 정의")
            elif 'unified_model = UnifiedMultiHeadModel(' in source:
                cells_to_remove.append(i)
                print(f"제거할 셀 {i}: 모델 생성")
            elif 'UnifiedMultiHeadModel(' in source and 'UnifiedMultiHeadModelWithMasking' not in source and 'loaded_model = UnifiedMultiHeadModel' in source:
                cells_to_remove.append(i)
                print(f"제거할 셀 {i}: 모델 로드")
        
        elif cell['cell_type'] == 'markdown':
            source = ''.join(cell.get('source', []))
            # UnifiedMultiHeadModel 관련 마크다운 (WithMasking 제외)
            if '## 16. 통합 멀티헤드 모델' in source and 'WithMasking' not in source and 'MultiModal' not in source:
                # 마크다운은 제거하지 않고 주석만 추가
                pass
    
    # 역순으로 제거 (인덱스가 변경되지 않도록)
    for idx in sorted(cells_to_remove, reverse=True):
        removed_cell = notebook['cells'].pop(idx)
        print(f"✅ 셀 {idx} 제거 완료")
    
    # 정리된 노트북 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"\n✅ 정리 완료!")
    print(f"   원본: {input_path}")
    print(f"   정리본: {output_path}")
    print(f"   제거된 셀 수: {len(cells_to_remove)}개")
    print(f"   남은 셀 수: {len(notebook['cells'])}개")

if __name__ == "__main__":
    clean_notebook()
