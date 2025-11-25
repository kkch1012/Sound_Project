"""
모델 평가 스크립트
학습된 모델의 성능을 상세히 평가
"""
import argparse
import sys
from pathlib import Path
import json
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.training.dataset import create_data_loaders
from app.ml.models import SoundClassifierCNN, SoundClassifierCRNN, SoundClassifierAttention


def parse_args():
    parser = argparse.ArgumentParser(description="모델 평가")
    parser.add_argument("--model_path", type=str, required=True,
                        help="모델 체크포인트 경로")
    parser.add_argument("--config_path", type=str, required=True,
                        help="설정 파일 경로")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="데이터 디렉토리")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="결과 저장 디렉토리")
    return parser.parse_args()


def evaluate_model(model, data_loader, device, idx_to_label):
    """모델 평가"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    """혼동 행렬 시각화"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"혼동 행렬 저장: {output_path}")


def plot_class_accuracy(y_true, y_pred, labels, output_path):
    """클래스별 정확도 시각화"""
    class_acc = {}
    for i, label in enumerate(labels):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_true[mask]).mean() * 100
            class_acc[label] = acc
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(len(class_acc)), list(class_acc.values()))
    plt.xticks(range(len(class_acc)), list(class_acc.keys()), rotation=45, ha='right')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Accuracy')
    plt.ylim(0, 100)
    
    # 값 표시
    for bar, acc in zip(bars, class_acc.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"클래스별 정확도 저장: {output_path}")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 설정 로드
    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    num_classes = config["num_classes"]
    idx_to_label = config["idx_to_label"]
    model_type = config["model_type"]
    input_shape = config.get("input_shape", [128, 216])
    
    # 레이블 목록
    labels = [idx_to_label[str(i)] for i in range(num_classes)]
    
    print(f"모델 타입: {model_type}")
    print(f"클래스 수: {num_classes}")
    
    # 모델 생성 및 로드
    if model_type == "cnn":
        model = SoundClassifierCNN(num_classes=num_classes, input_shape=tuple(input_shape))
    elif model_type == "crnn":
        model = SoundClassifierCRNN(num_classes=num_classes)
    elif model_type == "attention":
        model = SoundClassifierAttention(num_classes=num_classes, input_shape=tuple(input_shape))
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print("모델 로드 완료")
    
    # 데이터 로더 생성
    _, _, test_loader, _ = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=0.15,
        test_split=0.15,
        num_workers=4,
        target_shape=tuple(input_shape)
    )
    
    # 평가
    print("\n평가 중...")
    preds, labels_true, probs = evaluate_model(model, test_loader, device, idx_to_label)
    
    # 전체 정확도
    accuracy = (preds == labels_true).mean() * 100
    print(f"\n전체 정확도: {accuracy:.2f}%")
    
    # 분류 보고서
    report = classification_report(
        labels_true, preds, 
        target_names=labels,
        digits=4
    )
    print("\n분류 보고서:")
    print(report)
    
    # 보고서 저장
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"전체 정확도: {accuracy:.2f}%\n\n")
        f.write("분류 보고서:\n")
        f.write(report)
    print(f"보고서 저장: {report_path}")
    
    # 시각화
    plot_confusion_matrix(
        labels_true, preds, labels,
        output_dir / "confusion_matrix.png"
    )
    
    plot_class_accuracy(
        labels_true, preds, labels,
        output_dir / "class_accuracy.png"
    )
    
    print("\n평가 완료!")


if __name__ == "__main__":
    main()

