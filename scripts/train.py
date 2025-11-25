"""
모델 학습 스크립트
사용법: python scripts/train.py --config configs/train_config.yaml
"""
import argparse
import sys
from pathlib import Path
import json
import torch
import torch.nn as nn

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.training.dataset import create_data_loaders
from app.ml.training.trainer import Trainer, create_optimizer, create_scheduler
from app.ml.models import SoundClassifierCNN, SoundClassifierCRNN, SoundClassifierAttention


def parse_args():
    parser = argparse.ArgumentParser(description="차량 사운드 분류 모델 학습")
    
    # 데이터 설정
    parser.add_argument("--data_dir", type=str, default="data",
                        help="데이터 디렉토리 경로")
    parser.add_argument("--val_split", type=float, default=0.15,
                        help="검증 데이터 비율")
    parser.add_argument("--test_split", type=float, default=0.15,
                        help="테스트 데이터 비율")
    
    # 모델 설정
    parser.add_argument("--model_type", type=str, default="cnn",
                        choices=["cnn", "crnn", "attention"],
                        help="모델 타입")
    parser.add_argument("--input_shape", type=int, nargs=2, default=[128, 216],
                        help="입력 스펙트로그램 크기 (height width)")
    
    # 학습 설정
    parser.add_argument("--epochs", type=int, default=100,
                        help="총 에포크 수")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="배치 크기")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="학습률")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="가중치 감쇠")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adam", "adamw", "sgd"],
                        help="옵티마이저")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "step", "plateau", "warmup_cosine"],
                        help="학습률 스케줄러")
    parser.add_argument("--early_stopping", type=int, default=15,
                        help="조기 종료 인내심")
    
    # 기타 설정
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="체크포인트 저장 디렉토리")
    parser.add_argument("--experiment_name", type=str, default="sound_classifier",
                        help="실험 이름")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="데이터 로더 워커 수")
    parser.add_argument("--seed", type=int, default=42,
                        help="랜덤 시드")
    parser.add_argument("--device", type=str, default="auto",
                        help="학습 디바이스 (auto, cuda, cpu)")
    parser.add_argument("--use_class_weights", action="store_true",
                        help="클래스 가중치 사용 (불균형 데이터용)")
    
    return parser.parse_args()


def create_model(model_type: str, num_classes: int, input_shape: tuple) -> nn.Module:
    """모델 생성"""
    if model_type == "cnn":
        model = SoundClassifierCNN(
            num_classes=num_classes,
            input_shape=tuple(input_shape)
        )
    elif model_type == "crnn":
        model = SoundClassifierCRNN(
            num_classes=num_classes
        )
    elif model_type == "attention":
        model = SoundClassifierAttention(
            num_classes=num_classes,
            input_shape=tuple(input_shape)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def main():
    args = parse_args()
    
    # 시드 설정
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 디바이스 설정
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"\n{'='*60}")
    print("차량 사운드 분류 모델 학습")
    print(f"{'='*60}")
    print(f"디바이스: {device}")
    print(f"모델: {args.model_type}")
    print(f"데이터: {args.data_dir}")
    print(f"{'='*60}\n")
    
    # 데이터 로더 생성
    print("데이터 로딩 중...")
    train_loader, val_loader, test_loader, dataset_info = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        num_workers=args.num_workers,
        target_shape=tuple(args.input_shape),
        seed=args.seed
    )
    
    num_classes = dataset_info["num_classes"]
    print(f"클래스 수: {num_classes}")
    print(f"학습 샘플: {dataset_info['train_size']}")
    print(f"검증 샘플: {dataset_info['val_size']}")
    print(f"테스트 샘플: {dataset_info['test_size']}")
    
    # 모델 생성
    print(f"\n모델 생성 중... ({args.model_type})")
    model = create_model(args.model_type, num_classes, args.input_shape)
    
    # 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"총 파라미터: {total_params:,}")
    print(f"학습 가능 파라미터: {trainable_params:,}")
    
    # 손실 함수
    criterion = nn.CrossEntropyLoss()
    
    # 옵티마이저
    optimizer = create_optimizer(
        model, 
        optimizer_name=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 스케줄러
    scheduler = create_scheduler(
        optimizer,
        scheduler_name=args.scheduler,
        epochs=args.epochs
    )
    
    # 클래스 가중치
    class_weights = None
    if args.use_class_weights:
        class_weights = dataset_info["class_weights"]
        print("클래스 가중치 사용: 활성화")
    
    # 트레이너 생성
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=args.save_dir,
        experiment_name=args.experiment_name,
        use_amp=(device == "cuda"),
        class_weights=class_weights
    )
    
    # 학습 시작
    print("\n학습 시작...")
    history = trainer.train(
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping,
        save_best=True,
        verbose=True
    )
    
    # 테스트 평가
    print("\n테스트 데이터 평가 중...")
    trainer.val_loader = test_loader
    test_loss, test_acc = trainer.validate()
    print(f"테스트 Loss: {test_loss:.4f}")
    print(f"테스트 Accuracy: {test_acc:.2f}%")
    
    # 설정 저장
    config_path = Path(args.save_dir) / f"{args.experiment_name}_config.json"
    config = {
        "num_classes": num_classes,
        "label_to_idx": dataset_info["label_to_idx"],
        "idx_to_label": dataset_info["idx_to_label"],
        "state_labels": dataset_info["state_labels"],
        "model_type": args.model_type,
        "input_shape": args.input_shape,
        "test_accuracy": test_acc,
        "test_loss": test_loss
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n설정 저장됨: {config_path}")
    print(f"모델 저장됨: {args.save_dir}/{args.experiment_name}_best_model.pt")
    print("\n학습 완료!")


if __name__ == "__main__":
    main()

