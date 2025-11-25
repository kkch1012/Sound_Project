"""
모델 학습 모듈
학습, 검증, 조기 종료, 모델 저장 등
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
from typing import Optional, Dict, List, Tuple, Any, Union
from pathlib import Path
import json
from tqdm import tqdm


class EarlyStopping:
    """조기 종료 클래스"""
    
    def __init__(
        self, 
        patience: int = 10, 
        min_delta: float = 0.001,
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
    
    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta


class Trainer:
    """
    딥러닝 모델 학습기
    
    Features:
    - Mixed precision training (AMP)
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - Training history logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
        save_dir: str = "checkpoints",
        experiment_name: str = "experiment",
        use_amp: bool = True,
        class_weights: Optional[torch.Tensor] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.experiment_name = experiment_name
        self.use_amp = use_amp and device == "cuda"
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # 클래스 가중치 (불균형 데이터용)
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        
        # 학습 히스토리
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": []
        }
        
        # 저장 디렉토리 생성
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def train_epoch(self) -> Tuple[float, float]:
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp and self.scaler is not None:
                with autocast(device_type='cuda'):
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.*correct/total:.2f}%"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """검증"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in self.val_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        epochs: int = 100,
        early_stopping_patience: int = 15,
        save_best: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        모델 학습
        
        Args:
            epochs: 총 에포크 수
            early_stopping_patience: 조기 종료 인내심
            save_best: 최고 모델 저장 여부
            verbose: 상세 출력 여부
        
        Returns:
            학습 히스토리
        """
        early_stopping = EarlyStopping(
            patience=early_stopping_patience, 
            mode="min"
        )
        best_val_loss = float("inf")
        
        print(f"\n{'='*60}")
        print(f"Starting training: {self.experiment_name}")
        print(f"Device: {self.device}")
        print(f"Total epochs: {epochs}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            # 학습
            train_loss, train_acc = self.train_epoch()
            
            # 검증
            val_loss, val_acc = self.validate()
            
            # 학습률 스케줄링
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None and hasattr(self.scheduler, 'step'):
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 히스토리 기록
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)
            
            if verbose:
                print(f"Epoch [{epoch+1}/{epochs}]")
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
                print(f"  LR: {current_lr:.6f}")
            
            # 최고 모델 저장
            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(
                    epoch, 
                    is_best=True,
                    val_loss=val_loss,
                    val_acc=val_acc
                )
                if verbose:
                    print(f"  ✓ Best model saved!")
            
            # 조기 종료 확인
            if early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
            
            print()
        
        # 최종 결과 저장
        self._save_history()
        
        return self.history
    
    def save_checkpoint(
        self, 
        epoch: int, 
        is_best: bool = False,
        **kwargs
    ):
        """체크포인트 저장"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            **kwargs
        }
        
        if self.scheduler is not None and hasattr(self.scheduler, 'state_dict'):
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        filename = "best_model.pt" if is_best else f"checkpoint_epoch_{epoch}.pt"
        filepath = self.save_dir / f"{self.experiment_name}_{filename}"
        torch.save(checkpoint, filepath)
        
    def load_checkpoint(self, filepath: str):
        """체크포인트 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None and hasattr(self.scheduler, 'load_state_dict'):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if "history" in checkpoint:
            self.history = checkpoint["history"]
        
        return checkpoint.get("epoch", 0)
    
    def _save_history(self):
        """학습 히스토리 저장"""
        history_path = self.save_dir / f"{self.experiment_name}_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)


class MultiTaskTrainer(Trainer):
    """
    멀티태스크 학습기
    상태 분류 + 문제 분류 동시 학습
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        state_criterion: nn.Module,
        problem_criterion: nn.Module,
        optimizer: optim.Optimizer,
        state_weight: float = 0.3,
        problem_weight: float = 0.7,
        **kwargs
    ):
        super().__init__(
            model, train_loader, val_loader,
            criterion=problem_criterion,
            optimizer=optimizer,
            **kwargs
        )
        self.state_criterion = state_criterion
        self.problem_criterion = problem_criterion
        self.state_weight = state_weight
        self.problem_weight = problem_weight
        
    def train_epoch(self) -> Tuple[float, float]:
        """멀티태스크 학습"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in tqdm(self.train_loader, desc="Training"):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # batch_y는 (state_idx, problem_idx) 튜플이어야 함
            # 여기서는 간단히 problem_idx만 사용
            
            self.optimizer.zero_grad()
            
            outputs = self.model(batch_x)
            
            # 단일 태스크로 처리 (필요시 멀티태스크로 확장)
            loss = self.criterion(outputs, batch_y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
        
        return total_loss / len(self.train_loader), 100. * correct / total


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    lr: float = 1e-3,
    weight_decay: float = 0.01
) -> optim.Optimizer:
    """옵티마이저 생성"""
    optimizers = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD,
        "rmsprop": optim.RMSprop
    }
    
    optimizer_cls = optimizers.get(optimizer_name.lower(), optim.AdamW)
    
    if optimizer_name.lower() == "sgd":
        return optimizer_cls(
            model.parameters(), 
            lr=lr, 
            momentum=0.9, 
            weight_decay=weight_decay
        )
    
    return optimizer_cls(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
    )


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str = "cosine",
    epochs: int = 100,
    **kwargs
) -> Optional[Any]:
    """학습률 스케줄러 생성"""
    if scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs,
            eta_min=1e-6
        )
    elif scheduler_name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 30),
            gamma=kwargs.get("gamma", 0.1)
        )
    elif scheduler_name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    elif scheduler_name == "warmup_cosine":
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=kwargs.get("warmup_epochs", 5),
            total_epochs=epochs
        )
    return None


class WarmupCosineScheduler:
    """Warmup + Cosine Annealing 스케줄러"""
    
    def __init__(
        self, 
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.current_epoch = 0
        
    def step(self):
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * self.current_epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
            
    def state_dict(self):
        return {"current_epoch": self.current_epoch}
    
    def load_state_dict(self, state_dict):
        self.current_epoch = state_dict["current_epoch"]

