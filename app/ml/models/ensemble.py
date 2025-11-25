"""
앙상블 모델
여러 모델의 예측을 결합하여 더 강건한 분류
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple


class EnsembleClassifier(nn.Module):
    """
    여러 모델을 앙상블하는 분류기
    
    앙상블 방법:
    - Voting: 각 모델의 예측을 투표
    - Averaging: 확률을 평균
    - Weighted: 가중 평균
    - Stacking: 메타 분류기 사용
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        num_classes: int,
        ensemble_method: str = "weighted",
        weights: Optional[List[float]] = None
    ):
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.num_classes = num_classes
        self.ensemble_method = ensemble_method
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = nn.Parameter(
            torch.tensor(weights), 
            requires_grad=(ensemble_method == "learned_weighted")
        )
        
        # Stacking용 메타 분류기
        if ensemble_method == "stacking":
            self.meta_classifier = nn.Sequential(
                nn.Linear(num_classes * len(models), 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        앙상블 예측
        """
        # 각 모델의 예측 수집
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
            predictions.append(pred)
        
        if self.ensemble_method == "voting":
            return self._hard_voting(predictions)
        elif self.ensemble_method == "averaging":
            return self._soft_averaging(predictions)
        elif self.ensemble_method in ["weighted", "learned_weighted"]:
            return self._weighted_averaging(predictions)
        elif self.ensemble_method == "stacking":
            return self._stacking(predictions)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def _hard_voting(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """Hard voting: 각 모델의 예측 클래스를 투표"""
        votes = torch.stack([pred.argmax(dim=1) for pred in predictions], dim=0)
        # Mode (최빈값) 계산
        result = torch.mode(votes, dim=0)[0]
        # One-hot으로 변환
        return F.one_hot(result, num_classes=self.num_classes).float()
    
    def _soft_averaging(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """Soft voting: 확률 평균"""
        probs = torch.stack([F.softmax(pred, dim=1) for pred in predictions], dim=0)
        return probs.mean(dim=0)
    
    def _weighted_averaging(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """가중 평균"""
        weights = F.softmax(self.weights, dim=0)
        probs = torch.stack([F.softmax(pred, dim=1) for pred in predictions], dim=0)
        weighted_probs = (probs * weights.view(-1, 1, 1)).sum(dim=0)
        return weighted_probs
    
    def _stacking(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """Stacking: 메타 분류기 사용"""
        # 각 모델의 확률을 concatenate
        probs = torch.cat([F.softmax(pred, dim=1) for pred in predictions], dim=1)
        return self.meta_classifier(probs)
    
    def train_meta_classifier(
        self,
        train_loader,
        optimizer,
        criterion,
        epochs: int = 10,
        device: str = "cuda"
    ):
        """Stacking 메타 분류기 학습"""
        if self.ensemble_method != "stacking":
            raise ValueError("train_meta_classifier는 stacking 방법에서만 사용")
        
        # Base 모델들은 고정
        for model in self.models:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
        
        self.meta_classifier.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                # Base 모델 예측
                predictions = []
                for model in self.models:
                    with torch.no_grad():
                        pred = model(batch_x)
                    predictions.append(pred)
                
                # 메타 분류기 학습
                optimizer.zero_grad()
                output = self._stacking(predictions)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")


class MultiTaskClassifier(nn.Module):
    """
    멀티태스크 분류기
    계층적 분류 (상태 -> 세부 문제)
    
    예: 
    - Task 1: 상태 분류 (braking, idle, startup)
    - Task 2: 세부 문제 분류 (각 상태별 정상/비정상)
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_states: int,  # 상위 카테고리 수
        num_problems: int,  # 하위 문제 수
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.backbone = backbone
        
        # 상태 분류 헤드
        self.state_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_states)
        )
        
        # 문제 분류 헤드
        self.problem_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_problems)
        )
        
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            state_logits: (batch, num_states)
            problem_logits: (batch, num_problems)
        """
        # 백본으로 피처 추출
        features = self.backbone.extract_features(x)
        
        # 각 태스크 분류
        state_logits = self.state_head(features)
        problem_logits = self.problem_head(features)
        
        return state_logits, problem_logits
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """예측 결과를 딕셔너리로 반환"""
        state_logits, problem_logits = self.forward(x)
        
        return {
            "state": F.softmax(state_logits, dim=1),
            "state_pred": state_logits.argmax(dim=1),
            "problem": F.softmax(problem_logits, dim=1),
            "problem_pred": problem_logits.argmax(dim=1)
        }

