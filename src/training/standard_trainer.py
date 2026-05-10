import torch
import torch.nn as nn
from typing import Dict, Any
from .base_trainer import BaseTrainer
from .losses import AnatomicalLoss
from ..models.layers import SoftArgmax2D

class StandardTrainer(BaseTrainer):
    """
    Standard supervised trainer for pose estimation with optional anatomical constraints.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__(model, config, device, rank, world_size)
        self.optimizer = optimizer
        self.criterion = criterion
        
        # Anatomical constraints setup
        training_cfg = config.get("training", {})
        self.lambda_anatomical = training_cfg.get("lambda_anatomical", 0.0)
        self.warmup_epochs = training_cfg.get("warmup_epochs", 10)
        self.anatomical_mode = training_cfg.get("anatomical_mode", "hinge")
        
        if self.lambda_anatomical > 0:
            self.anatomical_criterion = AnatomicalLoss(
                device=device, mode=self.anatomical_mode
            ).to(device)
            self.soft_argmax = SoftArgmax2D().to(device)

    def _get_current_lambda_ana(self, epoch: int) -> float:
        # Linear warmup over configurable epochs
        if self.warmup_epochs <= 0:
            return self.lambda_anatomical
            
        if epoch < self.warmup_epochs:
            return self.lambda_anatomical * (epoch / self.warmup_epochs)
        return self.lambda_anatomical

    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        self.current_epoch = epoch # Store current epoch for steps
        return super().train_epoch(dataloader, epoch)

    def _train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        images = batch["image"].to(self.device)
        targets = batch["target"].to(self.device)
        
        outputs = self.model(images)
        loss_pose = self.criterion(outputs, targets)
        
        loss = loss_pose
        metrics = {"loss_pose": loss_pose.item()}
        
        if self.lambda_anatomical > 0:
            # Apply curriculum warmup
            curr_lambda = self._get_current_lambda_ana(self.current_epoch)
            
            # Extract coordinates differentiably
            pred_coords = self.soft_argmax(outputs)
            loss_ana = self.anatomical_criterion(pred_coords)
            loss = loss + curr_lambda * loss_ana
            metrics["loss_ana"] = loss_ana.item()
            metrics["lambda_ana"] = curr_lambda
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        metrics["loss"] = loss.item()
        return metrics

    def _val_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        images = batch["image"].to(self.device)
        targets = batch["target"].to(self.device)
        
        outputs = self.model(images)
        loss_pose = self.criterion(outputs, targets)
        
        loss = loss_pose
        metrics = {"loss_pose": loss_pose.item()}
        
        if self.lambda_anatomical > 0:
            curr_lambda = self._get_current_lambda_ana(self.current_epoch)
            pred_coords = self.soft_argmax(outputs)
            loss_ana = self.anatomical_criterion(pred_coords)
            loss = loss + curr_lambda * loss_ana
            metrics["loss_ana"] = loss_ana.item()
            metrics["lambda_ana"] = curr_lambda
            
        metrics["loss"] = loss.item()
        return metrics

    def _get_extra_checkpoint_data(self) -> Dict[str, Any]:
        return {"optimizer_state_dict": self.optimizer.state_dict()}

    def fit(self, train_loader, val_loader=None):
        for epoch in range(self.epochs):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = {}
            
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                
            if self.is_main:
                log_str = f"Epoch {epoch+1}: "
                log_str += " ".join([f"{k}={v:.4f}" for k, v in train_metrics.items()])
                if val_metrics:
                    log_str += " | " + " ".join([f"val_{k}={v:.4f}" for k, v in val_metrics.items()])
                print(log_str)
                
                val_loss = val_metrics.get("loss", float("inf"))
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                
                self.save_checkpoint(f"epoch_{epoch+1}", is_best=is_best)
                
                epoch_data = {"epoch": epoch+1, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
                self.update_history(epoch_data)
