import torch
import torch.nn as nn
from typing import Dict, Any
from .standard_trainer import StandardTrainer, generate_pytorch_heatmaps
from .losses import FeatureDistillationLoss, UncertaintyWeighting
from ..models import build_model
import copy

class DistillationTrainer(StandardTrainer):
    """
    DistillationTrainer transfers pose features from a frozen RGB teacher to an IR student.
    At training time:
      - Student receives IR image
      - Teacher receives aligned RGB image (from batch["image_aligned"])
      - Distillation loss (MSE) is computed between corresponding features from Stage 3 and Stage 4.
      - Dynamic multi-task uncertainty loss weighting balances heatmap loss and distillation loss.
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
        super().__init__(model, optimizer, criterion, config, device, rank, world_size)
        
        # Load the teacher config and model
        distill_cfg = config.get("distillation", {})
        teacher_checkpoint = distill_cfg.get("teacher_checkpoint", None)
        if teacher_checkpoint is None:
            raise ValueError("[DistillationTrainer] 'teacher_checkpoint' path must be provided in config['distillation'].")
            
        # Build frozen teacher model
        # Teacher is an RGB model (in_channels=3), so we configure it with 3 channels
        teacher_config = copy.deepcopy(config)
        
        # Override model settings for teacher
        if "model" in teacher_config:
            if "hrnet" in teacher_config["model"]:
                teacher_config["model"]["hrnet"]["in_channels"] = 3
                teacher_config["model"]["hrnet"]["pretrained"] = False
        else:
            teacher_config["model"] = {
                "name": "hrnet",
                "hrnet": {
                    "in_channels": 3,
                    "pretrained": False,
                    "architecture": "w32",
                    "num_joints": 14,
                    "heatmap_size": [64, 64],
                }
            }
            
        if self.is_main:
            print(f"[DistillationTrainer] Building RGB Teacher HRNet (in_channels=3) from: {teacher_checkpoint}")
            
        self.teacher = build_model(teacher_config).to(device)
        
        # Load weights
        checkpoint = torch.load(teacher_checkpoint, map_location=device)
        if "model_state_dict" in checkpoint:
            self.teacher.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            self.teacher.load_state_dict(checkpoint["state_dict"])
        else:
            self.teacher.load_state_dict(checkpoint)
            
        # Freeze teacher weights
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # Setup multi-stage feature distillation loss
        # Channels for HRNet Stage 3 & Stage 4 are 224 and 480 respectively
        self.distill_loss_fn = FeatureDistillationLoss(
            channels_student=[224, 480],
            channels_teacher=[224, 480]
        ).to(device)
        
        # Loss balance and dynamic uncertainty weighting parameters
        self.lambda_distill = distill_cfg.get("lambda_distill", 1.0)
        self.use_uncertainty = distill_cfg.get("use_uncertainty_weighting", True)
        
        if self.use_uncertainty:
            # Overwrite uncertainty_loss to handle pose heatmap + distillation
            self.tasks = ["pose", "distill"]
            self.uncertainty_loss = UncertaintyWeighting(len(self.tasks)).to(device)
            if self.is_main:
                print(f"[DistillationTrainer] Using Uncertainty Weighting for: {self.tasks}")

    def _train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        images = batch["image"].to(self.device)
        joints = batch["joints"].to(self.device)  # (B, 3, 14)
        
        # Track current sigma for metrics and dynamic curriculum
        sigma = self._get_current_sigma(self.current_epoch)
        
        # Generate target heatmaps dynamically with high precision on the GPU!
        targets = generate_pytorch_heatmaps(
            joints=joints, heatmap_size=(64, 64), image_size=(256, 256), sigma=sigma
        )
        
        # 1. Forward Student (IR modality)
        # We request intermediate Stage 3 and 4 feature maps using return_stages=True
        student_outputs, s_stage3, s_stage4 = self.model(images, return_stages=True)
        
        # Compute standard heatmap pose loss
        loss_pose = self.criterion(student_outputs, targets)
        
        # 2. Forward Teacher (Aligned RGB modality)
        # Teacher is evaluated in eval mode and doesn't calculate gradients
        loss_distill = torch.tensor(0.0, device=self.device)
        
        if "image_aligned" in batch:
            images_aligned = batch["image_aligned"].to(self.device)
            
            with torch.no_grad():
                _, t_stage3, t_stage4 = self.teacher(images_aligned, return_stages=True)
                
            # Compute distillation loss between student and teacher stages
            loss_distill = self.distill_loss_fn(
                student_features=[s_stage3, s_stage4],
                teacher_features=[t_stage3, t_stage4]
            )
            
        # 3. Combine losses
        if self.use_uncertainty:
            losses_dict = {
                "pose": loss_pose,
                "distill": loss_distill
            }
            # Balance automatically using learned log-variances
            total_loss, weighted_dict = self.uncertainty_loss(losses_dict)
            
            # Extract weighted loss values for monitoring
            metrics = {
                "loss": total_loss.item(),
                "loss_pose": loss_pose.item(),
                "loss_distill": loss_distill.item(),
                "w_pose": weighted_dict.get("w_pose", 0.0),
                "w_distill": weighted_dict.get("w_distill", 0.0),
                "sigma_pose": weighted_dict.get("sigma_pose", 1.0),
                "sigma_distill": weighted_dict.get("sigma_distill", 1.0),
                "sigma": sigma
            }
        else:
            # Fixed weighting
            total_loss = loss_pose + self.lambda_distill * loss_distill
            metrics = {
                "loss": total_loss.item(),
                "loss_pose": loss_pose.item(),
                "loss_distill": loss_distill.item(),
                "sigma": sigma
            }
            
        # Backward and step
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return metrics
