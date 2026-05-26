from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from src.models import build_model
from src.training.standard_trainer import StandardTrainer
from src.training.uda_trainer import UDATrainer
from src.training.cyclegan_trainer import CycleGANTrainer
from src.models.discriminator import DomainDiscriminator


def build_optimizer(
    model: nn.Module, trainer: Any, config: Dict[str, Any], rank: int = 0
) -> optim.Optimizer:
    """
    Builds the Adam optimizer with support for discriminative learning rates
    and only includes trainable parameters.
    """
    train_cfg = config.get("training", {})
    lr = train_cfg.get("lr", 0.0001)
    weight_decay = train_cfg.get("weight_decay", 0.0001)
    backbone_lr_ratio = train_cfg.get("backbone_lr_ratio", 1.0)

    params = list(model.parameters())
    if hasattr(trainer, "uncertainty_loss") and trainer.uncertainty_loss is not None:
        params += list(trainer.uncertainty_loss.parameters())

    trainable_params = [p for p in params if p.requires_grad]

    if backbone_lr_ratio != 1.0:
        head_params = []
        backbone_params = []

        # Split model parameters
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("head.") or name.startswith("decoder."):
                head_params.append(param)
            else:
                backbone_params.append(param)

        # Also add any uncertainty loss parameters to head_params (learned from scratch)
        if (
            hasattr(trainer, "uncertainty_loss")
            and trainer.uncertainty_loss is not None
        ):
            for param in trainer.uncertainty_loss.parameters():
                if param.requires_grad:
                    head_params.append(param)

        param_groups = [
            {"params": backbone_params, "lr": lr * backbone_lr_ratio},
            {"params": head_params, "lr": lr},
        ]

        if rank == 0:
            print(
                f"[Factory] Using Discriminative LR! Head parameters: {len(head_params)}, Backbone parameters: {len(backbone_params)}, Ratio: {backbone_lr_ratio}"
            )
        optimizer = optim.Adam(param_groups, weight_decay=weight_decay)
    else:
        if rank == 0:
            print(
                f"[Factory] Using Uniform LR! Total trainable tensors: {len(trainable_params)}"
            )
        optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)

    return optimizer


def create_trainer(
    config: Dict[str, Any], device: torch.device, rank: int = 0, world_size: int = 1
) -> Tuple[Any, nn.Module]:
    """
    Factory function to create the appropriate trainer based on config.
    Returns (trainer_instance, model_instance).
    """
    # 1. Build Model
    model = build_model(config).to(device)

    # 2. Extract Configs
    train_cfg = config.get("training", {})
    uda_cfg = config.get("uda", {})

    # 3. Setup Optimizer & Criterion
    lr = train_cfg.get("lr", 0.0001)
    weight_decay = train_cfg.get("weight_decay", 0.0001)
    criterion = nn.MSELoss()

    # 4. Decide Trainer Type
    training_type = config.get("training_type", "standard")
    use_uda = training_type == "uda" or uda_cfg.get("enabled", False)
    use_cyclegan = training_type == "cyclegan" or train_cfg.get("cyclegan", False)

    if use_cyclegan:
        trainer = CycleGANTrainer(
            config=config, device=device, rank=rank, world_size=world_size
        )
        # For CycleGAN, 'model' returned by factory is G_AB (contained in trainer)
        model = trainer.G_AB
    elif use_uda:
        # UDA Setup
        discriminator = DomainDiscriminator(in_channels=480).to(device)
        optimizer_d = optim.Adam(
            discriminator.parameters(), lr=lr, weight_decay=weight_decay
        )

        trainer = UDATrainer(
            model=model,
            discriminator=discriminator,
            optimizer=None,  # Will set below
            optimizer_d=optimizer_d,
            criterion=criterion,
            config=config,
            device=device,
            rank=rank,
            world_size=world_size,
        )
    else:
        # Standard Setup
        trainer = StandardTrainer(
            model=model,
            optimizer=None,  # Will set below
            criterion=criterion,
            config=config,
            device=device,
            rank=rank,
            world_size=world_size,
        )
        if hasattr(trainer, "uncertainty_loss") and rank == 0:
            print("[Factory] Added uncertainty weighting parameters to optimizer")

    # 5. Finalize Optimizer — filter out frozen parameters and apply discriminative lr/uniform lr
    if not use_cyclegan:
        optimizer = build_optimizer(model, trainer, config, rank)
        trainer.optimizer = optimizer

    if use_uda and rank == 0:
        print(
            f"[Factory] Created UDATrainer (Lambda Adv: {uda_cfg.get('lambda_adv', 0.001)})"
        )
    elif not use_uda and rank == 0:
        print("[Factory] Created StandardTrainer")

    return trainer, model
