from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from src.models import build_model
from src.training.standard_trainer import StandardTrainer
from src.training.uda_trainer import UDATrainer
from src.models.discriminator import DomainDiscriminator


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
    # Check if UDA is enabled in config
    use_uda = config.get("training_type") == "uda" or uda_cfg.get("enabled", False)

    if use_uda:
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
        params = list(model.parameters())
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
        params = list(model.parameters())
        if hasattr(trainer, "uncertainty_loss"):
            params += list(trainer.uncertainty_loss.parameters())
            if rank == 0:
                print("[Factory] Added uncertainty weighting parameters to optimizer")

    # 5. Finalize Optimizer — filter out frozen parameters
    trainable_params = [p for p in params if p.requires_grad]
    if rank == 0:
        print(f"[Factory] Total parameter tensors: {len(params)}, Trainable parameter tensors: {len(trainable_params)}")
    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    trainer.optimizer = optimizer

    if use_uda and rank == 0:
        print(
            f"[Factory] Created UDATrainer (Lambda Adv: {uda_cfg.get('lambda_adv', 0.001)})"
        )
    elif not use_uda and rank == 0:
        print("[Factory] Created StandardTrainer")

    return trainer, model
