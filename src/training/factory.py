from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from src.models import build_model
from src.training.standard_trainer import StandardTrainer
from src.training.uda_trainer import UDATrainer
from src.models.discriminator import DomainDiscriminator

def create_trainer(
    config: Dict[str, Any], 
    device: torch.device, 
    rank: int = 0, 
    world_size: int = 1
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
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    # 4. Decide Trainer Type
    # Check if UDA is enabled in config
    use_uda = config.get("training_type") == "uda" or uda_cfg.get("enabled", False)
    
    if use_uda:
        # UDA Setup
        # Discriminator in_channels should match model's bottleneck/feature layer
        # For HRNet-W32, features are concatenated streams (32+64+128+256 = 480)
        in_channels = 480 
        if config.get("model", {}).get("name") != "hrnet":
            # Potential fallback or different model handling
            pass
            
        discriminator = DomainDiscriminator(in_channels=in_channels).to(device)
        optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, weight_decay=weight_decay)
        
        trainer = UDATrainer(
            model=model,
            discriminator=discriminator,
            optimizer=optimizer,
            optimizer_d=optimizer_d,
            criterion=criterion,
            config=config,
            device=device,
            rank=rank,
            world_size=world_size
        )
        if rank == 0:
            print(f"[Factory] Created UDATrainer (Lambda Adv: {uda_cfg.get('lambda_adv', 0.001)})")
    else:
        # Standard Setup
        trainer = StandardTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            device=device,
            rank=rank,
            world_size=world_size
        )
        if rank == 0:
            print("[Factory] Created StandardTrainer")
        
    return trainer, model
