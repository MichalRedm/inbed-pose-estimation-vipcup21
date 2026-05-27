import json
from pathlib import Path
from .config_loader import load_config

PROJECT_ROOT = Path(__file__).parent.parent.parent
USER_CONFIG_PATH = PROJECT_ROOT / "configs" / "user_training.json"


def get_training_config():
    """
    Get training configuration, preferring user overrides if they exist.
    """
    # Start with defaults from default.yaml
    full_config = load_config()
    training_defaults = full_config.get("training", {})

    # Add remote default
    remote_defaults = full_config.get("remote", {})
    training_defaults["remote"] = remote_defaults.get("use_remote", False)

    # Check for user overrides
    if USER_CONFIG_PATH.exists():
        try:
            with open(USER_CONFIG_PATH, "r") as f:
                user_overrides = json.load(f)
                # Merge user overrides into defaults
                training_defaults.update(user_overrides)
        except Exception as e:
            print(f"Error loading user training config: {e}")

    return training_defaults


def get_display_metadata_for_config(config: dict) -> dict:
    """Heuristically determine display metadata based on config."""
    train_cfg = config.get("training", {})
    uda_cfg = config.get("uda", {})
    training_type = config.get("training_type", "standard")

    use_cyclegan = training_type == "cyclegan" or train_cfg.get("cyclegan", False)
    use_uda = training_type == "uda" or uda_cfg.get("enabled", False)

    if use_cyclegan:
        return {
            "charts": [
                {"key": "loss", "label": "Generator Loss", "color": "primary"},
                {
                    "key": "val_loss",
                    "label": "Val G Loss",
                    "color": "lime",
                    "dash": "5 3",
                },
                {
                    "key": "cycle_loss",
                    "label": "Cycle Consistency",
                    "color": "lime",
                    "dash": "2 2",
                },
                {
                    "key": "adv_loss",
                    "label": "Adversarial",
                    "color": "pink",
                    "dash": "4 4",
                },
            ],
            "highlights": [
                {"key": "loss", "label": "GENERATOR LOSS", "color": "primary"},
                {"key": "cycle_loss", "label": "CYCLE LOSS", "color": "lime"},
                {"key": "adv_loss", "label": "ADV LOSS", "color": "pink"},
                {"key": "d_loss", "label": "DISC LOSS", "color": "coral"},
            ],
            "primary_metric": "loss",
        }
    elif use_uda:
        return {
            "charts": [
                {"key": "loss", "label": "Pose Loss", "color": "primary"},
                {
                    "key": "adv_loss",
                    "label": "Domain Adv",
                    "color": "pink",
                    "dash": "4 4",
                },
                {"key": "val_pck", "label": "Val PCK", "color": "lime", "dash": "5 3"},
            ],
            "highlights": [
                {
                    "key": "val_pck",
                    "label": "VALIDATION PCK",
                    "color": "lime",
                    "suffix": "%",
                    "multiplier": 100,
                },
                {"key": "loss", "label": "POSE LOSS", "color": "primary"},
                {"key": "adv_loss", "label": "DOMAIN ADV", "color": "pink"},
            ],
            "primary_metric": "val_pck",
        }
    else:
        return {
            "charts": [
                {"key": "loss", "label": "Train Loss", "color": "primary"},
                {
                    "key": "val_loss",
                    "label": "Val Loss",
                    "color": "lime",
                    "dash": "5 3",
                },
            ],
            "highlights": [
                {
                    "key": "val_pck",
                    "label": "VALIDATION PCK",
                    "color": "lime",
                    "suffix": "%",
                    "multiplier": 100,
                },
                {"key": "loss", "label": "TRAIN LOSS", "color": "primary"},
                {"key": "sigma", "label": "SIGMA", "color": "pink"},
            ],
            "primary_metric": "val_pck",
        }


def save_training_config(config):
    """
    Save training configuration overrides to user_training.json.
    """
    USER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # We only save the hyperparameters we want to persist
    persistent_keys = ["lr", "epochs", "batch_size", "remote", "augmentation"]

    # Extract values from the root or from a nested 'training' object
    to_save = {}

    # 1. Check root level
    for k in persistent_keys:
        if k in config:
            to_save[k] = config[k]

    # 2. Check nested 'training' level (for compatibility with startTraining payload)
    if "training" in config:
        for k in persistent_keys:
            if k in config["training"]:
                to_save[k] = config["training"][k]

    with open(USER_CONFIG_PATH, "w") as f:
        json.dump(to_save, f, indent=4)

    return True
