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


def save_training_config(config):
    """
    Save training configuration overrides to user_training.json.
    """
    USER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # We only save the hyperparameters we want to persist
    persistent_keys = ["lr", "epochs", "batch_size", "remote", "augmentation"]
    to_save = {k: v for k, v in config.items() if k in persistent_keys}

    # If the config is nested (from the frontend), flatten it
    if "training" in config:
        for k, v in config["training"].items():
            if k in persistent_keys:
                to_save[k] = v

    # Special handling for augmentation if it's passed at root of config but inside training in default.yaml
    if "augmentation" in config and "augmentation" not in to_save:
        to_save["augmentation"] = config["augmentation"]

    with open(USER_CONFIG_PATH, "w") as f:
        json.dump(to_save, f, indent=4)

    return True
