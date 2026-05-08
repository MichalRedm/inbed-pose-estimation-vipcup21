import os
import yaml
from dotenv import load_dotenv


def load_config(config_path="configs/default.yaml"):
    """
    Load configuration from YAML and merge with environment variables.
    """
    load_dotenv()

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override with user training config if present
    user_config_path = os.path.join(
        os.path.dirname(os.path.dirname(config_path)), "configs", "user_training.json"
    )
    if os.path.exists(user_config_path):
        import json

        try:
            with open(user_config_path, "r") as f:
                user_overrides = json.load(f)
                # Training overrides
                if "training" in config:
                    # Update specific keys from user_overrides
                    # Note: user_training.json is currently flattened for some keys
                    for k, v in user_overrides.items():
                        if k in ["lr", "epochs", "batch_size", "augmentation"]:
                            config["training"][k] = v

                # Remote overrides
                if "remote" in config and "remote" in user_overrides:
                    config["remote"]["use_remote"] = user_overrides["remote"]
        except Exception as e:
            print(f"Error merging user config: {e}")

    # Override with environment variables if present
    # This is a simple recursive override logic
    _override_with_env(config)

    return config


def _override_with_env(config, prefix="APP"):
    for key, value in config.items():
        env_key = f"{prefix}_{key.upper()}"
        if isinstance(value, dict):
            _override_with_env(value, prefix=env_key)
        else:
            env_val = os.getenv(env_key)
            if env_val is not None:
                # Type casting based on YAML type
                if isinstance(value, bool):
                    config[key] = env_val.lower() in ("true", "1", "yes")
                elif isinstance(value, int):
                    config[key] = int(env_val)
                elif isinstance(value, float):
                    config[key] = float(env_val)
                else:
                    config[key] = env_val
