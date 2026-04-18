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
