from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from .wrapper import PoseDecodingWrapper

from .registry import MODEL_REGISTRY, register_model  # noqa: F401


def build_model(config: Dict[str, Any]) -> nn.Module:
    """
    Build a model based on the configuration.

    Args:
        config: Full project configuration dictionary.

    Returns:
        An instantiated PyTorch model.
    """
    model_cfg = config.get("model", {})
    name = model_cfg.get("name")

    if not name:
        raise ValueError("Model name not specified in configuration ('model.name').")

    if name not in MODEL_REGISTRY:
        # Import all modules in the package to ensure registration
        from . import hrnet  # noqa: F401
        from . import refined_hrnet  # noqa: F401
        from . import vitpose  # noqa: F401

        if name not in MODEL_REGISTRY:
            raise ValueError(
                f"Model '{name}' not found in registry. "
                f"Available models: {list(MODEL_REGISTRY.keys())}"
            )

    # Get specific model configuration (e.g., model.hrnet)
    specific_cfg = model_cfg.get(name, {})

    # Instantiate and return the model
    return MODEL_REGISTRY[name](specific_cfg)


def load_model_for_inference(
    checkpoint_path: str, device: torch.device, config: Optional[Dict[str, Any]] = None
) -> nn.Module:
    """
    Loads a model from a checkpoint and automatically wraps it in PoseDecodingWrapper
    if it's a heatmap-based model. Handles transient file corruption with retries.
    """
    import time

    state = None
    last_err = None
    for attempt in range(5):
        try:
            # We set weights_only=False because older checkpoints may contain
            # numpy scalars or other legacy objects that PyTorch 2.6+ blocks by default.
            state = torch.load(checkpoint_path, map_location=device, weights_only=False)
            break
        except Exception as e:
            last_err = e
            print(f"[load_model] Attempt {attempt + 1} failed: {e}. Retrying...")
            time.sleep(1.0)

    if state is None:
        raise RuntimeError(f"Failed to load checkpoint after 5 attempts: {last_err}")

    # Use config from checkpoint if available, else provided config
    if isinstance(state, dict) and "config" in state:
        model_config = state["config"]
    elif config:
        model_config = config
    else:
        from src.utils import load_config

        model_config = load_config()

    # --- FORCING PRETRAINED=FALSE FOR INFERENCE ---
    # We are loading our own weights anyway, so avoid downloading.
    if "model" in model_config:
        model_name = model_config.get("model", {}).get("name")
        if model_name and model_name in model_config["model"]:
            model_config["model"][model_name]["pretrained"] = False
    # ----------------------------------------------

    # Build base model
    model = build_model(model_config).to(device)

    # Load state dict
    # 3. Load weights with robust structural remapping for backward compatibility
    if isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]

        # Metadata keys to strip (handled separately)
        metadata_keys = ["decoding_config", "config", "best_optimized_pck"]
        filtered_state = {k: v for k, v in state_dict.items() if k not in metadata_keys}

        # --- Compatibility Remapping ---
        # Some older models used nested 'modules_list' in stages.
        # Newer models use flat nn.Sequential.
        remapped_state = {}
        model_keys = set(model.state_dict().keys())

        for k, v in filtered_state.items():
            new_k = k
            # Rule 1: modules_list.Y -> Y (legacy nested lists)
            if "modules_list." in k:
                new_k = new_k.replace("modules_list.", "")

            # Rule 2: fusion.fuse_layers -> fuse_layers.layers (architecture refactor)
            if "fusion.fuse_layers" in new_k:
                new_k = new_k.replace("fusion.fuse_layers", "fuse_layers.layers")

            if new_k in model_keys:
                remapped_state[new_k] = v
            elif k in model_keys:
                remapped_state[k] = v
            else:
                # If still no match, try stripping 'hrnet.' prefix if model is standalone
                # or adding 'hrnet.' if model is refined
                if k.startswith("hrnet.") and k[6:] in model_keys:
                    remapped_state[k[6:]] = v
                elif f"hrnet.{k}" in model_keys:
                    remapped_state[f"hrnet.{k}"] = v
                else:
                    # Keep original for strict=False to handle
                    remapped_state[k] = v

        load_res = model.load_state_dict(remapped_state, strict=False)

        # Verification: If many keys are missing, something is wrong
        missing = [k for k in load_res.missing_keys if "num_batches_tracked" not in k]
        total = len([k for k in model_keys if "num_batches_tracked" not in k])
        if len(missing) > total * 0.1:  # Allow 10% mismatch (e.g. some aux layers)
            print(
                f"WARNING: Major structural mismatch! {len(missing)}/{total} critical keys missing."
            )
            print(f"Sample missing: {missing[:5]}")
        elif len(missing) > 0:
            print(f"Loaded with {len(missing)} missing keys (remapping applied).")
        else:
            print("Model loaded with 100% key parity (remapping applied).")

    # Apply wrapper if it's a heatmap model
    if hasattr(model, "output_type") and model.output_type == "heatmap":
        decoding_cfg = {}
        if isinstance(state, dict) and "decoding_config" in state:
            decoding_cfg = state["decoding_config"]
        else:
            # Fallback to defaults or config
            train_cfg = model_config.get("training", {})
            decoding_cfg = {
                "method": train_cfg.get("decode_method", "argmax"),
                "temperature": train_cfg.get("decode_temperature", 10.0),
                "image_size": model_config.get("dataset", {}).get(
                    "image_size", [256, 256]
                ),
            }

        model = PoseDecodingWrapper(
            model,
            decode_method=decoding_cfg.get("method", "argmax"),
            temperature=decoding_cfg.get("temperature", 10.0),
            image_size=tuple(decoding_cfg.get("image_size", [256, 256])),
        )

    model.eval()
    return model
