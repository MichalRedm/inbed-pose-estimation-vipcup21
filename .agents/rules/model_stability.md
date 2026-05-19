# Model Stability and Backward Compatibility

To prevent "random mess" regressions in inference and ensure long-term research reproducibility, the following rules MUST be followed by all agents.

## 1. Zero-Regression Policy
- **Absolute Rule**: Any change to model architecture (e.g., refactoring `hrnet.py`) MUST maintain backward compatibility with older checkpoints (Loop 1-21+).
- **Mandatory Mapping**: If layer names or nesting change, you MUST implement a remapping rule in `src/models/__init__.py:load_model_for_inference`.

## 2. Self-Containment Standards
- **Checkpoint Metadata**: Every checkpoint MUST contain:
  - `model_state_dict`: The weights.
  - `config`: The full configuration used for training.
  - `decoding_config`: Explicit image size and decoding method (argmax/soft-argmax).
- **Inference Reliability**: `load_model_for_inference` is the single source of truth for loading models. Never load weights directly without using this utility.

## 3. Mandatory Testing
- **Compatibility Suite**: `tests/test_legacy_compatibility.py` MUST pass after any change to `src/models/`.
- **Visual Audit**: Before committing architecture changes, run `scratch/visual_inference_check.py` to confirm that older models still produce coherent skeletons.
- **Fail-Fast Parity**: The loader must report "100% key parity" for standard models. Any mismatch exceeding 10% of keys must be treated as a CRITICAL BLOCKER.

## 4. Model Registry
- **Centralized Discovery**: All models must be registered in `src/models/registry.py` using the `@register_model` decorator.
- **Isolation**: Avoid relative imports in registry-related code to prevent multiple registry instances in different execution contexts.

## 5. Atomic Loading and Race Conditions
- **Shadow Loading**: On Windows, the `InferenceService` MUST copy checkpoints to a temporary location (`scratch/inference_cache`) before loading. This prevents the API from locking files and blocking the trainer's save process.
- **Verify-then-Commit**: The trainer MUST verify every saved checkpoint (by attempting a `torch.load` on the temporary file) before renaming it to the final `best_model.pth` or `latest_model.pth`. This prevents corrupted archives from being finalized.
- **Race Condition Retries**: `load_model_for_inference` MUST implement a retry loop (default 5 attempts) to handle transient corruption errors during file writes.
- **Atomic Service**: `InferenceService.load_model` must perform atomic updates of the model instance.

## 6. Pre-trained Weights in Inference
- **Explicit Suppression**: When building a model for inference, `pretrained` MUST be forced to `False` in the configuration to prevent unnecessary downloads and network-related failures.

## 7. Coordinate Passthrough
- **No Double-Decoding**: If a model natively outputs coordinates (`output_type == "coordinates"`), the inference pipeline MUST bypass `decode_heatmaps`. The `PoseDecodingWrapper` and `InferenceService` both enforce this check to prevent `ValueError` shape mismatches.

## 8. Heatmap Decoding Parity
- **No Heatmap Decoding Mutations**: Any modification to `decode_heatmaps` inside `src/utils/pose.py` or classes like `SoftArgmax2D` MUST default to the exact standard global argmax/soft-argmax behaviors unless explicitly configured otherwise.
- **Parametric Default Protection**: Any new decoding features (such as local windowing, masking, or custom kernels) must default to disabled (`None` or `False`) so that legacy models loading these classes operate with 100% mathematical and behavioral parity compared to their original training settings.

