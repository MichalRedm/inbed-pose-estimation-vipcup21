# Current Status & Roadmap

## Implemented Features
- [x] **Project Infrastructure**: Relocated `remote_gpu.py` to `src/utils/`, initialized `src` packages, and set up `.venv` / `.env`.
- [x] **Data Loading**: Implemented `VIPCupDataset` (PyTorch) with coordinate shifting and modality support.
- [x] **Kaggle Integration**: Implemented `scripts/download_dataset.py` pointing to `awsaf49/ieee-vip-cup-2021-train-val-dataset`.
- [x] **Model Architecture**: Implemented simplified HRNet-W32 in `src/models/hrnet.py`.
- [x] **Training Pipeline**: Created `scripts/train.py` with robust heatmap-based MSE loss.
- [x] **Inference API**: Implemented FastAPI-based server for model access.
- [x] **Dashboard**: Full UI for inference, evaluation, and training management with joint visualization.
- [x] **Metrics**: Standardized on PCK@0.5 (torso-relative) and MPJPE.
- [x] **Iteration Loop**: Successfully executed `loop2_fixed_aug` with corrected augmentation logic.

## Current Technical Debts / Placeholders
- `src/models/hrnet.py` uses a simplified parallel stream architecture; could be extended to full HRNet-W32 for better performance.

## Future Tasks
- [ ] Implement Adversarial Domain Alignment to bridge the gap between uncovered and covered IR images.
- [ ] Explore Multi-Scale Heatmap Regression (dynamic sigma per joint).
- [ ] Fine-tune on specific cover types (e.g., thicker blankets).
- [ ] Integrate full HRNet-W32 parallel stream architecture.
