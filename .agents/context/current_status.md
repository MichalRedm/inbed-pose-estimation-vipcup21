# Current Status & Roadmap

## Implemented Features
- [x] **Project Infrastructure**: Relocated `remote_gpu.py` to `src/utils/`, initialized `src` packages, and set up `.venv` / `.env`.
- [x] **Data Loading**: Implemented `VIPCupDataset` (PyTorch) with coordinate shifting and modality support.
- [x] **Kaggle Integration**: Implemented `scripts/download_dataset.py` pointing to `awsaf49/ieee-vip-cup-2021-train-val-dataset`.
- [x] **Model Architecture**: Implemented simplified HRNet-W32 in `src/models/hrnet.py`.
- [x] **Training Pipeline**: Created `scripts/train.py` (Functional local loop).
- [x] **Repository Cleanup**: Removed legacy root notebooks and ignored project-specific `/data/`.
- [x] **Standardization**: Implemented **ACS (Agentic Collaboration Standard)** in the `.agents/` directory.
- [x] **CI Configuration**: Established `ruff` linting and formatting rules via dedicated `requirements-dev.txt`.
- [x] **Project Publication**: Created private GitHub repository and pushed the finalized codebase.
- [x] **Remote GPU Integration**: Implemented `remote_gpu.py` with Cloudflare Tunnel support and added a complete setup guide.
- [x] **Inference API**: Implemented FastAPI-based server for model access.

## Current Technical Debts / Placeholders
- `scripts/train.py` currently uses a placeholder regression loss; needs implementation of Gaussian heatmap targets.

## Future Tasks
- [ ] Implement robust heatmap-based MSE loss.
- [ ] Add Unsupervised Domain Adaptation (UDA) logic for covered subjects.
- [ ] Integrate full HRNet-W32 parallel stream architecture (currently simplified).
- [ ] Add visualization scripts for joint predictions.
