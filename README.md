# In-Bed Human Pose Estimation (IEEE VIP Cup 2021)

[![Python CI](https://github.com/MichalRedm/inbed-pose-estimation-vipcup21/actions/workflows/ci.yml/badge.svg)](https://github.com/MichalRedm/inbed-pose-estimation-vipcup21/actions/workflows/ci.yml)

A professional machine learning repository for multi-modal in-bed human pose estimation. This project is based on the **IEEE VIP Cup 2021** challenge and utilizes the **SLP (Simultaneously-collected Multimodal Lying Pose)** dataset.

## 🚀 Overview

This repository implements a high-resolution pose estimation pipeline using **HRNet-W32**. It supports multi-modal inputs (RGB, LWIR) and is designed with **Unsupervised Domain Adaptation (UDA)** in mind to handle subjects under different types of blankets/covers.

### Key Features
- **Professional Structure**: Modularized `src` package for data, models, training, and utilities.
- **Multi-Modal Support**: Seamlessly handle RGB and IR modalities.
- **Robust CI/CD**: Automated linting, formatting, and unit testing via GitHub Actions.
- **Inference API**: FastAPI-based server for real-time human pose estimation from images.
- **Remote Training support**: Modular utilities for managing training on remote GPU backends.
- **ACS Compliant**: Uses the Agentic Collaboration Standard for persistent project context.

---

## 🛠️ Setup

### Prerequisites
- Python 3.10+
- [Kaggle API Credentials](https://github.com/Kaggle/kaggle-api) (Required for dataset download)

### Installation
1. Clone the repository.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
4. Configure secondary credentials:
   Create a `.env` file in the root directory:
   ```env
   KAGGLE_USERNAME=your_username
   KAGGLE_API_TOKEN=your_token
   ```

---

## 📊 Dataset

We use the **Simultaneously-collected Multimodal Lying Pose (SLP)** dataset hosted on Kaggle.

**Download the dataset:**
```bash
python scripts/download_dataset.py
```
This will download and extract the dataset into `data/raw/`.

---

## 🏋️ Training

To start the training process locally:
```bash
python scripts/train.py
```
Configuration parameters (model specs, hyperparameters, dataset paths) are managed via `configs/default.yaml`.

---

## 🌐 Remote Training

The repository includes a provider-agnostic utility for training on remote GPU instances (e.g., Kaggle, RunPod, Lambda) via SSH or Cloudflare Tunnels.

### 1. Configure Connection
Create a `gpu_connection.json` file in the root directory (this file is Git-ignored):

```json
{
  "remote_gpu": {
    "type": "cloudflare_tunnel",
    "tunnel_hostname": "your-unique-hostname.trycloudflare.com",
    "ssh_user": "root",
    "ssh_key": "~/.ssh/id_ed25519"
  }
}
```
*   `type`: Use `"cloudflare_tunnel"` for hosts behind tunnels or `"ssh"` for direct access.
*   `tunnel_hostname`: The URL provided by the remote server.

### 2. Launch Remote Training
Run the orchestration script:
```bash
python scripts/remote_train.py
```
This script will:
- Establish a secure connection.
- Sync your current local code to the remote instance.
- Automatically setup the remote environment and dependencies.
- Execute `train.py` on the remote GPU.

### 3. Resuming Training
If the session is interrupted (e.g., tunnel disconnection), simply run `remote_train.py` again. Use the `--resume` flag (enabled by default) to automatically load the latest checkpoint from `models/checkpoints/` and continue training.

---

## 🧪 Testing & Code Quality

We use **Ruff** for linting/formatting and **Pytest** for unit testing.

**Run All Checks:**
```bash
# Linting
python -m ruff check .
# Formatting check
python -m ruff format --check .
# Unit tests
python -m pytest tests/
```

---

## ⚡ Inference API

The project includes a FastAPI-based server to serve the trained model.

### 1. Start the API
```bash
python scripts/run_api.py --port 8000
```

### 2. Make a Prediction
You can send an image to the `/predict` endpoint:
```bash
curl -X POST "http://localhost:8000/predict" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@path/to/your/image.jpg"
```
The API returns the predicted (x, y) coordinates for all 14 joints.

---

## 📂 Project Structure
```text
.
├── .github/          # GitHub Actions CI workflows
├── configs/          # Experiment configurations (YAML)
├── data/             # Dataset storage (Git-ignored)
├── scripts/          # Execution scripts (download, train, test)
├── src/              # Core source code
│   ├── data/         # PyTorch Datasets/Dataloaders
│   ├── models/       # Model architectures (HRNet)
│   ├── api/          # FastAPI application
│   ├── training/     # Trainer classes and loss functions
│   └── utils/        # Shared utilities
├── tests/            # Unit testing suite
└── requirements.txt  # Production dependencies
```

## 📜 License
This project is for academic/research purposes associated with the IEEE VIP Cup 2021.
## ML Dashboard

A professional React-based dashboard for monitoring training and visualizing inference results.

### Features
- **Training Monitor**: Real-time loss history charts, progress tracking, and hyperparameter control.
- **Inference Visualizer**: Upload images and visualize predicted 14-joint poses on a canvas.
- **Model Management**: List and select trained model checkpoints.
- **Modern Design**: Sentry-inspired dark mode UI with premium aesthetics.

### Setup & Running

1. **Start Backend API**:
   ```bash
   python src/api/main.py
   ```

2. **Start Dashboard**:
   ```bash
   cd dashboard
   npm install
   npm run dev
   ```
   The dashboard will be available at `http://localhost:5173`.

### Technology Stack
- **Frontend**: React, TypeScript, Vite.
- **Styling**: Vanilla CSS with CSS Variables.
- **Visualizations**: Recharts, HTML5 Canvas.
- **Icons**: Lucide React.
- **API**: Axios, FastAPI (Backend).
