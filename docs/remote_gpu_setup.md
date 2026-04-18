# 🚀 Kaggle Remote GPU Setup Guide

This guide explains how to connect your local development environment to a Kaggle GPU instance for high-performance training using the provided `src/utils/remote_gpu.py` utility.

---

## 1. Local Pre-requisites

### 🛠️ Install cloudflared
`cloudflared` is required to create a secure tunnel to Kaggle without exposing ports publicly.

- **Windows**: [Download the msi](https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.msi) and install it.
-   **Confirm Installation**:
    ```powershell
    cloudflared --version
    ```

### 🔑 SSH Key Pair
If you don't have an SSH key, generate one:
```powershell
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -C "kaggle-remote"
```
*(Press enter to skip the passphrase for automated login, or provide one for extra security).*

### 📦 Python Dependencies
Ensure your local environment has the required libraries:
```bash
pip install -r requirements-dev.txt
```
*(Specifically: `paramiko` and `scp`).*

---

## 2. Kaggle Setup

1.  **Open the Notebook**: Upload or open [kaggle-remote-gpu-server.ipynb](file:///d:/C/Users/Michał/Documents/GitHub/inbed-pose-estimation-vipcup21/kaggle-remote-gpu-server.ipynb) on Kaggle.
2.  **Add your Public Key**:
    -   On your local machine, copy your **public** key: `cat ~/.ssh/id_ed25519.pub`.
    -   On Kaggle, go to **Add-ons -> Secrets**.
    -   Add a new secret named `SSH_PUBLIC_KEY` and paste your key as the value.
3.  **Enable Internet & GPU**:
    -   In the Kaggle sidebar, ensure **Internet on** is enabled.
    -   Go to **Settings -> Accelerator** and select a GPU (T4 x2 or P100).
4.  **Run the Notebook**:
    -   Run all cells.
    -   The second-to-last cell will output a `ssh_config` block and a tunnel hostname (e.g., `xyz-abc-123.trycloudflare.com`).

---

## 3. Connecting from Local

### Option A: Automatic (using JSON)
Download the `gpu_connection.json` file from the `/kaggle/working/` directory in the Kaggle sidebar and save it to your local project root.

Run the verification script:
```bash
python scripts/verify_remote_gpu.py --json gpu_connection.json
```

### Option B: Manual Registration
You can register the backend directly in your Python code:

```python
from src.utils.remote_gpu import GPUManager

mgr = GPUManager()
mgr.add_backend('kaggle', {
    'type': 'cloudflare_tunnel',
    'tunnel_hostname': 'your-tunnel-id.trycloudflare.com',
    'ssh_user': 'root',
    'ssh_key': '~/.ssh/id_ed25519',
})

with mgr.use('kaggle') as gpu:
    gpu.run('nvidia-smi')
```

---

## 💡 Pro Tips
- **Session Limits**: Kaggle GPU sessions last 12 hours. You will need to re-run the notebook and update the `tunnel_hostname` when the session expires.
- **Pipelines**: Use `gpu.upload('./src', '/root/src')` to sync your code to the remote server before running `gpu.run('python src/train.py')`.
