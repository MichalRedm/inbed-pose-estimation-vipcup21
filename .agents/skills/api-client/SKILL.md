# API Client Skill

This skill provides instructions for interacting with the In-Bed Pose Estimation API using the dedicated CLI tool.

## Overview

The API CLI (`scripts/api_cli.py`) is the primary interface for an agent to trigger and monitor machine learning experiments. Using this tool ensures that:
1.  All experiments are tracked in the Dashboard.
2.  Telemetry (loss curves, logs) is visible to the USER in real-time.
3.  Remote GPU resources are managed correctly.

## Commands

### 1. Start Training
Trigger a new training run (local or remote).
```bash
python scripts/api_cli.py start --run_id <run_id> --config <config_path> [--remote] [--resume]
```

### 2. Monitor Status
Get a summary of the current training state.
```bash
python scripts/api_cli.py status
```

### 3. Fetch Logs
See the latest output from the training process.
```bash
python scripts/api_cli.py logs [-n <lines>]
```

### 4. Stop Training
Gracefully terminate the current session.
```bash
python scripts/api_cli.py stop
```

## Protocol for Agents

1.  **NEVER** run `scripts/train.py` or `scripts/remote_train.py` directly.
2.  **ALWAYS** use the API CLI to start training.
3.  **VALIDATE** connection before starting a remote run using the `status` command (ensure API is up).
4.  **RECORD** the `run_id` in the `state_tracker.md` before starting.
