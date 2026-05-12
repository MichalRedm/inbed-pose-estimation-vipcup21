import sqlite3
import json
import os
import time
from typing import Dict, Any, Optional, List

class LocalTracker:
    """
    Local SQLite-based telemetry tracker for MLOps.
    Provides persistent storage for run configurations and metrics.
    """
    def __init__(self, db_path: str = "results/telemetry.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    name TEXT,
                    config TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    run_id TEXT,
                    epoch INTEGER,
                    name TEXT,
                    value REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id)")

    def init_run(self, run_id: str, name: str, config: Dict[str, Any]):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, name, config) VALUES (?, ?, ?)",
                (run_id, name, json.dumps(config))
            )

    def log_metric(self, run_id: str, epoch: int, name: str, value: float):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO metrics (run_id, epoch, name, value) VALUES (?, ?, ?, ?)",
                (run_id, epoch, name, value)
            )

    def get_run_history(self, run_id: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT epoch, name, value FROM metrics WHERE run_id = ? ORDER BY epoch ASC",
                (run_id,)
            )
            history = {}
            for epoch, name, value in cursor.fetchall():
                if epoch not in history:
                    history[epoch] = {"epoch": epoch}
                history[epoch][name] = value
            return sorted(list(history.values()), key=lambda x: x["epoch"])

class JSONLStream:
    """
    Helper to emit structured logs for real-time dashboard updates.
    """
    def __init__(self, stream_path: str):
        self.stream_path = stream_path
        os.makedirs(os.path.dirname(self.stream_path), exist_ok=True)

    def emit(self, data: Dict[str, Any]):
        # Add [METRICS] prefix for easier identification if mixed with stdout
        payload = f"[METRICS] {json.dumps(data)}"
        try:
            with open(self.stream_path, "a") as f:
                f.write(payload + "\n")
            # Also print to stdout for standard log aggregation
            print(payload, flush=True)
        except Exception:
            pass
