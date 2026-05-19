"""
remote_gpu.py — Plug-and-play remote GPU connector for ML pipelines
====================================================================

Supports any SSH-accessible GPU server, including:
  • Kaggle  (via Cloudflare tunnel + this notebook)
  • Vast.ai, RunPod, Lambda Labs, CoreWeave, AWS, GCP, Azure VMs
  • Your own on-prem server

Quick start
-----------
    from remote_gpu import GPUManager

    mgr = GPUManager()

    # Register backends (one-time setup)
    mgr.add_backend('kaggle', {
        'type': 'cloudflare_tunnel',
        'tunnel_hostname': 'abc-def-123.trycloudflare.com',   # from notebook output
        'ssh_user': 'root',
        'ssh_key': '~/.ssh/id_ed25519',
    })
    mgr.add_backend('vast', {
        'type': 'ssh',
        'host': '12.34.56.78',
        'port': 22001,
        'ssh_user': 'root',
        'ssh_key': '~/.ssh/id_ed25519',
    })

    # Use a backend
    with mgr.use('kaggle') as gpu:
        gpu.upload('./my_project', '/root/my_project')
        result = gpu.run('cd /root/my_project && python train.py --epochs 20')
        print(result.stdout)
        gpu.download('/root/my_project/checkpoints', './checkpoints')

    # Or: auto-pick the first reachable backend
    with mgr.use_any() as gpu:
        gpu.run('python train.py')

Dependencies
------------
    pip install paramiko scp

Usage with connection JSON saved by the Kaggle notebook
---------------------------------------------------------
    mgr.add_backend_from_json('kaggle', '/path/to/gpu_connection.json')
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

# ── Optional imports — only needed at connection time ─────────────────────────
try:
    import paramiko
    from scp import SCPClient

    _HAS_PARAMIKO = True
except ImportError:
    _HAS_PARAMIKO = False


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RunResult:
    stdout: str
    stderr: str
    exit_code: int

    def ok(self) -> bool:
        return self.exit_code == 0

    def __repr__(self):
        lines = self.stdout.strip().splitlines()
        preview = lines[-1] if lines else "(empty)"
        return (
            f"<RunResult exit={self.exit_code} "
            f"stdout_lines={len(lines)} last={preview!r}>"
        )


@dataclass
class BackendConfig:
    name: str
    type: str  # 'ssh' | 'cloudflare_tunnel'
    host: str = ""
    port: int = 22
    ssh_user: str = "root"
    ssh_key: str = "~/.ssh/id_ed25519"
    # For cloudflare_tunnel type
    tunnel_hostname: str = ""
    # Extra metadata (GPU label, etc.)
    meta: dict = field(default_factory=dict)

    def effective_host(self) -> str:
        """Resolve actual TCP host to connect to."""
        if self.type == "cloudflare_tunnel":
            return self.tunnel_hostname or self.host
        return self.host


# ─────────────────────────────────────────────────────────────────────────────
# CloudflaredProxy — opens a local port via `cloudflared access tcp`
# ─────────────────────────────────────────────────────────────────────────────


class CloudflaredProxy:
    """
    Starts `cloudflared access tcp --hostname <h> --listener 127.0.0.1:<p>`
    so that paramiko can connect to localhost:<p> instead of needing
    ProxyCommand support.
    """

    def __init__(self, hostname: str, local_port: int = 0):
        import socket

        if local_port == 0:
            # Find a free port
            with socket.socket() as s:
                s.bind(("", 0))
                local_port = s.getsockname()[1]
        self.hostname = hostname
        self.local_port = local_port
        self._proc: subprocess.Popen | None = None
        self._executable = self._get_cloudflared_executable()

    def _get_cloudflared_executable(self) -> str:
        """Find the cloudflared executable, checking common Windows paths if needed."""
        # 1. Check if in PATH
        import shutil

        if shutil.which("cloudflared"):
            return "cloudflared"

        # 2. Check environment variable
        env_path = os.environ.get("GPU_CLOUDFLARED_PATH")
        if env_path and os.path.exists(env_path):
            return env_path

        # 3. Check common Windows paths
        common_paths = [
            r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
            r"C:\Program Files\cloudflared\cloudflared.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\bin\cloudflared.exe"),
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path

        # Default back to "cloudflared" and hope for the best
        return "cloudflared"

    def start(self, timeout: float = 15.0):
        cmd = [
            self._executable,
            "access",
            "tcp",
            "--hostname",
            self.hostname,
            "--listener",
            f"127.0.0.1:{self.local_port}",
        ]
        print(f"  Starting cloudflared tunnel: {' '.join(cmd)}")
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,  # Capture stderr for debugging
            text=True,
        )
        # Wait until the local port is accepting connections
        import socket

        deadline = time.time() + timeout
        while time.time() < deadline:
            # Check if process crashed
            if self._proc.poll() is not None:
                err = self._proc.stderr.read() if self._proc.stderr else "Unknown error"
                raise RuntimeError(f"cloudflared exited unexpectedly: {err}")

            try:
                with socket.create_connection(
                    ("127.0.0.1", self.local_port), timeout=1
                ):
                    # Give it a tiny bit more time to stabilize
                    time.sleep(0.5)
                    return  # ready
            except OSError:
                time.sleep(0.5)

        # If we reached here, it timed out
        if self._proc.poll() is None:
            self._proc.terminate()

        raise TimeoutError(
            f"cloudflared proxy on port {self.local_port} did not start "
            f"within {timeout}s. Is cloudflared installed and in PATH?"
        )

    def stop(self):
        if self._proc:
            self._proc.terminate()
            self._proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


# ─────────────────────────────────────────────────────────────────────────────
# GPUSession — represents an active connection to one GPU backend
# ─────────────────────────────────────────────────────────────────────────────


class GPUSession:
    """Active SSH session to a remote GPU. Use via GPUManager.use()."""

    def __init__(self, config: BackendConfig):
        if not _HAS_PARAMIKO:
            raise ImportError("paramiko and scp are required: pip install paramiko scp")
        self.config = config
        self._ssh: paramiko.SSHClient | None = None
        self._proxy: CloudflaredProxy | None = None

    # ── Connection lifecycle ──────────────────────────────────────────────────

    def connect(self):
        key_path = os.path.expanduser(self.config.ssh_key)

        if self.config.type == "cloudflare_tunnel":
            self._proxy = CloudflaredProxy(self.config.tunnel_hostname)
            self._proxy.start()
            connect_host = "127.0.0.1"
            connect_port = self._proxy.local_port
        else:
            connect_host = self.config.host
            connect_port = self.config.port

        key_path = os.path.expanduser(os.path.expandvars(key_path))
        print(f"  Using SSH key: {key_path} (exists={os.path.exists(key_path)})")
        self._ssh = paramiko.SSHClient()
        self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        print(
            f"  Attempting SSH connection to {connect_host}:{connect_port} as {self.config.ssh_user}..."
        )
        self._ssh.connect(
            hostname=connect_host,
            port=connect_port,
            username=self.config.ssh_user,
            key_filename=key_path,
            allow_agent=True,
            look_for_keys=False,
            timeout=30,
            banner_timeout=60,
        )
        # Prevent session timeout during long data downloads
        if self._ssh.get_transport():
            self._ssh.get_transport().set_keepalive(30)
            self._ssh.get_transport().set_timeout(30.0)
        print(f"Connected to [{self.config.name}]")

    def disconnect(self):
        if self._ssh:
            self._ssh.close()
            self._ssh = None
        if self._proxy:
            self._proxy.stop()
            self._proxy = None

    # ── Remote execution ──────────────────────────────────────────────────────

    # ── Remote execution ──────────────────────────────────────────────────────

    def run(
        self,
        command: str,
        timeout: int = 3600,
        stream: bool = True,
        check: bool = False,
    ) -> RunResult:
        """
        Execute a shell command on the remote GPU and return a RunResult.

        Args:
            command: The command string to execute
            timeout: Command timeout in seconds
            stream: If True, print stdout/stderr in real-time
            check: If True, raise RuntimeError if exit code is non-zero

        Notes:
            Commands are wrapped in ``bash -l -c '...'`` (a login shell) so that
            the remote ``~/.bash_profile`` is sourced automatically.
        """
        import sys
        import threading

        if not self._ssh:
            raise RuntimeError("Not connected. Use GPUManager.use() context manager.")

        # Wrap in a login shell so ~/.bash_profile is sourced
        wrapped = f"bash -l -c {shlex.quote(command)}"

        _, stdout_f, stderr_f = self._ssh.exec_command(wrapped, timeout=timeout)
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        if stream:

            def _stream(channel_file, storage, prefix=""):
                while True:
                    # Read what's available (blocks until at least 1 byte is available or EOF)
                    # Note: channel_file is a paramiko.ChannelFile
                    try:
                        chunk = channel_file.read(8192)
                        if not chunk:
                            break

                        data = chunk.decode(sys.stdout.encoding, errors="replace")
                        storage.append(data)

                        if prefix:
                            # For stderr, prefix each line
                            for line in data.splitlines(keepends=True):
                                print(f"{prefix}{line}", end="", flush=True)
                        else:
                            # For stdout, print raw (to preserve tqdm/ \r)
                            print(data, end="", flush=True)
                    except Exception:
                        break

            t_out = threading.Thread(target=_stream, args=(stdout_f, stdout_lines, ""))
            t_err = threading.Thread(
                target=_stream, args=(stderr_f, stderr_lines, "[stderr] ")
            )
            t_out.start()
            t_err.start()
            t_out.join()
            t_err.join()
        else:
            stdout_lines = stdout_f.readlines()
            stderr_lines = stderr_f.readlines()

        exit_code = stdout_f.channel.recv_exit_status()

        result = RunResult(
            stdout="".join(stdout_lines),
            stderr="".join(stderr_lines),
            exit_code=exit_code,
        )

        if check and exit_code != 0:
            raise RuntimeError(
                f"Command failed with exit code {exit_code}: {command}\nStderr: {result.stderr}"
            )

        return result

    def run_python(self, script: str, timeout: int = 3600) -> RunResult:
        """Run an inline Python script on the remote GPU."""
        remote_path = f"/tmp/_remote_script_{int(time.time())}.py"
        self.write_file(remote_path, script)
        return self.run(f"python {remote_path}", timeout=timeout)

    # ── File transfer ─────────────────────────────────────────────────────────

    def upload(self, local_path: str, remote_path: str, recursive: bool = True):
        """
        Upload a local file or directory to the remote GPU.
        Uses SCP under the hood.
        """
        with SCPClient(self._ssh.get_transport()) as scp:
            scp.put(local_path, remote_path=remote_path, recursive=recursive)
        print(f"Uploaded {local_path!r} -> {remote_path!r}")

    def download(self, remote_path: str, local_path: str, recursive: bool = True):
        """Download a file or directory from the remote GPU."""
        # Fix: Only create parent directory, not the local_path itself!
        # Otherwise scp always treats local_path as a destination directory.
        parent = os.path.dirname(local_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with SCPClient(self._ssh.get_transport()) as scp:
            scp.get(remote_path, local_path=local_path, recursive=recursive)
        print(f"Downloaded {remote_path!r} -> {local_path!r}")

    def sync_project(
        self,
        local_dir: str = ".",
        remote_dir: str = "/root/project",
        exclude: list[str] | None = None,
    ):
        """
        Sync local code to the remote GPU, excluding data, venv, and git.
        """
        import shutil
        import tempfile
        import tarfile

        # 1. Prepare a clean temporary directory with only the necessary files
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = os.path.join(tmp_dir, "sync_payload")
            os.makedirs(target)

            include = [
                "src",
                "scripts",
                "configs",
                "data",
                "results",
                "requirements.txt",
                "README.md",
                "gpu_connection.json",
                ".env",
            ]

            print("Preparing project tarball (excluding large data and checkpoints)...")
            for item in include:
                src_path = os.path.join(local_dir, item)
                if not os.path.exists(src_path):
                    continue

                dst_path = os.path.join(target, item)
                if os.path.isdir(src_path):
                    # For data, we strictly exclude 'raw'
                    # For results, we exclude '.pth' checkpoints to keep the payload small
                    ignore_patterns = ["processed_cache"]
                    if item == "data":
                        ignore_patterns.append("raw")
                    elif item == "results":
                        ignore_patterns.extend(["*.pth", "*.pt", "*.ckpt"])

                    shutil.copytree(
                        src_path,
                        dst_path,
                        ignore=shutil.ignore_patterns(*ignore_patterns),
                        dirs_exist_ok=True,
                    )
                else:
                    shutil.copy2(src_path, dst_path)

            # 2. Package into a tarball
            archive_path = os.path.join(tmp_dir, "project.tar.gz")
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(target, arcname=".")

            # 3. Create remote dir and upload
            self.run(f"mkdir -p {remote_dir}", check=True)
            print(
                f"Uploading project tarball ({os.path.getsize(archive_path) / 1024 / 1024:.1f} MB)..."
            )
            self.upload(archive_path, f"{remote_dir}/project.tar.gz", recursive=False)

            # 4. Extract on remote
            print("Extracting project on remote...")
            self.run(
                f"cd {remote_dir} && tar -xzf project.tar.gz && rm project.tar.gz",
                check=True,
            )

        print(f"Project synced successfully to {remote_dir}")

    def write_file(self, remote_path: str, content: str):
        """Write a text string directly to a file on the remote GPU."""
        sftp = self._ssh.open_sftp()
        with sftp.open(remote_path, "w") as f:
            f.write(content)
        sftp.close()

    def read_file(self, remote_path: str) -> str:
        """Read a text file from the remote GPU."""
        sftp = self._ssh.open_sftp()
        with sftp.open(remote_path, "r") as f:
            content = f.read()
        sftp.close()
        return content.decode() if isinstance(content, bytes) else content

    def gpu_info(self) -> str:
        """Return nvidia-smi output from the remote machine."""
        r = self.run("nvidia-smi", stream=False)
        return r.stdout

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()


# ─────────────────────────────────────────────────────────────────────────────
# GPUManager — registry of backends + smart selection
# ─────────────────────────────────────────────────────────────────────────────


class GPUManager:
    """
    Central registry for all your GPU backends.

    Example
    -------
        mgr = GPUManager()
        mgr.add_backend('kaggle', {
            'type': 'cloudflare_tunnel',
            'tunnel_hostname': 'abc-def-123.trycloudflare.com',
            'ssh_user': 'root',
            'ssh_key': '~/.ssh/id_ed25519',
        })
        with mgr.use('kaggle') as gpu:
            gpu.run('python train.py')
    """

    def __init__(self):
        self._backends: dict[str, BackendConfig] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def add_backend(self, name: str, cfg: dict[str, Any]):
        """Register a GPU backend by name."""
        self._backends[name] = BackendConfig(
            name=name,
            type=cfg.get("type", "ssh"),
            host=cfg.get("host", cfg.get("tunnel_hostname", "")),
            port=cfg.get("port", 22),
            ssh_user=cfg.get("ssh_user", "root"),
            ssh_key=cfg.get("ssh_key", "~/.ssh/id_ed25519"),
            tunnel_hostname=cfg.get("tunnel_hostname", cfg.get("host", "")),
            meta=cfg.get("meta", {}),
        )
        print(f"Registered backend: {name!r} [{cfg.get('type', 'ssh')}]")

    def add_backend_from_json(self, name: str, json_path: str):
        """
        Load backend config from a gpu_connection.json file.

        The JSON schema supports two backend types:

        Cloudflare tunnel (Kaggle, self-hosted)::

            {"type": "cloudflare_tunnel", "tunnel_hostname": "abc.trycloudflare.com",
             "ssh_user": "root", "port": 22, "gpu": "Tesla T4"}

        Direct SSH (RunPod, Vast.ai, Lambda Labs, on-prem)::

            {"type": "ssh", "host": "12.34.56.78", "port": 22001,
             "ssh_user": "root", "gpu": "A100"}
        """
        with open(os.path.expanduser(json_path)) as f:
            data = json.load(f)
        backend_type = data.get("type", "cloudflare_tunnel")
        self.add_backend(
            name,
            {
                "type": backend_type,
                "tunnel_hostname": data.get("tunnel_hostname", ""),
                "host": data.get("host", data.get("tunnel_hostname", "")),
                "ssh_user": data.get("ssh_user", "root"),
                "port": data.get("port", 22),
                "meta": {"gpu": data.get("gpu", "unknown")},
            },
        )

    def remove_backend(self, name: str):
        self._backends.pop(name, None)

    def list_backends(self) -> list[str]:
        return list(self._backends.keys())

    # ── Connection ────────────────────────────────────────────────────────────

    @contextmanager
    def use(self, name: str):
        """
        Context manager that opens and yields a GPUSession for the named backend.

        Usage:
            with mgr.use('kaggle') as gpu:
                gpu.run('python train.py')
        """
        if name not in self._backends:
            raise KeyError(
                f"Backend {name!r} not found. Available: {self.list_backends()}"
            )
        session = GPUSession(self._backends[name])
        try:
            yield session.__enter__()
        finally:
            session.__exit__(None, None, None)

    @contextmanager
    def use_any(self, preferred: list[str] | None = None):
        """
        Try backends in order (preferred list first, then all others) and
        open the first one that is reachable.

        Usage:
            with mgr.use_any(preferred=['kaggle', 'vast']) as gpu:
                gpu.run('python train.py')
        """
        order = (preferred or []) + [
            n for n in self._backends if n not in (preferred or [])
        ]
        for name in order:
            try:
                session = GPUSession(self._backends[name])
                session.connect()
                print(f"Auto-selected backend: {name!r}")
                try:
                    yield session
                finally:
                    session.disconnect()
                return
            except Exception as e:
                print(f"Backend {name!r} unreachable: {e}")
        raise RuntimeError("No reachable GPU backends found.")

    # ── Bulk operations ───────────────────────────────────────────────────────

    def ping_all(self) -> dict[str, bool]:
        """
        Check which backends are reachable (TCP connect only, no SSH handshake).
        Returns {name: reachable_bool}.
        """
        import socket

        results = {}
        for name, cfg in self._backends.items():
            try:
                if cfg.type == "cloudflare_tunnel":
                    # Just check DNS resolves — full reachability needs cloudflared
                    import socket

                    socket.getaddrinfo(cfg.tunnel_hostname, 443, timeout=5)
                    results[name] = True
                else:
                    with socket.create_connection((cfg.host, cfg.port), timeout=5):
                        results[name] = True
            except Exception:
                results[name] = False
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: load from environment variables
# ─────────────────────────────────────────────────────────────────────────────


def manager_from_env() -> GPUManager:
    """
    Build a GPUManager from environment variables.

    Reads JSON from GPU_BACKENDS env var:
        export GPU_BACKENDS='{"kaggle":{"type":"cloudflare_tunnel","tunnel_hostname":"abc.trycloudflare.com"}}'

    Or from a JSON file path:
        export GPU_BACKENDS_FILE=/path/to/backends.json
    """
    mgr = GPUManager()
    raw = os.environ.get("GPU_BACKENDS")
    file_path = os.environ.get("GPU_BACKENDS_FILE")

    if file_path and os.path.exists(file_path):
        with open(file_path) as f:
            raw = f.read()
    if raw:
        backends = json.loads(raw)
        for name, cfg in backends.items():
            mgr.add_backend(name, cfg)
    return mgr


# ─────────────────────────────────────────────────────────────────────────────
# CLI — quick test from the terminal
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test a remote GPU connection defined in gpu_connection.json"
    )
    parser.add_argument(
        "--json",
        default="gpu_connection.json",
        help="Path to gpu_connection.json produced by the Kaggle notebook",
    )
    parser.add_argument(
        "--key", default="~/.ssh/id_ed25519", help="Path to your SSH private key"
    )
    parser.add_argument(
        "--cmd",
        default='nvidia-smi && python -c "import torch; print(torch.cuda.get_device_name(0))"',
        help="Command to run on the remote GPU",
    )
    args = parser.parse_args()

    mgr = GPUManager()
    mgr.add_backend_from_json("kaggle", args.json)

    # override ssh_key if provided
    mgr._backends["kaggle"].ssh_key = args.key

    with mgr.use("kaggle") as gpu:
        print("\n── Remote GPU info ──────────────────────────────────────────")
        result = gpu.run(args.cmd)
        print("── Done ─────────────────────────────────────────────────────")
        if not result.ok():
            print(f"Exit code: {result.exit_code}")
            print(f"Stderr: {result.stderr[:500]}")
