from __future__ import annotations

import socket
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def mlflow_server(tmp_path: Path):
    """Spawn an ephemeral MLflow tracking server bound to a free port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    artifacts = tmp_path / "mlartifacts"
    artifacts.mkdir()
    proc = subprocess.Popen(
        [
            "mlflow",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--backend-store-uri",
            f"file://{tmp_path / 'mlruns'}",
            "--artifacts-destination",
            str(artifacts),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"{url}/health", timeout=0.5).close()
            break
        except OSError:
            time.sleep(0.2)
    else:
        proc.terminate()
        raise RuntimeError(f"mlflow server did not become ready on {url}")
    try:
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=5)
