from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.experiments.config import AttackConfig


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


class _TinyClassifier(nn.Module):
    """3x32x32 -> 10 linear classifier for fast unit tests."""

    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3 * 32 * 32, 10)
        self.call_count = 0
        self.seen_input_dtype: torch.dtype | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        self.seen_input_dtype = x.dtype
        return self.fc(self.flatten(x))


@pytest.fixture
def tiny_classifier() -> nn.Module:
    return _TinyClassifier()


@pytest.fixture
def tiny_classifier_factory():
    return _TinyClassifier


class _IdentityAttack:
    def __init__(self, config: AttackConfig) -> None:
        self.config = config

    def perturb(self, _model, x: torch.Tensor, _y: torch.Tensor) -> torch.Tensor:
        return x.detach().clone()


@pytest.fixture
def identity_attack_factory():
    return _IdentityAttack


class _DummyTracker:
    def __init__(self, *_args, **_kwargs) -> None:
        self.metrics: dict[str, float] = {}
        self.params: dict[str, object] = {}
        self.tags: dict[str, str] = {}

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def log_metrics(self, metrics, step=None) -> None:
        self.metrics.update(metrics)

    def log_params(self, params) -> None:
        self.params.update(params)

    def set_tags(self, tags) -> None:
        self.tags.update(tags)

    def log_artifact(self, *_args, **_kwargs) -> None:
        return None


@pytest.fixture
def dummy_tracker():
    return _DummyTracker()


@pytest.fixture
def dummy_tracker_factory():
    return _DummyTracker


@pytest.fixture
def tiny_batch():
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(2, 3, 32, 32, generator=generator)
    y = torch.tensor([0, 1], dtype=torch.long)
    return x, y


@pytest.fixture
def tiny_loader(tiny_batch):
    x, y = tiny_batch
    return DataLoader(TensorDataset(x, y), batch_size=2)


@pytest.fixture
def fgsm_config() -> AttackConfig:
    return AttackConfig("FGSM", 8 / 255, 8 / 255, 1, False, "Linf")


@pytest.fixture
def pgd_config() -> AttackConfig:
    return AttackConfig("PGD", 8 / 255, 2 / 255, 3, True, "Linf")


@pytest.fixture
def apgd_config() -> AttackConfig:
    return AttackConfig(
        "APGD-CE",
        epsilon=8 / 255,
        alpha=8 / 255,
        num_steps=5,
        random_start=True,
        norm="Linf",
        seed=42,
        rho=0.75,
        n_restarts=1,
    )


@pytest.fixture
def square_config() -> AttackConfig:
    return AttackConfig(
        "Square",
        epsilon=8 / 255,
        alpha=8 / 255,
        num_steps=16,
        random_start=True,
        norm="Linf",
        p_init=0.05,
        loss="margin",
        seed=42,
    )


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
            sys.executable,
            "-m",
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
