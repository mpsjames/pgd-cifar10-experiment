"""Collect reproducibility metadata about the runtime environment."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch


def _run_git(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def log_environment(repo_root: Path | None = None) -> dict[str, Any]:
    """Return reproducibility metadata, including Git state when available.

    Args:
        repo_root: Optional repository root used for Git queries.

    Returns:
        Dictionary containing torch/CUDA versions, device name, Python
        version, platform, Git availability, Git commit/dirty state, and a
        UTC timestamp.
    """
    commit = _run_git(["rev-parse", "HEAD"], repo_root)
    status = _run_git(["status", "--porcelain"], repo_root)
    git_available = status is not None

    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
    else:
        device = "cpu"
        cuda_version = None

    return {
        "torch_version": torch.__version__,
        "cuda_version": cuda_version,
        "device": device,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_available": git_available,
        "git_commit": commit if git_available else None,
        "git_dirty": bool(status) if git_available else None,
        "timestamp": datetime.now(UTC).isoformat(),
    }
