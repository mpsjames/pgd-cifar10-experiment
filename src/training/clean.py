"""Implement the clean-training loop and its optimizer/scheduler builders."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader

from src.experiments.config import TrainingConfig
from src.models.normalize_wrapper import NormalizedModel


def clean_train_epoch(
    model: NormalizedModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool = True,
) -> dict[str, float]:
    """Run one epoch of clean training.

    Args:
        model: `NormalizedModel` updated in place.
        loader: Training loader yielding `(x, y)` with raw `[0, 1]` images.
        optimizer: Optimizer to step each batch.
        scaler: AMP gradient scaler used when CUDA AMP is active.
        device: Target device for the epoch.
        use_amp: Whether AMP may be enabled on CUDA.

    Returns:
        Dictionary containing mean cross-entropy `loss` and clean `acc`.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    amp_active = use_amp and device.type == "cuda"
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_active):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        if amp_active:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach().item()) * y.size(0)
        total_correct += int((logits.detach().argmax(dim=1) == y).sum().item())
        total += y.size(0)
    return {"loss": total_loss / max(total, 1), "acc": total_correct / max(total, 1)}


def build_optimizer(
    model: torch.nn.Module, config: TrainingConfig
) -> torch.optim.Optimizer:
    """Build the optimizer declared by `TrainingConfig`.

    Args:
        model: Model whose parameters will be optimized.
        config: Training config. Reads `config.optimizer`, `config.lr`,
            `config.momentum`, and `config.weight_decay`.

    Returns:
        Configured torch optimizer instance.

    Raises:
        ValueError: When `config.optimizer` is unsupported.
    """
    if config.optimizer != "SGD":
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    return SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig):
    """Build the learning-rate scheduler declared by `TrainingConfig`.

    Args:
        optimizer: Optimizer whose learning rate should be scheduled.
        config: Training config. Reads `config.scheduler`, `config.epochs`,
            `config.lr_milestones`, and `config.lr_gamma`.

    Returns:
        Configured scheduler instance.

    Raises:
        ValueError: When the scheduler is unsupported or missing required
            milestone settings.
    """
    if config.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config.epochs)
    if config.scheduler == "multistep":
        if config.lr_milestones is None:
            raise ValueError(
                "multistep scheduler requires config.lr_milestones to be set; "
                "add lr_milestones (and lr_gamma) to the training YAML."
            )
        return MultiStepLR(
            optimizer, milestones=list(config.lr_milestones), gamma=config.lr_gamma
        )
    raise ValueError(f"Unsupported scheduler: {config.scheduler}")
