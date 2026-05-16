"""Implement Madry-style adversarial training and resume behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.attacks.base import BaseAttack
from src.attacks.verify import verify_perturbation
from src.experiments.config import TrainingConfig
from src.models.normalize_wrapper import NormalizedModel
from src.training.checkpoint import (
    capture_rng_state,
    load_resume_checkpoint,
    save_resume_checkpoint,
)
from src.training.clean import build_optimizer, build_scheduler


def adversarial_train_epoch(
    model: NormalizedModel,
    loader: DataLoader,
    optimizer: Optimizer,
    scaler: GradScaler,
    inner_attack: BaseAttack,
    device: torch.device,
    use_amp: bool = True,
) -> dict[str, float]:
    """Run one epoch of adversarial training with an inner attack.

    Args:
        model: `NormalizedModel` updated in place.
        loader: Training loader yielding raw `[0, 1]` CIFAR-10 batches.
        optimizer: Optimizer to step on adversarial inputs.
        scaler: AMP gradient scaler used when CUDA AMP is active.
        inner_attack: Attack used to generate adversarial examples for each
            batch.
        device: Target device.
        use_amp: Whether outer forward/backward passes may use AMP on CUDA.

    Returns:
        Dictionary containing mean adversarial `loss` and `acc_on_adv`.

    Notes:
        `verify_perturbation` is called on every batch so any attack bug fails
        before corrupting the training run.
    """
    total_loss = 0.0
    total_correct = 0
    total = 0
    amp_active = use_amp and device.type == "cuda"
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        model.eval()
        x_adv = inner_attack.perturb(model, x, y)
        verify_perturbation(
            x, x_adv, inner_attack.config.epsilon, inner_attack.config.norm
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_active):
            logits = model(x_adv)
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
    return {
        "loss": total_loss / max(total, 1),
        "acc_on_adv": total_correct / max(total, 1),
    }


def adversarial_train(
    model: NormalizedModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: TrainingConfig,
    tracker: Any,
    device: torch.device,
    arch: str,
    seed: int,
) -> Path:
    """Run Madry-style adversarial training with checkpoint-and-resume support.

    Args:
        model: `NormalizedModel` to train. Modified in place.
        train_loader: Training loader yielding raw `[0, 1]` CIFAR-10 batches.
        test_loader: Reserved for interface stability with the project plan.
            Currently unused by the loop.
        config: Adversarial-training config. Requires `config.inner_attack`
            whose name resolves to `"PGD"`.
        tracker: Active experiment tracker supporting `set_tags`,
            `log_params`, and `log_metrics`.
        device: Target training device.
        arch: Architecture name used in checkpoint naming.
        seed: Reproducibility seed used in checkpoint naming.

    Returns:
        Final adversarial-training checkpoint path.

    Raises:
        ValueError: When `config.inner_attack` is missing or not PGD.
        KeyboardInterrupt: Re-raised after a resume checkpoint is written.
        RuntimeError: Re-raised. CUDA OOMs trigger a resume checkpoint before
            propagation.
    """
    if config.inner_attack is None:
        raise ValueError("Adversarial training requires config.inner_attack")
    inner_name = config.inner_attack.name.upper()
    if inner_name != "PGD":
        raise ValueError(
            f"Adversarial training only supports PGD inner attack (plan §4.9); got name={config.inner_attack.name!r}"
        )
    from src.attacks.pgd import PGDAttack

    model.to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")
    inner_attack = PGDAttack(config.inner_attack)
    start_epoch = 0
    if config.resume_from is not None:
        start_epoch, _ = load_resume_checkpoint(
            config.resume_from, model, optimizer, scaler, scheduler
        )
        tracker.set_tags({"resumed_from_epoch": str(start_epoch)})
        tracker.log_params({"resumed_from_epoch": start_epoch})

    final_path = Path(f"checkpoints/adv/{arch}_madry_seed{seed}.pt")
    current_epoch = start_epoch
    try:
        for epoch in range(start_epoch, config.epochs):
            current_epoch = epoch
            metrics = adversarial_train_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                inner_attack,
                device,
                config.use_amp,
            )
            scheduler.step()
            tracker.log_metrics(metrics, step=epoch)
            if (epoch + 1) % config.save_every_epochs == 0:
                save_resume_checkpoint(
                    _resume_path(arch, seed, epoch),
                    model,
                    optimizer,
                    scaler,
                    scheduler,
                    epoch,
                    capture_rng_state(),
                )
        final_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "epochs": config.epochs}, final_path)
        return final_path
    except KeyboardInterrupt:
        save_resume_checkpoint(
            _resume_path(arch, seed, current_epoch),
            model,
            optimizer,
            scaler,
            scheduler,
            current_epoch,
            capture_rng_state(),
        )
        raise
    except RuntimeError as exc:
        if not _is_oom(exc):
            raise
        save_resume_checkpoint(
            _resume_path(arch, seed, current_epoch),
            model,
            optimizer,
            scaler,
            scheduler,
            current_epoch,
            capture_rng_state(),
        )
        raise


def _resume_path(arch: str, seed: int, epoch: int) -> Path:
    return Path("checkpoints/adv/_resume") / f"{arch}_seed{seed}_epoch{epoch}.pt"


def _is_oom(exc: RuntimeError) -> bool:
    return (
        isinstance(exc, torch.cuda.OutOfMemoryError)
        or "out of memory" in str(exc).lower()
    )
