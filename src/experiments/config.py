"""Declare immutable experiment configuration dataclasses.

These dataclasses are loaded from YAML once and then treated as frozen API
objects throughout training, evaluation, and notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class AttackConfig:
    """Describe one attack configuration loaded from `configs/attack/`.

    Fields:
        name: Attack family name such as `"FGSM"`, `"BIM"`, or `"PGD"`.
        epsilon: Linf budget in normalized image space `[0, 1]`.
        alpha: Step size in normalized image space `[0, 1]`.
        num_steps: Number of iterative updates. FGSM effectively uses 1.
        random_start: When True, initialize uniformly inside the epsilon ball.
        norm: Perturbation norm. Only `"Linf"` is supported in this project.
    """

    name: str
    epsilon: float
    alpha: float
    num_steps: int
    random_start: bool
    norm: Literal["Linf"]


@dataclass(frozen=True)
class ModelConfig:
    """Describe model-construction settings shared across experiments.

    Fields:
        arch: Architecture key consumed by `ARCH_BUILDERS`.
        checkpoint_path: Optional external checkpoint path. Most entry points
            construct the canonical path separately and leave this as `None`.
        num_classes: Number of output classes; CIFAR-10 uses 10.
        cifar_mean: Channel-wise CIFAR-10 mean used by `NormalizedModel`.
        cifar_std: Channel-wise CIFAR-10 standard deviation used by
            `NormalizedModel`.
    """

    arch: Literal["resnet18", "wrn_34_10", "resnet50", "vgg16_bn"]
    checkpoint_path: Path | None
    num_classes: int = 10
    cifar_mean: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    cifar_std: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)


@dataclass(frozen=True)
class TrainingConfig:
    """Describe training-loop hyperparameters for clean or adversarial runs.

    Fields:
        mode: Training phase, either `"clean"` or `"adversarial"`.
        epochs: Number of epochs to execute.
        batch_size: Per-step batch size.
        lr: Initial learning rate.
        weight_decay: Optimizer weight decay.
        optimizer: Optimizer family. Only `"SGD"` is supported.
        scheduler: Scheduler family: `"cosine"` or `"multistep"`.
        momentum: SGD momentum.
        use_amp: Whether mixed precision may be used on CUDA devices.
        grad_clip: Optional gradient clip value. Currently reserved for future
            use by training entry points.
        inner_attack: Required for adversarial training; `None` for clean
            training.
        resume_from: Optional resume-checkpoint path.
        save_every_epochs: Resume-checkpoint cadence.
        lr_milestones: Required when `scheduler == "multistep"`.
        lr_gamma: Multiplicative factor used by the multistep scheduler.
    """

    mode: Literal["clean", "adversarial"]
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    optimizer: Literal["SGD"]
    scheduler: Literal["cosine", "multistep"]
    momentum: float = 0.9
    use_amp: bool = True
    grad_clip: float | None = None
    inner_attack: AttackConfig | None = None
    resume_from: Path | None = None
    save_every_epochs: int = 5
    lr_milestones: tuple[int, ...] | None = None
    lr_gamma: float = 0.1


@dataclass(frozen=True)
class ExperimentConfig:
    """Bundle the resolved project configuration for one experiment run.

    Fields:
        experiment_id: Logical experiment identifier used in logs and outputs.
        seed: Global reproducibility seed.
        model: Model-construction settings.
        attack: Optional attack settings for evaluation-style entry points.
        training: Optional training settings for training entry points.
        output_dir: Root output directory declared by the composed config.
    """

    experiment_id: str
    seed: int
    model: ModelConfig
    attack: AttackConfig | None
    training: TrainingConfig | None
    output_dir: Path
