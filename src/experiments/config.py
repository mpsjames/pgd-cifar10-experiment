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
        name: Attack family name such as `"FGSM"`, `"BIM"`, `"PGD"`, or
            `"Square"`.
        epsilon: Linf budget in normalized image space `[0, 1]`.
        alpha: Step size in normalized image space `[0, 1]`. Unused by Square.
        num_steps: Number of iterative updates. FGSM effectively uses 1.
            For `Square`, this is the per-sample query budget.
        random_start: When True, initialize uniformly inside the epsilon ball.
        norm: Perturbation norm. Only `"Linf"` is supported in this project.
        p_init: Square-only initial fraction of pixels per perturbed square;
            `None` for non-Square attacks.
        loss: Square-only inner loss name (`"margin"` or `"cross_entropy"`);
            `None` for non-Square attacks.
        seed: Square-only per-attack RNG seed used to build a local
            `torch.Generator` (principles §4.1: no global-RNG mutation).
            `None` for non-Square attacks.
        rho: APGD-only checkpoint-progress threshold. `None` for other attacks.
        n_restarts: APGD-only number of random restarts. `None` for other
            attacks.
    """

    name: str
    epsilon: float
    alpha: float
    num_steps: int
    random_start: bool
    norm: Literal["Linf"]
    p_init: float | None = None
    loss: Literal["margin", "cross_entropy"] | None = None
    seed: int | None = None
    rho: float | None = None
    n_restarts: int | None = None


@dataclass(frozen=True)
class TrackingConfig:
    """Describe experiment tracking sinks loaded from `configs/base/default.yaml`.

    Fields:
        enable: Whether MLflow HTTP tracking is enabled. JSON and file logging
            remain active regardless of this flag.
        tracking_uri: MLflow tracking-server HTTP API URL.
        experiment_name: MLflow experiment name and JSON mirror namespace.
        http_request_timeout_s: Optional MLflow HTTP request timeout.
        http_request_max_retries: Optional MLflow HTTP retry count.
    """

    enable: bool
    tracking_uri: str
    experiment_name: str
    http_request_timeout_s: int | None = None
    http_request_max_retries: int | None = None


@dataclass(frozen=True)
class HardwareConfig:
    """Runtime execution settings (separate from algorithmic TrainingConfig).

    Fields:
        device: Compute device, "cuda" or "cpu".
        num_workers: DataLoader worker processes.
        pin_memory: Use page-locked memory for faster host->GPU copy.
        persistent_workers: Keep workers alive across epochs.
        prefetch_factor: Batches each worker pre-fetches; PyTorch default is 2.
        cudnn_benchmark: Enable autotuner for fastest conv kernels.
        cudnn_deterministic: Require deterministic algorithms.
        use_amp_override: When not None, force AMP on/off regardless of
            TrainingConfig.use_amp. Use False on Pascal GPUs (P40, P100) that
            lack Tensor Cores.
    """

    device: Literal["cuda", "cpu"] = "cuda"
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2
    cudnn_benchmark: bool = False
    cudnn_deterministic: bool = True
    use_amp_override: bool | None = None


@dataclass(frozen=True)
class ViTConfig:
    """ViT-Tiny architecture hyperparameters (meaningless for CNN variants).

    Fields:
        image_size: Input spatial resolution; CIFAR-10 native is 32.
        patch_size: Side length of each non-overlapping patch token.
        embed_dim: Token embedding dimension.
        depth: Number of transformer encoder blocks.
        num_heads: Number of attention heads (must divide embed_dim).
        mlp_ratio: Hidden-to-embed ratio inside the MLP block.
        dropout: Drop probability applied after attention and MLP residuals.
        attn_dropout: Drop probability inside `nn.MultiheadAttention`.
    """

    image_size: int = 32
    patch_size: int = 4
    embed_dim: int = 192
    depth: int = 12
    num_heads: int = 3
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attn_dropout: float = 0.0


@dataclass(frozen=True)
class WRNConfig:
    """WideResNet architecture hyperparameters (meaningless for ResNet/ViT)."""

    depth: int = 34
    widen_factor: int = 10
    dropout: float = 0.0


@dataclass(frozen=True)
class ModelConfig:
    """Describe model-construction settings shared across experiments.

    Fields:
        arch: Architecture key consumed by `build_model` and the legacy `ARCH_BUILDERS` registry.
        checkpoint_path: Optional external checkpoint path. Most entry points
            construct the canonical path separately and leave this as `None`.
        num_classes: Number of output classes; CIFAR-10 uses 10.
        cifar_mean: Channel-wise CIFAR-10 mean used by `Normalizer`.
        cifar_std: Channel-wise CIFAR-10 standard deviation used by
            `Normalizer`.
        vit: ViT-specific hyperparameters; `None` for non-ViT architectures.
        wrn: WideResNet-specific hyperparameters; `None` for non-WRN architectures.
    """

    arch: Literal["resnet18", "wrn_34_10", "vit_tiny"]
    checkpoint_path: Path | None
    num_classes: int = 10
    cifar_mean: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    cifar_std: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)
    vit: ViTConfig | None = None
    wrn: WRNConfig | None = None


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
        lr_milestones: Required when `scheduler == "multistep"`.
        lr_gamma: Multiplicative factor used by the multistep scheduler.
        val_every_n_epochs: Run validation every N epochs (and always on the
            final epoch). Default 1 = validate every epoch.
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
    lr_milestones: tuple[int, ...] | None = None
    lr_gamma: float = 0.1
    val_every_n_epochs: int = 1


@dataclass(frozen=True)
class ExperimentConfig:
    """Bundle the resolved project configuration for one experiment run.

    Fields:
        experiment_id: Logical experiment identifier used in logs and outputs.
        seed: Global reproducibility seed.
        model: Model-construction settings.
        attack: Optional attack settings for evaluation-style entry points.
        training: Optional training settings for training entry points.
        tracking: Experiment tracking settings.
        hardware: Runtime hardware preset (workers, pin_memory, cudnn flags,
            AMP override) selected via `configs/hardware/<name>.yaml`.
        output_dir: Root output directory declared by the composed config.
    """

    experiment_id: str
    seed: int
    model: ModelConfig
    attack: AttackConfig | None
    training: TrainingConfig | None
    tracking: TrackingConfig
    hardware: HardwareConfig
    output_dir: Path
