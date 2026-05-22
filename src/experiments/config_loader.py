"""Load YAML configuration fragments into frozen project dataclasses."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.experiments.config import (
    AttackConfig,
    ExperimentConfig,
    HardwareConfig,
    ModelConfig,
    TrackingConfig,
    TrainingConfig,
    ViTConfig,
)

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


def load_experiment_config(
    config_root: Path = CONFIG_ROOT,
    arch: str | None = None,
    attack: str | None = None,
    training: str | None = None,
    hardware: str | None = None,
    seed: int | None = None,
    experiment_id: str = "pgd_cifar10",
) -> ExperimentConfig:
    """Load and compose the experiment configuration for one run.

    Args:
        config_root: Root directory containing the YAML fragments.
        arch: Optional architecture override. When `None`, use the default from
            `configs/config.yaml`.
        attack: Optional attack override. When `None`, the default attack from
            `configs/config.yaml` is used — there is no way to disable attack
            loading via this argument. Use the returned config's `attack` field
            to detect or replace it post-load.
        training: Optional training override.
        hardware: Optional hardware preset override (stem under
            `configs/hardware/`). When `None`, use the default from
            `configs/config.yaml` (`gpu_default`).
        seed: Optional seed override.
        experiment_id: Logical experiment identifier attached to the returned
            config.

    Returns:
        Fully resolved frozen `ExperimentConfig`.
    """
    root = OmegaConf.load(config_root / "config.yaml")
    defaults = _defaults(root)
    arch_name = arch or defaults["architecture"]
    attack_name = attack or defaults.get("attack")
    training_name = training or defaults.get("training")
    hardware_name = hardware or defaults.get("hardware", "gpu_default")
    base = OmegaConf.load(config_root / "base" / f"{defaults['base']}.yaml")
    arch_cfg = OmegaConf.load(config_root / "architecture" / f"{arch_name}.yaml")
    attack_cfg = load_attack_config(attack_name, config_root) if attack_name else None
    training_cfg = load_training_config(training_name, config_root) if training_name else None
    if training_cfg is not None and arch_cfg.get("training") is not None:
        training_cfg = _override_training_from_arch(training_cfg, arch_cfg.training)
    hardware_cfg = load_hardware_config(hardware_name, config_root)

    run_seed = _validate_seed(int(seed if seed is not None else root.seed))
    model_cfg = _build_model_config(arch_cfg, base)
    return ExperimentConfig(
        experiment_id=experiment_id,
        seed=run_seed,
        model=model_cfg,
        attack=attack_cfg,
        training=training_cfg,
        tracking=_load_tracking_config(base),
        hardware=hardware_cfg,
    )


def load_hardware_config(name: str, config_root: Path = CONFIG_ROOT) -> HardwareConfig:
    """Load one hardware YAML preset into a `HardwareConfig`.

    Args:
        name: Hardware preset stem under `configs/hardware/`.
        config_root: Root config directory.

    Returns:
        Frozen `HardwareConfig` built from the resolved YAML mapping.

    Raises:
        TypeError: When the resolved config is not a mapping.
    """
    raw = OmegaConf.load(config_root / "hardware" / f"{name}.yaml")
    resolved = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError(f"Hardware config {name} did not resolve to a mapping")
    override = resolved.get("use_amp_override")
    device = str(resolved.get("device", "cuda"))
    if device not in {"cuda", "cpu"}:
        raise ValueError(f"hardware/{name}.yaml: device must be 'cuda' or 'cpu', got {device!r}")
    return HardwareConfig(
        device=device,  # type: ignore[arg-type]  # OmegaConf loses Literal["cuda","cpu"]
        num_workers=int(resolved.get("num_workers", 2)),
        pin_memory=bool(resolved.get("pin_memory", True)),
        persistent_workers=bool(resolved.get("persistent_workers", False)),
        prefetch_factor=int(resolved.get("prefetch_factor", 2)),
        cudnn_benchmark=bool(resolved.get("cudnn_benchmark", False)),
        cudnn_deterministic=bool(resolved.get("cudnn_deterministic", True)),
        use_amp_override=bool(override) if override is not None else None,
    )


def load_attack_config(name: str, config_root: Path = CONFIG_ROOT) -> AttackConfig:
    """Load one attack YAML file into an `AttackConfig`.

    Args:
        name: Attack config stem under `configs/attack/`.
        config_root: Root config directory.

    Returns:
        Frozen `AttackConfig` built from the resolved YAML mapping.

    Raises:
        TypeError: When the resolved config is not a mapping.
        ValueError: When required keys are missing or unexpected keys are
            present.
    """
    base = OmegaConf.load(config_root / "base" / "default.yaml")
    raw = OmegaConf.load(config_root / "attack" / f"{name}.yaml")
    cfg = OmegaConf.create({"base": base, "attack": raw})
    resolved = OmegaConf.to_container(cfg.attack, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError(f"Attack config {name} did not resolve to a mapping")
    return _build_attack_from_dict(resolved, f"attack/{name}.yaml")


def _build_attack_from_dict(resolved: dict, source: str) -> AttackConfig:
    """Build a frozen `AttackConfig` from a resolved YAML mapping.

    Args:
        resolved: Mapping of attack configuration keys.
        source: Human-readable source label used in error messages.

    Returns:
        Frozen `AttackConfig`.

    Raises:
        ValueError: When required keys are missing, unexpected keys are present,
            or the `loss` value is not a supported literal.
    """
    required_keys = {"name", "epsilon", "alpha", "num_steps", "random_start", "norm"}
    optional_keys = {
        "p_init",
        "loss",
        "seed",
        "rho",
        "n_restarts",
    }  # Attack-family-specific; ignored by other attacks.
    _require_keys(resolved, required_keys, source)
    extra = set(resolved) - (required_keys | optional_keys)
    if extra:
        raise ValueError(f"Unexpected AttackConfig keys in {source}: {sorted(extra)}")
    loss = resolved.get("loss")
    if loss is not None and loss not in {"margin", "cross_entropy"}:
        raise ValueError(f"{source}: 'loss' must be 'margin' or 'cross_entropy', got {loss!r}")
    epsilon = float(resolved["epsilon"])
    alpha = float(resolved["alpha"])
    num_steps = int(resolved["num_steps"])
    if epsilon < 0:
        raise ValueError(f"{source}: 'epsilon' must be >= 0, got {epsilon}")
    if alpha < 0:
        raise ValueError(f"{source}: 'alpha' must be >= 0, got {alpha}")
    if num_steps < 0:
        raise ValueError(f"{source}: 'num_steps' must be >= 0, got {num_steps}")
    p_init_val = float(resolved["p_init"]) if resolved.get("p_init") is not None else None
    if p_init_val is not None and not (0.0 < p_init_val <= 1.0):
        raise ValueError(f"{source}: 'p_init' must be in (0, 1], got {p_init_val}")
    return AttackConfig(
        name=str(resolved["name"]),
        epsilon=epsilon,
        alpha=alpha,
        num_steps=num_steps,
        random_start=bool(resolved["random_start"]),
        norm=str(resolved["norm"]),  # type: ignore[arg-type]  # OmegaConf loses Literal["Linf"] at runtime
        p_init=p_init_val,
        loss=loss,  # type: ignore[arg-type]  # mypy can't narrow str to Literal["margin","cross_entropy"]
        seed=int(resolved["seed"]) if resolved.get("seed") is not None else None,
        rho=float(resolved["rho"]) if resolved.get("rho") is not None else None,
        n_restarts=int(resolved["n_restarts"]) if resolved.get("n_restarts") is not None else None,
    )


_TRAINING_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "mode",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "optimizer",
        "scheduler",
        "momentum",
        "betas",
        "use_amp",
        "grad_clip",
        "inner_attack",
        "lr_milestones",
        "lr_gamma",
        "warmup_epochs",
        "min_lr",
        "label_smoothing",
        "mixup_alpha",
        "cutmix_alpha",
        "mixup_switch_prob",
        "val_every_n_epochs",
        "early_stopping_patience",
        "early_stopping_min_delta",
    }
)


def load_training_config(name: str, config_root: Path = CONFIG_ROOT) -> TrainingConfig:
    """Load one training YAML file into a `TrainingConfig`.

    Args:
        name: Training config stem under `configs/training/`.
        config_root: Root config directory.

    Returns:
        Frozen `TrainingConfig` with an embedded `AttackConfig` for
        `inner_attack` when present.

    Raises:
        TypeError: When the YAML does not resolve to a mapping or when
            `lr_milestones` has the wrong type.
        ValueError: When the YAML contains keys not in the allowed schema.
    """
    raw = OmegaConf.load(config_root / "training" / f"{name}.yaml")
    resolved = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError(f"Training config {name} did not resolve to a mapping")
    extra = set(resolved) - _TRAINING_CONFIG_KEYS
    if extra:
        raise ValueError(f"Unexpected TrainingConfig keys in training/{name}.yaml: {sorted(extra)}")
    inner = resolved.get("inner_attack")
    inner_attack = None
    if isinstance(inner, dict):
        inner_attack = _build_attack_from_dict(inner, f"training/{name}.yaml:inner_attack")
    if str(resolved["mode"]) == "adversarial" and inner_attack is None:
        raise ValueError(f"training/{name}.yaml: adversarial mode requires inner_attack")
    milestones_raw = resolved.get("lr_milestones")
    lr_milestones: tuple[int, ...] | None
    if milestones_raw is None:
        lr_milestones = None
    elif isinstance(milestones_raw, (list, tuple)):
        lr_milestones = tuple(int(m) for m in milestones_raw)
    else:
        raise TypeError(
            f"lr_milestones must be a list in training/{name}.yaml, "
            f"got {type(milestones_raw).__name__}"
        )
    betas_raw = resolved.get("betas")
    betas: tuple[float, float] = (0.9, 0.999)
    if betas_raw is not None:
        if not isinstance(betas_raw, (list, tuple)) or len(betas_raw) != 2:
            raise TypeError(
                f"betas must be a 2-element list in training/{name}.yaml, got {betas_raw!r}"
            )
        betas = (float(betas_raw[0]), float(betas_raw[1]))
    grad_clip_raw = resolved.get("grad_clip")
    _epochs = int(resolved["epochs"])
    _batch_size = int(resolved["batch_size"])
    _lr = float(resolved["lr"])
    _val_every = int(resolved.get("val_every_n_epochs", 1))
    _switch_prob = float(resolved.get("mixup_switch_prob", 0.5))
    if _epochs <= 0:
        raise ValueError(f"training/{name}.yaml: 'epochs' must be > 0, got {_epochs}")
    if _batch_size <= 0:
        raise ValueError(f"training/{name}.yaml: 'batch_size' must be > 0, got {_batch_size}")
    if _lr <= 0:
        raise ValueError(f"training/{name}.yaml: 'lr' must be > 0, got {_lr}")
    if _val_every <= 0:
        raise ValueError(
            f"training/{name}.yaml: 'val_every_n_epochs' must be > 0, got {_val_every}"
        )
    if not (0.0 <= _switch_prob <= 1.0):
        raise ValueError(
            f"training/{name}.yaml: 'mixup_switch_prob' must be in [0, 1], got {_switch_prob}"
        )
    return TrainingConfig(
        mode=str(resolved["mode"]),  # type: ignore[arg-type]  # OmegaConf loses Literal["clean","adversarial"]
        epochs=_epochs,
        batch_size=_batch_size,
        lr=_lr,
        weight_decay=float(resolved["weight_decay"]),
        optimizer=str(resolved["optimizer"]),  # type: ignore[arg-type]  # OmegaConf loses Literal["SGD","AdamW"]
        scheduler=str(resolved["scheduler"]),  # type: ignore[arg-type]  # OmegaConf loses Literal["cosine","multistep"]
        momentum=float(resolved.get("momentum", 0.9)),
        betas=betas,
        use_amp=bool(resolved.get("use_amp", True)),
        grad_clip=float(grad_clip_raw) if grad_clip_raw is not None else None,
        inner_attack=inner_attack,
        lr_milestones=lr_milestones,
        lr_gamma=float(resolved.get("lr_gamma", 0.1)),
        warmup_epochs=int(resolved.get("warmup_epochs", 0)),
        min_lr=float(resolved.get("min_lr", 0.0)),
        label_smoothing=float(resolved.get("label_smoothing", 0.0)),
        mixup_alpha=float(resolved.get("mixup_alpha", 0.0)),
        cutmix_alpha=float(resolved.get("cutmix_alpha", 0.0)),
        mixup_switch_prob=_switch_prob,
        val_every_n_epochs=_val_every,
        early_stopping_patience=int(resolved.get("early_stopping_patience", 0)),
        early_stopping_min_delta=float(resolved.get("early_stopping_min_delta", 0.0)),
    )


def _build_model_config(arch_cfg: Any, base: Any) -> ModelConfig:
    """Build a frozen `ModelConfig` from an architecture YAML and the base YAML.

    Reads architecture-specific hyperparameters from the optional `model:` block
    (ViT for `vit_tiny`); ResNet-18 has no extra fields.
    """
    arch = str(arch_cfg.arch)
    model_overrides = arch_cfg.get("model") or {}
    vit_cfg: ViTConfig | None = None
    if arch == "vit_tiny":
        vit_cfg = _build_vit_config(model_overrides)
    return ModelConfig(
        arch=arch,  # type: ignore[arg-type]  # OmegaConf loses Literal type info at runtime
        checkpoint_path=None,
        num_classes=int(base.dataset.num_classes),
        cifar_mean=tuple(float(v) for v in base.dataset.mean),  # type: ignore[arg-type]  # generator loses tuple Literal
        cifar_std=tuple(float(v) for v in base.dataset.std),  # type: ignore[arg-type]  # generator loses tuple Literal
        vit=vit_cfg,
    )


def _build_vit_config(model_overrides: Any) -> ViTConfig:
    dropout = _optional_float(model_overrides, "dropout")
    attn_dropout = _optional_float(model_overrides, "attn_dropout")
    drop_path = _optional_float(model_overrides, "drop_path")
    return ViTConfig(
        image_size=_optional_int(model_overrides, "image_size") or 32,
        patch_size=_optional_int(model_overrides, "patch_size") or 4,
        embed_dim=_optional_int(model_overrides, "embed_dim") or 192,
        depth=_optional_int(model_overrides, "depth") or 12,
        num_heads=_optional_int(model_overrides, "num_heads") or 3,
        mlp_ratio=_optional_float(model_overrides, "mlp_ratio") or 4.0,
        dropout=dropout if dropout is not None else 0.1,
        attn_dropout=attn_dropout if attn_dropout is not None else 0.0,
        drop_path=drop_path if drop_path is not None else 0.0,
    )


def _defaults(root: DictConfig) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in root.defaults:
        if isinstance(item, DictConfig):
            for key, value in item.items():
                if key != "_self_":
                    values[str(key)] = str(value)
    return values


_ARCH_TRAINING_OVERRIDABLE: dict[str, type] = {
    "batch_size": int,
    "use_amp": bool,
    "lr": float,
    "weight_decay": float,
    "optimizer": str,
    "scheduler": str,
    "momentum": float,
    "grad_clip": float,
    "warmup_epochs": int,
    "min_lr": float,
    "label_smoothing": float,
    "mixup_alpha": float,
    "cutmix_alpha": float,
    "mixup_switch_prob": float,
    "early_stopping_patience": int,
    "early_stopping_min_delta": float,
}


def _override_training_from_arch(config: TrainingConfig, arch_training: Any) -> TrainingConfig:
    """Layer arch-YAML `training:` block over a base `TrainingConfig`.

    Recognized scalar keys are merged via `dataclasses.replace`. `betas` is
    handled separately because it is a 2-tuple. Unknown keys raise so config
    typos surface immediately instead of being silently ignored.
    """
    updates: dict[str, Any] = {}
    for key in arch_training:
        key_str = str(key)
        if key_str == "betas":
            betas_raw = arch_training[key]
            try:
                betas_list = list(betas_raw)
            except TypeError as exc:
                raise TypeError(
                    f"betas override must be a 2-element list, got {betas_raw!r}"
                ) from exc
            if len(betas_list) != 2:
                raise TypeError(f"betas override must be a 2-element list, got {betas_raw!r}")
            updates["betas"] = (float(betas_list[0]), float(betas_list[1]))
            continue
        if key_str not in _ARCH_TRAINING_OVERRIDABLE:
            raise ValueError(f"Unknown arch training override key: {key_str!r}")
        caster = _ARCH_TRAINING_OVERRIDABLE[key_str]
        updates[key_str] = caster(arch_training[key])
    merged: TrainingConfig = replace(config, **updates)  # type: ignore[assignment]
    return merged


def _load_tracking_config(base: DictConfig) -> TrackingConfig:
    raw = base.get("mlflow")
    if raw is None:
        return TrackingConfig(
            enable=True,
            tracking_uri="http://127.0.0.1:5000",
            experiment_name="pgd_cifar10_multiarch",
        )
    required_keys = {"tracking_uri", "experiment_name"}
    optional_keys = {"enable", "http_request_timeout_s", "http_request_max_retries"}
    resolved = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("base/default.yaml mlflow section did not resolve to a mapping")
    _require_keys(resolved, required_keys, "base/default.yaml:mlflow")
    extra = set(resolved) - (required_keys | optional_keys)
    if extra:
        raise ValueError(f"Unexpected MLflow config keys in base/default.yaml: {sorted(extra)}")
    return TrackingConfig(
        enable=bool(resolved.get("enable", True)),
        tracking_uri=str(resolved["tracking_uri"]),
        experiment_name=str(resolved["experiment_name"]),
        http_request_timeout_s=(
            int(resolved["http_request_timeout_s"])
            if resolved.get("http_request_timeout_s") is not None
            else None
        ),
        http_request_max_retries=(
            int(resolved["http_request_max_retries"])
            if resolved.get("http_request_max_retries") is not None
            else None
        ),
    )


def _optional_int(data: Any, key: str) -> int | None:
    value = data.get(key) if hasattr(data, "get") else None
    return int(value) if value is not None else None


def _optional_float(data: Any, key: str) -> float | None:
    value = data.get(key) if hasattr(data, "get") else None
    return float(value) if value is not None else None


def _validate_seed(seed: int) -> int:
    if seed <= 0:
        raise ValueError(f"seed must be a positive non-zero integer, got {seed}")
    return seed


def _require_keys(data: dict[str, object], keys: set[str], source: str) -> None:
    missing = keys - set(data)
    if missing:
        raise ValueError(f"Missing required keys in {source}: {sorted(missing)}")
