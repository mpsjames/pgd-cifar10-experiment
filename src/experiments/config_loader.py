"""Load YAML configuration fragments into frozen project dataclasses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.experiments.config import (
    AttackConfig,
    ExperimentConfig,
    ModelConfig,
    TrackingConfig,
    TrainingConfig,
    ViTConfig,
    WRNConfig,
)


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


def load_experiment_config(
    config_root: Path = CONFIG_ROOT,
    arch: str | None = None,
    attack: str | None = None,
    training: str | None = None,
    seed: int | None = None,
    output_dir: Path | None = None,
    experiment_id: str = "pgd_cifar10",
) -> ExperimentConfig:
    """Load and compose the experiment configuration for one run.

    Args:
        config_root: Root directory containing the YAML fragments.
        arch: Optional architecture override. When `None`, use the default from
            `configs/config.yaml`.
        attack: Optional attack override.
        training: Optional training override.
        seed: Optional seed override.
        output_dir: Optional output-directory override.
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
    base = OmegaConf.load(config_root / "base" / f"{defaults['base']}.yaml")
    arch_cfg = OmegaConf.load(config_root / "architecture" / f"{arch_name}.yaml")
    attack_cfg = load_attack_config(attack_name, config_root) if attack_name else None
    training_cfg = (
        load_training_config(training_name, config_root) if training_name else None
    )
    if training_cfg is not None and arch_cfg.get("training") is not None:
        training_cfg = _override_training_batch(training_cfg, arch_cfg.training)

    model_cfg = _build_model_config(arch_cfg, base)
    return ExperimentConfig(
        experiment_id=experiment_id,
        seed=int(seed if seed is not None else root.seed),
        model=model_cfg,
        attack=attack_cfg,
        training=training_cfg,
        tracking=_load_tracking_config(base),
        output_dir=Path(output_dir if output_dir is not None else root.output_dir),
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
    required_keys = {"name", "epsilon", "alpha", "num_steps", "random_start", "norm"}
    optional_keys = {
        "p_init",
        "loss",
        "seed",
        "rho",
        "n_restarts",
    }  # Attack-family-specific; ignored by other attacks.
    _require_keys(resolved, required_keys, f"attack/{name}.yaml")
    extra = set(resolved) - (required_keys | optional_keys)
    if extra:
        raise ValueError(
            f"Unexpected AttackConfig keys in attack/{name}.yaml: {sorted(extra)}"
        )
    loss = resolved.get("loss")
    if loss is not None and loss not in {"margin", "cross_entropy"}:
        raise ValueError(
            f"attack/{name}.yaml: 'loss' must be 'margin' or 'cross_entropy', "
            f"got {loss!r}"
        )
    return AttackConfig(
        name=str(resolved["name"]),
        epsilon=float(resolved["epsilon"]),
        alpha=float(resolved["alpha"]),
        num_steps=int(resolved["num_steps"]),
        random_start=bool(resolved["random_start"]),
        norm=str(resolved["norm"]),  # type: ignore[arg-type]  # OmegaConf loses Literal["Linf"] at runtime
        p_init=float(resolved["p_init"]) if resolved.get("p_init") is not None else None,
        loss=loss,  # type: ignore[arg-type]  # mypy can't narrow str to Literal["margin","cross_entropy"]
        seed=int(resolved["seed"]) if resolved.get("seed") is not None else None,
        rho=float(resolved["rho"]) if resolved.get("rho") is not None else None,
        n_restarts=int(resolved["n_restarts"])
        if resolved.get("n_restarts") is not None
        else None,
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
    """
    raw = OmegaConf.load(config_root / "training" / f"{name}.yaml")
    resolved = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError(f"Training config {name} did not resolve to a mapping")
    inner = resolved.get("inner_attack")
    inner_attack = None
    if isinstance(inner, dict):
        inner_attack = AttackConfig(
            name=str(inner["name"]),
            epsilon=float(inner["epsilon"]),
            alpha=float(inner["alpha"]),
            num_steps=int(inner["num_steps"]),
            random_start=bool(inner["random_start"]),
            norm=str(inner["norm"]),  # type: ignore[arg-type]  # OmegaConf loses Literal["Linf"] at runtime
            seed=int(inner["seed"]) if inner.get("seed") is not None else None,
            rho=float(inner["rho"]) if inner.get("rho") is not None else None,
            n_restarts=int(inner["n_restarts"])
            if inner.get("n_restarts") is not None
            else None,
        )
    milestones_raw = resolved.get("lr_milestones")
    lr_milestones: tuple[int, ...] | None
    if milestones_raw is None:
        lr_milestones = None
    elif isinstance(milestones_raw, (list, tuple)):
        lr_milestones = tuple(int(m) for m in milestones_raw)
    else:
        raise TypeError(
            f"lr_milestones must be a list in training/{name}.yaml, got {type(milestones_raw).__name__}"
        )
    return TrainingConfig(
        mode=str(resolved["mode"]),  # type: ignore[arg-type]  # OmegaConf loses Literal["clean","adversarial"]
        epochs=int(resolved["epochs"]),
        batch_size=int(resolved["batch_size"]),
        lr=float(resolved["lr"]),
        weight_decay=float(resolved["weight_decay"]),
        optimizer=str(resolved["optimizer"]),  # type: ignore[arg-type]  # OmegaConf loses Literal["SGD"]
        scheduler=str(resolved["scheduler"]),  # type: ignore[arg-type]  # OmegaConf loses Literal["cosine","multistep"]
        momentum=float(resolved.get("momentum", 0.9)),
        use_amp=bool(resolved.get("use_amp", True)),
        grad_clip=resolved.get("grad_clip"),  # type: ignore[arg-type]  # dict[str,Any] lookup loses float|None
        inner_attack=inner_attack,
        resume_from=Path(resolved["resume_from"])
        if resolved.get("resume_from")
        else None,
        save_every_epochs=int(resolved.get("save_every_epochs", 5)),
        lr_milestones=lr_milestones,
        lr_gamma=float(resolved.get("lr_gamma", 0.1)),
    )


def _build_model_config(arch_cfg: Any, base: Any) -> ModelConfig:
    """Build a frozen `ModelConfig` from an architecture YAML and the base YAML.

    Reads architecture-specific hyperparameters from the optional `model:` block
    (ViT for `vit_tiny`, WRN for `wrn_34_10`); ResNet-18 has no extra fields.
    """
    arch = str(arch_cfg.arch)
    model_overrides = arch_cfg.get("model") or {}
    vit_cfg: ViTConfig | None = None
    wrn_cfg: WRNConfig | None = None
    if arch == "vit_tiny":
        vit_cfg = _build_vit_config(model_overrides)
    elif arch == "wrn_34_10":
        wrn_cfg = _build_wrn_config(model_overrides)
    return ModelConfig(
        arch=arch,  # type: ignore[arg-type]  # OmegaConf loses Literal type info at runtime
        checkpoint_path=None,
        num_classes=int(base.dataset.num_classes),
        cifar_mean=tuple(float(v) for v in base.dataset.mean),  # type: ignore[arg-type]  # generator loses tuple Literal
        cifar_std=tuple(float(v) for v in base.dataset.std),  # type: ignore[arg-type]  # generator loses tuple Literal
        vit=vit_cfg,
        wrn=wrn_cfg,
    )


def _build_vit_config(model_overrides: Any) -> ViTConfig:
    dropout = _optional_float(model_overrides, "dropout")
    attn_dropout = _optional_float(model_overrides, "attn_dropout")
    return ViTConfig(
        image_size=_optional_int(model_overrides, "image_size") or 32,
        patch_size=_optional_int(model_overrides, "patch_size") or 4,
        embed_dim=_optional_int(model_overrides, "embed_dim") or 192,
        depth=_optional_int(model_overrides, "depth") or 12,
        num_heads=_optional_int(model_overrides, "num_heads") or 3,
        mlp_ratio=_optional_float(model_overrides, "mlp_ratio") or 4.0,
        dropout=dropout if dropout is not None else 0.1,
        attn_dropout=attn_dropout if attn_dropout is not None else 0.0,
    )


def _build_wrn_config(model_overrides: Any) -> WRNConfig:
    dropout = _optional_float(model_overrides, "dropout")
    return WRNConfig(
        depth=_optional_int(model_overrides, "depth") or 34,
        widen_factor=_optional_int(model_overrides, "widen_factor") or 10,
        dropout=dropout if dropout is not None else 0.0,
    )


def _defaults(root: DictConfig) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in root.defaults:
        if isinstance(item, DictConfig):
            for key, value in item.items():
                if key != "_self_":
                    values[str(key)] = str(value)
    return values


def _override_training_batch(
    config: TrainingConfig, arch_training: Any
) -> TrainingConfig:
    batch_size = int(arch_training.get("batch_size", config.batch_size))
    use_amp = bool(arch_training.get("use_amp", config.use_amp))
    return TrainingConfig(
        mode=config.mode,
        epochs=config.epochs,
        batch_size=batch_size,
        lr=config.lr,
        weight_decay=config.weight_decay,
        optimizer=config.optimizer,
        scheduler=config.scheduler,
        momentum=config.momentum,
        use_amp=use_amp,
        grad_clip=config.grad_clip,
        inner_attack=config.inner_attack,
        resume_from=config.resume_from,
        save_every_epochs=config.save_every_epochs,
        lr_milestones=config.lr_milestones,
        lr_gamma=config.lr_gamma,
    )


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
        raise ValueError(
            f"Unexpected MLflow config keys in base/default.yaml: {sorted(extra)}"
        )
    return TrackingConfig(
        enable=bool(resolved.get("enable", True)),
        tracking_uri=str(resolved["tracking_uri"]),
        experiment_name=str(resolved["experiment_name"]),
        http_request_timeout_s=int(resolved["http_request_timeout_s"])
        if resolved.get("http_request_timeout_s") is not None
        else None,
        http_request_max_retries=int(resolved["http_request_max_retries"])
        if resolved.get("http_request_max_retries") is not None
        else None,
    )


def _optional_int(data: Any, key: str) -> int | None:
    value = data.get(key) if hasattr(data, "get") else None
    return int(value) if value is not None else None


def _optional_float(data: Any, key: str) -> float | None:
    value = data.get(key) if hasattr(data, "get") else None
    return float(value) if value is not None else None


def _require_keys(data: dict[str, object], keys: set[str], source: str) -> None:
    missing = keys - set(data)
    if missing:
        raise ValueError(f"Missing required keys in {source}: {sorted(missing)}")
