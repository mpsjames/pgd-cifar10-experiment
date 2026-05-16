"""Load YAML configuration fragments into frozen project dataclasses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.experiments.config import (
    AttackConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
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

    model_cfg = ModelConfig(
        arch=str(arch_cfg.arch),  # type: ignore[arg-type]
        checkpoint_path=None,
        num_classes=int(base.dataset.num_classes),
        cifar_mean=tuple(float(v) for v in base.dataset.mean),  # type: ignore[arg-type]
        cifar_std=tuple(float(v) for v in base.dataset.std),  # type: ignore[arg-type]
    )
    return ExperimentConfig(
        experiment_id=experiment_id,
        seed=int(seed if seed is not None else root.seed),
        model=model_cfg,
        attack=attack_cfg,
        training=training_cfg,
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
    _require_keys(
        resolved,
        {"name", "epsilon", "alpha", "num_steps", "random_start", "norm"},
        f"attack/{name}.yaml",
    )
    extra = set(resolved) - {
        "name",
        "epsilon",
        "alpha",
        "num_steps",
        "random_start",
        "norm",
    }
    if extra:
        raise ValueError(
            f"Unexpected AttackConfig keys in attack/{name}.yaml: {sorted(extra)}"
        )
    return AttackConfig(
        name=str(resolved["name"]),
        epsilon=float(resolved["epsilon"]),
        alpha=float(resolved["alpha"]),
        num_steps=int(resolved["num_steps"]),
        random_start=bool(resolved["random_start"]),
        norm=str(resolved["norm"]),  # type: ignore[arg-type]
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
            norm=str(inner["norm"]),  # type: ignore[arg-type]
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
        mode=str(resolved["mode"]),  # type: ignore[arg-type]
        epochs=int(resolved["epochs"]),
        batch_size=int(resolved["batch_size"]),
        lr=float(resolved["lr"]),
        weight_decay=float(resolved["weight_decay"]),
        optimizer=str(resolved["optimizer"]),  # type: ignore[arg-type]
        scheduler=str(resolved["scheduler"]),  # type: ignore[arg-type]
        momentum=float(resolved.get("momentum", 0.9)),
        use_amp=bool(resolved.get("use_amp", True)),
        grad_clip=resolved.get("grad_clip"),  # type: ignore[arg-type]
        inner_attack=inner_attack,
        resume_from=Path(resolved["resume_from"])
        if resolved.get("resume_from")
        else None,
        save_every_epochs=int(resolved.get("save_every_epochs", 5)),
        lr_milestones=lr_milestones,
        lr_gamma=float(resolved.get("lr_gamma", 0.1)),
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


def _require_keys(data: dict[str, object], keys: set[str], source: str) -> None:
    missing = keys - set(data)
    if missing:
        raise ValueError(f"Missing required keys in {source}: {sorted(missing)}")
