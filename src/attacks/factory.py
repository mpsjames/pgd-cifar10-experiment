"""Build attack objects from frozen attack configs."""

from __future__ import annotations

from src.attacks.apgd import APGDAttack
from src.attacks.base import BaseAttack
from src.attacks.fgsm import FGSMAttack
from src.attacks.pgd import PGDAttack
from src.attacks.square import SquareAttack
from src.experiments.config import AttackConfig


def replace_attack_epsilon(cfg: AttackConfig, epsilon: float) -> AttackConfig:
    """Return a copy of *cfg* with epsilon replaced and dependent fields adjusted.

    Args:
        cfg: Source attack config.
        epsilon: New Linf budget.

    Returns:
        New `AttackConfig` with updated `epsilon`, `alpha` (clamped to the new
        budget), and `random_start` (disabled when epsilon is zero).
    """
    import dataclasses

    return dataclasses.replace(
        cfg,
        epsilon=epsilon,
        alpha=min(cfg.alpha, epsilon) if epsilon > 0 else 0.0,
        random_start=cfg.random_start and epsilon > 0,
    )


class AttackFactory:
    """Registry-based factory for attack implementations.

    To add a new attack without modifying this class:
        AttackFactory.register("MY_ATTACK", MyAttack)
    """

    _registry: dict[str, type[BaseAttack]] = {
        "FGSM": FGSMAttack,
        "BIM": PGDAttack,
        "PGD": PGDAttack,
        "APGD-CE": APGDAttack,
        "SQUARE": SquareAttack,
    }

    @classmethod
    def build(cls, config: AttackConfig) -> BaseAttack:
        attack_cls = cls._registry.get(config.name.upper())
        if attack_cls is None:
            supported = ", ".join(sorted(cls._registry))
            raise ValueError(f"Unsupported attack: {config.name!r}. Supported: {supported}")
        return attack_cls(config)

    @classmethod
    def register(cls, name: str, attack_cls: type[BaseAttack]) -> None:
        """Register a new attack class under the given name."""
        cls._registry[name.upper()] = attack_cls
