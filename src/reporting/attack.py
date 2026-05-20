"""Attack construction helpers for notebook reporting."""

from __future__ import annotations

from dataclasses import replace

from src.attacks.pgd import PGDAttack
from src.experiments.config import AttackConfig
from src.experiments.config_loader import load_attack_config


def build_attack_for_report(attack_name: str):
    from src.attacks.factory import AttackFactory

    return AttackFactory.build(load_attack_config(attack_name))


def epsilon_grid() -> list[float]:
    """10 epsilon values spanning 0 to 16/255."""
    base_eps = load_attack_config("pgd_10").epsilon
    return [round(i * (2 * base_eps) / 9, 6) for i in range(10)]


def pgd_at_epsilon(epsilon: float) -> AttackConfig:
    base = load_attack_config("pgd_10")
    return replace(
        base,
        epsilon=epsilon,
        alpha=min(base.alpha, epsilon) if epsilon > 0 else 0.0,
        random_start=base.random_start and epsilon > 0,
    )


def pgd_attack_at_epsilon(epsilon: float) -> PGDAttack:
    return PGDAttack(pgd_at_epsilon(epsilon))
