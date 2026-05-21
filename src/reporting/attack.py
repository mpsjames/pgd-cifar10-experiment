"""Attack construction helpers for notebook reporting."""

from __future__ import annotations

from omegaconf import OmegaConf

from src.attacks.factory import replace_attack_epsilon
from src.attacks.pgd import PGDAttack
from src.experiments.config import AttackConfig
from src.experiments.config_loader import CONFIG_ROOT, load_attack_config


def build_attack_for_report(attack_name: str):
    from src.attacks.factory import AttackFactory

    return AttackFactory.build(load_attack_config(attack_name))


def epsilon_grid() -> list[float]:
    """Epsilon values from the canonical pgd_epsilon_sweep.yaml sweep config."""
    sweep = OmegaConf.load(CONFIG_ROOT / "sweeps" / "pgd_epsilon_sweep.yaml")
    return list(OmegaConf.to_container(sweep.epsilons, resolve=True))


def pgd_at_epsilon(epsilon: float) -> AttackConfig:
    base = load_attack_config("pgd_10")
    return replace_attack_epsilon(base, epsilon)


def pgd_attack_at_epsilon(epsilon: float) -> PGDAttack:
    return PGDAttack(pgd_at_epsilon(epsilon))
