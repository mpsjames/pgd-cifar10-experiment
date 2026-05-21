"""Shared constants for notebook reporting."""

from src.models.builders import ARCH_BUILDERS

ARCHES: list[str] = sorted(ARCH_BUILDERS)
SEED = 42
EPSILON_SWEEP_ARCHES = ["resnet18"]
NB03_SAMPLE_SIZE = 1000
NB05_SAMPLE_SIZE = 1000
NB06_NUM_SAMPLES = 8
SMOKE_SAMPLE_SIZE = 8
NB04_ATTACK_NAMES = [
    "fgsm",
    "bim_10",
    "pgd_10",
    "pgd_40",
    "pgd_100",
    "apgd_ce_10",
    "apgd_ce_100",
]
