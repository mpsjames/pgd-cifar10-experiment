"""Notebook helper for attack implementation validation (NB03)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.attacks.fgsm import FGSMAttack
from src.attacks.pgd import PGDAttack
from src.attacks.verify import verify_perturbation
from src.experiments.config import AttackConfig
from src.experiments.config_loader import load_attack_config
from src.reporting.constants import NB03_SAMPLE_SIZE, SEED
from src.reporting.data_loaders import evaluation_inputs
from src.reporting.io import ensure_result_dirs
from src.reporting.model_registry import build_fresh_model, reporting_model_pair
from src.utils.seed import set_all_seeds
from src.visualize.perturbation_panels import make_perturbation_panel


def nb03_attack_validation() -> Path:
    """Run FGSM/BIM/PGD validation and write invariant outputs."""
    ensure_result_dirs()
    set_all_seeds(SEED)
    models = reporting_model_pair("resnet18", SEED)
    model = models.clean.load_or_none()
    if model is None:
        model = build_fresh_model("resnet18")
    x, y = evaluation_inputs(NB03_SAMPLE_SIZE)
    # Explicit construction is intentional here: NB03 validates attack classes directly.
    attacks = [
        FGSMAttack(load_attack_config("fgsm")),
        PGDAttack(load_attack_config("bim_10")),
        PGDAttack(load_attack_config("pgd_10")),
    ]
    invariants = {
        "linf_holds": True,
        "pixel_domain_holds": True,
        "epsilon_zero_holds": False,
    }
    for attack in attacks:
        x_adv = attack.perturb(model, x, y)
        delta = (x_adv - x).abs().flatten(1).max(dim=1).values
        invariants["linf_holds"] = bool(
            invariants["linf_holds"] and (delta <= attack.config.epsilon + 1e-6).all()
        )
        invariants["pixel_domain_holds"] = bool(
            invariants["pixel_domain_holds"] and (x_adv >= 0).all() and (x_adv <= 1).all()
        )
        verify_perturbation(x, x_adv, attack.config.epsilon, attack.config.norm)

    pgd_zero_cfg = AttackConfig("PGD", 0.0, 0.0, 1, False, "Linf")
    zero = PGDAttack(pgd_zero_cfg).perturb(model, x, y)
    invariants["epsilon_zero_holds"] = bool(torch.equal(zero, x))
    fig = make_perturbation_panel(
        x[:1],
        attacks[-1].perturb(model, x[:1], y[:1]),
        "PGD-10 perturbation example\n"
        "CIFAR-10 test set, n=1000 sample protocol | ResNet-18 | "
        "epsilon=8/255, alpha=2/255, random_start=True | seeds={42}",
    )
    out = Path("results/figures/03_attack_validation.png")
    fig.savefig(out)
    plt.close(fig)
    Path("results/tables/03_invariants.json").write_text(
        json.dumps(invariants, indent=2), encoding="utf-8"
    )
    return out
