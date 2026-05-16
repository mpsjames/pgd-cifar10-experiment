from __future__ import annotations

import pytest
import torch
from torch import nn

from src.attacks.pgd import PGDAttack
from src.attacks.verify import verify_perturbation
from src.experiments.config import AttackConfig
from src.utils.seed import set_all_seeds


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
        self.seen_input_dtype: torch.dtype | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen_input_dtype = x.dtype
        return self.net(x)


def _pgd_config(
    random_start: bool = True,
    epsilon: float = 8 / 255,
    alpha: float = 2 / 255,
    steps: int = 3,
) -> AttackConfig:
    return AttackConfig(
        "PGD",
        epsilon=epsilon,
        alpha=alpha,
        num_steps=steps,
        random_start=random_start,
        norm="Linf",
    )


def test_linf_invariant() -> None:
    model = TinyClassifier()
    attack = PGDAttack(_pgd_config())
    x = torch.rand(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,), dtype=torch.long)
    x_adv = attack.perturb(model, x, y)
    verify_perturbation(x, x_adv, epsilon=8 / 255)


def test_zero_epsilon() -> None:
    model = TinyClassifier()
    attack = PGDAttack(_pgd_config(random_start=True, epsilon=0.0, alpha=0.0))
    x = torch.rand(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,), dtype=torch.long)
    x_adv = attack.perturb(model, x, y)
    assert torch.equal(x_adv, x)


def test_bim_is_pgd_no_random_deterministic() -> None:
    model = TinyClassifier()
    x = torch.rand(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,), dtype=torch.long)
    bim = PGDAttack(
        AttackConfig(
            "BIM",
            epsilon=8 / 255,
            alpha=2 / 255,
            num_steps=3,
            random_start=False,
            norm="Linf",
        )
    )
    pgd_no_random = PGDAttack(_pgd_config(random_start=False))
    assert torch.equal(bim.perturb(model, x, y), pgd_no_random.perturb(model, x, y))


def test_random_start_deterministic_with_seed() -> None:
    model = TinyClassifier()
    x = torch.rand(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,), dtype=torch.long)
    attack = PGDAttack(_pgd_config(random_start=True))
    set_all_seeds(42)
    a = attack.perturb(model, x, y)
    set_all_seeds(42)
    b = attack.perturb(model, x, y)
    assert torch.equal(a, b)


def test_pgd_runs_in_fp32_inside_autocast() -> None:
    model = TinyClassifier()
    attack = PGDAttack(_pgd_config(random_start=False, steps=1))
    x = torch.rand(2, 3, 32, 32)
    y = torch.randint(0, 10, (2,), dtype=torch.long)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        attack.perturb(model, x, y)
    assert model.seen_input_dtype == torch.float32


def test_torchattacks_deterministic_parity() -> None:
    torchattacks = pytest.importorskip("torchattacks")
    model = TinyClassifier().eval()
    x = torch.rand(2, 3, 32, 32)
    y = torch.randint(0, 10, (2,), dtype=torch.long)
    cfg = AttackConfig(
        "PGD",
        epsilon=8 / 255,
        alpha=2 / 255,
        num_steps=2,
        random_start=False,
        norm="Linf",
    )
    ours = PGDAttack(cfg).perturb(model, x, y)
    theirs = torchattacks.PGD(
        model, eps=cfg.epsilon, alpha=cfg.alpha, steps=cfg.num_steps, random_start=False
    )(x, y)
    assert torch.allclose(ours, theirs, atol=1e-5)


def test_torchattacks_asr_close() -> None:
    torchattacks = pytest.importorskip("torchattacks")
    model = TinyClassifier().eval()
    x = torch.rand(64, 3, 32, 32)
    y = torch.randint(0, 10, (64,), dtype=torch.long)
    cfg = AttackConfig(
        "PGD",
        epsilon=8 / 255,
        alpha=2 / 255,
        num_steps=2,
        random_start=True,
        norm="Linf",
    )
    set_all_seeds(42)
    ours = PGDAttack(cfg).perturb(model, x, y)
    set_all_seeds(42)
    theirs = torchattacks.PGD(
        model, eps=cfg.epsilon, alpha=cfg.alpha, steps=cfg.num_steps, random_start=True
    )(x, y)
    with torch.no_grad():
        ours_asr = (model(ours).argmax(dim=1) != y).float().mean().item()
        theirs_asr = (model(theirs).argmax(dim=1) != y).float().mean().item()
    assert abs(ours_asr - theirs_asr) < 0.02
