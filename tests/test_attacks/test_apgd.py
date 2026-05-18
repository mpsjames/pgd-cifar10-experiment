from __future__ import annotations

import torch
from torch import nn

from src.attacks.apgd import APGDAttack
from src.attacks.verify import verify_perturbation
from src.experiments.config import AttackConfig


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
        self.seen_input_dtype: torch.dtype | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen_input_dtype = x.dtype
        return self.net(x)


class ZeroGradientClassifier(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.flatten(1).sum(dim=1) * 0.0
        return z[:, None].expand(x.size(0), 10)


def _apgd_config(
    epsilon: float = 8 / 255,
    steps: int = 5,
    seed: int = 42,
    random_start: bool = True,
    n_restarts: int = 1,
) -> AttackConfig:
    return AttackConfig(
        name="APGD-CE",
        epsilon=epsilon,
        alpha=epsilon,
        num_steps=steps,
        random_start=random_start,
        norm="Linf",
        seed=seed,
        rho=0.75,
        n_restarts=n_restarts,
    )


def test_apgd_respects_linf_and_pixel_domain() -> None:
    torch.manual_seed(0)
    model = TinyClassifier()
    attack = APGDAttack(_apgd_config())
    x = torch.rand(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    x_adv = attack.perturb(model, x, y)

    verify_perturbation(x, x_adv, epsilon=8 / 255)
    assert x_adv.dtype == x.dtype
    assert x_adv.shape == x.shape


def test_apgd_zero_epsilon_identity() -> None:
    model = TinyClassifier()
    attack = APGDAttack(_apgd_config(epsilon=0.0))
    x = torch.rand(2, 3, 32, 32)
    y = torch.tensor([0, 1], dtype=torch.long)

    assert torch.equal(attack.perturb(model, x, y), x)


def test_apgd_loss_monotone_in_best() -> None:
    torch.manual_seed(0)
    model = TinyClassifier()
    attack = APGDAttack(_apgd_config(steps=6))
    x = torch.rand(3, 3, 32, 32)
    y = torch.tensor([0, 1, 2], dtype=torch.long)

    attack.perturb(model, x, y)

    for trajectory in attack.debug_best_loss_history:
        history = torch.stack(trajectory)
        assert torch.all(history[1:] >= history[:-1] - 1e-6)


def test_apgd_step_size_halves_on_no_improvement() -> None:
    model = ZeroGradientClassifier()
    attack = APGDAttack(_apgd_config(steps=5, random_start=False))
    x = torch.rand(2, 3, 32, 32)
    y = torch.tensor([0, 1], dtype=torch.long)

    attack.perturb(model, x, y)

    history = torch.stack(attack.debug_step_size_history[0])
    assert float(history.min().item()) < float(history[0].min().item())


def test_apgd_deterministic_per_seed() -> None:
    torch.manual_seed(0)
    model_a = TinyClassifier()
    torch.manual_seed(0)
    model_b = TinyClassifier()
    x = torch.rand(3, 3, 32, 32)
    y = torch.tensor([0, 1, 2], dtype=torch.long)
    cfg = _apgd_config(seed=123)

    adv_a = APGDAttack(cfg).perturb(model_a, x, y)
    adv_b = APGDAttack(cfg).perturb(model_b, x, y)

    assert torch.equal(adv_a, adv_b)


def test_apgd_rejects_non_linf() -> None:
    cfg = _apgd_config()
    object.__setattr__(cfg, "norm", "L2")
    try:
        APGDAttack(cfg)
    except ValueError as exc:
        assert "Linf" in str(exc)
    else:
        raise AssertionError("APGDAttack must reject non-Linf norms")


def test_apgd_history_is_partitioned_by_restart() -> None:
    torch.manual_seed(0)
    model = TinyClassifier()
    attack = APGDAttack(_apgd_config(steps=4, n_restarts=2))
    x = torch.rand(3, 3, 32, 32)
    y = torch.tensor([0, 1, 2], dtype=torch.long)

    attack.perturb(model, x, y)

    assert len(attack.debug_best_loss_history) == 2
    assert len(attack.debug_step_size_history) == 2
    for trajectory in attack.debug_best_loss_history:
        history = torch.stack(trajectory)
        assert torch.all(history[1:] >= history[:-1] - 1e-6)
