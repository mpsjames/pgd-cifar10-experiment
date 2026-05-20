"""Unit tests for the Linf Square Attack."""

from __future__ import annotations

import torch

from src.attacks.square import SquareAttack
from src.attacks.verify import verify_perturbation
from src.experiments.config import AttackConfig


def _square_config(
    epsilon: float = 8 / 255,
    num_steps: int = 16,
    p_init: float = 0.05,
    loss: str = "margin",
    seed: int = 42,
) -> AttackConfig:
    return AttackConfig(
        name="Square",
        epsilon=epsilon,
        alpha=epsilon,
        num_steps=num_steps,
        random_start=True,
        norm="Linf",
        p_init=p_init,
        loss=loss,  # type: ignore[arg-type]
        seed=seed,
    )


def test_square_respects_linf_and_pixel_domain(tiny_classifier) -> None:
    torch.manual_seed(0)
    model = tiny_classifier
    attack = SquareAttack(_square_config())
    x = torch.rand(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    x_adv = attack.perturb(model, x, y)

    verify_perturbation(x, x_adv, epsilon=8 / 255)
    assert x_adv.dtype == x.dtype
    assert x_adv.shape == x.shape


def test_square_is_deterministic_per_seed(tiny_classifier_factory) -> None:
    torch.manual_seed(0)
    model_a = tiny_classifier_factory()
    torch.manual_seed(0)
    model_b = tiny_classifier_factory()
    cfg = _square_config(num_steps=8, seed=123)
    x = torch.rand(3, 3, 32, 32)
    y = torch.tensor([0, 1, 2], dtype=torch.long)

    adv_a = SquareAttack(cfg).perturb(model_a, x, y)
    adv_b = SquareAttack(cfg).perturb(model_b, x, y)

    assert torch.equal(adv_a, adv_b)


def test_square_zero_epsilon_is_identity(tiny_classifier) -> None:
    model = tiny_classifier
    cfg = _square_config(epsilon=0.0)
    x = torch.rand(2, 3, 32, 32)
    y = torch.tensor([0, 1], dtype=torch.long)

    x_adv = SquareAttack(cfg).perturb(model, x, y)

    assert torch.equal(x_adv, x)


def test_square_query_budget_respected(tiny_classifier) -> None:
    model = tiny_classifier
    cfg = _square_config(num_steps=12)
    attack = SquareAttack(cfg)
    x = torch.rand(2, 3, 32, 32)
    y = torch.tensor([0, 1], dtype=torch.long)

    attack.perturb(model, x, y)

    # principles §4.9: report cost. Forward passes must not exceed num_steps.
    assert model.call_count <= cfg.num_steps
    assert attack.last_query_count <= cfg.num_steps


def test_square_rejects_non_linf() -> None:
    cfg = AttackConfig(
        name="Square",
        epsilon=1.0,
        alpha=1.0,
        num_steps=4,
        random_start=True,
        norm="Linf",  # constructor accepts this
        p_init=0.05,
        loss="margin",
        seed=7,
    )
    # Force-set the field via the underlying dict to simulate a misconfigured load.
    object.__setattr__(cfg, "norm", "L2")
    try:
        SquareAttack(cfg)
    except ValueError as exc:
        assert "Linf" in str(exc)
    else:
        raise AssertionError("SquareAttack must reject non-Linf norms")
