from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.attacks.pgd import PGDAttack
from src.evaluation.runner import AttackEvaluator
from src.experiments.config import AttackConfig
from src.models.normalizer import Normalizer


def test_attack_evaluator_runs_and_keeps_per_sample() -> None:
    inner = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    model = Normalizer(inner, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    x = torch.rand(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    attack = PGDAttack(
        AttackConfig(
            "PGD",
            epsilon=1 / 255,
            alpha=1 / 255,
            num_steps=1,
            random_start=False,
            norm="Linf",
        )
    )
    result = AttackEvaluator(model, attack, loader, torch.device("cpu"), keep_per_sample=True).run()
    assert result.n_samples == 4
    assert result.per_sample_linf is not None
    assert len(result.per_sample_linf) == 4
