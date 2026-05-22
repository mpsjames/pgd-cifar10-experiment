"""Evaluation helpers for reporting."""

from __future__ import annotations

import torch

from src.evaluation.attack_evaluator import AttackEvaluator, EvaluationResult
from src.reporting.constants import SEED
from src.reporting.loaders import evaluation_loader
from src.utils.seed import set_all_seeds


def evaluate_attack(
    model, attack, sample_size: int | None, seed: int, keep_per_sample: bool = False
) -> EvaluationResult:
    set_all_seeds(seed)
    loader = evaluation_loader(sample_size, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return AttackEvaluator(model, attack, loader, device, keep_per_sample=keep_per_sample).run()


def clean_accuracy(model, seed: int = SEED) -> float:
    loader = evaluation_loader(sample_size=None, seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return correct / max(total, 1)
