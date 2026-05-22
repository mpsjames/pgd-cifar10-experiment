from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.attacks.pgd import PGDAttack
from src.evaluation.attack_evaluator import AttackEvaluator
from src.experiments.config import AttackConfig
from src.experiments.config_loader import load_experiment_config
from src.experiments.runner import ExperimentRunner
from src.models.normalizer import Normalizer


def _make_model_and_loader():
    inner = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    model = Normalizer(inner, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    x = torch.rand(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=2, num_workers=0)
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
    return model, loader, attack


def test_attack_evaluator_runs_and_keeps_per_sample() -> None:
    model, loader, attack = _make_model_and_loader()
    result = AttackEvaluator(model, attack, loader, torch.device("cpu"), keep_per_sample=True).run()
    assert result.n_samples == 4
    assert result.per_sample_linf is not None
    assert len(result.per_sample_linf) == 4


def test_conditional_asr_excludes_already_wrong_samples() -> None:
    """conditional_asr counts only samples clean-correct that the attack flips."""
    model, loader, attack = _make_model_and_loader()
    result = AttackEvaluator(model, attack, loader, torch.device("cpu")).run()
    # asr = robust_error = 1 - robust_acc; includes samples already wrong under clean model
    assert result.asr == pytest.approx(1.0 - result.robust_acc)
    # conditional_asr is in [0, 1]
    assert 0.0 <= result.conditional_asr <= 1.0
    # conditional_asr <= asr: excludes already-wrong samples so it can only be equal or lower
    assert result.conditional_asr <= result.asr + 1e-6


def test_experiment_runner_honors_cpu_hardware_config() -> None:
    config = load_experiment_config(arch="resnet18", hardware="cpu")
    runner = ExperimentRunner(config, tracker=object())
    assert runner.device.type == "cpu"


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available on this host")
def test_experiment_runner_fails_fast_when_cuda_config_unavailable() -> None:
    config = load_experiment_config(arch="resnet18", hardware="gpu_default")
    with pytest.raises(RuntimeError, match="Configured device 'cuda' is unavailable"):
        ExperimentRunner(config, tracker=object())
