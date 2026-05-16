#!/usr/bin/env python
"""Run the PGD epsilon sweep experiment or its smoke-test variant."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from src.attacks.pgd import PGDAttack
from src.evaluation.runner import AttackEvaluator
from src.experiments.config import AttackConfig
from src.experiments.config_loader import load_attack_config, load_experiment_config
from src.models.builders import ARCH_BUILDERS, wrap_with_normalization
from src.tracking.mlflow_logger import ExperimentTracker
from src.utils.seed import set_all_seeds


def main() -> None:
    """Execute the epsilon sweep defined by YAML or CLI overrides.

    Smoke mode evaluates a tiny synthetic batch so the command remains usable
    in CPU-only CI.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-config",
        type=Path,
        default=Path("configs/sweeps/pgd_epsilon_sweep.yaml"),
    )
    parser.add_argument("--arch", action="append", default=None)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--epsilon", type=float, action="append", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--json-dir", type=Path, default=Path("results/logs"))
    args = parser.parse_args()

    sweep = OmegaConf.load(args.sweep_config)
    arches = args.arch or [str(item) for item in sweep.architectures]
    seeds = args.seed or [int(item) for item in sweep.seeds]
    epsilons = args.epsilon or [float(item) for item in sweep.epsilons]
    base_attack = load_attack_config(str(sweep.attack))
    if args.smoke:
        arches = arches[:1]
        seeds = seeds[:1]
        epsilons = epsilons[:2]

    for arch in arches:
        for seed in seeds:
            set_all_seeds(seed)
            exp_config = load_experiment_config(
                arch=arch, attack=str(sweep.attack), seed=seed
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = wrap_with_normalization(
                ARCH_BUILDERS[arch](exp_config.model.num_classes), exp_config.model
            ).to(device)
            loader = (
                _smoke_loader(args.batch_size, exp_config.model.num_classes)
                if args.smoke
                else _cifar_loader(args.batch_size, args.no_download)
            )
            for epsilon in epsilons:
                attack_config = AttackConfig(
                    name=base_attack.name,
                    epsilon=epsilon,
                    alpha=min(base_attack.alpha, epsilon) if epsilon > 0 else 0.0,
                    num_steps=base_attack.num_steps,
                    random_start=base_attack.random_start,
                    norm=base_attack.norm,
                )
                result = AttackEvaluator(
                    model, PGDAttack(attack_config), loader, device
                ).run()
                run_name = f"epsilon_sweep_{arch}_eps{epsilon:.6f}_seed{seed}"
                tags = {
                    "phase": "epsilon_sweep",
                    "arch": arch,
                    "seed": str(seed),
                    "epsilon": f"{epsilon:.12f}",
                }
                with ExperimentTracker(
                    "pgd_cifar10_multiarch",
                    run_name,
                    tags,
                    tracking_uri=args.tracking_uri,
                    json_dir=args.json_dir,
                ) as tracker:
                    tracker.log_metrics(
                        {
                            "asr": result.asr,
                            "robust_acc": result.robust_acc,
                            "n_samples": float(result.n_samples),
                        }
                    )
                print(
                    f"{run_name}: asr={result.asr:.4f}, robust_acc={result.robust_acc:.4f}"
                )


def _smoke_loader(batch_size: int, num_classes: int) -> DataLoader:
    n = min(batch_size, 8)
    x = torch.rand(n, 3, 32, 32)
    y = torch.randint(0, num_classes, (n,), dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=n)


def _cifar_loader(batch_size: int, no_download: bool) -> DataLoader:
    from src.data.cifar10 import get_cifar10_loaders

    _, loader = get_cifar10_loaders(batch_size, download=not no_download)
    return loader


if __name__ == "__main__":
    main()
