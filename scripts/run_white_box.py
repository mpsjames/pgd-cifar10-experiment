#!/usr/bin/env python
"""Run the canonical white-box PGD-10 evaluation entry point."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.attacks.pgd import PGDAttack
from src.data.cifar10 import get_cifar10_loaders
from src.evaluation.runner import AttackEvaluator
from src.experiments.config_loader import load_experiment_config
from src.models.builders import ARCH_BUILDERS, wrap_with_normalization
from src.utils.seed import set_all_seeds


def main() -> None:
    """Evaluate one architecture against PGD-10 on CIFAR-10 or smoke data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=sorted(ARCH_BUILDERS), required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    set_all_seeds(args.seed)
    exp_config = load_experiment_config(arch=args.arch, attack="pgd_10", seed=args.seed)
    if exp_config.attack is None:
        raise ValueError("attack config did not load")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke:
        x = torch.rand(min(args.batch_size, 8), 3, 32, 32)
        y = torch.randint(
            0, exp_config.model.num_classes, (x.size(0),), dtype=torch.long
        )
        test_loader = DataLoader(
            TensorDataset(x, y), batch_size=min(args.batch_size, 8)
        )
    else:
        _, test_loader = get_cifar10_loaders(
            args.batch_size, seed=args.seed, download=not args.no_download
        )
    model = wrap_with_normalization(
        ARCH_BUILDERS[args.arch](exp_config.model.num_classes), exp_config.model
    )
    attack = PGDAttack(exp_config.attack)
    result = AttackEvaluator(model, attack, test_loader, device).run()
    print(result)


if __name__ == "__main__":
    main()
