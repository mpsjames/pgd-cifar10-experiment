#!/usr/bin/env python
"""Run transfer-attack experiments across architectures or seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from src.attacks.pgd import PGDAttack
from src.attacks.verify import verify_perturbation
from src.evaluation.metrics import attack_success_rate, robust_accuracy
from src.experiments.config_loader import load_experiment_config
from src.models.builders import ARCH_BUILDERS, wrap_with_normalization
from src.tracking.mlflow_logger import ExperimentTracker
from src.utils.seed import set_all_seeds


def main() -> None:
    """Execute transfer-attack pairs declared in the YAML pair registries."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cross_arch", "cross_seed"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--json-dir", type=Path, default=Path("results/logs"))
    args = parser.parse_args()
    set_all_seeds(args.seed)
    pairs = _load_pairs(args.mode)
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]
    if not pairs:
        raise ValueError(f"No transfer pairs configured for mode={args.mode}")
    for pair in pairs:
        _run_pair(pair, args)


def _load_pairs(mode: str) -> list[dict]:
    path = Path(
        "configs/transfer/transfer_pairs.yaml"
        if mode == "cross_arch"
        else "configs/transfer/cross_seed_pairs.yaml"
    )
    cfg = OmegaConf.load(path)
    return [dict(item) for item in cfg.pairs]


def _run_pair(pair: dict, args: argparse.Namespace) -> None:
    if args.mode == "cross_arch":
        surrogate_arch = str(pair["surrogate"])
        victim_arch = str(pair["victim"])
        run_name = f"transfer_{surrogate_arch}_to_{victim_arch}_pgd10_seed{args.seed}"
        tags = {
            "phase": "transfer",
            "mode": "cross_arch",
            "surrogate": surrogate_arch,
            "victim": victim_arch,
            "seed": str(args.seed),
        }
    else:
        surrogate_arch = victim_arch = str(pair["arch"])
        run_name = f"transfer_seed_{surrogate_arch}_{pair['surrogate_seed']}to{pair['victim_seed']}_pgd10"
        tags = {
            "phase": "transfer",
            "mode": "cross_seed",
            "arch": surrogate_arch,
            "surrogate_seed": str(pair["surrogate_seed"]),
            "victim_seed": str(pair["victim_seed"]),
        }

    exp_config = load_experiment_config(
        arch=surrogate_arch, attack="pgd_10", seed=args.seed
    )
    if exp_config.attack is None:
        raise ValueError("transfer attack config did not load")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    surrogate = (
        wrap_with_normalization(
            ARCH_BUILDERS[surrogate_arch](exp_config.model.num_classes),
            exp_config.model,
        )
        .to(device)
        .eval()
    )
    victim_config = load_experiment_config(
        arch=victim_arch, attack="pgd_10", seed=args.seed
    ).model
    victim = (
        wrap_with_normalization(
            ARCH_BUILDERS[victim_arch](victim_config.num_classes), victim_config
        )
        .to(device)
        .eval()
    )
    loader = (
        _smoke_loader(args.batch_size, exp_config.model.num_classes)
        if args.smoke
        else _cifar_loader(args.batch_size, args.no_download)
    )
    attack = PGDAttack(exp_config.attack)

    predictions: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        x_adv = attack.perturb(surrogate, x, y)
        verify_perturbation(x, x_adv, exp_config.attack.epsilon, exp_config.attack.norm)
        with torch.no_grad():
            predictions.append(victim(x_adv).argmax(dim=1).cpu())
            labels.append(y.cpu())
    pred = torch.cat(predictions)
    lab = torch.cat(labels)
    metrics = {
        "asr": attack_success_rate(pred, lab),
        "robust_acc": robust_accuracy(pred, lab),
        "n_samples": float(lab.numel()),
    }
    with ExperimentTracker(
        "pgd_cifar10_multiarch",
        run_name,
        tags,
        tracking_uri=args.tracking_uri,
        json_dir=args.json_dir,
    ) as tracker:
        tracker.log_metrics(metrics)
    print(f"{run_name}: {metrics}")


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
