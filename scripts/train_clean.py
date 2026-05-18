#!/usr/bin/env python
"""Run the clean-training entry point for one architecture/seed pair."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.cifar10 import get_cifar10_loaders
from src.experiments.config_loader import load_experiment_config
from src.models.builders import ARCH_BUILDERS, build_model, wrap_with_normalization
from src.tracking.mlflow_logger import ExperimentTracker
from src.training.clean import build_optimizer, build_scheduler, clean_train_epoch
from src.utils.seed import set_all_seeds


def main() -> None:
    """Train one clean baseline model or a small smoke-test surrogate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=sorted(ARCH_BUILDERS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--json-dir", type=Path, default=Path("results/logs"))
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    set_all_seeds(args.seed)

    exp_config = load_experiment_config(
        arch=args.arch, training="clean", seed=args.seed
    )
    if exp_config.training is None:
        raise ValueError("clean training config did not load")
    config = exp_config.training
    batch_size = args.batch_size or config.batch_size
    config = replace(config, epochs=args.epochs, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = wrap_with_normalization(build_model(exp_config.model), exp_config.model).to(
        device
    )
    if args.smoke:
        x = torch.rand(min(batch_size, 8), 3, 32, 32)
        y = torch.randint(
            0, exp_config.model.num_classes, (x.size(0),), dtype=torch.long
        )
        train_loader = DataLoader(TensorDataset(x, y), batch_size=min(batch_size, 8))
    else:
        train_loader, _ = get_cifar10_loaders(
            batch_size, seed=args.seed, download=not args.no_download
        )
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    with ExperimentTracker(
        exp_config.tracking.experiment_name,
        f"train_clean_{args.arch}_seed{args.seed}",
        {"arch": args.arch, "phase": "train_clean"},
        tracking_uri=args.tracking_uri or exp_config.tracking.tracking_uri,
        json_dir=args.json_dir,
        enable=exp_config.tracking.enable and not args.no_mlflow,
        config=exp_config,
    ) as tracker:
        for epoch in range(args.epochs):
            metrics = clean_train_epoch(
                model, train_loader, optimizer, scaler, device, config.use_amp
            )
            scheduler.step()
            tracker.log_metrics(metrics, step=epoch)
    out = Path("checkpoints/clean") / f"{args.arch}_seed{args.seed}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, out)


if __name__ == "__main__":
    main()
