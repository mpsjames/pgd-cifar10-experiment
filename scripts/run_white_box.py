#!/usr/bin/env python
"""Run the canonical white-box adversarial evaluation entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.attacks.factory import build_attack
from src.data.cifar10 import get_cifar10_loaders
from src.evaluation.runner import AttackEvaluator
from src.experiments.notebook_reports import _load_model_from_path
from src.experiments.config_loader import load_experiment_config
from src.models.builders import ARCH_BUILDERS, build_model, wrap_with_normalization
from src.tracking.mlflow_logger import ExperimentTracker
from src.utils.seed import set_all_seeds


def main() -> None:
    """Evaluate one architecture against a white-box attack or smoke data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=sorted(ARCH_BUILDERS), required=True)
    parser.add_argument(
        "--attack",
        choices=["pgd_10", "pgd_40", "pgd_100", "apgd_ce_10", "apgd_ce_100"],
        default="apgd_ce_100",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--variant", choices=["clean", "adv"], default="clean")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--json-dir", type=Path, default=Path("results/logs"))
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    set_all_seeds(args.seed)
    exp_config = load_experiment_config(
        arch=args.arch, attack=args.attack, seed=args.seed
    )
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
    model = _load_checkpoint_or_smoke_model(
        args.arch, args.seed, args.variant, args.checkpoint, args.smoke
    ).to(device)
    attack = build_attack(exp_config.attack)
    result = AttackEvaluator(model, attack, test_loader, device).run()
    run_name = f"white_box_{args.attack}_{args.arch}_{args.variant}_seed{args.seed}"
    with ExperimentTracker(
        exp_config.tracking.experiment_name,
        run_name,
        {
            "phase": "white_box",
            "attack": args.attack,
            "arch": args.arch,
            "seed": str(args.seed),
            "variant": args.variant,
        },
        tracking_uri=args.tracking_uri or exp_config.tracking.tracking_uri,
        json_dir=args.json_dir,
        enable=exp_config.tracking.enable and not args.no_mlflow,
        config=exp_config,
    ) as tracker:
        tracker.log_metrics(
            {
                "asr": result.asr,
                "robust_acc": result.robust_acc,
                "linf_mean": result.linf_mean,
                "l2_mean": result.l2_mean,
                "time_per_image_ms": result.time_per_image_ms,
                "n_samples": float(result.n_samples),
            }
        )
    print(
        f"{run_name}: asr={result.asr:.4f}, robust_acc={result.robust_acc:.4f}, "
        f"time/img={result.time_per_image_ms:.2f}ms"
    )


def _checkpoint_path(arch: str, seed: int, variant: str) -> Path:
    if variant == "adv":
        return Path("checkpoints/adv") / f"{arch}_apgd_at_seed{seed}.pt"
    return Path("checkpoints/clean") / f"{arch}_seed{seed}.pt"


def _fresh_model_random(model_config):
    return wrap_with_normalization(build_model(model_config), model_config)


def _load_checkpoint_or_smoke_model(
    arch: str, seed: int, variant: str, checkpoint: Path | None, smoke: bool
):
    path = checkpoint or _checkpoint_path(arch, seed, variant)
    if path.exists():
        return _load_model_from_path(arch, path)
    if smoke:
        exp_config = load_experiment_config(arch=arch, attack="pgd_10", seed=seed)
        print(f"WARNING: smoke run on random weights; checkpoint not found: {path}")
        return _fresh_model_random(exp_config.model)
    raise FileNotFoundError(path)


if __name__ == "__main__":
    main()
