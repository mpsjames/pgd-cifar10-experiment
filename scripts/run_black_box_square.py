#!/usr/bin/env python
"""Run the Square Attack (Linf) — query-based black-box evaluation.

Threat model:
    Query-only access to victim logits — no gradients, no surrogate. Mirrors
    `scripts/run_white_box.py` for parity with the rest of the campaign.

Computational budget (principles §4.9):
    Default 5000 queries × 10000 test images per (arch, variant) is heavy on a
    4 GB GPU; expect roughly 6–10 GPU-hours per run for ResNet-18 and longer
    for WRN-34-10. Use `--num-queries` and `--smoke` to scale down for
    development.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.attacks.square import SquareAttack
from src.data.cifar10 import get_cifar10_loaders
from src.evaluation.runner import AttackEvaluator
from src.experiments.config import AttackConfig
from src.experiments.config_loader import load_attack_config, load_experiment_config
from src.experiments.notebook_reports import _load_model_from_path
from src.models.builders import ARCH_BUILDERS, build_model, wrap_with_normalization
from src.tracking.mlflow_logger import ExperimentTracker
from src.utils.seed import set_all_seeds


def main() -> None:
    """Evaluate one victim under the Linf Square Attack on CIFAR-10 or smoke data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=sorted(ARCH_BUILDERS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", choices=["clean", "adv"], default="clean")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-queries", type=int, default=None)
    parser.add_argument("--p-init", type=float, default=None)
    parser.add_argument(
        "--loss", choices=["margin", "cross_entropy"], default=None
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--json-dir", type=Path, default=Path("results/logs"))
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    set_all_seeds(args.seed)

    base_cfg = load_attack_config("square_5000")
    attack_cfg = AttackConfig(
        name=base_cfg.name,
        epsilon=base_cfg.epsilon,
        alpha=base_cfg.alpha,
        num_steps=int(args.num_queries) if args.num_queries is not None else base_cfg.num_steps,
        random_start=base_cfg.random_start,
        norm=base_cfg.norm,
        p_init=float(args.p_init) if args.p_init is not None else base_cfg.p_init,
        loss=args.loss or base_cfg.loss,  # type: ignore[arg-type]
        seed=args.seed,
    )

    exp_config = load_experiment_config(arch=args.arch, attack="pgd_10", seed=args.seed)
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
    attack = SquareAttack(attack_cfg)
    result = AttackEvaluator(model, attack, test_loader, device).run()
    run_name = f"square_{args.arch}_{args.variant}_seed{args.seed}_q{attack_cfg.num_steps}"
    tags = {
        "phase": "black_box_query",
        "attack": "square",
        "arch": args.arch,
        "seed": str(args.seed),
        "variant": args.variant,
        "num_queries": str(attack_cfg.num_steps),
        "loss": str(attack_cfg.loss),
    }
    with ExperimentTracker(
        exp_config.tracking.experiment_name,
        run_name,
        tags,
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
