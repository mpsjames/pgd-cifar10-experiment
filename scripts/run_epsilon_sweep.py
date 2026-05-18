#!/usr/bin/env python
"""Run the PGD epsilon sweep experiment or its smoke-test variant."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from src.attacks.factory import build_attack
from src.evaluation.runner import AttackEvaluator
from src.experiments.checkpoint_paths import adv_checkpoint_path, clean_checkpoint_path
from src.experiments.config import AttackConfig
from src.experiments.config_loader import load_attack_config, load_experiment_config
from src.models.builders import build_normalized_model, load_model_from_checkpoint
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
    parser.add_argument("--attack", default=None)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--epsilon", type=float, action="append", default=None)
    parser.add_argument("--variant", choices=["clean", "adv"], default="clean")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--json-dir", type=Path, default=Path("results/logs"))
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    set_all_seeds(args.seed[0] if args.seed else 42)

    sweep = OmegaConf.load(args.sweep_config)
    arches = args.arch or [str(item) for item in sweep.architectures]
    seeds = args.seed or [int(item) for item in sweep.seeds]
    epsilons = args.epsilon or [float(item) for item in sweep.epsilons]
    attack_name = args.attack or str(sweep.attack)
    base_attack = load_attack_config(attack_name)
    if args.smoke:
        arches = arches[:1]
        seeds = seeds[:1]
        epsilons = epsilons[:2]

    for arch in arches:
        for seed in seeds:
            set_all_seeds(seed)
            exp_config = load_experiment_config(
                arch=arch, attack=attack_name, seed=seed
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = _load_checkpoint_or_smoke_model(
                arch, seed, args.variant, exp_config, args.smoke
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
                    p_init=base_attack.p_init,
                    loss=base_attack.loss,
                    seed=base_attack.seed,
                    rho=base_attack.rho,
                    n_restarts=base_attack.n_restarts,
                )
                result = AttackEvaluator(
                    model, build_attack(attack_config), loader, device
                ).run()
                run_name = (
                    f"epsilon_sweep_{attack_name}_{arch}_eps{epsilon:.6f}_seed{seed}"
                )
                tags = {
                    "phase": "epsilon_sweep",
                    "arch": arch,
                    "seed": str(seed),
                    "epsilon": f"{epsilon:.12f}",
                }
                with ExperimentTracker(
                    exp_config.tracking.experiment_name,
                    run_name,
                    {**tags, "attack": attack_name},
                    tracking_uri=args.tracking_uri or exp_config.tracking.tracking_uri,
                    json_dir=args.json_dir,
                    enable=exp_config.tracking.enable and not args.no_mlflow,
                    config=exp_config,
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


def _checkpoint_path(arch: str, seed: int, variant: str) -> Path:
    if variant == "adv":
        return adv_checkpoint_path(arch, seed)
    return clean_checkpoint_path(arch, seed)


def _load_checkpoint_or_smoke_model(
    arch: str, seed: int, variant: str, exp_config, smoke: bool
):
    ckpt_path = _checkpoint_path(arch, seed, variant)
    if ckpt_path.exists():
        return load_model_from_checkpoint(exp_config.model, ckpt_path)
    if smoke:
        print(f"WARNING: smoke run on random weights; checkpoint not found: {ckpt_path}")
        return build_normalized_model(exp_config.model)
    raise FileNotFoundError(ckpt_path)


def _cifar_loader(batch_size: int, no_download: bool) -> DataLoader:
    from src.data.cifar10 import get_cifar10_loaders

    _, loader = get_cifar10_loaders(batch_size, download=not no_download)
    return loader


if __name__ == "__main__":
    main()
