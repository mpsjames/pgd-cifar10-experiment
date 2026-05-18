#!/usr/bin/env python
"""Run transfer-attack experiments across architectures or seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from src.attacks.factory import build_attack
from src.attacks.verify import verify_perturbation
from src.evaluation.metrics import attack_success_rate, robust_accuracy
from src.experiments.config_loader import load_experiment_config
from src.experiments.notebook_reports import _load_model_from_path
from src.models.builders import ARCH_BUILDERS, build_model, wrap_with_normalization
from src.tracking.mlflow_logger import ExperimentTracker
from src.utils.seed import set_all_seeds


def main() -> None:
    """Execute transfer-attack pairs declared in the YAML pair registries."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["cross_arch", "cross_seed", "gray_box"],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attack", default="pgd_10")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--json-dir", type=Path, default=Path("results/logs"))
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    set_all_seeds(args.seed)
    pairs = _load_pairs(args.mode)
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]
    if not pairs:
        raise ValueError(f"No transfer pairs configured for mode={args.mode}")
    for pair in pairs:
        _run_pair(pair, args)


_PAIR_FILES = {
    "cross_arch": "configs/transfer/transfer_pairs.yaml",
    "cross_seed": "configs/transfer/cross_seed_pairs.yaml",
    "gray_box": "configs/transfer/gray_box_pairs.yaml",
}


def _load_pairs(mode: str) -> list[dict]:
    cfg = OmegaConf.load(Path(_PAIR_FILES[mode]))
    return [dict(item) for item in cfg.pairs]


def _run_pair(pair: dict, args: argparse.Namespace) -> None:
    surrogate_variant = "clean"
    victim_variant = "clean"
    if args.mode == "cross_arch":
        surrogate_arch = str(pair["surrogate"])
        victim_arch = str(pair["victim"])
        surrogate_seed = victim_seed = args.seed
        run_name = (
            f"transfer_{surrogate_arch}_to_{victim_arch}_{args.attack}_seed{args.seed}"
        )
        tags = {
            "phase": "transfer",
            "mode": "cross_arch",
            "surrogate": surrogate_arch,
            "victim": victim_arch,
            "seed": str(args.seed),
        }
    elif args.mode == "cross_seed":
        surrogate_arch = victim_arch = str(pair["arch"])
        surrogate_seed = int(pair["surrogate_seed"])
        victim_seed = int(pair["victim_seed"])
        run_name = (
            f"transfer_seed_{surrogate_arch}_{pair['surrogate_seed']}"
            f"to{pair['victim_seed']}_{args.attack}"
        )
        tags = {
            "phase": "transfer",
            "mode": "cross_seed",
            "arch": surrogate_arch,
            "surrogate_seed": str(pair["surrogate_seed"]),
            "victim_seed": str(pair["victim_seed"]),
        }
    else:  # gray_box
        surrogate_arch = victim_arch = str(pair["arch"])
        surrogate_seed = int(pair["surrogate_seed"])
        victim_seed = int(pair["victim_seed"])
        surrogate_variant = str(pair.get("surrogate_variant", "clean"))
        victim_variant = str(pair.get("victim_variant", "clean"))
        run_name = (
            f"graybox_{surrogate_arch}"
            f"_s{surrogate_variant}{surrogate_seed}"
            f"_v{victim_variant}{victim_seed}_{args.attack}"
        )
        tags = {
            "phase": "transfer",
            "mode": "gray_box",
            "arch": surrogate_arch,
            "surrogate_seed": str(surrogate_seed),
            "victim_seed": str(victim_seed),
            "surrogate_variant": surrogate_variant,
            "victim_variant": victim_variant,
        }

    exp_config = load_experiment_config(
        arch=surrogate_arch, attack=args.attack, seed=args.seed
    )
    if exp_config.attack is None:
        raise ValueError("transfer attack config did not load")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    surrogate = _load_variant_or_smoke_model(
        surrogate_arch, surrogate_seed, surrogate_variant, exp_config.model, args.smoke
    )
    victim_config = load_experiment_config(
        arch=victim_arch, attack=args.attack, seed=victim_seed
    ).model
    victim = _load_variant_or_smoke_model(
        victim_arch, victim_seed, victim_variant, victim_config, args.smoke
    )
    surrogate = surrogate.to(device).eval()
    victim = victim.to(device).eval()
    loader = (
        _smoke_loader(args.batch_size, exp_config.model.num_classes)
        if args.smoke
        else _cifar_loader(args.batch_size, args.no_download)
    )
    attack = build_attack(exp_config.attack)

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
        exp_config.tracking.experiment_name,
        run_name,
        {**tags, "attack": args.attack},
        tracking_uri=args.tracking_uri or exp_config.tracking.tracking_uri,
        json_dir=args.json_dir,
        enable=exp_config.tracking.enable and not args.no_mlflow,
        config=exp_config,
    ) as tracker:
        tracker.log_metrics(metrics)
    print(f"{run_name}: {metrics}")


def _smoke_loader(batch_size: int, num_classes: int) -> DataLoader:
    n = min(batch_size, 8)
    x = torch.rand(n, 3, 32, 32)
    y = torch.randint(0, num_classes, (n,), dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=n)


def _clean_ckpt(arch: str, seed: int) -> Path:
    return Path("checkpoints/clean") / f"{arch}_seed{seed}.pt"


def _variant_ckpt(arch: str, seed: int, variant: str) -> Path:
    if variant == "adv":
        return Path("checkpoints/adv") / f"{arch}_apgd_at_seed{seed}.pt"
    if variant == "clean":
        return _clean_ckpt(arch, seed)
    raise ValueError(f"Unknown checkpoint variant: {variant!r}")


def _fresh_model_random(model_config):
    return wrap_with_normalization(build_model(model_config), model_config)


def _load_clean_or_smoke_model(arch: str, seed: int, model_config, smoke: bool):
    return _load_variant_or_smoke_model(arch, seed, "clean", model_config, smoke)


def _load_variant_or_smoke_model(
    arch: str, seed: int, variant: str, model_config, smoke: bool
):
    path = _variant_ckpt(arch, seed, variant)
    if path.exists():
        return _load_model_from_path(arch, path)
    if smoke:
        print(f"WARNING: smoke run on random weights; checkpoint not found: {path}")
        return _fresh_model_random(model_config)
    raise FileNotFoundError(path)


def _cifar_loader(batch_size: int, no_download: bool) -> DataLoader:
    from src.data.cifar10 import get_cifar10_loaders

    _, loader = get_cifar10_loaders(batch_size, download=not no_download)
    return loader


if __name__ == "__main__":
    main()
