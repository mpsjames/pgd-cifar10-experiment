"""Shared transfer-attack CLI implementation."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from src.attacks.factory import AttackFactory
from src.cli.runner import bootstrap
from src.experiments.runner import ExperimentRunner

TRANSFER_MODE_CONFIGS = {
    "cross_arch": "configs/transfer/transfer_pairs.yaml",
    "gray_box": "configs/transfer/gray_box_pairs.yaml",
}


def load_pairs(mode: str) -> list[dict]:
    cfg = OmegaConf.load(Path(TRANSFER_MODE_CONFIGS[mode]))
    return [dict(item) for item in cfg.pairs]


def run_pair(pair: dict, args) -> None:
    spec = pair_spec(pair, args)
    ctx = bootstrap(args, arch=spec["surrogate_arch"], attack=args.attack)
    if ctx.config.attack is None:
        raise ValueError("transfer attack config did not load")
    with ctx.build_tracker(spec["run_name"], {**spec["tags"], "attack": args.attack}) as tracker:
        result = ExperimentRunner(ctx.config, tracker).evaluate_transfer(
            AttackFactory.build(ctx.config.attack),
            surrogate_arch=spec["surrogate_arch"],
            victim_arch=spec["victim_arch"],
            surrogate_seed=spec["surrogate_seed"],
            victim_seed=spec["victim_seed"],
            surrogate_variant=spec["surrogate_variant"],
            victim_variant=spec["victim_variant"],
            batch_size=args.batch_size,
            smoke=args.smoke,
            no_download=args.no_download,
        )
    print(
        f"{spec['run_name']}: asr={result.asr:.4f}, "
        f"conditional_asr={result.conditional_asr:.4f}, robust_acc={result.robust_acc:.4f}"
    )


def pair_spec(pair: dict, args) -> dict:
    if args.mode == "cross_arch":
        return _cross_arch_spec(pair, args)
    return _gray_box_spec(pair, args)


def _cross_arch_spec(pair: dict, args) -> dict:
    surrogate_arch = str(pair["surrogate"])
    victim_arch = str(pair["victim"])
    return {
        "surrogate_arch": surrogate_arch,
        "victim_arch": victim_arch,
        "surrogate_seed": args.seed,
        "victim_seed": args.seed,
        "surrogate_variant": "clean",
        "victim_variant": "clean",
        "run_name": f"transfer_{surrogate_arch}_to_{victim_arch}_{args.attack}_seed{args.seed}",
        "tags": {
            "phase": "transfer",
            "mode": "cross_arch",
            "surrogate": surrogate_arch,
            "victim": victim_arch,
            "seed": str(args.seed),
        },
    }


def _gray_box_spec(pair: dict, args) -> dict:
    arch = str(pair["arch"])
    surrogate_seed = int(pair["surrogate_seed"])
    victim_seed = int(pair["victim_seed"])
    surrogate_variant = str(pair.get("surrogate_variant", "clean"))
    victim_variant = str(pair.get("victim_variant", "clean"))
    return {
        "surrogate_arch": arch,
        "victim_arch": arch,
        "surrogate_seed": surrogate_seed,
        "victim_seed": victim_seed,
        "surrogate_variant": surrogate_variant,
        "victim_variant": victim_variant,
        "run_name": (
            f"graybox_{arch}_s{surrogate_variant}{surrogate_seed}"
            f"_v{victim_variant}{victim_seed}_{args.attack}"
        ),
        "tags": {
            "phase": "transfer",
            "mode": "gray_box",
            "arch": arch,
            "surrogate_seed": str(surrogate_seed),
            "victim_seed": str(victim_seed),
            "surrogate_variant": surrogate_variant,
            "victim_variant": victim_variant,
        },
    }
