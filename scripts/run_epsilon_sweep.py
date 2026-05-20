#!/usr/bin/env python
"""Run the PGD epsilon sweep experiment or its smoke-test variant."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from src.cli.sweep import run_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PGD epsilon sweep")
    parser.add_argument(
        "--sweep-config",
        type=Path,
        default=Path("configs/sweeps/pgd_epsilon_sweep.yaml"),
    )
    parser.add_argument("--arch", action="append", default=None)
    parser.add_argument("--attack", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epsilon", type=float, action="append", default=None)
    parser.add_argument("--variant", choices=["clean", "adv"], default="clean")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--json-dir", type=Path, default=Path("results/logs"))
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    sweep = OmegaConf.load(args.sweep_config)
    arches = args.arch or [str(item) for item in sweep.architectures]
    seed = args.seed if args.seed is not None else int(sweep.seed)
    epsilons = args.epsilon or [float(item) for item in sweep.epsilons]
    attack_name = args.attack or str(sweep.attack)
    if args.smoke:
        arches, epsilons = arches[:1], epsilons[:2]
    run_sweep(args, arches, seed, epsilons, attack_name)


if __name__ == "__main__":
    main()
