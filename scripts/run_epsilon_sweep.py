#!/usr/bin/env python
"""Run the PGD epsilon sweep experiment or its smoke-test variant."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from src.cli.runner import build_common_parser
from src.cli.sweep import run_sweep


def main() -> None:
    parser = build_common_parser("Run PGD epsilon sweep")
    parser.set_defaults(batch_size=64, seed=None)
    parser.add_argument(
        "--sweep-config",
        type=Path,
        default=Path("configs/sweeps/pgd_epsilon_sweep.yaml"),
    )
    parser.add_argument("--arch", action="append", default=None)
    parser.add_argument("--attack", default=None)
    parser.add_argument("--epsilon", type=float, action="append", default=None)
    parser.add_argument("--variant", choices=["clean", "adv"], default="clean")
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
