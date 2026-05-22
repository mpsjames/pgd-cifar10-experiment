#!/usr/bin/env python
"""Run the Square Attack query-based black-box evaluation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from src.attacks.square import SquareAttack
from src.cli.attack_configs import square_config
from src.cli.runner import bootstrap, build_common_parser
from src.experiments.runner import ExperimentRunner
from src.models.builders import ARCH_BUILDERS


def square_context(args, attack_cfg):
    """Build script context whose tracked config matches the Square overrides."""
    ctx = bootstrap(args, arch=args.arch, attack="square_5000")
    return replace(ctx, config=replace(ctx.config, attack=attack_cfg))


def main() -> None:
    parser = build_common_parser("Evaluate the Linf Square Attack")
    parser.add_argument("--arch", choices=sorted(ARCH_BUILDERS), required=True)
    parser.add_argument("--variant", choices=["clean", "adv"], default="clean")
    parser.add_argument("--num-queries", type=int, default=None)
    parser.add_argument("--p-init", type=float, default=None)
    parser.add_argument("--loss", choices=["margin", "cross_entropy"], default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    attack_cfg = square_config(args)
    ctx = square_context(args, attack_cfg)
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
    with ctx.build_tracker(run_name, tags) as tracker:
        result = ExperimentRunner(ctx.config, tracker).evaluate_attack(
            SquareAttack(attack_cfg),
            checkpoint=args.checkpoint,
            variant=args.variant,
            batch_size=args.batch_size,
            smoke=args.smoke,
            no_download=args.no_download,
        )
    print(
        f"{run_name}: asr={result.asr:.4f}, conditional_asr={result.conditional_asr:.4f}, "
        f"robust_acc={result.robust_acc:.4f}, time/img={result.time_per_image_ms:.2f}ms"
    )


if __name__ == "__main__":
    main()
