#!/usr/bin/env python
"""Run the canonical white-box adversarial evaluation entry point."""

from __future__ import annotations

from pathlib import Path

from src.attacks.factory import AttackFactory
from src.cli.runner import bootstrap, build_common_parser
from src.experiments.runner import ExperimentRunner
from src.models.builders import ARCH_BUILDERS


def main() -> None:
    parser = build_common_parser("Evaluate a white-box attack")
    parser.add_argument("--arch", choices=sorted(ARCH_BUILDERS), required=True)
    parser.add_argument(
        "--attack",
        choices=["fgsm", "bim_10", "pgd_10", "pgd_40", "pgd_100", "apgd_ce_10", "apgd_ce_100"],
        default="apgd_ce_100",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--variant", choices=["clean", "adv"], default="clean")
    args = parser.parse_args()

    ctx = bootstrap(args, arch=args.arch, attack=args.attack)
    attack = AttackFactory.build(ctx.config.attack)
    run_name = f"white_box_{args.attack}_{args.arch}_{args.variant}_seed{args.seed}"
    tags = {
        "phase": "white_box",
        "attack": args.attack,
        "arch": args.arch,
        "seed": str(args.seed),
        "variant": args.variant,
    }
    with ctx.build_tracker(run_name, tags) as tracker:
        result = ExperimentRunner(ctx.config, tracker).evaluate_attack(
            attack,
            checkpoint=args.checkpoint,
            variant=args.variant,
            batch_size=args.batch_size,
            smoke=args.smoke,
            no_download=args.no_download,
        )
    print(
        f"{run_name}: asr={result.asr:.4f}, robust_acc={result.robust_acc:.4f}, "
        f"time/img={result.time_per_image_ms:.2f}ms"
    )


if __name__ == "__main__":
    main()
