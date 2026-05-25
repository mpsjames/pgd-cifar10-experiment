#!/usr/bin/env python
"""Run the canonical white-box adversarial evaluation entry point."""

from __future__ import annotations

from pathlib import Path

from src.attacks.factory import AttackFactory, replace_attack_epsilon
from src.cli.runner import bootstrap, build_common_parser
from src.experiments.runner import ExperimentRunner
from src.models.builders import ARCH_BUILDERS

_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"
_ATTACK_CHOICES = sorted(p.stem for p in (_CONFIG_ROOT / "attack").glob("*.yaml"))


def main() -> None:
    parser = build_common_parser("Evaluate a white-box attack")
    parser.add_argument("--arch", choices=sorted(ARCH_BUILDERS), required=True)
    parser.add_argument(
        "--attack",
        choices=_ATTACK_CHOICES,
        default="apgd_ce_100",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--variant", choices=["clean", "adv"], default="clean")
    parser.add_argument(
        "--no-attack",
        action="store_true",
        help="Zero out epsilon to measure clean accuracy with the same evaluation pipeline.",
    )
    args = parser.parse_args()

    ctx = bootstrap(args, arch=args.arch, attack=args.attack)
    attack_config = ctx.config.attack
    if args.no_attack:
        attack_config = replace_attack_epsilon(attack_config, 0.0)
    attack = AttackFactory.build(attack_config)
    attack_label = "none" if args.no_attack else args.attack
    run_name = f"white_box_{attack_label}_{args.arch}_{args.variant}_seed{args.seed}"
    tags = {
        "phase": "white_box",
        "attack": attack_label,
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
        f"{run_name}: asr={result.asr:.4f}, conditional_asr={result.conditional_asr:.4f}, "
        f"robust_acc={result.robust_acc:.4f}, time/img={result.time_per_image_ms:.2f}ms"
    )


if __name__ == "__main__":
    main()
