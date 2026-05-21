#!/usr/bin/env python
"""Run APGD adversarial training."""

from __future__ import annotations

from src.cli.runner import bootstrap, build_common_parser
from src.experiments.runner import ExperimentRunner
from src.models.builders import ARCH_BUILDERS


def main() -> None:
    parser = build_common_parser("Train adversarial CIFAR-10 classifier")
    parser.add_argument("--arch", choices=sorted(ARCH_BUILDERS), required=True)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    ctx = bootstrap(args, arch=args.arch, training="apgd_at")
    epochs = args.epochs
    if epochs is None and args.smoke:
        epochs = 1
    with ctx.build_tracker(
        f"train_adversarial_{args.arch}_seed{args.seed}",
        {
            "arch": args.arch,
            "phase": "train_apgd_at",
            "inner_attack": ctx.config.training.inner_attack.name,
        },
    ) as tracker:
        runner = ExperimentRunner(ctx.config, tracker)
        runner.train_adversarial(
            epochs=epochs,
            batch_size=args.batch_size,
            smoke=args.smoke,
            no_download=args.no_download,
        )


if __name__ == "__main__":
    main()
