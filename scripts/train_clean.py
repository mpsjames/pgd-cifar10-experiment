#!/usr/bin/env python
"""Run clean training for one architecture/seed pair."""

from __future__ import annotations

from src.cli.runner import bootstrap, build_common_parser
from src.experiments.runner import ExperimentRunner
from src.models.builders import ARCH_BUILDERS


def main() -> None:
    parser = build_common_parser("Train clean CIFAR-10 classifier")
    parser.add_argument("--arch", choices=sorted(ARCH_BUILDERS), required=True)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    ctx = bootstrap(args, arch=args.arch, training="clean")
    epochs = args.epochs
    if epochs is None and args.smoke:
        epochs = 1
    with ctx.build_tracker(
        f"train_clean_{args.arch}_seed{args.seed}",
        {"arch": args.arch, "phase": "train_clean"},
    ) as tracker:
        runner = ExperimentRunner(ctx.config, tracker)
        runner.train_clean(
            epochs=epochs,
            batch_size=args.batch_size,
            smoke=args.smoke,
            no_download=args.no_download,
        )


if __name__ == "__main__":
    main()
