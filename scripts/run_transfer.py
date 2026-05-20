#!/usr/bin/env python
"""Run transfer-attack experiments across architectures or seeds."""

from __future__ import annotations

from src.cli.runner import build_common_parser
from src.cli.transfer import (
    PAIR_FILES as _PAIR_FILES,
)
from src.cli.transfer import (
    load_pairs as _load_pairs,
)
from src.cli.transfer import (
    run_pair as _run_pair,
)


def main() -> None:
    parser = build_common_parser("Run transfer attacks")
    parser.add_argument("--mode", choices=sorted(_PAIR_FILES), required=True)
    parser.add_argument("--attack", default="pgd_10")
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()

    pairs = _load_pairs(args.mode)
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]
    if not pairs:
        raise ValueError(f"No transfer pairs configured for mode={args.mode}")
    for pair in pairs:
        _run_pair(pair, args)


if __name__ == "__main__":
    main()
