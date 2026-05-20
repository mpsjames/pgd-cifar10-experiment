"""CLI helpers for attack config overrides."""

from __future__ import annotations

import argparse
from dataclasses import replace

from src.experiments.config import AttackConfig
from src.experiments.config_loader import load_attack_config


def square_config(args: argparse.Namespace) -> AttackConfig:
    """Load `square_5000.yaml` and override fields supplied on the CLI."""
    base = load_attack_config("square_5000")
    return replace(
        base,
        num_steps=int(args.num_queries) if args.num_queries is not None else base.num_steps,
        p_init=float(args.p_init) if args.p_init is not None else base.p_init,
        loss=args.loss if args.loss is not None else base.loss,
        seed=args.seed,
    )
