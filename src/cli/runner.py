"""Shared CLI bootstrap for experiment scripts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from src.experiments.config import ExperimentConfig
from src.experiments.config_loader import load_experiment_config
from src.tracking.tracker import ExperimentTracker
from src.utils.seed import set_all_seeds


def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--json-dir", type=Path, default=Path("results/logs"))
    parser.add_argument("--no-mlflow", action="store_true")
    return parser


@dataclass(frozen=True)
class ScriptContext:
    args: argparse.Namespace
    config: ExperimentConfig

    def build_tracker(self, run_name: str, tags: dict[str, str]) -> ExperimentTracker:
        return ExperimentTracker(
            self.config.tracking.experiment_name,
            run_name,
            tags,
            tracking_uri=self.args.tracking_uri or self.config.tracking.tracking_uri,
            json_dir=self.args.json_dir,
            enable=self.config.tracking.enable and not self.args.no_mlflow,
            config=self.config,
        )


def bootstrap(
    args: argparse.Namespace,
    *,
    arch: str,
    attack: str | None = None,
    training: str | None = None,
) -> ScriptContext:
    set_all_seeds(args.seed)
    config = load_experiment_config(arch=arch, attack=attack, training=training, seed=args.seed)
    return ScriptContext(args=args, config=config)
