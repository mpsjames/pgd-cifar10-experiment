#!/usr/bin/env python
"""Run APGD adversarial training with the WRN fallback policy."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.cifar10 import get_cifar10_loaders
from src.experiments.config import TrainingConfig
from src.experiments.config_loader import load_experiment_config
from src.models.builders import ARCH_BUILDERS, build_model, wrap_with_normalization
from src.models.robustbench_loader import load_robustbench_wrn
from src.tracking.mlflow_logger import ExperimentTracker
from src.training.adversarial import adversarial_train
from src.utils.seed import set_all_seeds


WRN_FALLBACK_CODE = "WRN-34-10 OOM fallback decision"


def main() -> None:
    """Train one adversarially robust model or trigger the documented fallback.

    Notes:
        WRN-34-10 may retry at a smaller batch size and finally switch to the
        RobustBench fallback when 4 GB VRAM is insufficient.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=sorted(ARCH_BUILDERS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--json-dir", type=Path, default=Path("results/logs"))
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_config = load_experiment_config(
        arch=args.arch, training="apgd_at", seed=args.seed
    )
    if exp_config.training is None:
        raise ValueError("adversarial training config did not load")
    batch_size = args.batch_size or exp_config.training.batch_size
    with ExperimentTracker(
        exp_config.tracking.experiment_name,
        f"train_apgd_at_{args.arch}_seed{args.seed}",
        {
            "arch": args.arch,
            "phase": "train_adv",
            "inner_attack": exp_config.training.inner_attack.name
            if exp_config.training.inner_attack is not None
            else "",
        },
        tracking_uri=args.tracking_uri or exp_config.tracking.tracking_uri,
        json_dir=args.json_dir,
        enable=exp_config.tracking.enable and not args.no_mlflow,
        config=exp_config,
    ) as tracker:
        try:
            _run_training(
                args,
                exp_config,
                batch_size,
                device,
                tracker,
                grad_checkpointing=args.arch == "wrn_34_10",
            )
        except RuntimeError as exc:
            if args.arch != "wrn_34_10" or "out of memory" not in str(exc).lower():
                raise
            resume_path = _latest_resume_path(args.arch, args.seed)
            retry_args = argparse.Namespace(**{**vars(args), "resume": resume_path})
            if resume_path is None:
                resume_message = "retrying batch=16 from scratch (no resume checkpoint found)"
            else:
                resume_message = (
                    f"retrying batch=16 from latest resume checkpoint: {resume_path}"
                )
            print(
                f"{WRN_FALLBACK_CODE}: OOM at batch={batch_size}; {resume_message}."
            )
            tracker.set_tags(
                {
                    "fallback_triggered": "retry_batch16",
                    "wrn_oom_batch": str(batch_size),
                }
            )
            try:
                _run_training(
                    retry_args,
                    exp_config,
                    16,
                    device,
                    tracker,
                    grad_checkpointing=True,
                )
            except RuntimeError as second_exc:
                if "out of memory" not in str(second_exc).lower():
                    raise
                print(
                    f"{WRN_FALLBACK_CODE}: OOM persisted at batch=16; switching to RobustBench WRN fallback."
                )
                tracker.set_tags(
                    {"fallback_triggered": "true", "wrn_at_source": "robustbench"}
                )
                model = load_robustbench_wrn()
                out = Path(f"checkpoints/adv/wrn_34_10_apgd_at_seed{args.seed}.pt")
                out.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "source": "robustbench"}, out)


def _run_training(
    args,
    exp_config,
    batch_size: int,
    device: torch.device,
    tracker: ExperimentTracker,
    grad_checkpointing: bool,
) -> None:
    model = wrap_with_normalization(build_model(exp_config.model), exp_config.model).to(
        device
    )
    if args.smoke:
        x = torch.rand(min(batch_size, 4), 3, 32, 32)
        y = torch.randint(
            0, exp_config.model.num_classes, (x.size(0),), dtype=torch.long
        )
        train_loader = DataLoader(TensorDataset(x, y), batch_size=min(batch_size, 4))
        test_loader = train_loader
    else:
        train_loader, test_loader = get_cifar10_loaders(
            batch_size, seed=args.seed, download=not args.no_download
        )
    loaded = exp_config.training
    config = TrainingConfig(
        mode="adversarial",
        epochs=args.epochs,
        batch_size=batch_size,
        lr=loaded.lr,
        weight_decay=loaded.weight_decay,
        optimizer=loaded.optimizer,
        scheduler=loaded.scheduler,
        momentum=loaded.momentum,
        use_amp=loaded.use_amp,
        grad_clip=loaded.grad_clip,
        inner_attack=loaded.inner_attack,
        resume_from=args.resume,
        save_every_epochs=loaded.save_every_epochs,
        lr_milestones=loaded.lr_milestones,
        lr_gamma=loaded.lr_gamma,
    )
    tracker.set_tags(
        {
            "grad_checkpointing": str(grad_checkpointing).lower(),
            "batch_size": str(batch_size),
        }
    )
    adversarial_train(
        model,
        train_loader,
        test_loader,
        config,
        tracker,
        device,
        arch=args.arch,
        seed=args.seed,
    )


def _latest_resume_path(arch: str, seed: int) -> Path | None:
    resume_dir = Path("checkpoints/adv/_resume")
    pattern = re.compile(rf"^{re.escape(arch)}_seed{seed}_epoch(\d+)\.pt$")
    candidates: list[tuple[int, Path]] = []
    for path in resume_dir.glob(f"{arch}_seed{seed}_epoch*.pt"):
        match = pattern.match(path.name)
        if match is not None:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


if __name__ == "__main__":
    main()
