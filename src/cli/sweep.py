"""Shared epsilon-sweep CLI implementation."""

from __future__ import annotations

from dataclasses import replace

from src.attacks.factory import AttackFactory
from src.experiments.config_loader import load_attack_config, load_experiment_config
from src.experiments.runner import ExperimentRunner
from src.tracking.tracker import ExperimentTracker
from src.utils.seed import set_all_seeds


def run_sweep(
    args,
    arches: list[str],
    seed: int,
    epsilons: list[float],
    attack_name: str,
) -> None:
    base_attack = load_attack_config(attack_name)
    set_all_seeds(seed)
    for arch in arches:
        exp_config = load_experiment_config(arch=arch, attack=attack_name, seed=seed)
        for epsilon in epsilons:
            run_sweep_point(args, exp_config, arch, seed, attack_name, base_attack, epsilon)


def run_sweep_point(
    args,
    exp_config,
    arch: str,
    seed: int,
    attack_name: str,
    base,
    epsilon: float,
) -> None:
    attack_config = replace(
        base,
        epsilon=epsilon,
        alpha=min(base.alpha, epsilon) if epsilon > 0 else 0.0,
    )
    run_name = f"epsilon_sweep_{attack_name}_{arch}_eps{epsilon:.6f}_seed{seed}"
    tags = {
        "phase": "epsilon_sweep",
        "arch": arch,
        "seed": str(seed),
        "epsilon": f"{epsilon:.12f}",
        "attack": attack_name,
    }
    with ExperimentTracker(
        exp_config.tracking.experiment_name,
        run_name,
        tags,
        tracking_uri=args.tracking_uri or exp_config.tracking.tracking_uri,
        json_dir=args.json_dir,
        enable=exp_config.tracking.enable and not args.no_mlflow,
        config=exp_config,
    ) as tracker:
        result = ExperimentRunner(exp_config, tracker).evaluate_attack(
            AttackFactory.build(attack_config),
            variant=args.variant,
            batch_size=args.batch_size,
            smoke=args.smoke,
            no_download=args.no_download,
        )
    print(f"{run_name}: asr={result.asr:.4f}, robust_acc={result.robust_acc:.4f}")
