# PGD Adversarial Attack & Defense — Multi-Architecture Study on CIFAR-10

> Reproducible CIFAR-10 study of PGD-based attacks, adversarial training, and
> transfer behavior across four image-classification architectures.
>
> The repo is organized as production code under `src/`, thin CLI entrypoints
> under `scripts/`, and executable report notebooks that stay honest when the
> full checkpoint campaign has not been run yet.

## What this is

- A CIFAR-10-only implementation of FGSM, BIM, and PGD white-box evaluation, transfer attacks, epsilon sweeps, and Madry-style adversarial training.
- A reproducibility-oriented experiment harness with frozen configs, deterministic seeding, MLflow plus JSON tracking, and notebook reports backed by shared `src/` helpers.
- A multi-architecture benchmark covering `resnet18`, `wrn_34_10`, `resnet50`, and `vgg16_bn`.
- A smoke-test-friendly codebase: when full checkpoints are missing, scripts and notebooks emit pending markers instead of fabricating results.
- Non-goals: no L2/L1/L0 attacks, no targeted attacks, no AutoAttack baseline, no TRADES/MART-style defenses, and no ImageNet expansion.

## Quick start

```bash
git clone https://github.com/<user>/pgd-cifar10-experiment.git
cd pgd-cifar10-experiment
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,notebooks]'
pytest tests/ -q
bash scripts/reproduce.sh --smoke
```

## Reproducibility

This project reports `mean +- std` across 5 seeds for clean-model phases and
explicit single-seed disclosure for adversarial training (`seed=42`).

Full reproduction entrypoint:

```bash
bash scripts/reproduce.sh
```

The full campaign trains clean checkpoints for all four architectures across
seeds `{42, 123, 456, 789, 1024}`, runs single-seed adversarial training,
executes white-box and transfer evaluations, performs the epsilon sweep, and
executes all notebooks in place.

Hardware target: NVIDIA A1000 4 GB VRAM. Expected runtime is roughly
119-141 GPU-hours. WRN-34-10 adversarial training may fall back to the
RobustBench checkpoint path when the retry chain still OOMs; this is surfaced
through MLflow tags such as `fallback_triggered=true` and
`wrn_at_source=robustbench`.

## Architecture

```text
configs/    Hydra-style YAML composition consumed through OmegaConf
src/        Production code for attacks, data, models, training, evaluation, tracking, viz
scripts/    Thin CLI entrypoints for training and evaluation workflows
notebooks/  Executable report notebooks NB01-NB11
tests/      Pytest suite covering attacks, training, tracking, notebooks, and viz smoke paths
```

See [plan.md](plan.md) §2-§3 for the full module-boundary contract and
repository layout.

## Key contracts

- Reproducibility: entry points call `set_all_seeds(seed)` and tracking records Git/environment metadata.
- Raw-input attacks: attacks consume `[0, 1]` images, while `NormalizedModel` performs CIFAR-10 normalization inside `forward`.
- Fail-fast verification: attacked batches are checked with `verify_perturbation` before metrics are trusted.
- Frozen configs: experiment, training, model, and attack configs are immutable dataclasses after loading.
- Linf only: all attack/evaluation code assumes the project’s scoped perturbation norm.
- Honest notebooks: notebook helpers emit `full-campaign-pending` rather than inventing missing results.

## CLI

```text
scripts/train_clean.py          --arch {resnet18,wrn_34_10,resnet50,vgg16_bn} --seed N [--epochs E] [--smoke]
scripts/train_adversarial.py    --arch ... --seed N [--resume PATH] [--smoke]
scripts/run_white_box.py        --arch ... [--seed N] [--smoke]
scripts/run_transfer.py         --mode {cross_arch,cross_seed} [--max-pairs K] [--smoke]
scripts/run_epsilon_sweep.py    [--arch ARCH] [--seed N] [--epsilon EPS] [--smoke]
scripts/download_robustbench_wrn.py
scripts/reproduce.sh        [--smoke]
```

Examples:

```bash
python scripts/train_clean.py --arch resnet18 --seed 42 --epochs 100
python scripts/train_adversarial.py --arch resnet18 --seed 42 --epochs 100
python scripts/run_white_box.py --arch resnet18 --seed 42
python scripts/run_transfer.py --mode cross_seed --seed 42
python scripts/run_epsilon_sweep.py --arch resnet18 --seed 42
```

## Configuration

Root config: [configs/config.yaml](configs/config.yaml)

```yaml
defaults:
  - base: default
  - architecture: resnet18
  - attack: pgd_10
  - training: clean
  - _self_
```

Supporting fragments live under:

- `configs/architecture/`
- `configs/attack/`
- `configs/training/`
- `configs/transfer/`
- `configs/sweeps/`

`load_experiment_config`, `load_attack_config`, and `load_training_config`
resolve these YAML fragments into frozen dataclasses under `src/experiments/`.

## Experiment tracking

Dual sink:

- MLflow file backend under `mlruns/`
- JSON mirror under `results/logs/`

If the JSON sink fails, the MLflow run is preserved and tagged with
`json_sink_failed=true` instead of aborting the experiment.

```bash
mlflow ui --port 5000
```

## Documentation

- [plan.md](plan.md) — engineering implementation contract
- [principles.md](principles.md) — engineering rules and philosophy
- [docstring.md](docstring.md) — documentation standard for docstrings, comments, README, and notebooks
- [audit.md](audit.md) — compliance audit and scope checks

## Citation

```bibtex
@inproceedings{madry2018towards,
  title     = {Towards Deep Learning Models Resistant to Adversarial Attacks},
  author    = {Madry, Aleksander and Makelov, Aleksandar and Schmidt, Ludwig
               and Tsipras, Dimitris and Vladu, Adrian},
  booktitle = {ICLR},
  year      = {2018},
}
```

## License

See `LICENSE`.
