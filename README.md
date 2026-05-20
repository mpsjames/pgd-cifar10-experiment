# PGD Adversarial Attack & Defense — Multi-Architecture Study on CIFAR-10

> Reproducible CIFAR-10 study of PGD/APGD attacks, adversarial training, and
> transfer behavior across three image-classification architectures.
>
> The repo is organized as production code under `src/`, thin CLI entrypoints
> under `scripts/`, and executable report notebooks that stay honest when the
> full checkpoint campaign has not been run yet.

## What this is

- A CIFAR-10-only implementation of FGSM, BIM, PGD, APGD-CE white-box evaluation, transfer attacks, epsilon sweeps, and APGD adversarial training.
- A reproducibility-oriented experiment harness with frozen configs, deterministic seeding, MLflow plus JSON tracking, and notebook reports backed by shared `src/` helpers.
- A multi-architecture benchmark covering `resnet18`, `wrn_34_10`, and `vit_tiny`.
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

This project reports single-seed results with `seed=42` for all phases.

Full reproduction entrypoint:

```bash
bash scripts/reproduce.sh
```

The full campaign trains clean checkpoints for all three architectures with
`seed=42`, runs adversarial training, executes white-box and transfer
evaluations, performs the epsilon sweep, and executes all notebooks in place.

Hardware target: NVIDIA A1000 4 GB VRAM. Expected runtime is roughly
119-141 GPU-hours. WRN-34-10 adversarial training may fall back to the
RobustBench checkpoint path when the retry chain still OOMs; this is surfaced
through MLflow tags such as `fallback_triggered=true` and
`wrn_at_source=robustbench`.

## Architecture

```text
scripts/ -> src/cli/ -> src/experiments/runner.py
                    -> src/training/{CleanTrainer,AdversarialTrainer}
                    -> src/evaluation/AttackEvaluator
                    -> src/{attacks,models,data,tracking}
notebooks/ -> src/reporting.nb*_*
```

Stateful workflows use service objects (`ExperimentRunner`, trainers, and
`AttackEvaluator`). Pure transformations and serialization helpers remain
functions. Notebook code imports only from `src.reporting`; scripts share
bootstrap/checkpoint/smoke behavior through `src.cli`.

## Key contracts

- Reproducibility: entry points call `set_all_seeds(seed)` and tracking records Git/environment metadata.
- Raw-input attacks: attacks consume `[0, 1]` images, while `NormalizedModel` performs CIFAR-10 normalization inside `forward`.
- Fail-fast verification: attacked batches are checked with `verify_perturbation` before metrics are trusted.
- Frozen configs: experiment, training, model, and attack configs are immutable dataclasses after loading.
- Linf only: all attack/evaluation code assumes the project’s scoped perturbation norm.
- Honest notebooks: notebook helpers emit `full-campaign-pending` rather than inventing missing results.

## CLI

```text
scripts/train_clean.py          --arch {resnet18,wrn_34_10,vit_tiny} --seed N [--epochs E] [--smoke]
scripts/train_adversarial.py    --arch ... --seed N [--smoke]
scripts/run_white_box.py        --arch ... [--attack ATTACK] [--seed N] [--smoke]
scripts/run_transfer.py         --mode {cross_arch,gray_box} [--attack ATTACK] [--max-pairs K] [--smoke]
scripts/run_epsilon_sweep.py    [--arch ARCH] [--seed N] [--epsilon EPS] [--smoke]
scripts/run_black_box_square.py --arch ... [--variant clean|adv] [--num-queries N] [--smoke]
scripts/reproduce.sh        [--smoke]
```

Examples:

```bash
python scripts/train_clean.py --arch resnet18 --seed 42 --epochs 100
python scripts/train_adversarial.py --arch resnet18 --seed 42 --epochs 100
python scripts/run_white_box.py --arch resnet18 --seed 42
python scripts/run_transfer.py --mode cross_arch --seed 42
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

### Hardware presets

Override runtime settings via `--hardware <preset>`:

- `gpu_default` (default): conservative, deterministic, AMP per training config
- `cpu`: local testing without GPU
- `p40`: Tesla P40 cloud optimization (8 workers, persistent, cudnn benchmark, AMP off)

Example: `python scripts/train_clean.py --arch resnet18 --hardware p40 --batch-size 512`

## Experiment tracking

Tracking uses an MLflow HTTP server plus always-on local mirrors:

- MLflow API at `http://127.0.0.1:5000`
- JSON mirror under `results/logs/`
- Per-run log files plus rotating `results/logs/experiment.log`

If the JSON sink fails, the MLflow run is preserved and tagged with
`json_sink_failed=true` instead of aborting the experiment.

```bash
bash scripts/mlflow.sh
```

Browse the UI at `http://127.0.0.1:5000`. For one-off local or CI smoke runs
without the server, pass `--no-mlflow`; JSON and file logs still run.

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
