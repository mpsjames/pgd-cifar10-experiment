#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
PYTHON="${PYTHON:-.venv/bin/python}"
PYTEST="${PYTEST:-.venv/bin/pytest}"

"${PYTEST}" tests/ -q

if [[ "${MODE}" == "--smoke" || "${MODE}" == "smoke" ]]; then
  "${PYTHON}" scripts/train_clean.py --arch resnet18 --seed 42 --epochs 1 --batch-size 4 --smoke --no-mlflow
  "${PYTHON}" scripts/train_adversarial.py --arch resnet18 --seed 42 --epochs 1 --batch-size 4 --smoke --no-mlflow
  "${PYTHON}" scripts/run_white_box.py --arch vit_tiny --attack apgd_ce_10 --seed 42 --batch-size 4 --smoke --no-mlflow
  "${PYTHON}" scripts/run_transfer.py --mode gray_box --seed 42 --batch-size 4 --max-pairs 1 --smoke --no-mlflow
  "${PYTHON}" scripts/run_epsilon_sweep.py --arch resnet18 --seed 42 --attack apgd_ce_10 --batch-size 4 --smoke --no-mlflow
  "${PYTHON}" scripts/run_black_box_square.py --arch resnet18 --seed 42 --num-queries 8 --batch-size 4 --smoke --no-mlflow
  echo "Smoke reproduction completed."
  exit 0
fi

ARCHES=(resnet18 wrn_34_10 vit_tiny)

for arch in "${ARCHES[@]}"; do
  "${PYTHON}" scripts/train_clean.py --arch "${arch}" --seed 42 --epochs 100
done

for arch in "${ARCHES[@]}"; do
  "${PYTHON}" scripts/train_adversarial.py --arch "${arch}" --seed 42 --epochs 100
done

for arch in "${ARCHES[@]}"; do
  "${PYTHON}" scripts/run_white_box.py --arch "${arch}" --attack pgd_10 --seed 42
  "${PYTHON}" scripts/run_white_box.py --arch "${arch}" --attack apgd_ce_100 --seed 42
done

"${PYTHON}" scripts/run_transfer.py --mode cross_arch
"${PYTHON}" scripts/run_transfer.py --mode gray_box
"${PYTHON}" scripts/run_epsilon_sweep.py

for arch in "${ARCHES[@]}"; do
  for variant in clean adv; do
    "${PYTHON}" scripts/run_black_box_square.py --arch "${arch}" --seed 42 --variant "${variant}" --num-queries 5000
  done
done

for nb in notebooks/*.ipynb; do
  "${PYTHON}" -m jupyter nbconvert --to notebook --execute "${nb}" --inplace
done
