#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
PYTHON="${PYTHON:-.venv/bin/python}"
PYTEST="${PYTEST:-.venv/bin/pytest}"

"${PYTEST}" tests/ -q

if [[ "${MODE}" == "--smoke" || "${MODE}" == "smoke" ]]; then
  "${PYTHON}" scripts/train_clean.py --arch resnet18 --seed 42 --epochs 1 --batch-size 4 --smoke
  "${PYTHON}" scripts/run_white_box.py --arch resnet18 --seed 42 --batch-size 4 --smoke
  "${PYTHON}" scripts/run_transfer.py --mode cross_seed --seed 42 --batch-size 4 --max-pairs 1 --smoke
  "${PYTHON}" scripts/run_epsilon_sweep.py --arch resnet18 --seed 42 --batch-size 4 --smoke
  echo "Smoke reproduction completed."
  exit 0
fi

ARCHES=(resnet18 wrn_34_10 resnet50 vgg16_bn)
SEEDS=(42 123 456 789 1024)

for arch in "${ARCHES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    "${PYTHON}" scripts/train_clean.py --arch "${arch}" --seed "${seed}" --epochs 100
  done
done

for arch in "${ARCHES[@]}"; do
  "${PYTHON}" scripts/train_adversarial.py --arch "${arch}" --seed 42 --epochs 100
done

for arch in "${ARCHES[@]}"; do
  "${PYTHON}" scripts/run_white_box.py --arch "${arch}" --seed 42
done

"${PYTHON}" scripts/run_transfer.py --mode cross_arch
"${PYTHON}" scripts/run_transfer.py --mode cross_seed
"${PYTHON}" scripts/run_epsilon_sweep.py

for nb in notebooks/*.ipynb; do
  "${PYTHON}" -m jupyter nbconvert --to notebook --execute "${nb}" --inplace
done
