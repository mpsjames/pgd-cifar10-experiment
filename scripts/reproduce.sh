#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
PYTHON="${PYTHON:-.venv/bin/python}"
PYTEST="${PYTEST:-.venv/bin/pytest}"
HARDWARE="${HARDWARE:-}"
HARDWARE_ARGS=()
if [[ -n "${HARDWARE}" ]]; then
  HARDWARE_ARGS=(--hardware "${HARDWARE}")
fi

"${PYTEST}" tests/ -q

if [[ "${MODE}" == "--smoke" || "${MODE}" == "smoke" ]]; then
  # When HARDWARE is not set, fall back to cpu if CUDA is unavailable.
  if [[ -z "${HARDWARE}" ]]; then
    if ! "${PYTHON}" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
      HARDWARE="cpu"
      HARDWARE_ARGS=(--hardware cpu)
    fi
  fi
  "${PYTHON}" scripts/train_clean.py --arch resnet18 --seed 42 --epochs 1 --batch-size 4 --smoke --no-mlflow "${HARDWARE_ARGS[@]}"
  "${PYTHON}" scripts/train_adversarial.py --arch resnet18 --seed 42 --epochs 1 --batch-size 4 --smoke --no-mlflow "${HARDWARE_ARGS[@]}"
  "${PYTHON}" scripts/run_white_box.py --arch vit_tiny --attack apgd_ce_10 --seed 42 --batch-size 4 --smoke --no-mlflow "${HARDWARE_ARGS[@]}"
  "${PYTHON}" scripts/run_transfer.py --mode gray_box --seed 42 --batch-size 4 --max-pairs 1 --smoke --no-mlflow "${HARDWARE_ARGS[@]}"
  "${PYTHON}" scripts/run_epsilon_sweep.py --arch resnet18 --seed 42 --attack apgd_ce_10 --batch-size 4 --smoke --no-mlflow "${HARDWARE_ARGS[@]}"
  "${PYTHON}" scripts/run_black_box_square.py --arch resnet18 --seed 42 --num-queries 8 --batch-size 4 --smoke --no-mlflow "${HARDWARE_ARGS[@]}"
  echo "Smoke reproduction completed."
  exit 0
fi

ARCHES=(resnet18 vit_tiny)

for arch in "${ARCHES[@]}"; do
  "${PYTHON}" scripts/train_clean.py --arch "${arch}" --seed 42 --epochs 200 --batch-size 256 "${HARDWARE_ARGS[@]}"
done

for arch in "${ARCHES[@]}"; do
  "${PYTHON}" scripts/train_adversarial.py --arch "${arch}" --seed 42 --epochs 200 --batch-size 256 "${HARDWARE_ARGS[@]}"
done

WHITE_BOX_ATTACKS=(fgsm bim_10 pgd_10 pgd_40 pgd_100 apgd_ce_10 apgd_ce_100)

for arch in "${ARCHES[@]}"; do
  for attack in "${WHITE_BOX_ATTACKS[@]}"; do
    "${PYTHON}" scripts/run_white_box.py --arch "${arch}" --attack "${attack}" --seed 42 "${HARDWARE_ARGS[@]}"
  done
done

"${PYTHON}" scripts/run_transfer.py --mode cross_arch "${HARDWARE_ARGS[@]}"
"${PYTHON}" scripts/run_transfer.py --mode gray_box "${HARDWARE_ARGS[@]}"
"${PYTHON}" scripts/run_epsilon_sweep.py "${HARDWARE_ARGS[@]}"

for arch in "${ARCHES[@]}"; do
  for variant in clean adv; do
    "${PYTHON}" scripts/run_black_box_square.py --arch "${arch}" --seed 42 --variant "${variant}" --num-queries 5000 "${HARDWARE_ARGS[@]}"
  done
done

# Notebooks are executed manually or via CI.
# Do NOT run notebooks --inplace here; that would mutate tracked source files.
# To produce executed copies, run:
#   jupyter nbconvert --to notebook --execute notebooks/*.ipynb --output-dir results/executed_notebooks/
