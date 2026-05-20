#!/usr/bin/env bash
set -euo pipefail

HOST="${MLFLOW_HOST:-127.0.0.1}"
PORT="${MLFLOW_PORT:-5000}"
BACKEND="${MLFLOW_BACKEND_STORE_URI:-sqlite:///mlruns.db}"
ARTIFACTS="${MLFLOW_ARTIFACTS_DEST:-./mlartifacts}"

if [[ "${BACKEND}" == sqlite:///* ]]; then
  mkdir -p "$(dirname "${BACKEND#sqlite:///}")"
fi
mkdir -p "${ARTIFACTS}"

exec mlflow server \
  --host "${HOST}" \
  --port "${PORT}" \
  --backend-store-uri "${BACKEND}" \
  --artifacts-destination "${ARTIFACTS}"
