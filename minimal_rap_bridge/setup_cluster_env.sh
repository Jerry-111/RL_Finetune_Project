#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${1:-${ROOT_DIR}/.venv-pufferdrive-rap}"
MAP_DIR="${2:-resources/drive/binaries}"

echo "[bridge-setup] delegating to env bootstrap script..."
echo "[bridge-setup] root=${ROOT_DIR}"
echo "[bridge-setup] venv=${VENV_PATH}"
echo "[bridge-setup] map_dir=${MAP_DIR}"

bash "${ROOT_DIR}/envs/pufferdrive_rap_minimal/setup_env.sh" "${VENV_PATH}" "${MAP_DIR}"

echo
echo "[bridge-setup] done. Activate with:"
echo "  source \"${VENV_PATH}/bin/activate\""
