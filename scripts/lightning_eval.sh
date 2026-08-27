#!/usr/bin/env bash
#
# SoccerNet Game State Reconstruction - evaluation runner (Lightning Studio / Kaggle).
# Self-healing: patches a known dataset-task bug, downloads only the needed
# split(s), installs deps into a local .venv, and runs single-worker to avoid
# the cuDNN/torch_shm_manager issue on the Studio.
#
# This repo has a SINGLE entry config (`soccernet`): YOLO11-SNFT -> prtreid ->
# BoT-SORT·SOF -> GTA-Link -> BroadTrack -> jn_pipeline_gsr -> voting -> team.
#
# BroadTrack and the jersey stage each need a one-off provisioning step (both are
# Docker-free). Run them before the first evaluation:
#     bash scripts/setup_broadtrack.sh
#     bash scripts/setup_jn_gsr.sh
# Set SKIP_SETUP=1 to skip them once they are cached in persistent storage.

set -euo pipefail

SPLITS="${SPLITS:-test}"
NVID="${NVID:--1}"
RESULTS_DIR="${RESULTS_DIR:-eval_results}"
CONFIG_NAME="${CONFIG_NAME:-soccernet}"
VENV="${VENV:-.venv}"
DATA_DIR="data/SoccerNetGS"

echo "=================================================================="
echo " SoccerNet GSR evaluation | config=${CONFIG_NAME} splits=${SPLITS} nvid=${NVID}"
echo "=================================================================="

# 1. Dependencies into a local .venv (Python 3.9)
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi
uv venv --clear --python 3.9 "${VENV}"
uv pip install --python "${VENV}" -e .

# Everything below invokes the venv's interpreter DIRECTLY rather than through
# `uv run`. There is no uv.lock, so `uv run` performs its own fresh resolution
# and re-syncs the environment, silently undoing what `uv pip install` just
# produced (observed: it swapped 12 packages, breaking the
# torchmetrics -> transformers -> huggingface_hub import chain). Using the
# interpreter directly keeps one resolver in charge of the environment.
PYBIN="${VENV}/bin/python"
TRACKLAB_BIN="${VENV}/bin/tracklab"

# 2. Patch the gamestate-2025 -> 2024 bug in the installed TrackLab
SNGS=$("${PYBIN}" -c "import tracklab,os;print(os.path.join(os.path.dirname(tracklab.__file__),'wrappers','dataset','soccernet','soccernet_game_state.py'))")
sed -i 's/gamestate-2025/gamestate-2024/g' "${SNGS}"
echo "==> Patched dataset task name in ${SNGS}"

# 3. Download only the needed split(s) and unzip into place
mkdir -p "${DATA_DIR}"
for split in ${SPLITS}; do
  if [ ! -d "${DATA_DIR}/${split}" ]; then
    echo "==> Downloading SoccerNetGS split: ${split}"
    "${PYBIN}" -c "
from SoccerNet.Downloader import SoccerNetDownloader
d = SoccerNetDownloader(LocalDirectory='${DATA_DIR}')
d.downloadDataTask(task='gamestate-2024', split=['${split}'])
"
    unzip -o "${DATA_DIR}/gamestate-2024/${split}.zip" -d "${DATA_DIR}/${split}"
  else
    echo "==> Split '${split}' already present, skipping download."
  fi
done

# 4. One-off provisioning of the two subprocess-run stages
if [ "${SKIP_SETUP:-0}" != "1" ]; then
  echo "==> Provisioning BroadTrack (native build, no Docker)"
  bash scripts/setup_broadtrack.sh
  echo "==> Provisioning the jersey-number pipeline (own venv + weights)"
  bash scripts/setup_jn_gsr.sh
fi

# 5. Run evaluation per split (single-worker; data already present)
mkdir -p "${RESULTS_DIR}"
for split in ${SPLITS}; do
  echo "==> Evaluating split: ${split} (nvid=${NVID})"
  echo "n" | "${TRACKLAB_BIN}" -cn "${CONFIG_NAME}" \
      dataset.eval_set="${split}" \
      dataset.nvid="${NVID}" \
      num_cores=0 \
      2>&1 | tee "${RESULTS_DIR}/eval_${split}.log"
done

echo "==> Done. Logs (with metric tables) in ${RESULTS_DIR}/eval_<split>.log"
echo "==> Reference metrics:"
echo "    ${VENV}/bin/python scripts/reference_metrics.py \\"
echo "      --state states/sn-gamestate.pklz --dataset-path ${DATA_DIR} \\"
echo "      --eval-set ${SPLITS} --out reference_metrics"
