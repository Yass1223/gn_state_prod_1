#!/usr/bin/env bash
#
# jn_pipeline_gsr provisioning for sn-gamestate-lightning (Kaggle / Lightning.ai).
#
# The JN package needs its OWN interpreter (python 3.10, torch 2.0.1+cu118,
# mmcv 2.0.1, numpy 1.25.2, PARSeq) -- incompatible with this repo's env by
# design. This script:
#
#   1. ensures the package is vendored at plugins/jn_gsr
#      (copies from $JN_SRC if given and the folder is missing)
#   2. builds the package's own venv via its setup_env.py  (~7 GB;
#      JN_VENV overridable -- setup_kaggle.py convention: put it on the
#      ephemeral scratch disk on Kaggle)
#   3. fetches the hash-checked checkpoints into plugins/jn_gsr/models:
#      DBNet++ (111 MB) + legibility ResNet-34 (85 MB) + SATRN second
#      recogniser (48 MB, staged under models/recog2/ with its config
#      recovered from the checkpoint) via fetch_weights.py,
#      PARSeq (286 MB, load-gated through strhub) via stage_weights.py
#   4. runs the offline self-tests, then audits the staged PARSeq checkpoint
#      (audit_parseq.py --stages d1,d2: sha256 + size, and a strhub load that
#      asserts the model class, hparams, head width, tokenizer and parameter
#      count). d1/d2 need only the venv and the checkpoint, both of which
#      steps 2 and 3 have already produced.
#
# Usage:
#   bash scripts/setup_jn_gsr.sh
#   JN_SRC=/path/to/jn_pipeline_gsr bash scripts/setup_jn_gsr.sh   # vendor first
#   JN_VENV=/kaggle/tmp/.venv_jn    bash scripts/setup_jn_gsr.sh   # Kaggle scratch
#
# Weights come from the Ynniss HuggingFace repos (sha256-verified by the package
# itself); internet must be ON, HF_TOKEN exported if the repos are private.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JN_PKG="${JN_PKG:-${REPO_ROOT}/plugins/jn_gsr}"
export JN_VENV="${JN_VENV:-${JN_PKG}/.venv_jn}"

echo "=================================================================="
echo " jn_pipeline_gsr setup"
echo "   package: ${JN_PKG}"
echo "   venv:    ${JN_VENV}"
echo "=================================================================="

# --- 1. vendored package present? ---------------------------------------------
if [ ! -f "${JN_PKG}/jn_recognizer.py" ]; then
  if [ -n "${JN_SRC:-}" ] && [ -f "${JN_SRC}/jn_recognizer.py" ]; then
    echo "==> Vendoring package from ${JN_SRC}"
    mkdir -p "${JN_PKG}"
    cp -r "${JN_SRC}/." "${JN_PKG}/"
  else
    echo "ERROR: package not found at ${JN_PKG}." >&2
    echo "  Copy jn_pipeline_gsr there (Windows:  robocopy /E ^" >&2
    echo "    C:\\...\\jn_pipeline_gsr  <repo>\\plugins\\jn_gsr )" >&2
    echo "  or re-run with JN_SRC=/path/to/jn_pipeline_gsr" >&2
    exit 1
  fi
fi
for f in predict_tracklets.py setup_env.py fetch_weights.py stage_weights.py audit_parseq.py \
         jn_recognizer.py fuse_jn.py mmocr_reader.py; do
  [ -f "${JN_PKG}/${f}" ] || { echo "ERROR: ${JN_PKG}/${f} missing -- stale copy? Re-vendor." >&2; exit 1; }
done

cd "${JN_PKG}"

# --- 2. the package's own venv (its setup_env.py owns every pin) ---------------
PYBIN="${JN_VENV}/bin/python"
if [ ! -x "${PYBIN}" ]; then
  echo "==> Building the JN venv (this is the long step)"
  python3 setup_env.py
else
  echo "==> Venv present; verifying"
  python3 setup_env.py --check || { echo "==> Check failed; repairing"; python3 setup_env.py; }
fi

# --- 3. checkpoints (hash-checked by the package's own fetchers) ---------------
echo "==> Fetching DBNet++ + legibility + SATRN checkpoints"
"${PYBIN}" fetch_weights.py --out-dir models
[ -s models/recog2/best_recog_word_acc_epoch_10.pth ] \
  || { echo "ERROR: SATRN checkpoint not staged under models/recog2/" >&2; exit 1; }
echo "==> Fetching + load-gating the PARSeq checkpoint"
"${PYBIN}" stage_weights.py --out models/parseq_gsr_ft_s1.ckpt

# --- 4. self-tests + checkpoint audit ------------------------------------------
echo "==> Worker self-test (offline)"
"${PYBIN}" predict_tracklets.py --self-test
echo "==> Package offline self-tests"
"${PYBIN}" gsr_adapter.py
"${PYBIN}" evaluate_jn.py
"${PYBIN}" fuse_jn.py
"${PYBIN}" mmocr_reader.py
"${PYBIN}" jn_recognizer.py

# The staged checkpoint, asserted. stage_weights.py above REPORTS a hash
# mismatch and continues by design (a private fine-tune is legitimate); this is
# the arm that refuses. d2 also catches a strhub misroute -- load_from_checkpoint
# picks the model class by substring-matching the checkpoint PATH, parent
# directories included, so a repository checked out under a directory containing
# 'abinet'/'crnn'/'trba'/'trbc'/'vitstr' would build the wrong class and fail
# with a confusing shape error hours later instead of here.
# `set -e` is in force: a failure stops provisioning, which is intended.
echo "==> Checkpoint audit (provenance + shapes)"
"${PYBIN}" audit_parseq.py --stages d1,d2 --ckpt models/parseq_gsr_ft_s1.ckpt

cat <<EOF

==================================================================
 Done.
   worker      : ${JN_PKG}/predict_tracklets.py
   interpreter : ${PYBIN}
   models      : ${JN_PKG}/models
   recognisers : PARSeq models/parseq_gsr_ft_s1.ckpt
                 SATRN  models/recog2/best_recog_word_acc_epoch_10.pth
                 consolidation: vote_pool (the only rule of this build)

 The module config (configs/modules/jersey_number_detect/jn_gsr.yaml)
 expects the package at <repo>/plugins/jn_gsr. If JN_VENV was moved
 (e.g. Kaggle scratch), override at run time:
   tracklab -cn soccernet_jngsr \\
     modules.jersey_number_detect.cfg.venv_python=${PYBIN}

 GPU sharding is automatic (nvidia-smi): 2 workers on 2xT4,
 4 on 4xT4, 1 otherwise.
==================================================================
EOF
