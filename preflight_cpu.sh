#!/usr/bin/env bash
#
# Check-then-fetch preflight for gn_brdtr_pipeline on Lightning.ai (CPU phase).
#
# PHASE 1 audits every artifact the pipeline expects -- correct path, non-zero
# size, and where a checksum is published, the checksum. PHASE 2 downloads only
# what PHASE 1 flagged. PHASE 3 re-audits. Nothing here needs a GPU.
#
#   bash preflight_cpu.sh                  # audit, then fetch what's missing
#   CHECK_ONLY=1 bash preflight_cpu.sh     # audit only, download nothing
#   SPLITS="test valid" bash preflight_cpu.sh
#   SKIP_BROADTRACK=1 bash preflight_cpu.sh
#
# NOTE: the pipeline resolves project_dir as ${hydra:runtime.cwd}, so every path
# below is relative to the directory you later launch `tracklab` from. Run this
# from the repo root and launch from the repo root.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV="${VENV:-.venv}"; SPLITS="${SPLITS:-test}"
DATA="data/SoccerNetGS"
JN="plugins/jn_gsr";     BT="pretrained_models/broadtrack"
PY="uv run --python ${VENV}"

# Published checksums, copied from the repo's own fetchers.
SHA_DBNET="1e8ee32969a0264dd8d26918a3dea7ab9914c90033b087b1bf97c5eefecfe6c9"
SHA_LEGIB="b9c61dabaea4a6ec99528c5ae394f5875aecb8207de38484eccb0f977a373e41"
SHA_PARSEQ="22d936444e09b0358b5b7339c2971ab5e792fee9d53dd30a98917abcd3ee1887"
# SATRN second recogniser: only a 16-hex prefix is on record (observed by the
# audited fusion run of 2026-08-17; see plugins/jn_gsr/fetch_weights.py).
SHA_SATRN_PREFIX="9e8f73b300754c35"
SATRN_CKPT="recog2/best_recog_word_acc_epoch_10.pth"
# The staged PARSeq file. Its name MUST contain 'parseq' and none of 'abinet',
# 'crnn', 'trba', 'trbc', 'vitstr' -- strhub routes on the path string.
PARSEQ_CKPT="parseq_gsr_ft_s1.ckpt"

NEED_DATA=0; NEED_HF=0; NEED_TEAM=0; NEED_JN=0; NEED_BT=0; FAILED=0
G="\033[0;32m"; R="\033[0;31m"; Y="\033[0;33m"; N="\033[0m"
ok()   { printf "  ${G}OK${N}    %s\n" "$1"; }
bad()  { printf "  ${R}MISS${N}  %s\n" "$1"; FAILED=1; }
warn() { printf "  ${Y}WARN${N}  %s\n" "$1"; }
hdr()  { printf "\n\033[1m%s\033[0m\n" "$1"; }

# check_file <path> <min_bytes> [md5|sha256] [expected] -> 0 good, 1 needs fetch
check_file() {
  local p="$1" min="$2" kind="${3:-}" want="${4:-}" sz sum
  [ -f "$p" ] || { bad "$p  (absent)"; return 1; }
  sz=$(stat -c%s "$p")
  if [ "$sz" -lt "$min" ]; then
    bad "$p  (${sz} B -- truncated or an LFS pointer stub)"; return 1
  fi
  if [ -n "$kind" ]; then
    sum=$([ "$kind" = md5 ] && md5sum "$p" | cut -d' ' -f1 || sha256sum "$p" | cut -d' ' -f1)
    if [ "$sum" != "$want" ]; then
      bad "$p  ($kind mismatch: ${sum:0:16}... != ${want:0:16}...)"; return 1
    fi
    ok "$p  ($(du -h "$p" | cut -f1), $kind verified)"; return 0
  fi
  ok "$p  ($(du -h "$p" | cut -f1))"; return 0
}

######################## PHASE 1 -- AUDIT (no downloads) #######################
hdr "PHASE 1 - audit"
echo "  cwd: $(pwd)   <- tracklab must be launched from here"

hdr "1. toolchain"
command -v uv >/dev/null 2>&1 && ok "uv" || { warn "uv absent (will install)"; }
[ -d "${VENV}" ] && ok "${VENV}" || warn "${VENV} absent (will create)"

hdr "2. dataset  (${DATA}/<split>/SNGS-*/img1 + Labels-GameState.json)"
for s in ${SPLITS}; do
  case "$s" in train) exp=57;; valid) exp=59;; test) exp=49;; *) exp=0;; esac
  n=$(find "${DATA}/${s}" -maxdepth 1 -type d -name 'SNGS-*' 2>/dev/null | wc -l)
  if [ "$n" -eq 0 ]; then bad "${DATA}/${s}  (no SNGS-* sequences)"; NEED_DATA=1; continue; fi
  one=$(find "${DATA}/${s}" -maxdepth 1 -type d -name 'SNGS-*' | sort | head -1)
  if [ ! -d "${one}/img1" ]; then
    bad "${one}/img1 absent -- zip extracted to the wrong depth"; NEED_DATA=1; continue
  fi
  [ -f "${one}/Labels-GameState.json" ] || warn "${one}/Labels-GameState.json absent (needed for eval)"
  f=$(find "${one}/img1" -name '*.jpg' | wc -l)
  if [ "$exp" -gt 0 ] && [ "$n" -ne "$exp" ]; then
    warn "split '${s}': ${n} sequences, expected ${exp} -- partial extraction?"
  else
    ok "split '${s}': ${n} sequences, ${f} frames in $(basename "$one")"
  fi
done

hdr "3. detector  (HF cache, \${hf:} resolver)"
if hf_out=$(${PY} python - <<'PYEOF' 2>/dev/null
from huggingface_hub import try_to_load_from_cache
# Only the detector: no config references sports_model.pth.tar-60, and the successful
# end-to-end Kaggle run of 2026-09-02 never touched it (removed as a stale leftover of
# the upstream baseline; the tracker/split_merge ReID comes from Ynniss/osnet_ain, §5).
for f in ("yolov11_sn_best.pt",):
    p = try_to_load_from_cache("Ynniss/sn-gamestate-weights", f)
    print(("OK " if isinstance(p, str) else "MISS ") + f)
PYEOF
); then
  echo "$hf_out" | while read -r st f; do [ "$st" = OK ] && ok "hf: $f" || bad "hf: $f"; done
  echo "$hf_out" | grep -q MISS && NEED_HF=1
else
  bad "hf cache unreadable (huggingface_hub missing?)"; NEED_HF=1
fi

hdr "4. team-appearance model  (HF Ynniss/osnet_team, digest recorded by the audit)"
if team_out=$(${PY} python - <<'PYEOF' 2>/dev/null
from huggingface_hub import try_to_load_from_cache
p = try_to_load_from_cache("Ynniss/osnet_team", "osnet_team_best.pt")
print(("OK " if isinstance(p, str) else "MISS ") + "osnet_team_best.pt")
PYEOF
); then
  echo "$team_out" | while read -r st f; do [ "$st" = OK ] && ok "hf: $f" || bad "hf: $f"; done
  echo "$team_out" | grep -q MISS && NEED_TEAM=1
else
  bad "hf cache unreadable (huggingface_hub missing?)"; NEED_TEAM=1
fi

hdr "5. jersey stage  (${JN})"
[ -x "${JN}/.venv_jn/bin/python" ] && ok "${JN}/.venv_jn/bin/python" \
  || { bad "${JN}/.venv_jn/bin/python  (venv not built)"; NEED_JN=1; }
check_file "${JN}/models/best_icdar_hmean_epoch_10.pth" 100000000 sha256 "$SHA_DBNET" || NEED_JN=1
check_file "${JN}/models/sn_legibility.pth"              80000000 sha256 "$SHA_LEGIB" || NEED_JN=1
# 250 MB floor: the checkpoint is ~286 MB (272.74 MiB of storages + zip
# records). The floor exists to catch truncation and LFS pointer stubs, not to
# pin the size -- keep it BELOW the real size whenever the checkpoint changes.
if check_file "${JN}/models/${PARSEQ_CKPT}" 250000000; then
  s=$(sha256sum "${JN}/models/${PARSEQ_CKPT}" | cut -d' ' -f1)
  [ "$s" = "$SHA_PARSEQ" ] && ok "  parseq matches the upstream sha256" \
    || warn "  parseq sha differs from upstream (fine for a private fine-tune; \
the flag travels with the results)"
else NEED_JN=1; fi
# 40 MB floor: the checkpoint is ~47.7 MB. Prefix-checked, see SHA_SATRN_PREFIX.
if check_file "${JN}/models/${SATRN_CKPT}" 40000000; then
  s=$(sha256sum "${JN}/models/${SATRN_CKPT}" | cut -d' ' -f1)
  case "$s" in
    ${SHA_SATRN_PREFIX}*) ok "  satrn matches the recorded sha256 prefix";;
    *) warn "  satrn sha ${s:0:16}... differs from the recorded prefix ${SHA_SATRN_PREFIX} \
(a re-upload; the flag travels with the results)";;
  esac
  [ -f "${JN}/models/recog2/config_from_checkpoint.py" ] \
    && ok "${JN}/models/recog2/config_from_checkpoint.py" \
    || warn "${JN}/models/recog2/config_from_checkpoint.py absent (mmocr_reader recovers it at load)"
else NEED_JN=1; fi
# Fetched by the setup but NOT referenced by jn_recognizer.py -> never blocking.
[ -f "${JN}/models/resnet18_multi_single.pt" ] && ok "${JN}/models/resnet18_multi_single.pt (optional)" \
  || warn "${JN}/models/resnet18_multi_single.pt absent (optional, unused by the GSR recognizer)"

hdr "6. BroadTrack  (${BT})"
check_file "${BT}/models/nbjw_keypoint_model.pt" 1000000 || NEED_BT=1
check_file "${BT}/models/tvcalib_model.pt"       1000000 || NEED_BT=1
[ -d "${BT}/libtorch/lib" ] && ok "${BT}/libtorch/lib" || { bad "${BT}/libtorch/lib"; NEED_BT=1; }
[ -f "${BT}/src/scripts/compute_tripod.py" ] && ok "${BT}/src/scripts/compute_tripod.py" \
  || { bad "${BT}/src/scripts/compute_tripod.py"; NEED_BT=1; }
[ -x "${BT}/bin/broadtrack" ] && ok "${BT}/bin/broadtrack (built)" \
  || warn "${BT}/bin/broadtrack not built -- expected on CPU, build on the GPU"

if [ "${CHECK_ONLY:-0}" = "1" ]; then
  hdr "audit only (CHECK_ONLY=1) - nothing downloaded"
  [ "$FAILED" -eq 0 ] && echo "  all required artifacts present" || echo "  items marked MISS need fetching"
  exit $FAILED
fi
if [ "$FAILED" -eq 0 ]; then
  hdr "everything already present - nothing to download"; exit 0
fi

######################### PHASE 2 -- FETCH WHAT'S MISSING ######################
hdr "PHASE 2 - fetching only the flagged items"

command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="${HOME}/.local/bin:${PATH}"; }
[ -d "${VENV}" ] || uv venv --python 3.9 "${VENV}"
uv pip install -q --python "${VENV}" -e . && uv pip install -q --python "${VENV}" "huggingface_hub[cli]"
SNGS=$(${PY} python -c "import tracklab,os;print(os.path.join(os.path.dirname(tracklab.__file__),'wrappers','dataset','soccernet','soccernet_game_state.py'))" 2>/dev/null)
[ -n "$SNGS" ] && sed -i 's/gamestate-2025/gamestate-2024/g' "$SNGS" && ok "tracklab task-name patch applied"

if [ "$NEED_DATA" = 1 ]; then
  hdr "-> dataset  (SoccerNet server, Hugging Face SoccerNet/SN-GSR-2024 fallback)"
  mkdir -p "${DATA}"
  for s in ${SPLITS}; do
    n=$(find "${DATA}/${s}" -maxdepth 1 -type d -name 'SNGS-*' 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] && [ -d "$(find "${DATA}/${s}" -maxdepth 1 -type d -name 'SNGS-*'|sort|head -1)/img1" ] && continue
    ZIP="${DATA}/gamestate-2024/${s}.zip"
    ${PY} python - <<PYEOF || warn "SoccerNet server download failed; trying the Hugging Face mirror"
from SoccerNet.Downloader import SoccerNetDownloader
SoccerNetDownloader(LocalDirectory="${DATA}").downloadDataTask(task="gamestate-2024", split=["${s}"])
PYEOF
    # Absent, empty or truncated zip (no end-of-central-directory) -> fall back.
    if [ ! -s "${ZIP}" ] || ! ${PY} python -c "import zipfile; zipfile.ZipFile(r'${ZIP}').namelist()" >/dev/null 2>&1; then
      warn "falling back to Hugging Face: SoccerNet/SN-GSR-2024 ${s}.zip"
      ZIP=$(${PY} python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('SoccerNet/SN-GSR-2024', '${s}.zip', repo_type='dataset'))") \
        || { bad "Hugging Face dataset fallback failed for split '${s}'"; continue; }
    fi
    unzip -q -o "${ZIP}" -d "${DATA}/${s}"
  done
fi

if [ "$NEED_HF" = 1 ]; then
  hdr "-> detector (the only file the configs reference in this repo)"
  ${PY} hf download Ynniss/sn-gamestate-weights yolov11_sn_best.pt \
    || warn "HF fetch failed - private repo? export HF_TOKEN=hf_..."
fi

if [ "$NEED_TEAM" = 1 ]; then
  hdr "-> team-appearance model"
  ${PY} hf download Ynniss/osnet_team osnet_team_best.pt \
    || warn "HF fetch failed - private repo? export HF_TOKEN=hf_..."
fi

if [ "$NEED_JN" = 1 ]; then
  hdr "-> jersey stage"
  # setup_jn_gsr.sh runs venv -> weights -> self-tests. Its venv check imports
  # mmcv's DCNv2 CUDA op, which can fail without a GPU and abort before the
  # weights step, so fall back to the two fetchers directly (both hash-verify).
  bash scripts/setup_jn_gsr.sh
  if [ ! -s "${JN}/models/${PARSEQ_CKPT}" ] && [ -x "${JN}/.venv_jn/bin/python" ]; then
    warn "setup stopped early - fetching checkpoints directly"
    ( cd "${JN}" && ./.venv_jn/bin/python fetch_weights.py --out-dir models \
      && ./.venv_jn/bin/python stage_weights.py --out "models/${PARSEQ_CKPT}" )
  fi
fi

if [ "$NEED_BT" = 1 ] && [ "${SKIP_BROADTRACK:-0}" != "1" ]; then
  hdr "-> BroadTrack"
  # Order is apt -> clone+LFS+models -> libtorch -> cmake. A cmake failure on a
  # CPU box (no nvcc) still leaves weights and libtorch on disk; re-running on
  # the GPU skips straight to the build.
  bash scripts/setup_broadtrack.sh
fi

############################### PHASE 3 -- RE-AUDIT ############################
hdr "PHASE 3 - re-audit"
CHECK_ONLY=1 SPLITS="${SPLITS}" bash "$0"
