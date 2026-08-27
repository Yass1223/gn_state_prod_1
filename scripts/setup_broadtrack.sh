#!/usr/bin/env bash
#
# BroadTrack provisioning WITHOUT Docker (Kaggle / Lightning.ai friendly).
#
# The upstream Dockerfile is only a build recipe; Kaggle notebooks and Lightning Studios
# already give a root Linux shell with CUDA, so we execute that recipe natively:
#
#   1. clone github.com/evs-broadcast/BroadTrack with git-lfs   (weights come from EVS)
#   2. apt-get the same dependencies the Dockerfile installs
#   3. download libtorch 2.5.1+cu124
#   4. cmake + make + install
#   5. lay everything out under pretrained_models/broadtrack/ so the module config finds it
#
# Nothing from EVS is committed to this repo: their licence is noncommercial research with
# no redistribution, so the sources, the ~266 MB TorchScript weights and any generated
# calibration JSONs must stay out of public repos/datasets.
#
# Usage:
#   bash scripts/setup_broadtrack.sh                 # build into ./pretrained_models/broadtrack
#   BT_ROOT=/teamspace/studios/this_studio/bt bash scripts/setup_broadtrack.sh
#   SKIP_APT=1 bash scripts/setup_broadtrack.sh      # deps already installed
#
# The result is cacheable: copy $BT_ROOT to persistent storage (Lightning volume or a
# PRIVATE Kaggle dataset) and later runs skip the build entirely.
set -euo pipefail

BT_ROOT="${BT_ROOT:-$(pwd)/pretrained_models/broadtrack}"
BT_SRC="${BT_ROOT}/src"
BT_BIN="${BT_ROOT}/bin"
BT_MODELS="${BT_ROOT}/models"
LIBTORCH_DIR="${BT_ROOT}/libtorch"
LIBTORCH_URL="${LIBTORCH_URL:-https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip}"
JOBS="${JOBS:-$(nproc)}"

SUDO=""
if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi

echo "=================================================================="
echo " BroadTrack native build (no Docker)"
echo "   root:     ${BT_ROOT}"
echo "   jobs:     ${JOBS}"
echo "=================================================================="

mkdir -p "${BT_ROOT}" "${BT_BIN}" "${BT_MODELS}"

# --- 0. already built? ------------------------------------------------------------------
if [ -x "${BT_BIN}/broadtrack" ] && [ -s "${BT_MODELS}/nbjw_keypoint_model.pt" ]; then
  echo "==> Already provisioned at ${BT_ROOT}; nothing to do."
  echo "    (delete the folder to force a rebuild)"
  exit 0
fi

# --- 1. apt dependencies (same list as the upstream Dockerfile) --------------------------
if [ "${SKIP_APT:-0}" != "1" ]; then
  echo "==> Installing build dependencies"
  ${SUDO} apt-get update
  ${SUDO} apt-get install -y --no-install-recommends --no-install-suggests \
      git git-lfs cmake build-essential wget unzip \
      libboost-program-options-dev libboost-filesystem-dev \
      libeigen3-dev libflann-dev libgoogle-glog-dev \
      libgtest-dev libgmock-dev libsqlite3-dev libglew-dev \
      libceres-dev rapidjson-dev libopencv-dev
fi

# --- 2. sources + LFS weights from the official repo -------------------------------------
git lfs install --skip-repo || true
if [ ! -d "${BT_SRC}/.git" ]; then
  echo "==> Cloning evs-broadcast/BroadTrack (with LFS weights)"
  git clone https://github.com/evs-broadcast/BroadTrack.git "${BT_SRC}"
fi
( cd "${BT_SRC}" && git lfs pull )

# Sanity: LFS pointer files are ~134 bytes; the real keypoint model is ~266 MB.
for m in nbjw_keypoint_model.pt tvcalib_model.pt; do
  sz=$(stat -c%s "${BT_SRC}/models/${m}")
  if [ "${sz}" -lt 1000000 ]; then
    echo "ERROR: ${m} is only ${sz} bytes - git-lfs did not fetch the real weights." >&2
    echo "       Install git-lfs and re-run: cd ${BT_SRC} && git lfs pull" >&2
    exit 1
  fi
  cp -f "${BT_SRC}/models/${m}" "${BT_MODELS}/${m}"
  echo "==> models/${m}: ${sz} bytes OK"
done

# --- 2b. toolchain-portability patches to the EVS sources ---------------------------------
# Upstream builds inside cuda:12.4.1-cudnn-devel-ubuntu22.04, i.e. CUDA 12.4 and
# Ceres 2.0. On newer hosts (Ubuntu 24.04 => Ceres 2.2; CUDA >= 12.9) two upstream
# assumptions break:
#   * CUDA 12.9 dropped the standalone libnvToolsExt, so FindCUDAToolkit no longer
#     defines CUDA::nvToolsExt - but libtorch's Caffe2Config still references it,
#     and cmake dies in the generate step on a dangling target.
#   * Ceres 2.2 removed LocalParameterization: SubsetParameterization and
#     Problem::SetParameterization no longer exist, so CameraTracker.cpp fails to
#     compile. SubsetManifold/SetManifold are the sanctioned drop-in replacements
#     and are numerically identical (see Ceres version history).
# Both patches are idempotent, and the Ceres one is version-guarded so it stays a
# no-op on the upstream toolchain. NVTX is profiling instrumentation only, so the
# empty interface target has no functional effect.
echo "==> Applying toolchain-portability patches to the EVS sources"
python3 - "${BT_SRC}" <<'PYEOF'
import sys
from pathlib import Path

src = Path(sys.argv[1])

cml = src / "CMakeLists.txt"
s = cml.read_text()
if "CUDA::nvToolsExt INTERFACE IMPORTED" in s:
    print("    CMakeLists.txt: already patched")
else:
    stub = ("if(NOT TARGET CUDA::nvToolsExt)\n"
            "  add_library(CUDA::nvToolsExt INTERFACE IMPORTED GLOBAL)\n"
            "endif()\n\n")
    i = s.index("find_package(Torch")
    cml.write_text(s[:i] + stub + s[i:])
    print("    CMakeLists.txt: added CUDA::nvToolsExt stub (CUDA >= 12.9)")

ct = src / "CameraTracker.cpp"
s = ct.read_text()
old = ("        auto *fixedk1 = new ceres::SubsetParameterization(8, {7});\n"
       "        problem.SetParameterization(&cameraData[0], fixedk1);")
new = ("#if CERES_VERSION_MAJOR > 2 || (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 2)\n"
       "        auto *fixedk1 = new ceres::SubsetManifold(8, {7});\n"
       "        problem.SetManifold(&cameraData[0], fixedk1);\n"
       "#else\n"
       "        auto *fixedk1 = new ceres::SubsetParameterization(8, {7});\n"
       "        problem.SetParameterization(&cameraData[0], fixedk1);\n"
       "#endif")
if "SubsetManifold" in s:
    print("    CameraTracker.cpp: already patched")
elif old in s:
    ct.write_text(s.replace(old, new))
    print("    CameraTracker.cpp: migrated to the Ceres Manifold API (Ceres >= 2.2)")
else:
    sys.stderr.write("    WARNING: CameraTracker.cpp pattern not found - "
                     "upstream may have changed; build may fail on Ceres >= 2.2\n")
PYEOF

# --- 3. libtorch -------------------------------------------------------------------------
if [ ! -d "${LIBTORCH_DIR}" ]; then
  echo "==> Downloading libtorch (cu124)"
  wget -q --show-progress -O "${BT_ROOT}/libtorch.zip" "${LIBTORCH_URL}"
  unzip -q "${BT_ROOT}/libtorch.zip" -d "${BT_ROOT}"
  rm -f "${BT_ROOT}/libtorch.zip"
fi

# --- 3b. libnvrtc plain-name symlink ------------------------------------------------------
# libtorch ships NVRTC under a build-hashed name (libnvrtc-<hash>.so.12). At RUNTIME its
# lazy loader dlopen()s two candidates: "libnvrtc.so.12" and "libnvrtc-XXXXXXXX.so.12" for
# one SPECIFIC hash it was built against. When neither resolves it throws
#     RuntimeError: Error in dlopen for library libnvrtc.so.12and libnvrtc-XXXXXXXX.so.12
# and the binary aborts (SIGABRT, rc=-6). On a CUDA 13 host /usr/local/cuda/lib64 provides
# only libnvrtc.so.13, which is the wrong ABI version, so the system copy cannot satisfy it
# either. Creating the plain-name symlink next to the bundled library fixes it -- the file
# IS the correct CUDA 12 NVRTC, only its name differs.
#
# This surfaces LATE and only sometimes: NVRTC is loaded lazily, the first time TorchScript
# has to JIT-compile a kernel. BroadTrack hits that on "Reinit needed", so a sequence that
# never reinitialises calibrates all the way through and hides the problem entirely.
#
# TWO files need the treatment. Fixing only the first moves the error one level deeper:
#     nvrtc: error: failed to open libnvrtc-builtins.so.12.4
# NVRTC loads its builtins companion by EXACT version. libtorch ships it unversioned as
# libnvrtc-builtins.so, and a CUDA 13 host provides only libnvrtc-builtins.so.13.0.
for _nvrtc in "${LIBTORCH_DIR}"/lib/libnvrtc-*.so.12; do
  if [ -e "${_nvrtc}" ] && [ ! -e "${LIBTORCH_DIR}/lib/libnvrtc.so.12" ]; then
    ln -sf "$(basename "${_nvrtc}")" "${LIBTORCH_DIR}/lib/libnvrtc.so.12"
    echo "==> linked libnvrtc.so.12 -> $(basename "${_nvrtc}")"
  fi
done
if [ -e "${LIBTORCH_DIR}/lib/libnvrtc-builtins.so" ] \
   && [ ! -e "${LIBTORCH_DIR}/lib/libnvrtc-builtins.so.12.4" ]; then
  ln -sf libnvrtc-builtins.so "${LIBTORCH_DIR}/lib/libnvrtc-builtins.so.12.4"
  echo "==> linked libnvrtc-builtins.so.12.4 -> libnvrtc-builtins.so"
fi

# --- 4. build ----------------------------------------------------------------------------
echo "==> Building broadtrack"
mkdir -p "${BT_SRC}/build"
( cd "${BT_SRC}/build" \
  && cmake .. -DCMAKE_BUILD_TYPE=Release \
              -DCMAKE_PREFIX_PATH="${LIBTORCH_DIR}" \
              -DTorch_DIR="${LIBTORCH_DIR}/share/cmake/Torch" \
              -DCMAKE_INSTALL_PREFIX="${BT_ROOT}" \
  && make -j "${JOBS}" broadtrack \
  && make install )

if [ ! -x "${BT_BIN}/broadtrack" ]; then
  # `make install` only installs the `broadtrack` target; copy as a fallback.
  cp -f "${BT_SRC}/build/broadtrack" "${BT_BIN}/broadtrack"
fi
chmod +x "${BT_BIN}/broadtrack"

# --- 5. smoke test -----------------------------------------------------------------------
export LD_LIBRARY_PATH="${LIBTORCH_DIR}/lib:${LD_LIBRARY_PATH:-}"
echo "==> Smoke test: broadtrack --help"
"${BT_BIN}/broadtrack" --help || true

cat <<EOF

==================================================================
 Done.
   binary : ${BT_BIN}/broadtrack
   models : ${BT_MODELS}
   torch  : ${LIBTORCH_DIR}/lib   (module prepends this to LD_LIBRARY_PATH)

 The module config (configs/modules/calibration/broadtrack.yaml) expects these
 paths under \${model_dir}/broadtrack. If you used a custom BT_ROOT, override:
   tracklab -cn soccernet \\
     modules.calibration.cfg.binary=${BT_BIN}/broadtrack \\
     modules.calibration.cfg.keypoint_model=${BT_MODELS}/nbjw_keypoint_model.pt \\
     modules.calibration.cfg.line_model=${BT_MODELS}/tvcalib_model.pt \\
     modules.calibration.cfg.libtorch_lib=${LIBTORCH_DIR}/lib

 Reminder: EVS licence is noncommercial research, no redistribution.
 Do not publish the binary, the weights, or generated calibration JSONs.
==================================================================
EOF
