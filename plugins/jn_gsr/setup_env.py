#!/usr/bin/env python3
"""setup_env.py -- ONE Python 3.10 venv for the whole DBNet++ pipeline,
on a Lightning AI Studio.

WHY ONE ENV (this is the finding that makes the integration cheap):
The PARSeq venv and the MMOCR stack agree on every shared pin --

    python 3.10 | torch 2.0.1+cu118 | torchvision 0.15.2 | setuptools<81
    numpy 1.25.2 (satisfies MMOCR's numpy<2)

-- so mmcv 2.0.1 / mmengine 0.10.4 / mmdet 3.1.0 / mmocr 1.0.1 install ALONGSIDE
PARSeq in a single interpreter. No cross-process bridge, no temp-file handoff for
the in-memory crops the augmentation loop produces. Dropping ViTPose also removes
`transformers` from the crop path entirely.

The venv lives INSIDE the project folder (/teamspace/studios/this_studio/...), so
it survives Studio resets -- unlike anything under /home/zeus.

    python setup_env.py            # build or repair
    python setup_env.py --check    # verify only, no install
"""
import argparse
import os
import subprocess
import sys

# THE FIRST THING, before any child process is spawned. Kaggle's kernel presets
#   MPLBACKEND=module://matplotlib_inline.backend_inline
# and that module is NOT in this venv, so matplotlib raises ValueError at IMPORT
# time -- and torchmetrics (via pytorch-lightning) imports matplotlib. Every
# subprocess below inherits this environ, so setting it here fixes the whole
# tree. FORCE, not setdefault: the poisoned value has to be REPLACED.
# roi_dbnet/stage_utils/stage_weights/crop_classifier already do this for the
# runtime path; setup_env was the one entry point that did not.
os.environ["MPLBACKEND"] = "Agg"

VENV = os.environ.get("JN_VENV", ".venv_jn")   # overridable: Kaggle builds the
# venv on the ephemeral scratch disk (setup_kaggle.py sets JN_VENV) so its ~7 GB
# never counts against /kaggle/working's ~19.5 GB persisted budget.
PY = os.path.join(VENV, "bin", "python")


def sh(cmd, **kw):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True, **kw)


CONSTRAINTS = "constraints_jn.txt"


def build():
    sh(f"{sys.executable} -m pip install -q -U uv")
    sh("uv python install 3.10")
    if not os.path.exists(PY):
        sh(f"uv venv --seed --python 3.10 {VENV}")

    # ---- pip CONSTRAINTS, applied to EVERY install below --------------------
    # Without this, numpy is pinned early and then silently upgraded to 2.x by a
    # later package (scipy / pandas / opencv-python / ultralytics all pull
    # numpy>=2 on current releases). torch 2.0.1 and mmcv 2.0.1 are compiled
    # against the NumPy 1.x C ABI, so the upgrade does not raise -- it breaks
    # torch's numpy bridge at import time, several cells later, with a message
    # that does not name the cause. A constraints file is the only mechanism pip
    # honours across independent install invocations.
    with open(CONSTRAINTS, "w") as f:
        f.write("numpy<2\n"
                "opencv-python<5\n"       # opencv-python 5.x requires numpy>=2
                "opencv-python-headless<5\n")
    C = f"-c {CONSTRAINTS}"

    # uv venvs ship without setuptools; pkg_resources (needed by torch 2.0.1's
    # cpp_extension AND openmim) was removed in setuptools 82 -- pin below it.
    sh(f"{PY} -m pip install -q {C} 'setuptools<81' wheel")
    sh(f"{PY} -c 'import pkg_resources'")

    # numpy FIRST, so every later resolution sees it already satisfied
    sh(f"{PY} -m pip install -q {C} numpy==1.25.2")

    sh(f"{PY} -m pip install -q {C} --index-url https://download.pytorch.org/whl/cu118 "
       f"torch==2.0.1 torchvision==0.15.2")
    # mmcv prebuilt CUDA-op wheel matched to that exact torch/CUDA pair:
    sh(f"{PY} -m pip install -q {C} mmcv==2.0.1 "
       f"-f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html")
    sh(f"{PY} -m pip install -q {C} mmengine==0.10.4 mmdet==3.1.0 mmocr==1.0.1 openmim==0.3.9")
    # PARSeq side (same pins as the original parseq venv) + shared deps
    sh(f"{PY} -m pip install -q {C} pytorch-lightning==1.9.5 torchmetrics==1.0.3 "
       f"timm==0.9.5 nltk==3.8.1 pillow==10.0.0 pyyaml==6.0.1 lmdb==1.4.1")
    sh(f"{PY} -m pip install -q {C} yapf==0.40.1 imgaug==0.4.0 scipy h5py "
       f"opencv-python-headless pandas tqdm fire")
    # pipeline runtime deps:
    #   fire            -> str/parseq/tools/create_lmdb_dataset.py
    #   SoccerNet       -> the GSR-2024 download in stage_data.py
    #   huggingface_hub -> all four checkpoints (fetch_weights / stage_weights /
    #                      crop_classifier / legibility). A DIRECT dependency:
    #                      relying on a transitive one is how it disappears.
    #   gdown           -> the legibility Drive fallback
    # ultralytics is NOT installed. This pipeline has no YOLO stage -- boxes come
    # from Labels-GameState.json -- so it was pure dead weight, and it is what
    # produced the `filelock>=3.16.1` resolver conflicts in the build log.
    sh(f"{PY} -m pip install -q {C} SoccerNet huggingface_hub gdown")
    # Final re-assert: if anything above still dragged numpy forward, put it back
    # and fail loudly rather than leaving a subtly broken environment.
    sh(f"{PY} -m pip install -q {C} numpy==1.25.2")
    # NO Jupyter-kernel registration: Kaggle cannot switch a notebook's kernel,
    # so this venv is used only as a subprocess interpreter (PYBIN).
    if os.path.isdir("str/parseq"):
        sh(f"{PY} -m pip install -q {C} -e str/parseq --no-deps")
    else:
        print("[warn] str/parseq not found -- PARSeq (strhub) NOT installed")


def check():
    code = r'''
import os
os.environ["MPLBACKEND"] = "Agg"      # before ANY import; see module header
import sys
print("python", sys.version.split()[0])
assert sys.version_info[:2] == (3, 10), "expected the 3.10 venv"
import pkg_resources
import numpy
# HARD GATE. torch 2.0.1 / mmcv 2.0.1 are built against the NumPy 1.x C ABI; a
# 2.x numpy does not raise on install, it breaks torch's numpy bridge later.
assert numpy.__version__.startswith("1."), (
    f"numpy {numpy.__version__} in the venv, but torch 2.0.1 and mmcv 2.0.1 need "
    f"numpy<2. A later package upgraded it. Re-run setup_env.py (it now "
    f"applies a pip constraints file), or force it: "
    f"VENV/bin/pip install 'numpy==1.25.2'")
import torch, torchvision
print("torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| cuda", torch.version.cuda, "| numpy", numpy.__version__)
# prove the numpy bridge actually works, rather than trusting the version string
_t = torch.zeros(3); assert _t.numpy().shape == (3,), "torch<->numpy bridge broken"
print("torch<->numpy bridge: OK")
print("cuda available:", torch.cuda.is_available(),
      "| devices:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print("  GPU", i, torch.cuda.get_device_name(i))
import mmcv, mmengine, mmdet, mmocr
print("mmcv", mmcv.__version__, "| mmengine", mmengine.__version__,
      "| mmdet", mmdet.__version__, "| mmocr", mmocr.__version__)
from mmcv.ops import ModulatedDeformConv2d          # DCNv2 -- the backbone needs it
print("mmcv DCNv2 CUDA op: OK")
from mmocr.apis import TextDetInferencer            # DBNet++ (detection)
print("TextDetInferencer import: OK")
from mmocr.apis import TextRecInferencer            # SATRN (second recogniser)
print("TextRecInferencer import: OK")
import pytorch_lightning, timm
print("PL", pytorch_lightning.__version__, "| timm", timm.__version__)
try:
    # NOT `import strhub` -- strhub/ ships an __init__.py, so if the models/ and
    # data/ subpackages are absent (e.g. swallowed by an unanchored `models/`
    # .gitignore rule) the bare import still SUCCEEDS as a namespace package and
    # reports OK, while the jersey worker dies at runtime with
    # "No module named 'strhub.models'". Import the symbol actually used by
    # run_eval.build_models() so the check fails here instead.
    from strhub.models.utils import load_from_checkpoint
    import strhub; print("strhub (PARSeq): OK")
except Exception as e:
    print("strhub (PARSeq): BROKEN --", e)
    print("   -> str/parseq/strhub must contain models/ and data/ (vendored v1.1.0).")
    print("      Do NOT substitute upstream baudm/parseq main (v1.2.0): the")
    print("      SoccerNet PARSeq checkpoint is loaded via Lightning's")
    print("      load_from_checkpoint and is tied to the 1.1.0 model signature.")
import fire; print("fire (lmdb tool): OK")
try:
    import SoccerNet; print("SoccerNet (downloader): OK")
except Exception as e:
    print("SoccerNet: MISSING --", e)
import huggingface_hub
print("huggingface_hub:", huggingface_hub.__version__)
# The exact call shapes every fetcher uses -- verified here, not at download time.
from huggingface_hub import hf_hub_download, snapshot_download
import inspect
for _f in (hf_hub_download, snapshot_download):
    _p = list(inspect.signature(_f).parameters)
    assert _p[0] == "repo_id", (_f.__name__, _p[:3])
    assert "force_download" in _p, _f.__name__
print("huggingface_hub API shape: OK")
# Import the pipeline's OWN modules in the venv. An import-level break (a bad
# signature, a missing helper, a renamed constant) then surfaces HERE, in setup,
# instead of forty minutes into the GPU pass.
import crop_classifier, legibility, evaluate_jn, gsr_adapter
import roi_dbnet, dbnet_infer, common, stage_utils, run_eval
import fuse_jn, mmocr_reader, jn_recognizer, predict_tracklets
assert jn_recognizer.RULE == predict_tracklets.RULE == "vote_pool", (
    jn_recognizer.RULE, predict_tracklets.RULE)
assert jn_recognizer.SATRN_CKPT == predict_tracklets.DEFAULT_SATRN_CKPT
assert jn_recognizer.PARSEQ_CKPT == predict_tracklets.DEFAULT_PARSEQ_CKPT
print("pipeline modules import: OK (rule", jn_recognizer.RULE + ")")

ck = os.environ.get("JN_DBNET_CKPT", "best_icdar_hmean_epoch_10.pth")
print("checkpoint", ck, "present:", os.path.exists(ck),
      f"({os.path.getsize(ck)/1e6:.1f} MB)" if os.path.exists(ck) else "")
print("ENV OK")
'''
    # Explicit env, not just the inherited one: makes the guarantee local
    # to the call rather than a side effect of module import order.
    subprocess.run([PY, "-c", code], check=True,
                   env=dict(os.environ, MPLBACKEND="Agg"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only, no install")
    a = ap.parse_args()
    if not a.check:
        build()
    if not os.path.exists(PY):
        sys.exit(f"venv missing at {VENV} -- run without --check first")
    check()
    print(f"\n[setup] interpreter: {os.path.abspath(PY)}")
    print("[setup] run every pipeline step with THIS python (the notebook's PYBIN).")
    print("[setup] venv ready. Every stage runs through it as a "
          "subprocess; the Kaggle kernel is never switched.")
