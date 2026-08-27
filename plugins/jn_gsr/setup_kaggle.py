#!/usr/bin/env python3
"""setup_kaggle.py -- build the pipeline venv ON KAGGLE, reusing setup_env.

Why a separate entry point but NOT a separate pin list: the environment recipe
(python 3.10 + torch 2.0.1+cu118 + mmcv 2.0.1 + mmocr 1.0.1 + PL 1.9.5 + the
numpy<2 constraints file) lives ONCE, in setup_env.py. Duplicating it here
would drift. This wrapper only relocates the venv:

  * Kaggle's own kernel python cannot be swapped for a custom interpreter, so
    every pipeline step runs as `!{PYBIN} ...` -- the venv is a tool, not a
    kernel, exactly the "subprocess contract" the repo already verifies.
  * The venv (~7 GB with CUDA torch + mmcv) goes on the EPHEMERAL scratch disk
    (default /kaggle/tmp/venv_jn), NOT /kaggle/working: it would consume a third
    of the ~19.5 GB persisted budget and bloat every saved notebook version.
    Rebuilding it each session costs ~10 minutes; that is the right trade.

    python setup_kaggle.py                     # build + verify
    python setup_kaggle.py --venv /kaggle/tmp/venv_jn --check   # verify only

Prints PYBIN=<interpreter path> as its last line for the notebook to capture.
"""
import argparse
import os
import subprocess
import sys

# Before spawning setup_env.py. Kaggle's kernel presets MPLBACKEND to an inline
# backend whose module is absent from the venv; matplotlib raises at IMPORT
# time and torchmetrics imports matplotlib. See setup_env.py's header.
os.environ["MPLBACKEND"] = "Agg"

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv", default="/kaggle/tmp/venv_jn",
                    help="venv location (scratch disk; NOT /kaggle/working)")
    ap.add_argument("--check", action="store_true", help="verify only, no build")
    a = ap.parse_args()

    env = dict(os.environ, JN_VENV=a.venv, MPLBACKEND="Agg")
    setup = os.path.join(HERE, "setup_env.py")
    cmd = [sys.executable, setup] + (["--check"] if a.check else [])
    print(f"[setup_kaggle] venv -> {a.venv}  (delegating to setup_env.py)")
    r = subprocess.run(cmd, env=env, cwd=HERE)
    if r.returncode != 0:
        sys.exit(r.returncode)
    pybin = os.path.join(a.venv, "bin", "python")
    if not os.path.exists(pybin):
        sys.exit(f"[setup_kaggle] build reported success but {pybin} is missing")
    print(f"PYBIN={pybin}")


if __name__ == "__main__":
    main()
