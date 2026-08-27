#!/usr/bin/env python3
"""dual_gpu.py -- run one stage-worker per GPU, concurrently, and fail loudly.

The stage scripts pin themselves to cuda:0, so each worker is isolated with
CUDA_VISIBLE_DEVICES=<gpu> (inside the process, that GPU IS cuda:0) and told
which slice of the sequence list it owns via `--shard i --num-shards n`.
Output is streamed line-by-line with a [gpuN] prefix; the exit code is nonzero
if ANY worker fails, so a notebook `!` line stops the run instead of letting a
half-built cache masquerade as done.

    python dual_gpu.py --gpus 0,1 -- $PY stage_manifest.py --root data/gsr ...
    python dual_gpu.py --gpus 0   -- $PY stage_test.py ...     # degenerates to 1 worker

Stdlib only on purpose: it runs under whatever python launches it (the Kaggle
kernel's), not the pipeline venv.
"""
import argparse
import os
import subprocess
import sys
import threading


def stream(proc, tag, counter):
    """Prefix and forward worker output -- DROPPING whitespace-only lines.
    Measured on a real Kaggle run: the mmocr inferencer emits one bare newline
    per call, which was 99.3% of all captured lines (20,808 of 20,949) and
    would bloat a full run's saved notebook by ~half a million junk lines."""
    for line in iter(proc.stdout.readline, ""):
        if not line.strip():
            counter[0] += 1
            continue
        sys.stdout.write(f"[{tag}] {line}")
        sys.stdout.flush()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1", help="comma list, e.g. 0,1")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="-- followed by the worker command")
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        sys.exit("usage: dual_gpu.py --gpus 0,1 -- <python> <stage>.py <args...>")
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip() != ""]
    n = len(gpus)
    procs, threads = [], []
    dropped = [0]
    for i, g in enumerate(gpus):
        # PYTHONUNBUFFERED: children write to a pipe, so python would otherwise
        # block-buffer and progress lines would arrive in stale 4 KB bursts
        # MPLBACKEND: Kaggle presets an inline backend whose module is absent
        # from the pipeline venv, and matplotlib raises at IMPORT time. The
        # worker sanitises it too, but a worker that dies before reaching that
        # line dies with a confusing traceback, so force it at the source.
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, PYTHONUNBUFFERED="1",
                   MPLBACKEND="Agg")
        full = cmd + ["--shard", str(i), "--num-shards", str(n)]
        p = subprocess.Popen(full, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        t = threading.Thread(target=stream, args=(p, f"gpu{g}", dropped),
                             daemon=True)
        t.start()
        procs.append(p); threads.append(t)
    codes = [p.wait() for p in procs]
    for t in threads:
        t.join(timeout=5)
    bad = [(gpus[i], c) for i, c in enumerate(codes) if c != 0]
    if bad:
        sys.exit(f"[dual_gpu] worker(s) FAILED: {bad} -- fix and re-run "
                 f"(finished sequences are cached; only the rest recompute)")
    note = f" ({dropped[0]} blank output lines suppressed)" if dropped[0] else ""
    print(f"[dual_gpu] all {n} worker(s) finished OK{note}")


if __name__ == "__main__":
    main()
