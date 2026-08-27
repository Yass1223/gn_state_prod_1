# stage_utils.py -- shared plumbing for the Kaggle stage scripts
# (stage_data / stage_manifest / stage_crops / stage_test).
#
# The Kaggle path runs every pipeline stage as a `!{PYBIN} stage_*.py` subprocess
# (Kaggle's notebook editor cannot switch the Jupyter kernel to a custom venv, so
# nothing may import torch/mmocr IN the kernel). That inverts the Lightning
# notebook's design: state that used to live in the kernel (manifest, N_TRAIN,
# per-tracklet log-likelihoods) must live on DISK between stages -- which is also
# exactly what makes a run resumable across Kaggle sessions.
#
# Conventions enforced here:
#   * every stage sets the JN_* env vars from its CLI args BEFORE importing
#     roi_dbnet / dbnet_infer (both read the env at import time), reproducing
#     notebook section 0's "single source of truth" contract;
#   * every cached artifact embeds a config FINGERPRINT; consumers refuse to
#     merge caches produced under different thresholds/strides/checkpoints,
#     so a resumed session cannot silently mix incompatible partial results;
#   * JSON writes are atomic (tmp + rename), so a session killed mid-write
#     never leaves a truncated cache that a resume would trip over.

import argparse
import hashlib
import json
import os
import sys
import zlib
from pathlib import Path


# --------------------------------------------------------------- config env --
def add_common_args(ap: argparse.ArgumentParser):
    """The section-0 knobs every GPU stage needs, with section-0 defaults."""
    ap.add_argument("--det-thr", type=float, default=0.52,
                    help="DBNet++ score: ROI selection AND the legibility verdict")
    ap.add_argument("--roi-pad", type=float, default=0.12,
                    help="ROI margin per side, in units of glyph height")
    ap.add_argument("--player-pad", type=float, default=0.18,
                    help="player-box pad before detection (train == test)")
    ap.add_argument("--stride", type=int, default=5, help="keep every Nth frame")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="best_icdar_hmean_epoch_10.pth",
                    help="DBNet++ checkpoint")
    ap.add_argument("--maskrcnn", choices=("on", "off"), default="on",
                    help="multi-player frame filter")
    ap.add_argument("--maskrcnn-cover", type=float, default=0.10)
    ap.add_argument("--maskrcnn-drop-none", action="store_true",
                    help="require exactly 1 person (default keeps count<=1)")
    ap.add_argument("--maskrcnn-cache", type=int, default=800,
                    help="frames of Mask R-CNN masks kept hot; >= frames/sequence "
                         "(750 for GSR) makes each unique frame run ONCE per "
                         "sequence instead of thrashing the default 64-frame LRU")
    ap.add_argument("--shard", type=int, default=0, help="this worker's index")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="total workers (one per GPU); sequences are dealt "
                         "round-robin so 750-frame clips balance")


def apply_config_env(a):
    """Notebook section 0's contract, subprocess edition: env FIRST, then module
    attributes, so roi_dbnet/dbnet_infer (which read env at import) and any
    grandchild subprocess both see the same constants."""
    os.environ["JN_DET_THR"] = str(a.det_thr)
    os.environ["JN_PAD_FRAC"] = str(a.roi_pad)
    os.environ["JN_PLAYER_PAD"] = str(a.player_pad)
    os.environ["JN_DBNET_CKPT"] = str(a.ckpt)
    os.environ["MPLBACKEND"] = "Agg"   # FORCE, not setdefault: Kaggle presets
# MPLBACKEND=module://matplotlib_inline.backend_inline for its kernel, and that
# module is absent from this venv -- matplotlib then raises at IMPORT time.
# setdefault leaves the poisoned value in place; only assignment fixes it.
    import roi_dbnet, dbnet_infer          # noqa: E401  (import AFTER env)
    roi_dbnet.DET_THR, roi_dbnet.PAD_FRAC = a.det_thr, a.roi_pad
    dbnet_infer.PLAYER_PAD = a.player_pad
    return roi_dbnet, dbnet_infer


def fingerprint(a, extra=()):
    """Stable hash of everything that changes a stage's OUTPUT. Two caches with
    different fingerprints must never be merged."""
    parts = [f"det_thr={a.det_thr}", f"roi_pad={a.roi_pad}",
             f"player_pad={a.player_pad}", f"stride={a.stride}",
             f"seed={a.seed}",
             f"maskrcnn={a.maskrcnn}", f"cover={a.maskrcnn_cover}",
             f"drop_none={a.maskrcnn_drop_none}",
             f"ckpt={os.path.basename(str(a.ckpt))}"]
    parts += [str(x) for x in extra]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ file IO --
def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------- sequences --
def list_sequences(root, split):
    """Sorted sequence dirs for a split -- the parents of Labels-GameState.json,
    exactly as notebook section 3's seq_dirs() finds them."""
    import glob
    return sorted({os.path.dirname(p) for p in glob.glob(
        f"{root}/{split}/**/Labels-GameState.json", recursive=True)})


def shard_of(seqs, shard, num_shards):
    """Round-robin deal over the SORTED sequence list. Any worker owns
    seqs[shard::num_shards]; the union over workers is exactly seqs.
    GSR clips are uniform (750 frames each) so round-robin balances."""
    if not (0 <= shard < num_shards):
        sys.exit(f"--shard {shard} out of range for --num-shards {num_shards}")
    return list(seqs)[shard::num_shards]


def seq_rng_seed(seed, seq_name):
    """Deterministic per-SEQUENCE rng seed: independent of shard topology, so a
    2-GPU run, a 1-GPU run, and a resumed run all draw identical streams for a
    given sequence. (The OFF arm never consumes the rng, so its crops are
    byte-identical to the Lightning notebook's regardless.)"""
    return [int(seed), zlib.crc32(seq_name.encode()) & 0xFFFFFFFF]


def assert_fingerprint(cache_obj, want, what):
    got = cache_obj.get("fingerprint")
    if got != want:
        sys.exit(f"[stale-cache] {what}: fingerprint {got} != current {want}.\n"
                 f"  The cached file was produced under different thresholds/"
                 f"stride/checkpoint. Delete it (or the whole cache dir) and re-run.")


if __name__ == "__main__":
    # pure-logic self-tests
    class A:
        det_thr, roi_pad, player_pad, stride, seed = 0.52, 0.12, 0.18, 5, 0
        maskrcnn, maskrcnn_cover, maskrcnn_drop_none = "on", 0.10, False
        ckpt = "/x/best_icdar_hmean_epoch_10.pth"
    f1 = fingerprint(A())
    A.det_thr = 0.53
    assert fingerprint(A()) != f1, "fingerprint must react to threshold changes"
    A.det_thr = 0.52
    assert fingerprint(A()) == f1, "fingerprint must be stable"
    assert fingerprint(A(), extra=["arm=on"]) != f1

    seqs = [f"S{i:02d}" for i in range(7)]
    a = shard_of(seqs, 0, 2); b = shard_of(seqs, 1, 2)
    assert sorted(a + b) == seqs and not set(a) & set(b)
    assert shard_of(seqs, 0, 1) == seqs
    assert seq_rng_seed(0, "SNGS-021") == seq_rng_seed(0, "SNGS-021")
    assert seq_rng_seed(0, "SNGS-021") != seq_rng_seed(0, "SNGS-022")

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.json")
        atomic_json(p, {"fingerprint": "abc", "v": 1})
        assert load_json(p)["v"] == 1
        try:
            assert_fingerprint(load_json(p), "def", "test")
            raise AssertionError("should exit")
        except SystemExit:
            pass
    print("stage_utils: self-tests passed (fingerprint, shard partition, "
          "per-seq rng, atomic json, stale-cache gate)")
