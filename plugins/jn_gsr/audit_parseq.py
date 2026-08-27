#!/usr/bin/env python3
"""audit_parseq.py -- does the PARSeq swap actually take effect, and what did
it change?

Runs INSIDE .venv_jn (needs torch + strhub + the package's own modules), which
is why it lives here and not in scripts/.

A swap that silently does not happen is the failure mode this file exists to
rule out. Two caches keyed on `os.path.basename(--parseq)` alone would have
reused stale log-likelihoods; jn_cache/ keyed on the manifest alone would have
served the previous numbers for both the checkpoint and the legibility change.
Those key gaps are closed (run_eval.eval_fingerprint, jn_gsr_api._manifest_hash)
-- this script is the independent check that they were closed correctly.

    D1  provenance      staged sha256 + size, ASSERTED (stage_weights.py
                        reports and continues by design; this arm refuses)
    D2  load + shapes    strhub load, class, hparams, head width, tokenizer,
                        parameter count, resolved digit/EOS columns
    D3  reader           build_parseq_batch_reader over real ROI crops:
                        shape (2, 11), finite, and the share of probability
                        mass the 11 kept columns hold
    D4  ROI invariance   old vs new cache: roi / pl / ds / q / n_roi identical
    D5  effect           old vs new cache: frames whose img differs, tracklets
                        whose merged label differs
    D6  metrics          both arms x legibility 0.9 / 0.8 / 0.72 / 0.7

Typical use, after staging both checkpoints and running the worker once per
arm into SEPARATE cache directories (assert_fingerprint refuses to merge
caches whose fingerprints differ, which is the point):

    $PY audit_parseq.py --stages d1,d2
    $PY audit_parseq.py --stages d3 --root data/gsr --split test --max-crops 400
    $PY audit_parseq.py --stages d4,d5,d6 \\
        --cache-new work/eval_cache/new --parseq-new models/parseq_gsr_ft_s1.ckpt \\
        --cache-old work/eval_cache/old --parseq-old models/koshkina_sn_parseq.ckpt

D6 merges only -- it reads no weights from either checkpoint, they are there to
key their own caches. On a box where a cache outlived its 286/382 MB file,
declare the identity instead:  --parseq-sha-old <sha256 from the cache's
provenance.parseq.sha256>. A wrong declaration cannot pass; it just fails the
fingerprint like a wrong file would.

Exit status is 0 only if every selected stage passed. NOTHING here prints a
metric it did not compute: a stage that cannot run says so and fails.
"""
import argparse
import hashlib
import json
import os
import sys
import time

# Before ANY import that can reach matplotlib (see setup_env.py header).
os.environ["MPLBACKEND"] = "Agg"

# ---------------------------------------------------------------- expected --
# The published upstream artifact. Kept here as well as in stage_weights.py on
# purpose: this is the arm that ASSERTS, and it must not be silenced by an
# edit to the fetcher.
EXPECT_SHA256 = "22d936444e09b0358b5b7339c2971ab5e792fee9d53dd30a98917abcd3ee1887"
EXPECT_BYTES_MIN = 250_000_000        # truncation / LFS-stub floor
EXPECT_BYTES_MAX = 320_000_000

# Read from the checkpoint itself (archive/data.pkl), not from a model card:
# 175 fp32 tensors, 23,832,671 parameters, head [95, 384],
# text_embed.embedding.weight [97, 384], pos_queries [1, 26, 384].
EXPECT = {
    "model_class": "PARSeq",
    "img_size": [32, 128],
    "max_label_length": 25,
    "head_out_features": 95,
    "tokenizer_len": 97,
    "eos_id": 0,
    "pos_queries_shape": (1, 26, 384),
    "n_parameters": 23832671,
    # ACCEPTANCE for THIS checkpoint, not a permanent runtime invariant: it
    # holds because charset_train starts "0123456789" and Tokenizer puts EOS
    # first. run_eval.build_models keeps the general guard (ten distinct digit
    # columns, EOS not among them) and records the resolved indices.
    "digit_idx": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}

# The SUPERSEDED reference row: old checkpoint, legibility 0.9, GSR-2024 test,
# GT boxes, whole split, det 0.52, maxconf. D6 checks the harness against it
# BEFORE any other cell is read; if this does not reproduce, the staging or the
# harness differs from the run that produced it and nothing else is comparable.
REF_OLD_AT_090 = {"trk_acc": 0.8420, "numbered": 0.8501,
                  "minus1_f1": 0.83, "roi_kept": 0.572}

BAR = "=" * 74


def _hdr(t):
    print(f"\n{BAR}\n{t}\n{BAR}", flush=True)


def _sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


class Fail(Exception):
    """A stage assertion that did not hold."""


def _check(cond, what, got=None, want=None):
    if cond:
        print(f"  PASS  {what}" + (f"  ({got})" if got is not None else ""))
        return
    raise Fail(f"{what}: got {got!r}, expected {want!r}")


# ------------------------------------------------------------------- D1 -----
def d1_provenance(a, report):
    _hdr("D1  provenance -- the staged file is the published artifact")
    if not os.path.exists(a.ckpt):
        raise Fail(f"{a.ckpt} does not exist -- run stage_weights.py first")
    size = os.path.getsize(a.ckpt)
    t0 = time.time()
    digest = _sha256(a.ckpt)
    print(f"  path   {a.ckpt}")
    print(f"  size   {size} B ({size / 1e6:.1f} MB)")
    print(f"  sha256 {digest}  ({time.time() - t0:.1f}s)")
    report["d1"] = {"path": a.ckpt, "bytes": size, "sha256": digest}

    # strhub routes on the PATH string, so this is a provenance question too.
    lowered = os.path.abspath(a.ckpt).lower()
    clashes = [s for s in ("abinet", "crnn", "trba", "trbc", "vitstr")
               if s in lowered]
    _check("parseq" in lowered,
           "resolved path contains 'parseq'", lowered)
    _check(not clashes,
           "resolved path contains no competing model substring",
           clashes or "none", "none")
    _check(EXPECT_BYTES_MIN <= size <= EXPECT_BYTES_MAX,
           "size within the expected band", size,
           f"{EXPECT_BYTES_MIN}..{EXPECT_BYTES_MAX}")
    _check(digest == EXPECT_SHA256, "sha256 matches the published artifact",
           digest, EXPECT_SHA256)
    report["d1"]["ok"] = True


# ------------------------------------------------------------------- D2 -----
def d2_load(a, report):
    _hdr("D2  load + shape acceptance -- strhub builds the model we expect")
    import torch  # noqa: F401  (imported for the side effect of being present)
    from strhub.models.utils import load_from_checkpoint

    t0 = time.time()
    model = load_from_checkpoint(a.ckpt).eval()
    print(f"  loaded in {time.time() - t0:.1f}s")

    from common import _charset_indices
    digit_idx, eos = _charset_indices(model.tokenizer)
    n_params = sum(p.numel() for p in model.parameters())
    got = {
        "model_class": type(model).__name__,
        "img_size": list(model.hparams.img_size),
        "max_label_length": int(model.hparams.max_label_length),
        "head_out_features": int(model.head.out_features),
        "tokenizer_len": int(len(model.tokenizer)),
        "eos_id": int(model.tokenizer.eos_id),
        "pos_queries_shape": tuple(model.pos_queries.shape),
        "n_parameters": int(n_params),
        "digit_idx": [int(i) for i in digit_idx],
    }
    report["d2"] = {k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in got.items()}
    report["d2"]["resolved_eos"] = int(eos)

    for k, want in EXPECT.items():
        _check(got[k] == want, f"{k}", got[k], want)
    _check(int(eos) == EXPECT["eos_id"],
           "_charset_indices resolves EOS to the tokenizer's eos_id",
           int(eos), EXPECT["eos_id"])
    # The 11 columns the reader slices must all be inside the head's width.
    cols = list(digit_idx) + [int(eos)]
    _check(max(cols) < got["head_out_features"],
           "every sliced column is within the head width",
           f"max col {max(cols)} < {got['head_out_features']}")
    _check(len(set(digit_idx)) == 10 and eos not in digit_idx,
           "build_models' permanent guard holds "
           "(10 distinct digit columns, EOS not among them)")
    report["d2"]["ok"] = True
    return model


# ------------------------------------------------------------------- D3 -----
def _roi_crops(a, limit):
    """Real ROI crops off the staged split, through the PRODUCTION path
    (DetectorGate.roi_crop_scored), not a re-derivation of it."""
    from PIL import Image

    import stage_utils as U
    from common import subsample
    from dbnet_infer import DBNetDetector, DetectorGate
    from gsr_adapter import build_pool

    seq_dirs = U.list_sequences(a.root, a.split)
    if not seq_dirs:
        raise Fail(f"no sequences under {a.root}/{a.split} -- "
                   f"run stage_data.py --splits {a.split} first")
    det = DBNetDetector(weights=a.dbnet_ckpt, config=a.dbnet_cfg)
    gate = DetectorGate(det, thr=a.det_floor, pad_frac=a.roi_pad,
                        cache_max=a.det_cache)

    crops = []
    for sd in seq_dirs:
        pool = build_pool([os.path.join(sd, "Labels-GameState.json")])
        for tid in sorted(pool):
            for fr in subsample(pool[tid]["frames"], a.stride):
                try:
                    im = Image.open(fr["frame_path"]).convert("RGB")
                except OSError:
                    continue
                roi, _src, _ds = gate.roi_crop_scored(fr["frame_path"],
                                                      fr["xywh"], img=im)
                if roi is not None:
                    crops.append(roi)
                if len(crops) >= limit:
                    return crops
    return crops


def d3_reader(a, report, model=None):
    _hdr("D3  reader diagnostic -- shape, finiteness, and retained mass")
    import numpy as np

    if model is None:
        from strhub.models.utils import load_from_checkpoint
        model = load_from_checkpoint(a.ckpt).eval()
    try:
        import torch
        if torch.cuda.is_available():
            model = model.to("cuda:0")
    except ImportError:
        pass

    from common import build_parseq_batch_reader
    read_many = build_parseq_batch_reader(model)

    crops = _roi_crops(a, a.max_crops)
    _check(len(crops) > 0, "collected ROI crops", len(crops), ">0")
    print(f"  {len(crops)} ROI crops from {a.root}/{a.split}")

    logls = read_many(crops)
    _check(len(logls) == len(crops), "one reading per crop",
           len(logls), len(crops))

    arr = np.asarray([[t, u] for t, u in logls], dtype=np.float64)
    _check(arr.shape == (len(crops), 2, 11), "output shape is (N, 2, 11)",
           tuple(arr.shape), (len(crops), 2, 11))
    _check(bool(np.isfinite(arr).all()), "every log-likelihood is finite")

    # exp(logl).sum() per position: the share of probability mass the 11 kept
    # columns hold. With a 94-character charset this is <= 1 by construction;
    # how far below 1 is how much the recogniser spends on letters and
    # punctuation on football crops. Reported, not asserted -- it is a
    # measurement, and a threshold on it would be invented.
    mass = np.exp(arr).sum(axis=2)                      # [N, 2]
    stats = {
        "n_crops": int(len(crops)),
        "retained_mass_mean_pos0": float(mass[:, 0].mean()),
        "retained_mass_mean_pos1": float(mass[:, 1].mean()),
        "retained_mass_median_pos0": float(np.median(mass[:, 0])),
        "retained_mass_median_pos1": float(np.median(mass[:, 1])),
        "retained_mass_p05_pos0": float(np.percentile(mass[:, 0], 5)),
        "retained_mass_p05_pos1": float(np.percentile(mass[:, 1], 5)),
        "retained_mass_max": float(mass.max()),
    }
    report["d3"] = stats
    print(f"\n  retained probability mass over the 11 kept columns")
    print(f"    position 0 (tens):  mean {stats['retained_mass_mean_pos0']:.4f}"
          f"   median {stats['retained_mass_median_pos0']:.4f}"
          f"   p05 {stats['retained_mass_p05_pos0']:.4f}")
    print(f"    position 1 (units): mean {stats['retained_mass_mean_pos1']:.4f}"
          f"   median {stats['retained_mass_median_pos1']:.4f}"
          f"   p05 {stats['retained_mass_p05_pos1']:.4f}")
    # The +1e-9 floor inside the reader can lift a column marginally; allow it.
    _check(stats["retained_mass_max"] <= 1.0 + 1e-6,
           "retained mass never exceeds 1", stats["retained_mass_max"], "<=1")
    report["d3"]["ok"] = True


# --------------------------------------------------- cache loading (D4-D6) --
def _load_cache(path):
    if not os.path.isdir(path):
        raise Fail(f"cache directory {path} does not exist")
    files = sorted(f for f in os.listdir(path)
                   if f.endswith(".json") and f != "results.json")
    if not files:
        raise Fail(f"no sequence caches in {path}")
    out = {}
    for f in files:
        with open(os.path.join(path, f)) as fh:
            rec = json.load(fh)
        out[rec["sequence"]] = rec
    return out


def _frames_by_key(cache):
    """{(sequence, tid, frame_basename): record} -- the comparison key."""
    d = {}
    for seq, rec in cache.items():
        for tid, t in rec["tracklets"].items():
            for fr in t["frames"]:
                d[(seq, tid, fr["f"])] = fr
    return d


# ------------------------------------------------------------------- D4 -----
def d4_invariance(a, report):
    _hdr("D4  ROI invariance -- only the recogniser changed")
    new, old = _load_cache(a.cache_new), _load_cache(a.cache_old)
    _check(set(new) == set(old), "both arms cover the same sequences",
           f"{len(new)} vs {len(old)}")

    for seq in sorted(new):
        _check(new[seq]["n_roi"] == old[seq]["n_roi"],
               f"n_roi matches for {seq}",
               f"{new[seq]['n_roi']} vs {old[seq]['n_roi']}")

    fn, fo = _frames_by_key(new), _frames_by_key(old)
    _check(set(fn) == set(fo), "both arms cover the same (seq, tid, frame)",
           f"{len(fn)} vs {len(fo)}")

    diffs = {"roi": [], "pl": [], "ds": [], "q": []}
    for k in fn:
        A, B = fn[k], fo[k]
        for field in ("roi", "pl", "ds", "q"):
            if A.get(field) != B.get(field):
                diffs[field].append(k)
    report["d4"] = {"n_frames": len(fn),
                    **{f"n_{f}_differs": len(v) for f, v in diffs.items()},
                    "examples": {f: [list(x) for x in v[:5]]
                                 for f, v in diffs.items() if v}}
    for field, v in diffs.items():
        _check(not v, f"'{field}' is identical across both arms",
               f"{len(v)} of {len(fn)} differ", "0")
    print("\n  The detector and the legibility classifier are untouched by this\n"
          "  change and the worker is ungated, so this had to hold. If it had\n"
          "  not, every downstream comparison would be contaminated.")
    report["d4"]["ok"] = True


# ------------------------------------------------------------------- D5 -----
def d5_effect(a, report):
    _hdr("D5  effect -- the new weights are actually in the path")
    import evaluate_jn as E
    from legibility import frame_verdicts

    new, old = _load_cache(a.cache_new), _load_cache(a.cache_old)
    fn, fo = _frames_by_key(new), _frames_by_key(old)
    shared = [k for k in fn if k in fo and fn[k].get("roi")]
    n_img = sum(1 for k in shared if fn[k].get("img") != fo[k].get("img"))

    def merged(cache):
        lab = {}
        for seq in cache.values():
            for tid, t in seq["tracklets"].items():
                frames = t["frames"]
                keep = frame_verdicts([f["pl"] for f in frames],
                                      a.legibility_thr) if frames else []
                tens, units = [], []
                for f, k in zip(frames, keep):
                    if k and f.get("roi") and f.get("ds", 0.0) > a.det_thr:
                        tens.append(f["t"])
                        units.append(f["u"])
                lab[(seq["sequence"], tid)] = \
                    E.consolidate_tracklet_maxconf(tens, units)[0]
        return lab

    ln, lo = merged(new), merged(old)
    keys = set(ln) & set(lo)
    n_lab = sum(1 for k in keys if ln[k] != lo[k])

    report["d5"] = {"n_roi_frames": len(shared), "n_img_differs": n_img,
                    "n_tracklets": len(keys), "n_label_differs": n_lab,
                    "legibility_thr": a.legibility_thr,
                    "det_thr": a.det_thr}
    print(f"  ROI frames compared     {len(shared)}")
    print(f"  per-frame img differs   {n_img}"
          f"  ({100 * n_img / max(len(shared), 1):.2f}%)")
    print(f"  tracklets compared      {len(keys)}"
          f"  (merged at legibility > {a.legibility_thr}, det > {a.det_thr})")
    print(f"  merged label differs    {n_lab}"
          f"  ({100 * n_lab / max(len(keys), 1):.2f}%)")
    _check(n_img > 0, "the per-frame readings moved", n_img, ">0")
    _check(n_lab > 0, "at least one tracklet label moved", n_lab, ">0")
    print("\n  Zero difference would NOT have been a pass: with a different\n"
          "  fine-tune and a 94-character charset, exact agreement on every\n"
          "  tracklet is not plausible -- it would mean the new weights were\n"
          "  never loaded, or a cache was reused.")
    report["d5"]["ok"] = True


# ------------------------------------------------------------------- D6 -----
def d6_metrics(a, report):
    _hdr("D6  metrics -- both arms x four legibility thresholds")
    import run_eval

    arms = [("old", a.cache_old, a.parseq_old, a.parseq_sha_old),
            ("new", a.cache_new, a.parseq_new, a.parseq_sha_new)]
    thresholds = [float(t) for t in a.sweep.split(",")]
    cells, work = {}, os.path.join(a.out_dir, "_d6")
    os.makedirs(work, exist_ok=True)

    for arm, cache, parseq, sha in arms:
        if not (parseq or sha):
            raise Fail(f"--parseq-{arm} or --parseq-sha-{arm} is required: "
                       f"eval_fingerprint keys on the checkpoint's CONTENT, so "
                       f"each arm must be merged against the checkpoint its "
                       f"cache was built with (or a declaration of it) or "
                       f"assert_fingerprint will (correctly) refuse.")
        # The old arm's 382 MB checkpoint exists only to key its own cache --
        # nothing in a merge reads its weights. --parseq-sha-old lets a
        # merge-only box skip keeping it on disk.
        declared = ["--parseq-sha", sha] if sha else []
        for thr in thresholds:
            out = os.path.join(work, f"{arm}_{thr}.json")
            rc = run_eval.main([
                "--root", a.root, "--split", a.split, "--merge",
                "--cache", cache, "--parseq", parseq,
                "--ckpt", a.dbnet_ckpt, "--dbnet-cfg", a.dbnet_cfg,
                "--det-floor", str(a.det_floor), "--roi-pad", str(a.roi_pad),
                "--player-pad", str(a.player_pad), "--stride", str(a.stride),
                "--legibility-size", str(a.legibility_size),
                "--legibility-thr", str(thr), "--det-thr", str(a.det_thr),
                "--rule", "maxconf", "--out", out] + declared)
            if rc:
                raise Fail(f"merge failed for arm={arm} thr={thr} (rc={rc})")
            with open(out) as f:
                cells[(arm, thr)] = json.load(f)

    print(f"\n  arm   leg    trk_acc   numbered   -1 F1   roi kept")
    for arm, _c, _p in arms:
        for thr in thresholds:
            r = cells[(arm, thr)]
            print(f"  {arm:4s} {thr:5.2f}  {100 * r['trk_acc']:7.2f}%  "
                  f"{100 * r['numbered']:8.2f}%   {r['minus1_f1']:5.2f}   "
                  f"{100 * r['roi_kept']:6.1f}%")
    report["d6"] = {f"{arm}@{thr}": {k: cells[(arm, thr)][k] for k in
                                     ("trk_acc", "numbered", "minus1_f1",
                                      "roi_kept")}
                    for arm, _c, _p in arms for thr in thresholds}

    # The apparatus is validated BEFORE anything else in this table is read.
    print()
    if ("old", 0.9) not in cells:
        raise Fail("0.9 must be in --sweep: the superseded reference row is "
                   "what validates the harness, and nothing else in this "
                   "table can be trusted until it reproduces.")
    ref = cells[("old", 0.9)]
    for k, want in REF_OLD_AT_090.items():
        _check(abs(round(ref[k], 4) - want) < 5e-5,
               f"old checkpoint @ 0.9 reproduces {k}", round(ref[k], 4), want)
    print("\n  Apparatus validated. Read the rest of the table now, not before.\n"
          "  A gain in trk_acc alongside a fall in minus1_f1 is not a win: a\n"
          "  lower gate admits more frames, so fewer tracklets end with no\n"
          "  surviving ROI -- which works against the failure the legibility\n"
          "  gate was introduced to fix. Both numbers, then decide.")
    report["d6"]["ok"] = True


# ------------------------------------------------------------------- CLI ----
STAGES = {"d1": d1_provenance, "d2": d2_load, "d3": d3_reader,
          "d4": d4_invariance, "d5": d5_effect, "d6": d6_metrics}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="d1,d2",
                    help="comma-separated subset of " + ",".join(STAGES))
    ap.add_argument("--ckpt", default="models/parseq_gsr_ft_s1.ckpt")
    ap.add_argument("--out-dir", default="work/audit_parseq")
    ap.add_argument("--report", default=None,
                    help="default <out-dir>/audit_parseq.json")
    # D3 (and D6's merge arguments, which must match the worker's)
    ap.add_argument("--root", default="data/gsr")
    ap.add_argument("--split", default="test")
    ap.add_argument("--dbnet-ckpt", default="models/best_icdar_hmean_epoch_10.pth")
    ap.add_argument("--dbnet-cfg", default="mmocr_cfg/dbnetpp_infer.py")
    ap.add_argument("--det-floor", type=float, default=0.2)
    ap.add_argument("--det-thr", type=float, default=0.52)
    ap.add_argument("--roi-pad", type=float, default=0.12)
    ap.add_argument("--player-pad", type=float, default=0.18)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--legibility-size", type=int, default=256)
    ap.add_argument("--legibility-thr", type=float, default=0.72,
                    help="the gate D5 merges at (D6 sweeps its own)")
    ap.add_argument("--det-cache", type=int, default=512)
    ap.add_argument("--max-crops", type=int, default=400,
                    help="D3: how many real ROI crops to read")
    # D4/D5/D6
    ap.add_argument("--cache-new", default="work/eval_cache/new")
    ap.add_argument("--cache-old", default="work/eval_cache/old")
    ap.add_argument("--parseq-new", default="models/parseq_gsr_ft_s1.ckpt")
    ap.add_argument("--parseq-old", default="models/koshkina_sn_parseq.ckpt")
    ap.add_argument("--parseq-sha-new", default="",
                    help="D6: declare the new arm's checkpoint sha256 instead "
                         "of hashing the file (run_eval --parseq-sha). A wrong "
                         "value cannot pass -- the fingerprint will not match "
                         "the cache and the merge refuses.")
    ap.add_argument("--parseq-sha-old", default="",
                    help="D6: same for the old arm. A merge reads no weights, "
                         "so this lets a merge-only box drop the superseded "
                         "382 MB checkpoint and still reproduce its row.")
    ap.add_argument("--sweep", default="0.9,0.8,0.72,0.7",
                    help="D6 legibility thresholds; 0.9 is REQUIRED (it is "
                         "what validates the harness against the superseded "
                         "reference row)")
    a = ap.parse_args(argv)

    wanted = [s.strip().lower() for s in a.stages.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in STAGES]
    if unknown:
        ap.error(f"unknown stage(s) {unknown}; choose from {list(STAGES)}")

    os.makedirs(a.out_dir, exist_ok=True)
    report, failures, model = {"stages": wanted}, [], None
    for s in wanted:
        try:
            if s == "d2":
                model = d2_load(a, report)
            elif s == "d3":
                d3_reader(a, report, model=model)
            else:
                STAGES[s](a, report)
        except Fail as e:
            print(f"  FAIL  {e}", flush=True)
            report.setdefault(s, {})["ok"] = False
            report[s]["error"] = str(e)
            failures.append(s)
        except Exception as e:                      # a stage that cannot run
            print(f"  ERROR {s}: {type(e).__name__}: {e}", flush=True)
            report.setdefault(s, {})["ok"] = False
            report[s]["error"] = f"{type(e).__name__}: {e}"
            failures.append(s)

    path = a.report or os.path.join(a.out_dir, "audit_parseq.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    _hdr("summary")
    for s in wanted:
        print(f"  {s}  {'PASS' if report.get(s, {}).get('ok') else 'FAIL'}")
    print(f"\n  report -> {path}")
    if failures:
        print(f"\n  {len(failures)} stage(s) FAILED: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
