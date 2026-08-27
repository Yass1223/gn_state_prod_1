#!/usr/bin/env python3
"""run_eval.py -- jersey-number inference on SoccerNet GSR, FINAL configuration.

Stages (all weights frozen): GT player boxes -> ResNet-34 legibility gate
(p_legible cached per frame) -> DBNet++ text detection on the padded player
crop -> PARSeq digit recognition (11-way per-position log-likelihoods cached).

The worker is UNGATED: every subsampled frame is scored and cached; the single
final gate (legibility > 0.72, strictly greater, legibility.frame_verdicts) and
the maxconf consolidation are applied at --merge. A tracklet with no surviving
ROI frame is -1. Metrics reported: trk_acc, numbered, -1 F1, roi kept.
(The gate value is --legibility-thr; it is merge-time and NOT in the cache
fingerprint, so a sweep costs one re-merge, not one worker pass per cell.)

This configuration was selected on the v1/v2 whole-test-split ablation (the
ResNet-18 single/multi filter measured <= no-op and is removed; vote beat the
sum rule at every legibility threshold). The experimentation harness lives in
the v2 package; this build computes ONE configuration and nothing else.
"""
import argparse
import json
import os
import sys
import time

# Before any import that can reach matplotlib. Kaggle presets MPLBACKEND to an
# inline backend whose module is absent from this venv. FORCE, not setdefault.
os.environ["MPLBACKEND"] = "Agg"

import stage_utils as U

TOK = "0123456789E"
SCHEMA = 3   # final: single config, no filter stage, no per-frame "ds"


# --------------------------------------------------------------------- util --
def decode_frame(a, b):
    """Per-crop image-level label: argmax per position, 'E' terminates."""
    import numpy as np
    ti, ui = int(np.argmax(a)), int(np.argmax(b))
    lab = TOK[ti] + TOK[ui]
    lab = lab[:lab.index("E")] if "E" in lab else lab
    return lab if lab else "-1"


def _sha256(path, chunk=1 << 22):
    import hashlib
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _parseq_digest(a):
    """The PARSeq checkpoint's identity for the fingerprint.

    CONTENT, not just the name. A basename-only key lets a different checkpoint
    staged under the same filename reuse the cached log-likelihoods, which is
    precisely how a swap silently does not take effect. Hashing costs ~1 s on a
    286 MB file, once per process.

    --parseq-sha DECLARES that identity instead of reading it, for merge-only
    sessions where the cache survived but the 286 MB file did not (a resumed
    Kaggle session is the case this exists for). It is accepted at --merge ONLY,
    and it cannot fabricate agreement: a declared value that is not the one the
    cache was built under produces a different fingerprint and assert_fingerprint
    refuses, exactly as a wrong file would. What it deliberately CANNOT do is
    mint a new cache under an identity nobody verified -- the worker always
    hashes the file it is about to load."""
    declared = str(getattr(a, "parseq_sha", "") or "").strip().lower()
    is_merge = bool(getattr(a, "merge", False))

    if declared and not is_merge:
        sys.exit("[run] --parseq-sha is a MERGE-ONLY override. A worker pass "
                 "writes the cache, so it must hash the checkpoint it actually "
                 "loads rather than take your word for it. Drop --parseq-sha, "
                 "or add --merge.")
    if declared:
        if len(declared) < 16 or any(c not in "0123456789abcdef" for c in declared):
            sys.exit(f"[run] --parseq-sha {declared!r} is not a hex sha256 "
                     f"(need >= 16 hex chars; the full 64-char digest is what "
                     f"the cache's provenance record and stage_weights.py "
                     f"print).")
        print(f"[run] --parseq-sha: using the DECLARED checkpoint identity "
              f"{declared[:16]} (file not read). A wrong value cannot pass -- "
              f"the fingerprint simply will not match the cache.")
        return declared

    digest = _sha256(a.parseq)
    if digest is None:
        print(f"[run] WARNING: --parseq {a.parseq!r} is not readable. The "
              f"checkpoint's sha256 is part of the cache fingerprint, so this "
              f"run's fingerprint CANNOT match one produced with the "
              f"checkpoint present -- a --merge here will report a stale "
              f"cache. Stage the checkpoint, pass the --parseq the cache was "
              f"built with, or declare it with --parseq-sha.")
    return digest


def eval_fingerprint(a):
    """What INFERENCE depends on. --thr / --legibility-thr are excluded on
    purpose: they are applied at merge, so changing them must not invalidate the
    cache. `maskrcnn` is pinned to the constant 'ungated' because the worker
    never gates -- otherwise stage_utils.fingerprint would make the cache look
    configuration-dependent when it is not.

    NOTE: the PARSeq checkpoint's CONTENT is part of this, so a merge normally
    needs the checkpoint file to be READABLE, not merely the cache to exist.
    --parseq-sha lifts that for merge-only sessions; see _parseq_digest."""
    class _Shim:
        # v2: inference detects at the LOW floor and caches the top score, so
        # --det-thr joined --thr/--legibility-thr as a merge-time parameter.
        # The attribute keeps its name for stage_utils.fingerprint's field list.
        det_thr, roi_pad, player_pad = a.det_floor, a.roi_pad, a.player_pad
        stride, seed, ckpt = a.stride, a.seed, a.ckpt
        maskrcnn, maskrcnn_cover, maskrcnn_drop_none = "ungated", 0.10, False
    digest = _parseq_digest(a)
    extra = [
        f"schema={SCHEMA}",
        f"parseq={os.path.basename(str(a.parseq))}",
        f"parseq_sha={str(digest)[:16]}",
        f"leg_size={a.legibility_size}",
    ]
    # Appended ONLY for non-GSR datasets: keeps every pre-existing GSR cache
    # fingerprint byte-identical, while making a GSR cache and an SN-JNR cache
    # refuse to merge into one number if they ever share a --cache dir.
    if getattr(a, "dataset", "gsr") != "gsr":
        extra.append(f"dataset={a.dataset}")
    return U.fingerprint(_Shim(), extra=extra)


# ------------------------------------------------------------------- models --
def build_models(a):
    import torch

    import crop_classifier as CC
    # Before any torch.load. Checkpoints pickled on a numpy>=2 machine name
    # `numpy._core`, absent under the numpy 1.x this venv pins for the
    # torch/mmcv C ABI. Applies to every checkpoint, not just the classifier's.
    if CC.numpy2_pickle_compat():
        print("[run] numpy<2: mapping numpy._core -> numpy.core for pickles")

    import legibility as LG
    from dbnet_infer import DBNetDetector, DetectorGate

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[run] device={device}")

    t0 = time.time()
    det = DBNetDetector(weights=a.ckpt, config=a.dbnet_cfg)
    gate = DetectorGate(det, thr=a.det_floor, pad_frac=a.roi_pad,
                        cache_max=a.det_cache)
    print(f"[run] DBNet++ loaded in {time.time() - t0:.1f}s  "
          f"thr={gate.thr} pad={gate.pad_frac}")

    t0 = time.time()
    leg = LG.LegibilityClassifier(a.legibility_weights or None, device=device,
                                  threshold=a.legibility_thr,
                                  size=a.legibility_size)
    print(f"[run] legibility loaded in {time.time() - t0:.1f}s  "
          f"arch={leg.arch} n_out={leg.n_out} size={a.legibility_size} "
          f"thr={leg.threshold}")
    if leg.n_out != 1:
        raise RuntimeError(f"legibility head emits {leg.n_out} values; "
                           f"Koshkina's is a single sigmoid probability.")

    from strhub.models.utils import load_from_checkpoint
    t0 = time.time()
    model = load_from_checkpoint(a.parseq).eval().to(device)
    from common import (_charset_indices, build_parseq_batch_reader,
                        build_parseq_reader)
    read = build_parseq_reader(model)
    read_many = build_parseq_batch_reader(model)
    digit_idx, eos = _charset_indices(model.tokenizer)
    if len(set(digit_idx)) != 10 or eos in digit_idx:
        raise RuntimeError(f"PARSeq charset resolution failed: "
                           f"digits={digit_idx} eos={eos}")
    print(f"[run] PARSeq loaded in {time.time() - t0:.1f}s  "
          f"digits -> {digit_idx}, EOS -> {eos}, "
          f"img_size {list(model.hparams.img_size)}")

    return dict(device=device, det=det, gate=gate, leg=leg,
                model=model, read=read, read_many=read_many,
                provenance={
                    "dbnet": {"path": a.ckpt, "sha256": _sha256(a.ckpt)},
                    # digit_idx/eos are RESOLVED from the loaded tokenizer, so
                    # a checkpoint whose charset orders characters differently
                    # is visible in the record rather than only in the guard.
                    "parseq": {"path": a.parseq, "sha256": _sha256(a.parseq),
                               "model_class": type(model).__name__,
                               "img_size": list(model.hparams.img_size),
                               "digit_idx": [int(i) for i in digit_idx],
                               "eos_idx": int(eos),
                               "n_classes": int(model.head.out_features),
                               "tokenizer_len": int(len(model.tokenizer))},
                    "legibility": {"path": leg.path, "source": leg.source,
                                   "sha256": _sha256(leg.path),
                                   "arch": leg.arch, "n_out": leg.n_out,
                                   "size": a.legibility_size,
                                   "threshold": leg.threshold},
                })


# -------------------------------------------------------------------- worker --
def process_sequence(a, M, seq_dir):
    """One sequence -> the cache record. UNGATED: nothing is dropped here."""
    import numpy as np
    from PIL import Image

    from common import subsample
    from crop_classifier import crop_box

    seq = os.path.basename(seq_dir)
    from gsr_adapter import build_pool
    pool = build_pool([os.path.join(seq_dir, "Labels-GameState.json")])
    gate, read = M["gate"], M["read"]

    tracklets, lat = {}, {"leg": 0.0, "det": 0.0, "read": 0.0}
    n_frames = n_roi = n_degenerate = 0

    # v2: optional fp16 autocast around every model forward. Off by default so
    # results stay bit-comparable with fp32 caches; ~2x on T4 when enabled.
    import contextlib
    import torch
    amp = (lambda: torch.autocast("cuda", dtype=torch.float16)) \
        if (a.fp16 and torch.cuda.is_available()) else contextlib.nullcontext

    for tid in sorted(pool):
        t = pool[tid]
        frames = subsample(t["frames"], a.stride)
        if a.max_frames:
            frames = frames[:a.max_frames]
        if not frames:
            continue

        # --- ONE decode per (frame, box): classifiers AND detector share it --
        crops, idx = [], []
        rois, roi_idx, roi_ds, roi_q = [], [], [], []
        for i, fr in enumerate(frames):
            try:
                im = Image.open(fr["frame_path"]).convert("RGB")
            except Exception:
                continue
            c, _ = crop_box(im, fr["xywh"])
            if c is None:
                n_degenerate += 1
            else:
                crops.append(c)
                idx.append(i)
            # 3. DBNet++ on the 18%-padded player crop -> ROI + top score,
            #    detected at the LOW floor (a.det_floor); thresholding is merge
            t0 = time.time()
            with amp():
                roi, src, ds = gate.roi_crop_scored(fr["frame_path"],
                                                    fr["xywh"], img=im)
            lat["det"] += time.time() - t0
            if roi is not None:
                rois.append(roi)
                roi_idx.append(i)
                roi_ds.append(ds)
                # audit support: the WINNING quad, in padded-crop coords.
                # _detections is the gate's own memo -- this is a cache hit,
                # zero extra detector compute. Selection replicates
                # roi_from_detections exactly (>= floor, highest score wins).
                quads, scores, _c = gate._detections(fr["frame_path"],
                                                     fr["xywh"], img=im)
                keep = [(float(s), q) for s, q in zip(scores, quads)
                        if float(s) >= gate.thr]
                qtop = max(keep, key=lambda t: t[0])[1] if keep else None
                roi_q.append(None if qtop is None else
                             np.asarray(qtop, np.float64).reshape(-1, 2)[:4])

        # legibility. No crop -> 0.0, below any threshold -> illegible.
        p_leg = np.zeros(len(frames), np.float64)
        if crops:
            t0 = time.time()
            with amp():
                pl, _ = M["leg"].score(crops)
            lat["leg"] += time.time() - t0
            for j, i in enumerate(idx):
                p_leg[i] = float(pl[j])

        # 4. PARSeq, batched over the tracklet's ROIs -> per-position log-liks
        t0 = time.time()
        with amp():
            logls = M["read_many"](rois) if rois else []
        lat["read"] += time.time() - t0
        by_i = {i: (lg, ds, q)
                for i, lg, ds, q in zip(roi_idx, logls, roi_ds, roi_q)}

        per_frame = []
        for i, fr in enumerate(frames):
            n_frames += 1
            rec = {"f": os.path.basename(fr["frame_path"]),
                   "fp": fr["frame_path"],
                   "xywh": [round(float(v), 1) for v in fr["xywh"]],
                   "pl": round(float(p_leg[i]), 6)}
            if i not in by_i:
                rec["roi"] = False
                per_frame.append(rec)
                continue
            (tens, units), ds, qtop = by_i[i]
            n_roi += 1
            rec["roi"] = True
            # top-quad detector score at the low floor: makes the det gate a
            # merge-time parameter (any thr >= floor is exactly ds > thr)
            rec["ds"] = round(float(ds), 4)
            if qtop is not None:
                rec["q"] = [[round(float(x), 1), round(float(y), 1)]
                            for x, y in qtop]
            rec["t"] = [round(float(x), 4) for x in tens]
            rec["u"] = [round(float(x), 4) for x in units]
            rec["img"] = decode_frame(tens, units)
            per_frame.append(rec)

        tracklets[tid] = {"gt": str(t["number"]),
                          "n_all": len(t["frames"]),
                          "frames": per_frame}

    return {"schema": SCHEMA, "sequence": seq, "tracklets": tracklets,
            "n_frames": n_frames, "n_roi": n_roi,
            "n_degenerate_boxes": n_degenerate,
            "latency_s": {k: round(v, 2) for k, v in lat.items()},
            "detector_stats": dict(gate.stats)}


def work(a):
    from common import seed_everything
    seed_everything(a.seed)

    seq_dirs = U.list_sequences(a.root, a.split)
    stage_hint = "stage_data.py"
    if not seq_dirs:
        sys.exit(f"[run] no sequences under {a.root}/{a.split} -- "
                 f"run {stage_hint} --splits {a.split} first")
    mine = U.shard_of(seq_dirs, a.shard, a.num_shards)
    fp = eval_fingerprint(a)
    os.makedirs(a.cache, exist_ok=True)
    print(f"[run] shard {a.shard}/{a.num_shards}: {len(mine)} of "
          f"{len(seq_dirs)} sequences  fingerprint={fp}")

    todo = [s for s in mine
            if not os.path.exists(os.path.join(a.cache,
                                               os.path.basename(s) + ".json"))]
    print(f"[run] {len(mine) - len(todo)} already cached, {len(todo)} to do")
    if not todo:
        return 0

    M = build_models(a)
    for n, sd in enumerate(todo, 1):
        seq = os.path.basename(sd)
        t0 = time.time()
        rec = process_sequence(a, M, sd)
        rec["fingerprint"] = fp
        rec["provenance"] = M["provenance"]
        rec["seconds"] = round(time.time() - t0, 1)
        U.atomic_json(os.path.join(a.cache, seq + ".json"), rec)
        rate = rec["n_frames"] / max(rec["seconds"], 1e-9)
        print(f"[run] {n}/{len(todo)} {seq}: {len(rec['tracklets'])} tracklets, "
              f"{rec['n_frames']} frames, {rec['n_roi']} ROIs, "
              f"{rec['seconds']}s ({rate:.1f} frames/s)  "
              f"eta {(len(todo) - n) * rec['seconds'] / 60:.0f} min")
    return 0


# --------------------------------------------------------------------- merge --
# FROZEN PRODUCTION CONFIGURATION (chosen on the six-rule comparison run;
# see README):
#   filter: OFF
#   gate:   legibility > 0.72  AND  det > 0.52 (cached top-quad score 'ds')
#   rule:   maxconf -- score(L) = exp(best single-frame log-lik of L) x
#           (summed frame confidence of L); vote/wvote/sum remain selectable
#           via --rule for regression checks only
#   -1:     a tracklet with no surviving ROI frame
# Reported metrics: trk_acc (all tracklets, -1 as a class), numbered
# (numbered-only), -1 F1, roi kept. Nothing else is computed here.
def merge(a):
    import evaluate_jn as E
    from legibility import frame_verdicts

    fp = eval_fingerprint(a)
    if not os.path.isdir(a.cache):
        sys.exit(f"[merge] cache directory {a.cache} does not exist -- run the "
                 f"worker first (drop --merge), or pass the right --cache.")
    files = sorted(f for f in os.listdir(a.cache)
                   if f.endswith(".json") and f != "results.json")
    if not files:
        sys.exit(f"[merge] no sequence caches in {a.cache} -- run the worker "
                 f"first (drop --merge).")
    cache, prov = {}, None
    for f in files:
        rec = U.load_json(os.path.join(a.cache, f))
        U.assert_fingerprint(rec, fp, f)
        cache[rec["sequence"]] = rec
        prov = prov or rec.get("provenance")

    # Raw per-tracklet arrays, collected ONCE: the gate is merge-time, so any
    # threshold can be re-applied over the same cache.
    raw, gt = {}, {}
    for seq in cache.values():
        for tid, t in seq["tracklets"].items():
            gt[tid] = t["gt"]
            raw[tid] = t["frames"]

    def preds_at(lthr, dthr=0.0, rule=None):
        """Predictions at legibility > lthr AND (when dthr > 0) top-quad
        detector score ds > dthr. Both gates strictly greater. dthr must be
        >= the worker's --det-floor for the cached ds to be exact (the ROI is
        always the top quad); the notebook arms all satisfy this.
        `rule` (vote | wvote | sum) defaults to --rule; the gates are shared
        by every rule, so any rule re-merges over the same cache."""
        p, share = {}, {}
        for tid, frames in raw.items():
            keep = frame_verdicts([f["pl"] for f in frames],
                                  lthr) if frames else []
            tens, units = [], []
            for f, k in zip(frames, keep):
                if not (k and f.get("roi")):
                    continue
                if dthr > 0:
                    if "ds" not in f:
                        sys.exit(
                            "[merge] det gating requested but this cache "
                            "predates the 'ds' field (worker did not store "
                            "detector scores) -- re-run the worker to use "
                            "--det-thr, or gate at 0.")
                    if not f["ds"] > dthr:
                        continue
                tens.append(f["t"])
                units.append(f["u"])
            rl = rule or a.rule
            if rl == "wvote":
                p[tid], share[tid] = E.consolidate_tracklet_weighted_vote(
                    tens, units)
            elif rl == "slogconf":
                p[tid], share[tid] = E.consolidate_tracklet_slogconf(
                    tens, units)
            elif rl == "sumexp":
                p[tid], share[tid] = E.consolidate_tracklet_sumexp(
                    tens, units)
            elif rl == "maxconf":
                p[tid], share[tid] = E.consolidate_tracklet_maxconf(
                    tens, units)
            elif rl == "sum":
                lab, conf = E.consolidate_tracklet(tens, units)
                # avg_conf is -inf on empty tracklets; keep the sidecar
                # json finite with a sentinel
                p[tid], share[tid] = lab, (conf if conf > -998.0 else -999.0)
            else:
                p[tid], share[tid] = E.consolidate_tracklet_vote(tens, units)
        return p, share

    def det_kept_frac(dthr):
        """Fraction of ROI frames passing the det gate (before legibility) --
        shows how much each arm actually cuts."""
        roi = [f for fr in raw.values() for f in fr if f.get("roi")]
        if not roi or dthr <= 0:
            return 1.0
        return sum(f.get("ds", 0.0) > dthr for f in roi) / len(roi)

    # ---- predict-only: EvalAI submission, no metrics -------------------
    # For splits with hidden labels (SN-JNR challenge). Scoring against the
    # all-'-1' placeholder gt the adapter emits there would print meaningless
    # numbers, so this branch writes ONLY the submission dict
    # {player_id: int} (-1 == not visible; EvalAI challenge 1952 format) plus
    # a provenance sidecar. Coercion: valid jerseys are 1..99, so a predicted
    # "0"/"00" -- reachable because PARSeq's charset starts at '0' -- maps to
    # -1, with the affected count printed and the tids listed in the sidecar.
    if a.predict_only:
        pred, share = preds_at(a.legibility_thr, a.det_thr)
        sub, coerced = {}, []
        for tid, lab in pred.items():
            v = int(lab)
            if v == 0:
                coerced.append(tid)
                v = -1
            sub[tid] = v
        U.atomic_json(a.out, sub)
        meta = {
            "config": {"dataset": a.dataset, "filter": "off",
                       "legibility_thr": a.legibility_thr,
                       "rule": a.rule, "stride": a.stride,
                       "det_floor": a.det_floor,
                       "det_thr": a.det_thr},
            "n_tracklets": len(sub),
            "n_not_visible": sum(v == -1 for v in sub.values()),
            "zero_coerced_to_minus1": coerced,
            "vote_share": {t: round(float(s), 4) for t, s in share.items()},
            "fingerprint": fp,
            "provenance": prov,
        }
        meta_path = (a.out[:-5] if a.out.endswith(".json") else a.out) \
            + ".meta.json"
        U.atomic_json(meta_path, meta)
        nv = meta["n_not_visible"]
        print(f"\n[merge] predict-only: {len(sub)} tracklets, {nv} "
              f"not-visible ({100 * nv / max(len(sub), 1):.1f}%), "
              f"{len(coerced)} zero-coerced to -1  "
              f"(thr {a.legibility_thr}, {a.rule})")
        print(f"[merge] submission -> {a.out}")
        print(f"[merge] provenance -> {meta_path}")
        return

    pred, share = preds_at(a.legibility_thr, a.det_thr)
    m = E.cell_metrics([], [], pred, gt)

    out = {
        "config": {"dataset": a.dataset, "filter": "off",
                   "legibility_thr": a.legibility_thr,
                   "rule": a.rule, "stride": a.stride,
                   "det_floor": a.det_floor,
                   "det_thr": a.det_thr},
        "n_tracklets": len(gt),
        "trk_acc": round(m["tracklet_accuracy"], 4),
        "numbered": round(m["tracklet_accuracy_numbered"], 4),
        "minus1_f1": round(m["minus1_f1"], 4),
        "roi_kept": round(det_kept_frac(a.det_thr), 4),
        "fingerprint": fp,
        "provenance": prov,
    }


    print(f"\n[merge] {len(cache)} sequences, {len(gt)} tracklets  "
          f"(config: filter off, legibility > {a.legibility_thr}, {a.rule})\n")
    print(f"  trk_acc    {100 * out['trk_acc']:6.2f}%")
    print(f"  numbered   {100 * out['numbered']:6.2f}%")
    print(f"  -1 F1      {out['minus1_f1']:6.2f}")
    print(f"  roi kept   {100 * out['roi_kept']:6.1f}%   (det > {a.det_thr})")
    if a.out:
        U.atomic_json(a.out, out)
        print(f"\n[merge] wrote {a.out}")
    return 0


# ---------------------------------------------------------------------- CLI --
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    U.add_common_args(ap)
    ap.add_argument("--root", default="data/gsr")
    ap.add_argument("--split", default="test")
    ap.add_argument("--dataset", choices=("gsr", "snjnr"), default="gsr",
                    help="gsr: frames + GT boxes (Labels-GameState.json). "
                         "snjnr: SN-JNR pre-cropped player tracklets; one "
                         "player folder per 'sequence', full-image box, "
                         "--player-pad inert. NOT comparable across datasets.")
    # --det-thr is ALREADY declared by U.add_common_args (worker side).
    # At merge it is the frozen det gate: keep ROI frames with cached
    # top-quad score ds > this. 0.52 is the production value. Must be >=
    # --det-floor for ds to be exact. Merge-time (not in the fingerprint).
    ap.set_defaults(det_thr=0.52)
    ap.add_argument("--predict-only", action="store_true",
                    help="merge writes an EvalAI submission dict "
                         "{player_id: int, -1 = not visible} plus a "
                         ".meta.json sidecar, and computes NO metrics -- for "
                         "splits with hidden labels (SN-JNR challenge)")
    ap.add_argument("--rule",
                    choices=("maxconf", "vote", "wvote", "sum",
                             "slogconf", "sumexp"),
                    default="maxconf",
                    help="merge-time consolidation rule. PRODUCTION DEFAULT "
                         "maxconf: score(L) = exp(best single-frame log-lik "
                         "of L) x summed frame confidence of L. The others "
                         "are kept selectable for regression checks. "
                         "Merge-time like --legibility-thr (not in the "
                         "cache fingerprint).")
    ap.add_argument("--cache", default="work/eval_cache")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--out", default=None, help="results path (merge)")
    ap.add_argument("--parseq", default="models/parseq_gsr_ft_s1.ckpt")
    ap.add_argument("--parseq-sha", default="",
                    help="MERGE ONLY: declare the PARSeq checkpoint's sha256 "
                         "instead of hashing the file, for merge-only sessions "
                         "where the cache outlived the 286 MB checkpoint. Take "
                         "the value from the cache's provenance record "
                         "(provenance.parseq.sha256) or from stage_weights.py. "
                         "A wrong value cannot pass: the fingerprint will not "
                         "match and the merge refuses.")
    ap.add_argument("--dbnet-cfg", default="mmocr_cfg/dbnetpp_infer.py")
    ap.add_argument("--det-floor", type=float, default=0.2,
                    help="v2: worker-side detection floor. The ROI is cut and "
                         "PARSeq is read for the TOP quad >= floor, its score "
                         "cached as 'ds'; --det-thr then gates at merge like "
                         "--thr/--legibility-thr. Part of the fingerprint.")
    ap.add_argument("--fp16", dest="fp16", action="store_true", default=True,
                    help="autocast fp16 around all model forwards (default on; "
                         "reproduced the fp32 ablation bit-identically on T4)")
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--det-cache", type=int, default=512,
                    help="DBNet++ detections kept hot per worker")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="cap frames per tracklet (smoke tests only)")
    # --- legibility (the ONE gate of the final configuration) ------------
    ap.add_argument("--legibility-weights", default="")
    ap.add_argument("--legibility-thr", type=float, default=0.72,
                    help="final config: strictly-greater cut on p_legible. "
                         "MERGE-TIME (deliberately outside the fingerprint), "
                         "so a threshold sweep re-merges one worker pass.")
    ap.add_argument("--legibility-size", type=int, default=256,
                    help="256 per legibility.py; a wrong value shifts every "
                         "probability without crashing")
    a = ap.parse_args(argv)

    inert = [n for n, v, d in (("--maskrcnn", a.maskrcnn, "on"),
                               ("--maskrcnn-cover", a.maskrcnn_cover, 0.10),
                               ("--maskrcnn-cache", a.maskrcnn_cache, 800),
                               ("--maskrcnn-drop-none", a.maskrcnn_drop_none,
                                False)) if v != d]
    if inert:
        print(f"[run] NOTE: {', '.join(inert)} are inert here -- the worker "
              f"always runs ungated and gating happens at merge.")

    U.apply_config_env(a)
    return merge(a) if a.merge else work(a)


if __name__ == "__main__":
    sys.exit(main())
