# jn_recognizer.py -- the production jersey-number module.
#
# The single integration surface for a GSR-style pipeline: one class, one
# call per tracklet. The operating point is:
#
#     legibility > 0.72       (strictly greater; CONSTRUCTOR ARGUMENT
#                              `legibility_thr`, surfaced all the way up to
#                              jn_gsr.yaml -- see the note below)
#     DBNet++ top quad > 0.52 (strictly greater; SN-JNR test argmax; measured
#                              neutral-to-slightly-positive on both datasets)
#     two recognisers on the SAME surviving ROI crops:
#         A  PARSeq  parseq_gsr_ft_s1.ckpt            (Ynniss/final_parseq_jn)
#         B  SATRN   best_recog_word_acc_epoch_10.pth (Ynniss/satrn_small,
#                    mmocr 1.x; architecture read from the config the
#                    checkpoint carries: SATRN encoder + NRTR decoder)
#     vote_pool consolidation (fuse_jn.py): every surviving frame is decoded
#         by both models; score(L) = n_A(L) + n_B(L), the number of pooled
#         frame decodes that read L. Ties: more pooled votes (equal by
#         construction), larger summed raw log-likelihood, then label string.
#
# This is the ONLY consolidation rule in this build. The single-model
# maxconf / wvote / vote rules of the previous build are no longer selectable
# here; they remain in evaluate_jn.py and run_eval.py --rule for the
# evaluation harness.
#
# Measured on GT-box tracklets (fusion run of 2026-08-17, kaggle-fusion-1
# notebook; legibility > 0.72, det > 0.52, stride 5, both audits 0 fail):
#     GSR-2024 test (1038 trk)   PARSeq maxconf 87.48 / 90.83 / -1 F1 0.82
#                                vote_pool      87.86 / 91.34 / 0.82
#     SN jersey-2023 test (1211) PARSeq maxconf 84.72 / 83.76 / 0.81
#                                vote_pool      85.30 / 84.58 / 0.81
# (trk_acc / numbered / -1 F1). The rule was chosen on those test splits; no
# number exists yet for predicted tracklets through the host pipeline.
#
# Usage:
#     from jn_recognizer import JerseyNumberRecognizer
#     rec = JerseyNumberRecognizer("models")            # loads all 4 nets
#     number, confidence, n_used = rec.predict(tracklet)
#
#     `tracklet` is an iterable of frames; each frame is either
#         (pil_image_or_path, xywh)   with xywh = (x, y, w, h) player box
#         pil_image_or_path           alone -> treated as an ALREADY-CROPPED
#                                     player image (xywh=None)
#     `number` is "1".."99" or "-1" (not visible); `confidence` is the
#     winning label's share of the pooled frame votes, i.e.
#     (n_A(L) + n_B(L)) / (2 * n_used), 0.0 when no frame survives;
#     `n_used` is the number of frames that entered the vote (each read by
#     both models).
#
# Efficiency note vs the evaluation harness: the harness runs both readers on
# every detected frame and gates at merge time. Here the gates run FIRST and
# the readers see only the survivors -- typically ~20% of detected frames at
# the frozen thresholds. Predictions are identical: the gates are conjunctive
# and the vote sees the same surviving set.
#
# The pre-cropped input mode matches the SN-JNR validation setting exactly:
# the player pad is inert on already-cropped images (there is no surrounding
# frame to pad into), so expect SN-JNR-like rather than GSR-like behaviour.

import os
from types import SimpleNamespace

import numpy as np
from PIL import Image

from common import subsample
from crop_classifier import crop_box
from fuse_jn import fuse_stats, label_stats, votes_of
from legibility import frame_verdicts

# The operating point. `LEGIBILITY_THR` is the DEFAULT for the constructor
# argument of the same name, not a constant to edit: jn_gsr.yaml ->
# jn_gsr_api -> predict_tracklets.py --legibility-thr -> here, and the value
# is part of the host cache key, so changing it in the config actually
# re-runs. Editing this line instead would leave that chain stale.
#
# 0.72 is the configured gate for this build; det 0.52 is the SN-JNR test
# argmax, neutral on GSR test. Both were fixed before the fusion run above.
LEGIBILITY_THR = 0.72
DET_THR = 0.52

# The one consolidation rule of this build (a fuse_jn.RULES member).
RULE = "vote_pool"

# Default staged locations inside models_dir (fetch_weights.py / stage_weights.py).
PARSEQ_CKPT = "parseq_gsr_ft_s1.ckpt"
SATRN_CKPT = "recog2/best_recog_word_acc_epoch_10.pth"

# Sentinel "whole image" box for pre-cropped inputs; crop/pad code clips it.
_FULL = (0.0, 0.0, 1e9, 1e9)


class JerseyNumberRecognizer:
    """Frozen-config tracklet -> jersey number. See module docstring."""

    def __init__(self, models_dir="models", stride=5, fp16=True,
                 models=None,
                 dbnet_ckpt="best_icdar_hmean_epoch_10.pth",
                 parseq_ckpt=PARSEQ_CKPT,
                 satrn_ckpt=SATRN_CKPT,
                 legibility_ckpt="sn_legibility.pth",
                 dbnet_cfg="mmocr_cfg/dbnetpp_infer.py",
                 legibility_thr=LEGIBILITY_THR):
        """models_dir must hold the four validated checkpoints (fetch with
        fetch_weights.py + stage_weights.py). `models` injects prebuilt
        {"gate","leg","read_many","read_many2"} -- the offline-test hook; when
        given, nothing is loaded and fp16 is ignored. stride: read every Nth
        frame (5 == the validated setting).
        satrn_ckpt: the second recogniser, a path relative to models_dir (or
        absolute). Its mmocr config and character dictionary are resolved by
        mmocr_reader (a .py / .txt beside the checkpoint, else the config
        stored inside the checkpoint and mmocr's own dictionary file).
        legibility_thr: the strictly-greater p_legible cut (0.72). The SAME
        value gates in predict() and is handed to build_models, so the
        classifier's recorded threshold in the provenance record cannot drift
        from the one that actually decided which frames were read."""
        self.rule = RULE
        self.stride = max(1, int(stride))
        self.legibility_thr = float(legibility_thr)
        self._fp16 = bool(fp16)
        if models is not None:
            for k in ("gate", "leg", "read_many", "read_many2"):
                if k not in models:
                    raise KeyError(f"models hook lacks {k!r}")
            self._M = models
            return
        import run_eval  # reuse the validated builder -- no duplicate logic
        a = SimpleNamespace(
            ckpt=os.path.join(models_dir, dbnet_ckpt),
            parseq=os.path.join(models_dir, parseq_ckpt),
            legibility_weights=os.path.join(models_dir, legibility_ckpt),
            dbnet_cfg=dbnet_cfg, legibility_thr=self.legibility_thr,
            legibility_size=256, det_floor=0.2, roi_pad=0.12,
            player_pad=0.18, det_cache=512)
        M = run_eval.build_models(a)
        # The second recogniser, on the same device as PARSeq. mmocr_reader
        # refuses to load if the dictionary lacks a digit, if END collides
        # with a digit, or if the dictionary size differs from the
        # checkpoint's classifier width -- the failure modes that would
        # otherwise corrupt every vote silently.
        from mmocr_reader import build_recog2_reader
        satrn_path = satrn_ckpt if os.path.isabs(satrn_ckpt) \
            else os.path.join(models_dir, satrn_ckpt)
        if not os.path.exists(satrn_path):
            raise FileNotFoundError(
                f"[jn] second recogniser checkpoint not found: {satrn_path} "
                f"-- run fetch_weights.py --out-dir {models_dir}")
        r2, read_many2 = build_recog2_reader(satrn_path, device=M["device"])
        M["read_many2"] = read_many2
        M["provenance"]["satrn"] = {
            "path": r2.pth, "sha256": run_eval._sha256(r2.pth),
            "config": r2.cfg_path, "dict_file": r2.dict_file,
            "arch": r2.arch, "n_classes": r2.n_classes,
            "digit_idx": r2.digit_idx, "end_idx": r2.end_idx}
        self._M = M

    # ------------------------------------------------------------------ #
    def _amp(self):
        if not self._fp16:
            import contextlib
            return contextlib.nullcontext()
        try:
            import torch
            if torch.cuda.is_available():
                return torch.autocast("cuda", dtype=torch.float16)
        except ImportError:
            pass
        import contextlib
        return contextlib.nullcontext()

    @staticmethod
    def _norm(frame):
        """Accept (img, xywh) | (img,) | bare img; path or PIL."""
        img, xywh = (frame if isinstance(frame, (tuple, list)) and
                     len(frame) == 2 else (frame, None))
        if isinstance(img, (str, os.PathLike)):
            key, im = str(img), Image.open(img).convert("RGB")
        else:
            key, im = f"mem-{id(img)}", img.convert("RGB")
        return key, im, (_FULL if xywh is None else tuple(map(float, xywh)))

    @staticmethod
    def consolidate(logls_a, logls_b):
        """vote_pool over two readers' outputs for the SAME frames.
        -> (label, share). logls_x: [(tens_logl[11], units_logl[11]), ...]."""
        if len(logls_a) != len(logls_b):
            raise RuntimeError(f"[jn] readers disagree on frame count: "
                               f"{len(logls_a)} vs {len(logls_b)}")
        if not logls_a:
            return "-1", 0.0
        sa = label_stats([t for t, _ in logls_a], [u for _, u in logls_a])
        sb = label_stats([t for t, _ in logls_b], [u for _, u in logls_b])
        label = fuse_stats(sa, sb, rules=(RULE,))[RULE]
        pooled = votes_of(sa, label) + votes_of(sb, label)
        return label, pooled / (len(logls_a) + len(logls_b))

    def predict(self, tracklet):
        """-> (number "1".."99" | "-1", vote_share, n_frames_in_vote)."""
        M = self._M
        frames = subsample(list(tracklet), self.stride)

        # pass 1: decode once; legibility crops + detector (score only kept)
        crops, ci, det = [], [], []
        for i, frame in enumerate(frames):
            try:
                key, im, xywh = self._norm(frame)
            except Exception:
                det.append(None)
                continue
            c, _ = crop_box(im, xywh)
            if c is not None:
                crops.append(c)
                ci.append(i)
            with self._amp():
                roi, _src, ds = M["gate"].roi_crop_scored(key, xywh, img=im)
            det.append(None if roi is None else (roi, float(ds)))

        pl = np.zeros(len(frames))
        if crops:
            with self._amp():
                scores, _ = M["leg"].score(crops)
            for j, i in enumerate(ci):
                pl[i] = float(scores[j])

        # gates FIRST (identical semantics to the harness merge: both strict)
        leg_ok = frame_verdicts(pl, self.legibility_thr)
        rois = [d[0] for i, d in enumerate(det)
                if d is not None and leg_ok[i] and d[1] > DET_THR]

        # both readers on the survivors, then the pooled vote
        if not rois:
            return "-1", 0.0, 0
        with self._amp():
            logls = M["read_many"](rois)
            logls2 = M["read_many2"](rois)
        label, share = self.consolidate(logls, logls2)
        return label, float(share), len(rois)


# ------------------------------ self-tests ------------------------------- #
if __name__ == "__main__":
    # Offline: stubbed models, no torch/GPU/checkpoints. Verifies gate order,
    # strictness at both thresholds, the pooled vote and its share, the -1
    # path, pre-cropped mode, stride, and that both readers see the same ROIs.
    def _logl(hot, p=0.9):
        """11-vector of log-probs: p on `hot`, rest of mass spread evenly."""
        import math
        v = [math.log((1 - p) / 10)] * 11
        v[hot] = math.log(p)
        return v

    class StubGate:
        def __init__(self, plan):            # key-index -> (ds | None)
            self.plan, self.calls = plan, 0

        def roi_crop_scored(self, key, xywh, img=None):
            ds = self.plan[self.calls]
            self.calls += 1
            if ds is None:
                return None, "none", 0.0
            return img, "det", ds            # roi = the image itself (stub)

    class StubLeg:
        def __init__(self, pls):
            self.pls = list(pls)

        def score(self, crops):
            out, self.pls = self.pls[:len(crops)], self.pls[len(crops):]
            return out, None

    class Reader:
        """each surviving roi -> a (digit, p); records the ROIs it was given"""
        def __init__(self, spec):
            self.spec, self.seen = list(spec), []

        def __call__(self, rois):
            self.seen.append([id(r) for r in rois])
            out = []
            for _ in rois:
                d, p = self.spec.pop(0)
                out.append((_logl(d, p), _logl(10, p)))
            return out

    def models(gate, leg, ra, rb):
        return {"gate": gate, "leg": leg, "read_many": ra, "read_many2": rb}

    imgs = [Image.new("RGB", (40, 60), (i * 9, 50, 50)) for i in range(6)]
    tr = [(im, (0, 0, 40, 60)) for im in imgs]

    # 1) gates strict at both thresholds; pooled vote; both readers see the
    #    identical ROI list. frame0 in; frame1 pl=0.72 NOT > 0.72 -> leg gate;
    #    frame2 ds=0.52 NOT > 0.52 -> det gate; frame3 in; frame4 no roi;
    #    frame5 in. 3 survivors: A reads 7,7,9 ; B reads 7,9,9 -> 3 vs 3 votes;
    #    tie -> summed raw log-lik: 7's frames carry p .9/.9/.9, 9's .6/.6/.6
    #    -> 7 wins with share 3/6.
    ra = Reader([(7, .9), (7, .9), (9, .6)])
    rb = Reader([(7, .9), (9, .6), (9, .6)])
    rec = JerseyNumberRecognizer(models=models(
        StubGate([0.9, 0.9, 0.52, 0.9, None, 0.9]),
        StubLeg([0.99, 0.72, 0.99, 0.99, 0.99, 0.99]), ra, rb), stride=1)
    n, conf, used = rec.predict(tr)
    assert (n, used) == ("7", 3) and abs(conf - 3 / 6) < 1e-9, (n, conf, used)
    assert ra.seen == rb.seen and len(ra.seen[0]) == 3
    assert rec.rule == "vote_pool"

    # 1b) a clear majority: A 7,7,9 ; B 7,7,7 -> 7 with 5/6
    rec = JerseyNumberRecognizer(models=models(
        StubGate([0.9, 0.9, 0.9]), StubLeg([0.99, 0.99, 0.99]),
        Reader([(7, .9), (7, .9), (9, .9)]),
        Reader([(7, .9), (7, .9), (7, .9)])), stride=1)
    n, conf, used = rec.predict(tr[:3])
    assert (n, used) == ("7", 3) and abs(conf - 5 / 6) < 1e-9, (n, conf, used)

    # 1c) B outvotes A: A says 4 once, B says 8 with more confident frames on
    #     a 1-1 tie -> strength decides -> 8 (the second model can win)
    rec = JerseyNumberRecognizer(models=models(
        StubGate([0.9]), StubLeg([0.99]),
        Reader([(4, .6)]), Reader([(8, .95)])), stride=1)
    assert rec.predict(tr[:1])[0] == "8"

    # 2) nothing survives -> -1 with zero confidence, readers never called
    ra, rb = Reader([]), Reader([])
    rec = JerseyNumberRecognizer(models=models(
        StubGate([0.4]), StubLeg([0.99]), ra, rb), stride=1)
    assert rec.predict([tr[0]]) == ("-1", 0.0, 0)
    assert ra.seen == [] and rb.seen == []

    # 3) pre-cropped mode: bare images, full-image sentinel reaches the gate
    seen = []

    class SpyGate(StubGate):
        def roi_crop_scored(self, key, xywh, img=None):
            seen.append(xywh)
            return super().roi_crop_scored(key, xywh, img)

    rec = JerseyNumberRecognizer(models=models(
        SpyGate([0.9, 0.9]), StubLeg([0.99, 0.99]),
        Reader([(4, .9), (4, .9)]), Reader([(4, .9), (4, .9)])), stride=1)
    n, conf, used = rec.predict(imgs[:2])
    assert (n, conf, used) == ("4", 1.0, 2) and seen == [_FULL, _FULL]

    # 4) stride honoured: 6 frames @ stride 5 -> frames 0 and 5 only
    g = StubGate([0.9, 0.9])
    rec = JerseyNumberRecognizer(models=models(
        g, StubLeg([0.99, 0.99]),
        Reader([(2, .9), (2, .9)]), Reader([(2, .9), (2, .9)])), stride=5)
    n, _, used = rec.predict(imgs)
    assert (n, used, g.calls) == ("2", 2, 2)

    # 5) two-digit decode through the real vote
    rec = JerseyNumberRecognizer(models=models(
        StubGate([0.9]), StubLeg([0.99]),
        lambda rois: [(_logl(1), _logl(0))],
        lambda rois: [(_logl(1), _logl(0))]), stride=1)
    assert rec.predict([imgs[0]])[0] == "10"

    # 6) the legibility cut sits exactly at the DEFAULT and is strict:
    #    0.7201 survives, 0.72 does not.
    assert LEGIBILITY_THR == 0.72, LEGIBILITY_THR
    for pl_value, expect in ((0.7201, ("4", 1)), (0.72, ("-1", 0))):
        r = JerseyNumberRecognizer(models=models(
            StubGate([0.9]), StubLeg([pl_value]),
            Reader([(4, .9)]), Reader([(4, .9)])), stride=1)
        n, _, used = r.predict([tr[0]])
        assert (n, used) == expect, (pl_value, n, used)

    # 7) legibility_thr is a CONSTRUCTOR ARGUMENT, not a module constant read
    #    at call time: the same pl=0.8 frame passes at 0.72 and fails at 0.9.
    for thr, expect in ((0.72, ("4", 1)), (0.9, ("-1", 0))):
        r = JerseyNumberRecognizer(models=models(
            StubGate([0.9]), StubLeg([0.8]),
            Reader([(4, .9)]), Reader([(4, .9)])), stride=1,
            legibility_thr=thr)
        assert r.legibility_thr == thr
        n, _, used = r.predict([tr[0]])
        assert (n, used) == expect, (thr, n, used)

    # 8) a second reader that drops a frame is an error, not a silent vote
    rec = JerseyNumberRecognizer(models=models(
        StubGate([0.9, 0.9]), StubLeg([0.99, 0.99]),
        Reader([(4, .9), (4, .9)]), lambda rois: [(_logl(4), _logl(10))]),
        stride=1)
    try:
        rec.predict(tr[:2])
        raise AssertionError("length mismatch must raise")
    except RuntimeError:
        pass

    # 9) the models hook must carry both readers
    try:
        JerseyNumberRecognizer(models={"gate": None, "leg": None,
                                       "read_many": None})
        raise AssertionError("missing read_many2 must raise")
    except KeyError:
        pass

    # 10) consolidate() agrees with fuse_jn's own vote_pool on the same frames
    from fuse_jn import fuse
    A = [(_logl(2, .8), _logl(3, .8)), (_logl(2, .8), _logl(3, .8)),
         (_logl(4, .95), _logl(10, .95))]
    B = [(_logl(2, .7), _logl(3, .7)), (_logl(4, .7), _logl(10, .7)),
         (_logl(4, .7), _logl(10, .7))]
    ref = fuse([t for t, _ in A], [u for _, u in A],
               [t for t, _ in B], [u for _, u in B], rules=("vote_pool",))
    lab, share = JerseyNumberRecognizer.consolidate(A, B)
    assert lab == ref["vote_pool"] and abs(share - 3 / 6) < 1e-9, (lab, share)

    print("jn_recognizer: all self-tests passed (strict gates at 0.72/0.52, "
          "gate order, vote_pool label + pooled share, second model can win, "
          "-1 path, pre-cropped sentinel, stride, two-digit decode, "
          "legibility_thr is a constructor knob, reader count mismatch "
          "raises, hook requires both readers, agrees with fuse_jn)")
