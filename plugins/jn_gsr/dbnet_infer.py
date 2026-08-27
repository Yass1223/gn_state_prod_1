# dbnet_infer.py -- DBNet++ detector wrapper + the ROI provider that replaces
# ViTPose AND the legibility gate.
#
# Checkpoint: ./best_icdar_hmean_epoch_10.pth (the Lightning working folder),
# overridable via DBNetDetector(weights=...) or $JN_DBNET_CKPT.
#
# Three objects:
#   DBNetDetector  -- thin TextDetInferencer wrapper; detect(pil) -> (quads, scores)
#                     in the coordinate frame of the image passed in.
#   DetectorGate   -- manifest-time inclusion decision with the SAME signature the
#                     legibility gate had, is_legible(frame_path, xywh) -> bool, so
#                     gsr_train_data.build_manifest needs no modification. Caches the
#                     detection so the test-time ROI cut does not re-run the model.
#   make_pose_fn   -- the callable two_stage_materialize expects: crop -> ROI | None.
#
# WHY INCLUSION HAPPENS AT MANIFEST TIME (arm-consistency, verified):
#   ViTPose's torso_crop had fallback=True, so it essentially always returned a
#   crop -- pose_fn was TOTAL. DBNet++ at 0.52 with no fallback is PARTIAL. In
#   two_stage_variant the ON arm detects on the JITTERED crop and the OFF arm on
#   the CLEAN one, so if inclusion were decided per-variant the ON arm would get
#   K_ON independent attempts per frame against the OFF arm's one, and would
#   recover ~10-30% more frames -- a one-directional bias in favour of the very
#   arm whose benefit is being measured. Deciding inclusion ONCE on the clean crop
#   restores identical frame populations. It is also exactly where the legibility
#   gate used to sit, so the structure is unchanged.

import os

import numpy as np
from PIL import Image

from gsr_adapter import pad_box
from roi_dbnet import DET_THR, PAD_FRAC, roi_from_detections

DEFAULT_CKPT = "best_icdar_hmean_epoch_10.pth"
DEFAULT_CFG = "mmocr_cfg/dbnetpp_infer.py"
PLAYER_PAD = float(os.environ.get("JN_PLAYER_PAD", 0.18))
                        # player-box pad before detection (same constant as the
                        # old torso path: common.LEG_PAD -- framing parity);
                        # env override so subprocess tools match section 0


def resolve_ckpt(weights=None):
    """Checkpoint in the CURRENT folder by default (Lightning working dir)."""
    w = weights or os.environ.get("JN_DBNET_CKPT") or DEFAULT_CKPT
    if not os.path.exists(w):
        raise FileNotFoundError(
            f"DBNet++ checkpoint not found: {w!r}\n"
            f"Upload best_icdar_hmean_epoch_10.pth into this folder, or pass "
            f"weights=... / set $JN_DBNET_CKPT.")
    return w


class DBNetDetector:
    """TextDetInferencer wrapper. detect() returns (quads, scores) where quads is
    a list of (4,2) float arrays in the coordinate frame of the image given."""

    def __init__(self, weights=None, config=DEFAULT_CFG, device=None):
        os.environ["MPLBACKEND"] = "Agg"   # FORCE: Kaggle/Studio kernels preset a
        # backend whose module is absent from the venv; mmdet pulls in pyplot.
        from mmocr.apis import TextDetInferencer
        import torch
        self.weights = resolve_ckpt(weights)
        self.config = config
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.inferencer = TextDetInferencer(model=config, weights=self.weights,
                                            device=self.device)
        self.n_calls = 0

    def detect(self, img):
        """img: PIL.Image | np.ndarray (RGB). -> (quads, scores)."""
        arr = np.asarray(img.convert("RGB")) if isinstance(img, Image.Image) \
            else np.asarray(img)
        bgr = np.ascontiguousarray(arr[:, :, ::-1])   # inferencer expects BGR;
        # ::-1 alone yields a negative-stride VIEW, which cv2/mmcv ops can reject
        out = self.inferencer(bgr, return_datasamples=False, progress_bar=False)
        self.n_calls += 1
        pred = out["predictions"][0]
        polys = pred.get("polygons", []) or []
        scores = list(pred.get("scores", [1.0] * len(polys)))
        quads = []
        for p in polys:
            q = np.asarray(p, np.float32).reshape(-1, 2)
            if len(q) != 4:                            # guard: non-quad polygon
                import cv2
                q = cv2.boxPoints(cv2.minAreaRect(q)).astype(np.float32)
            quads.append(q)
        return quads, scores


def padded_player_crop(frame_pil, box_xywh, frac=PLAYER_PAD):
    """The 18%-padded PLAYER crop -- the image the detector sees, at train and
    test alike. Returns (crop_pil | None, (x0, y0))."""
    W, H = frame_pil.size
    x0, y0, x1, y1 = pad_box(*box_xywh, W, H, frac)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None, (0, 0)
    return frame_pil.crop((x0, y0, x1, y1)), (x0, y0)


class DetectorGate:
    """Detector-backed frame inclusion + ROI cut, with a shared cache.

    is_legible(frame_path, xywh) -> bool     (drop-in for LegibilityGate)
    roi_crop(frame_path, xywh)   -> (roi_pil | None, source)
    """

    def __init__(self, detector, thr=DET_THR, pad_frac=PAD_FRAC, cache_max=4096):
        self.det = detector
        self.thr = thr
        self.pad_frac = pad_frac
        self._cache = {}
        self._cache_max = cache_max
        self.stats = {"calls": 0, "hits": 0, "det": 0, "none": 0}

    @staticmethod
    def _key(frame_path, xywh):
        return (str(frame_path), tuple(round(float(v), 3) for v in xywh))

    def _detections(self, frame_path, xywh, img=None):
        k = self._key(frame_path, xywh)
        if k in self._cache:
            self.stats["hits"] += 1
            return self._cache[k]
        # v2: callers that already decoded the frame pass it in -- the JPEG is
        # decoded ONCE per (frame, box) across classifiers and detector.
        if img is None:
            img = Image.open(frame_path).convert("RGB")
        crop, _ = padded_player_crop(img, xywh)
        val = ([], [], None) if crop is None else (*self.det.detect(crop), crop)
        if len(self._cache) >= self._cache_max:
            self._cache.pop(next(iter(self._cache)))
        self._cache[k] = val
        self.stats["calls"] += 1
        return val

    def is_legible(self, frame_path, box_xywh, pad=None):
        """Frame-inclusion verdict. With the legibility classifier removed, the
        detector score threshold IS the legibility decision."""
        _, scores, _ = self._detections(frame_path, box_xywh)
        ok = any(float(s) >= self.thr for s in scores)
        self.stats["det" if ok else "none"] += 1
        return ok

    def roi_crop(self, frame_path, box_xywh, img=None):
        """Test-time ROI. Reuses the cached detection when the frame already went
        through is_legible, so the model runs once per (frame, box)."""
        roi, src, _ = self.roi_crop_scored(frame_path, box_xywh, img=img)
        return roi, src

    def roi_crop_scored(self, frame_path, box_xywh, img=None):
        """(roi | None, source, best_score). best_score is the highest detector
        score >= self.thr (0.0 when none). With the gate constructed at a LOW
        floor (v2: --det-floor), storing this per frame makes the operating
        threshold a merge-time parameter: the ROI is always the highest-scoring
        quad's, so gating at any thr >= floor is exactly `best_score >= thr`."""
        quads, scores, crop = self._detections(frame_path, box_xywh, img=img)
        if crop is None:
            return None, "none", 0.0
        best = max((float(s) for s in scores if float(s) >= self.thr),
                   default=0.0)
        roi, src = roi_from_detections(crop, quads, scores,
                                       thr=self.thr, pad_frac=self.pad_frac)
        return roi, src, best


def make_pose_fn(detector, thr=DET_THR, pad_frac=PAD_FRAC):
    """The callable two_stage_materialize.two_stage_variant expects:
    player_crop -> ROI | None. Detects on whatever image it is handed, so the ON
    arm's jittered crops get a genuine per-variant detection (which is the point
    of the two-stage split); a variant with no detection is skipped by the
    existing `if sample is None: continue` path in materialize()."""
    def pose_fn(player_crop):
        quads, scores = detector.detect(player_crop)
        roi, _src = roi_from_detections(player_crop, quads, scores,
                                        thr=thr, pad_frac=pad_frac)
        return roi
    return pose_fn


if __name__ == "__main__":
    # Pure-logic tests: no mmocr, no GPU, no checkpoint. The detector is stubbed,
    # so this exercises the gate/cache/ROI wiring only.
    import math

    def rot_quad(cx, cy, w, h, deg):
        t = math.radians(deg); c, s = math.cos(t), math.sin(t)
        pts = [(-w/2, -h/2), (-w/2, h/2), (w/2, h/2), (w/2, -h/2)]
        return np.array([(cx + x*c - y*s, cy + x*s + y*c) for x, y in pts], np.float32)

    class StubDet:
        def __init__(self, score): self.score = score; self.n = 0
        def detect(self, img):
            self.n += 1
            return [rot_quad(40, 50, 30, 22, 8)], [self.score]

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "f.png")
        Image.fromarray(np.full((300, 200, 3), 100, np.uint8)).save(fp)
        box = (40, 60, 100, 150)

        # above threshold -> legible, and roi_crop reuses the cached detection
        det = StubDet(0.80); gate = DetectorGate(det)
        assert gate.is_legible(fp, box) is True
        assert det.n == 1, "detector should have run once"
        roi, src = gate.roi_crop(fp, box)
        assert src == "det" and roi is not None
        assert det.n == 1, f"roi_crop must hit the cache, detector ran {det.n}x"
        assert gate.stats["hits"] == 1

        # below threshold -> not legible, and roi_crop agrees (one source of truth)
        det2 = StubDet(0.51); gate2 = DetectorGate(det2)
        assert gate2.is_legible(fp, box) is False
        assert gate2.roi_crop(fp, box) == (None, "none")

        # exactly at threshold is INCLUSIVE (>= 0.52)
        det3 = StubDet(DET_THR); gate3 = DetectorGate(det3)
        assert gate3.is_legible(fp, box) is True, "0.52 must be inclusive"

        # make_pose_fn: returns an ROI above thr, None below
        pf_hi = make_pose_fn(StubDet(0.9)); pf_lo = make_pose_fn(StubDet(0.1))
        crop = Image.fromarray(np.full((120, 90, 3), 80, np.uint8))
        assert pf_hi(crop) is not None and pf_lo(crop) is None

        # padded_player_crop: 18% pad, clipped at frame edges
        img = Image.fromarray(np.full((1000, 1000, 3), 0, np.uint8))
        c, (ox, oy) = padded_player_crop(img, (100, 100, 100, 100))
        assert (ox, oy) == (82, 82) and c.size == (136, 136), (ox, oy, c.size)

        # resolve_ckpt raises a useful error when the checkpoint is absent
        try:
            resolve_ckpt(os.path.join(d, "missing.pth")); raise AssertionError("should raise")
        except FileNotFoundError as e:
            assert "best_icdar_hmean_epoch_10.pth" in str(e)

    print("dbnet_infer: pure-logic tests passed (gate verdict, cache reuse, "
          "threshold inclusivity, pose_fn, player pad, missing-ckpt error)")
