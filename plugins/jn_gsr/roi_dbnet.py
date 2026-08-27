# roi_dbnet.py -- DBNet++ polygon -> recognizer input (FINAL specification).
#
# Replaces BOTH pose_vitpose_hf.ViTPoseTorso and common.padded_torso_crop as the
# ROI provider; with the legibility gate removed, the score threshold below is
# also the frame-inclusion (legibility) decision.
#
# Locked decisions:
#   * DET_THR = 0.52 on the detector score; below it a frame contributes nothing
#     (an all-'none' tracklet consolidates to -1)
#   * several quads -> highest score wins
#   * geometry = corner-extreme AABB; NO rotation is ever applied to pixels
#   * padding in units of GLYPH HEIGHT (the quad edge nearest image-vertical),
#     not AABB height, which inflates with lean
#   * integer-grid extraction, rounded OUTWARD, replicate border:
#     ZERO interpolation here -- the recognizer's own BICUBIC resize is the
#     single resample in the entire path
#   * byte-identical constants and code path at train (inside two_stage_variant,
#     per augmented variant) and at test
#
# Pure numpy + PIL: import- and test-safe without a GPU, like common.py.
# The detector call itself (mmocr TextDetInferencer) stays outside this module.

import os

import numpy as np
from PIL import Image

# Environment overrides exist so that notebook section 0 -- which runs in the
# KERNEL -- also governs the `!{PYBIN} verify_pipeline.py roi` audit and the
# `!{PYBIN} viz_check.py` panels, which run as SUBPROCESSES. IPython's `!` lines
# inherit os.environ, so setting JN_DET_THR there reaches every tool; module
# attributes alone would not. Without this, tuning DET_THR in section 0 would
# leave the audit silently auditing 0.52.
DET_THR = float(os.environ.get("JN_DET_THR", 0.52))    # ROI selection AND legibility
PAD_FRAC = float(os.environ.get("JN_PAD_FRAC", 0.12))  # margin, in glyph heights
MIN_SIDE = 4       # reject degenerate detections (pixels, pre-padding)


def _aabb(quad):
    q = np.asarray(quad, np.float32).reshape(-1, 2)
    return float(q[:, 0].min()), float(q[:, 1].min()), \
        float(q[:, 0].max()), float(q[:, 1].max())


def _glyph_height(quad):
    """Length of the quad edge whose direction lies closest to image-vertical.
    Convention-free: uses the corner points only, never minAreaRect's angle.
    Wrong only past 45 deg of lean, where the cost is a slightly-off margin --
    never a rotated crop ("use the angle only where a mistake is cheap")."""
    q = np.asarray(quad, np.float32).reshape(-1, 2)
    e1, e2 = q[1] - q[0], q[3] - q[0]           # two adjacent edges

    def verticality(v):
        n = float(np.hypot(v[0], v[1]))
        return abs(float(v[1])) / n if n > 0 else 0.0

    e = e1 if verticality(e1) >= verticality(e2) else e2
    return float(np.hypot(e[0], e[1]))


def roi_from_detections(crop_pil, quads, scores,
                        thr=DET_THR, pad_frac=PAD_FRAC):
    """(player crop, DBNet++ polygons + scores) -> (roi_pil | None, source).

    source in {'det', 'none'}. 'none' is the frame-level illegibility verdict.
    quads: iterable of flat len-8 or (4,2) point arrays in CROP coordinates.
    """
    keep = [(float(s), q) for s, q in zip(scores, quads) if float(s) >= thr]
    if not keep:
        return None, "none"
    _, q = max(keep, key=lambda t: t[0])                 # highest score wins

    x0, y0, x1, y1 = _aabb(q)
    if (x1 - x0) < MIN_SIDE or (y1 - y0) < MIN_SIDE:
        return None, "none"

    m = pad_frac * _glyph_height(q)
    X0, Y0 = int(np.floor(x0 - m)), int(np.floor(y0 - m))   # round OUTWARD:
    X1, Y1 = int(np.ceil(x1 + m)), int(np.ceil(y1 + m))     # padding never truncates

    a = np.asarray(crop_pil.convert("RGB"))
    H, W = a.shape[:2]
    core = a[max(0, Y0):min(H, Y1), max(0, X0):min(W, X1)]
    if core.size == 0:
        return None, "none"
    pad_t, pad_l = max(0, -Y0), max(0, -X0)
    pad_b, pad_r = max(0, Y1 - H), max(0, X1 - W)
    out = np.pad(core, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="edge")
    return Image.fromarray(out), "det"


# ------------------------------- self-tests --------------------------------
if __name__ == "__main__":
    import math

    def rot_quad(cx, cy, w, h, deg):
        t = math.radians(deg)
        c, s = math.cos(t), math.sin(t)
        pts = [(-w / 2, -h / 2), (-w / 2, h / 2), (w / 2, h / 2), (w / 2, -h / 2)]
        return np.array([(cx + x * c - y * s, cy + x * s + y * c)
                         for x, y in pts], np.float32)

    crop = Image.fromarray(np.full((200, 160, 3), 90, np.uint8))

    # 1) all detections below threshold -> ('none')
    q = rot_quad(80, 100, 40, 30, 10)
    roi, src = roi_from_detections(crop, [q], [0.519])
    assert roi is None and src == "none", "threshold gate failed"

    # 2) highest score wins among several
    q_lo = rot_quad(40, 60, 30, 22, 5)
    q_hi = rot_quad(110, 120, 36, 26, -8)
    roi, src = roi_from_detections(crop, [q_lo, q_hi], [0.60, 0.90])
    ax = _aabb(q_hi)
    m = PAD_FRAC * _glyph_height(q_hi)
    exp_w = int(np.ceil(ax[2] + m)) - int(np.floor(ax[0] - m))
    exp_h = int(np.ceil(ax[3] + m)) - int(np.floor(ax[1] - m))
    assert src == "det" and roi.size == (exp_w, exp_h), "selection/size failed"

    # 3) glyph height: at modest lean it is the ~vertical edge (the 26 side),
    #    NOT the AABB height (which is larger)
    gh = _glyph_height(rot_quad(0, 0, 36, 26, 15))
    assert abs(gh - 26.0) < 1e-4, f"glyph height {gh} != 26"
    aabb_h = _aabb(rot_quad(0, 0, 36, 26, 15))
    assert (aabb_h[3] - aabb_h[1]) > 26.0, "AABB height should exceed glyph height"

    # 4) flat len-8 polygon input (mmocr's native format) accepted
    flat = rot_quad(80, 100, 40, 30, 0).reshape(-1)
    roi, src = roi_from_detections(crop, [flat], [0.9])
    assert src == "det", "flat polygon input failed"

    # 5) replicate border at the crop edge: requested extent preserved exactly
    q_edge = rot_quad(8, 8, 30, 24, 12)          # padded AABB exits top-left
    roi, src = roi_from_detections(crop, [q_edge], [0.9])
    ax = _aabb(q_edge)
    m = PAD_FRAC * _glyph_height(q_edge)
    exp_w = int(np.ceil(ax[2] + m)) - int(np.floor(ax[0] - m))
    exp_h = int(np.ceil(ax[3] + m)) - int(np.floor(ax[1] - m))
    assert roi.size == (exp_w, exp_h), "edge replication changed the extent"

    # 6) degenerate detection rejected
    tiny = rot_quad(80, 100, 3, 2, 0)
    roi, src = roi_from_detections(crop, [tiny], [0.99])
    assert roi is None and src == "none", "degenerate guard failed"

    # 7) determinism: identical input -> identical bytes
    r1, _ = roi_from_detections(crop, [q_hi], [0.9])
    r2, _ = roi_from_detections(crop, [q_hi], [0.9])
    assert np.array_equal(np.asarray(r1), np.asarray(r2)), "non-deterministic"

    print("roi_dbnet: all 7 self-tests passed "
          "(threshold, argmax-score, glyph-height, flat-input, "
          "edge-replicate, degenerate, determinism)")
