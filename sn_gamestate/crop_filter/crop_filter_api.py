"""Single / multi label for every detector box (tracked-only contaminator rule).

Runs directly after ``track`` (BoT-SORT), before ``split_merge``. Every detection of a
frame receives three columns, tracked or not:

    crop_rT     max over contaminators B of inter(T, B) / area(T)
    crop_rB     max over contaminators B of inter(T, B) / area(B)
    crop_single True iff crop_rT <= thr_target and crop_rB < thr_other
    crop_trigger index (detections.index label) of the box that attained the
                ratio which made T multi, or NaN when T is single / no contaminator

A crop is "single" when it shows one person. Downstream stages that must ignore
multi crops (the team embedding and the role/team rules) read ``crop_single``, and
the ``split_merge`` stage clusters and merges on single crops only; no stage
recomputes overlaps. The jersey stage does not read it (it uses every crop).

Contaminator set C(T) — which boxes may make T multi — is selected by
``contam_mode``:

    all              every other box in the frame (the original rule)
    conf             boxes with bbox_conf > conf_thr_other
    tracked          boxes carrying a track_id                (the adopted rule)
    tracked_or_conf  tracked, or untracked with bbox_conf > conf_thr_other

No detection is removed by any mode; the rule only decides which boxes may veto
another box's single label. A box excluded from C(T) is still labelled itself.

The rule, thresholds (rT <= 0.25, rB < 0.40) and the four modes are the ones of
the splitter notebook's ``label_single`` (tracklet-splitter_dbscan_4_raw.ipynb,
``contam_mode="tracked"``) and of the contaminator-selection note: in the audited
samples the tracked-only rule removed 98.1 % (valid) / 89.4 % (test) of
false-positive triggers while hiding no real overlap. The stage is placed after
the tracker because "has a track_id" exists only from that point on.

Two facts recorded for the audit rather than hidden:

* ``split_merge`` can later leave a detection unassigned (track_id NaN). A box that
  vetoed another box here may therefore be untracked by the time the role stage
  runs. ``crop_trigger`` lets the audit count such cases; the label is NOT
  recomputed.
* Boxes with a non-positive area get rT = rB = 0 and are labelled single, exactly
  as the notebook's ``np.maximum(area, 1e-9)`` would.
"""
import logging

import numpy as np
import pandas as pd

from tracklab.pipeline.videolevel_module import VideoLevelModule

log = logging.getLogger(__name__)

MODES = ("all", "conf", "tracked", "tracked_or_conf")


def label_single_frame(boxes_ltwh, tracked, conf, thr_target, thr_other,
                       contam_mode, conf_thr_other):
    """One frame. Returns (single[bool], rT, rB, trigger_pos) with trigger_pos the
    0-based position of the box that fired the rule, or -1.

    Same arithmetic as the splitter notebook's ``label_single`` (inter/area with
    the diagonal zeroed, excluded contaminators' columns zeroed)."""
    n = len(boxes_ltwh)
    single = np.ones(n, dtype=bool)
    r_t = np.zeros(n, dtype=np.float32)
    r_b = np.zeros(n, dtype=np.float32)
    trig = np.full(n, -1, dtype=np.int64)
    if n < 2:
        return single, r_t, r_b, trig
    b = np.asarray(boxes_ltwh, dtype=np.float64)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(x2[:, None], x2[None, :]) - np.maximum(x1[:, None], x1[None, :]), 0, None)
    ih = np.clip(np.minimum(y2[:, None], y2[None, :]) - np.maximum(y1[:, None], y1[None, :]), 0, None)
    inter = iw * ih
    np.fill_diagonal(inter, 0.0)
    keep = None
    if contam_mode == "tracked":
        keep = np.asarray(tracked, dtype=bool)
    elif contam_mode == "tracked_or_conf":
        keep = np.asarray(tracked, dtype=bool) | (np.asarray(conf, dtype=np.float64) > conf_thr_other)
    elif contam_mode == "conf" and conf_thr_other > 0:
        keep = np.asarray(conf, dtype=np.float64) > conf_thr_other
    if keep is not None and not keep.all():
        inter[:, ~keep] = 0.0          # these boxes cannot make another box multi
    area = np.maximum(b[:, 2] * b[:, 3], 1e-9)
    ratio_t = inter / area[:, None]    # inter / area(T)
    ratio_b = inter / area[None, :]    # inter / area(B)
    rt = ratio_t.max(axis=1)
    rb = ratio_b.max(axis=1)
    r_t[:] = rt
    r_b[:] = rb
    single[:] = (rt <= thr_target) & (rb < thr_other)
    # the box that fired: rule T first (rT > thr_target), else rule B
    fired_t = rt > thr_target
    fired_b = ~fired_t & (rb >= thr_other)
    trig[fired_t] = ratio_t[fired_t].argmax(axis=1)
    trig[fired_b] = ratio_b[fired_b].argmax(axis=1)
    return single, r_t, r_b, trig


class CropFilter(VideoLevelModule):
    """Per-detection single/multi label; see the module docstring."""

    input_columns = ["bbox_ltwh", "image_id", "track_id"]
    output_columns = ["crop_single", "crop_rT", "crop_rB", "crop_trigger"]

    def __init__(self, cfg, device=None, tracking_dataset=None, **kwargs):
        super().__init__()
        self.thr_target = float(cfg.thr_target)
        self.thr_other = float(cfg.thr_other)
        self.contam_mode = str(cfg.contam_mode)
        self.conf_thr_other = float(getattr(cfg, "conf_thr_other", 0.0))
        if self.contam_mode not in MODES:
            raise ValueError(f"[crop_filter] contam_mode={self.contam_mode!r}; "
                             f"expected one of {MODES}")
        if self.contam_mode in ("conf", "tracked_or_conf") and self.conf_thr_other <= 0:
            raise ValueError(f"[crop_filter] contam_mode={self.contam_mode!r} needs "
                             f"conf_thr_other > 0 (got {self.conf_thr_other})")
        log.info(f"[crop_filter] rT <= {self.thr_target}, rB < {self.thr_other}, "
                 f"contaminators: {self.contam_mode}"
                 + (f" (conf > {self.conf_thr_other})"
                    if self.contam_mode in ("conf", "tracked_or_conf") else ""))

    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        n = len(detections)
        single = np.ones(n, dtype=bool)
        r_t = np.zeros(n, dtype=np.float32)
        r_b = np.zeros(n, dtype=np.float32)
        trigger = np.full(n, np.nan)
        if n:
            tracked_all = detections["track_id"].notna().to_numpy()
            conf_all = (detections["bbox_conf"].to_numpy(dtype=np.float64)
                        if "bbox_conf" in detections.columns else np.zeros(n))
            index = detections.index.to_numpy()
            pos_of = pd.Series(np.arange(n), index=detections.index)
            for _, grp in detections.groupby("image_id", sort=False):
                pos = pos_of.loc[grp.index].to_numpy()
                boxes = np.stack([np.asarray(v, dtype=np.float64) for v in grp["bbox_ltwh"]])
                s, rt, rb, tg = label_single_frame(
                    boxes, tracked_all[pos], conf_all[pos], self.thr_target,
                    self.thr_other, self.contam_mode, self.conf_thr_other)
                single[pos] = s
                r_t[pos] = rt
                r_b[pos] = rb
                fired = tg >= 0
                if fired.any():
                    trigger[pos[fired]] = index[pos[tg[fired]]]
        out = detections.copy()
        out["crop_single"] = single
        out["crop_rT"] = r_t
        out["crop_rB"] = r_b
        out["crop_trigger"] = trigger
        n_tracked = int(out["track_id"].notna().sum())
        n_single_tracked = int((out["crop_single"] & out["track_id"].notna()).sum())
        log.info(f"[crop_filter] {n} detections, {n_tracked} tracked; single among "
                 f"tracked: {n_single_tracked} ({n_single_tracked / max(n_tracked, 1):.1%})")
        return out
