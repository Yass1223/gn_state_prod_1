"""Linear interpolation of tracklet gaps (official BoT-SORT ``tools/interpolation.py``).

The official BoT-SORT release ships DTI (disconnected track interpolation) as a
separate offline step that the GTA-Link reference (sjc042/gta-link) does not
include, so this pipeline had neither. It fills short gaps inside a finished
tracklet by linearly interpolating the box between the detections that bracket
the gap, which recovers detections lost to occlusion and raises DetA/recall
without touching association.

Semantics (from the official script)
------------------------------------
Per final ``track_id``, walk the tracklet's detections in temporal order; for
each consecutive pair whose frame gap ``dt`` satisfies ``1 < dt < n_dti``, insert
``dt - 1`` rows whose ``bbox_ltwh`` is the linear blend of the bracketing boxes.
Only tracklets with at least ``n_min`` real detections are interpolated. Two
deliberate departures from the official script, both specified for this pipeline:
``n_min`` is inclusive (``>= n_min``, the script uses ``>``), and interpolated
rows carry the mean ``bbox_conf`` of their two endpoints rather than a hard 1.0,
so a synthesized box never outranks the real detections around it.

Frames are ordered by the *rank* of ``image_id`` in the video's frame list rather
than by raw ``image_id`` arithmetic, and every synthesized row is placed on an
``image_id`` that actually exists in ``metadatas``. Downstream stages resolve a
frame path by ``image_id``, so a synthesized row must never point at a frame that
is not there.

What downstream stages require of a synthesized row (audited, 2026-08-06)
-------------------------------------------------------------------------
Pipeline order is ``bbox_detector -> reid -> track -> gta_link -> [here] ->
calibration -> jersey_number_detect -> tracklet_agg -> team -> team_side``.
``reid`` runs BEFORE ``track``, so a row created here can never have the columns
``reid`` produces (``embeddings``, ``visibility_scores``, ``body_masks``,
``role_detection``, ``role_confidence``).

* ``calibration`` (``BroadTrackCalibration``) — needs ``bbox_ltwh``, ``image_id``.
  Satisfied; synthesized rows are projected to ``bbox_pitch`` like any other.
* ``jersey_number_detect`` (``JNGsrTrackletRecognizer``) — needs ``track_id``,
  ``bbox_ltwh``, ``image_id``, ``role_detection``. The role filter uses
  ``grp["role_detection"].mode()``, which skips NaN, so the tracklet's real
  detections still decide the role and the synthesized crops are ordinary image
  regions. Satisfied.
* ``tracklet_agg`` (``MajorityVoteTracklet``) — votes per ``track_id`` and writes
  the winner to every row of the tracklet, so synthesized rows receive ``role``
  and ``jersey_number`` from their tracklet. Satisfied.
* ``team`` (``TrackletTeamClustering``) — **NOT satisfied.** It runs
  ``np.mean(np.vstack(group.embeddings.values), axis=0)`` over each tracklet.
  With one embedding-less row in the group, ``np.vstack`` raises
  ``ValueError: all the input array dimensions except for the concatenation axis
  must match exactly, but along dimension 1, the array at index 0 has size 512
  and the array at index 1 has size 1``.
* ``team_side`` — needs ``team_cluster``, ``bbox_pitch``, ``role``; all derived
  per tracklet. Satisfied.

Consequence, and why this module still lives in the pipeline
------------------------------------------------------------
The tracking-metrics path is unaffected: ``scripts/reference_metrics.py
--skip gsr jersey_number calibration`` needs only ``track_id``, ``bbox_ltwh`` and
``image_id``, and a state that stops after this stage never reaches ``team``.
That is exactly the scoring path the GTA/interpolation tuning uses, so the stage
can be swept in place with ``enabled: true`` and no other change.

Turning it on in the FULL pipeline additionally requires ``team`` to tolerate
rows without ``embeddings`` (drop them from the tracklet mean, or carry the
nearest real embedding forward). That is a separate decision and is deliberately
NOT made here; enabling the flag logs a warning naming the constraint.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tracklab.pipeline.videolevel_module import VideoLevelModule

log = logging.getLogger(__name__)

# Columns a synthesized row can legitimately carry. Everything else stays NaN/None
# on purpose — see the audit in the module docstring.
_CARRIED = ("track_id", "video_id", "category_id")


def frame_ranks(metadatas: pd.DataFrame):
    """``(image_id -> 0-based rank, rank -> image_id)`` in true temporal order.

    Ordered by frame filename when available (``000001.jpg`` ... is the canonical
    SoccerNet order) and by ``image_id`` otherwise. Raw ``image_id`` differences
    are NOT used as frame gaps: they are only guaranteed to be ordered, not
    contiguous.
    """
    if "file_path" in metadatas.columns:
        keys = metadatas["file_path"].map(lambda p: Path(str(p)).name)
    else:
        keys = pd.Series(metadatas.index, index=metadatas.index)
    ordered = list(keys.sort_values().index)
    return {img: i for i, img in enumerate(ordered)}, ordered


def interpolate_detections(detections: pd.DataFrame, metadatas: pd.DataFrame,
                           n_dti: int = 25, n_min: int = 5) -> pd.DataFrame:
    """Return ``detections`` plus the synthesized gap rows (originals untouched)."""
    rank_of, id_at = frame_ranks(metadatas)

    work = detections[detections["track_id"].notna()].copy()
    work["_rank"] = work["image_id"].map(rank_of)
    unmapped = int(work["_rank"].isna().sum())
    if unmapped:
        log.warning(
            f"[interpolation] {unmapped} detection(s) reference an image_id that is "
            f"not in this video's metadata; they cannot anchor an interpolation."
        )
        work = work[work["_rank"].notna()]

    next_index = (int(detections.index.max()) + 1) if len(detections) else 0
    new_rows, n_gaps = [], 0

    for tid, grp in work.groupby("track_id"):
        if len(grp) < n_min:
            continue
        grp = grp.sort_values("_rank")
        ranks = grp["_rank"].astype(int).to_numpy()
        boxes = np.stack([np.asarray(b, dtype=float) for b in grp["bbox_ltwh"]])
        confs = (grp["bbox_conf"].to_numpy(dtype=float)
                 if "bbox_conf" in grp.columns else np.full(len(grp), np.nan))

        for i in range(1, len(ranks)):
            dt = ranks[i] - ranks[i - 1]
            if not (1 < dt < n_dti):
                continue
            n_gaps += 1
            lo_box, hi_box = boxes[i - 1], boxes[i]
            conf = float(np.nanmean([confs[i - 1], confs[i]]))
            template = grp.iloc[i - 1]
            for j in range(1, int(dt)):
                row = {c: template[c] for c in _CARRIED if c in grp.columns}
                row["track_id"] = tid
                row["image_id"] = id_at[ranks[i - 1] + j]
                row["bbox_ltwh"] = lo_box + (hi_box - lo_box) * (j / dt)
                row["bbox_conf"] = conf
                row["interpolated"] = True
                new_rows.append(pd.Series(row, name=next_index))
                next_index += 1

    out = detections.copy()
    out["interpolated"] = False
    if not new_rows:
        log.info("[interpolation] no gap satisfied "
                 f"1 < dt < {n_dti} on a tracklet of >= {n_min} detections")
        return out

    added = pd.DataFrame(new_rows)
    out = pd.concat([out, added], axis=0)
    out["interpolated"] = out["interpolated"].fillna(False).astype(bool)
    log.info(
        f"[interpolation] filled {n_gaps} gap(s) with {len(added)} synthesized "
        f"detection(s) (n_dti={n_dti}, n_min={n_min}); "
        f"{len(detections)} -> {len(out)} rows"
    )
    return out


class LinearInterpolation(VideoLevelModule):
    """Offline DTI as a pipeline stage, directly after ``gta_link``.

    ``enabled: false`` is a true no-op on the detections; it only stamps the
    ``interpolated`` column (all False) so the module's output contract holds.
    """

    input_columns = ["track_id", "bbox_ltwh", "image_id"]
    output_columns = ["interpolated"]

    def __init__(self, cfg, device=None, tracking_dataset=None):
        self.cfg = cfg
        self.enabled = bool(getattr(cfg, "enabled", False))
        self.n_dti = int(getattr(cfg, "n_dti", 25))
        self.n_min = int(getattr(cfg, "n_min", 5))
        if self.enabled:
            log.warning(
                "[interpolation] enabled: synthesized rows carry no `embeddings`, "
                "and TrackletTeamClustering vstacks that column per tracklet - the "
                "`team` stage will raise unless it is made to tolerate them. Safe "
                "for tracking-only states (reference_metrics.py --skip gsr "
                "jersey_number calibration). See the module docstring."
            )

    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        if not self.enabled or len(detections) == 0 \
                or "track_id" not in detections.columns:
            out = detections.copy()
            out["interpolated"] = False
            return out
        return interpolate_detections(detections, metadatas,
                                      n_dti=self.n_dti, n_min=self.n_min)
