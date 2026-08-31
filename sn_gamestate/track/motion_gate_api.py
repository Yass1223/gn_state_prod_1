"""Motion-continuity gate as a pipeline stage (``motion_gate``).

Runs directly after ``split_merge`` (final ``track_id``) and before ``calibration``.
Switchable with ``cfg.enabled``. For every final trajectory, a detection that is
not in motion continuity with at least ``min_support`` other detections of the
same trajectory -- judged in camera-motion-compensated image coordinates -- is
DISABLED: its ``track_id`` becomes NaN (the pipeline's convention for a detection
no later stage should use), the original id is kept in
``track_id_pre_motion_gate`` and the row is flagged ``motion_gate_disabled``.

Camera motion: the per-frame 2x3 SOF warps recorded by the track stage in its
diagnostics sidecar (``audit/track/<video>.json``, fields tx/ty/a00/a01/a10/a11
per frame). The same warps that steered the tracker steer this gate; nothing is
re-estimated. If the gate is enabled and the sidecar is absent or predates the
full-warp record, the stage FAILS LOUDLY instead of silently judging motion in
un-compensated coordinates. A frame individually missing from the sidecar
contributes an identity step and is counted.

Trajectories with at most ``min_support`` detections are skipped whole (the
criterion is unsatisfiable there) and counted. Roles do not exist yet at this
point of the pipeline, so the gate applies to every trajectory.

With ``enabled: false`` the stage writes its columns (all-false flags, NaN
pre-gate ids), writes its sidecar with ``enabled: false`` and changes nothing --
a true no-op the audit can still verify.

Audit sidecar: ``<audit_dir>/<sequence>.json`` with the settings that ran, the
warp coverage, and the per-trajectory disabled counts; checked by the ``audit``
stage against the config and the state.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tracklab.pipeline.videolevel_module import VideoLevelModule

from sn_gamestate.track import motion_gate as mg

log = logging.getLogger(__name__)


def sequence_name(metadatas: pd.DataFrame) -> str:
    """Same rule as the audit and the other stages (SNGS-xxx from the frame path
    when available, else video_id), so the sidecar name matches."""
    if len(metadatas) and "file_path" in metadatas.columns:
        return Path(str(metadatas["file_path"].iloc[0])).parent.parent.name
    if len(metadatas) and "video_id" in metadatas.columns:
        return str(metadatas["video_id"].iloc[0])
    return "unknown"


def load_warps(track_sidecar_dir, video_id, seq):
    """{image_id(str): 2x3 warp} from the tracker sidecar, or None when the sidecar
    is absent/unreadable. Raises RuntimeError when frames exist but none carries the
    full warp (a pre-full-warp sidecar)."""
    if track_sidecar_dir is None:
        return None
    for name in (str(video_id), seq):
        p = Path(track_sidecar_dir) / f"{name}.json"
        if p.is_file():
            break
    else:
        return None
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.error(f"[motion_gate] could not read tracker sidecar '{p}': {exc}")
        return None
    warps, n_frames = {}, 0
    for fr in data.get("frames", []):
        n_frames += 1
        keys = ("a00", "a01", "tx", "a10", "a11", "ty")
        if all(k in fr for k in keys):
            warps[str(fr.get("image_id"))] = np.array(
                [[fr["a00"], fr["a01"], fr["tx"]], [fr["a10"], fr["a11"], fr["ty"]]],
                dtype=np.float64)
    if n_frames and not warps:
        raise RuntimeError(
            f"[motion_gate] tracker sidecar '{p}' records {n_frames} frame(s) but none "
            f"carries the full 2x3 warp (a00/a01/a10/a11/tx/ty) - it predates the "
            f"full-warp record; re-run tracking or disable the motion gate")
    return warps


class MotionGate(VideoLevelModule):
    input_columns = ["track_id", "bbox_ltwh", "image_id"]
    output_columns = ["track_id", "track_id_pre_motion_gate", "motion_gate_disabled"]

    def __init__(self, cfg, device, tracking_dataset=None, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.enabled = bool(cfg.enabled)
        self.min_support = int(cfg.min_support)
        self.window = int(cfg.window)
        self.speed_max_px = float(cfg.speed_max_px)
        self.slack_px = float(cfg.slack_px)
        if self.min_support < 1:
            raise ValueError(f"[motion_gate] min_support must be >= 1, got {self.min_support}")
        if self.window < 1 or self.speed_max_px <= 0 or self.slack_px < 0:
            raise ValueError("[motion_gate] window must be >= 1, speed_max_px > 0, slack_px >= 0")
        self.audit_dir = Path(str(cfg.audit_dir)) if getattr(cfg, "audit_dir", None) else None
        if self.audit_dir:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
        d = getattr(cfg, "track_sidecar_dir", None)
        self.track_sidecar_dir = Path(str(d)) if d else None
        log.info(f"[motion_gate] enabled {self.enabled}; min_support {self.min_support}, "
                 f"window {self.window}, speed_max_px {self.speed_max_px}, slack_px {self.slack_px}")

    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        seq = sequence_name(metadatas)
        record = dict(sequence=seq, ran=False,
                      settings=dict(enabled=self.enabled, min_support=self.min_support,
                                    window=self.window, speed_max_px=self.speed_max_px,
                                    slack_px=self.slack_px),
                      inputs=dict(detections=int(len(detections)), tracked=0, tracklets=0,
                                  frames=int(len(metadatas)), warps=0, warp_steps_missing=0),
                      outputs=dict(disabled=0, tracklets_touched=0, trajectories_skipped_short=0))
        out = detections.copy()
        out["track_id_pre_motion_gate"] = np.nan
        out["motion_gate_disabled"] = False

        if len(out) == 0 or "track_id" not in out.columns:
            self._write(record)
            return out
        tracked_mask = out["track_id"].notna()
        record["inputs"]["tracked"] = int(tracked_mask.sum())
        record["inputs"]["tracklets"] = int(out.loc[tracked_mask, "track_id"].nunique())
        if not self.enabled:
            log.info(f"[motion_gate] {seq}: disabled by config; no detection touched")
            self._write(record)
            return out
        if record["inputs"]["tracked"] == 0:
            self._write(record)
            return out

        warps = load_warps(self.track_sidecar_dir, metadatas["video_id"].iloc[0]
                           if "video_id" in metadatas.columns else seq, seq)
        if warps is None:
            raise RuntimeError(
                f"[motion_gate] {seq}: no tracker sidecar under '{self.track_sidecar_dir}' - "
                f"the gate cannot compensate camera motion; re-run tracking with the sidecar "
                f"enabled or set modules.motion_gate.cfg.enabled=false")
        record["inputs"]["warps"] = int(len(warps))

        work = out[tracked_mask]
        frame_order = [int(i) for i in metadatas.sort_index().index]
        # tracker sidecar keys are str(image_id); frame keys here are the int image ids
        warps_by_frame = {}
        for k, v in warps.items():
            try:
                warps_by_frame[int(k)] = v
            except (TypeError, ValueError):
                pass
        tids = work["track_id"].to_numpy(dtype=np.float64)
        if not np.all(tids == np.round(tids)):
            raise RuntimeError(f"[motion_gate] {seq}: non-integer track_id values")
        disabled, rep = mg.gate_video(
            tids.astype(np.int64), work["image_id"].to_numpy(dtype=np.int64),
            np.stack([np.asarray(b, dtype=np.float64)[:4] for b in work["bbox_ltwh"]]),
            frame_order, warps_by_frame,
            self.min_support, self.window, self.speed_max_px, self.slack_px)
        record["ran"] = True
        record["inputs"]["warp_steps_missing"] = rep["warp_steps_missing"]

        idx = work.index[disabled]
        out.loc[idx, "track_id_pre_motion_gate"] = out.loc[idx, "track_id"]
        out.loc[idx, "motion_gate_disabled"] = True
        out.loc[idx, "track_id"] = np.nan
        record["outputs"].update(
            disabled=int(disabled.sum()),
            tracklets_touched=int(sum(1 for t in rep["per_trajectory"] if t["disabled"])),
            trajectories_skipped_short=rep["trajectories_skipped_short"])
        record["per_trajectory"] = rep["per_trajectory"]
        log.info(f"[motion_gate] {seq}: {record['inputs']['tracked']} tracked detections in "
                 f"{record['inputs']['tracklets']} trajectories; disabled "
                 f"{record['outputs']['disabled']} in {record['outputs']['tracklets_touched']} "
                 f"trajector(ies); {rep['trajectories_skipped_short']} short trajectories skipped; "
                 f"{rep['warp_steps_missing']} warp step(s) missing of {len(frame_order) - 1}")
        self._write(record)
        return out

    def _write(self, record):
        if self.audit_dir is None:
            return
        try:
            (self.audit_dir / f"{record['sequence']}.json").write_text(
                json.dumps(record, indent=2, default=str))
        except Exception as exc:                       # never break a run over telemetry
            log.warning(f"[motion_gate] could not write the audit sidecar: {exc}")
