"""Off-pitch tracklet gate (``pitch_gate`` stage).

The detector is a single-class person model and no upstream stage removes a
detection, so a tracklet standing on the bench, behind the goal or among the
photographers is exported as an athlete and receives a role from ``role_team``
like everybody else. Pitch coordinates exist only after ``calibration``
(``bbox_pitch``), so this stage runs directly after it and BEFORE
``tracklet_split``: every later stage groups by ``track_id``, a gated tracklet
must be invisible to all of them (splitting, team embedding/clustering, jersey
recognition, refinement, role/side rules, voting, evaluation export, radar),
and gating first means the splitter never works on off-pitch tracklets. The
splitter rewrites ``track_id`` afterwards and keeps this stage's output per
row in ``track_id_presplit``.

Rule, per final ``track_id``: the mean of the projected bottom-middle points
(``bbox_pitch['x_bottom_middle']`` / ``['y_bottom_middle']``, metres, pitch
centre at the origin) over the tracklet's rows that carry a finite projection.
The tracklet is off-pitch iff

    |mean_x| > PITCH_HALF_LEN + margin_m   or   |mean_y| > PITCH_HALF_WID + margin_m

with the 105 x 68 m pitch of ``sn_gamestate.team.rules`` and ``margin_m`` from the
config. The mean over the whole tracklet is used rather than a per-frame test so
that projection error at the far touchline cannot flicker single frames of an
on-pitch player. A tracklet without any finite projection is kept (the gate has
no evidence) and counted for the audit.

Outputs, on every row:

    track_id_pregate    the ``track_id`` the stage received (NaN on untracked rows)
    pitch_mean_x/y      the tracklet mean position (NaN on untracked rows and on
                        tracklets without a projection)
    pitch_gate_offpitch True on the rows of an off-pitch tracklet (the rule's
                        outcome, recorded whether or not the gate is enabled)
    track_id            unchanged when ``enabled`` is false; NaN on the rows of an
                        off-pitch tracklet when ``enabled`` is true. Rows are never
                        deleted (no stage of this pipeline removes a detection).

Untracked rows stay NaN throughout. With ``enabled: false`` the stage is a no-op
on ``track_id``; the columns and the sidecar are still written so the state and
the audit have the same shape either way.

Sidecar ``<audit_dir>/<sequence>.json``: the switch and margin that ran, the pitch
dimensions used, tracklet and row counts (received, with/without a projection,
off-pitch, gated) and a per-tracklet breakdown. The run audit recomputes the
means and the rule from the detections, checks them against the columns and the
sidecar, and checks that ``track_id`` was (or was not) changed exactly as the
switch says.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tracklab.pipeline.videolevel_module import VideoLevelModule

from sn_gamestate.team.rules import PITCH_HALF_LEN, PITCH_HALF_WID

log = logging.getLogger(__name__)


def sequence_name(metadatas: pd.DataFrame) -> str:
    """Same rule as the audit, tracklet_split and team_embed (``SNGS-xxx`` from the
    frame path when available, else ``video_id``), so the sidecar name matches."""
    if len(metadatas) and "file_path" in metadatas.columns:
        return Path(str(metadatas["file_path"].iloc[0])).parent.parent.name
    if len(metadatas) and "video_id" in metadatas.columns:
        return str(metadatas["video_id"].iloc[0])
    return "unknown"


def pitch_xy(bp):
    """(x, y) of the bottom-middle projection, NaN when absent."""
    if isinstance(bp, dict):
        try:
            return float(bp.get("x_bottom_middle", np.nan)), float(bp.get("y_bottom_middle", np.nan))
        except (TypeError, ValueError):
            return np.nan, np.nan
    return np.nan, np.nan


def tracklet_mean_position(bbox_pitch_values):
    """Mean (x, y) over the entries with a finite projection, and their count.
    (NaN, NaN, 0) when no entry has one."""
    xy = np.array([pitch_xy(b) for b in bbox_pitch_values], dtype=float).reshape(-1, 2)
    ok = np.isfinite(xy).all(axis=1)
    if not ok.any():
        return np.nan, np.nan, 0
    m = xy[ok].mean(axis=0)
    return float(m[0]), float(m[1]), int(ok.sum())


def is_off_pitch(mean_x, mean_y, margin_m,
                 half_len=PITCH_HALF_LEN, half_wid=PITCH_HALF_WID):
    """The gate rule. False when the mean is not finite (no evidence -> keep)."""
    if not (np.isfinite(mean_x) and np.isfinite(mean_y)):
        return False
    return bool(abs(mean_x) > half_len + margin_m or abs(mean_y) > half_wid + margin_m)


def gate_tracklets(detections, margin_m, half_len=PITCH_HALF_LEN, half_wid=PITCH_HALF_WID):
    """Per-row mean position and off-pitch flag, aligned to ``detections.index``.

    Returns (mean_x, mean_y, off_pitch, per_tracklet) where per_tracklet is a list
    of dicts (track_id, n_rows, n_positions, mean_x, mean_y, off_pitch). Rows
    without a track_id get NaN / False."""
    n = len(detections)
    mean_x = pd.Series(np.nan, index=detections.index, dtype=float)
    mean_y = pd.Series(np.nan, index=detections.index, dtype=float)
    off = pd.Series(False, index=detections.index, dtype=bool)
    per = []
    if n == 0 or "track_id" not in detections.columns:
        return mean_x, mean_y, off, per
    has_pitch = "bbox_pitch" in detections.columns
    tracked = detections[detections["track_id"].notna()]
    for tid, grp in tracked.groupby("track_id", sort=True):
        if has_pitch:
            mx, my, k = tracklet_mean_position(grp["bbox_pitch"].to_numpy())
        else:
            mx, my, k = np.nan, np.nan, 0
        o = is_off_pitch(mx, my, margin_m, half_len, half_wid)
        mean_x.loc[grp.index] = mx
        mean_y.loc[grp.index] = my
        off.loc[grp.index] = o
        per.append(dict(track_id=float(tid), n_rows=int(len(grp)), n_positions=int(k),
                        mean_x=None if not np.isfinite(mx) else round(mx, 4),
                        mean_y=None if not np.isfinite(my) else round(my, 4),
                        off_pitch=bool(o)))
    return mean_x, mean_y, off, per


class PitchGate(VideoLevelModule):
    """Per-tracklet off-pitch gate on the mean projected position; see the module docstring."""

    input_columns = ["track_id", "bbox_pitch"]
    output_columns = ["track_id", "track_id_pregate", "pitch_gate_offpitch",
                      "pitch_mean_x", "pitch_mean_y"]

    def __init__(self, cfg, device=None, tracking_dataset=None, **kwargs):
        super().__init__()
        self.enabled = bool(getattr(cfg, "enabled", True))
        self.margin_m = float(getattr(cfg, "margin_m", 0.0))
        if not np.isfinite(self.margin_m) or self.margin_m < 0:
            raise ValueError(f"[pitch_gate] margin_m must be a finite value >= 0, got {self.margin_m}")
        self.audit_dir = Path(str(cfg.audit_dir)) if getattr(cfg, "audit_dir", None) else None
        if self.audit_dir:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"[pitch_gate] enabled={self.enabled}, margin {self.margin_m} m around "
                 f"{2 * PITCH_HALF_LEN:g} x {2 * PITCH_HALF_WID:g} m")

    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        seq = sequence_name(metadatas)
        out = detections.copy()
        if "track_id" not in out.columns:
            raise RuntimeError(f"[pitch_gate] {seq}: track_id column missing - the track "
                               f"stage must run before pitch_gate")
        if "bbox_pitch" not in out.columns:
            raise RuntimeError(f"[pitch_gate] {seq}: bbox_pitch column missing - the calibration "
                               f"stage must run before pitch_gate")
        mean_x, mean_y, off, per = gate_tracklets(out, self.margin_m)
        out["track_id_pregate"] = out["track_id"].astype(float)
        out["pitch_mean_x"] = mean_x
        out["pitch_mean_y"] = mean_y
        out["pitch_gate_offpitch"] = off
        n_tracked_before = int(out["track_id"].notna().sum())
        rows_off = int(off.sum())
        if self.enabled and rows_off:
            out.loc[off, "track_id"] = np.nan
        n_tracked_after = int(out["track_id"].notna().sum())

        n_off = sum(1 for p in per if p["off_pitch"])
        n_nopos = sum(1 for p in per if p["n_positions"] == 0)
        record = dict(
            sequence=seq, enabled=self.enabled, margin_m=self.margin_m,
            pitch=dict(half_len=PITCH_HALF_LEN, half_wid=PITCH_HALF_WID),
            tracklets=len(per), tracklets_with_position=len(per) - n_nopos,
            tracklets_without_position=n_nopos, tracklets_off_pitch=n_off,
            tracklets_gated=n_off if self.enabled else 0,
            rows_tracked_before=n_tracked_before, rows_off_pitch=rows_off,
            rows_gated=n_tracked_before - n_tracked_after, rows_tracked_after=n_tracked_after,
            per_tracklet=per)
        log.info(f"[pitch_gate] {seq}: {len(per)} tracklets, {n_off} off-pitch "
                 f"({rows_off} rows), {n_nopos} without a projection (kept); "
                 f"{'gated' if self.enabled else 'gate disabled, nothing changed'}: "
                 f"{record['rows_gated']} rows lost their track_id")
        self._write(record)
        return out

    def _write(self, record):
        if self.audit_dir is None:
            return
        try:
            (self.audit_dir / f"{record['sequence']}.json").write_text(
                json.dumps(record, indent=2, default=str))
        except Exception as exc:                       # never break a run over telemetry
            log.warning(f"[pitch_gate] could not write the audit sidecar: {exc}")
