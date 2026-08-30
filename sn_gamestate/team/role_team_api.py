"""Role, team cluster and team side per tracklet (``role_team`` stage).

Replaces the prtreid role vote, the k-means team clustering and the mean-position
side labelling with the notebook's rule chain (``sn_gamestate.team.rules``). Per
sequence:

1. tracklet table ``D`` — one row per final ``track_id``: positions from
   ``bbox_pitch`` of the rows on the stride grid (``n`` = their count), the median
   ``team_embedding`` over the sampled crops labelled ``crop_single`` (fallback:
   the ``MIN_CROPS`` lowest ``crop_rT`` crops);
2. ``rules.run_sequence(D, params)``;
3. columns on every tracked row: ``role`` in {player, goalkeeper, referee};
   ``team_cluster`` (0/1 for outfield players, NaN otherwise); ``team`` in
   {left, right} for players and goalkeepers, None for referees.

Every tracked row receives a role (the GS encoder asserts a known role on every
exported detection). Untracked rows keep NaN and are dropped by the encoder.

Multi crops are ignored here, as in the notebook. The jersey stage uses every crop.

Sidecar ``<audit_dir>/<sequence>.json``: per tracklet the role, reason (``why``),
team, ``n``, single-crop count and fallback, outlier flags, ``z``; per sequence
the parameters that ran, ``s_ok``, the DBSCAN eps, the naming cues and the chosen
left cluster, plus counts. The run audit reads it.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tracklab.pipeline.videolevel_module import VideoLevelModule

from sn_gamestate.team import rules
from sn_gamestate.team.team_embed_api import frame_index, sequence_name

log = logging.getLogger(__name__)

TEAM_NAMES = {0: "left", 1: "right"}


def _pitch_xy(bp):
    if isinstance(bp, dict):
        return float(bp.get("x_bottom_middle", np.nan)), float(bp.get("y_bottom_middle", np.nan))
    return np.nan, np.nan


class RoleTeamAssignment(VideoLevelModule):
    input_columns = ["track_id", "image_id", "bbox_pitch", "team_embedding", "crop_single", "crop_rT"]
    output_columns = ["role", "team_cluster", "team"]

    def __init__(self, cfg, device=None, tracking_dataset=None, **kwargs):
        super().__init__()
        params = dict(cfg.params) if getattr(cfg, "params", None) is not None else {}
        self.params = rules.check_params(params)
        self.stride = int(getattr(cfg, "pos_stride", rules.POS_STRIDE))
        self.crops_per_track = int(getattr(cfg, "crops_per_track", rules.CROPS_PER_TRK))
        self.audit_dir = Path(str(cfg.audit_dir)) if getattr(cfg, "audit_dir", None) else None
        if self.audit_dir:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"[role_team] params {self.params}")

    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        seq = sequence_name(metadatas)
        out = detections.copy()
        out["role"] = None
        out["team_cluster"] = np.nan
        out["team"] = None
        record = dict(sequence=seq, params=self.params, stride=self.stride, tracklets=0,
                      tracklets_off_grid=0, tracklets_no_embedding=0, per_tracklet=[], sequence_level={})
        tracked = out.dropna(subset=["track_id"])
        if len(tracked) == 0:
            log.warning(f"[role_team] {seq}: no tracked detection")
            self._write(record)
            return out
        fidx = frame_index(metadatas)
        rows, no_emb = [], []
        for tid, grp in tracked.groupby("track_id"):
            frames = fidx.reindex(grp["image_id"].to_numpy()).to_numpy().astype(float)
            if np.isnan(frames).any():
                raise RuntimeError(f"[role_team] {seq}: detection image_id without frame metadata")
            pos_rows, crop_rows, off_grid = rules.sample_tracklet_rows(frames, self.stride, self.crops_per_track)
            record["tracklets"] += 1
            record["tracklets_off_grid"] += int(off_grid)
            pos = grp.iloc[pos_rows]
            xy = np.array([_pitch_xy(b) for b in pos["bbox_pitch"]], dtype=float).reshape(-1, 2)
            crops = grp.iloc[crop_rows]
            have = [i for i, e in enumerate(crops["team_embedding"]) if isinstance(e, np.ndarray) and e.size]
            if not have:
                no_emb.append(tid)
                continue
            crops = crops.iloc[have]
            rows.append(rules.tracklet_row(
                tid, xy[:, 0], xy[:, 1], len(pos_rows), np.stack(crops["team_embedding"].to_numpy()),
                crops["crop_single"].to_numpy(bool), crops["crop_rT"].to_numpy(float)))
        record["tracklets_no_embedding"] = len(no_emb)
        if no_emb:
            # A tracked tracklet without any embedded crop cannot be placed by the
            # rules; it is labelled player with no team so the export stays valid,
            # and the audit reports it (FAIL: the embed stage must cover every tracklet).
            log.error(f"[role_team] {seq}: {len(no_emb)} tracklet(s) without team_embedding "
                      f"-> player, no team: {no_emb[:8]}")
            for tid in no_emb:
                out.loc[out["track_id"] == tid, "role"] = "player"
        if rows:
            D = pd.DataFrame(rows)
            R = rules.run_sequence(D, self.params)
            for j, tid in enumerate(D["track"]):
                role = rules.ROLE_NAMES[int(R["role"][j])]
                team = int(R["team"][j])
                sel = out["track_id"] == tid
                out.loc[sel, "role"] = role
                out.loc[sel, "team"] = TEAM_NAMES.get(team) if role != "referee" else None
                if role == "player" and team >= 0:
                    out.loc[sel, "team_cluster"] = float(team)
                record["per_tracklet"].append(dict(
                    track_id=float(tid), role=role, why=str(R["why"][j]), team=TEAM_NAMES.get(team),
                    n=int(D.n[j]), n_single=int(D.n_single[j]), filt_fallback=bool(D.filt_fallback[j]),
                    mx=_f(D.mx[j]), sx=_f(D.sx[j]), my=_f(D.my[j]), sy=_f(D.sy[j]), q75=_f(D.q75[j]),
                    out_rule=bool(R["out_rule"][j]), out_db=bool(R["out_db"][j]),
                    confirmed=bool(R["confirmed"][j]), gk_candidate=bool(R["gk_c"][j]), z=_f(R["z"][j])))
            roles = [r["role"] for r in record["per_tracklet"]]
            record["sequence_level"] = dict(
                s_ok=bool(R["s_ok"]), named_left_cluster=R["named"], dbscan_eps=_f(R.get("eps")),
                cues={k: (None if v is None else int(v)) for k, v in R["cues"].items()},
                n_player=roles.count("player"), n_goalkeeper=roles.count("goalkeeper"),
                n_referee=roles.count("referee"),
                n_left=sum(1 for r in record["per_tracklet"] if r["team"] == "left" and r["role"] != "referee"),
                n_right=sum(1 for r in record["per_tracklet"] if r["team"] == "right" and r["role"] != "referee"),
                distance_median=_f(R.get("m")), distance_mad=_f(R.get("s")))
            log.info(f"[role_team] {seq}: {len(D)} tracklets -> {roles.count('player')} players, "
                     f"{roles.count('goalkeeper')} goalkeepers, {roles.count('referee')} referees; "
                     f"left cluster {R['named']}, cues {R['cues']}")
        self._write(record)
        return out

    def _write(self, record):
        if self.audit_dir:
            (self.audit_dir / f"{record['sequence']}.json").write_text(json.dumps(record, indent=2, default=str))


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None
