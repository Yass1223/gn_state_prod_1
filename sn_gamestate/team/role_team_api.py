"""Role and team SIDE per finished trajectory (``role_team`` stage).

Runs AFTER ``traj_refine``, on the pipeline's final trajectories -- the whole
point of its position: roles and sides are decided once, where the temporal and
positional evidence is strongest, and never gate the merge. Inputs per
trajectory: the team CLUSTER id assigned by ``team_embed`` and propagated by
``traj_refine``; since batch 11 the TEAM CLUSTERING IS RECOMPUTED HERE, on the
final trajectories (kmeans2_threshold over per-trajectory descriptors), and the
recomputed ``team_cluster`` / ``team_cluster_nearest`` overwrite the columns
(constant per final trajectory). Pitch positions (``bbox_pitch``) sampled on the
stride grid. In this architecture "appearance outlier" MEANS "unclustered" --
the clustering already made that call.

Per sequence:

1. Geometry statistics per trajectory: mean/std of x and y, the 75th percentile
   of |x| (goal-depth cue) and the sampled position count, exactly the
   notebook's statistics (``rules.sample_tracklet_rows`` grid).
2. ASSISTANT referees: eligible trajectories (n >= tau_n) hugging a touchline
   (|mean y| >= tau_a * max|mean y|, y-std <= tau_a_sy), one per side;
   accepted when unclustered (``confirm: outlier``) or by margin (tau_m).
3. GOALKEEPERS: one per half -- the deepest eligible trajectory with
   q75(|x|) >= PEN_X and |mean y| <= PEN_Y, accepted when unclustered or by
   margin; an unclustered second candidate within 4 m of the same depth is
   confirmed too.
4. MAIN referee (one per sequence): among the remaining UNCLUSTERED
   trajectories, the candidates whose sampled y-range stays inside the band
   the trajectory means span, symmetrically (2.14):

       max(y_ref) <= 0.9 * max_i(mean_y_i)   and
       min(y_ref) >= 0.9 * min_i(mean_y_i)

   (max/min over the means of ALL trajectories with y positions). Among the
   candidates, the one nearest the assistant referees (mean distance of mean
   positions); with no assistants found, the most central candidate
   (smallest |mean y|).
5. Everything else is a PLAYER. Side: the two team clusters are named left and
   right by the sequence-level cue chain over the player trajectories' mean x
   (sign/quantile/mean vote, keeper and centroid cues available via
   ``side_rule``); a clustered player takes its cluster's side; an unclustered
   player takes the side of its NEAREST centroid (``team_cluster_nearest``,
   modal over its rows -- a flagged fallback); a player with no embedding at
   all takes the side of its mean-x half. Goalkeepers take the side of their
   half; referees have no side.

Columns written on every tracked row: ``role`` in {player, goalkeeper,
referee}; ``team`` in {left, right} (None for referees). ``team_cluster``
stays as ``traj_refine`` left it. Sidecar
``<audit_dir>/<sequence>.json``: per-trajectory role/why/team/cluster/
fallback flags, sequence-level naming cues, the (2.14) band, and counts; the
run audit reads it.
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
DEFAULTS = dict(tau_n=6, tau_a=0.72, tau_a_sy=6.0, tau_m=0.15,
                confirm="outlier", side_rule="vote", band=0.9, gk_depth_m=4.0,
                cluster_method="kmeans2_threshold", outlier_k=3.25)


def _pitch_xy(bp):
    if isinstance(bp, dict):
        return float(bp.get("x_bottom_middle", np.nan)), float(bp.get("y_bottom_middle", np.nan))
    return np.nan, np.nan


def _cues(mxp, lab, myp=None, gk=None):
    """The notebook's side-naming cues over player trajectories: which cluster
    (0/1) is the LEFT team. ``mxp``/``myp`` mean x/y per player trajectory,
    ``lab`` its cluster id (0/1), ``gk`` the (mx, my) of the single goalkeeper
    when exactly one exists (the positional part of the notebook's keeper
    cue -- the appearance part is gone with the embeddings)."""
    def sign():
        v = [np.nansum(np.sign(mxp[lab == c])) for c in (0, 1)]
        return None if (not np.isfinite(v).all() or abs(v[0] - v[1]) < 1) else int(np.argmin(v))

    def mean():
        c = [np.nanmean(mxp[lab == k]) if (lab == k).any() else np.nan for k in (0, 1)]
        return None if (not np.isfinite(c).all() or c[0] == c[1]) else int(np.argmin(c))

    def quant():
        q = []
        for k in (0, 1):
            v = mxp[lab == k]
            v = v[np.isfinite(v)]
            q.append(np.mean([np.percentile(v, 10), np.percentile(v, 90)]) if len(v) >= 2 else np.nan)
        return None if (not np.isfinite(q).all() or q[0] == q[1]) else int(np.argmin(q))

    def keeper():
        if gk is None or not np.isfinite(gk[0]):
            return None
        kx = gk[0]
        ky = gk[1] if np.isfinite(gk[1]) else 0.0
        yy = myp if myp is not None else np.zeros_like(mxp)
        dist = []
        for k in (0, 1):
            sel = lab == k
            dd = np.hypot(mxp[sel] - kx, np.nan_to_num(yy[sel], nan=0.0) - ky)
            dist.append(np.nanmean(dd) if sel.any() else np.nan)
        if not np.isfinite(dist).all() or dist[0] == dist[1]:
            return None
        own = int(np.argmin(dist))
        return own if kx < 0 else 1 - own

    return dict(sign=sign(), mean=mean(), quantile=quant(), keeper=keeper())


class RoleTeamAssignment(VideoLevelModule):
    input_columns = ["track_id", "image_id", "bbox_pitch", "team_embedding"]
    output_columns = ["role", "team", "team_cluster", "team_cluster_nearest"]

    def __init__(self, cfg, device=None, tracking_dataset=None, **kwargs):
        super().__init__()
        p = dict(DEFAULTS)
        p.update(dict(cfg.params) if getattr(cfg, "params", None) is not None else {})
        self.params = p
        self.stride = int(getattr(cfg, "pos_stride", rules.POS_STRIDE))
        self.audit_dir = Path(str(cfg.audit_dir)) if getattr(cfg, "audit_dir", None) else None
        if self.audit_dir:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"[role_team] per-trajectory roles and sides AFTER traj_refine; "
                 f"params {self.params}; unclustered == appearance outlier")

    # ------------------------------------------------------------------ main --
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        seq = sequence_name(metadatas)
        P = self.params
        out = detections.copy()
        out["role"] = None
        out["team"] = None
        record = dict(sequence=seq, params=dict(P), stride=self.stride,
                      trajectories=0, per_trajectory=[], sequence_level={})
        tracked = out.dropna(subset=["track_id"])
        if len(tracked) == 0:
            log.warning(f"[role_team] {seq}: no tracked detection")
            self._write(record)
            return out
        fidx = frame_index(metadatas)

        # --- 1. per-trajectory geometry + labels ---------------------------
        T = []
        for tid, grp in tracked.groupby("track_id"):
            frames = fidx.reindex(grp["image_id"].to_numpy()).to_numpy().astype(float)
            if np.isnan(frames).any():
                raise RuntimeError(f"[role_team] {seq}: detection image_id without frame metadata")
            pos_rows, _, _ = rules.sample_tracklet_rows(frames, self.stride, 1)
            pos = grp.iloc[pos_rows]
            xy = np.array([_pitch_xy(b) for b in pos["bbox_pitch"]], dtype=float).reshape(-1, 2)
            px, py = xy[:, 0], xy[:, 1]
            px, py = px[np.isfinite(px)], py[np.isfinite(py)]
            es = [e for e in grp["team_embedding"]
                  if isinstance(e, np.ndarray) and e.size]
            if es:
                z = np.median(np.asarray(es, dtype=np.float32), axis=0)
                z = z / (np.linalg.norm(z) + 1e-9)
            else:
                z = None
            T.append(dict(
                tid=float(tid), n=int(len(pos_rows)),
                mx=px.mean() if len(px) else np.nan,
                sx=px.std() if len(px) else np.nan,
                q75=np.percentile(np.abs(px), 75) if len(px) else np.nan,
                my=py.mean() if len(py) >= 3 else np.nan,
                sy=py.std() if len(py) >= 3 else np.nan,
                ymax=py.max() if len(py) else np.nan,
                ymin=py.min() if len(py) else np.nan,
                emb=z, cluster=None, nearest=None))
        record["trajectories"] = len(T)
        n = len(T)
        # --- 1b. TEAM CLUSTERING, recomputed on the FINAL trajectories -----
        # (batch 11): descriptor = L2-normalised median of the trajectory's
        # embedded crops (all fragments merged in), 2-means + the robust MAD
        # threshold -- the same kmeans2_threshold rule as team_embed, now on
        # whole trajectories. team_embed's fragment-level clustering keeps
        # feeding traj_refine; the labels HERE are the ones roles/sides use
        # and the ones written to the output columns.
        have = [j for j in range(n) if T[j]["emb"] is not None]
        k_out = float(self.params["outlier_k"])
        clus = dict(method=str(self.params["cluster_method"]), outlier_k=k_out,
                    trajectories=n, embedded=len(have),
                    no_embedding=n - len(have), clustered=0,
                    unclustered_threshold=0, s_ok=None, m=None, s=None,
                    sizes=[0, 0])
        if len(have) >= 2:
            E = np.stack([T[j]["emb"] for j in have]).astype(np.float32)
            km = rules.kmeans2(E)
            d_all = np.linalg.norm(E[:, None] - km.cluster_centers_[None], axis=2)
            d = d_all.min(1)
            lab = d_all.argmin(1)
            m_c = float(np.median(d))
            s_c = float(np.median(np.abs(d - m_c)))
            s_ok = s_c >= 0.05 * m_c
            outl = (d > m_c + k_out * s_c) if s_ok else np.zeros(len(d), bool)
            for i, j in enumerate(have):
                T[j]["nearest"] = float(lab[i])
                T[j]["cluster"] = None if outl[i] else float(lab[i])
            clus.update(clustered=int((~outl).sum()),
                        unclustered_threshold=int(outl.sum()),
                        s_ok=bool(s_ok), m=round(m_c, 6), s=round(s_c, 6),
                        sizes=[int(((lab == c) & ~outl).sum()) for c in (0, 1)])
        record["cluster"] = clus
        mx = np.array([t["mx"] for t in T])
        my = np.array([t["my"] for t in T])
        sy = np.array([t["sy"] for t in T])
        q75 = np.array([t["q75"] for t in T])
        nn_ = np.array([t["n"] for t in T], dtype=float)
        unclustered = np.array([t["cluster"] is None for t in T])
        role = np.array(["player"] * n, dtype=object)
        why = np.array(["player"] * n, dtype=object)

        def accept(j, margin):
            if P["confirm"] != "outlier" or unclustered[j]:
                return True
            return margin >= P["tau_m"]

        # --- 2. assistants --------------------------------------------------
        elig = nn_ >= P["tau_n"]
        absmy = np.abs(my)
        maxmy = np.nanmax(absmy) if np.isfinite(absmy).any() else np.nan
        assist = elig & np.isfinite(my) & (absmy >= P["tau_a"] * maxmy) & (sy <= P["tau_a_sy"])
        a_margin = np.where(np.isfinite(my), absmy / (maxmy + 1e-9) - P["tau_a"], -1.0)
        for sgn in (-1, 1):
            idx = np.where(assist & (np.sign(my) == sgn))[0]
            if len(idx) > 1:
                keep = idx[np.argmax(absmy[idx])]
                assist[np.setdiff1d(idx, [keep])] = False
        for j in np.where(assist)[0]:
            if accept(j, a_margin[j]):
                role[j], why[j] = "referee", "assistant"
            else:
                assist[j] = False
                why[j] = "assistant_rejected"

        # --- 3. goalkeepers -------------------------------------------------
        gk_c = elig & (role == "player") & np.isfinite(mx) & \
            (q75 >= rules.PEN_X) & (np.abs(my) <= rules.PEN_Y)
        gk_margin = (q75 - rules.PEN_X) / 10
        for sgn in (-1, 1):
            side = np.where(gk_c & (np.sign(mx) == sgn) & (role == "player"))[0]
            if len(side) == 0:
                continue
            best = side[np.argmax(np.abs(mx[side]))]
            if accept(best, gk_margin[best]):
                role[best], why[best] = "goalkeeper", "gk_extreme"
            else:
                why[best] = "gk_rejected"
            for j in side:
                if j == best:
                    continue
                if (unclustered[j] and role[best] == "goalkeeper"
                        and abs(mx[j]) >= abs(mx[best]) - P["gk_depth_m"]):
                    role[j], why[j] = "goalkeeper", "gk_confirmed"
                else:
                    why[j] = "gk_rejected"

        # --- 4. MAIN referee: unclustered + inside the symmetric band (2.14)
        #        + nearest to the assistants -------------------------------
        means_y = my[np.isfinite(my)]
        band_hi = P["band"] * float(np.max(means_y)) if len(means_y) else np.nan
        band_lo = P["band"] * float(np.min(means_y)) if len(means_y) else np.nan
        ymax = np.array([t["ymax"] for t in T])
        ymin = np.array([t["ymin"] for t in T])
        cand = np.where(unclustered & (role == "player") & np.isfinite(ymax)
                        & np.isfinite(ymin) & (ymax <= band_hi)
                        & (ymin >= band_lo))[0] if np.isfinite(band_hi) else np.array([], int)
        main_ref = None
        if len(cand):
            a_idx = np.where(assist & (role == "referee"))[0]
            if len(a_idx):
                da = [np.nanmean(np.hypot(mx[j] - mx[a_idx],
                                          np.nan_to_num(my[j], nan=0.0)
                                          - np.nan_to_num(my[a_idx], nan=0.0)))
                      for j in cand]
                main_ref = int(cand[int(np.nanargmin(da))])
            else:
                main_ref = int(cand[int(np.nanargmin(np.abs(my[cand])))])
            role[main_ref], why[main_ref] = "referee", "main_2.14"

        # --- 5. sides: name the clusters, then per-trajectory teams --------
        players = np.where(role == "player")[0]
        lab = np.array([T[j]["cluster"] if T[j]["cluster"] is not None else np.nan
                        for j in range(n)])
        pl = players[np.isfinite(lab[players])]
        gk_idx = np.where(role == "goalkeeper")[0]
        gk_pos = ((mx[gk_idx[0]], my[gk_idx[0]]) if len(gk_idx) == 1 else None)
        cues = (_cues(mx[pl], lab[pl].astype(int), my[pl], gk_pos) if len(pl)
                else dict(sign=None, mean=None, quantile=None, keeper=None))
        if P["side_rule"] == "keeper":
            left = cues["keeper"]
            left = cues["quantile"] if left is None else left
            left = cues["mean"] if left is None else left
        else:                                    # vote (default for other values)
            votes = [c for c in (cues["sign"], cues["quantile"], cues["mean"])
                     if c is not None]
            left = (0 if votes.count(0) > votes.count(1)
                    else 1 if votes.count(1) > votes.count(0)
                    else cues["mean"]) if votes else None
        if left is None:
            left = 0
        team = np.array([None] * n, dtype=object)
        n_fallback_nearest = n_fallback_half = 0
        for j in range(n):
            if role[j] == "referee":
                continue
            if role[j] == "goalkeeper":
                team[j] = "left" if (np.isfinite(mx[j]) and mx[j] < 0) else "right"
                continue
            cl = T[j]["cluster"]
            if cl is None:
                cl = T[j]["nearest"]
                if cl is not None:
                    n_fallback_nearest += 1
                    why[j] = "player_nearest_centroid"
                else:
                    n_fallback_half += 1
                    why[j] = "player_half_fallback"
                    team[j] = "left" if (np.isfinite(mx[j]) and mx[j] < 0) else "right"
                    continue
            team[j] = "left" if int(cl) == left else "right"

        # --- apply + sidecar ------------------------------------------------
        out["team_cluster"] = np.nan
        out["team_cluster_nearest"] = np.nan
        for j, t in enumerate(T):
            sel = out["track_id"] == t["tid"]
            out.loc[sel, "role"] = role[j]
            out.loc[sel, "team"] = team[j]
            if t["cluster"] is not None:
                out.loc[sel, "team_cluster"] = t["cluster"]
            if t["nearest"] is not None:
                out.loc[sel, "team_cluster_nearest"] = t["nearest"]
            record["per_trajectory"].append(dict(
                track_id=t["tid"], role=str(role[j]), why=str(why[j]),
                team=team[j], cluster=t["cluster"], nearest=t["nearest"],
                n=t["n"], mx=_f(t["mx"]), my=_f(t["my"]), sy=_f(t["sy"]),
                q75=_f(t["q75"]), ymax=_f(t["ymax"]), ymin=_f(t["ymin"])))
        roles = [r["role"] for r in record["per_trajectory"]]
        record["sequence_level"] = dict(
            named_left_cluster=int(left), cues=cues,
            band=[_f(band_lo), _f(band_hi)],
            main_referee=(T[main_ref]["tid"] if main_ref is not None else None),
            n_player=roles.count("player"), n_goalkeeper=roles.count("goalkeeper"),
            n_referee=roles.count("referee"),
            n_unclustered=int(unclustered.sum()),
            n_fallback_nearest=n_fallback_nearest, n_fallback_half=n_fallback_half,
            n_left=sum(1 for r in record["per_trajectory"] if r["team"] == "left"),
            n_right=sum(1 for r in record["per_trajectory"] if r["team"] == "right"))
        log.info(f"[role_team] {seq}: {n} trajectories -> "
                 f"{roles.count('player')} players, {roles.count('goalkeeper')} "
                 f"goalkeepers, {roles.count('referee')} referees "
                 f"(main {'found' if main_ref is not None else 'none'}); left "
                 f"cluster {left}, cues {cues}; fallbacks: {n_fallback_nearest} "
                 f"nearest-centroid, {n_fallback_half} half")
        self._write(record)
        return out

    def _write(self, record):
        if self.audit_dir:
            (self.audit_dir / f"{record['sequence']}.json").write_text(
                json.dumps(record, indent=2, default=str))


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None
