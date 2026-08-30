"""Role and team assignment rules on tracklets — the notebook's ``run_sequence``.

Port of ``role-team-gt-tracks-3.ipynb`` (cells 11 and 13): the tracklet table
builder and the per-sequence rule chain. Arithmetic, thresholds, tie-breaks and
branch order are the notebook's; only the input plumbing (pipeline columns instead
of cached CSV/NPZ) and the output form (dict arrays plus a per-tracklet reason)
differ. ``FROZEN_PARAMS`` holds the values the notebook froze on the tuning split
and scored once on test (three-class macro-F1 0.926 on the realistic condition).

Steps per sequence (one tracklet = one row of ``D``):

1. k-means (k=2) on the tracklet descriptors; ``d`` = distance to the nearer centre.
2. position candidates: assistant (across-pitch extreme, low y-spread, one per
   side), goalkeeper (75 % quantile of |x| past the penalty-area depth, |y| inside
   its width; deepest per half, further same-half outliers within 4 m confirmed).
3. appearance outliers: distance rule ``d > m + k*s`` (m, s = median / MAD of d over
   tracklets no position rule proposed; disabled when ``s < 0.05*m``), and DBSCAN
   noise on cosine distance (eps from the knee of the kth-neighbour curve).
4. roles in order: assistant, goalkeeper, centre referee (largest normalised
   movement range above ``tau_r``; needs >= 6 eligible tracklets and, with
   ``centre_confirm=outlier``, appearance support), then the extra referee channel
   (``extra_ref``: both signals, or ``z > k_strong``; pool may include rejected
   keeper candidates; capped at ``max_ref`` per sequence, largest ``d`` first).
   Assistant / goalkeeper acceptance needs an outlier flag, or a position margin
   >= ``tau_m`` (``confirm=outlier``).
5. k-means again on outfield players only -> two clusters.
6. naming (``side_rule``): which cluster is the left team; ``keeper`` = the cluster
   nearer the single goalkeeper is his team and his half fixes the sides, falling
   back to the retreated-end quantile cue, then the cluster mean, then an
   embedding-space keeper cue, then cluster 0.

Positions are the projected bottom-middle points (``bbox_pitch``) at stride
``POS_STRIDE``; ``n`` (the length gate ``tau_n``) counts those sampled positions,
as in the notebook.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans

ROLE_OUT, ROLE_GK, ROLE_REF = 0, 1, 2
ROLE_NAMES = ("player", "goalkeeper", "referee")

SEED = 42
POS_STRIDE = 5          # frames between sampled positions / crops (notebook POS_STRIDE)
CROPS_PER_TRK = 16      # crops embedded per tracklet, evenly spaced in time
MIN_CROPS = 3           # fewer single crops than this -> the MIN_CROPS lowest-rT crops
PITCH_HALF_LEN, PITCH_HALF_WID = 52.5, 34.0
PEN_X, PEN_Y = PITCH_HALF_LEN - 16.5, 20.16   # penalty-area depth (from centre) and half-width

# The notebook's frozen configuration (cell 17 output, "frozen parameters").
FROZEN_PARAMS = dict(
    k=3.25, dbscan_min=4, dbscan_scale=1.5, tau_n=5, tau_a=0.85, tau_a_sy=3.0,
    tau_r=2.0, tau_m=0.30, confirm="outlier", extra_ref="strong", k_strong=4.0,
    extra_pool="all", centre_confirm="outlier", max_ref=3, side_rule="keeper",
)
PARAM_CHOICES = dict(confirm=("geometry", "outlier"),
                     extra_ref=("none", "both", "rule", "strong"),
                     extra_pool=("nogk", "all"),
                     centre_confirm=("margin", "outlier"),
                     side_rule=("sign", "mean", "quantile", "vote", "centroid", "keeper", "vote2"))


def check_params(P):
    """Raise on an unknown key, a missing key, or a categorical value outside the
    notebook's option set. Every value must be one the rules can execute."""
    unknown = set(P) - set(FROZEN_PARAMS)
    missing = set(FROZEN_PARAMS) - set(P)
    if unknown or missing:
        raise ValueError(f"role/team params: unknown {sorted(unknown)}, missing {sorted(missing)}")
    for key, choices in PARAM_CHOICES.items():
        if P[key] not in choices:
            raise ValueError(f"role/team params: {key}={P[key]!r} not in {choices}")
    return {key: (type(FROZEN_PARAMS[key])(P[key]) if not isinstance(FROZEN_PARAMS[key], str) else str(P[key]))
            for key in FROZEN_PARAMS}


# ------------------------------------------------------------- sampling ------

def sample_tracklet_rows(frames, stride=POS_STRIDE, crops=CROPS_PER_TRK):
    """Which rows of one tracklet are used, given its frame indices (0-based).

    Returns (pos_rows, crop_rows): positions of the rows (into ``frames``) that
    carry a pitch position (every ``stride``-th frame, as the notebook's
    ``(frame - 1) % POS_STRIDE == 0`` on 1-based frames) and the <= ``crops`` rows
    among them, evenly spaced in time, whose crops are embedded
    (``np.linspace(0, n - 1, min(crops, n))``, deduplicated, as in the notebook).

    Tracklets with no frame on the stride grid do not exist in the notebook (its
    tables were built on the grid). Here every tracked row must receive a role, so
    such a tracklet falls back to all of its rows; the audit reports how many.
    """
    frames = np.asarray(frames, dtype=np.int64)
    order = np.argsort(frames, kind="stable")
    on_grid = order[(frames[order] % stride) == 0]
    fallback = len(on_grid) == 0
    pos_rows = order if fallback else on_grid
    idx = np.linspace(0, len(pos_rows) - 1, min(crops, len(pos_rows))).astype(int)
    crop_rows = pos_rows[sorted(set(idx.tolist()))]
    return pos_rows, crop_rows, fallback


# ------------------------------------------------------- tracklet table ------

def tracklet_row(track_id, px, py, n_positions, emb_rows, single, r_t):
    """One row of ``D`` from a tracklet's positions and its sampled-crop embeddings.

    ``px``/``py``: pitch coordinates of the sampled positions (NaN when absent);
    ``n_positions``: their count (the notebook's ``n = len(gT)``);
    ``emb_rows``: (m, dim) embeddings of the sampled crops, ``single``/``r_t`` their
    crop labels and overlap ratios. Median descriptor over single crops, falling
    back to the MIN_CROPS lowest-rT crops (notebook ``tracklets()``)."""
    single = np.asarray(single, dtype=bool)
    r_t = np.asarray(r_t, dtype=float)
    sel = np.where(single)[0]
    fallback = len(sel) < MIN_CROPS
    if fallback:
        sel = np.argsort(r_t, kind="stable")[:MIN_CROPS]
    # float32 throughout, as the notebook (median of the float32 crop descriptors)
    e = np.median(np.asarray(emb_rows, dtype=np.float32)[sel], axis=0)
    e = e / (np.linalg.norm(e) + 1e-9)
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    px, py = px[np.isfinite(px)], py[np.isfinite(py)]
    return dict(
        track=track_id, n=int(n_positions), n_single=int(len(sel)), filt_fallback=bool(fallback),
        mx=px.mean() if len(px) else np.nan, sx=px.std() if len(px) else np.nan,
        q75=np.percentile(np.abs(px), 75) if len(px) else np.nan,
        my=py.mean() if len(py) >= 3 else np.nan, sy=py.std() if len(py) >= 3 else np.nan,
        emb=np.asarray(e, dtype=np.float32),
    )


# ------------------------------------------------------------ the rules ------

def knee_eps(E, kth=3):
    d = 1 - E @ E.T
    np.fill_diagonal(d, np.inf)
    nn = np.sort(np.sort(d, axis=1)[:, :kth], axis=1)[:, -1]
    s = np.sort(nn)
    if len(s) < 4:
        return float(np.median(s)) if len(s) else 0.5
    # knee: point with the largest distance to the chord from first to last
    x = np.linspace(0, 1, len(s))
    y = (s - s[0]) / (s[-1] - s[0] + 1e-9)
    dist = np.abs((y[-1] - y[0]) * x - (x[-1] - x[0]) * y + x[-1] * y[0] - y[-1] * x[0])
    return float(s[int(np.argmax(dist))])


def kmeans2(E):
    return KMeans(2, n_init=10, random_state=SEED).fit(E)


def run_sequence(D, P):
    """D: tracklet table for one sequence (columns emb, mx, sx, my, sy, q75, n).
    Returns per-tracklet arrays and sequence-level cues, as the notebook."""
    n = len(D)
    E = np.stack(D.emb.to_numpy())          # float32, as the notebook
    mx, sx, my, sy, q75, nn_ = [D[c].to_numpy(float) for c in ("mx", "sx", "my", "sy", "q75", "n")]
    pred = np.full(n, ROLE_OUT)
    why = np.array(["player"] * n, dtype=object)
    confirmed = np.zeros(n, bool)
    if n < 2:
        return dict(role=pred, why=why, team=np.full(n, -1), out_rule=np.zeros(n, bool),
                    out_db=np.zeros(n, bool), confirmed=confirmed, named=None, s_ok=False,
                    z=np.full(n, -np.inf), gk_c=np.zeros(n, bool), cues={}, eps=None)
    # --- 1. partition ---
    km = kmeans2(E)
    d_all = np.linalg.norm(E[:, None] - km.cluster_centers_[None], axis=2)
    d = d_all.min(1)
    # --- 2. geometry candidates (positive evidence only) ---
    elig = nn_ >= P["tau_n"]
    absmy = np.abs(my)
    maxmy = np.nanmax(absmy) if np.isfinite(absmy).any() else np.nan
    assist = elig & np.isfinite(my) & (absmy >= P["tau_a"] * maxmy) & (sy <= P["tau_a_sy"])
    a_margin = np.where(np.isfinite(my), absmy / (maxmy + 1e-9) - P["tau_a"], -1.0)
    # one assistant per side (sign of my)
    for sgn in (-1, 1):
        idx = np.where(assist & (np.sign(my) == sgn))[0]
        if len(idx) > 1:
            keep = idx[np.argmax(absmy[idx])]
            assist[np.setdiff1d(idx, [keep])] = False
    gk_c = elig & ~assist & np.isfinite(mx) & (q75 >= PEN_X) & (np.abs(my) <= PEN_Y)
    gk_margin = (q75 - PEN_X) / 10
    # --- 3. outlier channels; scale from tracklets passing no geometry rule ---
    base = ~assist & ~gk_c
    ref_d = d[base] if base.sum() >= 3 else d
    m = np.median(ref_d)
    s = np.median(np.abs(ref_d - m))
    s_ok = s >= 0.05 * m
    out_rule = (d > m + P["k"] * s) if s_ok else np.zeros(n, bool)
    out_db = np.zeros(n, bool)
    eps = None
    if n >= 4:
        eps = max(1e-3, knee_eps(E) * P["dbscan_scale"])
        lab = DBSCAN(eps=eps, min_samples=P["dbscan_min"], metric="cosine").fit_predict(E)
        out_db = lab == -1
    is_out = out_rule | out_db

    def accept(j, margin):
        ok = is_out[j] if P["confirm"] == "outlier" else True
        if not ok and margin >= P["tau_m"]:
            ok = True
        if ok:
            confirmed[j] = is_out[j]
        return ok

    # --- 4a assistant ---
    for j in np.where(assist)[0]:
        if accept(j, a_margin[j]):
            pred[j] = ROLE_REF
            why[j] = "assistant"
        else:
            why[j] = "assistant_rejected"
    # --- 4b goalkeeper: one per half, deepest ---
    for sgn in (-1, 1):
        side = np.where(gk_c & (np.sign(mx) == sgn) & (pred == ROLE_OUT))[0]
        if len(side) == 0:
            continue
        best = side[np.argmax(np.abs(mx[side]))]
        if accept(best, gk_margin[best]):
            pred[best] = ROLE_GK
            why[best] = "gk_extreme"
        else:
            why[best] = "gk_rejected"
        for j in side:
            if j == best:
                continue
            if is_out[j] and abs(mx[j]) >= abs(mx[best]) - 4.0 and pred[best] == ROLE_GK:
                pred[j] = ROLE_GK
                why[j] = "gk_confirmed"
                confirmed[j] = True
            else:
                why[j] = "gk_rejected"
    # --- 4c centre referee: largest normalised sx + sy above tau_r, one per sequence ---
    rest = np.where((pred == ROLE_OUT) & elig & np.isfinite(sx))[0]
    if len(rest) >= 6:
        msx = np.nanmedian(sx[rest])
        msy = np.nanmedian(sy[rest]) if np.isfinite(sy[rest]).any() else np.nan
        score = sx[rest] / (msx + 1e-9) + (
            np.nan_to_num(sy[rest], nan=msy if np.isfinite(msy) else 0) / (msy + 1e-9)
            if np.isfinite(msy) else 0)
        top = rest[np.argmax(score)]
        # score is ~2 for a typical player; tau_r is the margin above that
        r_margin = (score.max() - 2.0) - P["tau_r"]
        c_ok = is_out[top] if P.get("centre_confirm", "margin") == "outlier" else accept(top, r_margin / 2)
        if r_margin >= 0 and c_ok:
            pred[top] = ROLE_REF
            why[top] = "centre"
            confirmed[top] = is_out[top]
        elif r_margin >= 0:
            why[top] = "centre_rejected"
    # --- 4d. extra referee channel: remaining appearance outliers ---
    z = (d - m) / (s + 1e-9) if s_ok else np.full(n, -np.inf)
    if P.get("extra_ref", "none") != "none":
        if P["extra_ref"] == "both":
            flag = out_rule & out_db
        elif P["extra_ref"] == "rule":
            flag = out_rule
        else:
            flag = (out_rule & out_db) | (z > P.get("k_strong", 4.0))   # strong
        pool = (pred == ROLE_OUT) & elig
        if P.get("extra_pool", "nogk") == "nogk":
            pool &= ~gk_c
        cand = np.where(flag & pool)[0]
        room = int(P.get("max_ref", 3)) - int((pred == ROLE_REF).sum())
        if room > 0 and len(cand):
            # most extreme appearance first (largest d)
            for j in cand[np.argsort(-d[cand])][:room]:
                pred[j] = ROLE_REF
                why[j] = "ref_outlier"
                confirmed[j] = True
    # --- 5. refit on players only, assign by nearest centroid ---
    players = np.where(pred == ROLE_OUT)[0]
    team = np.full(n, -1)
    named = None
    cues_out = {}
    if len(players) >= 2:
        km2 = kmeans2(np.ascontiguousarray(E[players]))
        lab = km2.labels_
        # --- 6. naming: which cluster is the left team ---
        mxp = mx[players]

        def cue_sign():
            v = [np.nansum(np.sign(mxp[lab == c])) for c in (0, 1)]
            return None if (not np.isfinite(v).all() or abs(v[0] - v[1]) < 1) else int(np.argmin(v))

        def cue_mean():
            c = [np.nanmean(mxp[lab == k]) for k in (0, 1)]
            return None if (not np.isfinite(c).all() or c[0] == c[1]) else int(np.argmin(c))

        def cue_quant():
            q = []
            for k in (0, 1):
                v = mxp[lab == k]
                v = v[np.isfinite(v)]
                q.append(np.mean([np.percentile(v, 10), np.percentile(v, 90)]) if len(v) >= 2 else np.nan)
            return None if (not np.isfinite(q).all() or q[0] == q[1]) else int(np.argmin(q))

        def cue_centroid():
            g = np.nanmean(mxp)
            c = [np.nanmean(mxp[lab == k]) - g for k in (0, 1)]
            return None if (not np.isfinite(c).all() or c[0] == c[1]) else int(np.argmin(c))

        gks_ = np.where(pred == ROLE_GK)[0]

        def cue_keeper():
            # exactly one keeper: the cluster whose players sit closer to him (mean
            # tracklet position) is his team; his half then fixes the sides.
            if len(gks_) != 1 or not np.isfinite(mx[gks_[0]]):
                return None
            kx, ky = mx[gks_[0]], my[gks_[0]]
            if not np.isfinite(ky):
                ky = 0.0
            dist = []
            for k in (0, 1):
                sel = lab == k
                dd = np.hypot(mxp[sel] - kx, np.nan_to_num(my[players][sel], nan=0.0) - ky)
                dist.append(np.nanmean(dd) if len(dd) else np.nan)
            if not np.isfinite(dist).all() or dist[0] == dist[1]:
                return None
            own = int(np.argmin(dist))
            return own if kx < 0 else 1 - own

        rule = P.get("side_rule", "vote")
        if rule == "centroid":
            left = cue_centroid()
            left = cue_mean() if left is None else left
        elif rule == "keeper":
            left = cue_keeper()
            left = cue_quant() if left is None else left
            left = cue_mean() if left is None else left
        elif rule == "vote2":
            cues = [c for c in (cue_keeper(), cue_quant(), cue_mean()) if c is not None]
            left = None if not cues else (0 if cues.count(0) > cues.count(1) else 1 if cues.count(1) > cues.count(0) else cue_quant())
        elif rule == "sign":
            left = cue_sign()
            left = cue_mean() if left is None else left
        elif rule == "mean":
            left = cue_mean()
        elif rule == "quantile":
            left = cue_quant()
            left = cue_mean() if left is None else left
        else:
            cues = [c for c in (cue_sign(), cue_quant(), cue_mean()) if c is not None]
            left = None if not cues else (0 if cues.count(0) > cues.count(1) else 1 if cues.count(1) > cues.count(0) else cue_mean())
        gks = np.where(pred == ROLE_GK)[0]
        if left is None and len(gks) == 1 and np.isfinite(mx[gks[0]]):
            dk = np.linalg.norm(km2.cluster_centers_ - E[gks[0]], axis=1)
            own = int(np.argmin(dk))
            left = own if mx[gks[0]] < 0 else 1 - own
        if left is None:
            left = 0
        named = left
        cues_out = dict(sign=cue_sign(), mean=cue_mean(), quantile=cue_quant(),
                        centroid=cue_centroid(), keeper=cue_keeper())
        team[players] = np.where(lab == left, 0, 1)
    for j in np.where(pred == ROLE_GK)[0]:
        team[j] = 0 if (np.isfinite(mx[j]) and mx[j] < 0) else 1
    return dict(role=pred, why=why, team=team, out_rule=out_rule, out_db=out_db, confirmed=confirmed,
                named=named, s_ok=bool(s_ok), z=z, gk_c=gk_c, cues=cues_out, eps=eps,
                m=float(m), s=float(s))
