"""Reference for the equivalence test: the ORIGINAL notebook code, verbatim.

Cells 11 (tracklets) and 13 (DEFAULT_PARAMS, knee_eps, kmeans2, run_sequence,
five_class) of role-team-gt-tracks-3.ipynb, preceded only by the constants those
cells read from the notebook's config cell. Not used by the pipeline; the port that
runs there is sn_gamestate/team/rules.py, and test_rules_equivalence.py asserts the
two produce identical outputs.
"""
import numpy as np, pandas as pd
from sklearn.cluster import KMeans, DBSCAN
SEED=42; MIN_CROPS=3; FILTER_RT, FILTER_RB = 0.25, 0.40
PITCH_HALF_LEN, PITCH_HALF_WID = 52.5, 34.0
PEN_X, PEN_Y = PITCH_HALF_LEN - 16.5, 20.16
ROLE_OUT, ROLE_GK, ROLE_REF = 0, 1, 2
def load_cache(split, cond):
    cdir = os.path.join(CACHE_DIR, split)
    T = pd.read_csv(os.path.join(cdir, "dets_%s.csv" % cond))
    S = pd.read_csv(os.path.join(cdir, "samples_%s.csv" % cond))
    Z = np.load(os.path.join(cdir, "emb_%s.npz" % cond))
    return T, S, {k: Z[k] for k in Z.files}

def tracklets(T, S, Z, emb_name, rt=FILTER_RT, rb=FILTER_RB):
    """One row per tracklet: cls, n (positions), position stats, median embedding, filter info."""
    single = (S.rT <= rt) & (S.rB < rb)
    rows, embs, roles = [], [], []
    if emb_name == "concat":
        Zs = [Z["emb_osnet_team"]] + ([Z["emb_prtreid"]] if "emb_prtreid" in Z else [])
        Zc = np.concatenate([z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9) for z in Zs], 1)
    else:
        Zc = Z["emb_" + emb_name]
    pos = T.groupby(["sequence", "track"], sort=False)
    for (seq, tr), gS in S.groupby(["sequence", "track"], sort=False):
        gT = pos.get_group((seq, tr))
        sel = np.where(single.loc[gS.index].to_numpy())[0]
        fallback = len(sel) < MIN_CROPS
        if fallback:
            sel = np.argsort(gS.rT.to_numpy())[:MIN_CROPS]
        ii = gS.index.to_numpy()[sel]
        e = np.median(Zc[ii], axis=0); e = e / (np.linalg.norm(e) + 1e-9)
        px = gT.px.to_numpy(float); py = gT.py.to_numpy(float)
        px, py = px[np.isfinite(px)], py[np.isfinite(py)]
        rows.append(dict(sequence=seq, track=tr, cls=int(gT.cls.mode().iloc[0]), n=len(gT),
                         n_single=int(len(sel)), filt_fallback=fallback,
                         mx=px.mean() if len(px) else np.nan, sx=px.std() if len(px) else np.nan,
                         q75=np.percentile(np.abs(px), 75) if len(px) else np.nan,
                         my=py.mean() if len(py) >= 3 else np.nan, sy=py.std() if len(py) >= 3 else np.nan,
                         px_fallback=float(gT.fallback.mean()),
                         induced_px=float(np.nanmedian(np.abs(gT.px - gT.gt_px))) if "gt_px" in gT else 0.0))
        embs.append(e)
        if "role_prt" in Z:
            r = Z["role_prt"][ii]; roles.append(np.nanmean(r, axis=0) if np.isfinite(r).any() else np.full(4, np.nan))
    D = pd.DataFrame(rows); D["emb"] = list(np.stack(embs))
    D["role_prt"] = list(np.stack(roles)) if roles else [None] * len(D)
    return D
DEFAULT_PARAMS = dict(k=2.5, dbscan_min=3, dbscan_scale=1.0, tau_n=10, tau_a=0.9, tau_a_sy=4.0,
                      tau_r=1.5, tau_m=0.15, confirm="outlier",   # confirm in {geometry, outlier}
                      extra_ref="none",      # {none, both, rule, strong}: remaining outlier tracklets -> referee
                      k_strong=4.0,          # 'strong': distance rule at a larger multiple, no DBSCAN agreement needed
                      extra_pool="nogk",     # {nogk, all}: 'all' lets rejected keeper candidates into the referee step
                      centre_confirm="margin",  # {margin, outlier}: 'outlier' = centre referee needs appearance support
                      max_ref=3,             # cap on referees per sequence (annotations show up to 3 officials in view)
                      side_rule="vote")      # {sign, mean, quantile, vote, centroid, keeper, vote2}

def knee_eps(E, kth=3):
    d = 1 - E @ E.T; np.fill_diagonal(d, np.inf)
    nn = np.sort(np.sort(d, axis=1)[:, :kth], axis=1)[:, -1]
    s = np.sort(nn)
    if len(s) < 4: return float(np.median(s)) if len(s) else 0.5
    # knee: point with the largest distance to the chord from first to last
    x = np.linspace(0, 1, len(s)); y = (s - s[0]) / (s[-1] - s[0] + 1e-9)
    dist = np.abs((y[-1] - y[0]) * x - (x[-1] - x[0]) * y + x[-1] * y[0] - y[-1] * x[0])
    return float(s[int(np.argmax(dist))])

KM_CACHE = {}
def kmeans2(E):
    key = (E.shape, E.tobytes()[:4096], float(E.sum()))
    if key not in KM_CACHE:
        KM_CACHE[key] = KMeans(2, n_init=10, random_state=SEED).fit(E)
    return KM_CACHE[key]

def run_sequence(D, P):
    """D: tracklet table for one sequence. Returns per-tracklet dict arrays."""
    n = len(D)
    E = np.stack(D.emb.to_numpy())
    mx, sx, my, sy, q75, nn_ = [D[c].to_numpy(float) for c in ("mx", "sx", "my", "sy", "q75", "n")]
    pred = np.full(n, ROLE_OUT); why = np.array(["player"] * n, dtype=object)
    confirmed = np.zeros(n, bool)
    if n < 2:
        return dict(role=pred, why=why, team=np.full(n, -1), out_rule=np.zeros(n, bool),
                    out_db=np.zeros(n, bool), confirmed=confirmed, named=None, s_ok=False)
    # --- 1. partition ---
    km = kmeans2(E)
    d_all = np.linalg.norm(E[:, None] - km.cluster_centers_[None], axis=2)
    d = d_all.min(1)
    # --- 2. geometry candidates (positive evidence only) ---
    elig = nn_ >= P["tau_n"]
    absmy = np.abs(my); maxmy = np.nanmax(absmy) if np.isfinite(absmy).any() else np.nan
    assist = elig & np.isfinite(my) & (absmy >= P["tau_a"] * maxmy) & (sy <= P["tau_a_sy"])
    a_margin = np.where(np.isfinite(my), absmy / (maxmy + 1e-9) - P["tau_a"], -1.0)
    # one assistant per side (sign of my)
    for sgn in (-1, 1):
        idx = np.where(assist & (np.sign(my) == sgn))[0]
        if len(idx) > 1:
            keep = idx[np.argmax(absmy[idx])]; assist[np.setdiff1d(idx, [keep])] = False
    gk_c = elig & ~assist & np.isfinite(mx) & (q75 >= PEN_X) & (np.abs(my) <= PEN_Y)
    gk_margin = (q75 - PEN_X) / 10
    # --- 3. outlier channels; scale from tracklets passing no geometry rule ---
    base = ~assist & ~gk_c
    ref_d = d[base] if base.sum() >= 3 else d
    m = np.median(ref_d); s = np.median(np.abs(ref_d - m)); s_ok = s >= 0.05 * m
    out_rule = (d > m + P["k"] * s) if s_ok else np.zeros(n, bool)
    out_db = np.zeros(n, bool)
    if n >= 4:
        eps = max(1e-3, knee_eps(E) * P["dbscan_scale"])
        lab = DBSCAN(eps=eps, min_samples=P["dbscan_min"], metric="cosine").fit_predict(E)
        out_db = lab == -1
    is_out = out_rule | out_db
    def accept(j, margin, tag):
        ok = is_out[j] if P["confirm"] == "outlier" else True
        if not ok and margin >= P["tau_m"]: ok = True
        if ok: confirmed[j] = is_out[j]
        return ok
    # --- 4a assistant ---
    for j in np.where(assist)[0]:
        if accept(j, a_margin[j], "assistant"): pred[j] = ROLE_REF; why[j] = "assistant"
        else: why[j] = "assistant_rejected"
    # --- 4b goalkeeper: one per half, deepest ---
    for sgn in (-1, 1):
        side = np.where(gk_c & (np.sign(mx) == sgn) & (pred == ROLE_OUT))[0]
        if len(side) == 0: continue
        best = side[np.argmax(np.abs(mx[side]))]
        if accept(best, gk_margin[best], "gk"): pred[best] = ROLE_GK; why[best] = "gk_extreme"
        else: why[best] = "gk_rejected"
        for j in side:
            if j == best: continue
            if is_out[j] and abs(mx[j]) >= abs(mx[best]) - 4.0 and pred[best] == ROLE_GK:
                pred[j] = ROLE_GK; why[j] = "gk_confirmed"; confirmed[j] = True
            else: why[j] = "gk_rejected"
    # --- 4c centre referee: largest normalised sx + sy above tau_r, one per sequence ---
    rest = np.where((pred == ROLE_OUT) & elig & np.isfinite(sx))[0]
    if len(rest) >= 6:
        msx = np.nanmedian(sx[rest]); msy = np.nanmedian(sy[rest]) if np.isfinite(sy[rest]).any() else np.nan
        score = sx[rest] / (msx + 1e-9) + (np.nan_to_num(sy[rest], nan=msy if np.isfinite(msy) else 0) / (msy + 1e-9) if np.isfinite(msy) else 0)
        top = rest[np.argmax(score)]
        # score is ~2 for a typical player; tau_r is the margin above that
        r_margin = (score.max() - 2.0) - P["tau_r"]
        c_ok = is_out[top] if P.get("centre_confirm", "margin") == "outlier" else accept(top, r_margin / 2, "centre")
        if r_margin >= 0 and c_ok:
            pred[top] = ROLE_REF; why[top] = "centre"; confirmed[top] = is_out[top]
        elif r_margin >= 0:
            why[top] = "centre_rejected"
    # --- 4d. extra referee channel (tuned on valid): remaining tracklets that appearance flags as
    # outliers and geometry does not place in a penalty area. Motivated by the valid diagnostic of
    # the first run: 124 of 127 referees were outliers by the distance rule, but the one-centre /
    # one-assistant-per-side rules let at most three through and the rest fell back to player.
    z = (d - m) / (s + 1e-9) if s_ok else np.full(n, -np.inf)     # distance score in units of s
    if P.get("extra_ref", "none") != "none":
        if P["extra_ref"] == "both":     flag = out_rule & out_db
        elif P["extra_ref"] == "rule":   flag = out_rule
        else:                            flag = (out_rule & out_db) | (z > P.get("k_strong", 4.0))   # strong
        pool = (pred == ROLE_OUT) & elig
        if P.get("extra_pool", "nogk") == "nogk": pool &= ~gk_c
        cand = np.where(flag & pool)[0]
        room = int(P.get("max_ref", 3)) - int((pred == ROLE_REF).sum())
        if room > 0 and len(cand):
            # most extreme appearance first (largest d)
            for j in cand[np.argsort(-d[cand])][:room]:
                pred[j] = ROLE_REF; why[j] = "ref_outlier"; confirmed[j] = True
    # --- 5. refit on players only, assign by nearest centroid ---
    players = np.where(pred == ROLE_OUT)[0]
    team = np.full(n, -1)
    named = None; cues_out = {}
    if len(players) >= 2:
        km2 = kmeans2(np.ascontiguousarray(E[players]))
        lab = km2.labels_
        # --- 6. naming: which cluster is the left team ---
        # sign: per-tracklet sign vote on mean x. mean: cluster mean of mean x. quantile: compare the
        # retreated end of each cluster (mean of the 10th and 90th percentiles of tracklet mean x):
        # when both teams sit in one half, the team attacking that goal keeps its defenders behind,
        # so its low quantile is further left. vote: majority of the three, mean as tie-break.
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
                v = mxp[lab == k]; v = v[np.isfinite(v)]
                q.append(np.mean([np.percentile(v, 10), np.percentile(v, 90)]) if len(v) >= 2 else np.nan)
            return None if (not np.isfinite(q).all() or q[0] == q[1]) else int(np.argmin(q))
        def cue_centroid():
            # centroid of each cluster's player positions relative to the centroid of all players
            g = np.nanmean(mxp)
            c = [np.nanmean(mxp[lab == k]) - g for k in (0, 1)]
            return None if (not np.isfinite(c).all() or c[0] == c[1]) else int(np.argmin(c))
        gks_ = np.where(pred == ROLE_GK)[0]
        def cue_keeper():
            # exactly one keeper: the cluster whose players sit closer to him (mean tracklet position)
            # is his team; his half then fixes the sides. Uses tracklet means, not per-frame positions.
            if len(gks_) != 1 or not np.isfinite(mx[gks_[0]]): return None
            kx, ky = mx[gks_[0]], my[gks_[0]]
            if not np.isfinite(ky): ky = 0.0
            dist = []
            for k in (0, 1):
                sel = lab == k
                dd = np.hypot(mxp[sel] - kx, np.nan_to_num(my[players][sel], nan=0.0) - ky)
                dist.append(np.nanmean(dd) if len(dd) else np.nan)
            if not np.isfinite(dist).all() or dist[0] == dist[1]: return None
            own = int(np.argmin(dist))
            return own if kx < 0 else 1 - own
        rule = P.get("side_rule", "vote")
        if rule == "centroid": left = cue_centroid(); left = cue_mean() if left is None else left
        elif rule == "keeper": left = cue_keeper(); left = cue_quant() if left is None else left; left = cue_mean() if left is None else left
        elif rule == "vote2":
            cues = [c for c in (cue_keeper(), cue_quant(), cue_mean()) if c is not None]
            left = None if not cues else (0 if cues.count(0) > cues.count(1) else 1 if cues.count(1) > cues.count(0) else cue_quant())
        elif rule == "sign":   left = cue_sign();  left = cue_mean() if left is None else left
        elif rule == "mean":   left = cue_mean()
        elif rule == "quantile": left = cue_quant(); left = cue_mean() if left is None else left
        else:
            cues = [c for c in (cue_sign(), cue_quant(), cue_mean()) if c is not None]
            left = None if not cues else (0 if cues.count(0) > cues.count(1) else 1 if cues.count(1) > cues.count(0) else cue_mean())
        gks = np.where(pred == ROLE_GK)[0]
        if left is None and len(gks) == 1 and np.isfinite(mx[gks[0]]):
            dk = np.linalg.norm(km2.cluster_centers_ - E[gks[0]], axis=1)
            own = int(np.argmin(dk))
            left = own if mx[gks[0]] < 0 else 1 - own
        if left is None: left = 0
        named = left
        cues_out = dict(sign=cue_sign(), mean=cue_mean(), quantile=cue_quant(), centroid=cue_centroid(), keeper=cue_keeper())
        team[players] = np.where(lab == left, 0, 1)
    for j in np.where(pred == ROLE_GK)[0]:
        team[j] = 0 if (np.isfinite(mx[j]) and mx[j] < 0) else 1
    return dict(role=pred, why=why, team=team, out_rule=out_rule, out_db=out_db, confirmed=confirmed,
                named=named, s_ok=s_ok, z=z, gk_c=gk_c, cues=cues_out)

def five_class(role, team):
    if role == ROLE_REF: return 2
    if role == ROLE_GK: return 3 if team == 0 else 4
    return 0 if team == 0 else 1
