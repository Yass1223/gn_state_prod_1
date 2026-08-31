"""Verbatim reference code from splitter-plus-merger-final.ipynb.

Part 1: ``cosine_dist_to_centers``, ``contiguity_fix`` and ``split_tracklet`` from the
"Core functions" cell. Part 2: ``_unit``, ``fragment_stats``, ``_dist``,
``merge_sequence``, ``pass1_conflicts``, ``pass2_place`` from the "merge functions"
cell, and the per-sequence body of ``run_full`` as ``run_full_one``. Kept unchanged
(other than dropping the evaluation-only functions) as the reference the port in
``sn_gamestate/track/split_merge.py`` is compared to by ``tests/test_split_merge.py``.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


# ---------- Part 1: splitter ----------
def cosine_dist_to_centers(E, centers):
    # E (n,d) unit vectors, centers (k,d); returns (n,k) cosine distance
    c = centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-9)
    return 1.0 - E @ c.T

def contiguity_fix(frames, labels, min_run):
    """Post-process: cut each cluster into time-contiguous runs (by row order); runs shorter than
    min_run that sit inside another cluster's span are reassigned to the neighbouring cluster."""
    lab = labels.copy(); n = len(lab)
    if n == 0: return lab
    changed = True; it = 0
    while changed and it < 10:
        changed = False; it += 1
        # runs in row order
        starts = [0] + [i for i in range(1, n) if lab[i] != lab[i - 1]]
        ends = starts[1:] + [n]
        for s, e in zip(starts, ends):
            if e - s >= min_run: continue
            left = lab[s - 1] if s > 0 else None
            right = lab[e] if e < n else None
            if left is not None and left == right and left != lab[s]:
                lab[s:e] = left; changed = True
            elif left is None and right is not None and (e - s) < min_run:
                lab[s:e] = right; changed = True
            elif right is None and left is not None and (e - s) < min_run:
                lab[s:e] = left; changed = True
    return lab

def split_tracklet(E, single, frames, conf, eps, min_samples, noise="attach",
                   use_conf_mask=False, conf_thr=0.3, contiguity_min_run=0, eps_mode="global", eps_k=5, eps_q=0.9, D=None):
    """E: (n,512) unit embeddings of one tracklet, row order == frame order.
    Returns cluster label per row (0..k-1), k, n_noise_single."""
    n = len(E)
    labels = np.zeros(n, dtype=np.int64)
    core = single & (np.linalg.norm(E, axis=1) > 1e-6)
    if use_conf_mask: core &= (conf > conf_thr)
    ci = np.where(core)[0]
    n_noise = 0
    if len(ci) >= max(2, min_samples):
        if D is None:
            Dc = np.clip(1.0 - E[ci] @ E[ci].T, 0.0, 2.0).astype(np.float64)
        else:
            Dc = D[np.ix_(ci, ci)]
        if eps_mode == "kdist":
            # per-tracklet eps: quantile of the k-th nearest cosine distance; eps acts as a multiplier
            Dk = Dc.copy(); np.fill_diagonal(Dk, np.inf)
            kk = min(eps_k, len(ci) - 1)
            kd = np.sort(Dk, axis=1)[:, kk - 1]
            eps_use = max(float(np.quantile(kd, eps_q)) * eps, 1e-4)
        else:
            eps_use = eps
        lab = DBSCAN(eps=eps_use, min_samples=min_samples, metric="precomputed").fit_predict(Dc)
        k = int(lab.max()) + 1 if lab.max() >= 0 else 0
        if k == 0:
            labels[:] = 0; return labels, 1, int((lab < 0).sum())
        n_noise = int((lab < 0).sum())
        centers = np.stack([E[ci][lab == c].mean(axis=0) for c in range(k)])
        if noise == "attach" and n_noise:
            lab[lab < 0] = cosine_dist_to_centers(E[ci][lab < 0], centers).argmin(axis=1)
        elif noise == "own" and n_noise:
            lab[lab < 0] = np.arange(k, k + n_noise); k += n_noise
            centers = np.stack([E[ci][lab == c].mean(axis=0) for c in range(k)])
        labels[ci] = lab
        rest = np.where(~core)[0]
        if len(rest):
            labels[rest] = cosine_dist_to_centers(E[rest], centers).argmin(axis=1)
        if contiguity_min_run > 0 and k > 1:
            labels = contiguity_fix(frames, labels, contiguity_min_run)
            # relabel compactly
            u, labels = np.unique(labels, return_inverse=True); k = len(u)
        return labels, k, n_noise
    return labels, 1, 0


# ---------- Part 2: merger ----------
def _unit(E):
    """fp32 unit rows; ok=False where the norm is ~0 (those rows carry no signal)."""
    X = np.asarray(E, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    ok = n[:, 0] > 1e-6
    U = np.zeros_like(X)
    U[ok] = X[ok] / n[ok]
    return U, ok


def fragment_stats(rec):
    """Per fragment: all row positions, clean row positions, sum of unit vectors and
    count over clean rows, and the set of frames those clean rows occupy."""
    U, ok = _unit(rec["E"])
    fr, sg, fz = rec["frag"], rec["single"], rec["frames"]
    stats = {}
    order = np.argsort(fr, kind="stable")
    bounds = np.flatnonzero(np.r_[True, fr[order][1:] != fr[order][:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        rows = order[a:b]
        srows = rows[sg[rows] & ok[rows]]
        st = dict(rows=rows, n_single=int(len(srows)))
        if len(srows):
            st["sum"] = U[srows].sum(axis=0).astype(np.float64)
            st["n"] = int(len(srows))
            st["frames"] = set(int(x) for x in fz[srows])
        else:
            st["sum"], st["n"], st["frames"] = None, 0, set()
        stats[int(fr[rows[0]])] = st
    return stats, U, ok


def _dist(Ti, Tj):
    return 1.0 - float((Ti["sum"] / Ti["n"]) @ (Tj["sum"] / Tj["n"]))


def merge_sequence(rec, tau):
    """Plain agglomerative merging of one sequence's fragments.
    Returns (traj_of_row, cents, info); rows of fragments with no clean detection get -1."""
    stats, U, ok = fragment_stats(rec)
    main = [f for f, s in stats.items() if s["n_single"] > 0]
    T = [dict(frags=[f], sum=stats[f]["sum"].copy(), n=stats[f]["n"],
              frames=set(stats[f]["frames"])) for f in main]
    k = len(T)
    alive = np.ones(k, dtype=bool)
    D = np.full((k, k), np.inf)
    for i in range(k):
        for j in range(i + 1, k):
            if not (T[i]["frames"] & T[j]["frames"]):
                D[i, j] = D[j, i] = _dist(T[i], T[j])
    n_merges = 0
    while alive.sum() > 1:
        sub = D.copy()
        sub[~alive, :] = np.inf
        sub[:, ~alive] = np.inf
        i, j = np.unravel_index(np.argmin(sub), sub.shape)
        if not np.isfinite(sub[i, j]) or sub[i, j] > tau:
            break
        T[i]["frags"] += T[j]["frags"]
        T[i]["sum"] += T[j]["sum"]
        T[i]["n"] += T[j]["n"]
        T[i]["frames"] |= T[j]["frames"]
        alive[j] = False
        D[j, :] = np.inf
        D[:, j] = np.inf
        for m in range(k):
            if m != i and alive[m]:
                v = np.inf if (T[i]["frames"] & T[m]["frames"]) else _dist(T[i], T[m])
                D[i, m] = D[m, i] = v
        n_merges += 1
    traj_of_row = np.full(len(rec["frag"]), -1, dtype=np.int64)
    cents = {}
    t = 0
    for i in np.where(alive)[0]:
        for f in T[i]["frags"]:
            traj_of_row[stats[f]["rows"]] = t
        cents[t] = (T[i]["sum"].copy(), T[i]["n"])
        t += 1
    info = dict(n_fragments=len(stats), n_ordinary=len(main),
                n_allmulti=len(stats) - len(main), n_traj=t, n_merges=n_merges)
    return traj_of_row, cents, info


def pass1_conflicts(rec, traj_of_row, cents):
    """One detection per frame per trajectory. Clean detections always win; among
    conflicting overlapping detections the nearest to the trajectory mean stays and
    the rest are set aside. Extra clean detections in one frame are not expected
    (counted if seen; nearest kept, others set aside)."""
    U, ok = _unit(rec["E"])
    sg, fz = rec["single"], rec["frames"]
    pool, n_disc, n_anom = [], 0, 0
    df = pd.DataFrame({"t": traj_of_row, "f": fz, "pos": np.arange(len(fz))})
    df = df[df["t"] >= 0]
    for (t, f), g in df.groupby(["t", "f"]):
        if len(g) < 2:
            continue
        rows = g["pos"].values
        s, gh = rows[sg[rows]], rows[~sg[rows]]
        m = cents[t][0] / cents[t][1]

        def _nearest(c):
            d = np.where(ok[c], 1.0 - U[c] @ m, np.inf)
            return c[int(np.argmin(d))]

        if len(s) >= 1:
            if len(s) > 1:
                keep = _nearest(s)
                extra = [r for r in s if r != keep]
                pool += extra
                n_anom += len(extra)
            pool += list(gh)
            n_disc += len(gh)
        else:
            keep = _nearest(gh)
            drop = [r for r in gh if r != keep]
            pool += drop
            n_disc += len(drop)
    for r in pool:
        traj_of_row[r] = -1
    return np.array(sorted(pool), dtype=np.int64), dict(discarded=n_disc, single_anomaly=n_anom)


def pass2_place(rec, traj_of_row, cents, pool_rows):
    """Place set-aside detections, ascending by distance to the nearest trajectory mean
    whose frame slot is free; occupancy counts every assigned detection. No free
    admissible trajectory -> stays -1."""
    U, ok = _unit(rec["E"])
    fz = rec["frames"]
    pool = np.asarray(sorted(set(int(x) for x in pool_rows)), dtype=np.int64)
    tids = sorted(cents)
    if len(pool) == 0 or len(tids) == 0:
        return dict(placed=0, unassigned=int(len(pool)))
    C = np.stack([cents[t][0] / cents[t][1] for t in tids]).astype(np.float32)
    occupied = set(zip(traj_of_row[traj_of_row >= 0].tolist(), fz[traj_of_row >= 0].tolist()))
    Dm = np.full((len(pool), len(tids)), np.inf)
    okp = ok[pool]
    Dm[okp] = 1.0 - U[pool[okp]] @ C.T
    for r, p in enumerate(pool):
        for c, t in enumerate(tids):
            if (t, int(fz[p])) in occupied:
                Dm[r, c] = np.inf
    placed = 0
    while np.isfinite(Dm).any():
        r, c = np.unravel_index(np.argmin(Dm), Dm.shape)
        p, t = int(pool[r]), int(tids[c])
        traj_of_row[p] = t
        occupied.add((t, int(fz[p])))
        placed += 1
        Dm[r, :] = np.inf
        Dm[np.where(fz[pool] == fz[p])[0], c] = np.inf
    return dict(placed=placed, unassigned=int(len(pool) - placed))


def run_full_one(rec, tau):
    """The per-sequence body of the notebook's ``run_full`` (merge + pass 1 + pass 2)."""
    t, c, info = merge_sequence(rec, tau)
    pool, r1 = pass1_conflicts(rec, t, c)
    allmulti_rows = np.where((t < 0) & ~np.isin(np.arange(len(t)), pool))[0]
    r2 = pass2_place(rec, t, c, np.concatenate([pool, allmulti_rows]))
    return t, info, r1, r2
