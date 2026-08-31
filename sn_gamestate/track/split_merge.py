"""Tracklet splitting and fragment merging on appearance -- the algorithm.

Port of the production part of ``splitter-plus-merger-final.ipynb`` (Part 1
``split_tracklet`` with its default options, Part 2 ``fragment_stats``,
``merge_sequence``, ``pass1_conflicts``, ``pass2_place`` and the driver
``run_full``). Function names, signatures and bodies follow the notebook so the
port can be checked against it line by line; ``tests/test_split_merge.py`` runs
the notebook cells verbatim against this module on random inputs. Everything
the notebook does only for evaluation (ground-truth matching, homogeneity,
pairwise scores, oracles, CSV outputs) is left out.

Inputs, per video (aligned arrays, one entry per TRACKED detection):

    E          (n, d) appearance embeddings (unit vectors; a zero row means the
               detection could not be embedded and carries no signal)
    single     (n,)   bool, the crop filter's ``crop_single`` label
    frames     (n,)   int, the frame key (``image_id``); equality is all that
               matters -- two detections share a frame iff the keys are equal
    track_ids  (n,)   int, the tracker's ``track_id``

Steps
-----
1. Split (``split_tracklet``): per tracklet, DBSCAN with ``eps`` and
   ``min_samples`` on the precomputed cosine-distance matrix of its clean
   detections (``single`` and non-zero). Noise points and the non-clean
   detections are attached to the nearest cluster centroid. A tracklet with
   fewer than ``max(2, min_samples)`` clean detections is one fragment.
2. Merge (``merge_sequence``): agglomerative merging of the fragments that hold
   at least one clean detection. Group distance = 1 minus the dot product of
   the two groups' mean unit vectors over clean detections (equal to the
   average pairwise cosine distance between members). Two groups may merge only
   if their clean frame sets are disjoint. Merging stops when the closest
   admissible pair is farther than ``tau``. Non-clean detections take no part
   in this step; they inherit their fragment's trajectory.
3. Pass 1 (``pass1_conflicts``): one detection per (trajectory, frame). A
   clean detection always wins; among non-clean ones the closest to the
   trajectory mean stays and the rest are set aside.
4. Pass 2 (``pass2_place``): the set-aside detections and every detection of a
   fragment with no clean detection are placed, in ascending order of distance,
   into the nearest trajectory whose frame slot is free. A detection with no
   admissible trajectory stays unassigned (-1).

No spatial or motion constraint is used anywhere.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

# Fragment id of cluster ``lab`` of tracklet ``tid`` is ``tid * FRAG_BASE + lab``; it
# keeps fragments ordered by (track_id, cluster) exactly as the notebook's
# ``(base + tid) * 10000 + lab`` does within one sequence.
FRAG_BASE = 10000


# ------------------------------------------------------------------ split (Part 1)

def cosine_dist_to_centers(E, centers):
    """E (n,d) unit vectors, centers (k,d); returns (n,k) cosine distance."""
    c = centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-9)
    return 1.0 - E @ c.T


def split_tracklet(E, single, eps, min_samples):
    """E: (n,d) float32 embeddings of ONE tracklet, single: (n,) bool.

    Returns (labels, k, n_noise): cluster label per row (0..k-1), the number of
    clusters, and the number of DBSCAN noise points among the clean rows. This
    is the notebook's ``split_tracklet`` with ``noise="attach"``,
    ``use_conf_mask=False``, ``contiguity_min_run=0`` and ``eps_mode="global"``.
    """
    E = np.asarray(E, dtype=np.float32)
    n = len(E)
    labels = np.zeros(n, dtype=np.int64)
    core = np.asarray(single, dtype=bool) & (np.linalg.norm(E, axis=1) > 1e-6)
    ci = np.where(core)[0]
    if len(ci) >= max(2, min_samples):
        Dc = np.clip(1.0 - E[ci] @ E[ci].T, 0.0, 2.0).astype(np.float64)
        lab = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit_predict(Dc)
        k = int(lab.max()) + 1 if lab.max() >= 0 else 0
        if k == 0:
            labels[:] = 0
            return labels, 1, int((lab < 0).sum())
        n_noise = int((lab < 0).sum())
        centers = np.stack([E[ci][lab == c].mean(axis=0) for c in range(k)])
        if n_noise:
            lab[lab < 0] = cosine_dist_to_centers(E[ci][lab < 0], centers).argmin(axis=1)
        labels[ci] = lab
        rest = np.where(~core)[0]
        if len(rest):
            labels[rest] = cosine_dist_to_centers(E[rest], centers).argmin(axis=1)
        return labels, k, n_noise
    return labels, 1, 0


def split_video(E, single, frames, track_ids, eps, min_samples):
    """Split every tracklet of one video. Returns (frag, per_tracklet) where
    ``frag`` is the fragment id per row and ``per_tracklet`` a list of dicts
    (track_id, n, n_single, k, noise) in track_id order."""
    E = np.asarray(E)
    single = np.asarray(single, dtype=bool)
    frames = np.asarray(frames, dtype=np.int64)
    track_ids = np.asarray(track_ids, dtype=np.int64)
    frag = np.zeros(len(E), dtype=np.int64)
    per_tracklet = []
    for tid in np.unique(track_ids):
        idx = np.where(track_ids == tid)[0]
        idx = idx[np.argsort(frames[idx], kind="stable")]
        Ei = np.asarray(E[idx], dtype=np.float32)
        lab, k, nn = split_tracklet(Ei, single[idx], eps, min_samples)
        frag[idx] = int(tid) * FRAG_BASE + lab
        n_core = int((single[idx] & (np.linalg.norm(Ei, axis=1) > 1e-6)).sum())
        per_tracklet.append(dict(track_id=int(tid), n=int(len(idx)),
                                 n_single=int(single[idx].sum()), n_core=n_core,
                                 k=int(k), noise=int(nn)))
    return frag, per_tracklet


# ------------------------------------------------------------------ merge (Part 2)

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
    """Plain agglomerative merging of one video's fragments.
    Returns (traj_of_row, cents, info); rows of fragments with no clean detection get -1.
    ``info["merge_distances"]`` lists the distance of every accepted merge, in order."""
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
    merge_distances = []
    while alive.sum() > 1:
        sub = D.copy()
        sub[~alive, :] = np.inf
        sub[:, ~alive] = np.inf
        i, j = np.unravel_index(np.argmin(sub), sub.shape)
        if not np.isfinite(sub[i, j]) or sub[i, j] > tau:
            break
        merge_distances.append(float(sub[i, j]))
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
    frags_of_traj = {}
    t = 0
    for i in np.where(alive)[0]:
        for f in T[i]["frags"]:
            traj_of_row[stats[f]["rows"]] = t
        cents[t] = (T[i]["sum"].copy(), T[i]["n"])
        frags_of_traj[t] = list(T[i]["frags"])
        t += 1
    info = dict(n_fragments=len(stats), n_ordinary=len(main),
                n_allmulti=len(stats) - len(main), n_traj=t, n_merges=n_merges,
                merge_distances=merge_distances, frags_of_traj=frags_of_traj)
    return traj_of_row, cents, info


def pass1_conflicts(rec, traj_of_row, cents):
    """One detection per frame per trajectory. Clean detections always win; among
    conflicting non-clean detections the nearest to the trajectory mean stays and
    the rest are set aside. Extra clean detections in one frame are not expected
    (counted if seen; nearest kept, others set aside). Mutates ``traj_of_row``."""
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
    admissible trajectory -> stays -1. Mutates ``traj_of_row``."""
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


def merge_video(rec, tau):
    """Merging + pass 1 + pass 2 for one video (the notebook's ``run_full`` body for
    one sequence). Returns (traj_of_row, report)."""
    t, c, info = merge_sequence(rec, tau)
    pool, r1 = pass1_conflicts(rec, t, c)
    allmulti_rows = np.where((t < 0) & ~np.isin(np.arange(len(t)), pool))[0]
    r2 = pass2_place(rec, t, c, np.concatenate([pool, allmulti_rows]))
    report = dict(merge=info, pass1=r1,
                  pass2=dict(placed=r2["placed"], unassigned=r2["unassigned"],
                             from_pass1=int(len(pool)), from_allmulti=int(len(allmulti_rows))))
    return t, report


# ---------------------------------------------------------------------- driver

def split_merge_video(E, single, frames, track_ids, eps, min_samples, tau):
    """Whole method for one video. Returns (traj_of_row, report).

    ``traj_of_row`` is the trajectory index per input row (0..n_traj-1, or -1 for
    a detection that could not be placed). ``report`` holds the split, merge and
    pass counts the stage writes to its audit sidecar.
    """
    E = np.asarray(E)
    single = np.asarray(single, dtype=bool)
    frames = np.asarray(frames, dtype=np.int64)
    track_ids = np.asarray(track_ids, dtype=np.int64)
    n = len(E)
    if not (len(single) == len(frames) == len(track_ids) == n):
        raise ValueError("E, single, frames and track_ids must have one entry per detection")
    frag, per_tracklet = split_video(E, single, frames, track_ids, eps, min_samples)
    rec = dict(E=E, frag=frag, single=single, frames=frames)
    traj, rep = merge_video(rec, tau)
    rep["split"] = dict(per_tracklet=per_tracklet, fragments=int(len(np.unique(frag))),
                        tracklets=int(len(per_tracklet)),
                        tracklets_split=int(sum(1 for p in per_tracklet if p["k"] > 1)),
                        fragments_max_per_tracklet=int(max((p["k"] for p in per_tracklet), default=0)),
                        noise_points=int(sum(p["noise"] for p in per_tracklet)),
                        tracklets_below_min_samples=int(sum(
                            1 for p in per_tracklet if p["n_core"] < max(2, min_samples))))
    rep["frag"] = frag
    return traj, rep
