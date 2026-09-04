"""Tracklet splitting -- Stage 1 of the refinement method, and nothing else.

The pipeline's ONE merge lives in ``traj_refine`` (Stage 2); this module only
breaks tracklets that hold more than one identity into fragments. It performs
no fragment-to-fragment merging of any kind. The only attachments it makes are
Stage 1's own outlier handling, which completes the split:

Per tracklet (inputs: unit appearance embeddings ``u = e/||e||`` of ALL its
detections -- clean and multi-player alike -- and the crop filter's
``crop_single`` label):

1. DBSCAN(eps, min_samples) on the precomputed cosine-distance matrix of ALL
   detections, yielding fragments and noise points.
2. Every noise detection -- single or multi crop -- is assigned to the fragment
   with the nearest centroid.
3. Every fragment containing ONLY multi-player detections is dissolved: each of
   its detections is assigned, individually, to the nearest remaining fragment.
4. Centroids are recomputed from clean detections only (the convention every
   later stage uses).

Centroid convention (shared with ``traj_refine``): mean unit embedding over a
fragment's clean, non-zero detections; over its non-zero detections when it has
no clean one; a fragment with no non-zero embedding has no centroid and cannot
attract outliers. Zero-embedding detections (unreadable frame, degenerate box)
are at cosine distance 1 from everything; when they end up as noise their
attachment falls to the deterministic tie rule (lowest fragment label).

Degenerate cases, all deterministic:
* a tracklet with fewer than ``max(2, min_samples)`` detections, or whose
  DBSCAN result is all noise, is ONE fragment;
* a tracklet whose fragments are ALL multi-only has no dissolution target and
  keeps its DBSCAN fragments as they are;
* if no fragment has a valid centroid, noise attaches to the lowest label.

Fragment ids follow the ``split_merge`` convention ``tid * FRAG_BASE + label``
so a fragment's source tracklet is recoverable by integer division.

Multi-player detections are otherwise IGNORED until Stage 3 (duplicate-frame
resolution inside ``traj_refine``): they contribute nothing to centroids here
beyond the fallback, and nothing to any later distance or disjointness test.

Input invariant (the tracker's): one detection per (tracklet, frame). The
driver validates it and raises on violation, because every later one-per-frame
guarantee builds on it.
"""
import numpy as np
from sklearn.cluster import DBSCAN

FRAG_BASE = 10000


def _unit(E):
    n = np.linalg.norm(E, axis=1, keepdims=True)
    return np.where(n > 1e-9, E / np.maximum(n, 1e-9), 0.0)


def _centroid(U, single, rows):
    """Clean-first centroid over ``rows`` of unit matrix ``U`` (None if the
    rows hold no non-zero embedding)."""
    rows = np.asarray(rows, dtype=np.int64)
    nz = np.linalg.norm(U[rows], axis=1) > 1e-6
    clean = np.asarray(single, dtype=bool)[rows] & nz
    if clean.any():
        return U[rows[clean]].mean(axis=0)
    if nz.any():
        return U[rows[nz]].mean(axis=0)
    return None


def split_tracklet(U, single, eps, min_samples):
    """One tracklet -> fragment label per detection (0..k-1), plus counts.

    ``U`` (n, d) unit embeddings of ALL the tracklet's detections, in row
    order; ``single`` (n,) bool. Returns ``(labels, k, n_noise, n_dissolved)``.
    """
    n = len(U)
    single = np.asarray(single, dtype=bool)
    if n < max(2, int(min_samples)):
        return np.zeros(n, dtype=np.int64), 1, 0, 0
    D = np.clip(1.0 - U @ U.T, 0.0, 2.0)
    lab = DBSCAN(eps=eps, min_samples=int(min_samples),
                 metric="precomputed").fit_predict(D)
    lab = lab.astype(np.int64)
    clusters = sorted(int(c) for c in np.unique(lab) if c >= 0)
    if not clusters:
        return np.zeros(n, dtype=np.int64), 1, 0, 0

    def cent(c):
        return _centroid(U, single, np.where(lab == c)[0])

    cents = {c: cent(c) for c in clusters}

    # 2. noise -> nearest fragment centroid (single or multi crop alike)
    n_noise = int((lab == -1).sum())
    for i in np.where(lab == -1)[0]:
        best, best_d = None, None
        for c in clusters:
            mu = cents[c]
            d = 1.0 if mu is None else float(1.0 - U[i] @ mu)
            if best is None or d < best_d - 1e-12:
                best, best_d = c, d
        lab[i] = best                      # ties fall to the lowest label

    # 3. dissolve all-multi fragments into the nearest remaining fragment
    remaining = [c for c in clusters if single[lab == c].any()]
    n_dissolved = 0
    if remaining and len(remaining) < len(clusters):
        cents = {c: _centroid(U, single, np.where(lab == c)[0]) for c in remaining}
        for c in clusters:
            if c in remaining:
                continue
            n_dissolved += 1
            for i in np.where(lab == c)[0]:
                best, best_d = None, None
                for r in remaining:
                    mu = cents[r]
                    d = 1.0 if mu is None else float(1.0 - U[i] @ mu)
                    if best is None or d < best_d - 1e-12:
                        best, best_d = r, d
                lab[i] = best
        clusters = remaining

    # compact relabel 0..k-1 preserving cluster order
    order = {c: k for k, c in enumerate(sorted(set(int(x) for x in lab)))}
    lab = np.array([order[int(x)] for x in lab], dtype=np.int64)
    return lab, len(order), n_noise, n_dissolved


def split_video(E, single, frames, track_ids, eps, min_samples):
    """All tracklets of one video.

    Aligned arrays over TRACKED detections: ``E`` (n, d) embeddings (rows may
    be zero), ``single`` (n,) bool, ``frames`` (n,) int (equality == same
    frame), ``track_ids`` (n,) int. Raises on a duplicated (tracklet, frame)
    pair -- the tracker invariant every later stage builds on.

    Returns ``(frag, per_tracklet)``: ``frag[i] = tid * FRAG_BASE + label`` and
    one report entry per tracklet.
    """
    E = np.asarray(E, dtype=np.float32)
    single = np.asarray(single, dtype=bool)
    frames = np.asarray(frames, dtype=np.int64)
    track_ids = np.asarray(track_ids, dtype=np.int64)
    n = len(E)
    if not (len(single) == len(frames) == len(track_ids) == n):
        raise ValueError("E, single, frames and track_ids must have one entry "
                         "per detection")
    pairs = set()
    for t, f in zip(track_ids, frames):
        key = (int(t), int(f))
        if key in pairs:
            raise ValueError(f"tracklet {t} holds two detections in frame {f}; "
                             f"the tracker invariant is broken")
        pairs.add(key)

    U = _unit(E)
    frag = np.empty(n, dtype=np.int64)
    per_tracklet = []
    for tid in np.unique(track_ids):
        idx = np.where(track_ids == tid)[0]
        lab, k, n_noise, n_dissolved = split_tracklet(
            U[idx], single[idx], eps, min_samples)
        frag[idx] = int(tid) * FRAG_BASE + lab
        per_tracklet.append(dict(
            track_id=int(tid), n=int(len(idx)),
            n_single=int(single[idx].sum()),
            n_multi=int((~single[idx]).sum()), k=int(k),
            noise=int(n_noise), dissolved_allmulti=int(n_dissolved)))
    return frag, per_tracklet
