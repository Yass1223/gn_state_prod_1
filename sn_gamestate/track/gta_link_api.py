"""GTA-Link offline tracklet stitching as a TrackLab ``VideoLevelModule``.

Faithful port of the validated notebook (§11d.5). For one video it:

1. Re-extracts an OSNet-AIN appearance feature for every detection crop.
2. Optionally SPLITS tracklets that contain an id switch (``use_split``), then builds
   one EMA-averaged, L2-normalised embedding per tracklet.
3. Computes pairwise cosine distance, then *forbids* merging any pair that overlaps in
   time, and gates temporally-disjoint pairs by a spatial cut (centre gap must be below
   ``spatial_thresh * sqrt(frame gap)``).
4. Connects the tracklets and re-assigns ``track_id`` so each merged group shares one id.

Connect modes (``connect_mode``)
--------------------------------
``agglomerative`` (default, the notebook's behaviour): one shot of
``AgglomerativeClustering`` on the gated distance matrix.

``iterative``: the sjc042/gta-link reference — repeatedly take the globally closest
surviving pair below ``appearance_thresh``, gate it, pin a failed pair at
``appearance_thresh`` so it is never retried, and on success merge it (frames unioned,
EMA recomputed over the merged frame-sorted features, the survivor's row and column
recomputed against everything still alive). Because every candidate is gated against
the CURRENT merged frame set, this mode cannot produce the transitive per-frame
collisions described below — ``_dedup_per_frame`` must null zero rows, and logs an
error if it ever does.

Split (``use_split``)
---------------------
The reference GTA-Link is Split + Connect; this module had only Connect. With
``use_split: true``, every tracklet of at least ``split_len_thres`` detections is
DBSCAN-clustered (cosine, on feature-wise standardised per-detection embeddings)
before Connect runs: more than one cluster means the tracker glued two identities
together, and the tracklet is broken into one fragment per cluster for Connect to
re-stitch correctly. Because a split turns one ``track_id`` into several
tracklets, the final relabelling assigns ids per detection row, not per original
``track_id``.

Per-frame collision guard
-------------------------
Step 3 forbids merging tracklets that *directly* overlap in time, but agglomerative
average-linkage can still place two time-overlapping tracklets in the same cluster
*transitively* (A–B and B–C both merge, dragging A and C together even though A and C
share a frame). When that happens the relabelling would put two boxes with the same id
in one frame. Following the notebook, we resolve any such ``(image_id, track_id)``
collision by keeping the detection from the longer-lived original tracklet and setting
the colliding row's ``track_id`` to NaN (equivalent to the notebook's row drop). NaN
track_ids are handled downstream (the team visualiser skips them, pandas ``groupby``
drops them, and the SoccerNet eval encoding drops them), so this stays coherent.

Appearance model
----------------
The same OSNet-AIN used by the tracker, built from the same shared module
(``sn_gamestate.reid.osnet_ain``) so a detection's embedding here is identical to the one
the tracker saw for it: same checkpoint, same letterbox geometry, same arithmetic.
This runs as its own pipeline stage right after ``track`` and leaves the prtreid ``reid``
module — consumed by team, role and jersey — completely untouched.
"""
import logging
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler

from tracklab.pipeline.videolevel_module import VideoLevelModule
from tracklab.utils.cv2 import cv2_load_image

# The tracker's appearance model, reused here so both stages embed a crop identically.
from sn_gamestate.reid import osnet_ain

log = logging.getLogger(__name__)


def _closest_frames(fa: np.ndarray, fb: np.ndarray):
    """``(index into fa, index into fb, frame gap)`` of the closest frame pair.

    Both arrays are sorted ascending. Vectorised: for each frame of ``fa`` take the
    two neighbouring insertion points in ``fb`` and keep the global minimum.
    """
    pos = np.searchsorted(fb, fa)
    lo = np.clip(pos - 1, 0, len(fb) - 1)
    hi = np.clip(pos, 0, len(fb) - 1)
    d_lo = np.abs(fa - fb[lo])
    d_hi = np.abs(fa - fb[hi])
    take_hi = d_hi < d_lo
    d = np.where(take_hi, d_hi, d_lo)
    ib = np.where(take_hi, hi, lo)
    ia = int(np.argmin(d))
    return ia, int(ib[ia]), int(d[ia])


class _Tracklet:
    """One tracklet's frame-sorted detections plus its EMA appearance embedding.

    ``rows`` holds the detection index labels this tracklet owns — the relabelling
    step assigns ids per row, not per original ``track_id``, because Split can turn
    one ``track_id`` into several tracklets. ``ids`` carries the ORIGINAL
    ``track_id``\\ s folded in, for provenance in the logs.
    ``emb`` is the raw EMA output (the agglomerative path normalises the whole
    matrix at once, as it always has); ``unit`` is its L2-normalised form, used by
    the iterative path which compares one row at a time.
    """

    __slots__ = ("ids", "rows", "frames", "feats", "boxes", "emb", "unit")

    def __init__(self, ids, rows, frames, feats, boxes, ema_alpha):
        self.ids = list(ids)
        self.rows = np.asarray(rows)
        self.frames = np.asarray(frames)
        self.feats = np.asarray(feats)
        self.boxes = np.asarray(boxes, dtype=float)

        emb = self.feats[0].copy()
        for v in self.feats[1:]:
            emb = ema_alpha * emb + (1.0 - ema_alpha) * v
            emb /= (np.linalg.norm(emb) + 1e-6)
        self.emb = emb
        self.unit = emb / (np.linalg.norm(emb) + 1e-6)

    def extent(self):
        """``(emb, first_frame, last_frame, first_box, last_box)`` — ``_gate``'s tuple."""
        return (self.emb, int(self.frames[0]), int(self.frames[-1]),
                self.boxes[0], self.boxes[-1])

    def merged_with(self, other: "_Tracklet", ema_alpha: float) -> "_Tracklet":
        """Union of both tracklets, frame-sorted, with the EMA recomputed over it."""
        frames = np.concatenate([self.frames, other.frames])
        o = np.argsort(frames, kind="stable")
        return _Tracklet(
            self.ids + other.ids,
            np.concatenate([self.rows, other.rows])[o],
            frames[o],
            np.concatenate([self.feats, other.feats])[o],
            np.concatenate([self.boxes, other.boxes])[o],
            ema_alpha,
        )

    def subset(self, sel, ema_alpha: float) -> "_Tracklet":
        """The detections at positions ``sel``, as a tracklet of their own."""
        return _Tracklet(list(self.ids), self.rows[sel], self.frames[sel],
                         self.feats[sel], self.boxes[sel], ema_alpha)


class GTALink(VideoLevelModule):
    input_columns = ["track_id", "bbox_ltwh", "image_id"]
    output_columns = ["track_id"]

    def __init__(self, cfg, device, tracking_dataset=None):
        self.cfg = cfg
        self.device = device

        # The tracker's appearance model: same module, same checkpoint, same
        # letterbox preprocessing, same fp16-autocast arithmetic, so a crop embeds
        # to the same vector in both stages. A checkpoint-pin mismatch between the
        # two module configs is a run-audit FAIL, not a silent divergence.
        self.embedder = osnet_ain.from_config(
            cfg, device, batch_size=int(getattr(cfg, "batch_size", 64))
        )

        self.appearance_thresh = float(cfg.appearance_thresh)
        self.spatial_thresh = float(cfg.spatial_thresh)
        self.min_tracklet_len = int(cfg.min_tracklet_len)
        self.ema_alpha = float(cfg.ema_alpha)
        self.linkage = getattr(cfg, "linkage", "average")
        self.batch_size = int(getattr(cfg, "batch_size", 64))

        self.connect_mode = str(getattr(cfg, "connect_mode", "agglomerative")).lower()
        if self.connect_mode not in ("agglomerative", "iterative"):
            raise ValueError(
                f"connect_mode must be 'agglomerative' or 'iterative', "
                f"got {self.connect_mode!r}"
            )

        self.use_split = bool(getattr(cfg, "use_split", False))
        self.split_eps = float(getattr(cfg, "split_eps", 0.6))
        self.split_min_samples = int(getattr(cfg, "split_min_samples", 4))
        self.split_max_k = int(getattr(cfg, "split_max_k", 3))
        self.split_len_thres = int(getattr(cfg, "split_len_thres", 30))

    # ------------------------------------------------------------------ features
    @torch.no_grad()
    def _extract_features(self, dets: pd.DataFrame, metadatas: pd.DataFrame) -> np.ndarray:
        """Appearance feature per row, aligned to ``dets.index`` (zeros on failure)."""
        feats = np.zeros((len(dets), self.embedder.dim), dtype=np.float32)
        id2path = (metadatas["file_path"].to_dict()
                   if "file_path" in metadatas.columns else {})
        pos = {idx: i for i, idx in enumerate(dets.index)}
        n_missing_path = n_unreadable = 0

        for image_id, group in dets.groupby("image_id"):
            path = id2path.get(image_id)
            if path is None:
                n_missing_path += 1
                continue
            img = cv2_load_image(path)
            if img is None:
                n_unreadable += 1
                continue
            # cv2_load_image returns RGB; the OSNet-AIN preprocessing does BGR2RGB
            # itself, so it must be handed BGR or every channel arrives swapped.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            crops, rows = [], []
            for idx, det in group.iterrows():
                l, t, w, h = [float(v) for v in det["bbox_ltwh"]]
                patch = osnet_ain.crop_ltrb(img, (l, t, l + w, t + h))
                if patch is None:
                    continue  # degenerate box: leave a zero feature so it never merges
                crops.append(patch)
                rows.append(idx)
            if crops:
                # Already L2-normalised by the embedder, which also does the batching.
                for r, f in zip(rows, self.embedder.embed(crops)):
                    feats[pos[r]] = f

        if n_missing_path or n_unreadable:
            log.warning(
                f"[GTA-Link] skipped {n_missing_path} frame(s) with no file_path and "
                f"{n_unreadable} unreadable frame(s); their detections keep zero features."
            )
        # A zero row has cosine distance 1.0 to everything, so it never merges and
        # drags average linkage around silently. Surface the rate; at 100% the
        # appearance model or the frame paths are broken and the whole stage is noise.
        if len(feats):
            n_zero = int((~feats.any(axis=1)).sum())
            if n_zero == len(feats):
                raise RuntimeError(
                    f"[GTA-Link] all {len(feats)} detection crops produced an all-zero "
                    f"appearance feature - the ReID backbone or the frame paths are "
                    f"broken; refusing to stitch tracklets on empty embeddings."
                )
            if n_zero > 0.05 * len(feats):
                log.warning(
                    f"[GTA-Link] {n_zero}/{len(feats)} ({n_zero / len(feats):.1%}) "
                    f"detections have an all-zero appearance feature (degenerate box, "
                    f"unreadable frame, or missing file_path); they cannot be merged."
                )
        return feats

    # ------------------------------------------------------------------ gating
    def _gate(self, embs, tids) -> np.ndarray:
        """Symmetric merge-forbidden mask over the tracklets listed in ``tids``.

        Decided per UNORDERED pair: tracklets overlapping in time may never merge;
        a temporally disjoint pair is gated once — on its (earlier -> later)
        transition — and that single verdict is written to both ``[i, j]`` and
        ``[j, i]``. Deciding each ordered cell on its own leaves the
        (later, earlier) cell ungated, and since
        ``AgglomerativeClustering(metric="precomputed")`` does not symmetrize its
        input, the resulting clustering depends on tracklet index order.
        """
        n = len(tids)
        forbidden = np.zeros((n, n), dtype=bool)
        np.fill_diagonal(forbidden, True)
        for i in range(n):
            _, si, ei, first_i, last_i = embs[tids[i]]
            for j in range(i + 1, n):
                _, sj, ej, first_j, last_j = embs[tids[j]]
                if not (ei < sj or ej < si):         # overlap in time -> never merge
                    block = True
                else:                                 # disjoint -> one spatial gate
                    if ei < sj:                       # i ends before j starts
                        end_box, start_box, frame_gap = last_i, first_j, sj - ei
                    else:                             # j ends before i starts
                        end_box, start_box, frame_gap = last_j, first_i, si - ej
                    a = end_box[:2] + end_box[2:] / 2.0      # last centre of the earlier
                    b = start_box[:2] + start_box[2:] / 2.0  # first centre of the later
                    gap = max(1.0, abs(frame_gap) ** 0.5)
                    block = bool(np.linalg.norm(a - b) > self.spatial_thresh * gap)
                forbidden[i, j] = forbidden[j, i] = block
        return forbidden

    # ------------------------------------------------------------------ clustering
    def _cluster(self, dist: np.ndarray):
        # sklearn renamed `affinity` -> `metric` in 1.2; support both.
        kw = dict(n_clusters=None, distance_threshold=self.appearance_thresh,
                  linkage=self.linkage)
        try:
            return AgglomerativeClustering(metric="precomputed", **kw).fit_predict(dist)
        except TypeError:
            return AgglomerativeClustering(affinity="precomputed", **kw).fit_predict(dist)

    # ------------------------------------------------------------------ tracklets
    def _build_tracklets(self, work: pd.DataFrame, feats: np.ndarray) -> "OrderedDict":
        """``track_id -> _Tracklet`` for every tracklet of >= ``min_tracklet_len``."""
        order = {idx: i for i, idx in enumerate(work.index)}
        tracks = OrderedDict()
        for tid, g in work.groupby("track_id"):
            g = g.sort_values("image_id")
            rows = [order[i] for i in g.index]
            if len(rows) < self.min_tracklet_len:
                continue
            tracks[tid] = _Tracklet(
                [tid],
                g.index.to_numpy(),
                g["image_id"].to_numpy(dtype=np.int64),
                feats[rows],
                np.stack([np.asarray(b, dtype=float) for b in g["bbox_ltwh"]]),
                self.ema_alpha,
            )
        return tracks

    # ------------------------------------------------------------------ split
    def _detect_id_switch(self, feats: np.ndarray):
        """Per-detection cluster labels if a tracklet holds >1 identity, else None.

        Reference GTA-Link ``detect_id_switch``: DBSCAN over the tracklet's
        per-detection embeddings, standardised feature-wise first. Noise points are
        assigned to their nearest cluster centre, and the cluster count is capped at
        ``split_max_k`` by repeatedly merging the two closest centres.
        """
        if len(feats) < max(2, self.split_min_samples):
            return None
        scaled = StandardScaler().fit_transform(feats)
        # All-identical (or all-zero) features standardise to all-zero rows, where
        # the cosine metric is undefined. Nothing to split in that case anyway.
        if not np.all(np.linalg.norm(scaled, axis=1) > 0):
            log.warning("[GTA-Link] split: a tracklet's detections have no "
                        "appearance variance; not splitting it")
            return None

        labels = DBSCAN(eps=self.split_eps, min_samples=self.split_min_samples,
                        metric="cosine").fit_predict(scaled)
        real = sorted(set(labels.tolist()) - {-1})
        if len(real) < 2:                       # one identity, or nothing but noise
            return None

        centres = np.stack([scaled[labels == l].mean(axis=0) for l in real])
        noise = np.flatnonzero(labels == -1)
        if len(noise):                          # noise -> nearest centre
            d = np.linalg.norm(scaled[noise][:, None, :] - centres[None, :, :], axis=2)
            labels[noise] = np.asarray(real)[np.argmin(d, axis=1)]

        while len(set(labels.tolist())) > self.split_max_k:
            keys = sorted(set(labels.tolist()))
            C = np.stack([scaled[labels == k].mean(axis=0) for k in keys])
            D = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
            np.fill_diagonal(D, np.inf)
            a, b = np.unravel_index(int(np.argmin(D)), D.shape)
            labels[labels == keys[b]] = keys[a]

        return labels if len(set(labels.tolist())) > 1 else None

    def _split_tracklets(self, tracks: "OrderedDict") -> "OrderedDict":
        """Break tracklets that contain an id switch, BEFORE Connect stitches them.

        Reuses the features ``_extract_features`` already produced, so the only
        added cost is one DBSCAN per long tracklet.
        """
        out, n_split, n_frag = OrderedDict(), 0, 0
        for tid, tr in tracks.items():
            labels = (self._detect_id_switch(tr.feats)
                      if len(tr.frames) >= self.split_len_thres else None)
            if labels is None:
                out[tid] = tr
                continue
            n_split += 1
            for k, lab in enumerate(sorted(set(labels.tolist()))):
                out[(tid, k)] = tr.subset(np.flatnonzero(labels == lab),
                                          self.ema_alpha)
                n_frag += 1
        if n_split:
            log.info(
                f"[GTA-Link] split: {n_split} tracklet(s) held an id switch, broken "
                f"into {n_frag} fragment(s) ({len(tracks)} -> {len(out)} tracklets)"
            )
        return out

    # ------------------------------------------------------------------ connect
    def _connect_agglomerative(self, tracks: "OrderedDict") -> list:
        """Single-shot clustering of the gated distance matrix (the default path)."""
        tids = list(tracks)
        E = np.stack([tracks[t].emb for t in tids])
        E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-6)
        dist = 1.0 - E @ E.T

        embs = OrderedDict((t, tracks[t].extent()) for t in tids)
        gated = dist.copy()
        gated[self._gate(embs, tids)] = 1.0
        np.fill_diagonal(gated, 0.0)
        # Both consumers of `gated` assume a metric: AgglomerativeClustering with
        # metric="precomputed" does NOT symmetrize its input, so an asymmetric
        # matrix makes the result depend on tracklet index order.
        assert np.allclose(gated, gated.T), "gated distance matrix is not symmetric"

        groups = OrderedDict()
        for t, lab in zip(tids, self._cluster(gated)):
            groups.setdefault(lab, []).append(tracks[t].rows)
        return [np.concatenate(r) for r in groups.values()]

    def _connect_iterative(self, tracks: "OrderedDict") -> list:
        """Reference GTA-Link Connect: greedy global-minimum merge with recompute.

        Repeatedly take the globally closest surviving pair. Below
        ``appearance_thresh`` it is gated; a pair that fails the gate has both its
        cells pinned at ``appearance_thresh`` so it can never be selected again,
        and a pair that passes is merged (frames unioned, EMA recomputed over the
        merged frame-sorted features, row/col of the survivor recomputed against
        everything still alive). Unlike average-linkage clustering this can never
        merge two tracklets that share a frame, not even transitively: every
        candidate is gated against the CURRENT merged frame set.
        """
        items = list(tracks.values())
        n = len(items)
        U = np.stack([t.unit for t in items])
        D = 1.0 - U @ U.T
        np.fill_diagonal(D, np.inf)

        alive = np.ones(n, dtype=bool)
        n_merged = n_rejected = 0
        while True:
            i, j = np.unravel_index(int(np.argmin(D)), D.shape)
            if not D[i, j] < self.appearance_thresh:
                break
            if self._pair_blocked(items[i], items[j]):
                D[i, j] = D[j, i] = self.appearance_thresh   # pinned: never retried
                n_rejected += 1
                continue

            items[i] = items[i].merged_with(items[j], self.ema_alpha)
            alive[j] = False
            D[j, :] = D[:, j] = np.inf
            # Row/col recompute: the survivor's appearance changed, so every
            # distance that involves it is stale (this also clears its pins).
            live = np.flatnonzero(alive)
            live = live[live != i]
            if len(live):
                d = 1.0 - np.stack([items[k].unit for k in live]) @ items[i].unit
                D[i, live] = d
                D[live, i] = d
            n_merged += 1

        log.info(
            f"[GTA-Link] iterative connect: {n_merged} merge(s), "
            f"{n_rejected} pair(s) rejected by the gate"
        )
        return [items[k].rows for k in np.flatnonzero(alive)]

    def _pair_blocked(self, a: "_Tracklet", b: "_Tracklet") -> bool:
        """Temporal + spatial gate for one (possibly already merged) pair.

        Merged tracklets are not intervals — Connect can fold a tracklet from the
        start of the clip together with one from the end — so the temporal test is
        a frame-SET intersection and the spatial test is applied at the pair's
        closest approach in time. For two contiguous disjoint tracklets that is
        exactly the (last centre of the earlier, first centre of the later) rule
        used by the agglomerative path.
        """
        if np.intersect1d(a.frames, b.frames).size:
            return True
        ia, ib, frame_gap = _closest_frames(a.frames, b.frames)
        ca = a.boxes[ia][:2] + a.boxes[ia][2:] / 2.0
        cb = b.boxes[ib][:2] + b.boxes[ib][2:] / 2.0
        gap = max(1.0, float(frame_gap) ** 0.5)
        return bool(np.linalg.norm(ca - cb) > self.spatial_thresh * gap)

    # ------------------------------------------------------------------ main
    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        if len(detections) == 0 or "track_id" not in detections.columns:
            return detections
        work = detections[detections["track_id"].notna()].copy()
        if work["track_id"].nunique() < 2:
            return detections

        feats = self._extract_features(work, metadatas)
        tracks = self._build_tracklets(work, feats)
        if self.use_split:
            tracks = self._split_tracklets(tracks)
        if len(tracks) < 2:
            return detections

        if self.connect_mode == "iterative":
            groups = self._connect_iterative(tracks)
        else:
            groups = self._connect_agglomerative(tracks)

        # Collision-free relabelling, per DETECTION rather than per original
        # track_id: Split can break one track_id into several tracklets, so a
        # per-id map cannot express the result. Tracklets connected together share
        # a fresh id; anything no tracklet claimed (below min_tracklet_len) keeps
        # its identity with a fresh id of its own. NaN track_ids stay NaN.
        new_ids = pd.Series(np.nan, index=detections.index, dtype=float)
        next_id = 1
        for rows in groups:
            new_ids.loc[rows] = next_id
            next_id += 1
        unclaimed = detections["track_id"].notna() & new_ids.isna()
        for tid in detections.loc[unclaimed, "track_id"].unique():
            new_ids.loc[unclaimed & (detections["track_id"] == tid)] = next_id
            next_id += 1

        out = detections.copy()
        out["track_id"] = new_ids

        # Per-frame collision guard (see module docstring): if transitive clustering put
        # two time-overlapping tracklets under one new id, keep the detection from the
        # longer-lived original tracklet in each colliding frame and NaN the rest.
        n_nulled = self._dedup_per_frame(detections, out)
        if self.connect_mode == "iterative" and n_nulled:
            # Iterative Connect gates every candidate against the CURRENT merged
            # frame set, so a per-frame collision is structurally impossible here.
            # If one appears, the merge bookkeeping is wrong - not a tuning issue.
            log.error(
                f"[GTA-Link] iterative connect produced {n_nulled} per-frame id "
                f"collision(s); the frame-set gate makes this impossible, so the "
                f"merge bookkeeping is broken."
            )

        n_before = int(detections["track_id"].dropna().nunique())
        n_after = int(out["track_id"].dropna().nunique())
        log.info(
            f"[GTA-Link] tracklets {n_before} -> {n_after} "
            f"(merged {n_before - n_after}); per-frame de-dup nulled {n_nulled} detection(s)"
        )
        return out

    # ------------------------------------------------------------------ de-dup
    @staticmethod
    def _dedup_per_frame(detections: pd.DataFrame, out: pd.DataFrame) -> int:
        """NaN colliding (image_id, new track_id) rows, keeping the longest-lived original.

        Returns the number of detection rows set to NaN. Mutates ``out`` in place.
        """
        orig = detections["track_id"]
        keep = orig.notna() & out["track_id"].notna()
        if not keep.any():
            return 0

        img = detections["image_id"]
        # Original-tracklet lifetime (frame span) and support (detection count), keyed by
        # the ORIGINAL track_id, for deterministic keeper selection on collisions.
        img_int = img[orig.notna()].astype(int)
        life, count = {}, {}
        for tid, g in img_int.groupby(orig[orig.notna()]):
            life[tid] = int(g.max() - g.min() + 1)
            count[tid] = int(g.shape[0])

        frame = pd.DataFrame(
            {"orig": orig[keep].values,
             "new": out["track_id"][keep].values,
             "img": img[keep].astype(str).values},
            index=detections.index[keep],
        )

        to_nan = []
        for _, g in frame.groupby(["img", "new"]):
            if len(g) <= 1:
                continue
            best_idx, best_key = None, None
            for idx, row in g.iterrows():
                o = row["orig"]
                key = (life.get(o, 0), count.get(o, 0), -float(o))  # max life, count; min id
                if best_key is None or key > best_key:
                    best_key, best_idx = key, idx
            to_nan.extend([idx for idx in g.index if idx != best_idx])

        if to_nan:
            out.loc[to_nan, "track_id"] = np.nan
        return len(to_nan)
