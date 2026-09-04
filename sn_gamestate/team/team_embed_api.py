"""Team-appearance embeddings and TEAM CLUSTERS per fragment (``team_embed`` stage).

Runs after ``tracklet_split``: the tracklets it sees are the splitter's
fragments. For every fragment, its SINGLE crops (``crop_single``) on the
position grid (every ``POS_STRIDE``-th frame, at most ``CROPS_PER_TRK`` evenly
spaced) are embedded with ``osnet_team``; a fragment with no single crop gets
no embedding. The fragment descriptor is the L2-normalised median of its
embedded crops (float32, the notebook's convention).

The fragment descriptors of the sequence are then clustered into TEAM CLUSTERS
-- anonymous kit ids, no left/right naming and no roles (those are assigned
after ``traj_refine``, on finished trajectories). ``team_cluster_nearest``
additionally records every embedded fragment's nearest centroid BEFORE the
threshold -- the role/side stage's fallback for unclustered trajectories.
Method ``kmeans2_threshold``:
2-means (the notebook's seeded k-means), then the robust distance rule -- with
``d`` each fragment's Euclidean distance to its nearest centroid, ``m`` the
median of ``d`` and ``s`` its MAD, a fragment is UNCLUSTERED when
``s >= 0.05*m`` (the scale is meaningful) and ``d > m + outlier_k*s``; so
referees and other kit outliers get no cluster. Fragments without an embedding
are unclustered too. ``cluster_method`` is a config switch so alternative
clusterings can be compared.

Output columns: ``team_embedding`` (float32 vector on the sampled single rows,
None elsewhere) and ``team_cluster`` (0.0/1.0 on every row of a clustered
fragment, NaN otherwise). Sampling is ``rules.sample_tracklet_rows`` on the
fragment's single rows.

A per-sequence sidecar ``<audit_dir>/<sequence>.json`` records what ran: the
checkpoint digest and source, input size, crops sampled and embedded, crops that
clamped to nothing (skipped, left None), and tracklets that had no frame on the
grid (fell back to all their rows). The run audit compares it with the config.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tracklab.pipeline.videolevel_module import VideoLevelModule
from tracklab.utils.cv2 import cv2_load_image

from sn_gamestate.reid import osnet_team
from sn_gamestate.team import rules

log = logging.getLogger(__name__)


def sequence_name(metadatas: pd.DataFrame) -> str:
    if len(metadatas) and "file_path" in metadatas.columns:
        return Path(str(metadatas["file_path"].iloc[0])).parent.parent.name
    if len(metadatas) and "video_id" in metadatas.columns:
        return str(metadatas["video_id"].iloc[0])
    return "unknown"


def frame_index(metadatas: pd.DataFrame) -> pd.Series:
    """0-based frame index per image_id: the dataset's ``frame`` column when present,
    else the rank of the image in file order."""
    if "frame" in metadatas.columns and metadatas["frame"].notna().all():
        return metadatas["frame"].astype(int)
    order = metadatas.sort_index().index if "file_path" not in metadatas.columns else \
        metadatas.sort_values("file_path").index
    return pd.Series(np.arange(len(order)), index=order).reindex(metadatas.index)


class TeamEmbedding(VideoLevelModule):
    input_columns = ["track_id", "bbox_ltwh", "image_id", "crop_single"]
    output_columns = ["team_embedding", "team_cluster", "team_cluster_nearest"]

    def __init__(self, cfg, device, tracking_dataset=None, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.stride = int(getattr(cfg, "pos_stride", rules.POS_STRIDE))
        self.crops_per_track = int(getattr(cfg, "crops_per_track", rules.CROPS_PER_TRK))
        self.batch_size = int(getattr(cfg, "batch_size", 128))
        self.cluster_method = str(getattr(cfg, "cluster_method", "kmeans2_threshold"))
        if self.cluster_method not in ("kmeans2_threshold",):
            raise ValueError(f"[team_embed] unknown cluster_method "
                             f"{self.cluster_method!r}")
        self.outlier_k = float(getattr(cfg, "outlier_k", 3.25))
        self.audit_dir = Path(str(cfg.audit_dir)) if getattr(cfg, "audit_dir", None) else None
        if self.audit_dir:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.model = None      # built on first use so a config error surfaces at construction, weights at run

    def _model(self):
        if self.model is None:
            self.model = osnet_team.from_config(self.cfg, self.device, batch_size=self.batch_size)
        return self.model

    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        seq = sequence_name(metadatas)
        out = detections.copy()
        out["team_embedding"] = None
        record = dict(sequence=seq, stride=self.stride, crops_per_track=self.crops_per_track,
                      tracklets=0, tracklets_off_grid=0, fragments_no_single=0,
                      crops_sampled=0, crops_embedded=0,
                      crops_empty=0, frames_read=0, embedder=None,
                      cluster=dict(method=self.cluster_method, outlier_k=self.outlier_k))
        tracked = out.dropna(subset=["track_id"])
        if len(tracked) == 0:
            log.warning(f"[team_embed] {seq}: no tracked detection; nothing embedded")
            self._write(record)
            return out
        model = self._model()
        record["embedder"] = dict(model.info)
        fidx = frame_index(metadatas)
        id2path = {idx: str(p) for idx, p in metadatas["file_path"].items()}
        # which rows to embed, per tracklet
        rows_by_image = {}
        for tid, grp in tracked.groupby("track_id"):
            record["tracklets"] += 1
            sgrp = grp[grp["crop_single"].astype(bool)]
            if len(sgrp) == 0:
                record["fragments_no_single"] += 1
                continue                       # no single crop -> no embedding
            frames = fidx.reindex(sgrp["image_id"].to_numpy()).to_numpy()
            if np.isnan(frames.astype(float)).any():
                raise RuntimeError(f"[team_embed] {seq}: detection image_id without frame metadata")
            _, crop_rows, off_grid = rules.sample_tracklet_rows(frames, self.stride, self.crops_per_track)
            record["tracklets_off_grid"] += int(off_grid)
            for r in sgrp.index[crop_rows]:
                rows_by_image.setdefault(sgrp.at[r, "image_id"], []).append(r)
        record["crops_sampled"] = sum(len(v) for v in rows_by_image.values())
        # embed frame by frame (RGB, as loaded)
        emb_col = {}
        pending, pending_rows = [], []
        for image_id, rows in rows_by_image.items():
            path = id2path.get(image_id)
            if path is None:
                raise RuntimeError(f"[team_embed] {seq}: image_id {image_id} has no file_path")
            img = cv2_load_image(path)
            record["frames_read"] += 1
            for r in rows:
                crop = osnet_team.crop_rgb(img, out.at[r, "bbox_ltwh"])
                if crop.size == 0 or crop.shape[0] < 1 or crop.shape[1] < 1:
                    record["crops_empty"] += 1
                    continue
                pending.append(np.ascontiguousarray(crop))
                pending_rows.append(r)
            if len(pending) >= self.batch_size:
                for r, e in zip(pending_rows, model.embed(pending)):
                    emb_col[r] = e
                pending, pending_rows = [], []
        if pending:
            for r, e in zip(pending_rows, model.embed(pending)):
                emb_col[r] = e
        record["crops_embedded"] = len(emb_col)
        col = pd.Series(emb_col, dtype=object).reindex(out.index)
        out["team_embedding"] = col.where(col.notna(), None)
        bad = sum(1 for e in emb_col.values() if not np.isfinite(e).all() or not np.any(e))
        record["embeddings_nonfinite_or_zero"] = int(bad)

        # ------------------------------------------------- team clustering --
        # Fragment descriptor: L2-normalised median of its embedded single
        # crops. 2-means over the descriptors; the robust distance rule leaves
        # kit outliers (referees etc.) UNCLUSTERED.
        out["team_cluster"] = np.nan
        # nearest centroid id BEFORE thresholding: the role/side stage's
        # fallback for trajectories left unclustered (player, nearest kit)
        out["team_cluster_nearest"] = np.nan
        frag_ids, descs = [], []
        for tid, grp in out[out["track_id"].notna()].groupby("track_id"):
            es = [e for e in grp["team_embedding"]
                  if isinstance(e, np.ndarray) and e.size and np.isfinite(e).all()
                  and np.any(e)]
            if not es:
                continue
            e = np.median(np.asarray(es, dtype=np.float32), axis=0)
            n = float(np.linalg.norm(e))
            if n < 1e-9:
                continue
            frag_ids.append(tid)
            descs.append((e / n).astype(np.float32))
        clus = record["cluster"]
        clus.update(fragments=int(record["tracklets"]), embedded=len(frag_ids),
                    clustered=0, unclustered_threshold=0, s_ok=None,
                    m=None, s=None, sizes=[0, 0])
        if len(frag_ids) >= 2:
            E = np.stack(descs)
            km = rules.kmeans2(E)
            d_all = np.linalg.norm(E[:, None] - km.cluster_centers_[None], axis=2)
            lab = d_all.argmin(1)
            d = d_all.min(1)
            m = float(np.median(d))
            s = float(np.median(np.abs(d - m)))
            s_ok = s >= 0.05 * m
            outlier = (d > m + self.outlier_k * s) if s_ok else np.zeros(len(d), bool)
            for tid, l, is_out in zip(frag_ids, lab, outlier):
                out.loc[out["track_id"] == tid, "team_cluster_nearest"] = float(l)
                if not is_out:
                    out.loc[out["track_id"] == tid, "team_cluster"] = float(l)
            clus.update(clustered=int((~outlier).sum()),
                        unclustered_threshold=int(outlier.sum()),
                        s_ok=bool(s_ok), m=round(m, 6), s=round(s, 6),
                        sizes=[int(((lab == c) & ~outlier).sum()) for c in (0, 1)],
                        centroid_gap=round(float(np.linalg.norm(
                            km.cluster_centers_[0] - km.cluster_centers_[1])), 6))
        n_unclustered = int(record["tracklets"]) - int(clus["clustered"])
        log.info(f"[team_embed] {seq}: {record['tracklets']} fragments, "
                 f"{record['crops_embedded']} single crops embedded from "
                 f"{record['frames_read']} frames ({record['crops_empty']} empty, "
                 f"{record['fragments_no_single']} fragment(s) without a single crop); "
                 f"clusters {clus['sizes']}, {n_unclustered} unclustered "
                 f"({self.cluster_method}, k {self.outlier_k}); no sides, no roles "
                 f"here - they are assigned after traj_refine")
        self._write(record)
        return out

    def _write(self, record):
        if self.audit_dir:
            (self.audit_dir / f"{record['sequence']}.json").write_text(json.dumps(record, indent=2, default=str))
