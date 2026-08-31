"""Team-appearance embeddings of sampled crops per tracklet (``team_embed`` stage).

For every tracklet (final ``track_id``, after ``split_merge``) the rows on the position
grid (every ``POS_STRIDE``-th frame) are taken and at most ``CROPS_PER_TRK`` of
them, evenly spaced in time, are embedded with ``osnet_team`` — single and multi
crops alike, exactly as the notebook embedded every sampled crop and applied the
single filter afterwards from the stored ratios. The role/team stage then keeps
the single ones (``crop_single``) and falls back to the lowest-rT ones.

Output column ``team_embedding``: a float32 vector on the sampled rows, None
elsewhere. Sampling is ``rules.sample_tracklet_rows`` — the same function the
role/team stage uses to decide which rows carry a position — so the two stages
cannot disagree about which crops belong to a tracklet.

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
    input_columns = ["track_id", "bbox_ltwh", "image_id"]
    output_columns = ["team_embedding"]

    def __init__(self, cfg, device, tracking_dataset=None, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.stride = int(getattr(cfg, "pos_stride", rules.POS_STRIDE))
        self.crops_per_track = int(getattr(cfg, "crops_per_track", rules.CROPS_PER_TRK))
        self.batch_size = int(getattr(cfg, "batch_size", 128))
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
                      tracklets=0, tracklets_off_grid=0, crops_sampled=0, crops_embedded=0,
                      crops_empty=0, frames_read=0, embedder=None)
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
            frames = fidx.reindex(grp["image_id"].to_numpy()).to_numpy()
            if np.isnan(frames.astype(float)).any():
                raise RuntimeError(f"[team_embed] {seq}: detection image_id without frame metadata")
            _, crop_rows, off_grid = rules.sample_tracklet_rows(frames, self.stride, self.crops_per_track)
            record["tracklets"] += 1
            record["tracklets_off_grid"] += int(off_grid)
            for r in grp.index[crop_rows]:
                rows_by_image.setdefault(grp.at[r, "image_id"], []).append(r)
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
        log.info(f"[team_embed] {seq}: {record['tracklets']} tracklets, {record['crops_embedded']} crops "
                 f"embedded from {record['frames_read']} frames "
                 f"({record['crops_empty']} empty, {record['tracklets_off_grid']} off-grid tracklets)")
        self._write(record)
        return out

    def _write(self, record):
        if self.audit_dir:
            (self.audit_dir / f"{record['sequence']}.json").write_text(json.dumps(record, indent=2, default=str))
