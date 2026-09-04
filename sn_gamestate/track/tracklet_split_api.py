"""Tracklet splitting as a pipeline stage (``tracklet_split``).

Runs directly after ``track`` and ``crop_filter`` and before ``calibration``;
it writes the ``track_id`` every later stage groups by. This stage is Stage 1
of the refinement method and NOTHING ELSE: the pipeline's one and only merge
is the label-aware ``traj_refine`` stage. The algorithm is in
``sn_gamestate/track/tracklet_split.py``; this module supplies its inputs and
applies its output:

1. Appearance: one OSNet-AIN embedding per tracked detection, from the same
   shared module and checkpoint pin as the tracker
   (``sn_gamestate/reid/osnet_ain``), so a crop embeds to the same vector in
   both stages. A detection whose box clamps to nothing, or whose frame cannot
   be read, keeps an all-zero feature; it is at cosine distance 1 from
   everything and its placement falls to the deterministic tie rules.
2. Clean/overlapping label: the crop filter's ``crop_single`` column, read,
   never recomputed.
3. Split, per tracklet: DBSCAN over ALL detections; noise (single or multi
   crop) assigned to the nearest fragment centroid; all-multi fragments
   dissolved, detection by detection, into the nearest remaining fragment;
   centroids clean-only throughout (see the algorithm module).
4. Relabel: fragments become the new trajectories, numbered ``1..T`` in
   (source tracklet, fragment) order. EVERY tracked detection stays assigned
   -- the splitter never unassigns a row and never merges fragments. The
   incoming id is kept per row in ``track_id_presplit`` so the audit can
   verify each fragment has exactly one source tracklet.

By construction the output holds at most one detection per (``image_id``,
``track_id``) -- fragments partition tracklets, which hold one detection per
frame (validated; a violation raises). A fragment may hold no clean detection
only in the degenerate all-multi-tracklet case; the count is reported, not
hidden.

Audit sidecar: ``<audit_dir>/<sequence>.json`` with the settings and embedder
digest that ran, input counts, the per-tracklet split report and the output
counts. The run audit compares it with the config and the detections.
"""
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from tracklab.pipeline.videolevel_module import VideoLevelModule
from tracklab.utils.cv2 import cv2_load_image

from sn_gamestate.reid import osnet_ain
from sn_gamestate.track import tracklet_split as ts

log = logging.getLogger(__name__)


def sequence_name(metadatas: pd.DataFrame) -> str:
    """Same rule as the audit and the team_embed stage (``SNGS-xxx`` from the
    frame path when available, else ``video_id``)."""
    if len(metadatas) and "file_path" in metadatas.columns:
        return Path(str(metadatas["file_path"].iloc[0])).parent.parent.name
    if len(metadatas) and "video_id" in metadatas.columns:
        return str(metadatas["video_id"].iloc[0])
    return "unknown"


class TrackletSplit(VideoLevelModule):
    input_columns = ["track_id", "bbox_ltwh", "image_id", "crop_single"]
    output_columns = ["track_id", "track_id_presplit"]

    def __init__(self, cfg, device, tracking_dataset=None, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.eps = float(cfg.eps)
        self.min_samples = int(cfg.min_samples)
        self.batch_size = int(getattr(cfg, "batch_size", 64))
        if not (0.0 < self.eps <= 2.0):
            raise ValueError(f"[tracklet_split] eps must be in (0, 2], got {self.eps}")
        if self.min_samples < 2:
            raise ValueError(f"[tracklet_split] min_samples must be >= 2, "
                             f"got {self.min_samples}")
        self.audit_dir = Path(str(cfg.audit_dir)) if getattr(cfg, "audit_dir", None) else None
        if self.audit_dir:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
        # The tracker's appearance model: same module, same checkpoint pin, same
        # preprocessing and fp16-autocast arithmetic. A pin mismatch between the
        # two module configs is a run-audit FAIL.
        self.embedder = osnet_ain.from_config(cfg, device, batch_size=self.batch_size)
        log.info(f"[tracklet_split] eps {self.eps}, min_samples {self.min_samples}; "
                 f"embedder {self.embedder.info.get('sha256', '?')[:16]} "
                 f"({self.embedder.info.get('precision')}); split only, the "
                 f"pipeline's one merge is traj_refine")

    # ------------------------------------------------------------------ features
    @torch.no_grad()
    def _extract_features(self, dets: pd.DataFrame, metadatas: pd.DataFrame,
                          record: dict) -> np.ndarray:
        """Appearance feature per row, aligned to ``dets.index`` (zeros on failure)."""
        feats = np.zeros((len(dets), self.embedder.dim), dtype=np.float32)
        id2path = (metadatas["file_path"].to_dict()
                   if "file_path" in metadatas.columns else {})
        pos = {idx: i for i, idx in enumerate(dets.index)}
        n_missing_path = n_unreadable = n_degenerate = 0

        for image_id, group in dets.groupby("image_id"):
            path = id2path.get(image_id)
            if path is None:
                n_missing_path += 1
                continue
            img = cv2_load_image(path)
            if img is None:
                n_unreadable += 1
                continue
            # cv2_load_image returns RGB; the OSNet-AIN preprocessing does
            # BGR2RGB itself, so it must be handed BGR.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            crops, rows = [], []
            for idx, det in group.iterrows():
                l, t, w, h = [float(v) for v in det["bbox_ltwh"]]
                patch = osnet_ain.crop_ltrb(img, (l, t, l + w, t + h))
                if patch is None:
                    n_degenerate += 1
                    continue
                crops.append(patch)
                rows.append(idx)
            if crops:
                for r, f in zip(rows, self.embedder.embed(crops)):
                    feats[pos[r]] = f

        n_zero = int((~feats.any(axis=1)).sum()) if len(feats) else 0
        record["inputs"].update(frames_without_path=n_missing_path,
                                frames_unreadable=n_unreadable,
                                degenerate_boxes=n_degenerate,
                                zero_embeddings=n_zero)
        if n_missing_path or n_unreadable:
            log.warning(f"[tracklet_split] skipped {n_missing_path} frame(s) with no "
                        f"file_path and {n_unreadable} unreadable frame(s); their "
                        f"detections keep zero features.")
        if len(feats) and n_zero == len(feats):
            raise RuntimeError(
                f"[tracklet_split] all {len(feats)} detection crops produced an "
                f"all-zero appearance feature - the appearance model or the frame "
                f"paths are broken; refusing to split tracklets on empty embeddings.")
        if len(feats) and n_zero > 0.05 * len(feats):
            log.warning(f"[tracklet_split] {n_zero}/{len(feats)} "
                        f"({n_zero / len(feats):.1%}) detections have an all-zero "
                        f"appearance feature")
        return feats

    # ------------------------------------------------------------------ main
    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        seq = sequence_name(metadatas)
        record = dict(sequence=seq, ran=False,
                      settings=dict(eps=self.eps, min_samples=self.min_samples,
                                    batch_size=self.batch_size),
                      embedder=dict(self.embedder.info),
                      inputs=dict(detections=int(len(detections)), tracked=0,
                                  tracklets=0,
                                  crop_single_present=bool("crop_single" in detections.columns),
                                  single_tracked=0, multi_tracked=0),
                      outputs=dict(fragments=0, rows_assigned=0,
                                   frame_collisions=0,
                                   fragments_without_clean=0))
        out = detections.copy()
        out["track_id_presplit"] = out["track_id"] if "track_id" in out.columns else np.nan
        if len(detections) == 0 or "track_id" not in detections.columns:
            log.warning(f"[tracklet_split] {seq}: no detections or no track_id "
                        f"column; nothing to do")
            self._write(record)
            return out
        if "crop_single" not in detections.columns:
            raise RuntimeError(f"[tracklet_split] {seq}: crop_single column missing - "
                               f"the crop_filter stage must run before tracklet_split")

        work = out[out["track_id"].notna()].copy()
        work = work.sort_values(["track_id", "image_id"], kind="stable")
        record["inputs"]["tracked"] = int(len(work))
        record["inputs"]["tracklets"] = int(work["track_id"].nunique())
        if len(work) == 0:
            log.warning(f"[tracklet_split] {seq}: no tracked detection; nothing to do")
            self._write(record)
            return out

        single = work["crop_single"].astype(bool).to_numpy()
        frames = work["image_id"].to_numpy(dtype=np.int64)
        tids = work["track_id"].to_numpy(dtype=np.float64)
        if not np.all(tids == np.round(tids)):
            raise RuntimeError(f"[tracklet_split] {seq}: non-integer track_id values")
        tids = tids.astype(np.int64)
        record["inputs"]["single_tracked"] = int(single.sum())
        record["inputs"]["multi_tracked"] = int((~single).sum())
        record["inputs"]["tracklets_without_single"] = int(
            sum(1 for _, g in work.groupby("track_id")
                if not g["crop_single"].astype(bool).any()))

        feats = self._extract_features(work, metadatas, record)
        frag, per_tracklet = ts.split_video(feats, single, frames, tids,
                                            self.eps, self.min_samples)
        record["ran"] = True

        # Relabel: fragments -> 1..T in (source tracklet, fragment) order.
        uniq = sorted(int(f) for f in np.unique(frag))
        newid = {f: i + 1 for i, f in enumerate(uniq)}
        out.loc[work.index, "track_id"] = [float(newid[int(f)]) for f in frag]

        # Self-checks on the structural guarantees of the split.
        tracked_out = out[out["track_id"].notna()]
        coll = int(tracked_out.duplicated(subset=["image_id", "track_id"]).sum())
        if coll:
            log.error(f"[tracklet_split] {seq}: {coll} (image_id, track_id) "
                      f"collision(s) in the output; fragments partition tracklets, "
                      f"so the bookkeeping is broken")
        if int(tracked_out["track_id"].notna().sum()) != len(work):
            log.error(f"[tracklet_split] {seq}: tracked row count changed - the "
                      f"splitter must only relabel")
        origin = tracked_out.groupby("track_id")["track_id_presplit"].nunique()
        n_multi_origin = int((origin > 1).sum())
        if n_multi_origin:
            log.error(f"[tracklet_split] {seq}: {n_multi_origin} fragment(s) mix "
                      f"detections from more than one source tracklet")
        clean_per = tracked_out.groupby("track_id")["crop_single"].apply(
            lambda s: bool(s.astype(bool).any()))
        n_noclean = int((~clean_per).sum())

        record["split"] = dict(
            per_tracklet=per_tracklet,
            fragments=int(len(uniq)),
            tracklets_split=int(sum(1 for p in per_tracklet if p["k"] > 1)),
            noise=int(sum(p["noise"] for p in per_tracklet)),
            dissolved_allmulti=int(sum(p["dissolved_allmulti"] for p in per_tracklet)))
        record["outputs"].update(fragments=int(tracked_out["track_id"].nunique()),
                                 rows_assigned=int(len(work)),
                                 frame_collisions=coll,
                                 fragments_multi_origin=n_multi_origin,
                                 fragments_without_clean=n_noclean)
        log.info(f"[tracklet_split] {seq}: {record['inputs']['tracklets']} tracklets "
                 f"-> {record['split']['fragments']} fragments "
                 f"({record['split']['tracklets_split']} split, "
                 f"{record['split']['noise']} noise attached, "
                 f"{record['split']['dissolved_allmulti']} all-multi dissolved, "
                 f"{n_noclean} fragment(s) without a clean detection); no merging "
                 f"here - traj_refine is the pipeline's one merge")
        self._write(record)
        return out

    def _write(self, record):
        if self.audit_dir is None:
            return
        try:
            (self.audit_dir / f"{record['sequence']}.json").write_text(
                json.dumps(record, indent=2, default=str))
        except Exception as exc:               # never break a run over telemetry
            log.warning(f"[tracklet_split] could not write the audit sidecar: {exc}")
