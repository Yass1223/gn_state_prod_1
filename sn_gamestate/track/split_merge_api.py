"""Tracklet splitting and fragment merging as a pipeline stage (``split_merge``).

Runs directly after ``track`` and ``crop_filter`` and before ``calibration``; it
writes the final ``track_id`` every later stage groups by. The algorithm is in
``sn_gamestate/track/split_merge.py`` (a port of the production part of
``splitter-plus-merger-final.ipynb``); this module supplies its inputs from the
pipeline and turns its output into ``track_id``:

1. Appearance: one OSNet-AIN embedding per tracked detection, from the same
   shared module and checkpoint pin as the tracker (``sn_gamestate/reid/osnet_ain``),
   so a crop embeds to the same vector in both stages. A detection whose box
   clamps to nothing, or whose frame cannot be read, keeps an all-zero feature
   and takes no part in clustering or merging.
2. Clean/overlapping label: the crop filter's ``crop_single`` column, as
   produced upstream (rT <= 0.25, rB < 0.40, tracked-only contaminators). It is
   read, never recomputed.
3. Frame key: ``image_id``. The method only tests frame equality (two
   detections share a frame iff their keys are equal), so no frame numbering is
   needed.
4. Split (DBSCAN per tracklet) -> merge (agglomerative, appearance, disjoint
   frame sets, threshold ``tau``) -> pass 1 (one detection per trajectory and
   frame) -> pass 2 (placement of set-aside and clean-less detections).
5. Relabel: trajectory ``t`` becomes ``track_id = t + 1``; a detection the
   method could not place gets ``track_id = NaN`` (every downstream stage groups
   by ``track_id`` and drops NaN, and the evaluation encoding drops it too).
   Untracked detections stay NaN.

By construction the output holds at most one detection per (``image_id``,
``track_id``) and every trajectory contains at least one clean detection; the
stage checks both on its own output and logs an error if either fails.

Audit sidecar
-------------
Per video the stage writes ``<audit_dir>/<sequence>.json`` with the settings and
embedder digest that ran, the input counts (tracked detections, tracklets, clean
and overlapping detections, zero embeddings), the split report (fragments per
tracklet, noise points), the merge report (ordinary and clean-less fragments,
merges and their distances, trajectories), the two passes (set aside, placed,
unassigned), the output counts and a per-trajectory breakdown (source
tracklets, fragments, rows, clean rows). The run audit compares it with the
config and with the detections it receives.
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
from sn_gamestate.track import split_merge as sm

log = logging.getLogger(__name__)


def sequence_name(metadatas: pd.DataFrame) -> str:
    """Same rule as the audit and the team_embed stage (``SNGS-xxx`` from the frame
    path when available, else ``video_id``), so the sidecar name matches."""
    if len(metadatas) and "file_path" in metadatas.columns:
        return Path(str(metadatas["file_path"].iloc[0])).parent.parent.name
    if len(metadatas) and "video_id" in metadatas.columns:
        return str(metadatas["video_id"].iloc[0])
    return "unknown"


class SplitMerge(VideoLevelModule):
    input_columns = ["track_id", "bbox_ltwh", "image_id", "crop_single"]
    output_columns = ["track_id"]

    def __init__(self, cfg, device, tracking_dataset=None, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.eps = float(cfg.eps)
        self.min_samples = int(cfg.min_samples)
        self.tau = float(cfg.tau)
        self.batch_size = int(getattr(cfg, "batch_size", 64))
        if not (0.0 < self.eps <= 2.0):
            raise ValueError(f"[split_merge] eps must be in (0, 2], got {self.eps}")
        if self.min_samples < 2:
            raise ValueError(f"[split_merge] min_samples must be >= 2, got {self.min_samples}")
        if not (0.0 <= self.tau <= 2.0):
            raise ValueError(f"[split_merge] tau must be in [0, 2], got {self.tau}")
        self.audit_dir = Path(str(cfg.audit_dir)) if getattr(cfg, "audit_dir", None) else None
        if self.audit_dir:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
        # The tracker's appearance model: same module, same checkpoint pin, same
        # letterbox preprocessing and fp16-autocast arithmetic. A pin mismatch
        # between the two module configs is a run-audit FAIL.
        self.embedder = osnet_ain.from_config(cfg, device, batch_size=self.batch_size)
        log.info(f"[split_merge] eps {self.eps}, min_samples {self.min_samples}, "
                 f"tau {self.tau}; embedder {self.embedder.info.get('sha256', '?')[:16]} "
                 f"({self.embedder.info.get('precision')})")

    # ------------------------------------------------------------------ features
    @torch.no_grad()
    def _extract_features(self, dets: pd.DataFrame, metadatas: pd.DataFrame, record: dict) -> np.ndarray:
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
            # cv2_load_image returns RGB; the OSNet-AIN preprocessing does BGR2RGB
            # itself, so it must be handed BGR or every channel arrives swapped.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            crops, rows = [], []
            for idx, det in group.iterrows():
                l, t, w, h = [float(v) for v in det["bbox_ltwh"]]
                patch = osnet_ain.crop_ltrb(img, (l, t, l + w, t + h))
                if patch is None:
                    n_degenerate += 1
                    continue  # degenerate box: zero feature, never clustered or merged
                crops.append(patch)
                rows.append(idx)
            if crops:
                for r, f in zip(rows, self.embedder.embed(crops)):
                    feats[pos[r]] = f

        n_zero = int((~feats.any(axis=1)).sum()) if len(feats) else 0
        record["inputs"].update(frames_without_path=n_missing_path, frames_unreadable=n_unreadable,
                                degenerate_boxes=n_degenerate, zero_embeddings=n_zero)
        if n_missing_path or n_unreadable:
            log.warning(f"[split_merge] skipped {n_missing_path} frame(s) with no file_path and "
                        f"{n_unreadable} unreadable frame(s); their detections keep zero features.")
        if len(feats) and n_zero == len(feats):
            raise RuntimeError(
                f"[split_merge] all {len(feats)} detection crops produced an all-zero "
                f"appearance feature - the appearance model or the frame paths are broken; "
                f"refusing to split or merge tracklets on empty embeddings.")
        if len(feats) and n_zero > 0.05 * len(feats):
            log.warning(f"[split_merge] {n_zero}/{len(feats)} ({n_zero / len(feats):.1%}) "
                        f"detections have an all-zero appearance feature; they are excluded "
                        f"from clustering and merging and can only be placed in pass 2.")
        return feats

    # ------------------------------------------------------------------ main
    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        seq = sequence_name(metadatas)
        record = dict(sequence=seq, ran=False,
                      settings=dict(eps=self.eps, min_samples=self.min_samples, tau=self.tau,
                                    batch_size=self.batch_size),
                      embedder=dict(self.embedder.info),
                      inputs=dict(detections=int(len(detections)), tracked=0, tracklets=0,
                                  crop_single_present=bool("crop_single" in detections.columns),
                                  single_tracked=0, multi_tracked=0),
                      outputs=dict(tracklets=0, rows_assigned=0, rows_unassigned=0,
                                   frame_collisions=0, trajectories_without_clean=0))
        if len(detections) == 0 or "track_id" not in detections.columns:
            log.warning(f"[split_merge] {seq}: no detections or no track_id column; nothing to do")
            self._write(record)
            return detections
        if "crop_single" not in detections.columns:
            raise RuntimeError(f"[split_merge] {seq}: crop_single column missing - the crop_filter "
                               f"stage must run before split_merge")

        work = detections[detections["track_id"].notna()].copy()
        work = work.sort_values(["track_id", "image_id"], kind="stable")
        record["inputs"]["tracked"] = int(len(work))
        record["inputs"]["tracklets"] = int(work["track_id"].nunique())
        if len(work) == 0:
            log.warning(f"[split_merge] {seq}: no tracked detection; nothing to do")
            self._write(record)
            return detections

        single = work["crop_single"].astype(bool).to_numpy()
        frames = work["image_id"].to_numpy(dtype=np.int64)
        tids = work["track_id"].to_numpy(dtype=np.float64)
        if not np.all(tids == np.round(tids)):
            raise RuntimeError(f"[split_merge] {seq}: non-integer track_id values")
        tids = tids.astype(np.int64)
        record["inputs"]["single_tracked"] = int(single.sum())
        record["inputs"]["multi_tracked"] = int((~single).sum())
        record["inputs"]["tracklets_without_single"] = int(
            sum(1 for _, g in work.groupby("track_id") if not g["crop_single"].astype(bool).any()))

        feats = self._extract_features(work, metadatas, record)
        traj, rep = sm.split_merge_video(feats, single, frames, tids,
                                         self.eps, self.min_samples, self.tau)
        record["ran"] = True

        # Relabel: trajectory t -> t + 1; unplaced -> NaN; untracked rows stay NaN.
        new_ids = pd.Series(np.nan, index=detections.index, dtype=float)
        assigned = traj >= 0
        new_ids.loc[work.index[assigned]] = traj[assigned].astype(float) + 1.0
        out = detections.copy()
        out["track_id"] = new_ids

        # Self-checks on the structural guarantees of pass 1 and of the merge.
        tracked_out = out[out["track_id"].notna()]
        coll = int(tracked_out.duplicated(subset=["image_id", "track_id"]).sum())
        clean_per_traj = tracked_out.groupby("track_id")["crop_single"].apply(lambda s: bool(s.astype(bool).any()))
        no_clean = int((~clean_per_traj).sum())
        if coll:
            log.error(f"[split_merge] {seq}: {coll} (image_id, track_id) collision(s) in the "
                      f"output; pass 1 guarantees none, so the bookkeeping is broken")
        if no_clean:
            log.error(f"[split_merge] {seq}: {no_clean} trajector(ies) without a clean detection; "
                      f"only fragments with a clean detection can found a trajectory")

        # Sidecar content.
        per_tracklet = rep["split"].pop("per_tracklet")
        record["split"] = dict(rep["split"], per_tracklet=per_tracklet)
        mi = rep["merge"]
        record["merge"] = dict(ordinary_fragments=mi["n_ordinary"], allmulti_fragments=mi["n_allmulti"],
                               fragments=mi["n_fragments"], merges=mi["n_merges"],
                               trajectories=mi["n_traj"],
                               merge_distances=[round(d, 4) for d in mi["merge_distances"]])
        record["pass1"] = dict(set_aside=rep["pass1"]["discarded"],
                               single_anomaly=rep["pass1"]["single_anomaly"])
        record["pass2"] = dict(rep["pass2"])
        record["outputs"].update(tracklets=int(tracked_out["track_id"].nunique()),
                                 rows_assigned=int(assigned.sum()),
                                 rows_unassigned=int((~assigned).sum()),
                                 frame_collisions=coll, trajectories_without_clean=no_clean)
        frag = rep["frag"]
        per_traj = []
        for t, frs in sorted(mi["frags_of_traj"].items()):
            rows = np.where(traj == t)[0]
            src = sorted(set(int(f) // sm.FRAG_BASE for f in frs))
            per_traj.append(dict(track_id=int(t) + 1, n_fragments=int(len(frs)),
                                 source_track_ids=src, n_rows=int(len(rows)),
                                 n_single=int(single[rows].sum()),
                                 n_rows_from_own_fragments=int(np.isin(frag[rows], frs).sum())))
        record["per_trajectory"] = per_traj

        if rep["pass1"]["single_anomaly"]:
            log.warning(f"[split_merge] {seq}: {rep['pass1']['single_anomaly']} clean detection(s) "
                        f"shared a frame inside one trajectory (set aside)")
        log.info(f"[split_merge] {seq}: {record['inputs']['tracklets']} tracklets -> "
                 f"{record['split']['fragments']} fragments ({record['split']['tracklets_split']} split) -> "
                 f"{record['merge']['trajectories']} trajectories ({record['merge']['merges']} merges); "
                 f"pass 1 set aside {record['pass1']['set_aside']}, pass 2 placed "
                 f"{record['pass2']['placed']}, unassigned {record['pass2']['unassigned']}; "
                 f"zero embeddings {record['inputs']['zero_embeddings']}")
        self._write(record)
        return out

    def _write(self, record):
        if self.audit_dir is None:
            return
        try:
            (self.audit_dir / f"{record['sequence']}.json").write_text(
                json.dumps(record, indent=2, default=str))
        except Exception as exc:                       # never break a run over telemetry
            log.warning(f"[split_merge] could not write the audit sidecar: {exc}")
