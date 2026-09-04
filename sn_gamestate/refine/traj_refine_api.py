"""Label-aware trajectory refinement as a pipeline stage (``traj_refine``).

Runs after ``jersey_number_detect`` and before ``tracklet_agg``: the merges it
performs need the team side (``role_team``), the jersey number with its pooled
maxconf candidate statistics (``jn_gsr_api``), and the exit/entry geometry --
evidence that does not exist at ``split_merge``'s position. The algorithm is in
``sn_gamestate/refine/traj_refine.py``; this module supplies its inputs and
applies its output:

1. Appearance: one OSNet-AIN embedding per tracked detection, from the same
   shared module and checkpoint pin as the tracker and ``split_merge``
   (``sn_gamestate/reid/osnet_ain``), so ``tau`` keeps the cosine-distance
   scale it was tuned on. A degenerate box or unreadable frame keeps an
   all-zero feature; a trajectory with no usable embedding never merges.
2. Per-trajectory labels: ``team`` (first non-null; per-track constant by the
   role_team contract), ``jersey_number_detection`` (idem), the jersey stage's
   ``jersey_number_candidates`` (pooled ``[label, mx, conf_sum, votes]``
   stats), and scope = per-track modal ``role`` in ``cfg.roles``.
3. Chronology: the dataset's frame index (``team_embed_api.frame_index``), so
   frame equality and time order are the dataset's, not ``image_id`` order.
   The image width for the re-enter test is read off the first frame the
   feature extraction loads.
4. Refine: phase 2a (same-team same-number merges in descending pair maxconf;
   overlap conflicts resolved by maxconf, the loser walking down its candidate
   list) then phase 2b (agglomerative merging under time-disjointness,
   re-enter consistency and vacuous-when-unknown team/number agreement),
   both within ``tau``.
5. Apply: rows of a merged cluster take the smallest constituent ``track_id``.
   For every in-scope cluster the resolved number, its vote share and its
   maxconf score are written to ALL member rows (so the downstream majority
   vote cannot be flipped by row counts); a merged cluster additionally
   unifies ``team`` (the known side), ``role`` (row majority, tie -> player)
   and ``team_cluster`` (modal non-null value). Referee trajectories and
   unassigned rows are untouched.

Snapshots for the audit: the incoming ids and jersey columns are kept in
``track_id_prerefine``, ``jersey_number_detection_prerefine`` and
``jersey_number_confidence_prerefine`` (the pattern ``pitch_gate`` uses with
``track_id_pregate``), so every earlier stage can still be audited against the
state it actually produced. With ``cfg.enabled: false`` the stage only writes
the snapshots and its sidecar -- the attribution switch for A/B metric runs.

Stage 3 (duplicate-frame resolution) runs inside this stage after the merger:
merging requires disjointness of CLEAN frames only (multi-player detections
are ignored until here), so multi rows can collide; 3a keeps one detection per
(frame, trajectory) -- the clean one when present, else the multi nearest the
clean-first centroid -- and 3b places the held rows into the nearest in-scope
trajectory with a free frame slot, unassigning (track_id -> NaN) the rest. The
output therefore holds at most one detection per (``image_id``, ``track_id``)
over ALL detections; the stage checks this and per-cluster label constancy on
its own output and logs an error if either fails. Tracked rows out = tracked
rows in minus the unassigned count -- the ONLY way this stage drops a row.

Audit sidecar: ``<audit_dir>/<sequence>.json`` with the settings and embedder
digest that ran, input counts, the merge/conflict/rejection log of both
phases, output counts and a per-cluster breakdown. The run audit compares it
with the composed config and the detections it receives.
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

from sn_gamestate.refine import traj_refine as tr
from sn_gamestate.reid import osnet_ain
from sn_gamestate.team.team_embed_api import frame_index, sequence_name

log = logging.getLogger(__name__)


def _is_null(v):
    if v is None:
        return True
    if isinstance(v, (list, tuple, np.ndarray, dict, str)):
        return False
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


class TrajRefine(VideoLevelModule):
    input_columns = ["track_id", "bbox_ltwh", "image_id", "crop_single",
                     "role", "team", "jersey_number_detection",
                     "jersey_number_confidence", "jersey_number_candidates"]
    output_columns = ["track_id", "track_id_prerefine",
                      "jersey_number_detection", "jersey_number_confidence",
                      "jersey_number_maxconf",
                      "jersey_number_detection_prerefine",
                      "jersey_number_confidence_prerefine",
                      "role", "team", "team_cluster"]

    def __init__(self, cfg, device, tracking_dataset=None, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.enabled = bool(getattr(cfg, "enabled", True))
        self.tau = float(cfg.tau)
        self.use_reenter = bool(getattr(cfg, "use_reenter", True))
        self.edge_margin = float(getattr(cfg, "edge_margin", 0.02))
        self.roles = [str(r) for r in getattr(cfg, "roles", ["player", "goalkeeper"])]
        self.batch_size = int(getattr(cfg, "batch_size", 64))
        if not (0.0 <= self.tau <= 2.0):
            raise ValueError(f"[traj_refine] tau must be in [0, 2], got {self.tau}")
        if not (0.0 <= self.edge_margin < 0.5):
            raise ValueError(f"[traj_refine] edge_margin must be in [0, 0.5), "
                             f"got {self.edge_margin}")
        self.audit_dir = Path(str(cfg.audit_dir)) if getattr(cfg, "audit_dir", None) else None
        if self.audit_dir:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
        # Same appearance model, pin and arithmetic as track / split_merge; a
        # pin mismatch against split_merge is a run-audit FAIL. Built lazily so
        # a disabled stage costs nothing.
        self._embedder = None
        log.info(f"[traj_refine] enabled {self.enabled}, tau {self.tau}, "
                 f"use_reenter {self.use_reenter}, edge_margin {self.edge_margin}, "
                 f"roles {self.roles}")

    def _model(self):
        if self._embedder is None:
            self._embedder = osnet_ain.from_config(self.cfg, self.device,
                                                   batch_size=self.batch_size)
        return self._embedder

    # ------------------------------------------------------------- features --
    @torch.no_grad()
    def _extract_features(self, dets: pd.DataFrame, metadatas: pd.DataFrame,
                          record: dict):
        """(features aligned to ``dets.index``, image width or None)."""
        model = self._model()
        record["embedder"] = dict(model.info)
        feats = np.zeros((len(dets), model.dim), dtype=np.float32)
        id2path = (metadatas["file_path"].to_dict()
                   if "file_path" in metadatas.columns else {})
        pos = {idx: i for i, idx in enumerate(dets.index)}
        n_missing_path = n_unreadable = n_degenerate = 0
        img_w = None

        for image_id, group in dets.groupby("image_id"):
            path = id2path.get(image_id)
            if path is None:
                n_missing_path += 1
                continue
            img = cv2_load_image(path)
            if img is None:
                n_unreadable += 1
                continue
            if img_w is None:
                img_w = float(img.shape[1])
            # cv2_load_image returns RGB; OSNet-AIN preprocessing expects BGR.
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
                for r, f in zip(rows, model.embed(crops)):
                    feats[pos[r]] = f

        n_zero = int((~feats.any(axis=1)).sum()) if len(feats) else 0
        record["inputs"].update(frames_without_path=n_missing_path,
                                frames_unreadable=n_unreadable,
                                degenerate_boxes=n_degenerate,
                                zero_embeddings=n_zero, img_w=img_w)
        if n_missing_path or n_unreadable:
            log.warning(f"[traj_refine] skipped {n_missing_path} frame(s) with no "
                        f"file_path and {n_unreadable} unreadable frame(s)")
        if len(feats) and n_zero == len(feats):
            raise RuntimeError(
                f"[traj_refine] all {len(feats)} detection crops produced an "
                f"all-zero appearance feature - the appearance model or the frame "
                f"paths are broken; refusing to merge trajectories on empty "
                f"embeddings.")
        if img_w is None and self.use_reenter:
            log.warning("[traj_refine] no frame could be read; the re-enter "
                        "condition is vacuous for this sequence")
        return feats, img_w

    # --------------------------------------------------------------- tracks --
    def _track_info(self, work: pd.DataFrame, record: dict):
        tracks = {}
        for tid, grp in work.groupby("_tid"):
            role_mode = grp["role"].mode() if "role" in grp.columns else pd.Series(dtype=object)
            role = role_mode.iloc[0] if len(role_mode) else None
            teams = [v for v in grp["team"] if not _is_null(v)] if "team" in grp.columns else []
            numbers = [v for v in grp["jersey_number_detection"] if not _is_null(v)]
            cand = next((v for v in grp["jersey_number_candidates"]
                         if isinstance(v, (list, tuple)) and len(v)), None)
            tracks[int(tid)] = dict(
                team=str(teams[0]) if teams else None,
                number=str(numbers[0]) if numbers else None,
                cand=cand or [],
                scope=role in self.roles)
        record["inputs"]["in_scope"] = sum(1 for t in tracks.values() if t["scope"])
        record["inputs"]["numbered"] = sum(1 for t in tracks.values() if t["number"])
        record["inputs"]["with_team"] = sum(1 for t in tracks.values() if t["team"])
        return tracks

    # ------------------------------------------------------------------ main --
    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        seq = sequence_name(metadatas)
        out = detections.copy()
        # Snapshots first, unconditionally: the audit compares every earlier
        # stage against the state it actually produced.
        out["track_id_prerefine"] = out["track_id"]
        out["jersey_number_detection_prerefine"] = out["jersey_number_detection"]
        out["jersey_number_confidence_prerefine"] = out["jersey_number_confidence"]
        if "jersey_number_maxconf" not in out.columns:
            out["jersey_number_maxconf"] = 0.0

        record = dict(sequence=seq, ran=False,
                      settings=dict(enabled=self.enabled, tau=self.tau,
                                    use_reenter=self.use_reenter,
                                    edge_margin=self.edge_margin,
                                    roles=list(self.roles)),
                      embedder=None,
                      inputs=dict(detections=int(len(detections)), tracked=0,
                                  tracklets=0),
                      outputs=dict(tracklets=0, merges=0, merges_2a=0,
                                   merges_2b=0, conflicts=0, rejected_2a=0,
                                   rows_relabelled=0, rows_unassigned=0,
                                   frame_collisions=0))
        if not self.enabled:
            log.info(f"[traj_refine] {seq}: disabled; snapshots written, "
                     f"nothing changed")
            self._write(record)
            return out
        if len(detections) == 0 or "track_id" not in detections.columns:
            log.warning(f"[traj_refine] {seq}: no detections or no track_id "
                        f"column; nothing to do")
            self._write(record)
            return out
        for col in ("crop_single", "role", "team", "jersey_number_detection",
                    "jersey_number_candidates"):
            if col not in detections.columns:
                raise RuntimeError(f"[traj_refine] {seq}: {col} column missing - "
                                   f"the stage must run after crop_filter, "
                                   f"role_team and jersey_number_detect")

        work = out[out["track_id"].notna()].copy()
        record["inputs"]["tracked"] = int(len(work))
        record["inputs"]["tracklets"] = int(work["track_id"].nunique())
        if len(work) == 0:
            log.warning(f"[traj_refine] {seq}: no tracked detection; nothing to do")
            self._write(record)
            return out

        tids = work["track_id"].to_numpy(dtype=np.float64)
        if not np.all(tids == np.round(tids)):
            raise RuntimeError(f"[traj_refine] {seq}: non-integer track_id values")
        work["_tid"] = tids.astype(np.int64)
        fidx = frame_index(metadatas)
        frames = fidx.reindex(work["image_id"].to_numpy()).to_numpy(dtype=float)
        if np.isnan(frames).any():
            raise RuntimeError(f"[traj_refine] {seq}: detection image_id without "
                               f"frame metadata")
        frames = frames.astype(np.int64)
        boxes = np.stack([np.asarray(b, dtype=np.float64)
                          for b in work["bbox_ltwh"]])
        single = work["crop_single"].astype(bool).to_numpy()
        tracks = self._track_info(work, record)
        feats, img_w = self._extract_features(work, metadatas, record)

        new_tid, resolved, rep = tr.refine_video(
            feats, single, frames, boxes, work["_tid"].to_numpy(), tracks,
            img_w, self.tau, self.use_reenter, self.edge_margin)
        record["ran"] = True

        # ----------------------------------------------------------- apply --
        n_relabel = 0
        un_rows = work.index[new_tid < 0]
        out.loc[un_rows, "track_id"] = np.nan          # stage-3b unassigned
        for key, res in resolved.items():
            rows = work.index[new_tid == key]          # incl. rows adopted in 3b
            changed = int((work.loc[rows, "_tid"] != key).sum())
            n_relabel += changed
            if changed or len(res["tids"]) > 1:
                out.loc[rows, "track_id"] = float(key)
            if not tracks[min(res["tids"])]["scope"]:
                continue                      # out of scope: rows untouched
            merged = len(res["tids"]) > 1 or changed > 0
            out.loc[rows, "jersey_number_detection"] = res["number"]
            out.loc[rows, "jersey_number_confidence"] = res["confidence"]
            out.loc[rows, "jersey_number_maxconf"] = res["maxconf"]
            if merged:
                if res["team"] is not None:
                    out.loc[rows, "team"] = res["team"]
                roles = out.loc[rows, "role"].mode()
                role = roles.iloc[0] if len(roles) == 1 else "player"
                out.loc[rows, "role"] = role
                tc = out.loc[rows, "team_cluster"]
                tc = tc[tc.notna()]
                out.loc[rows, "team_cluster"] = (float(tc.mode().iloc[0])
                                                 if len(tc) else np.nan)

        # ------------------------------------------------------ self-checks --
        tracked_out = out[out["track_id"].notna()]
        coll = int(tracked_out.duplicated(subset=["image_id", "track_id"]).sum())
        if coll:
            log.error(f"[traj_refine] {seq}: {coll} (image_id, track_id) "
                      f"collision(s) in the output; time-disjointness is a merge "
                      f"condition, so the bookkeeping is broken")
        n_incoherent = 0
        for tid, grp in tracked_out.groupby("track_id"):
            for col in ("jersey_number_detection", "team", "role"):
                vals = {str(v) for v in grp[col] if not _is_null(v)}
                if len(vals) > 1:
                    n_incoherent += 1
                    break
        if n_incoherent:
            log.error(f"[traj_refine] {seq}: {n_incoherent} cluster(s) with a "
                      f"non-constant number/team/role after propagation")
        n_unassigned = int((new_tid < 0).sum())
        if int(tracked_out["track_id"].notna().sum()) != len(work) - n_unassigned:
            log.error(f"[traj_refine] {seq}: tracked rows out "
                      f"({int(tracked_out['track_id'].notna().sum())}) != tracked in "
                      f"({len(work)}) - stage-3 unassigned ({n_unassigned})")

        merges_2a = sum(1 for m in rep["merges"] if m["phase"] == "2a")
        merges_2b = sum(1 for m in rep["merges"] if m["phase"] == "2b")
        record["phase2a"] = dict(merges=merges_2a, conflicts=len(rep["conflicts"]),
                                 rejected=len(rep["rejected_2a"]),
                                 conflict_log=rep["conflicts"],
                                 rejected_log=rep["rejected_2a"])
        record["phase2b"] = dict(merges=merges_2b)
        record["stage3"] = dict(rep["stage3"])
        record["merge_log"] = rep["merges"]
        record["no_centroid"] = rep["no_centroid"]
        record["out_of_scope"] = rep["out_of_scope"]
        record["outputs"].update(
            tracklets=int(tracked_out["track_id"].nunique()),
            merges=len(rep["merges"]), merges_2a=merges_2a, merges_2b=merges_2b,
            conflicts=len(rep["conflicts"]), rejected_2a=len(rep["rejected_2a"]),
            rows_relabelled=int(n_relabel), rows_unassigned=n_unassigned,
            frame_collisions=coll, clusters_incoherent=n_incoherent)
        record["per_cluster"] = [
            dict(track_id=int(k), source_track_ids=res["tids"],
                 n_rows=int((new_tid == k).sum()), team=res["team"],
                 number=res["number"],
                 confidence=round(res["confidence"], 6),
                 maxconf=round(res["maxconf"], 6))
            for k, res in sorted(resolved.items()) if len(res["tids"]) > 1]

        log.info(f"[traj_refine] {seq}: {record['inputs']['tracklets']} trajectories "
                 f"-> {record['outputs']['tracklets']} "
                 f"({merges_2a} 2a + {merges_2b} 2b merges, "
                 f"{len(rep['conflicts'])} number conflict(s) resolved, "
                 f"{len(rep['rejected_2a'])} 2a pair(s) rejected, "
                 f"{len(rep['no_centroid'])} without a centroid, "
                 f"{rep['out_of_scope']} out of scope; stage 3: "
                 f"{rep['stage3']['held']} held, {rep['stage3']['placed']} placed, "
                 f"{rep['stage3']['unassigned']} unassigned)")
        self._write(record)
        return out

    def _write(self, record):
        if self.audit_dir is None:
            return
        try:
            (self.audit_dir / f"{record['sequence']}.json").write_text(
                json.dumps(record, indent=2, default=str))
        except Exception as exc:                   # never break a run over telemetry
            log.warning(f"[traj_refine] could not write the audit sidecar: {exc}")
