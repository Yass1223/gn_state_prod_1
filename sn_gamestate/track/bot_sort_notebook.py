"""Notebook-faithful BoT-SORT (SOF) for the SoccerNet GSR pipeline.

Why this module exists
----------------------
The validated CMC-ablation notebook (yolov11-bot-sort-cmc.ipynb, §11d) runs **boxmot's**
BoT-SORT. tracklab 1.3.24 bundles a *different* BoT-SORT fork (top-level ``bot_sort``
package) whose first association is the JDE/FairMOT recipe::

    dists = fuse_motion(kf, embedding_distance(...), ..., lambda_=0.985)

i.e. ``0.985 * cosine + 0.015 * Mahalanobis`` with chi-square gating — the original
BoT-SORT ``min(gated IoU, gated appearance)`` block is commented out in that file, and
``proximity_thresh`` / ``appearance_thresh`` are consequently inert in the main
association. That is NOT what produced the notebook's HOTA 0.6687/0.6947.

This subclass restores the notebook's (= boxmot's = the original BoT-SORT paper's)
association while reusing the fork's building blocks (STrack, KalmanFilter, GMC,
matching, ReIDDetectMultiBackend), all of which are formula-identical to boxmot:

1. **Detection split**: first pass ``conf > track_high_thresh``; BYTE second pass
   ``track_low_thresh < conf < track_high_thresh`` — ``track_low_thresh`` is exposed
   (the fork hard-codes 0.1; the notebook configured 0.05).
2. **First association**: ``ious = 1 - IoU``; mask ``ious > proximity_thresh``;
   ``emb = cosine``; ``emb[emb > appearance_thresh] = 1``; ``emb[mask] = 1``;
   ``dists = min(ious, emb)``; LAP at ``match_thresh``. No motion fusion, no
   ``lambda_``, no score fusion (boxmot ``fuse_first_associate=False``).
3. **Second association**: IoU only at 0.5 (identical in both implementations).
4. **Unconfirmed tracks**: score-fused IoU vs ``cosine / 2`` gated, LAP at 0.7
   (identical in both implementations).
5. **New tracks / removal**: ``score >= new_track_thresh`` to spawn;
   removal after ``max_time_lost = int(frame_rate / 30 * track_buffer)`` frames
   (= 50 for the notebook's frame_rate=25, track_buffer=60).

Box convention fix
------------------
The fork feeds ``ultralytics.xyxy2xywh`` output (**center**-xywh) into
``STrack(tlwh=...)`` (expects **top-left**-xywh). The (w/2, h/2) shift is applied
consistently everywhere, and the output stage un-does it, so boxes come out right —
but the GMC affine is then applied around shifted points and every KF state is
translated. Here detections enter as true top-left tlwh, so CMC/KF operate on true
centers exactly as in boxmot, and ReID crops are cut from the true ltrb box with the
same ``max(0, int(.)) / min(W, int(.))`` clamping as the notebook's injector.

I/O contract
------------
``update()`` keeps the tracklab wrapper contract: input Nx7
``[l, t, r, b, conf, class, tracklab_id]`` (torch tensor), output rows
``[l, t, r, b, track_id, class, conf, tracklab_id]``.

TensorRT
--------
``NotebookBotSORT`` (the tracklab wrapper below) can transparently swap the ReID
weights for a TensorRT engine (``use_tensorrt`` + ``trt_engine`` in the module
config): tracklab's ``ReIDDetectMultiBackend`` routes on the ``.engine`` suffix and
runs it natively. Falls back to the ``.pt`` path with a warning if the engine file
is absent, so the pipeline never hard-fails.
"""
import logging
from pathlib import Path

import numpy as np
import torch

import bot_sort.bot_sort as _fork
from bot_sort import matching
from bot_sort.basetrack import TrackState
from bot_sort.bot_sort import (
    STrack,
    joint_stracks,
    sub_stracks,
    remove_duplicate_stracks,
)

from tracklab.wrappers.track.bot_sort_api import BotSORT as _TracklabBotSORT
from tracklab.utils.cv2 import cv2_load_image

from sn_gamestate.track.masked_sof import MaskedSOF

log = logging.getLogger(__name__)


class NotebookBoTSORT(_fork.BoTSORT):
    """boxmot-parity BoT-SORT on top of the tracklab fork's components."""

    def __init__(self, *args, track_low_thresh: float = 0.1,
                 masked_sof: bool = False, **kwargs):
        # The fork's __init__ builds ReIDDetectMultiBackend + GMC(cmc_method) and
        # stores every shared hyperparameter; it has no track_low_thresh and no
        # masked_sof, so both are popped here before the parent call. lambda_ is
        # accepted by the parent but unused here.
        super().__init__(*args, **kwargs)
        self.track_low_thresh = float(track_low_thresh)

        # Optional repo-local CMC. The fork's sparseOptFlow ignores `detections`
        # (keypoints land on the players), has a trivially-true point-count guard
        # and no `H is None` guard; MaskedSOF is boxmot 19's behaviour instead.
        # See sn_gamestate/track/masked_sof.py.
        if masked_sof:
            self.gmc = MaskedSOF(downscale=2)
            log.info("[BoT-SORT] CMC: repo-local MaskedSOF (downscale=2)")

    # ---------------------------------------------------------------- features
    def _get_features_ltrb(self, ltrb_boxes, img):
        """ReID features for true (l, t, r, b) boxes.

        Mirrors the notebook injector's crop rule: ints, clamped to [0, W] / [0, H];
        degenerate boxes get a tiny black crop so batch order stays aligned.
        """
        h_img, w_img = img.shape[:2]
        crops = []
        for box in ltrb_boxes:
            x1 = max(0, int(box[0]))
            y1 = max(0, int(box[1]))
            x2 = min(w_img, int(box[2]))
            y2 = min(h_img, int(box[3]))
            if x2 <= x1 or y2 <= y1:
                crops.append(np.zeros((2, 2, 3), dtype=img.dtype))
            else:
                crops.append(img[y1:y2, x1:x2])
        if not crops:
            return np.zeros((0, 512), dtype=np.float32)
        feats = self.model(crops)  # ReIDDetectMultiBackend handles preprocessing
        if isinstance(feats, torch.Tensor):
            feats = feats.cpu().float().numpy()
        return feats

    # ------------------------------------------------------------------ update
    def update(self, output_results, img):
        self.frame_id += 1
        activated_stracks, refind_stracks = [], []
        lost_stracks, removed_stracks = [], []

        if isinstance(output_results, torch.Tensor):
            output_results = output_results.cpu().numpy()
        output_results = np.asarray(output_results, dtype=np.float64)
        if output_results.ndim == 1:
            # A 1-D array is either empty OR a single detection row that numpy/torch
            # already squeezed. Reshaping both to (0, 7) silently dropped the single
            # detection (and with it any track it would have started or continued).
            if output_results.size == 0:
                output_results = output_results.reshape(0, 7)
            elif output_results.size == 7:
                output_results = output_results.reshape(1, 7)
            else:
                raise ValueError(
                    "expected a 1-D detection array of size 0 or 7 "
                    f"[l,t,r,b,conf,class,tracklab_id], got shape "
                    f"{output_results.shape}"
                )

        ltrb = output_results[:, 0:4]
        confs = output_results[:, 4]
        clss = output_results[:, 5]
        tracklab_ids = output_results[:, 6]

        self.height, self.width = img.shape[:2]

        # -- notebook/boxmot detection split (BYTE) ---------------------------
        first_mask = confs > self.track_high_thresh
        second_mask = np.logical_and(
            confs > self.track_low_thresh, confs < self.track_high_thresh
        )

        ltrb_first, ltrb_second = ltrb[first_mask], ltrb[second_mask]
        scores_first, scores_second = confs[first_mask], confs[second_mask]
        clss_first, clss_second = clss[first_mask], clss[second_mask]
        ids_first, ids_second = tracklab_ids[first_mask], tracklab_ids[second_mask]

        def _to_tlwh(b):  # true top-left xywh
            out = b.copy()
            out[:, 2:4] = out[:, 2:4] - out[:, 0:2]
            return out

        tlwh_first = _to_tlwh(ltrb_first) if len(ltrb_first) else ltrb_first
        tlwh_second = _to_tlwh(ltrb_second) if len(ltrb_second) else ltrb_second

        # -- appearance features for the high-confidence set ------------------
        if len(ltrb_first) > 0:
            features_first = self._get_features_ltrb(ltrb_first, img)
            detections = [
                STrack(tlwh, s, c, np.asarray(f, dtype=np.float32), tracklab_id=tid)
                for (tlwh, s, c, f, tid) in zip(
                    tlwh_first, scores_first, clss_first, features_first, ids_first
                )
            ]
        else:
            detections = []

        # -- split confirmed / unconfirmed ------------------------------------
        unconfirmed, tracked_stracks = [], []
        for track in self.tracked_stracks:
            (tracked_stracks if track.is_activated else unconfirmed).append(track)

        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)

        # -- camera motion compensation (SOF) ---------------------------------
        warp = self.gmc.apply(img, ltrb_first)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # -- Step 2: first association (paper/boxmot: min of gated IoU & emb) --
        ious_dists = matching.iou_distance(strack_pool, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh
        if len(strack_pool) and len(detections):
            emb_dists = matching.embedding_distance(strack_pool, detections)
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # -- Step 3: BYTE second association (IoU only, thresh 0.5) -----------
        if len(tlwh_second) > 0:
            detections_second = [
                STrack(tlwh, s, c, tracklab_id=tid)
                for (tlwh, s, c, tid) in zip(
                    tlwh_second, scores_second, clss_second, ids_second
                )
            ]
        else:
            detections_second = []

        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, _ = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # -- unconfirmed tracks (identical in fork & boxmot) -------------------
        detections = [detections[i] for i in u_detection]
        ious_dists = matching.iou_distance(unconfirmed, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh
        ious_dists = matching.fuse_score(ious_dists, detections)
        if len(unconfirmed) and len(detections):
            emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=0.7
        )
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # -- Step 4: init new tracks ------------------------------------------
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # -- Step 5: lifetime management ---------------------------------------
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # -- merge bookkeeping (same as fork/boxmot) ----------------------------
        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        # -- output: true ltrb (STrack now holds true top-left tlwh) -----------
        outputs = []
        for t in self.tracked_stracks:
            if not t.is_activated:
                continue
            tlwh = t.tlwh
            outputs.append(
                [
                    tlwh[0],
                    tlwh[1],
                    tlwh[0] + tlwh[2],
                    tlwh[1] + tlwh[3],
                    t.track_id,
                    t.cls,
                    t.score,
                    t.tracklab_id,
                ]
            )
        return outputs


class NotebookBotSORT(_TracklabBotSORT):
    """tracklab wrapper for :class:`NotebookBoTSORT` (drop-in for tracklab.wrappers.BotSORT).

    Adds: notebook association (via NotebookBoTSORT), configurable
    ``track_low_thresh``, ``>=`` on the ``min_confidence`` pre-filter (the notebook
    kept detections at exactly the floor), and optional TensorRT ReID weights.
    """

    # Same issue as YOLOUltralyticsSNFT: tracklab derives Module.level from the
    # DIRECT base class name, and BotSORT yields "bot" instead of "image", which
    # makes EngineDatapipe.__len__ raise. BotSORT is itself an ImageLevelModule.
    level = "image"

    def reset(self):
        # tracklab rebuilds the tracker per video, so this is the clip boundary:
        # let the outgoing CMC emit its identity-fallback rate before it is dropped.
        prev_gmc = getattr(getattr(self, "model", None), "gmc", None)
        if hasattr(prev_gmc, "report"):
            prev_gmc.report()

        weights = Path(self.cfg.model_weights)
        if bool(getattr(self.cfg, "use_tensorrt", False)):
            engine = Path(str(getattr(self.cfg, "trt_engine", "") or ""))
            if engine.suffix == ".engine" and engine.is_file():
                log.info(f"[BoT-SORT] ReID via TensorRT engine: {engine}")
                weights = engine
            else:
                log.warning(
                    f"[BoT-SORT] use_tensorrt=true but engine not found at "
                    f"'{engine}'. Falling back to PyTorch weights "
                    f"({weights.name}). Build it with scripts/build_trt_engines.py."
                )
        self.model = NotebookBoTSORT(
            weights,
            self.device,
            self.cfg.fp16,
            **self.cfg.hyperparams,
        )

    @torch.no_grad()
    def process(self, batch, detections, metadatas):
        import pandas as pd
        from tracklab.utils.coordinates import ltrb_to_ltwh

        if len(detections) == 0:
            return []
        inputs = batch["input"][0]  # Nx7 [l,t,r,b,conf,class,tracklab_id]
        # ">=" (not ">") so detections at exactly the floor survive, matching the
        # notebook's det files which keep conf >= TRACK_DET_CONF.
        inputs = inputs[inputs[:, 4] >= self.cfg.min_confidence]
        image = cv2_load_image(metadatas["file_path"].values[0])
        results = self.model.update(inputs, image)
        results = np.asarray(results)  # N'x8 [l,t,r,b,track_id,class,conf,idx]
        if results.size:
            track_bbox_ltwh = [ltrb_to_ltwh(x) for x in results[:, :4]]
            track_bbox_conf = list(results[:, 6])
            track_ids = list(results[:, 4])
            idxs = list(results[:, 7].astype(int))
            assert set(idxs).issubset(
                detections.index
            ), "Mismatch of indexes during the tracking. The results should match the detections."
            results = pd.DataFrame(
                {
                    "track_bbox_ltwh": track_bbox_ltwh,
                    "track_bbox_conf": track_bbox_conf,
                    "track_id": track_ids,
                    "idxs": idxs,
                }
            )
            results.set_index("idxs", inplace=True, drop=True)
            return results
        return []
