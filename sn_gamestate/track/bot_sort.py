"""BoT-SORT with sparse-optical-flow camera motion, as the reference notebook runs it.

What this is
------------
boxmot's `BotSort` driven from the tracklab pipeline, with appearance supplied from
outside. It replaces a subclass of tracklab's bundled `bot_sort` fork, which is a
different tracker: its first association is the JDE/FairMOT recipe
`fuse_motion(kf, cosine, lambda_=0.985)` with the paper's `min(gated IoU, gated
appearance)` block commented out, so `proximity_thresh` and `appearance_thresh` are
inert there. Rather than re-derive boxmot's behaviour on top of the fork, this module
calls boxmot.

Three inversions of control, all deliberate and all matching the reference:

* **Embeddings are injected.** `reid_model` is a guard object that raises if the
  tracker ever tries to compute its own, and features are passed to `update()`. Only
  detections above `track_high_thresh` are embedded; the rest carry a zero row, which
  is what the first association gate sees for them.
* **Camera motion is computed outside.** One `SOF` instance per video, called once per
  frame, its warp handed to the tracker through a feed object installed as `cmc`. The
  constructor is checked to have built a real `SOF` before the feed replaces it, so a
  wrong `cmc_method` fails loudly instead of silently disabling compensation.
* **No confidence pre-filter.** boxmot's own split is the only floor. The detector's
  `conf` already floors at 0.1 and `track_low_thresh` is below that, so an extra
  `min_confidence` gate would be a second, redundant threshold.

Two details that are easy to get wrong
--------------------------------------
`tracklab.utils.cv2.cv2_load_image` returns **RGB**. boxmot's `BaseCMC.preprocess`
does `COLOR_BGR2GRAY` and the OSNet-AIN preprocessing does `COLOR_BGR2RGB`, so both
expect BGR. The frame is converted once, here, and the BGR array is what everything
downstream sees.

boxmot returns `[x1, y1, x2, y2, id, conf, cls, det_ind]`. Confidence is column 5 and
the class is column 6 — the reverse of the tracklab fork's output, where reading
column 6 as confidence would silently write class ids into `track_bbox_conf`.

Camera motion, for the record
-----------------------------
boxmot 19's `SOF.apply(img, dets)` accepts `dets` and never reads it: all three
`goodFeaturesToTrack` calls pass `mask=None`, and `BaseCMC.generate_mask` is never
called. The reference notebook additionally passes `None`. So the camera motion here is
estimated on unmasked keypoints, players included, and `scale=0.15` means the estimate
runs on a frame downsampled to 15% with the translation divided back up.

Inherited contract
------------------
`preprocess`, the collate function and the dataloader all come from tracklab's
`BotSORT` wrapper, which is why that class is still the base. The one thing assumed
about it and not verified here is that its `__init__` only stores `cfg`/`device` and
calls `reset()`; if it also reads `cfg.model_weights`, this will fail at construction
naming that key.
"""
import atexit
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from boxmot.motion.cmc import get_cmc_method
from boxmot.trackers.botsort.botsort import BotSort

from tracklab.wrappers.track.bot_sort_api import BotSORT as _TracklabBotSORT
from tracklab.utils.cv2 import cv2_load_image

from sn_gamestate.reid import osnet_ain

log = logging.getLogger(__name__)

IDENTITY = np.eye(2, 3, dtype=np.float32)


class _WarpFeed:
    """Stands in for the tracker's own CMC and returns the warp computed this frame."""

    def __init__(self):
        self.H = IDENTITY.copy()

    def apply(self, img, dets=None):
        return self.H


class _NoFeatures:
    """Fails loudly if the tracker tries to compute appearance itself."""

    def get_features(self, xyxys, img):
        raise RuntimeError(
            "the tracker asked for appearance features, but embeddings are supplied "
            "per frame; something bypassed the injection path"
        )

    def warmup(self, *args, **kwargs):
        pass

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self


class _Diagnostics:
    """Per-video record of what only the tracker can see.

    Most of what can go quietly wrong here leaves no trace in the saved state: how often
    camera-motion estimation fell back to identity, how many crops clipped to nothing,
    whether every frame actually reached the tracker. Without a record, an audit can only
    report those as unresolved, so they are written through to a sidecar on every frame
    and the in-pipeline `audit` stage reads them back. Write-through, not end-of-video:
    each video passes through every stage - the audit included - before the tracker
    sees the next video, so anything buffered until the next video (or process exit)
    would reach disk only after the audit that needs it (observed on Kaggle: 5/5
    sequences failed with "no tracker sidecar" while tracking itself was healthy).

    Writing is best-effort: a failure here degrades the audit, and must never take down
    a tracking run.

    Every frame's FULL 2x3 SOF warp is recorded (tx, ty, a00, a01, a10, a11): the
    motion_gate stage compensates camera motion with exactly the warps that ran here,
    so the two stages cannot disagree about the camera's motion.
    """

    def __init__(self, out_dir, settings: dict):
        self.dir = Path(out_dir) if out_dir else None
        self.settings = settings
        self.video = None
        self.frames = []
        if self.dir is not None:
            # The last video has no following reset() to flush it.
            atexit.register(self.flush)

    def begin(self, video_id):
        if self.dir is None or video_id == self.video:
            return
        self.flush()
        self.video = video_id

    def record(self, **row):
        if self.dir is None:
            return
        self.frames.append(row)
        # Write-through (see the class docstring): the sidecar must already be on
        # disk when this video's audit runs, which is before the next begin().
        self._write()

    def _write(self):
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / f"{self.video}.json").write_text(json.dumps({
                "video_id": str(self.video),
                "settings": self.settings,
                "n_frames": len(self.frames),
                "frames": self.frames,
            }))
        except Exception as exc:                       # never break a run over telemetry
            log.warning(f"[BoT-SORT] could not write the audit sidecar: {exc}")

    def flush(self):
        if self.dir is None or self.video is None or not self.frames:
            return
        self._write()
        self.frames = []
        self.video = None


class BotSortSOF(_TracklabBotSORT):
    """tracklab stage wrapping boxmot's BoT-SORT with OSNet-AIN appearance."""

    # tracklab derives Module.level from the direct base class name, and "BotSORT"
    # yields "bot" instead of "image", which makes EngineDatapipe.__len__ raise.
    level = "image"

    def reset(self):
        """Called per video: a fresh tracker and a fresh motion estimator each clip."""
        hyperparams = {k: v for k, v in self.cfg.hyperparams.items()}

        # The appearance model is clip-independent, so it is built once and reused;
        # rebuilding it per video would reload the checkpoint for every clip.
        if getattr(self, "embedder", None) is None:
            self.embedder = osnet_ain.from_config(
                self.cfg, self.device,
                batch_size=int(getattr(self.cfg, "embed_batch_size", 64)),
            )

        tracker = BotSort(reid_model=_NoFeatures(), with_reid=True, **hyperparams)

        built = type(getattr(tracker, "cmc", None)).__name__.lower()
        if built != "sof":
            raise RuntimeError(
                f"cmc_method={hyperparams.get('cmc_method')!r} built {built!r}, "
                f"expected 'sof'; camera motion would not be what was configured"
            )
        if abs(float(tracker.appearance_thresh)
               - float(hyperparams["appearance_thresh"])) > 1e-9:
            raise RuntimeError("appearance_thresh did not reach the tracker")

        self.warp_feed = _WarpFeed()
        tracker.cmc = self.warp_feed          # only after a real SOF was confirmed built
        self.cmc = get_cmc_method("sof")(scale=float(self.cfg.sof_scale))
        self.model = tracker
        self.track_high_thresh = float(hyperparams["track_high_thresh"])
        self._bad_det_ind = 0

        # Built once, like the embedder: it owns an atexit hook and spans every video.
        if getattr(self, "diagnostics", None) is None:
            self.diagnostics = _Diagnostics(
                getattr(self.cfg, "audit_dir", None),
                {"hyperparams": hyperparams, "sof_scale": float(self.cfg.sof_scale),
                 "embedder": dict(getattr(self.embedder, "info", {}))},
            )

    @torch.no_grad()
    def process(self, batch, detections, metadatas):
        import pandas as pd
        from tracklab.utils.coordinates import ltrb_to_ltwh

        # cv2_load_image returns RGB; boxmot's CMC and the AIN preprocessing both
        # expect BGR. Convert once and use the BGR frame everywhere below.
        image = cv2.cvtColor(cv2_load_image(metadatas["file_path"].values[0]),
                             cv2.COLOR_RGB2BGR)
        self.diagnostics.begin(
            metadatas["video_id"].values[0] if "video_id" in metadatas.columns
            else "unknown")

        inputs = batch["input"][0] if len(detections) else np.zeros((0, 7))
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.cpu().numpy()
        # float64 throughout: the last column carries DataFrame index labels, which
        # float32 cannot represent exactly beyond 2**24.
        inputs = np.asarray(inputs, dtype=np.float64)
        if inputs.ndim == 1:
            if inputs.size == 0:
                inputs = inputs.reshape(0, 7)
            elif inputs.size == 7:
                inputs = inputs.reshape(1, 7)
            else:
                raise ValueError(
                    "expected a 1-D detection array of size 0 or 7 "
                    f"[l,t,r,b,conf,class,tracklab_id], got shape {inputs.shape}"
                )

        dets = np.ascontiguousarray(inputs[:, [0, 1, 2, 3, 4, 5]], dtype=np.float32)
        tracklab_ids = inputs[:, 6]

        embeddings = np.zeros((len(dets), self.embedder.dim), dtype=np.float32)
        high = np.flatnonzero(dets[:, 4] > self.track_high_thresh)
        n_clipped = n_tiny = 0
        if len(high):
            crops = []
            for box in dets[high, :4]:
                patch = osnet_ain.crop_ltrb(image, box)
                if patch is None:
                    n_clipped += 1
                    patch = osnet_ain.EMPTY_CROP
                elif patch.shape[0] < 16 or patch.shape[1] < 8:
                    n_tiny += 1
                crops.append(patch)
            embeddings[high] = self.embedder.embed(crops)

        # Every frame gets a warp and an update, including frames with no detections.
        # Returning early on an empty frame would leave the optical-flow estimator's
        # previous frame two steps back, so the next warp would describe twice the
        # motion, and Kalman prediction and track ageing would both skip a step.
        warp = self.cmc.apply(image, None)
        self.warp_feed.H = warp

        # Copies because BoT-SORT normalises the feature array it is handed in place.
        results = np.asarray(self.model.update(dets.copy(), image, embeddings.copy()))
        results = results.reshape(-1, 8) if results.size else results.reshape(0, 8)

        det_ind = results[:, 7].astype(int)
        valid = (det_ind >= 0) & (det_ind < len(tracklab_ids))
        n_dropped = int((~valid).sum())
        if n_dropped:
            self._bad_det_ind += n_dropped
            if self._bad_det_ind % 100 < n_dropped:
                log.warning(
                    f"[BoT-SORT] {self._bad_det_ind} output row(s) so far carried a "
                    f"detection index outside this frame's detections; dropped"
                )
            results = results[valid]
            det_ind = det_ind[valid]

        keypoints = getattr(self.cmc, "prev_keypoints", None)
        self.diagnostics.record(
            image_id=str(metadatas.index[0]),
            n_det=int(len(dets)), n_high=int(len(high)),
            identity=bool(np.array_equal(warp, IDENTITY)),
            tx=float(warp[0, 2]), ty=float(warp[1, 2]), a00=float(warp[0, 0]),
            a01=float(warp[0, 1]), a10=float(warp[1, 0]), a11=float(warp[1, 1]),
            corners=int(len(keypoints)) if keypoints is not None else 0,
            clipped=n_clipped, tiny=n_tiny,
            zero_emb=int((~embeddings[high].any(axis=1)).sum()) if len(high) else 0,
            n_out=int(len(results)), dropped=n_dropped,
        )

        if len(results) == 0:
            return []

        idxs = tracklab_ids[det_ind].astype(int).tolist()
        assert set(idxs).issubset(detections.index), (
            "Mismatch of indexes during the tracking. "
            "The results should match the detections."
        )
        out = pd.DataFrame(
            {
                "track_bbox_ltwh": [ltrb_to_ltwh(row) for row in results[:, :4]],
                "track_bbox_conf": list(results[:, 5]),
                "track_id": list(results[:, 4]),
                "idxs": idxs,
            }
        )
        out.set_index("idxs", inplace=True, drop=True)
        return out
