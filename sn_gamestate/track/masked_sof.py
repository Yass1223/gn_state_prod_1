"""Repo-local masked sparse-optical-flow CMC for BoT-SORT.

Why this module exists
----------------------
``bot_sort/gmc.py`` in tracklab 1.3.24 is a verbatim copy of the original BoT-SORT
repo, and ``applySparseOptFlow`` carries that copy's defects:

* the ``detections`` argument is accepted and then **ignored** — the call is
  ``goodFeaturesToTrack(frame, mask=None, ...)``. Keypoints are therefore sampled
  on the players too, so the "camera" affine is fitted partly to the players' own
  motion. The BoT-SORT paper's GMC explicitly excludes dynamic objects;
* the point-count guard is ``np.size(prevPoints, 0) == np.size(prevPoints, 0)``,
  which is trivially true and guards nothing;
* ``cv2.estimateAffinePartial2D`` returns ``H = None`` on degenerate or
  RANSAC-starved correspondences, and the next line dereferences it
  (``H[0, 2] *= downscale``) — a TypeError mid-clip;
* the "not enough matching points" path only ``print``s, so a clip that silently
  ran with no camera compensation is indistinguishable from a healthy one.

boxmot 19's SOF — the implementation behind the notebook's reference numbers —
masks the detections out of the keypoint search, fits with RANSAC, gates on both
inlier count and inlier ratio, guards ``H is None``, and re-detects keypoints
every frame. This module is that behaviour, repo-local, plus a per-clip CMC
health counter that survives into production.

Contract
--------
``MaskedSOF(downscale=2).apply(raw_frame, detections) -> H`` (2x3 float64), a
drop-in for the fork's ``GMC.apply``: ``raw_frame`` is the full-resolution image
as the tracker loaded it, ``detections`` are **ltrb boxes in original image
coordinates**. Every failure path returns the identity warp — never an exception,
never a partly-applied transform.

Colour note: ``tracklab.utils.cv2.cv2_load_image`` returns **RGB**, so the
grayscale conversion uses ``COLOR_RGB2GRAY``. The fork applies ``COLOR_BGR2GRAY``
to the same RGB array, which swaps the R and B luminance weights — harmless for
flow, but there is no reason to reproduce it.
"""
import logging
import weakref

import cv2
import numpy as np

log = logging.getLogger(__name__)

# goodFeaturesToTrack settings, identical to boxmot 19 / the BoT-SORT reference.
_GFTT = dict(
    maxCorners=1000,
    qualityLevel=0.01,
    minDistance=1,
    blockSize=3,
    useHarrisDetector=False,
    k=0.04,
)


def _emit_report(stats: dict) -> None:
    """Log the CMC health line once. Shared by ``report()`` and the finalizer."""
    if stats["reported"] or stats["attempts"] == 0:
        return
    stats["reported"] = True
    n_id, n_att = stats["identity"], stats["attempts"]
    log.info(
        f"[MaskedSOF] CMC health: {n_id}/{n_att} frames ({n_id / n_att:.1%}) fell "
        f"back to an identity warp"
    )


class MaskedSOF:
    """Sparse optical flow CMC that excludes detection boxes from the keypoints.

    One instance per clip. The first frame has no predecessor and returns identity
    without counting as a failure.
    """

    def __init__(
        self,
        downscale: int = 2,
        ransac_reproj_thresh: float = 3.0,
        min_pairs: int = 4,
        min_inliers: int = 10,
        min_inlier_ratio: float = 0.25,
        warn_every: int = 50,
    ):
        self.downscale = max(1, int(downscale))
        self.ransac_reproj_thresh = float(ransac_reproj_thresh)
        self.min_pairs = int(min_pairs)
        self.min_inliers = int(min_inliers)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.warn_every = max(1, int(warn_every))

        self.prev_frame = None
        self.prev_keypoints = None

        # Counters live in a plain dict so the finalizer can read them without
        # holding a reference to `self` (which would stop it from ever running).
        self.stats = {"frames": 0, "attempts": 0, "identity": 0, "reported": False}
        self._warned = {}
        # Backstop for the final clip: tracklab rebuilds the tracker per video, so
        # report() is normally driven from the wrapper's reset(); this covers the
        # last instance, which nothing replaces.
        weakref.finalize(self, _emit_report, self.stats)

    # ------------------------------------------------------------------ reporting
    def report(self) -> None:
        """Emit the per-clip identity-fallback rate (idempotent)."""
        _emit_report(self.stats)

    def _warn(self, reason: str) -> None:
        """Rate-limited warning: first occurrence, then every ``warn_every``-th."""
        n = self._warned[reason] = self._warned.get(reason, 0) + 1
        if n == 1 or n % self.warn_every == 0:
            log.warning(
                f"[MaskedSOF] frame {self.stats['frames']}: {reason} -> identity "
                f"warp (occurrence {n} of this reason in this clip)"
            )

    # ------------------------------------------------------------------ internals
    def _prepare(self, raw_frame: np.ndarray) -> np.ndarray:
        frame = raw_frame
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if self.downscale > 1:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (max(1, w // self.downscale),
                                       max(1, h // self.downscale)))
        return frame

    def _mask(self, shape, detections) -> np.ndarray:
        """255 everywhere except inside the detection boxes (downscaled, clamped)."""
        h, w = shape
        mask = np.full((h, w), 255, dtype=np.uint8)
        if detections is None or len(detections) == 0:
            return mask
        boxes = np.asarray(detections, dtype=np.float64).reshape(-1, 4)
        boxes = boxes / self.downscale
        for l, t, r, b in boxes:
            # floor/ceil so a box is fully covered rather than fringed by keypoints
            x1, y1 = max(0, int(np.floor(l))), max(0, int(np.floor(t)))
            x2, y2 = min(w, int(np.ceil(r))), min(h, int(np.ceil(b)))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 0
        return mask

    def _estimate(self, frame: np.ndarray) -> np.ndarray:
        """Affine from prev_keypoints to the current frame, or identity."""
        eye = np.eye(2, 3, dtype=np.float64)
        self.stats["attempts"] += 1

        prev_pts = self.prev_keypoints
        n_prev = 0 if prev_pts is None else len(prev_pts)
        if n_prev < self.min_pairs:
            self._warn(f"only {n_prev} keypoint(s) on the previous frame "
                       f"(< {self.min_pairs})")
            self.stats["identity"] += 1
            return eye

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, frame, prev_pts, None
        )
        if curr_pts is None or status is None:
            self._warn("pyramidal LK returned no correspondences")
            self.stats["identity"] += 1
            return eye

        keep = status.reshape(-1).astype(bool)
        p0 = prev_pts.reshape(-1, 2)[keep]
        p1 = curr_pts.reshape(-1, 2)[keep]
        if len(p0) < self.min_pairs:
            self._warn(f"only {len(p0)} tracked keypoint pair(s) "
                       f"(< {self.min_pairs})")
            self.stats["identity"] += 1
            return eye

        H, inliers = cv2.estimateAffinePartial2D(
            p0, p1, method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_reproj_thresh,
        )
        if H is None:
            self._warn("estimateAffinePartial2D returned None")
            self.stats["identity"] += 1
            return eye

        n_in = 0 if inliers is None else int(np.count_nonzero(inliers))
        if n_in < self.min_inliers:
            self._warn(f"{n_in} RANSAC inlier(s) (< {self.min_inliers})")
            self.stats["identity"] += 1
            return eye
        ratio = n_in / len(p0)
        if ratio < self.min_inlier_ratio:
            self._warn(f"RANSAC inlier ratio {ratio:.2f} "
                       f"(< {self.min_inlier_ratio:.2f})")
            self.stats["identity"] += 1
            return eye

        # The affine was fitted in downscaled pixels; only the translation column
        # carries units, so it is the only part that scales back up.
        H = np.asarray(H, dtype=np.float64)
        H[0, 2] *= self.downscale
        H[1, 2] *= self.downscale
        return H

    # ------------------------------------------------------------------ interface
    def apply(self, raw_frame: np.ndarray, detections=None) -> np.ndarray:
        """Camera warp (2x3) taking the previous frame's coordinates to this one."""
        self.stats["frames"] += 1
        frame = self._prepare(raw_frame)
        keypoints = cv2.goodFeaturesToTrack(
            frame, mask=self._mask(frame.shape[:2], detections), **_GFTT
        )

        if self.prev_frame is None:            # first frame of the clip
            H = np.eye(2, 3, dtype=np.float64)
        else:
            H = self._estimate(frame)

        # Re-detect on every frame (boxmot behaviour) rather than carrying the
        # LK-tracked points forward, so masking is re-applied against the boxes of
        # the frame the points will be tracked FROM.
        self.prev_frame = frame
        self.prev_keypoints = keypoints
        return H
