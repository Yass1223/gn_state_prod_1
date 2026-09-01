"""Motion-continuity gate on final trajectories -- the algorithm.

Judges, per final trajectory, whether each detection is in motion continuity with
enough of the trajectory's other detections, in CAMERA-MOTION-COMPENSATED image
coordinates, and marks the ones that are not. The camera motion comes in as the
per-frame 2x3 SOF warps the tracker computed (prev -> curr, boxmot's ``multi_gmc``
convention); no motion is re-estimated here.

Definitions
-----------
* Position of a detection: the bottom-middle of its box (the feet point),
  ``(l + w/2, t + h)``.
* Stabilised coordinates: with ``W_f`` the warp of frame ``f`` (mapping frame
  ``f-1`` coordinates to frame ``f`` coordinates), the cumulative transform to the
  first frame is ``M_first = I`` and ``M_f = M_prev @ inv(W_f)``; a point ``p`` of
  frame ``f`` becomes ``q = M_f p`` in first-frame coordinates. A frame with no
  recorded warp contributes an identity step (counted by the caller).
* Support: detection ``i`` is supported by detection ``j`` of the same trajectory
  iff ``0 < |f_i - f_j| <= window`` and
  ``||q_i - q_j|| <= speed_max_px * |f_i - f_j| + slack_px``.
* Gate: a detection with fewer than ``min_support`` supporters is disabled. A
  trajectory with at most ``min_support`` detections is skipped whole (the
  criterion is unsatisfiable there: a detection can have at most n-1 supporters).

``split_merge`` guarantees one detection per (trajectory, frame), so each
trajectory is a clean time series.
"""
import numpy as np


def chain_warps(frame_ids, warps):
    """Cumulative transforms to the first frame.

    frame_ids: iterable of frame keys in ascending temporal order (every frame of
    the video, not only frames with detections). warps: dict frame_id -> 2x3
    array-like, the SOF warp of that frame (prev -> curr); a missing entry is an
    identity step. Returns (M, n_missing) where M maps frame_id -> 3x3 transform
    into first-frame coordinates.
    """
    M = {}
    acc = np.eye(3)
    n_missing = 0
    for k, fid in enumerate(frame_ids):
        if k > 0:
            w = warps.get(fid)
            if w is None:
                n_missing += 1
            else:
                W = np.eye(3)
                W[:2, :] = np.asarray(w, dtype=np.float64)
                acc = acc @ np.linalg.inv(W)
        M[fid] = acc.copy()
    return M, n_missing


def stabilised_positions(bboxes_ltwh, frame_of_row, M):
    """Bottom-middle points, mapped into first-frame coordinates.

    bboxes_ltwh: (n, 4); frame_of_row: (n,) frame key per row; M: from chain_warps.
    """
    b = np.asarray(bboxes_ltwh, dtype=np.float64)
    p = np.stack([b[:, 0] + b[:, 2] / 2.0, b[:, 1] + b[:, 3]], axis=1)
    q = np.empty_like(p)
    for i, fid in enumerate(frame_of_row):
        m = M[fid]
        v = m @ np.array([p[i, 0], p[i, 1], 1.0])
        q[i] = v[:2] / v[2]
    return q


def support_counts(q, frames, window, speed_max_px, slack_px):
    """Supporters per detection of ONE trajectory.

    q: (n, 2) stabilised positions; frames: (n,) integer frame numbers.
    Returns (n,) int: for each detection, how many others are within `window`
    frames and within ``speed_max_px * dt + slack_px`` distance.
    """
    q = np.asarray(q, dtype=np.float64)
    f = np.asarray(frames, dtype=np.int64)
    n = len(f)
    if n <= 1:
        return np.zeros(n, dtype=np.int64)
    df = np.abs(f[:, None] - f[None, :])
    d = np.linalg.norm(q[:, None, :] - q[None, :, :], axis=2)
    ok = (df > 0) & (df <= window) & (d <= speed_max_px * df + slack_px)
    return ok.sum(axis=1).astype(np.int64)


def gate_video(track_ids, frames, bboxes_ltwh, all_frame_ids, warps,
               min_support, window, speed_max_px, slack_px):
    """Apply the gate to one video.

    track_ids: (n,) final trajectory id per TRACKED detection (ints);
    frames: (n,) frame key per detection; bboxes_ltwh: (n, 4);
    all_frame_ids: every frame of the video in temporal order; warps: frame -> 2x3.

    Returns (disabled, report): disabled is an (n,) bool mask, report a dict with
    warp coverage and per-trajectory counts.
    """
    track_ids = np.asarray(track_ids, dtype=np.int64)
    frames = np.asarray(frames, dtype=np.int64)
    n = len(track_ids)
    disabled = np.zeros(n, dtype=bool)
    M, n_missing = chain_warps(all_frame_ids, warps)
    q = stabilised_positions(bboxes_ltwh, frames, M)
    # frame NUMBER of each detection = its position in the video's frame order, so the
    # time arithmetic is exact whatever the frame-key numbering scheme is.
    pos_of = {fid: k for k, fid in enumerate(all_frame_ids)}
    fnum = np.array([pos_of[int(f)] for f in frames], dtype=np.int64)
    per_traj = []
    for tid in np.unique(track_ids):
        rows = np.where(track_ids == tid)[0]
        if len(rows) <= min_support:
            per_traj.append(dict(track_id=int(tid), n=int(len(rows)),
                                 skipped_short=True, disabled=0, support_min=None))
            continue
        sup = support_counts(q[rows], fnum[rows], window, speed_max_px, slack_px)
        bad = sup < min_support
        disabled[rows[bad]] = True
        per_traj.append(dict(track_id=int(tid), n=int(len(rows)), skipped_short=False,
                             disabled=int(bad.sum()), support_min=int(sup.min())))
    report = dict(frames=len(list(all_frame_ids)), warp_steps_missing=int(n_missing),
                  trajectories=int(len(per_traj)),
                  trajectories_skipped_short=int(sum(1 for t in per_traj if t["skipped_short"])),
                  detections=int(n), disabled=int(disabled.sum()), per_trajectory=per_traj)
    return disabled, report
