"""Tests of the motion_gate stage: algorithm, CMC handling, stage + audit contract.

    .venv/bin/python tests/test_motion_gate.py

1. ``test_algorithm``: camera-motion chaining stabilises a world-static point
   exactly; support counting matches a hand computation at the boundary; the gate
   keeps a smoothly moving player under a strong camera pan (where uncompensated
   image speed alone would disable everything), disables an injected outlier,
   skips trajectories with at most ``min_support`` detections, and enforces the
   ``min_support`` boundary.
2. ``test_stage_and_audit`` (needs tracklab, as test_stages.py): the ``MotionGate``
   stage reads the tracker sidecar's full warps, flags/NaNs exactly the rule's
   rows, keeps the original id in ``track_id_pre_motion_gate``, is a verified
   no-op with ``enabled: false``, fails loudly on a pre-full-warp sidecar; the
   audit's motion_gate check passes on the stage's output and fails on five broken
   variants; ``ids_as_split_merge_left`` reconstructs the split_merge view through
   both gates.
"""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sn_gamestate.track import motion_gate as mg          # noqa: E402

MIN_SUPPORT, WINDOW, SPEED, SLACK = 3, 25, 40.0, 20.0
PAN = 70.0            # px/frame camera pan: uncompensated displacement (~65 px/frame after the
                      # player's own ~5) exceeds SPEED*dt+SLACK at every dt, so compensation is decisive


def pan_warps(frame_ids, dx=PAN):
    """Constant horizontal pan: a world-static point at p in frame f-1 appears at
    p - dx in frame f, so the prev->curr warp is a translation by -dx."""
    return {int(f): np.array([[1.0, 0.0, -dx], [0.0, 1.0, 0.0]]) for f in frame_ids[1:]}


def boxes_at(points):
    """A 20x60 box whose bottom-middle is the given image point."""
    return np.stack([[x - 10.0, y - 60.0, 20.0, 60.0] for x, y in points])


def test_algorithm():
    frames = list(range(100, 160))                      # frame keys need not start at 0
    warps = pan_warps(frames)
    # --- stabilisation is exact for a world-static point under the pan
    M, n_missing = mg.chain_warps(frames, warps)
    assert n_missing == 0
    world = np.array([400.0, 300.0])
    img = [world - np.array([PAN, 0.0]) * k for k in range(len(frames))]
    q = mg.stabilised_positions(boxes_at(img), frames, M)
    assert np.allclose(q, world, atol=1e-6), "a world-static point must be constant after chaining"
    # missing warps are identity steps and are counted
    _, n_missing2 = mg.chain_warps(frames, {k: w for k, w in warps.items() if k % 2 == 0})
    assert n_missing2 == sum(1 for f in frames[1:] if f % 2 != 0)

    # --- support boundary: distance exactly speed*dt+slack is a supporter, +eps is not
    qs = np.array([[0.0, 0.0], [SPEED * 1 + SLACK, 0.0], [SPEED * 2 + SLACK + 1e-6, 0.0]])
    sup = mg.support_counts(qs, np.array([0, 1, 2]), WINDOW, SPEED, SLACK)
    # rows 0<->1: dt=1, d=60 <= 60 ok; 0<->2: dt=2, d=120.000001 > 120 no; 1<->2: dt=1, d=60.000001... 
    d12 = qs[2, 0] - qs[1, 0]
    assert d12 <= SPEED + SLACK, "test construction: 1<->2 must be a supporter pair"
    assert sup.tolist() == [1, 2, 1], sup

    # --- gate: world-smooth player under the pan is kept only thanks to compensation
    rng = np.random.default_rng(0)
    n = len(frames)
    world_traj = np.stack([np.linspace(200, 500, n), np.full(n, 400.0)], axis=1)  # ~5 px/frame
    img_traj = [world_traj[k] - np.array([PAN, 0.0]) * k for k in range(n)]
    tids = np.full(n, 7)
    disabled, rep = mg.gate_video(tids, frames, boxes_at(img_traj), frames, warps,
                                  MIN_SUPPORT, WINDOW, SPEED, SLACK)
    assert not disabled.any(), "a world-smooth trajectory must survive the pan"
    # the same trajectory WITHOUT compensation is wiped out (proves CMC is decisive)
    disabled_nc, _ = mg.gate_video(tids, frames, boxes_at(img_traj), frames, {},
                                   MIN_SUPPORT, WINDOW, SPEED, SLACK)
    assert disabled_nc.all(), "without warps the pan alone exceeds the speed bound"
    # net uncompensated displacement per frame must beat the bound already at dt=1:
    assert (PAN - 5.1) * 1 > SPEED * 1 + SLACK, "test construction: pan minus player speed must exceed the dt=1 bound"

    # --- an injected outlier (wrong box far away in world coords) is disabled, alone.
    #     It must exceed the window's maximum reach SPEED*WINDOW+SLACK (= 1020 px):
    #     nearer offsets are still 'plausible' for the largest in-window frame gaps.
    img_out = list(img_traj)
    img_out[30] = img_out[30] + np.array([0.0, 2000.0])
    disabled_o, rep_o = mg.gate_video(tids, frames, boxes_at(img_out), frames, warps,
                                      MIN_SUPPORT, WINDOW, SPEED, SLACK)
    assert disabled_o[30] and disabled_o.sum() == 1, np.where(disabled_o)[0]
    assert rep_o["disabled"] == 1 and rep_o["per_trajectory"][0]["disabled"] == 1

    # --- short trajectory (n == min_support) is skipped whole
    disabled_s, rep_s = mg.gate_video(np.full(3, 2), frames[:3], boxes_at(img_traj[:3]),
                                      frames, warps, MIN_SUPPORT, WINDOW, SPEED, SLACK)
    assert not disabled_s.any() and rep_s["trajectories_skipped_short"] == 1

    # --- min_support boundary: 4 clustered detections -> each has 3 supporters -> kept;
    #     a 5th far detection has 0 -> disabled
    pts = [world, world + [5, 0], world + [0, 5], world + [5, 5], world + [5000, 0]]
    img5 = [np.array(p) - np.array([PAN, 0.0]) * k for k, p in enumerate(pts)]
    disabled_b, _ = mg.gate_video(np.full(5, 3), frames[:5], boxes_at(img5), frames, warps,
                                  MIN_SUPPORT, WINDOW, SPEED, SLACK)
    assert disabled_b.tolist() == [False, False, False, False, True]
    print("algorithm: chaining, support boundary, CMC-decisive kept/wiped, outlier, "
          "short-skip, min_support boundary - ok")


# --------------------------------------------------------------------- stage + audit

def build_state(tmp, with_outlier=True):
    """60 frames, tracker sidecar with full pan warps, detections shaped like the
    split_merge output: trajectory 1 (world-smooth, one injected outlier), trajectory
    2 (short: 3 rows), 2 untracked rows."""
    frames = 1000 + np.arange(60)
    meta = pd.DataFrame(dict(video_id="SNGS-000", frame=np.arange(60),
                             file_path=[str(tmp / "SNGS-000" / "img1" / f"{k + 1:06d}.jpg")
                                        for k in range(60)]),
                        index=frames)
    meta.index.name = "image_id"
    side = {"video_id": "SNGS-000", "settings": {}, "n_frames": 60, "frames": [
        dict(image_id=str(int(f)), identity=False,
             a00=1.0, a01=0.0, tx=-PAN if k else 0.0, a10=0.0, a11=1.0, ty=0.0)
        for k, f in enumerate(frames)]}
    (tmp / "audit" / "track").mkdir(parents=True, exist_ok=True)
    (tmp / "audit" / "track" / "SNGS-000.json").write_text(json.dumps(side))

    n = 60
    world = np.stack([np.linspace(200, 500, n), np.full(n, 400.0)], axis=1)
    img = [world[k] - np.array([PAN, 0.0]) * k for k in range(n)]
    if with_outlier:
        img[30] = img[30] + np.array([0.0, 2000.0])   # beyond SPEED*WINDOW+SLACK from every neighbour
    rows = []
    for k in range(n):
        rows.append(dict(image_id=int(frames[k]), track_id=1.0,
                         bbox_ltwh=np.array([img[k][0] - 10, img[k][1] - 60, 20.0, 60.0])))
    for k in range(3):
        rows.append(dict(image_id=int(frames[k]), track_id=2.0,
                         bbox_ltwh=np.array([100.0, 100.0, 20.0, 60.0])))
    rows.append(dict(image_id=int(frames[0]), track_id=np.nan,
                     bbox_ltwh=np.array([0.0, 0.0, 5.0, 5.0])))
    rows.append(dict(image_id=int(frames[1]), track_id=np.nan,
                     bbox_ltwh=np.array([0.0, 0.0, 5.0, 5.0])))
    det = pd.DataFrame(rows)
    det["crop_single"] = True
    det.index = np.arange(5000, 5000 + len(det))
    return det, meta


def make_stage(tmp, enabled):
    from sn_gamestate.track.motion_gate_api import MotionGate
    cfg = SimpleNamespace(enabled=enabled, min_support=MIN_SUPPORT, window=WINDOW,
                          speed_max_px=SPEED, slack_px=SLACK,
                          track_sidecar_dir=str(tmp / "audit" / "track"),
                          audit_dir=str(tmp / "audit" / "motion_gate"))
    return MotionGate(cfg, "cpu")


def test_stage_and_audit():
    from sn_gamestate.audit.run_audit_api import RunAudit, ids_as_split_merge_left
    tmp = Path(tempfile.mkdtemp())
    det, meta = build_state(tmp)
    stage = make_stage(tmp, enabled=True)
    out = stage.process(det, meta)
    outlier_idx = det.index[30]
    assert bool(out.loc[outlier_idx, "motion_gate_disabled"]) is True
    assert np.isnan(out.loc[outlier_idx, "track_id"])
    assert out.loc[outlier_idx, "track_id_pre_motion_gate"] == 1.0
    assert int(out["motion_gate_disabled"].sum()) == 1, "only the outlier is disabled"
    assert out.loc[det["track_id"] == 2.0, "track_id"].notna().all(), "short trajectory untouched"
    assert out.loc[det["track_id"].isna(), "track_id"].isna().all()
    side = json.loads((tmp / "audit" / "motion_gate" / "SNGS-000.json").read_text())
    assert side["ran"] and side["outputs"]["disabled"] == 1
    assert side["outputs"]["trajectories_skipped_short"] == 1
    assert side["inputs"]["tracked"] == 63 and side["inputs"]["warps"] == 60
    print("stage: outlier disabled, short trajectory skipped, untracked untouched, sidecar ok")

    # split_merge view reconstruction through motion_gate + an emulated pitch_gate
    withpg = out.copy()
    withpg["track_id_pregate"] = withpg["track_id"].astype(float)
    withpg.loc[withpg["track_id"] == 2.0, "track_id"] = np.nan     # pitch_gate clears traj 2
    view = ids_as_split_merge_left(withpg)
    assert len(view) == 63 and view.loc[outlier_idx, "track_id"] == 1.0
    assert set(view["track_id"].unique()) == {1.0, 2.0}
    print("ids_as_split_merge_left: both gates undone, 63 rows, ids restored")

    cfg = SimpleNamespace(out_dir=str(tmp / "audit"), jn_cache_dir=str(tmp / "jn"),
                          calib_dir=str(tmp / "calib"), models_dir=str(tmp / "models"),
                          thresholds={},
                          track_sidecar_dir=str(tmp / "audit" / "track"),
                          motion_gate_sidecar_dir=str(tmp / "audit" / "motion_gate"),
                          expected_motion_gate=dict(enabled=True, min_support=MIN_SUPPORT,
                                                    window=WINDOW, speed_max_px=SPEED,
                                                    slack_px=SLACK))
    tracked_in = ids_as_split_merge_left(out)
    tracked_out = out.dropna(subset=["track_id"])
    c = RunAudit(cfg)._check_motion_gate("SNGS-000", out, tracked_in, tracked_out, meta)
    assert c.verdict == "PASS", (c.verdict, c.note)
    assert c.observed["rule_recompute_mismatches"] == 0 and c.observed["flagged_rows"] == 1
    print(f"audit motion_gate: PASS (disabled share {c.observed['disabled_share']})")

    def verdict(cfg_, det_):
        ti = ids_as_split_merge_left(det_)
        return RunAudit(cfg_)._check_motion_gate("SNGS-000", det_, ti, det_.dropna(subset=["track_id"]), meta)

    # 1) parameter mismatch
    bad_cfg = SimpleNamespace(**{**cfg.__dict__,
                                 "expected_motion_gate": dict(cfg.expected_motion_gate, min_support=4)})
    assert verdict(bad_cfg, out).verdict == "FAIL"
    # 2) a flag flipped off (rule recompute + consistency)
    bad = out.copy(); bad.loc[outlier_idx, "motion_gate_disabled"] = False
    assert verdict(cfg, bad).verdict == "FAIL"
    # 3) a row flagged that the rule keeps
    bad = out.copy()
    extra = det.index[5]
    bad.loc[extra, "motion_gate_disabled"] = True
    bad.loc[extra, "track_id_pre_motion_gate"] = 1.0
    bad.loc[extra, "track_id"] = np.nan
    assert verdict(cfg, bad).verdict == "FAIL"
    # 4) missing sidecar
    bad_cfg = SimpleNamespace(**{**cfg.__dict__, "motion_gate_sidecar_dir": str(tmp / "nowhere")})
    assert verdict(bad_cfg, out).verdict == "FAIL"
    # 5) config says off but rows are flagged
    bad_cfg = SimpleNamespace(**{**cfg.__dict__,
                                 "expected_motion_gate": dict(cfg.expected_motion_gate, enabled=False)})
    assert verdict(bad_cfg, out).verdict == "FAIL"
    print("audit motion_gate: five negative controls FAIL")

    # --- enabled: false is a verified no-op
    tmp2 = Path(tempfile.mkdtemp())
    det2, meta2 = build_state(tmp2)
    out2 = make_stage(tmp2, enabled=False).process(det2, meta2)
    assert not out2["motion_gate_disabled"].any()
    assert out2["track_id"].equals(det2["track_id"])
    cfg_off = SimpleNamespace(**{**cfg.__dict__,
                                 "track_sidecar_dir": str(tmp2 / "audit" / "track"),
                                 "motion_gate_sidecar_dir": str(tmp2 / "audit" / "motion_gate"),
                                 "expected_motion_gate": dict(cfg.expected_motion_gate, enabled=False)})
    c2 = RunAudit(cfg_off)._check_motion_gate("SNGS-000", out2, ids_as_split_merge_left(out2),
                                              out2.dropna(subset=["track_id"]), meta2)
    assert c2.verdict == "PASS", (c2.verdict, c2.note)
    print("gate off: verified no-op, audit PASS")

    # --- a pre-full-warp tracker sidecar makes the enabled stage fail loudly
    tmp3 = Path(tempfile.mkdtemp())
    det3, meta3 = build_state(tmp3)
    p = tmp3 / "audit" / "track" / "SNGS-000.json"
    old = json.loads(p.read_text())
    for fr in old["frames"]:
        fr.pop("a01"); fr.pop("a10"); fr.pop("a11")
    p.write_text(json.dumps(old))
    try:
        make_stage(tmp3, enabled=True).process(det3, meta3)
        raise AssertionError("expected RuntimeError on a pre-full-warp sidecar")
    except RuntimeError as exc:
        assert "full 2x3 warp" in str(exc)
    print("pre-full-warp sidecar: stage fails loudly")


if __name__ == "__main__":
    test_algorithm()
    test_stage_and_audit()
    print("ALL MOTION_GATE TESTS PASSED")
