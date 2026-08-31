"""pitch_gate stage: rule, stage contract, switch, sidecar, and the audit check.

Synthetic detections only (no images, no model): 8 tracklets with projected
positions, one on the bench (y ~ 40 m), one behind a goal (x ~ 60 m), one at a
throw-in (y ~ 35 m, inside the 5 m margin), one straddling the touchline with a
mean inside, one without any projection, plus untracked rows.
"""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sn_gamestate.pitch_gate.pitch_gate_api import (          # noqa: E402
    PitchGate, gate_tracklets, is_off_pitch, tracklet_mean_position)
from sn_gamestate.team.rules import PITCH_HALF_LEN, PITCH_HALF_WID  # noqa: E402
from sn_gamestate.audit.run_audit_api import RunAudit          # noqa: E402

MARGIN = 5.0


def make_data(n_frames=40):
    rng = np.random.default_rng(0)
    meta = pd.DataFrame([dict(image_id=1000 + f, video_id=0, frame=f,
                              file_path=f"/data/SNGS-000/img1/{f + 1:06d}.jpg") for f in range(n_frames)]
                        ).set_index("image_id")
    # track_id -> (mean x, mean y, spread); None = no projection
    spec = {0: (-30.0, 0.0, 2.0),      # midfield
            1: (45.0, 0.0, 2.0),       # keeper, inside
            2: (0.0, 40.0, 1.0),       # bench: off (|y| 40 > 34 + 5)
            3: (60.0, 5.0, 1.0),       # behind the goal: off (|x| 60 > 52.5 + 5)
            4: (0.0, 35.0, 0.5),       # throw-in: inside the margin, kept
            5: (10.0, 33.0, 3.0),      # straddles the touchline frame by frame, mean inside
            6: None,                   # no projection at all: kept
            7: (-58.0, 0.0, 0.0)}      # exactly 0.5 m past the margin: off
    rows = []
    for f in range(n_frames):
        for tid, s in spec.items():
            if s is None:
                bp = None
            else:
                mx, my, sd = s
                bp = {"x_bottom_middle": mx + (rng.normal(0, sd) if sd else 0.0),
                      "y_bottom_middle": my + (rng.normal(0, sd) if sd else 0.0)}
            rows.append(dict(image_id=1000 + f, bbox_ltwh=np.array([10.0 * tid, 50.0, 20.0, 50.0]),
                             bbox_conf=0.9, track_id=float(tid), crop_single=True, bbox_pitch=bp))
        rows.append(dict(image_id=1000 + f, bbox_ltwh=np.array([200.0, 50.0, 20.0, 50.0]), bbox_conf=0.2,
                         track_id=np.nan, crop_single=True,
                         bbox_pitch={"x_bottom_middle": 70.0, "y_bottom_middle": 0.0}))  # untracked, ignored
    det = pd.DataFrame(rows)
    det.index = np.arange(5000, 5000 + len(det))
    # tracklet 5: force the per-frame flicker (half the frames outside) while the mean stays inside
    sel = det.index[(det.track_id == 5.0)]
    for i, idx in enumerate(sel):
        det.at[idx, "bbox_pitch"] = {"x_bottom_middle": 10.0, "y_bottom_middle": 36.0 if i % 2 else 30.0}
    return det, meta


def test_rule():
    assert not is_off_pitch(0.0, 0.0, MARGIN)
    assert not is_off_pitch(PITCH_HALF_LEN + MARGIN, 0.0, MARGIN)          # boundary is inside (strict >)
    assert is_off_pitch(PITCH_HALF_LEN + MARGIN + 1e-6, 0.0, MARGIN)
    assert not is_off_pitch(0.0, PITCH_HALF_WID + MARGIN, MARGIN)
    assert is_off_pitch(0.0, -(PITCH_HALF_WID + MARGIN) - 1e-6, MARGIN)
    assert not is_off_pitch(np.nan, 0.0, MARGIN) and not is_off_pitch(0.0, np.nan, MARGIN)
    assert is_off_pitch(-53.0, 0.0, 0.0) and not is_off_pitch(-53.0, 0.0, 1.0)
    mx, my, k = tracklet_mean_position([{"x_bottom_middle": 1.0, "y_bottom_middle": 2.0}, None,
                                        {"x_bottom_middle": 3.0, "y_bottom_middle": 4.0},
                                        {"x_bottom_middle": float("nan"), "y_bottom_middle": 0.0}, "junk"])
    assert (mx, my, k) == (2.0, 3.0, 2)
    assert tracklet_mean_position([None, {}])[2] == 0
    print("pitch_gate: rule ok (strict boundary, NaN -> keep, mean over finite rows only)")


def _run_stage():
    """Runs the stage enabled and disabled with the asserts of the stage contract;
    returns (tmp, det, meta, out_enabled, out_disabled) for the audit test."""
    tmp = Path(tempfile.mkdtemp())
    det, meta = make_data()
    n_tracked_in = int(det.track_id.notna().sum())

    # --- enabled
    pg = PitchGate(SimpleNamespace(enabled=True, margin_m=MARGIN, audit_dir=str(tmp / "audit/pitch_gate")))
    out = pg.process(det, meta)
    assert len(out) == len(det) and out.index.equals(det.index), "no row may be added or removed"
    for c in ("track_id_pregate", "pitch_gate_offpitch", "pitch_mean_x", "pitch_mean_y"):
        assert c in out.columns
    assert out["track_id_pregate"].equals(det["track_id"].astype(float))
    off_ids = set(out.loc[out.pitch_gate_offpitch, "track_id_pregate"])
    assert off_ids == {2.0, 3.0, 7.0}, off_ids
    assert out.loc[out.pitch_gate_offpitch, "track_id"].isna().all()
    kept = out[~out.pitch_gate_offpitch & out.track_id_pregate.notna()]
    assert (kept.track_id == kept.track_id_pregate).all()
    assert out.loc[det.track_id.isna(), "track_id"].isna().all() and not out.loc[det.track_id.isna(), "pitch_gate_offpitch"].any()
    assert set(out.track_id.dropna()) == {0.0, 1.0, 4.0, 5.0, 6.0}
    assert out.loc[out.track_id_pregate == 6.0, "pitch_mean_x"].isna().all(), "no projection -> NaN mean, kept"
    assert out.loc[out.track_id_pregate == 5.0, "pitch_mean_y"].between(30, 34).all(), "mean, not per frame"
    assert abs(out.loc[out.track_id_pregate == 7.0, "pitch_mean_x"].iloc[0] + 58.0) < 1e-9
    assert out["pitch_gate_offpitch"].dtype == bool
    rec = json.loads((tmp / "audit/pitch_gate/SNGS-000.json").read_text())
    assert rec["enabled"] is True and rec["margin_m"] == MARGIN
    assert rec["tracklets"] == 8 and rec["tracklets_off_pitch"] == 3 and rec["tracklets_gated"] == 3
    assert rec["tracklets_without_position"] == 1
    assert rec["rows_tracked_before"] == n_tracked_in and rec["rows_gated"] == 3 * 40
    assert rec["rows_tracked_after"] == n_tracked_in - 3 * 40
    assert {p["track_id"] for p in rec["per_tracklet"] if p["off_pitch"]} == {2.0, 3.0, 7.0}
    print("pitch_gate enabled: gated {2,3,7}, kept throw-in / touchline-straddler / no-projection, sidecar ok")

    # --- disabled: same columns, track_id untouched
    pg0 = PitchGate(SimpleNamespace(enabled=False, margin_m=MARGIN, audit_dir=str(tmp / "audit/pitch_gate_off")))
    out0 = pg0.process(det, meta)
    assert out0["track_id"].equals(det["track_id"].astype(float))
    assert set(out0.loc[out0.pitch_gate_offpitch, "track_id_pregate"]) == {2.0, 3.0, 7.0}, "rule still recorded"
    rec0 = json.loads((tmp / "audit/pitch_gate_off/SNGS-000.json").read_text())
    assert rec0["enabled"] is False and rec0["tracklets_off_pitch"] == 3 and rec0["tracklets_gated"] == 0
    assert rec0["rows_gated"] == 0 and rec0["rows_tracked_after"] == n_tracked_in
    print("pitch_gate disabled: track_id untouched, rule recorded, sidecar ok")

    # --- margin 0 gates the throw-in and the touchline straddler too; margin 30 gates nothing
    _, _, off_m0, _ = gate_tracklets(det, 0.0)
    assert set(det.loc[off_m0, "track_id"]) == {2.0, 3.0, 4.0, 7.0}
    _, _, off_m30, _ = gate_tracklets(det, 30.0)
    assert not off_m30.any()
    # invalid margin
    try:
        PitchGate(SimpleNamespace(enabled=True, margin_m=-1.0, audit_dir=None))
        raise AssertionError("negative margin accepted")
    except ValueError:
        pass
    print("pitch_gate: margin sweep and validation ok")
    return tmp, det, meta, out, out0


def _audit(tmp, sidecar_dir, enabled, margin, thresholds=None):
    cfg = SimpleNamespace(out_dir=str(tmp / "audit"), jn_cache_dir=str(tmp / "jn"), calib_dir=str(tmp / "calib"),
                          models_dir=str(tmp / "models"), thresholds=thresholds or {},
                          pitch_gate_sidecar_dir=str(sidecar_dir),
                          expected_pitch_gate=dict(enabled=enabled, margin_m=margin))
    return RunAudit(cfg)


def _audit_checks(tmp, det, meta, out, out0):
    sc, sc0 = tmp / "audit/pitch_gate", tmp / "audit/pitch_gate_off"
    # 1 of 8 tracklets has no projection (12.5 %): WARN at the default 5 % threshold,
    # PASS once the threshold admits it. Both are the intended behaviour.
    c = _audit(tmp, sc, True, MARGIN)._check_pitch_gate("SNGS-000", out)
    assert c.verdict == "WARN" and "no projection" in c.note, (c.verdict, c.note)
    lenient = dict(pitch_gate_no_position_warn=0.2)
    c = _audit(tmp, sc, True, MARGIN, lenient)._check_pitch_gate("SNGS-000", out)
    assert c.verdict == "PASS", c.note
    assert c.observed["tracklets_gated"] == 3 and c.observed["rows_gated"] == 120
    assert c.observed["tracklets_without_position"] == 1 and c.observed["off_pitch_track_ids"] == [2.0, 3.0, 7.0]
    c0 = _audit(tmp, sc0, False, MARGIN, lenient)._check_pitch_gate("SNGS-000", out0)
    assert c0.verdict == "PASS" and "disabled" in c0.note, (c0.verdict, c0.note)   # INFO note, PASS verdict
    # an implausible gated share (calibration suspect) is a WARN
    strict = dict(pitch_gate_no_position_warn=0.2, pitch_gate_gated_warn=0.3)
    c = _audit(tmp, sc, True, MARGIN, strict)._check_pitch_gate("SNGS-000", out)
    assert c.verdict == "WARN" and "off-pitch at margin" in c.note, (c.verdict, c.note)
    # negative controls
    assert _audit(tmp, sc, True, 4.0)._check_pitch_gate("SNGS-000", out).verdict == "FAIL"        # margin mismatch
    assert _audit(tmp, sc, False, MARGIN)._check_pitch_gate("SNGS-000", out).verdict == "FAIL"    # switch mismatch
    assert _audit(tmp, tmp / "nowhere", True, MARGIN)._check_pitch_gate("SNGS-000", out).verdict == "FAIL"  # no sidecar
    bad = out.copy(); bad.loc[bad.track_id_pregate == 2.0, "track_id"] = 2.0               # gated row kept its id
    assert _audit(tmp, sc, True, MARGIN)._check_pitch_gate("SNGS-000", bad).verdict == "FAIL"
    bad = out.copy(); bad.loc[bad.track_id_pregate == 0.0, "track_id"] = np.nan             # on-pitch row lost its id
    assert _audit(tmp, sc, True, MARGIN)._check_pitch_gate("SNGS-000", bad).verdict == "FAIL"
    bad = out.copy(); bad.loc[bad.track_id_pregate == 4.0, "pitch_gate_offpitch"] = True    # flag disagrees with rule
    assert _audit(tmp, sc, True, MARGIN)._check_pitch_gate("SNGS-000", bad).verdict == "FAIL"
    bad = out.copy(); bad.loc[bad.track_id_pregate == 0.0, "pitch_mean_x"] += 1.0           # stored mean tampered
    assert _audit(tmp, sc, True, MARGIN)._check_pitch_gate("SNGS-000", bad).verdict == "FAIL"
    bad = out0.copy(); bad.loc[bad.track_id_pregate == 2.0, "track_id"] = np.nan            # disabled but changed
    assert _audit(tmp, sc0, False, MARGIN)._check_pitch_gate("SNGS-000", bad).verdict == "FAIL"
    bad = out.drop(columns=["pitch_mean_y"])
    assert _audit(tmp, sc, True, MARGIN)._check_pitch_gate("SNGS-000", bad).verdict == "FAIL"
    # the split_merge audit must see the pre-gate ids: the rebuilt frame holds all 8 tracklets
    pre = out[out["track_id_pregate"].notna()].copy(); pre["track_id"] = pre["track_id_pregate"]
    assert pre["track_id"].nunique() == 8 and out["track_id"].nunique() == 5
    print("audit pitch_gate: WARN at the default no-projection threshold, PASS (enabled / disabled with note) "
          "with a lenient one, WARN on an implausible gated share, nine negative controls FAIL")


def test_stage_enabled_and_disabled():
    _run_stage()


def test_audit():
    _audit_checks(*_run_stage())


if __name__ == "__main__":
    test_rule()
    _audit_checks(*_run_stage())
    print("ALL PITCH_GATE TESTS PASSED")
