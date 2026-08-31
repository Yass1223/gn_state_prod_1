"""Tests of the jersey stage's single-crop selection and of the audit's jersey check.

    .venv/bin/python tests/test_jersey_single_crops.py

1. ``test_manifest``: ``JNGsrTrackletRecognizer._build_manifest`` with
   ``single_crops_only`` on keeps only crop_single detections of role-eligible
   tracklets, skips a tracklet with no single crop, reports the counts, and folds the
   flag into the cache key; with the flag off every detection is kept.
2. ``test_audit_jersey``: with a fabricated cache blob in the stage's format, the
   audit's jersey check passes on a consistent state and fails on: a number on a
   tracklet without a single crop, a blob without the flag, a blob with the other
   value, a manifest tracklet count that disagrees with the state, and a manifest crop
   count that disagrees with the single crops of the eligible tracklets.

Needs the project venv (tracklab is imported by both modules); no torch, GPU or
JN checkpoints are used (the recogniser workers are never launched).
"""
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sn_gamestate.jersey.jn_gsr_api import JNGsrTrackletRecognizer, RULE   # noqa: E402
from sn_gamestate.audit.run_audit_api import RunAudit                       # noqa: E402


def make_state(tmp):
    """4 tracklets: 1 player with single+multi crops, 2 goalkeeper all single,
    3 player all multi (no single crop), 4 referee (role-excluded); 2 untracked rows."""
    frames = np.arange(10)
    meta = pd.DataFrame(dict(video_id="SNGS-000", frame=frames,
                             file_path=[str(tmp / "SNGS-000" / "img1" / f"{f + 1:06d}.jpg") for f in frames]),
                        index=1000 + frames)
    rows = []
    for f in frames:
        rows.append(dict(image_id=1000 + f, track_id=1.0, role="player", crop_single=(f % 2 == 0)))
        rows.append(dict(image_id=1000 + f, track_id=2.0, role="goalkeeper", crop_single=True))
        rows.append(dict(image_id=1000 + f, track_id=3.0, role="player", crop_single=False))
        rows.append(dict(image_id=1000 + f, track_id=4.0, role="referee", crop_single=True))
    rows.append(dict(image_id=1000, track_id=np.nan, role=None, crop_single=True))
    rows.append(dict(image_id=1001, track_id=np.nan, role=None, crop_single=False))
    det = pd.DataFrame(rows)
    det["bbox_ltwh"] = [np.array([10.0, 10.0, 20.0, 40.0])] * len(det)
    det["bbox_conf"] = 0.9
    return det, meta


def make_stage(tmp, single_crops_only):
    models = tmp / "models"; (models / "recog2").mkdir(parents=True, exist_ok=True)
    (models / "parseq_gsr_ft_s1.ckpt").write_bytes(b"parseq")
    (models / "recog2" / "best_recog_word_acc_epoch_10.pth").write_bytes(b"satrn")
    cfg = SimpleNamespace(pipeline_dir=str(tmp / "plugin"), venv_python=str(tmp / "no_python"),
                          models_dir=str(models), cache_dir=str(tmp / "jn_cache"),
                          single_crops_only=single_crops_only)
    return JNGsrTrackletRecognizer(cfg, "cpu")


def test_manifest():
    tmp = Path(tempfile.mkdtemp())
    det, meta = make_state(tmp)
    st = make_stage(tmp, True)
    manifest, stats = st._build_manifest(det, meta)
    assert set(manifest) == {"1.0", "2.0"}, set(manifest)
    assert len(manifest["1.0"]) == 5 and len(manifest["2.0"]) == 10
    assert stats == dict(tracklets_skipped_role=1, tracklets_skipped_no_single=1,
                         frames_excluded_multi=15, manifest_tracklets=2, manifest_frames=15), stats
    # chronological
    assert [Path(f[0]).name for f in manifest["1.0"]] == [f"{f + 1:06d}.jpg" for f in range(0, 10, 2)]
    h_on = st._manifest_hash(manifest, RULE, st.stride)
    st_off = make_stage(tmp, False)
    manifest_off, stats_off = st_off._build_manifest(det, meta)
    assert set(manifest_off) == {"1.0", "2.0", "3.0"} and len(manifest_off["1.0"]) == 10
    assert stats_off["frames_excluded_multi"] == 0 and stats_off["tracklets_skipped_no_single"] == 0
    assert st_off._manifest_hash(manifest, RULE, st.stride) != h_on, "the flag is part of the cache key"
    # missing column is a contract error, not a silent fallback
    try:
        st._build_manifest(det.drop(columns=["crop_single"]), meta)
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass
    print("manifest: single crops only, counts, cache key, contract - ok")


def write_blob(tmp, stage, det, meta, manifest, stats, results, **override):
    seq = "SNGS-000"
    mhash = stage._manifest_hash(manifest, RULE, stage.stride)
    blob = {"manifest_sha256": mhash, "rule": RULE, "stride": stage.stride,
            "legibility_thr": stage.legibility_thr, "single_crops_only": stage.single_crops_only,
            "manifest_stats": stats,
            "parseq_ckpt": stage.parseq_ckpt, "parseq_sha256": stage._ckpt_id(stage.parseq_ckpt),
            "satrn_ckpt": stage.satrn_ckpt, "satrn_sha256": stage._ckpt_id(stage.satrn_ckpt),
            "results": results}
    blob.update(override)
    for k in [k for k, v in override.items() if v is None]:
        blob.pop(k)
    for old in (tmp / "jn_cache").glob(f"{seq}.*.json"):
        old.unlink()
    (tmp / "jn_cache").mkdir(exist_ok=True)
    (tmp / "jn_cache" / f"{seq}.{mhash[:12]}.json").write_text(json.dumps(blob))


def apply_results(det, results):
    det = det.copy()
    det["jersey_number_detection"] = None
    det["jersey_number_confidence"] = 0.0
    for tid, r in results.items():
        m = det["track_id"].map(lambda v: None if pd.isna(v) else str(v)) == tid
        if str(r["number"]).isdigit():
            det.loc[m, "jersey_number_detection"] = str(int(r["number"]))
            det.loc[m, "jersey_number_confidence"] = r["confidence"]
    return det


def test_audit_jersey():
    tmp = Path(tempfile.mkdtemp())
    det, meta = make_state(tmp)
    st = make_stage(tmp, True)
    manifest, stats = st._build_manifest(det, meta)
    results = {"1.0": {"number": "7", "confidence": 0.9, "n_used": 3},
               "2.0": {"number": "-1", "confidence": 0.0, "n_used": 2}}
    out = apply_results(det, results)
    tracked = out.dropna(subset=["track_id"])
    write_blob(tmp, st, det, meta, manifest, stats, results)
    cfg = SimpleNamespace(out_dir=str(tmp / "audit"), jn_cache_dir=str(tmp / "jn_cache"),
                          calib_dir=str(tmp / "calib"), models_dir=str(tmp / "models"),
                          jn_roles=["player", "goalkeeper"], jn_single_crops_only=True, thresholds={})
    c = RunAudit(cfg)._check_jersey("SNGS-000", tracked)
    assert c.verdict in ("PASS", "WARN"), (c.verdict, c.note)
    assert "provenance" in c.note or c.verdict == "PASS"   # only the provenance file may WARN here
    assert c.observed["eligible_tracklets"] == 2 and c.observed["role_eligible_without_single_crop"] == 1
    assert c.observed["manifest_stats"]["manifest_frames"] == 15
    print(f"audit jersey: {c.verdict} ({c.note})")

    def verdict(tracked_, **override):
        write_blob(tmp, st, det, meta, manifest, stats, results, **override)
        return RunAudit(cfg)._check_jersey("SNGS-000", tracked_)

    # 1) a number on the tracklet with no single crop
    bad = tracked.copy(); bad.loc[bad["track_id"] == 3.0, "jersey_number_detection"] = "9"
    r = verdict(bad); assert r.verdict == "FAIL" and "without a single crop carry" in r.note, r.note
    # 2) blob without the flag
    r = verdict(tracked, single_crops_only=None); assert r.verdict == "FAIL" and "does not record" in r.note
    # 3) blob with the other value
    r = verdict(tracked, single_crops_only=False); assert r.verdict == "FAIL" and "!= configured" in r.note
    # 4) manifest tracklet count disagrees with the state
    r = verdict(tracked, manifest_stats=dict(stats, manifest_tracklets=3)); assert r.verdict == "FAIL"
    # 5) manifest crop count disagrees with the single crops of the eligible tracklets
    r = verdict(tracked, manifest_stats=dict(stats, manifest_frames=20)); assert r.verdict == "FAIL"
    # 6) with the flag off, tracklet 3 is eligible and must be in the blob -> missing -> FAIL
    cfg.jn_single_crops_only = False
    r = verdict(tracked, single_crops_only=False)
    assert r.verdict == "FAIL" and "absent from the blob" in r.note, r.note
    print("audit jersey: six negative controls FAIL")


if __name__ == "__main__":
    test_manifest()
    test_audit_jersey()
    print("ALL JERSEY SINGLE-CROP TESTS PASSED")
