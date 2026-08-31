"""Tests of the split_merge stage: port == notebook, behaviour, stage + audit contract.

    .venv/bin/python tests/test_split_merge.py

1. ``test_port_equals_notebook``: on random videos, the port in
   ``sn_gamestate/track/split_merge.py`` produces the same fragment labels, the same
   trajectories and the same pass counts as the notebook cells kept verbatim in
   ``notebook_split_merge_reference.py`` (numpy / pandas / scikit-learn only).
2. ``test_behaviour``: a two-identity tracklet is split; same-identity tracklets with
   disjoint frames merge; a frame-overlapping one does not; pass 1 keeps one detection
   per trajectory and frame with the clean one winning; pass 2 places a clean-less
   tracklet and leaves a detection with no free slot unassigned.
3. ``test_stage_and_audit``: the ``SplitMerge`` stage with a stubbed feature extractor
   (needs torch + tracklab, as test_stages.py) writes ``track_id`` and its sidecar, and
   the audit's split_merge check passes on it and fails on four broken variants.
"""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sn_gamestate.track import split_merge as sm            # noqa: E402
import notebook_split_merge_reference as nb                  # noqa: E402

EPS, MIN_SAMPLES, TAU = 0.2, 5, 0.60
D = 16


def _ident(rng, n_ident):
    C = rng.normal(size=(n_ident, D))
    return C / np.linalg.norm(C, axis=1, keepdims=True)


def _draw(rng, centre, n, spread):
    X = centre[None, :] + spread * rng.normal(size=(n, D))
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


def random_video(rng, n_tracks=12, n_ident=8, n_frames=120):
    """Tracklets of 1 or 2 identities over a frame range, with overlapping (multi)
    rows and a few zero embeddings; several tracklets per frame."""
    C = _ident(rng, n_ident)
    E, single, frames, tids = [], [], [], []
    for tid in range(1, n_tracks + 1):
        start = int(rng.integers(0, n_frames - 20))
        length = int(rng.integers(6, 40))
        fr = np.arange(start, min(n_frames, start + length))
        n = len(fr)
        ids = [int(rng.integers(n_ident))]
        if rng.random() < 0.4:
            ids.append(int(rng.integers(n_ident)))
        cut = n if len(ids) == 1 else int(rng.integers(1, n))
        X = np.vstack([_draw(rng, C[ids[0]], cut, 0.15), _draw(rng, C[ids[-1]], n - cut, 0.15)])
        sg = rng.random(n) > 0.25
        zero = rng.random(n) < 0.03
        X[zero] = 0.0
        E.append(X); single.append(sg); frames.append(fr); tids.append(np.full(n, tid))
    return (np.vstack(E), np.concatenate(single), np.concatenate(frames),
            np.concatenate(tids))


def notebook_pipeline(E, single, frames, tids):
    """The notebook's driver: split per tracklet (run_configs, one config), fragment
    ids ``tid * 10000 + label``, then run_full's per-sequence body."""
    frag = np.zeros(len(E), dtype=np.int64)
    ks, noises = {}, {}
    for tid in np.unique(tids):
        idx = np.where(tids == tid)[0]
        idx = idx[np.argsort(frames[idx], kind="stable")]
        Ei = E[idx].astype(np.float32)
        Dm = np.clip(1.0 - Ei @ Ei.T, 0.0, 2.0).astype(np.float64) if len(idx) > 1 \
            else np.zeros((len(idx), len(idx)))
        lab, k, nn = nb.split_tracklet(Ei, single[idx], frames[idx], np.ones(len(idx)),
                                       EPS, MIN_SAMPLES, D=Dm)
        frag[idx] = int(tid) * 10000 + lab
        ks[int(tid)], noises[int(tid)] = k, nn
    rec = dict(E=E, frag=frag, single=single, frames=frames)
    t, info, r1, r2 = nb.run_full_one(rec, TAU)
    return frag, ks, noises, t, info, r1, r2


def test_port_equals_notebook(n_videos=40):
    rng = np.random.default_rng(0)
    n_rows = n_split = n_merge = n_unassigned = 0
    for v in range(n_videos):
        E, single, frames, tids = random_video(rng, n_tracks=int(rng.integers(4, 16)))
        frag_nb, ks, noises, t_nb, info_nb, r1_nb, r2_nb = notebook_pipeline(E, single, frames, tids)
        t_pt, rep = sm.split_merge_video(E, single, frames, tids, EPS, MIN_SAMPLES, TAU)
        assert np.array_equal(rep["frag"], frag_nb), f"video {v}: fragment labels differ"
        for p in rep["split"]["per_tracklet"]:
            assert (p["k"], p["noise"]) == (ks[p["track_id"]], noises[p["track_id"]]), (v, p)
        assert np.array_equal(t_pt, t_nb), f"video {v}: trajectories differ"
        m = rep["merge"]
        for key in ("n_fragments", "n_ordinary", "n_allmulti", "n_traj", "n_merges"):
            assert m[key] == info_nb[key], (v, key, m[key], info_nb[key])
        assert rep["pass1"] == r1_nb, (v, rep["pass1"], r1_nb)
        assert (rep["pass2"]["placed"], rep["pass2"]["unassigned"]) == (r2_nb["placed"], r2_nb["unassigned"]), v
        assert len(m["merge_distances"]) == m["n_merges"] and all(d <= TAU for d in m["merge_distances"])
        n_rows += len(E); n_split += rep["split"]["tracklets_split"]
        n_merge += m["n_merges"]; n_unassigned += rep["pass2"]["unassigned"]
    assert n_split > 0 and n_merge > 0, "the random videos exercised neither a split nor a merge"
    print(f"port == notebook on {n_videos} random videos ({n_rows} rows, {n_split} tracklets split, "
          f"{n_merge} merges, {n_unassigned} unassigned)")


def build_scenario(rng):
    """Hand-built video with a known answer (see the module docstring, point 2). The
    identities are orthogonal unit vectors (cosine distance 1.0 > TAU), so the expected
    answer does not depend on the seed; the seed only draws the within-identity noise."""
    C = np.eye(D)[:3]
    A, B = C[0], C[1]
    rows = []  # (tid, frame, identity, single)
    rows += [(1, f, A, True) for f in range(0, 30)] + [(1, f, B, True) for f in range(30, 60)]  # id switch
    rows += [(2, f, A, True) for f in range(70, 100)]                                            # A again, disjoint
    rows += [(3, f, A, True) for f in range(20, 50)]                                             # A, overlaps 1
    rows += [(1, 85, A, False)]                                                                  # multi row of 1 on a frame 2 owns
    rows += [(4, f, A, False) for f in range(105, 115)]                                          # all-multi tracklet
    tids = np.array([r[0] for r in rows]); frames = np.array([r[1] for r in rows])
    single = np.array([r[3] for r in rows])
    E = np.vstack([_draw(rng, r[2], 1, 0.10) for r in rows])
    return E, single, frames, tids


def test_behaviour():
    rng = np.random.default_rng(1)
    E, single, frames, tids = build_scenario(rng)
    traj, rep = sm.split_merge_video(E, single, frames, tids, EPS, MIN_SAMPLES, TAU)
    per = {p["track_id"]: p for p in rep["split"]["per_tracklet"]}
    assert per[1]["k"] == 2, "tracklet 1 holds two identities and must split"
    assert per[2]["k"] == 1 and per[3]["k"] == 1
    assert per[4]["k"] == 1 and per[4]["n_core"] == 0, "all-multi tracklet stays one fragment"
    frag = rep["frag"]
    frag_1A = frag[(tids == 1) & (frames < 30)]; frag_1B = frag[(tids == 1) & (frames >= 30) & single]
    assert len(set(frag_1A)) == 1 and len(set(frag_1B)) == 1 and frag_1A[0] != frag_1B[0]
    t = {}
    for name, mask in (("1A", (tids == 1) & (frames < 30)), ("1B", (tids == 1) & (frames >= 30) & (frames < 60)),
                       ("2", tids == 2), ("3", tids == 3)):
        vals = set(traj[mask].tolist()); assert len(vals) == 1, (name, vals); t[name] = vals.pop()
    assert t["1A"] == t["2"], "identity A on disjoint frames must merge"
    assert t["1B"] != t["1A"], "identity B must not join A"
    assert t["3"] not in (t["1A"], t["1B"]), "tracklet 3 shares frames with both and must stay apart"
    assert rep["merge"]["n_merges"] == 1 and rep["merge"]["n_traj"] == 3
    # pass 1: the multi row of tracklet 1 at frame 85 collides with the clean row of tracklet 2
    multi_row = np.where((tids == 1) & (frames == 85))[0][0]
    assert rep["pass1"]["discarded"] == 1 and rep["pass1"]["single_anomaly"] == 0
    # pass 2: A's trajectory holds frame 85 (tracklet 2) and tracklet 3 does not reach
    # it, but B's trajectory ends at 59, so frame 85 is free there and in tracklet 3's;
    # the row is placed in the nearer of the two.
    assert traj[multi_row] in (t["1B"], t["3"]) and traj[multi_row] != t["1A"]
    # the all-multi tracklet 4 (identity A, frames 105-114) is placed into an identity-A
    # trajectory (1A+2 or 3, both A; nearest mean decides per row), never into B's
    assert set(traj[tids == 4].tolist()) <= {t["1A"], t["3"]} and (traj[tids == 4] >= 0).all()
    assert rep["pass2"]["unassigned"] == 0
    # invariants
    assigned = traj >= 0
    pairs = list(zip(traj[assigned].tolist(), frames[assigned].tolist()))
    assert len(pairs) == len(set(pairs)), "one detection per trajectory and frame"
    for tr in set(traj[assigned].tolist()):
        assert single[(traj == tr)].any(), "every trajectory holds a clean detection"
    # a detection with no free slot anywhere stays unassigned: give every trajectory a
    # clean detection on frame 85 (B gets one through tracklet 1), then add an
    # overlapping detection of a new identity on frame 85
    B = np.eye(D)[1]                                     # identity B of build_scenario
    E3 = np.vstack([E, _draw(rng, B, 1, 0.1), _draw(rng, np.eye(D)[5], 1, 0.1)])
    traj3, rep3 = sm.split_merge_video(
        E3, np.append(single, [True, False]), np.append(frames, [85, 85]), np.append(tids, [1, 5]),
        EPS, MIN_SAMPLES, TAU)
    assert traj3[-1] == -1 and rep3["pass2"]["unassigned"] >= 1
    print("behaviour: split, merge, no-merge on shared frames, pass 1, pass 2, unassigned - ok")


def test_stage_and_audit():
    from sn_gamestate.track import split_merge_api as api
    from sn_gamestate.audit.run_audit_api import RunAudit
    tmp = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(2)
    E, single, frames, tids = build_scenario(rng)
    n = len(E)
    # detections in the pipeline's layout, shuffled, plus untracked rows
    order = rng.permutation(n)
    det = pd.DataFrame(dict(image_id=1000 + frames[order], bbox_ltwh=[np.array([10.0, 10.0, 20.0, 40.0])] * n,
                            bbox_conf=0.9, track_id=tids[order].astype(float), crop_single=single[order]))
    det = pd.concat([det, pd.DataFrame(dict(image_id=[1000, 1001], bbox_ltwh=[np.array([0.0, 0.0, 5.0, 5.0])] * 2,
                                            bbox_conf=0.2, track_id=np.nan, crop_single=True))], ignore_index=True)
    det.index = np.arange(5000, 5000 + len(det))
    feats_by_index = {idx: E[order][i] for i, idx in enumerate(det.index[:n])}
    meta = pd.DataFrame(dict(video_id="SNGS-000", frame=np.arange(120),
                             file_path=[str(tmp / "SNGS-000" / "img1" / f"{f + 1:06d}.jpg") for f in range(120)]),
                        index=1000 + np.arange(120))
    meta.index.name = "image_id"

    stage = api.SplitMerge.__new__(api.SplitMerge)
    stage.cfg, stage.device = None, "cpu"
    stage.eps, stage.min_samples, stage.tau, stage.batch_size = EPS, MIN_SAMPLES, TAU, 64
    stage.audit_dir = tmp / "audit" / "split_merge"; stage.audit_dir.mkdir(parents=True)
    SHA = "a" * 64
    stage.embedder = SimpleNamespace(info={"sha256": SHA, "precision": "fp32"}, dim=D)

    def fake_features(dets, metadatas, record):
        F = np.stack([feats_by_index[i] for i in dets.index]).astype(np.float32)
        record["inputs"].update(frames_without_path=0, frames_unreadable=0, degenerate_boxes=0,
                                zero_embeddings=int((~F.any(axis=1)).sum()))
        return F
    stage._extract_features = fake_features

    out = stage.process(det, meta)
    assert len(out) == len(det) and list(out.index) == list(det.index)
    assert out.loc[det["track_id"].isna(), "track_id"].isna().all(), "untracked rows stay NaN"
    tracked = out.dropna(subset=["track_id"])
    assert tracked["track_id"].nunique() == 3
    assert not tracked.duplicated(subset=["image_id", "track_id"]).any()
    assert tracked.groupby("track_id")["crop_single"].apply(lambda s: s.astype(bool).any()).all()
    side = json.loads((stage.audit_dir / "SNGS-000.json").read_text())
    assert side["ran"] and side["inputs"]["tracked"] == n and side["outputs"]["tracklets"] == 3
    assert side["merge"]["merges"] == 1 and len(side["per_trajectory"]) == 3
    print(f"stage: {side['inputs']['tracklets']} tracklets -> {side['split']['fragments']} fragments -> "
          f"{side['merge']['trajectories']} trajectories; sidecar written")

    cfg = SimpleNamespace(out_dir=str(tmp / "audit"), jn_cache_dir=str(tmp / "jn"), calib_dir=str(tmp / "calib"),
                          models_dir=str(tmp / "models"), thresholds={},
                          split_merge_sidecar_dir=str(stage.audit_dir),
                          expected_split_merge=dict(eps=EPS, min_samples=MIN_SAMPLES, tau=TAU, ain_sha256=SHA))
    c = RunAudit(cfg)._check_split_merge("SNGS-000", out, tracked)
    assert c.verdict == "PASS", (c.verdict, c.note)
    print(f"audit split_merge: PASS ({c.observed['merges']} merges, "
          f"unassigned share {c.observed['unassigned_share']})")
    # negative controls
    cfg.expected_split_merge = dict(eps=EPS, min_samples=MIN_SAMPLES, tau=0.55, ain_sha256=SHA)
    assert RunAudit(cfg)._check_split_merge("SNGS-000", out, tracked).verdict == "FAIL", "tau mismatch"
    cfg.expected_split_merge = dict(eps=EPS, min_samples=MIN_SAMPLES, tau=TAU, ain_sha256="b" * 64)
    assert RunAudit(cfg)._check_split_merge("SNGS-000", out, tracked).verdict == "FAIL", "digest mismatch"
    cfg.expected_split_merge = dict(eps=EPS, min_samples=MIN_SAMPLES, tau=TAU, ain_sha256=SHA)
    assert RunAudit(cfg)._check_split_merge("SNGS-999", out, tracked).verdict == "FAIL", "missing sidecar"
    bad = out.copy()
    ids = sorted(bad["track_id"].dropna().unique())
    bad.loc[bad["track_id"] == ids[1], "track_id"] = ids[0]        # two trajectories under one id
    assert RunAudit(cfg)._check_split_merge("SNGS-000", bad, bad.dropna(subset=["track_id"])).verdict == "FAIL", \
        "tracklet count / collisions"
    bad = out.copy()
    bad.loc[bad["track_id"] == ids[2], "crop_single"] = False       # a trajectory without a clean row
    assert RunAudit(cfg)._check_split_merge("SNGS-000", bad, bad.dropna(subset=["track_id"])).verdict == "FAIL", \
        "trajectory without clean detection"
    print("audit split_merge: five negative controls FAIL")


if __name__ == "__main__":
    test_port_equals_notebook()
    test_behaviour()
    test_stage_and_audit()
    print("ALL SPLIT_MERGE TESTS PASSED")
