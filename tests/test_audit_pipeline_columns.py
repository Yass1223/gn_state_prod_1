"""Test scripts/audit_pipeline_columns.py against synthetic-but-realistic inputs.

Builds a tracker state (.pklz), role_team + team_embed sidecars, and a jersey cache
blob that mirror the real column/field layout, then runs the audit and asserts the
verdicts — for a healthy run and for four deliberately broken variants (each must
turn exactly the intended check to FAIL).
"""
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "scripts" / "audit_pipeline_columns.py"
SEQ = "SNGS-117"


def build(root, *, break_case=None):
    """Create state + sidecars + blob under root. Returns dict of paths/args."""
    root = Path(root)
    frames_dir = root / "data" / "SoccerNetGS" / "test" / SEQ / "img1"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n_frames = 10
    for f in range(n_frames):
        Image.fromarray(rng.integers(0, 255, (200, 320, 3), np.uint8)).save(frames_dir / f"{f+1:06d}.jpg")

    # tracklets: 8 players (2 teams), 1 GK, 1 referee
    tracklets = {}
    def kit(team):  # not used for embeddings here, just structure
        return team
    roles = {}
    teams = {}
    for tid in range(1, 11):
        if tid <= 8:
            roles[tid] = "player"; teams[tid] = "left" if tid % 2 else "right"
        elif tid == 9:
            roles[tid] = "goalkeeper"; teams[tid] = "left"
        else:
            roles[tid] = "referee"; teams[tid] = None

    det_rows, img_rows = [], []
    image_ids = list(range(1000, 1000 + n_frames))
    for k, f in enumerate(image_ids):
        img_rows.append(dict(image_id=f, video_id=SEQ, frame=k,
                             file_path=str(frames_dir / f"{k+1:06d}.jpg"),
                             parameters={"cam": [1, 2, 3]}))
        for tid in range(1, 11):
            emb = rng.normal(size=8).astype(np.float32); emb /= np.linalg.norm(emb)
            det_rows.append(dict(
                image_id=f, video_id=SEQ, track_id=float(tid),
                bbox_ltwh=np.array([20 + 25 * tid + rng.normal(0, 1), 40.0, 18.0, 45.0]),
                bbox_conf=0.9,
                crop_single=bool(k % 2 == 0), crop_rT=float(rng.uniform(0, 0.5)),
                crop_rB=float(rng.uniform(0, 0.5)), crop_trigger=np.nan,
                bbox_pitch={"x_bottom_middle": float(rng.uniform(-40, 40)),
                            "y_bottom_middle": float(rng.uniform(-5, 5))},
                team_embedding=emb,
                role=roles[tid],
                team=teams[tid],
                team_cluster=(0.0 if teams[tid] == "left" else 1.0) if roles[tid] == "player" else np.nan,
                jersey_number_detection=None, jersey_number_confidence=0.0,
                jersey_number=np.nan))
        # add a couple of untracked detections
        det_rows.append(dict(image_id=f, video_id=SEQ, track_id=np.nan,
                             bbox_ltwh=np.array([5.0, 5.0, 10.0, 20.0]), bbox_conf=0.2,
                             crop_single=True, crop_rT=0.0, crop_rB=0.0, crop_trigger=np.nan,
                             bbox_pitch=None, team_embedding=None, role=None, team=None,
                             team_cluster=np.nan, jersey_number_detection=None,
                             jersey_number_confidence=0.0, jersey_number=np.nan))
    det = pd.DataFrame(det_rows)
    # jersey columns are object in the real pipeline (str numbers / None); make them
    # object here so assigning a string number doesn't clash with an all-NaN float column.
    for col in ("jersey_number_detection", "jersey_number", "team", "role"):
        det[col] = det[col].astype(object)
    img = pd.DataFrame(img_rows).set_index("image_id")

    # jersey: players 1..8 numbered; propagate into detection + voted columns
    blob_results = {}
    for tid in range(1, 11):
        if roles[tid] in ("player", "goalkeeper"):
            num = str(tid)
            n_used = 5
            if break_case == "zero_frames" and tid == 1:
                n_used = 0
            blob_results[str(float(tid))] = {"number": num, "confidence": 0.8, "n_used": n_used}
            # populate detection + voted, unless a break case suppresses it
            mask = det["track_id"] == float(tid)
            if not (break_case == "blob_not_in_detection" and tid == 2):
                det.loc[mask, "jersey_number_detection"] = num
                det.loc[mask, "jersey_number_confidence"] = 0.8
                if not (break_case == "voting_drop" and tid == 3):
                    det.loc[mask, "jersey_number"] = num
    # break: referee carries a team
    if break_case == "referee_team":
        det.loc[det["track_id"] == 10.0, "team"] = "left"

    # write state .pklz
    state = root / "state.pklz"
    with zipfile.ZipFile(state, "w", zipfile.ZIP_DEFLATED) as zf:
        b = io.BytesIO(); det.to_pickle(b); zf.writestr(f"{SEQ}.pkl", b.getvalue())
        b = io.BytesIO(); img.to_pickle(b); zf.writestr(f"{SEQ}_image.pkl", b.getvalue())
        zf.writestr("summary.json", "{}")

    # role_team sidecar
    rt_dir = root / "audit" / "role_team"; rt_dir.mkdir(parents=True, exist_ok=True)
    per = []
    for tid in range(1, 11):
        per.append(dict(track_id=float(tid), role=roles[tid],
                        why=("gk_extreme" if roles[tid] == "goalkeeper"
                             else "ref_outlier" if roles[tid] == "referee" else "player"),
                        team=teams[tid], n=10, n_single=5, filt_fallback=False,
                        mx=float(rng.uniform(-40, 40)), sx=5.0, my=0.0, sy=2.0, q75=10.0,
                        out_rule=bool(roles[tid] == "referee"), out_db=False,
                        confirmed=bool(roles[tid] == "referee"),
                        gk_candidate=bool(roles[tid] == "goalkeeper"), z=1.5))
    if break_case == "referee_team":
        for t in per:
            if t["role"] == "referee":
                t["team"] = "left"
    lvl = dict(s_ok=True, named_left_cluster=1, dbscan_eps=0.3,
               cues={"sign": 0, "mean": 1, "quantile": 1, "centroid": 1, "keeper": 0},
               n_player=8, n_goalkeeper=1, n_referee=1, n_left=4, n_right=4,
               distance_median=0.5, distance_mad=0.1)
    (rt_dir / f"{SEQ}.json").write_text(json.dumps(dict(
        sequence=SEQ, params={}, per_tracklet=per, sequence_level=lvl,
        tracklets=10, tracklets_no_embedding=0, tracklets_off_grid=0)))

    # team_embed sidecar
    te_dir = root / "audit" / "team_embed"; te_dir.mkdir(parents=True, exist_ok=True)
    (te_dir / f"{SEQ}.json").write_text(json.dumps(dict(
        sequence=SEQ, tracklets=10, tracklets_with_embedding=10, crops_embedded=50,
        crops_empty=0, tracklets_off_grid=0, embeddings_nonfinite_or_zero=0,
        embedder={"sha256": "abc123", "source": "x", "input_hw": [128, 64], "embedding_dim": 8})))

    # jersey cache blob
    jn_dir = root / "jn_cache"; jn_dir.mkdir(parents=True, exist_ok=True)
    (jn_dir / f"{SEQ}.deadbeef0000.json").write_text(json.dumps(dict(
        manifest_sha256="x", rule="vote_pool", stride=5, legibility_thr=0.72,
        parseq_ckpt="p", parseq_sha256="a"*64, satrn_ckpt="s", satrn_sha256="b"*64,
        results=blob_results)))

    return dict(state=str(state), audit_dir=str(root / "audit"),
                role_team_dir=str(rt_dir), team_embed_dir=str(te_dir),
                jn_cache_dir=str(jn_dir), out=str(root / "report"))


def run_audit(paths, n_crops=8):
    cmd = [sys.executable, str(SCRIPT), "--state", paths["state"],
           "--audit-dir", paths["audit_dir"], "--role-team-dir", paths["role_team_dir"],
           "--team-embed-dir", paths["team_embed_dir"], "--jn-cache-dir", paths["jn_cache_dir"],
           "--out", paths["out"], "--n-crops", str(n_crops)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    rep = json.loads((Path(paths["out"]) / "column_audit.json").read_text())
    return r.returncode, r.stdout, rep


def verdicts(rep, section):
    return {c["check"]: c["verdict"] for c in rep[section]}


if __name__ == "__main__":
    import tempfile
    # 1) healthy run -> no FAIL, crop sheet drawn
    p = build(tempfile.mkdtemp())
    rc, out, rep = run_audit(p)
    assert rc == 0, f"healthy run should pass, got rc={rc}\n{out}"
    assert rep["crop_sheet"]["drawn"] > 0, "crop sheet not drawn"
    assert os.path.exists(Path(p["out"]) / "crop_check.png")
    # every column check present and not FAIL
    colv = verdicts(rep, "columns")
    assert all(v != "FAIL" for v in colv.values()), colv
    print("healthy: PASS (rc=0, crop sheet drawn, no column FAIL)")

    # 2) referee carries a team -> column team/referee-empty AND team_role invariant FAIL
    p = build(tempfile.mkdtemp(), break_case="referee_team")
    rc, out, rep = run_audit(p)
    assert rc != 0
    assert verdicts(rep, "columns")["column:team/referee-empty"] == "FAIL"
    assert any(c["verdict"] == "FAIL" and "referee" in c["note"] for c in rep["team_role"])
    print("referee_team: FAIL fired in COLUMNS and TEAM/ROLE")

    # 3) blob numbered but detection empty -> jersey hop 2->3 FAIL
    p = build(tempfile.mkdtemp(), break_case="blob_not_in_detection")
    rc, out, rep = run_audit(p)
    assert rc != 0
    assert any("blob but empty" in c["note"] for c in rep["jersey"] if c["verdict"] == "FAIL"), \
        [c["note"] for c in rep["jersey"]]
    print("blob_not_in_detection: FAIL fired in JERSEY hop 2->3")

    # 4) detection set but voting dropped it -> jersey hop 3->4 FAIL
    p = build(tempfile.mkdtemp(), break_case="voting_drop")
    rc, out, rep = run_audit(p)
    assert rc != 0
    assert any("dropped by voting" in c["note"] for c in rep["jersey"] if c["verdict"] == "FAIL"), \
        [c["note"] for c in rep["jersey"]]
    print("voting_drop: FAIL fired in JERSEY hop 3->4")

    # 5) numbered from 0 frames -> jersey n_used FAIL
    p = build(tempfile.mkdtemp(), break_case="zero_frames")
    rc, out, rep = run_audit(p)
    assert rc != 0
    assert any("n_used==0" in c["note"] for c in rep["jersey"] if c["verdict"] == "FAIL"), \
        [c["note"] for c in rep["jersey"]]
    print("zero_frames: FAIL fired in JERSEY (n_used==0)")

    print("\nALL AUDIT TESTS PASSED")
