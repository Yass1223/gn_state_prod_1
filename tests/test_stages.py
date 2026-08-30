"""Local end-to-end test of the three new stages on synthetic data.

* a synthetic osnet_team checkpoint in the notebook's format (config + state_dict)
  saved from the SAME architecture, so load_state_dict must report nothing missing
  and nothing unexpected;
* 60 synthetic frames on disk; detections with the pipeline's columns
  (image_id, bbox_ltwh, bbox_conf, track_id, bbox_pitch);
* crop_filter -> team_embed -> role_team, then the column contract the GS encoder
  and the audit rely on.
"""
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sn_gamestate.reid.osnet_ain  # noqa: F401,E402
from sn_gamestate.reid import osnet_team                                    # noqa: E402
from sn_gamestate.crop_filter.crop_filter_api import CropFilter, label_single_frame  # noqa: E402
from sn_gamestate.team.team_embed_api import TeamEmbedding                  # noqa: E402
from sn_gamestate.team.role_team_api import RoleTeamAssignment              # noqa: E402
from sn_gamestate.team import rules                                          # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import notebook_reference as nbref                                            # noqa: E402


def make_checkpoint(path):
    torch.manual_seed(0)
    backbone, _ = sn_gamestate.reid.osnet_ain.build_backbone("osnet_x1_0")
    net = osnet_team.TeamNet(backbone, 512, 256)
    torch.save({"config": dict(img_h=128, img_w=64, emb_dim=256, letterbox=True,
                               mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                "state_dict": net.state_dict(),
                "test_metrics": {"worst5": [("SNGS-118", 0.5)]}}, path)


def make_data(root, n_frames=60, n_tracks=8):
    rng = np.random.default_rng(0)
    img_dir = root / "SNGS-000" / "img1"; img_dir.mkdir(parents=True)
    meta = []
    for f in range(n_frames):
        Image.fromarray(rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)).save(img_dir / f"{f + 1:06d}.jpg")
        meta.append(dict(image_id=1000 + f, video_id=0, frame=f, file_path=str(img_dir / f"{f + 1:06d}.jpg")))
    metadatas = pd.DataFrame(meta).set_index("image_id")
    rows = []
    for f in range(n_frames):
        for tr in range(n_tracks):
            l, t = 20 + 30 * tr + rng.normal(0, 2), 60 + rng.normal(0, 2)
            rows.append(dict(image_id=1000 + f, bbox_ltwh=np.array([l, t, 20.0, 50.0]), bbox_conf=0.9,
                             track_id=float(tr) if tr < n_tracks - 1 else np.nan,
                             bbox_pitch={"x_bottom_middle": (-45.0 if tr == 0 else 45.0 if tr == 1 else rng.uniform(-30, 30)),
                                         "y_bottom_middle": rng.uniform(-5, 5)}))
        # an untracked low-confidence box overlapping track 2 (must NOT make it multi)
        rows.append(dict(image_id=1000 + f, bbox_ltwh=np.array([20 + 60 + 5.0, 65.0, 20.0, 50.0]), bbox_conf=0.15,
                         track_id=np.nan, bbox_pitch=None))
        # a tracked box overlapping track 3 strongly on even frames (must make it multi)
        if f % 2 == 0:
            rows.append(dict(image_id=1000 + f, bbox_ltwh=np.array([20 + 90 + 3.0, 62.0, 20.0, 50.0]), bbox_conf=0.8,
                             track_id=99.0, bbox_pitch={"x_bottom_middle": 1.0, "y_bottom_middle": 1.0}))
    detections = pd.DataFrame(rows)
    detections.index = np.arange(5000, 5000 + len(detections))
    return detections, metadatas


def test_label_single_matches_notebook():
    """label_single_frame == the splitter notebook's label_single on random frames."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = int(rng.integers(1, 12))
        boxes = np.column_stack([rng.uniform(0, 300, n), rng.uniform(0, 300, n), rng.uniform(5, 60, n), rng.uniform(5, 120, n)])
        tracked = rng.random(n) > 0.4
        conf = rng.uniform(0, 1, n)
        for mode, thr in (("tracked", 0.0), ("all", 0.0), ("conf", 0.3), ("tracked_or_conf", 0.5)):
            s, rt, rb, _ = label_single_frame(boxes, tracked, conf, 0.25, 0.40, mode, thr)
            det = pd.DataFrame(dict(l=boxes[:, 0], t=boxes[:, 1], w=boxes[:, 2], h=boxes[:, 3],
                                    conf=conf, track_id=np.where(tracked, 1.0, np.nan), frame=0))
            s2, rt2, rb2 = notebook_label_single(det, 0.25, 0.40, thr, mode)
            assert np.array_equal(s, s2) and np.allclose(rt, rt2) and np.allclose(rb, rb2), mode
    print("label_single_frame == splitter notebook label_single (200 frames x 4 modes)")


def notebook_label_single(det, thr_target, thr_other, conf_thr_other, contam_mode):
    """Verbatim from tracklet-splitter_dbscan_4_raw.ipynb cell 3."""
    single = np.ones(len(det), dtype=bool)
    r_t = np.zeros(len(det), dtype=np.float32); r_b = np.zeros(len(det), dtype=np.float32)
    boxes = det[["l", "t", "w", "h"]].values.astype(np.float64)
    conf = det["conf"].values.astype(np.float64)
    tracked = det["track_id"].notna().values
    for f, idx in pd.Series(np.arange(len(det))).groupby(det["frame"].values):
        idx = idx.values
        if len(idx) < 2: continue
        b = boxes[idx]
        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
        iw = np.clip(np.minimum(x2[:, None], x2[None, :]) - np.maximum(x1[:, None], x1[None, :]), 0, None)
        ih = np.clip(np.minimum(y2[:, None], y2[None, :]) - np.maximum(y1[:, None], y1[None, :]), 0, None)
        inter = iw * ih; np.fill_diagonal(inter, 0.0)
        keep = None
        if contam_mode == "tracked":
            keep = tracked[idx]
        elif contam_mode == "tracked_or_conf":
            keep = tracked[idx] | (conf[idx] > conf_thr_other)
        elif contam_mode == "conf" and conf_thr_other > 0:
            keep = conf[idx] > conf_thr_other
        if keep is not None and not keep.all():
            inter[:, ~keep] = 0.0
        area = np.maximum(b[:, 2] * b[:, 3], 1e-9)
        rt = (inter / area[:, None]).max(axis=1)
        rb = (inter / area[None, :]).max(axis=1)
        r_t[idx] = rt; r_b[idx] = rb
        single[idx] = (rt <= thr_target) & (rb < thr_other)
    return single, r_t, r_b


def test_stages():
    tmp = Path(tempfile.mkdtemp())
    ckpt = tmp / "osnet_team_best.pt"; make_checkpoint(ckpt)
    det, meta = make_data(tmp / "data")
    cf = CropFilter(SimpleNamespace(thr_target=0.25, thr_other=0.40, contam_mode="tracked", conf_thr_other=0.0))
    det = cf.process(det, meta)
    for c in ("crop_single", "crop_rT", "crop_rB", "crop_trigger"):
        assert c in det.columns and len(det[c]) == len(det)
    t2 = det[det.track_id == 2.0]; t3 = det[det.track_id == 3.0]
    assert t2["crop_single"].all(), "untracked low-conf box must not veto track 2"
    even = t3[t3["image_id"] % 2 == 0]
    assert (~even["crop_single"]).all(), "tracked overlapping box must make track 3 multi on even frames"
    assert set(even["crop_trigger"].dropna().map(lambda i: det.at[int(i), "track_id"])) == {99.0}
    print("crop_filter: contract ok (tracked-only veto verified both ways)")

    te = TeamEmbedding(SimpleNamespace(team_local_path=str(ckpt), team_sha256=None, team_repo="x", team_file="y",
                                       team_revision=None, audit_dir=str(tmp / "audit_embed"),
                                       pos_stride=5, crops_per_track=16, batch_size=32), device="cpu")
    det = te.process(det, meta)
    emb = det["team_embedding"]
    n_emb = sum(1 for e in emb if isinstance(e, np.ndarray))
    assert n_emb > 0 and all(e.shape == (256,) and e.dtype == np.float32 for e in emb if isinstance(e, np.ndarray))
    assert all(abs(np.linalg.norm(e) - 1) < 1e-4 for e in emb if isinstance(e, np.ndarray))
    # exactly <=16 crops per tracklet, on the stride grid
    for tid, g in det.dropna(subset=["track_id"]).groupby("track_id"):
        k = sum(1 for e in g["team_embedding"] if isinstance(e, np.ndarray))
        assert 1 <= k <= 16, (tid, k)
        assert all(meta.at[i, "frame"] % 5 == 0 for i, e in zip(g["image_id"], g["team_embedding"]) if isinstance(e, np.ndarray))
    rec = (tmp / "audit_embed" / "SNGS-000.json").read_text()
    assert "sha256" in rec and "crops_embedded" in rec
    print(f"team_embed: {n_emb} crops embedded, sidecar written")

    # flip-TTA / letterbox parity with the notebook's functions on the same weights
    model = te.model
    img = np.asarray(Image.open(meta["file_path"].iloc[0]).convert("RGB"))
    crop = osnet_team.crop_rgb(img, det["bbox_ltwh"].iloc[0])
    ref = notebook_embed(model, [crop])
    got = model.embed([crop])
    assert np.allclose(ref, got, atol=1e-6), "embedding differs from the notebook's embed_osnet"
    print("team_embed: letterbox + flip TTA == notebook embed_osnet")

    rt = RoleTeamAssignment(SimpleNamespace(params=dict(rules.FROZEN_PARAMS), audit_dir=str(tmp / "audit_role"),
                                            pos_stride=5, crops_per_track=16))
    det = rt.process(det, meta)
    tracked = det.dropna(subset=["track_id"])
    assert tracked["role"].isin(["player", "goalkeeper", "referee"]).all(), "every tracked row needs a role"
    assert det.loc[det.track_id.isna(), "role"].isna().all()
    pg = tracked[tracked.role != "referee"]
    assert pg["team"].isin(["left", "right"]).all()
    assert tracked.loc[tracked.role == "referee", "team"].isna().all()
    for tid, g in tracked.groupby("track_id"):
        assert g["role"].nunique() == 1 and g["team"].astype(str).nunique() == 1
    print("role_team:", tracked.groupby("track_id")[["role", "team"]].first().to_dict("index"))
    rec = (tmp / "audit_role" / "SNGS-000.json").read_text()
    assert '"per_tracklet"' in rec and '"cues"' in rec
    print("role_team: contract ok, sidecar written")


@torch.no_grad()
def notebook_embed(model, crops):
    """Notebook embed_osnet (cell 3) against model.net / model.letterbox."""
    import torch.nn.functional as Fn
    xs = np.stack([model.letterbox(c) for c in crops])
    xt = torch.from_numpy(xs)
    e = model.net(xt) + model.net(torch.flip(xt, dims=[3]))
    return Fn.normalize(e, dim=1).cpu().numpy()


if __name__ == "__main__":
    test_label_single_matches_notebook()
    test_stages()
    print("ALL LOCAL STAGE TESTS PASSED")
