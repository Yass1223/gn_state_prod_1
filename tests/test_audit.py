"""Audit checks for crop_filter / pitch_gate / team_embed / role_team on the synthetic run."""
import sys, tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_stages as T
from sn_gamestate.crop_filter.crop_filter_api import CropFilter
from sn_gamestate.pitch_gate.pitch_gate_api import PitchGate
from sn_gamestate.team.team_embed_api import TeamEmbedding
from sn_gamestate.team.role_team_api import RoleTeamAssignment
from sn_gamestate.team import rules
from sn_gamestate.audit.run_audit_api import RunAudit

tmp = Path(tempfile.mkdtemp())
ckpt = tmp / "osnet_team_best.pt"; T.make_checkpoint(ckpt)
det, meta = T.make_data(tmp / "data")
det = CropFilter(SimpleNamespace(thr_target=0.25, thr_other=0.40, contam_mode="tracked", conf_thr_other=0.0)).process(det, meta)
# every synthetic tracklet stands inside the pitch: the gate must change nothing here
det = PitchGate(SimpleNamespace(enabled=True, margin_m=5.0, audit_dir=str(tmp / "audit/pitch_gate"))).process(det, meta)
assert det["track_id"].equals(det["track_id_pregate"]) and not det["pitch_gate_offpitch"].any()
te = TeamEmbedding(SimpleNamespace(team_local_path=str(ckpt), team_sha256=None, team_repo="x", team_file="y", team_revision=None,
                                   audit_dir=str(tmp / "audit/team_embed"), pos_stride=5, crops_per_track=16, batch_size=32), device="cpu")
det = te.process(det, meta)
det = RoleTeamAssignment(SimpleNamespace(params=dict(rules.FROZEN_PARAMS), audit_dir=str(tmp / "audit/role_team"), pos_stride=5, crops_per_track=16)).process(det, meta)
det["jersey_number_detection"] = None; det["jersey_number_confidence"] = 0.0; det["jersey_number"] = np.nan
cfg = SimpleNamespace(out_dir=str(tmp / "audit"), jn_cache_dir=str(tmp / "jn"), calib_dir=str(tmp / "calib"), models_dir=str(tmp / "models"),
                      thresholds={}, track_sidecar_dir=None, expected_tracker={},
                      expected_crop_filter=dict(thr_target=0.25, thr_other=0.40, contam_mode="tracked"),
                      team_embed_sidecar_dir=str(tmp / "audit/team_embed"),
                      expected_team_embed=dict(sha256=None, pos_stride=5, crops_per_track=16,
                                               cluster_method="kmeans2_threshold", outlier_k=3.25),
                      role_team_sidecar_dir=str(tmp / "audit/role_team"), expected_role_team=dict(params=dict(rules.FROZEN_PARAMS)),
                      pitch_gate_sidecar_dir=str(tmp / "audit/pitch_gate"), expected_pitch_gate=dict(enabled=True, margin_m=5.0))
audit = RunAudit(cfg)
import json
audit.process(det, meta)
rep = json.loads((tmp / "audit" / "SNGS-000.json").read_text())
by = {c["component"]: c for c in rep["checks"]}
TE_NAME = "team_embed (osnet_team + team clustering)"
RT_NAME = "role_team (per-trajectory roles + sides, after traj_refine)"
for name in ("crop_filter", "pitch_gate", TE_NAME, RT_NAME):
    print(f"{by[name]['verdict']:4} {name}: {by[name]['note'] or 'ok'}")
assert by["crop_filter"]["verdict"] == "PASS"
assert by["pitch_gate"]["verdict"] == "PASS" and by["pitch_gate"]["observed"]["tracklets_gated"] == 0
assert by[TE_NAME]["verdict"] in ("PASS", "WARN") and "not pinned" in by[TE_NAME]["note"]
assert by[RT_NAME]["verdict"] in ("PASS", "WARN")   # WARN = half fallback / one team, legitimate on random-noise frames
# negative controls: a wrong parameter, a missing embedding, a wrong threshold
cfg.expected_role_team = dict(params=dict(rules.FROZEN_PARAMS, k=2.5))
assert RunAudit(cfg)._check_role_team("SNGS-000", det.dropna(subset=["track_id"])).verdict == "FAIL"
cfg.expected_role_team = dict(params=dict(rules.FROZEN_PARAMS))
cfg.expected_crop_filter = dict(thr_target=0.99, thr_other=0.999, contam_mode="tracked")  # would relabel the multi crops
assert RunAudit(cfg)._check_crop_filter(det, det.dropna(subset=["track_id"])).verdict == "FAIL"
cfg.expected_team_embed = dict(sha256="0" * 64, pos_stride=5, crops_per_track=16,
                               cluster_method="kmeans2_threshold", outlier_k=3.25)
assert RunAudit(cfg)._check_team_embed("SNGS-000", det.dropna(subset=["track_id"])).verdict == "FAIL"
d2 = det.copy(); d2.loc[d2.track_id == 2.0, "role"] = None
cfg.expected_team_embed = dict(sha256=None, pos_stride=5, crops_per_track=16,
                               cluster_method="kmeans2_threshold", outlier_k=3.25)
assert RunAudit(cfg)._check_role_team("SNGS-000", d2.dropna(subset=["track_id"])).verdict == "FAIL"
cfg.expected_pitch_gate = dict(enabled=False, margin_m=5.0)   # switch that ran != configured
assert RunAudit(cfg)._check_pitch_gate("SNGS-000", det).verdict == "FAIL"
cfg.expected_pitch_gate = dict(enabled=True, margin_m=5.0)
# role_team audits the FINAL trajectories directly (the stage is the last
# labelling stage in this architecture; there are no role/team snapshots):
# genuine per-trajectory variance must FAIL.
d3 = det.dropna(subset=["track_id"]).copy()
counts = d3[d3["team"].isin(["left", "right"])].groupby("track_id").size()
tids2 = counts[counts >= 2].index
assert len(tids2), "fixture must hold a track with >= 2 team-labelled rows"
r = d3.index[(d3["track_id"] == tids2[0]) & d3["team"].isin(["left", "right"])][0]
d3.loc[r, "team"] = "right" if d3.loc[r, "team"] == "left" else "left"
c_live = RunAudit(cfg)._check_role_team("SNGS-000", d3)
assert c_live.verdict == "FAIL" and "team varies" in str(c_live.note)
print("audit: positives PASS, six negative controls FAIL (wrong param, wrong "
      "thresholds, wrong pin, missing role, wrong gate switch, team variance)")
