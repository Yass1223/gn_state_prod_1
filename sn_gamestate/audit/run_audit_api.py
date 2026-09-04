"""Run audit: turns every silent degradation of this pipeline into a verdict.

Neither BroadTrack nor the jersey stage aborts the run when it fails: each logs
an ERROR and returns, and the pipeline runs to completion with empty columns.
The role/team stage labels every tracked row, but a tracklet the team-embedding
stage never reached falls back to "player, no team"; the radar only draws rows
with a defined team. A run can therefore finish, evaluate and render while a
component silently did nothing. This module is the LAST stage of the pipeline
and, per sequence, states for each component what it was supposed to do, what
was observed, and a verdict:

    PASS   observed what the component is supposed to produce
    WARN   produced, but degraded beyond the configured threshold
    FAIL   absent, empty, inconsistent, or a recorded failure
    INFO   observation only (no independent expectation can be checked here)

It never modifies detections. Per sequence it writes ``<out_dir>/<seq>.json``
and logs a table. ``scripts/verify_run_integrity.py`` reads those files and
refuses the run's metrics on any FAIL.

Thresholds live in the module config (``modules/audit/run_audit.yaml``) with
the reason for each; the JSON keeps the raw counts so they can be revisited.

The jersey-number component (the stage most recently changed) is checked in
the most detail: the per-sequence cache blob written by ``jn_gsr_api`` must
exist, carry ``rule == vote_pool`` and the real sha256 of BOTH staged
checkpoints, answer every eligible tracklet, and agree with the detection
columns; the two provenance files written at provisioning must record a
SATRN download that is a torch container with a recovered config and a
matching hash, and a PARSeq checkpoint that matches upstream.
"""
import glob
import hashlib
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from tracklab.pipeline.videolevel_module import VideoLevelModule

from sn_gamestate.visualization.pitch import radar_color

log = logging.getLogger(__name__)

RULE = "vote_pool"
PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _is_nan(v):
    """True for None/NaN scalars only; arrays, dicts and strings are values."""
    if v is None:
        return True
    if isinstance(v, (np.ndarray, list, tuple, dict, str)):
        return False
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _is_number(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _tid(v):
    """Canonical tracklet id string: jn_gsr_api keys its manifest/blob with
    str(track_id), which is '3.0' for a float column and '3' for an int one."""
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else str(v)
    except (TypeError, ValueError):
        return str(v)


def _share(num, den):
    return float(num) / float(den) if den else 0.0


def _sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


class Check:
    def __init__(self, component, expected):
        self.component, self.expected = component, expected
        self.observed, self.verdict, self.note = {}, PASS, ""

    def set(self, verdict, note=""):
        order = {INFO: 0, PASS: 1, WARN: 2, FAIL: 3}
        if order[verdict] > order[self.verdict]:
            self.verdict = verdict
        if note:
            self.note = (self.note + "; " if self.note else "") + note
        return self

    def to_dict(self):
        return {"component": self.component, "expected": self.expected,
                "observed": self.observed, "verdict": self.verdict,
                "note": self.note}


class RunAudit(VideoLevelModule):
    """Final stage: per-sequence verdict for every component. Read-only."""

    input_columns = []       # column presence is itself a finding, checked at runtime
    output_columns = []

    def __init__(self, cfg, device=None, tracking_dataset=None, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.out_dir = Path(str(cfg.out_dir))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.jn_cache_dir = Path(str(cfg.jn_cache_dir))
        self.calib_dir = Path(str(cfg.calib_dir))
        # Score below which the calibration stage rejects a frame (broadtrack.yaml
        # min_score); the audit counts such frames in the camera JSON.
        self.calib_min_score = float(getattr(cfg, "calib_min_score", 0.3))
        self.models_dir = Path(str(cfg.models_dir))
        self.parseq_ckpt = str(getattr(cfg, "parseq_ckpt", "parseq_gsr_ft_s1.ckpt"))
        self.satrn_ckpt = str(getattr(cfg, "satrn_ckpt",
                                      "recog2/best_recog_word_acc_epoch_10.pth"))
        self.jn_roles = set(getattr(cfg, "jn_roles", ["player", "goalkeeper"]))
        # jersey stage crop selection (jn_gsr.yaml single_crops_only); see _check_jersey
        self.jn_single_crops_only = bool(getattr(cfg, "jn_single_crops_only", True))
        # Sidecars of the team-embedding and role/team stages (must equal those
        # modules' audit_dir) and what their configs declare, resolved by OmegaConf.
        d = getattr(cfg, "team_embed_sidecar_dir", None)
        self.team_embed_sidecar_dir = Path(str(d)) if d else None
        d = getattr(cfg, "role_team_sidecar_dir", None)
        self.role_team_sidecar_dir = Path(str(d)) if d else None
        self.expected_crop_filter = {str(k): v for k, v in
                                     (getattr(cfg, "expected_crop_filter", {}) or {}).items()}
        self.expected_team_embed = {str(k): v for k, v in
                                    (getattr(cfg, "expected_team_embed", {}) or {}).items()}
        self.expected_role_team = {str(k): v for k, v in
                                   (getattr(cfg, "expected_role_team", {}) or {}).items()}
        # Pitch-gate stage sidecar (written by sn_gamestate.pitch_gate.pitch_gate_api)
        # and the switch / margin it must have run with.
        d = getattr(cfg, "pitch_gate_sidecar_dir", None)
        self.pitch_gate_sidecar_dir = Path(str(d)) if d else None
        self.expected_pitch_gate = {str(k): v for k, v in
                                    (getattr(cfg, "expected_pitch_gate", {}) or {}).items()}
        # Trajectory-refinement stage sidecar (sn_gamestate.refine.traj_refine_api)
        # and the settings / checkpoint it must have run with. The refine stage
        # relabels track_id AFTER pitch_gate/role_team/jersey, so those checks run
        # on its per-row snapshots (track_id_prerefine, jersey snapshots).
        d = getattr(cfg, "traj_refine_sidecar_dir", None)
        self.traj_refine_sidecar_dir = Path(str(d)) if d else None
        self.expected_traj_refine = {str(k): v for k, v in
                                     (getattr(cfg, "expected_traj_refine", {}) or {}).items()}
        thr = getattr(cfg, "thresholds", None) or {}
        g = lambda k, d: float(thr[k]) if k in thr else d  # noqa: E731
        self.thr = {
            "empty_frames_warn": g("empty_frames_warn", 0.20),
            "single_share_warn": g("single_share_warn", 0.30),
            "embed_missing_warn": g("embed_missing_warn", 0.01),
            "off_grid_warn": g("off_grid_warn", 0.05),
            "tracked_warn": g("tracked_warn", 0.50),
            "pitch_missing_warn": g("pitch_missing_warn", 0.05),
            "jn_min_eligible_for_zero_fail": g("jn_min_eligible_for_zero_fail", 10),
            "radar_skipped_tracked_warn": g("radar_skipped_tracked_warn", 0.05),
            "cmc_identity_warn": g("cmc_identity_warn", 0.02),
            "crop_clipped_warn": g("crop_clipped_warn", 0.001),
            "zero_emb_warn": g("zero_emb_warn", 0.01),
            "tracklet_split_zero_emb_warn": g("tracklet_split_zero_emb_warn", 0.01),
            "pitch_gate_gated_warn": g("pitch_gate_gated_warn", 0.50),
            "pitch_gate_no_position_warn": g("pitch_gate_no_position_warn", 0.05),
            "calib_lost_frames_warn": g("calib_lost_frames_warn", 0.10),
        }
        # Splitter stage sidecar (written by sn_gamestate.track.tracklet_split_api)
        # and the settings/checkpoint it must have run with.
        d = getattr(cfg, "tracklet_split_sidecar_dir", None)
        self.tracklet_split_sidecar_dir = Path(str(d)) if d else None
        self.expected_tracklet_split = {str(k): v for k, v in
                                        (getattr(cfg, "expected_tracklet_split", {}) or {}).items()}
        # Tracker diagnostics sidecars (written by sn_gamestate.track.bot_sort) and
        # the declared tracker/embedder configuration to hold the run against. The
        # expected values arrive resolved by OmegaConf interpolation from the track
        # and tracklet_split module configs (see modules/audit/run_audit.yaml), so a
        # mismatch between what RAN and what the configs DECLARE is detectable.
        sidecar = getattr(cfg, "track_sidecar_dir", None)
        self.track_sidecar_dir = Path(str(sidecar)) if sidecar else None
        self.expected_tracker = {str(k): v for k, v in
                                 (getattr(cfg, "expected_tracker", {}) or {}).items()}
        self._ckpt_sha = {}

    # ------------------------------------------------------------ helpers --
    def _ckpt_id(self, rel):
        if rel not in self._ckpt_sha:
            p = self.models_dir / rel
            try:
                self._ckpt_sha[rel] = _sha256(p)
            except OSError:
                self._ckpt_sha[rel] = None
        return self._ckpt_sha[rel]

    @staticmethod
    def _seq_name(metadatas):
        if len(metadatas) and "file_path" in metadatas.columns:
            return Path(str(metadatas["file_path"].iloc[0])).parent.parent.name
        if len(metadatas) and "video_id" in metadatas.columns:
            return str(metadatas["video_id"].iloc[0])
        return "unknown"

    @staticmethod
    def _per_track_constant(tracked, col):
        """[track ids] whose `col` takes more than one distinct non-null value."""
        if col not in tracked.columns:
            return None
        bad = []
        for tid, grp in tracked.groupby("track_id"):
            vals = {str(v) for v in grp[col] if not _is_nan(v)}
            if len(vals) > 1:
                bad.append(str(tid))
        return bad

    # ------------------------------------------------------------- checks --
    def _check_detector(self, det, meta):
        c = Check("bbox_detector", "at least one detection on (nearly) every frame")
        n_frames = len(meta)
        frames_with = det["image_id"].nunique() if "image_id" in det.columns else 0
        c.observed = {"frames": n_frames, "frames_with_detections": frames_with,
                      "detections": len(det)}
        if len(det) == 0 or frames_with == 0:
            return c.set(FAIL, "no detections at all")
        empty = _share(n_frames - frames_with, n_frames)
        c.observed["empty_frame_share"] = round(empty, 4)
        if empty > self.thr["empty_frames_warn"]:
            c.set(WARN, f"{empty:.1%} frames without detections")
        return c

    def _check_crop_filter(self, det, tracked):
        c = Check("crop_filter", "crop_single / crop_rT / crop_rB on every detection; thresholds and "
                                 "contaminator mode as configured; only tracked boxes veto (tracked-only)")
        for col in ("crop_single", "crop_rT", "crop_rB", "crop_trigger"):
            if col not in det.columns:
                c.observed[col] = "column missing"
                c.set(FAIL, f"{col} column missing")
        if c.verdict == FAIL:
            return c
        c.observed["expected"] = dict(self.expected_crop_filter)
        null = int(det["crop_single"].apply(_is_nan).sum())
        c.observed["crop_single_null"] = null
        if len(det) and null:
            c.set(FAIL, f"crop_single missing on {null} detections")
        thr_t = self.expected_crop_filter.get("thr_target")
        thr_b = self.expected_crop_filter.get("thr_other")
        single = det["crop_single"].astype(bool)
        if thr_t is not None and thr_b is not None:
            recomputed = (det["crop_rT"].astype(float) <= float(thr_t)) & (det["crop_rB"].astype(float) < float(thr_b))
            mism = int((recomputed != single).sum())
            c.observed["labels_inconsistent_with_ratios"] = mism
            if mism:
                c.set(FAIL, f"{mism} labels disagree with the stored ratios at the configured thresholds")
        share = _share(int(single.reindex(tracked.index).fillna(False).astype(bool).sum()), len(tracked)) if "track_id" in det.columns else 0.0
        c.observed["single_share_tracked"] = round(share, 4)
        if len(tracked) and share < self.thr["single_share_warn"]:
            c.set(WARN, f"only {share:.1%} of tracked detections labelled single")
        # tracked-only rule: every veto must come from a box that was tracked when the
        # filter ran. traj_refine can later unassign a detection (track_id NaN) in
        # stage 3b; count it. The pitch gate also clears track_id, so the id as the
        # splitter left it (track_id_pregate) is used when the gate stage ran.
        mode = str(self.expected_crop_filter.get("contam_mode", "tracked"))
        trig = det["crop_trigger"]
        fired = trig.notna()
        c.observed["multi_labels"] = int((~single).sum())
        c.observed["multi_without_trigger"] = int(((~single) & ~fired).sum())
        if int(((~single) & ~fired).sum()):
            c.set(FAIL, "multi labels with no recorded trigger box")
        if mode == "tracked" and fired.any():
            tid_now = det["track_id_pregate"] if "track_id_pregate" in det.columns else det["track_id"]
            trig_idx = trig[fired].astype(int)
            valid = trig_idx.isin(det.index)
            if not valid.all():
                c.set(FAIL, f"{int((~valid).sum())} trigger indices not in the frame")
            now_untracked = int(tid_now.reindex(trig_idx[valid]).isna().sum())
            c.observed["vetoes_by_box_now_untracked"] = now_untracked
            if now_untracked:
                c.set(INFO, f"{now_untracked} vetoes came from boxes later left unassigned")
        return c

    def _check_track(self, det, tracked):
        c = Check("track + tracklet_split", "most detections carry a track_id; "
                                         "tracklets are non-trivial")
        if "track_id" not in det.columns:
            c.observed["track_id"] = "column missing"
            return c.set(FAIL, "track_id column missing")
        share = _share(len(tracked), len(det))
        lens = tracked.groupby("track_id").size() if len(tracked) else pd.Series(dtype=int)
        c.observed = {"tracked_share": round(share, 4), "tracklets": int(len(lens)),
                      "tracklet_len_median": float(lens.median()) if len(lens) else 0.0,
                      "tracklet_len_min": int(lens.min()) if len(lens) else 0}
        if len(det) and len(tracked) == 0:
            return c.set(FAIL, "no detection has a track_id")
        if share < self.thr["tracked_warn"]:
            c.set(WARN, f"only {share:.1%} of detections tracked")
        return c

    # ------------------------------------------------ tracker internals --
    def _find_track_sidecar(self, seq, metadatas):
        """(path, parsed json) of this sequence's tracker sidecar, else (None, None).

        The sidecar is named after the tracker's `video_id`, which may differ from
        the audit's path-derived sequence name, so the match is by exact filename
        first, then by the json's own `video_id` against the sequence name or any
        video_id present in `metadatas`.
        """
        d = self.track_sidecar_dir
        if d is None or not d.is_dir():
            return None, None
        vids = ({str(v) for v in metadatas["video_id"].unique()}
                if "video_id" in metadatas.columns else set())
        exact = d / f"{seq}.json"
        paths = ([exact] if exact.is_file() else []) + sorted(
            p for p in d.glob("*.json") if p != exact)
        for p in paths:
            try:
                data = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            vid = str(data.get("video_id"))
            if p == exact or vid == seq or vid in vids:
                return p, data
        return None, None

    def _check_tracker_internals(self, seq, metadatas):
        c = Check(
            "track internals (BoT-SORT \u00b7 SOF + OSNet-AIN)",
            "track and tracklet_split configs pin the same OSNet-AIN weights (both stages "
            "build the same embedder module, so the arithmetic is identical by "
            "construction); a diagnostics sidecar covers every frame; the settings "
            "and checkpoint digest that RAN equal the ones the configs declare; "
            "camera motion mostly non-identity; no clipped crops, zero embeddings or "
            "dropped rows")
        exp = self.expected_tracker

        # Config-level identity between the two embedding stages first: each stage
        # sha-verifies its own load at runtime, so equal pins guarantee equal weights
        # even before the sidecar is opened.
        for a, b, what in (("ain_sha256_track", "ain_sha256_tracklet_split", "ain_sha256"),
                           ("ain_file_track", "ain_file_tracklet_split", "ain_file"),
                           ("ain_revision_track", "ain_revision_tracklet_split", "ain_revision")):
            va, vb = exp.get(a), exp.get(b)
            c.observed[what] = {"track": va, "tracklet_split": vb}
            if va in (None, "", "None") or vb in (None, "", "None"):
                c.set(FAIL, f"{what} not pinned in both track and tracklet_split configs")
            elif str(va) != str(vb):
                c.set(FAIL, f"track and tracklet_split disagree on {what}")

        path, data = self._find_track_sidecar(seq, metadatas)
        c.observed["sidecar"] = str(path) if path else None
        if data is None:
            return c.set(FAIL, "no tracker sidecar for this sequence - the tracker's "
                               "internals ran unobserved (audit_dir unset, unwritable, "
                               "or the track stage did not run)")

        frames = data.get("frames") or []
        n = len(frames)
        n_meta = int(len(metadatas))
        c.observed.update({"frames_recorded": n, "frames_in_sequence": n_meta})
        if n < n_meta:
            c.set(FAIL, f"sidecar covers {n} of {n_meta} frames - "
                        f"{n_meta - n} frame(s) never reached the tracker")

        st = data.get("settings") or {}
        hp = st.get("hyperparams") or {}
        emb = st.get("embedder") or {}
        ran_app = hp.get("appearance_thresh")
        want_app = exp.get("appearance_thresh")
        c.observed["ran_appearance_thresh"] = ran_app
        if want_app is not None and ran_app is not None \
                and abs(float(ran_app) - float(want_app)) > 1e-9:
            c.set(FAIL, f"appearance_thresh that ran ({ran_app}) "
                        f"!= configured ({want_app})")
        ran_scale = st.get("sof_scale")
        want_scale = exp.get("sof_scale")
        c.observed["ran_sof_scale"] = ran_scale
        if want_scale is not None and ran_scale is not None \
                and abs(float(ran_scale) - float(want_scale)) > 1e-9:
            c.set(FAIL, f"sof_scale that ran ({ran_scale}) != configured ({want_scale})")
        got_sha = emb.get("sha256")
        want_sha = exp.get("ain_sha256_track")
        c.observed["ran_embedder_sha256"] = got_sha
        # Recorded, not asserted: fp16 autocast is baked in and a CPU-only run
        # legitimately resolves to fp32, so the arithmetic is informational.
        c.observed["ran_embedder_precision"] = emb.get("precision")
        if want_sha not in (None, "", "None") and got_sha is not None \
                and str(got_sha) != str(want_sha):
            c.set(FAIL, f"embedder sha256 that ran ({got_sha}) "
                        f"!= configured ({want_sha})")

        if n:
            ident = sum(1 for f in frames[1:] if f.get("identity"))
            att = max(1, n - 1)   # frame 1 is identity by construction; not counted
            share = ident / att
            c.observed.update({"identity_warps": int(ident),
                               "identity_share": round(share, 4)})
            if n > 1 and ident == att:
                c.set(FAIL, "camera motion NEVER produced a warp (all identity)")
            elif share > self.thr["cmc_identity_warn"]:
                c.set(WARN, f"{share:.1%} frames fell back to an identity warp")

            n_high = sum(int(f.get("n_high", 0)) for f in frames)
            clipped = sum(int(f.get("clipped", 0)) for f in frames)
            zero = sum(int(f.get("zero_emb", 0)) for f in frames)
            dropped = sum(int(f.get("dropped", 0)) for f in frames)
            corners = sum(int(f.get("corners", 0)) for f in frames)
            c.observed.update({"high_conf_detections": n_high,
                               "clipped_crops": clipped, "zero_embeddings": zero,
                               "dropped_rows": dropped})
            if n_high:
                if zero == n_high:
                    c.set(FAIL, "every high-confidence detection embedded to zero")
                elif _share(zero, n_high) > self.thr["zero_emb_warn"]:
                    c.set(WARN, f"{_share(zero, n_high):.1%} zero embeddings "
                                f"among high-confidence detections")
                if _share(clipped, n_high) > self.thr["crop_clipped_warn"]:
                    c.set(WARN, f"{_share(clipped, n_high):.1%} crops "
                                f"clipped to nothing")
            if dropped:
                c.set(WARN, f"{dropped} tracker output row(s) dropped "
                            f"(detection index outside the frame)")
            if n > 1 and corners == 0:
                c.set(WARN, "the motion estimator never found a keypoint")
        return c

    # ------------------------------------------------- split / merge stage --
    def _check_tracklet_split(self, seq, det, tracked):
        """Inputs, internals and outputs of the tracklet_split stage.

        The stage is Stage 1 of the refinement method and SPLIT ONLY: the
        pipeline's one merge is ``traj_refine``, so any merging evidence here
        (a merge threshold in the settings, merge or pass sections in the
        sidecar, dropped rows) is a FAIL. ``tracked`` must hold the track ids
        as the splitter LEFT them: when the pitch_gate stage ran afterwards,
        ``process`` rebuilds it from ``track_id_pregate``."""
        c = Check("tracklet_split (DBSCAN split, no merging)",
                  "sidecar for the sequence; settings (eps, min_samples) and checkpoint "
                  "that ran equal the configured ones; NO merge threshold anywhere; "
                  "crop_single received; per-tracklet fragment counts sum to the "
                  "fragment total; every tracked row stays assigned; fragments equal "
                  "the trajectories in the state; one detection per frame per "
                  "fragment; every fragment has exactly one source tracklet "
                  "(track_id_presplit)")
        exp = self.expected_tracklet_split
        c.observed["expected"] = dict(exp)
        data = self._read_sidecar(self.tracklet_split_sidecar_dir, seq)
        c.observed["sidecar"] = (str(self.tracklet_split_sidecar_dir / f"{seq}.json")
                                if self.tracklet_split_sidecar_dir else None)
        if data is None:
            return c.set(FAIL, "no tracklet_split sidecar for this sequence (stage did "
                               "not run, audit_dir unset, or unwritable)")

        # --- settings and checkpoint that ran; a merge threshold is a violation
        st = data.get("settings") or {}
        emb = data.get("embedder") or {}
        c.observed["ran"] = dict(st)
        c.observed["ran_embedder_sha256"] = emb.get("sha256")
        c.observed["ran_embedder_precision"] = emb.get("precision")
        for key in ("eps", "min_samples"):
            want, got = exp.get(key), st.get(key)
            if want is None or got is None:
                c.set(FAIL, f"{key} not declared (config) or not recorded (sidecar)")
            elif abs(float(got) - float(want)) > 1e-9:
                c.set(FAIL, f"{key} that ran ({got}) != configured ({want})")
        if "tau" in st or "tau" in exp:
            c.set(FAIL, "a merge threshold (tau) is configured or ran on the splitter; "
                        "the splitter must not merge - the pipeline's one merge is "
                        "traj_refine")
        want_sha = exp.get("ain_sha256")
        if want_sha in (None, "", "None"):
            c.set(FAIL, "ain_sha256 not pinned in the tracklet_split config")
        elif not emb.get("sha256"):
            c.set(FAIL, "sidecar records no embedder digest")
        elif str(emb.get("sha256")) != str(want_sha):
            c.set(FAIL, f"embedder sha256 that ran ({emb.get('sha256')}) != "
                        f"configured ({want_sha})")

        # --- inputs the stage received
        inp = data.get("inputs") or {}
        n_in = int(inp.get("tracked") or 0)
        c.observed["inputs"] = dict(inp)
        if not data.get("ran", False):
            if n_in == 0 and len(tracked) == 0:
                return c.set(INFO, "no tracked detection reached the stage")
            return c.set(FAIL, "the stage recorded that it did not run on this sequence")
        if inp.get("crop_single_present") is not True:
            c.set(FAIL, "the stage did not receive the crop filter's crop_single column")
        n_zero = int(inp.get("zero_embeddings") or 0)
        if n_in:
            if n_zero == n_in:
                c.set(FAIL, "every tracked detection embedded to zero")
            elif _share(n_zero, n_in) > self.thr["tracklet_split_zero_emb_warn"]:
                c.set(WARN, f"{_share(n_zero, n_in):.1%} tracked detections with an "
                            f"all-zero embedding (they attach by the deterministic "
                            f"tie rules)")
        if int(inp.get("frames_without_path") or 0) or int(inp.get("frames_unreadable") or 0):
            c.set(WARN, f"{inp.get('frames_without_path')} frame(s) without path, "
                        f"{inp.get('frames_unreadable')} unreadable")

        # --- internal consistency of the split; merging evidence is a FAIL
        sp = data.get("split") or {}
        outp = data.get("outputs") or {}
        c.observed["split"] = {k: v for k, v in sp.items() if k != "per_tracklet"}
        c.observed["outputs"] = dict(outp)
        for bad in ("merge", "pass1", "pass2"):
            if bad in data:
                c.set(FAIL, f"the splitter sidecar holds a '{bad}' section; merging "
                            f"and placement do not happen in this stage")
        if "merges" in outp or "rows_unassigned" in outp and int(outp.get("rows_unassigned") or 0):
            c.set(FAIL, "the splitter sidecar reports merges or unassigned rows")
        n_trk_in = int(inp.get("tracklets") or 0)
        n_frag = int(sp.get("fragments") or 0)
        per = sp.get("per_tracklet") or []
        if n_frag < n_trk_in:
            c.set(FAIL, f"{n_frag} fragments from {n_trk_in} tracklets (the split can "
                        f"only add)")
        if per and sum(int(p.get("k") or 0) for p in per) != n_frag:
            c.set(FAIL, "the per-tracklet fragment counts do not sum to the fragment "
                        "total")
        if int(outp.get("fragments") or 0) != n_frag:
            c.set(FAIL, f"outputs.fragments ({outp.get('fragments')}) != "
                        f"split.fragments ({n_frag})")
        if int(outp.get("rows_assigned") or 0) != n_in:
            c.set(FAIL, f"rows_assigned ({outp.get('rows_assigned')}) != tracked input "
                        f"({n_in}); the splitter keeps every row")
        if int(outp.get("frame_collisions") or 0):
            c.set(FAIL, f"the stage reported {outp.get('frame_collisions')} frame "
                        f"collision(s) in its output")
        if int(outp.get("fragments_multi_origin") or 0):
            c.set(FAIL, f"{outp.get('fragments_multi_origin')} fragment(s) mix "
                        f"detections from more than one source tracklet")

        # --- outputs, recomputed from the detections the audit receives
        if "track_id" not in det.columns:
            return c.set(FAIL, "track_id column missing")
        n_tracked_now = int(len(tracked))
        n_frag_now = int(tracked["track_id"].nunique()) if n_tracked_now else 0
        c.observed["detections_tracked_now"] = n_tracked_now
        c.observed["detections_fragments_now"] = n_frag_now
        if n_tracked_now != n_in:
            c.set(FAIL, f"{n_tracked_now} tracked detections in the state, the stage "
                        f"received {n_in}; the splitter never drops a row")
        if n_frag_now != n_frag:
            c.set(FAIL, f"{n_frag_now} trajectories in the state, the stage reports "
                        f"{n_frag} fragments")
        if n_tracked_now:
            coll = int(tracked.duplicated(subset=["image_id", "track_id"]).sum())
            c.observed["frame_collisions_recomputed"] = coll
            if coll:
                c.set(FAIL, f"{coll} (image_id, track_id) collision(s): two "
                            f"detections of one fragment in one frame")
            if "track_id_presplit" in det.columns:
                origin = tracked.join(det["track_id_presplit"], how="left") \
                    if "track_id_presplit" not in tracked.columns else tracked
                per_origin = origin.groupby("track_id")["track_id_presplit"].nunique()
                n_mixed = int((per_origin > 1).sum())
                c.observed["fragments_multi_origin_recomputed"] = n_mixed
                if n_mixed:
                    c.set(FAIL, f"{n_mixed} fragment(s) hold detections from more "
                                f"than one source tracklet")
            else:
                c.set(FAIL, "track_id_presplit column missing from the state; the "
                            "splitter's origin snapshot did not survive")
            if "crop_single" in tracked.columns:
                has_clean = tracked.groupby("track_id")["crop_single"].apply(
                    lambda s: bool(s.astype(bool).any()))
                n_noclean = int((~has_clean).sum())
                c.observed["fragments_without_clean_recomputed"] = n_noclean
                if n_noclean != int(outp.get("fragments_without_clean") or 0):
                    c.set(FAIL, f"fragments without a clean detection: recomputed "
                                f"{n_noclean}, sidecar "
                                f"{outp.get('fragments_without_clean')}")
                lens = tracked.groupby("track_id").size()
                c.observed["fragment_len_median"] = float(lens.median())
                c.observed["fragment_len_min"] = int(lens.min())
            else:
                c.set(FAIL, "crop_single column missing from the state")
        c.observed["tracklets_split"] = sp.get("tracklets_split")
        return c

    # --------------------------------------------------- pitch gate stage --
    def _check_pitch_gate(self, seq, det):
        """Recomputes the gate from the detections and holds the stage to its switch.

        Columns present; sidecar present with the switch and margin equal to the
        configured ones; per pre-gate tracklet the mean projected position and the
        rule outcome recomputed from bbox_pitch equal the stored columns; with the
        gate enabled every off-pitch row has track_id NaN and every other tracked
        row kept its id, with the gate disabled track_id equals track_id_pregate
        everywhere; counts equal the sidecar's; gated share and share of tracklets
        without a projection within thresholds."""
        from sn_gamestate.pitch_gate.pitch_gate_api import gate_tracklets
        c = Check("pitch_gate", "track_id_pregate / pitch_gate_offpitch / pitch_mean_x / pitch_mean_y on "
                                "every row; sidecar with the switch and margin that ran equal to the "
                                "configured ones; mean position and rule recomputed from bbox_pitch equal "
                                "the stored columns; track_id cleared on off-pitch tracklets iff enabled; "
                                "counts equal the sidecar's")
        exp = self.expected_pitch_gate
        c.observed["expected"] = dict(exp)
        for col in ("track_id_pregate", "pitch_gate_offpitch", "pitch_mean_x", "pitch_mean_y"):
            if col not in det.columns:
                c.observed[col] = "column missing"
                c.set(FAIL, f"{col} column missing")
        if "track_id" not in det.columns or "bbox_pitch" not in det.columns:
            c.set(FAIL, "track_id or bbox_pitch column missing")
        if c.verdict == FAIL:
            return c
        data = self._read_sidecar(self.pitch_gate_sidecar_dir, seq)
        c.observed["sidecar"] = (str(self.pitch_gate_sidecar_dir / f"{seq}.json")
                                if self.pitch_gate_sidecar_dir else None)
        if data is None:
            return c.set(FAIL, "no pitch_gate sidecar for this sequence (stage did not run, "
                               "audit_dir unset, or unwritable)")
        ran_enabled, ran_margin = data.get("enabled"), data.get("margin_m")
        c.observed["ran"] = dict(enabled=ran_enabled, margin_m=ran_margin, pitch=data.get("pitch"))
        want_enabled, want_margin = exp.get("enabled"), exp.get("margin_m")
        if want_enabled is None or ran_enabled is None:
            c.set(FAIL, "enabled not declared (config) or not recorded (sidecar)")
        elif bool(want_enabled) != bool(ran_enabled):
            c.set(FAIL, f"enabled that ran ({ran_enabled}) != configured ({want_enabled})")
        if want_margin is None or ran_margin is None:
            c.set(FAIL, "margin_m not declared (config) or not recorded (sidecar)")
        elif abs(float(ran_margin) - float(want_margin)) > 1e-9:
            c.set(FAIL, f"margin_m that ran ({ran_margin}) != configured ({want_margin})")
        if ran_margin is None:
            return c
        margin = float(ran_margin)
        enabled = bool(ran_enabled)

        # --- recompute the gate on the ids as the stage received them
        pre = det.copy()
        pre["track_id"] = pre["track_id_pregate"]
        mean_x, mean_y, off, per = gate_tracklets(pre, margin)
        got_x = det["pitch_mean_x"].astype(float)
        got_y = det["pitch_mean_y"].astype(float)
        got_off = det["pitch_gate_offpitch"].astype(bool)
        same_x = (np.isclose(mean_x, got_x, atol=1e-6, equal_nan=True))
        same_y = (np.isclose(mean_y, got_y, atol=1e-6, equal_nan=True))
        n_mean_mismatch = int((~(same_x & same_y)).sum())
        n_rule_mismatch = int((off.to_numpy() != got_off.to_numpy()).sum())
        c.observed.update(rows=int(len(det)), mean_mismatch_rows=n_mean_mismatch,
                          rule_mismatch_rows=n_rule_mismatch)
        if n_mean_mismatch:
            c.set(FAIL, f"{n_mean_mismatch} rows whose stored mean position differs from the "
                        f"one recomputed from bbox_pitch")
        if n_rule_mismatch:
            c.set(FAIL, f"{n_rule_mismatch} rows whose pitch_gate_offpitch differs from the rule "
                        f"at margin {margin}")

        # --- track_id changed exactly as the switch says. The gate must be held
        # to the ids as IT left them: when the traj_refine stage ran afterwards
        # (and relabelled merged trajectories), those are in track_id_prerefine.
        tid_now = (det["track_id_prerefine"] if "track_id_prerefine" in det.columns
                   else det["track_id"])
        tid_pre = det["track_id_pregate"]
        if enabled:
            off_with_id = int((got_off & tid_now.notna()).sum())
            kept = (~got_off) & tid_pre.notna()
            kept_changed = int((kept & ~(tid_now == tid_pre)).sum())
            new_ids = int((tid_pre.isna() & tid_now.notna()).sum())
            c.observed.update(offpitch_rows_still_tracked=off_with_id,
                              onpitch_rows_with_changed_id=kept_changed, rows_with_new_id=new_ids)
            if off_with_id:
                c.set(FAIL, f"{off_with_id} off-pitch rows still carry a track_id although the gate is enabled")
            if kept_changed or new_ids:
                c.set(FAIL, f"the gate changed track_id on {kept_changed + new_ids} rows it must not touch")
        else:
            changed = int((~((tid_now == tid_pre) | (tid_now.isna() & tid_pre.isna()))).sum())
            c.observed["rows_with_changed_id"] = changed
            if changed:
                c.set(FAIL, f"gate disabled but track_id differs from track_id_pregate on {changed} rows")

        # --- counts against the sidecar (on the same pre-refine ids)
        n_trk = len(per)
        n_off = sum(1 for p in per if p["off_pitch"])
        n_nopos = sum(1 for p in per if p["n_positions"] == 0)
        rows_gated_now = int((tid_pre.notna() & tid_now.isna()).sum())
        c.observed.update(tracklets=n_trk, tracklets_off_pitch=n_off,
                          tracklets_without_position=n_nopos,
                          tracklets_gated=n_off if enabled else 0, rows_gated=rows_gated_now,
                          tracked_rows_before=int(tid_pre.notna().sum()),
                          tracked_rows_after=int(tid_now.notna().sum()),
                          off_pitch_track_ids=[p["track_id"] for p in per if p["off_pitch"]])
        for key, val in (("tracklets", n_trk), ("tracklets_off_pitch", n_off),
                         ("tracklets_without_position", n_nopos),
                         ("tracklets_gated", n_off if enabled else 0), ("rows_gated", rows_gated_now)):
            rec = data.get(key)
            if rec is None or int(rec) != int(val):
                c.set(FAIL, f"sidecar {key} ({rec}) != recomputed ({val})")
        share_off = _share(n_off, n_trk)
        share_nopos = _share(n_nopos, n_trk)
        c.observed.update(off_pitch_share=round(share_off, 4), no_position_share=round(share_nopos, 4))
        if n_trk and share_off > self.thr["pitch_gate_gated_warn"]:
            c.set(WARN, f"{share_off:.1%} of the tracklets are off-pitch at margin {margin} m "
                        f"(implausible; check the calibration)")
        if n_trk and share_nopos > self.thr["pitch_gate_no_position_warn"]:
            c.set(WARN, f"{share_nopos:.1%} of the tracklets have no projection and could not be gated")
        if n_trk and not enabled and n_off:
            c.set(INFO, f"gate disabled: {n_off} off-pitch tracklet(s) kept")
        return c

    def _check_calibration(self, seq, tracked):
        c = Check("calibration", "bbox_pitch on every tracked detection; "
                                 "non-empty camera JSON for the sequence")
        if "bbox_pitch" not in tracked.columns:
            c.observed["bbox_pitch"] = "column missing"
            c.set(FAIL, "bbox_pitch column missing")
        else:
            has = int(tracked["bbox_pitch"].apply(lambda b: isinstance(b, dict)).sum())
            miss = _share(len(tracked) - has, len(tracked))
            c.observed.update({"tracked_with_bbox_pitch": has,
                               "missing_share": round(miss, 4)})
            if len(tracked) and has == 0:
                c.set(FAIL, "no tracked detection has bbox_pitch")
            elif miss > self.thr["pitch_missing_warn"]:
                c.set(WARN, f"{miss:.1%} tracked detections without bbox_pitch")
        cj = self.calib_dir / f"{seq}.json"
        c.observed["camera_json"] = str(cj)
        if not cj.is_file():
            c.set(FAIL, "camera JSON absent")
        else:
            try:
                data = json.loads(cj.read_text())
                c.observed["camera_json_frames"] = len(data) if hasattr(data, "__len__") else None
                if not data:
                    c.set(FAIL, "camera JSON empty")
                elif isinstance(data, dict):
                    # The binary records every frame, lost ones included (main.cpp:
                    # score < 0.3 = tracking lost, reinit attempted; after > 5 lost
                    # frames with score < 0.2 the camera is reset to the prior). The
                    # calibration stage rejects frames below min_score and reuses the
                    # last accepted camera; a large lost share means many reused
                    # (stale) cameras, so the projections of this sequence are suspect.
                    scores = [float(v.get("score", 0.0)) for v in data.values()
                              if isinstance(v, dict) and _is_number(v.get("score", None))]
                    n_reinit = sum(1 for v in data.values()
                                   if isinstance(v, dict) and bool(v.get("reinit", False)))
                    n_lost = sum(1 for sc in scores if sc < self.calib_min_score)
                    share_lost = _share(n_lost, len(scores))
                    c.observed.update({
                        "frames_with_score": len(scores),
                        "frames_below_min_score": n_lost,
                        "lost_share": round(share_lost, 4),
                        "frames_reinit": n_reinit,
                        "score_median": round(float(np.median(scores)), 4) if scores else None,
                        "score_min": round(min(scores), 4) if scores else None,
                        "min_score": self.calib_min_score})
                    if scores and share_lost > self.thr["calib_lost_frames_warn"]:
                        c.set(WARN, f"{share_lost:.1%} of the calibrated frames are below "
                                    f"min_score={self.calib_min_score} (tracking lost; their "
                                    f"detections use the last accepted camera)")
            except (OSError, json.JSONDecodeError) as e:
                c.set(FAIL, f"camera JSON unreadable: {e}")
        return c

    def _eligible_tids(self, tracked):
        """Tracklets the jersey stage recognises: role (per-track constant, from the
        role_team stage) in jn_roles and, with jn_single_crops_only, at least one
        crop_single detection (the stage hands only single crops to the recognisers
        and skips a tracklet that has none)."""
        if "role" not in tracked.columns:
            return set()
        out = set()
        for tid, grp in tracked.groupby("track_id"):
            mode = grp["role"].mode()
            if not (len(mode) and mode.iloc[0] in self.jn_roles):
                continue
            if self.jn_single_crops_only and not self._has_single(grp):
                continue
            out.add(_tid(tid))
        return out

    @staticmethod
    def _has_single(grp):
        return "crop_single" in grp.columns and bool(grp["crop_single"].astype(bool).any())

    def _role_eligible_without_single(self, tracked):
        """Role-eligible tracklets that hold no single crop (left unnumbered by design
        when single crops only)."""
        if "role" not in tracked.columns:
            return set()
        out = set()
        for tid, grp in tracked.groupby("track_id"):
            mode = grp["role"].mode()
            if len(mode) and mode.iloc[0] in self.jn_roles and not self._has_single(grp):
                out.add(_tid(tid))
        return out

    def _check_jersey(self, seq, tracked):
        """``tracked`` must hold the rows as the jersey stage saw them: pre-refine
        ids, and -- through ``jn_col`` -- the pre-refine number/confidence snapshots
        when the traj_refine stage ran afterwards (it reassigns conflicting
        numbers and propagates merged ones, which is ITS contract, checked in
        ``_check_traj_refine``)."""
        c = Check("jersey_number_detect",
                  f"per-sequence cache blob with rule={RULE}, real sha256 of both "
                  f"checkpoints, single_crops_only as configured, an answer for every "
                  f"eligible tracklet, no number on a tracklet without a single crop, "
                  f"per-track constant jersey_number_detection consistent with the blob")
        jn_col = ("jersey_number_detection_prerefine"
                  if "jersey_number_detection_prerefine" in tracked.columns
                  else "jersey_number_detection")
        c.observed["number_column"] = jn_col
        eligible = self._eligible_tids(tracked)
        c.observed["eligible_tracklets"] = len(eligible)
        c.observed["single_crops_only"] = self.jn_single_crops_only
        no_single = self._role_eligible_without_single(tracked)
        if self.jn_single_crops_only:
            c.observed["role_eligible_without_single_crop"] = len(no_single)
            if "crop_single" not in tracked.columns:
                c.set(FAIL, "crop_single column missing; single-crop eligibility cannot be checked")
            if no_single:
                c.set(INFO, f"{len(no_single)} role-eligible tracklet(s) hold no single crop and are "
                            f"left unnumbered by design")

        for col in ("jersey_number_detection", "jersey_number_confidence"):
            if col not in tracked.columns:
                c.observed[col] = "column missing"
                c.set(FAIL, f"{col} column missing")
        if c.verdict == FAIL:
            return c

        # per-track constancy + coverage from the detection columns
        bad = self._per_track_constant(tracked, jn_col)
        c.observed["tracks_with_inconsistent_number"] = len(bad)
        if bad:
            c.set(FAIL, f"{len(bad)} tracks with >1 {jn_col} value")
        numbered = set()
        for tid, grp in tracked.groupby("track_id"):
            vals = [v for v in grp[jn_col] if not _is_nan(v)]
            if vals:
                numbered.add(_tid(tid))
        c.observed["eligible_numbered"] = len(numbered & eligible)
        c.observed["numbered_outside_eligible"] = len(numbered - eligible)
        if self.jn_single_crops_only and (numbered & no_single):
            c.set(FAIL, f"{len(numbered & no_single)} tracklet(s) without a single crop carry a "
                        f"number; with single_crops_only such a tracklet never reaches the recognisers")
        if numbered - eligible - no_single:
            c.set(WARN, "numbers on tracklets outside the role filter")
        if (len(eligible) >= self.thr["jn_min_eligible_for_zero_fail"]
                and not (numbered & eligible)):
            c.set(FAIL, f"0 of {len(eligible)} eligible tracklets numbered")

        # the cache blob jn_gsr_api wrote for this sequence
        blobs = sorted(glob.glob(str(self.jn_cache_dir / f"{seq}.*.json")),
                       key=os.path.getmtime)
        c.observed["cache_blobs"] = len(blobs)
        if not blobs:
            return c.set(FAIL, "no jersey cache blob for this sequence "
                               "(stage did not run, or its workers failed)")
        try:
            blob = json.loads(open(blobs[-1]).read())
        except (OSError, json.JSONDecodeError) as e:
            return c.set(FAIL, f"cache blob unreadable: {e}")
        c.observed["cache_blob"] = os.path.basename(blobs[-1])
        c.observed["rule"] = blob.get("rule")
        if blob.get("rule") != RULE:
            c.set(FAIL, f"blob rule {blob.get('rule')!r} != {RULE!r}")
        if "single_crops_only" not in blob:
            c.set(FAIL, "blob does not record single_crops_only (written by an older stage)")
        elif bool(blob.get("single_crops_only")) != self.jn_single_crops_only:
            c.set(FAIL, f"blob single_crops_only={blob.get('single_crops_only')} != "
                        f"configured {self.jn_single_crops_only}")
        ms = blob.get("manifest_stats") or {}
        if ms:
            c.observed["manifest_stats"] = dict(ms)
            n_eligible_blob = int(ms.get("manifest_tracklets") or 0)
            if n_eligible_blob != len(eligible):
                c.set(FAIL, f"blob manifest held {n_eligible_blob} tracklets, the state has "
                            f"{len(eligible)} eligible ones")
            if self.jn_single_crops_only and "crop_single" in tracked.columns:
                elig_rows = tracked[tracked["track_id"].map(_tid).isin(eligible)]
                n_single_rows = int(elig_rows["crop_single"].astype(bool).sum())
                if int(ms.get("manifest_frames") or 0) != n_single_rows:
                    c.set(FAIL, f"blob manifest held {ms.get('manifest_frames')} crops, the eligible "
                                f"tracklets hold {n_single_rows} single crops")
        for key, rel in (("parseq_sha256", self.parseq_ckpt),
                         ("satrn_sha256", self.satrn_ckpt)):
            v = str(blob.get(key, ""))
            ok_hex = bool(_HEX64.match(v))
            c.observed[key] = v[:16] + ("..." if ok_hex else "")
            if not ok_hex:
                c.set(FAIL, f"{key} is not a real digest ({v[:24]!r})")
                continue
            disk = self._ckpt_id(rel)
            if disk is None:
                c.set(WARN, f"{rel} not readable now; cannot compare {key}")
            elif disk != v:
                c.set(FAIL, f"{key} in blob != staged file {rel}")
        results = {_tid(k): v for k, v in (blob.get("results", {}) or {}).items()}
        missing = eligible - set(results)
        c.observed["eligible_missing_from_blob"] = len(missing)
        if missing:
            c.set(FAIL, f"{len(missing)} eligible tracklets absent from the blob")
        n_used0 = sum(1 for r in results.values() if int(r.get("n_used", 0)) == 0)
        minus1 = sum(1 for r in results.values() if str(r.get("number")) == "-1")
        c.observed.update({"blob_tracklets": len(results), "blob_minus1": minus1,
                           "blob_n_used_zero": n_used0})
        # detection columns agree with the blob
        disagree = 0
        for tid, grp in tracked.groupby("track_id"):
            r = results.get(_tid(tid))
            if r is None:
                continue
            vals = {str(int(float(v))) for v in grp[jn_col]
                    if not _is_nan(v)}
            want = str(r.get("number"))
            want = str(int(want)) if want.isdigit() else None
            if want is None and vals:
                disagree += 1
            elif want is not None and vals != {want}:
                disagree += 1
        c.observed["tracks_disagreeing_with_blob"] = disagree
        if disagree:
            c.set(FAIL, f"{disagree} tracks whose column differs from the blob")

        # provisioning provenance (written once by fetch_weights / stage_weights)
        fp = self.models_dir / "fetch_weights_provenance.json"
        if not fp.is_file():
            c.set(WARN, "fetch_weights_provenance.json absent")
        else:
            try:
                recs = {r["key"]: r for r in json.loads(fp.read_text())["weights"]}
                s = recs.get("satrn")
                if s is None:
                    c.set(FAIL, "SATRN not in fetch_weights provenance")
                else:
                    c.observed["satrn_provenance"] = {
                        "sha256_matches": s.get("sha256_matches"),
                        "container": s.get("container"),
                        "config_from_checkpoint": bool(s.get("config_from_checkpoint"))}
                    if s.get("sha256_matches") is False:
                        c.set(WARN, "SATRN download hash differs from the recorded prefix")
                    if s.get("container") == "archive":
                        c.set(FAIL, "SATRN download is a file archive, not a checkpoint")
                    if not s.get("config_from_checkpoint"):
                        c.set(WARN, "no config recovered from the SATRN checkpoint")
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
                c.set(WARN, f"fetch_weights_provenance.json unreadable: {e}")
        wp = self.models_dir / "weights_provenance.json"
        if not wp.is_file():
            c.set(WARN, "weights_provenance.json (PARSeq) absent")
        else:
            try:
                prov = json.loads(wp.read_text())
                c.observed["parseq_matches_upstream"] = prov.get("matches_upstream")
                if prov.get("matches_upstream") is not True:
                    c.set(FAIL, "PARSeq checkpoint does not match upstream")
            except (OSError, json.JSONDecodeError) as e:
                c.set(WARN, f"weights_provenance.json unreadable: {e}")
        return c

    def _check_traj_refine(self, seq, det, tracked, tracked_prerefine):
        """Inputs, internals and outputs of the traj_refine stage.

        Snapshots present; sidecar present with the settings and embedder digest
        that ran equal to the configured ones and to tracklet_split's checkpoint pin;
        tracked rows out equal tracked rows in minus the stage-3b unassigned count
        (the only way this stage drops a row); tracklets
        after == tracklets before - merges; every accepted merge distance <= tau;
        no (image_id, track_id) collision; number/team/role constant per final
        track; counts equal the sidecar's; disabled => everything untouched."""
        c = Check("traj_refine (label-aware trajectory refinement)",
                  "track_id_prerefine + jersey snapshots on every row; sidecar with the "
                  "settings and checkpoint that ran equal to the configured ones and to "
                  "tracklet_split's pin; tracked rows out == in - unassigned; tracklets_out == "
                  "tracklets_in - merges; merge distances <= tau; one detection per frame "
                  "per trajectory; number/team/role constant per final track; "
                  "disabled => track_id and jersey columns untouched")
        exp = self.expected_traj_refine
        c.observed["expected"] = dict(exp)
        for col in ("track_id_prerefine", "jersey_number_detection_prerefine",
                    "jersey_number_confidence_prerefine"):
            if col not in det.columns:
                c.observed[col] = "column missing"
                c.set(FAIL, f"{col} column missing")
        if c.verdict == FAIL:
            return c
        data = self._read_sidecar(self.traj_refine_sidecar_dir, seq)
        c.observed["sidecar"] = (str(self.traj_refine_sidecar_dir / f"{seq}.json")
                                if self.traj_refine_sidecar_dir else None)
        if data is None:
            return c.set(FAIL, "no traj_refine sidecar for this sequence (stage did not "
                               "run, audit_dir unset, or unwritable)")

        # --- settings and checkpoint that ran
        st = data.get("settings") or {}
        emb = data.get("embedder") or {}
        c.observed["ran"] = dict(st)
        c.observed["ran_embedder_sha256"] = emb.get("sha256") if emb else None
        want_enabled = exp.get("enabled")
        ran_enabled = st.get("enabled")
        if want_enabled is None or ran_enabled is None:
            c.set(FAIL, "enabled not declared (config) or not recorded (sidecar)")
        elif bool(want_enabled) != bool(ran_enabled):
            c.set(FAIL, f"enabled that ran ({ran_enabled}) != configured ({want_enabled})")
        for key in ("tau", "edge_margin"):
            want, got = exp.get(key), st.get(key)
            if want is None or got is None:
                c.set(FAIL, f"{key} not declared (config) or not recorded (sidecar)")
            elif abs(float(got) - float(want)) > 1e-9:
                c.set(FAIL, f"{key} that ran ({got}) != configured ({want})")
        want, got = exp.get("use_reenter"), st.get("use_reenter")
        if want is not None and got is not None and bool(want) != bool(got):
            c.set(FAIL, f"use_reenter that ran ({got}) != configured ({want})")
        want_roles = exp.get("roles")
        got_roles = st.get("roles")
        if want_roles is not None and got_roles is not None \
                and sorted(map(str, want_roles)) != sorted(map(str, got_roles)):
            c.set(FAIL, f"roles that ran ({got_roles}) != configured ({want_roles})")

        enabled = bool(ran_enabled)
        tid_now, tid_pre = det["track_id"], det["track_id_prerefine"]
        if not enabled or not data.get("ran", False):
            changed = int((~((tid_now == tid_pre) | (tid_now.isna() & tid_pre.isna()))).sum())
            c.observed["rows_with_changed_id"] = changed
            if changed:
                c.set(FAIL, f"stage disabled or not run but track_id differs from "
                            f"track_id_prerefine on {changed} rows")
            if "jersey_number_detection" in det.columns:
                a = det["jersey_number_detection"].map(lambda v: None if _is_nan(v) else str(v))
                b = det["jersey_number_detection_prerefine"].map(
                    lambda v: None if _is_nan(v) else str(v))
                n_diff = int((a != b).sum())
                if n_diff:
                    c.set(FAIL, f"stage disabled or not run but jersey_number_detection "
                                f"differs from its snapshot on {n_diff} rows")
            if not enabled:
                return c.set(INFO, "stage disabled by configuration")
            return c.set(FAIL, "the stage recorded that it did not run on this sequence")

        # --- checkpoint pin: equal to its own config AND to tracklet_split's
        want_sha = exp.get("ain_sha256")
        want_sha_sm = exp.get("ain_sha256_tracklet_split")
        if want_sha in (None, "", "None"):
            c.set(FAIL, "ain_sha256 not pinned in the traj_refine config")
        elif want_sha_sm not in (None, "", "None") and str(want_sha) != str(want_sha_sm):
            c.set(FAIL, "traj_refine and tracklet_split configs pin different weights")
        if not emb or not emb.get("sha256"):
            c.set(FAIL, "sidecar records no embedder digest")
        elif want_sha not in (None, "", "None") and str(emb.get("sha256")) != str(want_sha):
            c.set(FAIL, f"embedder sha256 that ran ({emb.get('sha256')}) "
                        f"!= configured ({want_sha})")

        # --- row accounting: out = in - stage-3b unassigned; no row may GAIN an id
        outp = data.get("outputs") or {}
        inp = data.get("inputs") or {}
        n_pre_rows = int(tid_pre.notna().sum())
        n_now_rows = int(tid_now.notna().sum())
        n_lost = int((tid_pre.notna() & tid_now.isna()).sum())
        n_gained = int((tid_pre.isna() & tid_now.notna()).sum())
        n_unassigned = int(outp.get("rows_unassigned") or 0)
        c.observed.update(tracked_rows_before=n_pre_rows, tracked_rows_after=n_now_rows,
                          rows_losing_id=n_lost, rows_gaining_id=n_gained,
                          rows_unassigned_sidecar=n_unassigned)
        if n_gained:
            c.set(FAIL, f"{n_gained} row(s) gained a track_id in the refine stage")
        if n_lost != n_unassigned:
            c.set(FAIL, f"{n_lost} row(s) lost their track_id but the sidecar reports "
                        f"{n_unassigned} stage-3b unassigned")
        if n_now_rows != n_pre_rows - n_unassigned:
            c.set(FAIL, f"tracked rows after ({n_now_rows}) != before ({n_pre_rows}) "
                        f"- unassigned ({n_unassigned})")

        # --- merge arithmetic against the sidecar
        n_trk_pre = int(tracked_prerefine["track_id"].nunique()) if len(tracked_prerefine) else 0
        n_trk_now = int(tracked["track_id"].nunique()) if len(tracked) else 0
        n_merges = int(outp.get("merges") or 0)
        c.observed.update(tracklets_before=n_trk_pre, tracklets_after=n_trk_now,
                          merges=n_merges, conflicts=outp.get("conflicts"),
                          rejected_2a=outp.get("rejected_2a"),
                          no_centroid=len(data.get("no_centroid") or []),
                          out_of_scope=data.get("out_of_scope"))
        if int(inp.get("tracklets") or 0) != n_trk_pre:
            c.set(FAIL, f"sidecar received {inp.get('tracklets')} tracklets, the snapshot "
                        f"holds {n_trk_pre}")
        if int(outp.get("tracklets") or 0) != n_trk_now:
            c.set(FAIL, f"sidecar reports {outp.get('tracklets')} tracklets, the state "
                        f"holds {n_trk_now}")
        if n_trk_now != n_trk_pre - n_merges:
            c.set(FAIL, f"tracklets after ({n_trk_now}) != before ({n_trk_pre}) - "
                        f"merges ({n_merges})")
        tau = exp.get("tau")
        dists = [m.get("distance") for m in (data.get("merge_log") or [])
                 if m.get("distance") is not None]
        if len(dists) != n_merges:
            c.set(FAIL, f"{len(dists)} merge distances recorded for {n_merges} merges")
        if dists:
            c.observed["merge_distance_max"] = round(float(max(dists)), 4)
            if tau is not None and max(dists) > float(tau) + 1e-6:
                c.set(FAIL, f"a merge was accepted at distance {max(dists):.4f} > tau {tau}")
        if int(outp.get("frame_collisions") or 0) or int(outp.get("clusters_incoherent") or 0):
            c.set(FAIL, f"the stage reported {outp.get('frame_collisions')} frame "
                        f"collision(s) and {outp.get('clusters_incoherent')} incoherent "
                        f"cluster(s) in its own output")

        # --- outputs, recomputed from the detections
        if len(tracked):
            coll = int(tracked.duplicated(subset=["image_id", "track_id"]).sum())
            c.observed["frame_collisions_recomputed"] = coll
            if coll:
                c.set(FAIL, f"{coll} (image_id, track_id) collision(s) after refinement")
            for col in ("jersey_number_detection", "team", "role"):
                bad = self._per_track_constant(tracked, col)
                if bad is None:
                    c.set(FAIL, f"{col} column missing")
                elif bad:
                    c.set(FAIL, f"{col} varies within {len(bad)} refined tracks")
        return c

    def _check_tracklet_agg(self, tracked):
        c = Check("tracklet_agg", "jersey_number constant per track and equal to the "
                                  "per-track detection")
        for col in ("jersey_number",):
            if col not in tracked.columns:
                c.observed[col] = "column missing"
                c.set(FAIL, f"{col} column missing")
                continue
            bad = self._per_track_constant(tracked, col)
            c.observed[f"{col}_inconsistent_tracks"] = len(bad)
            if bad:
                c.set(FAIL, f"{col} varies within {len(bad)} tracks")
        if {"jersey_number", "jersey_number_detection"} <= set(tracked.columns):
            mism = 0
            for tid, grp in tracked.groupby("track_id"):
                a = {str(int(float(v))) for v in grp["jersey_number_detection"] if not _is_nan(v)}
                b = {str(int(float(v))) for v in grp["jersey_number"] if not _is_nan(v)}
                if a != b:
                    mism += 1
            c.observed["tracks_jn_vs_detection_mismatch"] = mism
            if mism:
                c.set(FAIL, f"{mism} tracks where jersey_number != detection")
        return c

    def _read_sidecar(self, directory, seq):
        if directory is None or not directory.is_dir():
            return None
        p = directory / f"{seq}.json"
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def _check_team_embed(self, seq, tracked):
        c = Check("team_embed (osnet_team)",
                  "team_embedding on the sampled crops of EVERY tracklet; the checkpoint "
                  "that ran equals the configured one; no empty/zero embeddings")
        if "team_embedding" not in tracked.columns:
            c.observed["team_embedding"] = "column missing"
            return c.set(FAIL, "team_embedding column missing")
        n_tr = tracked["track_id"].nunique()
        covered = 0
        for tid, grp in tracked.groupby("track_id"):
            if any(isinstance(e, np.ndarray) and e.size for e in grp["team_embedding"]):
                covered += 1
        c.observed.update({"tracklets": int(n_tr), "tracklets_with_embedding": int(covered)})
        if n_tr and covered == 0:
            c.set(FAIL, "no tracklet has a team embedding")
        elif _share(n_tr - covered, n_tr) > self.thr["embed_missing_warn"]:
            c.set(FAIL, f"{n_tr - covered} tracklet(s) without any team embedding")
        elif n_tr - covered:
            c.set(WARN, f"{n_tr - covered} tracklet(s) without any team embedding")
        data = self._read_sidecar(self.team_embed_sidecar_dir, seq)
        c.observed["sidecar"] = str(self.team_embed_sidecar_dir / f"{seq}.json") if self.team_embed_sidecar_dir else None
        if data is None:
            return c.set(FAIL, "no team_embed sidecar for this sequence (stage did not run, "
                               "audit_dir unset, or unwritable)")
        emb = data.get("embedder") or {}
        c.observed.update({"ran_sha256": emb.get("sha256"), "ran_source": emb.get("source"),
                           "ran_input_hw": emb.get("input_hw"), "ran_dim": emb.get("embedding_dim"),
                           "crops_sampled": data.get("crops_sampled"), "crops_embedded": data.get("crops_embedded"),
                           "crops_empty": data.get("crops_empty"),
                           "tracklets_off_grid": data.get("tracklets_off_grid"),
                           "nonfinite_or_zero": data.get("embeddings_nonfinite_or_zero")})
        want = self.expected_team_embed.get("sha256")
        if want not in (None, "", "None"):
            if str(emb.get("sha256")) != str(want):
                c.set(FAIL, f"embedder sha256 that ran ({emb.get('sha256')}) != configured ({want})")
        else:
            c.set(INFO, "team_sha256 not pinned in the config; digest recorded only")
        if not emb.get("sha256"):
            c.set(FAIL, "sidecar records no embedder digest")
        if data.get("tracklets") and data.get("crops_embedded", 0) == 0:
            c.set(FAIL, "stage ran but embedded no crop")
        if data.get("embeddings_nonfinite_or_zero"):
            c.set(FAIL, f"{data['embeddings_nonfinite_or_zero']} non-finite or all-zero embeddings")
        n_off = int(data.get("tracklets_off_grid") or 0)
        if data.get("tracklets") and _share(n_off, data["tracklets"]) > self.thr["off_grid_warn"]:
            c.set(WARN, f"{n_off} tracklet(s) had no frame on the stride grid (all rows used)")
        for key in ("pos_stride", "crops_per_track"):
            want = self.expected_team_embed.get(key)
            got = data.get("stride" if key == "pos_stride" else key)
            if want is not None and got is not None and int(want) != int(got):
                c.set(FAIL, f"{key} that ran ({got}) != configured ({want})")
        return c

    def _check_role_team(self, seq, tracked):
        c = Check("role_team", "every tracked row has a role in {player, goalkeeper, referee}; players "
                               "and goalkeepers have team in {left, right}, referees none; role and "
                               "team constant per track; both teams present; parameters that ran "
                               "equal the configured ones; per-clip caps respected; audited on the "
                               "pre-refine role/team snapshots when traj_refine wrote them")
        for col in ("role", "team", "team_cluster"):
            if col not in tracked.columns:
                c.observed[col] = "column missing"
                c.set(FAIL, f"{col} column missing")
        if c.verdict == FAIL:
            return c
        # Audit role_team's OWN output. traj_refine later rewrites team/role/
        # team_cluster on merged clusters and on rows stage 3b moves between
        # trajectories, so once its snapshots exist the live columns grouped by
        # the pre-refine ids no longer equal what this stage produced (run-7
        # false FAIL: "team varies within 8 tracks" from 3b-adopted rows).
        # Read the snapshots instead; copy first -- the caller shares this
        # frame with the team_embed and jersey checks.
        snap = ("role_prerefine", "team_prerefine", "team_cluster_prerefine")
        if all(s in tracked.columns for s in snap):
            tracked = tracked.copy()
            tracked["role"] = tracked["role_prerefine"]
            tracked["team"] = tracked["team_prerefine"]
            tracked["team_cluster"] = tracked["team_cluster_prerefine"]
            c.observed["columns_audited"] = "prerefine snapshots"
        else:
            c.observed["columns_audited"] = "live (no role/team snapshots in the state)"
        role = tracked["role"]
        bad_role = int((~role.isin(["player", "goalkeeper", "referee"])).sum())
        c.observed["rows_without_valid_role"] = bad_role
        if bad_role:
            c.set(FAIL, f"{bad_role} tracked rows without a valid role")
        pg = tracked[role.isin(["player", "goalkeeper"])]
        has = pg["team"].isin(["left", "right"])
        c.observed.update({"player_gk_rows": int(len(pg)), "rows_missing_team": int((~has).sum()),
                           "left_tracklets": int(pg[pg["team"] == "left"]["track_id"].nunique()),
                           "right_tracklets": int(pg[pg["team"] == "right"]["track_id"].nunique()),
                           "referee_tracklets": int(tracked[role == "referee"]["track_id"].nunique()),
                           "goalkeeper_tracklets": int(tracked[role == "goalkeeper"]["track_id"].nunique())})
        if len(pg) and int((~has).sum()):
            c.set(FAIL, f"{int((~has).sum())} player/GK rows without team")
        ref_team = int(tracked[(role == "referee") & tracked["team"].isin(["left", "right"])].shape[0])
        if ref_team:
            c.set(FAIL, f"{ref_team} referee rows carry a team")
        for col in ("role", "team"):
            bad = self._per_track_constant(tracked, col)
            c.observed[f"{col}_inconsistent_tracks"] = len(bad)
            if bad:
                c.set(FAIL, f"{col} varies within {len(bad)} tracks")
        if len(pg) and (c.observed["left_tracklets"] == 0 or c.observed["right_tracklets"] == 0):
            c.set(WARN, "only one team present")
        data = self._read_sidecar(self.role_team_sidecar_dir, seq)
        c.observed["sidecar"] = str(self.role_team_sidecar_dir / f"{seq}.json") if self.role_team_sidecar_dir else None
        if data is None:
            return c.set(FAIL, "no role_team sidecar for this sequence (stage did not run, "
                               "audit_dir unset, or unwritable)")
        ran = data.get("params") or {}
        want = self.expected_role_team.get("params") or {}
        diff = {}
        for k, v in want.items():
            got = ran.get(k)
            if got is None:
                diff[k] = (got, v)
            elif _is_number(got) and _is_number(v):
                if abs(float(got) - float(v)) > 1e-9:
                    diff[k] = (got, v)
            elif str(got) != str(v):
                diff[k] = (got, v)
        c.observed["params_ran"] = ran
        if diff:
            c.set(FAIL, f"parameters that ran differ from the config: {diff}")
        lvl = data.get("sequence_level") or {}
        c.observed.update({"s_ok": lvl.get("s_ok"), "named_left_cluster": lvl.get("named_left_cluster"),
                           "cues": lvl.get("cues"), "dbscan_eps": lvl.get("dbscan_eps"),
                           "tracklets_no_embedding": data.get("tracklets_no_embedding"),
                           "tracklets_off_grid": data.get("tracklets_off_grid")})
        if data.get("tracklets_no_embedding"):
            c.set(FAIL, f"{data['tracklets_no_embedding']} tracklet(s) reached the rules without "
                        f"a team embedding (labelled player, no team)")
        if lvl and lvl.get("s_ok") is False:
            c.set(WARN, "distance rule disabled: degenerate spread of embedding distances")
        max_ref = want.get("max_ref")
        if max_ref is not None and c.observed["referee_tracklets"] > int(max_ref):
            c.set(FAIL, f"{c.observed['referee_tracklets']} referees > max_ref {max_ref}")
        per = data.get("per_tracklet") or []
        reasons = {}
        for r in per:
            reasons[r.get("why")] = reasons.get(r.get("why"), 0) + 1
        c.observed["reasons"] = reasons
        if per and not tracked.empty and len(per) + int(data.get("tracklets_no_embedding") or 0) != tracked["track_id"].nunique():
            c.set(FAIL, f"sidecar covers {len(per)} tracklets, detections hold {tracked['track_id'].nunique()}")
        return c

    def _check_visualization(self, det, tracked):
        c = Check("visualization (radar)", "every tracked row with a pitch position "
                                            "is drawn with a team/referee colour")
        drawable = tracked[tracked["bbox_pitch"].apply(lambda b: isinstance(b, dict))] \
            if "bbox_pitch" in tracked.columns else tracked.iloc[0:0]
        skipped = int(sum(1 for _, r in drawable.iterrows() if radar_color(r) is None))
        c.observed = {"untracked_rows_not_drawn": int(len(det) - len(tracked)),
                      "tracked_rows_with_pitch": int(len(drawable)),
                      "tracked_rows_skipped_no_colour": skipped}
        share = _share(skipped, len(drawable))
        c.observed["skipped_share"] = round(share, 4)
        if share > self.thr["radar_skipped_tracked_warn"]:
            c.set(WARN, f"{share:.1%} tracked rows have no team/referee colour")
        return c

    # ---------------------------------------------------------------- main --
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        seq = self._seq_name(metadatas)
        det = detections
        tracked = det.dropna(subset=["track_id"]) if "track_id" in det.columns else det.iloc[0:0]
        # The ids as the splitter left them: the pitch gate clears track_id on off-pitch
        # tracklets and keeps the original in track_id_pregate, so the tracklet_split
        # check compares its sidecar with that column when the gate stage ran.
        if "track_id_pregate" in det.columns:
            tracked_pregate = det[det["track_id_pregate"].notna()].copy()
            tracked_pregate["track_id"] = tracked_pregate["track_id_pregate"]
        else:
            tracked_pregate = tracked
        # The ids as the pitch gate left them: the traj_refine stage relabels merged
        # trajectories and keeps the incoming id in track_id_prerefine, so the
        # team_embed / role_team / jersey checks (stages that ran BEFORE refine)
        # compare against that column when the refine stage ran.
        if "track_id_prerefine" in det.columns:
            tracked_prerefine = det[det["track_id_prerefine"].notna()].copy()
            tracked_prerefine["track_id"] = tracked_prerefine["track_id_prerefine"]
        else:
            tracked_prerefine = tracked
        checks = [
            self._check_detector(det, metadatas),
            self._check_track(det, tracked),
            self._check_tracker_internals(seq, metadatas),
            self._check_tracklet_split(seq, det, tracked_pregate),
            self._check_crop_filter(det, tracked_pregate),
            self._check_calibration(seq, tracked_pregate),
            self._check_pitch_gate(seq, det),
            self._check_team_embed(seq, tracked_prerefine),
            self._check_role_team(seq, tracked_prerefine),
            self._check_jersey(seq, tracked_prerefine),
            self._check_traj_refine(seq, det, tracked, tracked_prerefine),
            self._check_tracklet_agg(tracked),
            self._check_visualization(det, tracked),
        ]
        report = {"sequence": seq, "detections": int(len(det)),
                  "tracked": int(len(tracked)), "rule": RULE,
                  "checks": [c.to_dict() for c in checks],
                  "summary": {v: sum(1 for c in checks if c.verdict == v)
                              for v in (PASS, WARN, FAIL, INFO)}}
        out = self.out_dir / f"{seq}.json"
        out.write_text(json.dumps(report, indent=2, default=str))

        width = max(len(c.component) for c in checks)
        lines = [f"[audit] {seq}: " + ", ".join(f"{k}={v}" for k, v in report["summary"].items())]
        for c in checks:
            lines.append(f"[audit]   {c.verdict:4} {c.component:<{width}}  {c.note or 'ok'}")
        msg = "\n".join(lines)
        (log.error if report["summary"][FAIL] else
         log.warning if report["summary"][WARN] else log.info)(msg)
        return detections
