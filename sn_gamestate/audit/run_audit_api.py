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
        self.models_dir = Path(str(cfg.models_dir))
        self.parseq_ckpt = str(getattr(cfg, "parseq_ckpt", "parseq_gsr_ft_s1.ckpt"))
        self.satrn_ckpt = str(getattr(cfg, "satrn_ckpt",
                                      "recog2/best_recog_word_acc_epoch_10.pth"))
        self.jn_roles = set(getattr(cfg, "jn_roles", ["player", "goalkeeper"]))
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
        }
        # Tracker diagnostics sidecars (written by sn_gamestate.track.bot_sort) and
        # the declared tracker/embedder configuration to hold the run against. The
        # expected values arrive resolved by OmegaConf interpolation from the track
        # and gta_link module configs (see modules/audit/run_audit.yaml), so a
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
        share = _share(int((single & det["track_id"].notna()).sum()), len(tracked)) if "track_id" in det.columns else 0.0
        c.observed["single_share_tracked"] = round(share, 4)
        if len(tracked) and share < self.thr["single_share_warn"]:
            c.set(WARN, f"only {share:.1%} of tracked detections labelled single")
        # tracked-only rule: every veto must come from a box that was tracked when the
        # filter ran. GTA-Link's collision guard can NaN a track_id afterwards; count it.
        mode = str(self.expected_crop_filter.get("contam_mode", "tracked"))
        trig = det["crop_trigger"]
        fired = trig.notna()
        c.observed["multi_labels"] = int((~single).sum())
        c.observed["multi_without_trigger"] = int(((~single) & ~fired).sum())
        if int(((~single) & ~fired).sum()):
            c.set(FAIL, "multi labels with no recorded trigger box")
        if mode == "tracked" and fired.any():
            tid_now = det["track_id"]
            trig_idx = trig[fired].astype(int)
            valid = trig_idx.isin(det.index)
            if not valid.all():
                c.set(FAIL, f"{int((~valid).sum())} trigger indices not in the frame")
            now_untracked = int(tid_now.reindex(trig_idx[valid]).isna().sum())
            c.observed["vetoes_by_box_now_untracked"] = now_untracked
            if now_untracked:
                c.set(INFO, f"{now_untracked} vetoes came from boxes untracked after GTA-Link")
        return c

    def _check_track(self, det, tracked):
        c = Check("track + gta_link", "most detections carry a track_id; "
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
            "track and gta_link configs pin the same OSNet-AIN weights (both stages "
            "build the same embedder module, so the arithmetic is identical by "
            "construction); a diagnostics sidecar covers every frame; the settings "
            "and checkpoint digest that RAN equal the ones the configs declare; "
            "camera motion mostly non-identity; no clipped crops, zero embeddings or "
            "dropped rows")
        exp = self.expected_tracker

        # Config-level identity between the two embedding stages first: each stage
        # sha-verifies its own load at runtime, so equal pins guarantee equal weights
        # even before the sidecar is opened.
        for a, b, what in (("ain_sha256_track", "ain_sha256_gta", "ain_sha256"),
                           ("ain_file_track", "ain_file_gta", "ain_file"),
                           ("ain_revision_track", "ain_revision_gta", "ain_revision")):
            va, vb = exp.get(a), exp.get(b)
            c.observed[what] = {"track": va, "gta_link": vb}
            if va in (None, "", "None") or vb in (None, "", "None"):
                c.set(FAIL, f"{what} not pinned in both track and gta_link configs")
            elif str(va) != str(vb):
                c.set(FAIL, f"track and gta_link disagree on {what}")

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
            except (OSError, json.JSONDecodeError) as e:
                c.set(FAIL, f"camera JSON unreadable: {e}")
        return c

    def _eligible_tids(self, tracked):
        """Tracklets the jersey stage recognises: role (per-track constant, from the
        role_team stage) in jn_roles."""
        if "role" not in tracked.columns:
            return set()
        out = set()
        for tid, grp in tracked.groupby("track_id"):
            mode = grp["role"].mode()
            if len(mode) and mode.iloc[0] in self.jn_roles:
                out.add(_tid(tid))
        return out

    def _check_jersey(self, seq, tracked):
        c = Check("jersey_number_detect",
                  f"per-sequence cache blob with rule={RULE}, real sha256 of both "
                  f"checkpoints, an answer for every eligible tracklet, per-track "
                  f"constant jersey_number_detection consistent with the blob")
        eligible = self._eligible_tids(tracked)
        c.observed["eligible_tracklets"] = len(eligible)

        for col in ("jersey_number_detection", "jersey_number_confidence"):
            if col not in tracked.columns:
                c.observed[col] = "column missing"
                c.set(FAIL, f"{col} column missing")
        if c.verdict == FAIL:
            return c

        # per-track constancy + coverage from the detection columns
        bad = self._per_track_constant(tracked, "jersey_number_detection")
        c.observed["tracks_with_inconsistent_number"] = len(bad)
        if bad:
            c.set(FAIL, f"{len(bad)} tracks with >1 jersey_number_detection value")
        numbered = set()
        for tid, grp in tracked.groupby("track_id"):
            vals = [v for v in grp["jersey_number_detection"] if not _is_nan(v)]
            if vals:
                numbered.add(_tid(tid))
        c.observed["eligible_numbered"] = len(numbered & eligible)
        c.observed["numbered_outside_eligible"] = len(numbered - eligible)
        if numbered - eligible:
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
            vals = {str(int(float(v))) for v in grp["jersey_number_detection"]
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
                               "equal the configured ones; per-clip caps respected")
        for col in ("role", "team", "team_cluster"):
            if col not in tracked.columns:
                c.observed[col] = "column missing"
                c.set(FAIL, f"{col} column missing")
        if c.verdict == FAIL:
            return c
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
        checks = [
            self._check_detector(det, metadatas),
            self._check_track(det, tracked),
            self._check_tracker_internals(seq, metadatas),
            self._check_crop_filter(det, tracked),
            self._check_calibration(seq, tracked),
            self._check_team_embed(seq, tracked),
            self._check_role_team(seq, tracked),
            self._check_jersey(seq, tracked),
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
