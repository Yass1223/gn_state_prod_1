#!/usr/bin/env python
"""Full-pipeline column and stage-I/O audit for the SoccerNet GSR pipeline.

Read-only. Runs after a finished ``tracklab -cn soccernet`` and inspects the saved
tracker state together with the per-stage sidecars, turning every silent
degradation into an explicit PASS/WARN/FAIL verdict. It complements
``scripts/verify_run_integrity.py`` (which reads logs + the audit stage's JSON) by
going down to the level of individual columns and the per-tracklet reasoning of the
team/role and jersey stages.

Four audits, each a section in the report:

1. COLUMNS   -- every column each stage is contracted to write is present, has a
                sane dtype, and is populated on the rows where it must be
                (e.g. bbox_pitch on tracked rows, role on every tracked row, team
                only on players/goalkeepers, jersey_number on eligible tracklets).
2. JERSEY    -- the number is traced through all four hops: manifest input
                (eligible tracklets, crops each) -> worker blob
                (number/confidence/n_used) -> detection columns
                (jersey_number_detection/confidence) -> final jersey_number after
                tracklet_agg voting. Every drop between hops is reported.
3. TEAM/ROLE -- per tracklet from the role_team sidecar: the geometry inputs
                (mx/my/sx/sy/q75, n, single-crop count, embedding present), the
                decision (why), the outlier evidence (out_rule/out_db/z/confirmed),
                and the team with the naming cue that decided the sides. Structural
                invariants enforced (referee has no team; player/GK has one; role
                in {player,goalkeeper,referee}).
4. CROPS     -- a labelled contact sheet of N random tracked crops (mix of
                crop_single True/False) with their single/multi label, rT/rB,
                role, team and jersey, written to <out>/crop_check.png for visual
                verification of the single/multi filter and the assignments.

Usage:
    python scripts/audit_pipeline_columns.py \
        --state state.pklz \
        --dataset-path data/SoccerNetGS --eval-set test \
        --audit-dir audit --role-team-dir audit/role_team \
        --team-embed-dir audit/team_embed --jn-cache-dir jn_cache \
        --out audit_report --n-crops 24

Exit 0 = no FAIL. Non-zero = at least one FAIL (do not trust the run).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"
GREEN, YELLOW, RED, DIM, BOLD, RESET = (
    "\033[0;32m", "\033[0;33m", "\033[0;31m", "\033[2m", "\033[1m", "\033[0m")
VERDICT_COLOR = {PASS: GREEN, WARN: YELLOW, FAIL: RED, INFO: DIM}

JN_ROLES = ("player", "goalkeeper")
VALID_ROLES = ("player", "goalkeeper", "referee")


# --------------------------------------------------------------------------- util

class Check:
    """One audit line: what was expected, what was observed, a verdict."""

    def __init__(self, name, expected):
        self.name = name
        self.expected = expected
        self.observed = {}
        self.verdict = PASS
        self.note = ""

    def set(self, verdict, note=""):
        order = {PASS: 0, INFO: 1, WARN: 2, FAIL: 3}
        if order[verdict] >= order[self.verdict]:
            self.verdict = verdict
            if note:
                self.note = note
        return self

    def to_dict(self):
        return dict(check=self.name, expected=self.expected, verdict=self.verdict,
                    note=self.note, observed=self.observed)


def _is_nan(v):
    if v is None:
        return True
    if isinstance(v, (np.ndarray, list, tuple, dict, str)):
        return False
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _share(a, b):
    return 0.0 if b == 0 else a / b


def load_state(pklz_path):
    """(detections, images) concatenated over videos. Same layout as
    scripts/reference_metrics.load_state: '<vid>.pkl' + '<vid>_image.pkl'."""
    dets, imgs = [], []
    with zipfile.ZipFile(pklz_path) as zf:
        names = zf.namelist()
        for n in names:
            if not n.endswith(".pkl") or n.endswith("_image.pkl"):
                continue
            with zf.open(n) as fp:
                dets.append(pd.read_pickle(fp))
            im = n[:-4] + "_image.pkl"
            if im in names:
                with zf.open(im) as fp:
                    imgs.append(pd.read_pickle(fp))
    if not dets:
        raise SystemExit(f"no '<video_id>.pkl' members in {pklz_path}")
    return pd.concat(dets), (pd.concat(imgs) if imgs else pd.DataFrame())


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------- 1. COLUMNS

# (column, where it must be populated, produced-by). "where" is a predicate name
# resolved below; None means "every row".
def audit_columns(det, img):
    checks = []
    tracked = det["track_id"].notna() if "track_id" in det.columns else pd.Series(False, index=det.index)
    n_tracked = int(tracked.sum())

    def col_check(col, must_on, producer, dtype_hint=None, elem_ok=None):
        c = Check(f"column:{col}", f"written by {producer}; populated on {must_on['desc']}")
        if col not in det.columns:
            return c.set(FAIL, "column missing").to_dict()
        series = det[col]
        mask = must_on["mask"]
        n_expected = int(mask.sum())
        # null count on the rows where it must be populated
        if elem_ok is not None:
            populated = series[mask].map(lambda v: elem_ok(v))
            n_null = int((~populated).sum())
        else:
            n_null = int(series[mask].map(_is_nan).sum())
        c.observed.update(rows_expected=n_expected, rows_null=n_null,
                          dtype=str(series.dtype))
        if n_expected == 0:
            c.set(INFO, "no rows require this column in this run")
        elif n_null == n_expected:
            c.set(FAIL, f"all {n_expected} required rows are null")
        elif n_null:
            c.set(WARN, f"{n_null}/{n_expected} required rows null")
        return c.to_dict()

    everyrow = dict(desc="every detection", mask=pd.Series(True, index=det.index))
    trackedrows = dict(desc="tracked rows", mask=tracked)

    def is_ltwh(v):
        return isinstance(v, (list, tuple, np.ndarray)) and len(np.asarray(v).ravel()) == 4

    def is_pitch(v):
        return isinstance(v, dict) and "x_bottom_middle" in v

    def is_vec(v):
        return isinstance(v, np.ndarray) and v.size > 0 and np.isfinite(v).all()

    # detector / tracker
    checks.append(col_check("bbox_ltwh", everyrow, "bbox_detector", elem_ok=is_ltwh))
    checks.append(col_check("bbox_conf", everyrow, "bbox_detector"))
    # track_id: expected on *some* rows; FAIL only if zero tracked
    c = Check("column:track_id", "written by track+tracklet_split (refined by traj_refine); some detections tracked")
    if "track_id" not in det.columns:
        c.set(FAIL, "column missing")
    else:
        c.observed.update(tracked=n_tracked, total=int(len(det)),
                          tracked_share=round(_share(n_tracked, len(det)), 4))
        if n_tracked == 0:
            c.set(FAIL, "no detection has a track_id")
    checks.append(c.to_dict())
    # crop_filter: every detection
    for col in ("crop_single", "crop_rT", "crop_rB"):
        checks.append(col_check(col, everyrow, "crop_filter"))
    # calibration: bbox_pitch on tracked rows (role/team need it there)
    checks.append(col_check("bbox_pitch", trackedrows, "calibration", elem_ok=is_pitch))
    # pitch_gate: pre-gate id and off-pitch flag on every row; with the gate enabled
    # no off-pitch row may still carry a track_id
    checks.append(col_check("pitch_gate_offpitch", everyrow, "pitch_gate"))
    c = Check("column:track_id_pregate", "written by pitch_gate; the id tracklet_split left, on every pre-gate tracked row")
    if "track_id_pregate" not in det.columns or "pitch_gate_offpitch" not in det.columns:
        c.set(FAIL, "column missing")
    else:
        pre = det["track_id_pregate"].notna()
        off = det["pitch_gate_offpitch"].map(lambda v: bool(v) if not _is_nan(v) else False)
        c.observed.update(tracked_before_gate=int(pre.sum()), tracked_after_gate=n_tracked,
                          off_pitch_rows=int(off.sum()), off_pitch_rows_still_tracked=int((off & tracked).sum()),
                          new_ids_after_gate=int((~pre & tracked).sum()))
        if int((~pre & tracked).sum()):
            c.set(FAIL, "rows tracked now that had no track_id before the gate")
        if int(off.sum()) and int((off & tracked).sum()) == int(off.sum()):
            c.set(INFO, "off-pitch rows kept their track_id (gate disabled)")
        elif int((off & tracked).sum()):
            c.set(FAIL, f"{int((off & tracked).sum())} off-pitch rows still tracked while others were gated")
    checks.append(c.to_dict())
    # image parameters
    c = Check("column:parameters", "written by calibration on image rows")
    if len(img) == 0 or "parameters" not in img.columns:
        c.set(FAIL, "image 'parameters' column missing")
    else:
        n_null = int(img["parameters"].map(_is_nan).sum())
        c.observed.update(image_rows=int(len(img)), null=n_null)
        if n_null == len(img):
            c.set(FAIL, "all image rows lack camera parameters")
        elif n_null:
            c.set(WARN, f"{n_null}/{len(img)} image rows lack parameters")
    checks.append(c.to_dict())
    # team_embed: team_embedding on tracked rows (sampled subset -> WARN not FAIL if partial)
    c = Check("column:team_embedding", "written by team_embed on sampled tracked crops")
    if "team_embedding" not in det.columns:
        c.set(FAIL, "column missing")
    else:
        n_vec = int(det.loc[tracked, "team_embedding"].map(is_vec).sum())
        c.observed.update(tracked=n_tracked, embedded=n_vec)
        if n_tracked and n_vec == 0:
            c.set(FAIL, "no tracked row has a team embedding")
    checks.append(c.to_dict())
    # role_team: role on every tracked row; team only on player/GK; team_cluster on players
    checks.append(col_check("role", trackedrows, "role_team"))
    if "role" in det.columns:
        pg = tracked & det["role"].isin(JN_ROLES)
        checks.append(col_check("team", dict(desc="player/GK tracked rows", mask=pg), "role_team"))
        # referee must NOT carry a team
        c = Check("column:team/referee-empty", "referees carry no team")
        refs = tracked & (det["role"] == "referee")
        n_bad = int(det.loc[refs, "team"].map(lambda v: not _is_nan(v)).sum()) if "team" in det.columns else 0
        c.observed.update(referee_rows=int(refs.sum()), referee_with_team=n_bad)
        if n_bad:
            c.set(FAIL, f"{n_bad} referee rows carry a team")
        checks.append(c.to_dict())
        # role values valid
        c = Check("column:role/values", f"role in {VALID_ROLES}")
        bad = tracked & ~det["role"].isin(VALID_ROLES) & det["role"].map(lambda v: not _is_nan(v))
        c.observed.update(invalid_role_rows=int(bad.sum()))
        if int(bad.sum()):
            c.set(FAIL, f"{int(bad.sum())} tracked rows have a role outside {VALID_ROLES}")
        checks.append(c.to_dict())
    # jersey detection columns (present on every detection; populated on eligible)
    for col in ("jersey_number_detection", "jersey_number_confidence"):
        c = Check(f"column:{col}", "written by jersey_number_detect (None/0.0 default)")
        if col not in det.columns:
            c.set(FAIL, "column missing")
        checks.append(c.to_dict())
    # final voted jersey_number (present on tracked rows; value only where numbered)
    c = Check("column:jersey_number", "written by tracklet_agg (voted from _detection)")
    if "jersey_number" not in det.columns:
        c.set(FAIL, "column missing")
    else:
        n_val = int(det.loc[tracked, "jersey_number"].map(lambda v: not _is_nan(v)).sum())
        c.observed.update(tracked=n_tracked, numbered_rows=n_val)
    checks.append(c.to_dict())
    return checks


# --------------------------------------------------------------- 2. JERSEY I/O

def audit_jersey(det, img, jn_cache_dir):
    """Trace the number through: manifest input -> worker blob -> detection cols -> voted."""
    checks = []
    seqs = sorted(set(img["video_id"])) if "video_id" in img.columns else []
    if "track_id" not in det.columns or "role" not in det.columns:
        c = Check("jersey:preconditions", "track_id and role present")
        return [c.set(FAIL, "need track_id and role columns to audit jersey").to_dict()]

    for seq in seqs:
        seq_img = img[img["video_id"] == seq]
        img_ids = set(seq_img.index)
        seq_det = det[det["image_id"].isin(img_ids)]
        tracked = seq_det[seq_det["track_id"].notna()]
        # eligible = tracklets whose (constant) role is player/GK
        eligible = set()
        for tid, grp in tracked.groupby("track_id"):
            mode = grp["role"].mode()
            if len(mode) and mode.iloc[0] in JN_ROLES:
                eligible.add(str(tid))

        c = Check(f"jersey:{seq}",
                  "input (eligible tracklets, crops) -> blob (number/conf/n_used) "
                  "-> detection cols -> voted jersey_number, no silent drop between hops")
        c.observed["eligible_tracklets"] = len(eligible)

        blobs = sorted(glob.glob(str(Path(jn_cache_dir) / f"{seq}.*.json")), key=os.path.getmtime) \
            if jn_cache_dir else []
        if not blobs:
            c.set(FAIL, "no jersey cache blob (stage did not run or workers failed)")
            checks.append(c.to_dict())
            continue
        blob = read_json(blobs[-1]) or {}
        results = blob.get("results", {}) or {}
        c.observed["cache_blob"] = os.path.basename(blobs[-1])
        c.observed["blob_tracklets"] = len(results)

        # HOP 1->2: every eligible tracklet has a blob entry
        missing_blob = eligible - set(results)
        c.observed["eligible_missing_from_blob"] = len(missing_blob)
        if missing_blob:
            c.set(WARN, f"{len(missing_blob)} eligible tracklets absent from the worker blob")

        # blob stats
        numbered_blob = {t for t, r in results.items()
                         if str(r.get("number", "-1")).isdigit() and str(r.get("number")) != "-1"}
        zero_used = [t for t in numbered_blob if int(results[t].get("n_used", 0)) == 0]
        confs = [float(results[t].get("confidence", 0.0)) for t in numbered_blob]
        c.observed.update(blob_numbered=len(numbered_blob),
                          blob_numbered_with_zero_frames=len(zero_used),
                          blob_conf_min=round(min(confs), 4) if confs else None,
                          blob_conf_mean=round(float(np.mean(confs)), 4) if confs else None)
        if zero_used:
            c.set(FAIL, f"{len(zero_used)} tracklets numbered from 0 frames (n_used==0)")

        # HOP 2->3: blob number present on the detection column for that tracklet
        det_by_tid = {str(t): g for t, g in tracked.groupby("track_id")}
        drop_2_3 = []
        for t in numbered_blob:
            g = det_by_tid.get(t)
            if g is None:
                drop_2_3.append(t)
                continue
            vals = {str(v) for v in g["jersey_number_detection"] if not _is_nan(v)} \
                if "jersey_number_detection" in g.columns else set()
            if not vals:
                drop_2_3.append(t)
        c.observed["blob_numbered_but_detection_empty"] = len(drop_2_3)
        if drop_2_3:
            c.set(FAIL, f"{len(drop_2_3)} tracklets numbered in the blob but empty in "
                        f"jersey_number_detection")

        # HOP 3->4: detection number survives voting into jersey_number
        drop_3_4 = []
        if "jersey_number" in tracked.columns and "jersey_number_detection" in tracked.columns:
            for t, g in det_by_tid.items():
                d = {str(v) for v in g["jersey_number_detection"] if not _is_nan(v)}
                v = {str(v) for v in g["jersey_number"] if not _is_nan(v)}
                if d and not v:
                    drop_3_4.append(t)
        c.observed["detection_set_but_voted_empty"] = len(drop_3_4)
        if drop_3_4:
            c.set(FAIL, f"{len(drop_3_4)} tracklets have a detected number dropped by voting")

        # end-to-end coverage
        voted_numbered = 0
        if "jersey_number" in tracked.columns:
            for t, g in det_by_tid.items():
                if any(not _is_nan(v) for v in g["jersey_number"]):
                    voted_numbered += 1
        c.observed["voted_numbered_tracklets"] = voted_numbered
        c.observed["coverage_eligible"] = round(_share(voted_numbered, len(eligible)), 4)
        if len(eligible) >= 5 and voted_numbered == 0:
            c.set(FAIL, f"0 of {len(eligible)} eligible tracklets ended up numbered")
        checks.append(c.to_dict())
    return checks


# ------------------------------------------------------------- 3. TEAM / ROLE

def audit_team_role(det, role_team_dir, team_embed_dir, img):
    checks = []
    seqs = sorted(set(img["video_id"])) if "video_id" in img.columns else []
    for seq in seqs:
        c = Check(f"team_role:{seq}",
                  "per-tracklet role from geometry+appearance outliers; team from the "
                  "player k-means with a documented naming cue; invariants hold")
        rt = read_json(Path(role_team_dir) / f"{seq}.json") if role_team_dir else None
        if rt is None:
            c.set(FAIL, "no role_team sidecar for this sequence")
            checks.append(c.to_dict())
            continue
        per = rt.get("per_tracklet", []) or []
        lvl = rt.get("sequence_level", {}) or {}
        c.observed["tracklets"] = len(per)
        # reason breakdown = how each role was determined
        reasons = {}
        for t in per:
            reasons[t.get("why")] = reasons.get(t.get("why"), 0) + 1
        c.observed["role_reasons"] = reasons
        # outlier evidence summary
        n_rule = sum(1 for t in per if t.get("out_rule"))
        n_db = sum(1 for t in per if t.get("out_db"))
        n_conf = sum(1 for t in per if t.get("confirmed"))
        c.observed.update(outlier_by_distance_rule=n_rule, outlier_by_dbscan=n_db,
                          outliers_confirmed=n_conf,
                          distance_median=lvl.get("distance_median"),
                          distance_mad=lvl.get("distance_mad"), s_ok=lvl.get("s_ok"),
                          dbscan_eps=lvl.get("dbscan_eps"))
        # team naming: which cue decided the left side
        c.observed.update(named_left_cluster=lvl.get("named_left_cluster"),
                          naming_cues=lvl.get("cues"),
                          counts=dict(player=lvl.get("n_player"), goalkeeper=lvl.get("n_goalkeeper"),
                                      referee=lvl.get("n_referee"),
                                      left=lvl.get("n_left"), right=lvl.get("n_right")))
        if lvl.get("s_ok") is False:
            c.set(WARN, "distance rule disabled (degenerate embedding-distance spread)")
        # invariants against the sidecar itself
        bad_role = [t for t in per if t.get("role") not in VALID_ROLES]
        ref_team = [t for t in per if t.get("role") == "referee" and t.get("team") in ("left", "right")]
        pg_noteam = [t for t in per if t.get("role") in JN_ROLES and t.get("team") not in ("left", "right")]
        c.observed.update(invalid_role=len(bad_role), referee_with_team=len(ref_team),
                          player_gk_without_team=len(pg_noteam))
        if bad_role:
            c.set(FAIL, f"{len(bad_role)} tracklets with an invalid role")
        if ref_team:
            c.set(FAIL, f"{len(ref_team)} referees carry a team")
        if pg_noteam:
            c.set(FAIL, f"{len(pg_noteam)} players/GKs without a team")
        # both teams present?
        if (lvl.get("n_left") or 0) == 0 or (lvl.get("n_right") or 0) == 0:
            c.set(WARN, "only one team present among players")
        # cross-check sidecar vs the actual detection columns for this sequence
        if "video_id" in img.columns and "role" in det.columns:
            img_ids = set(img[img["video_id"] == seq].index)
            td = det[det["image_id"].isin(img_ids) & det["track_id"].notna()]
            col_roles = {}
            for tid, g in td.groupby("track_id"):
                m = g["role"].mode()
                col_roles[str(float(tid))] = m.iloc[0] if len(m) else None
            side_roles = {str(float(t["track_id"])): t["role"] for t in per if t.get("track_id") is not None}
            mism = sum(1 for k in side_roles if k in col_roles and col_roles[k] != side_roles[k])
            c.observed["sidecar_vs_column_role_mismatch"] = mism
            if mism:
                c.set(FAIL, f"{mism} tracklets: sidecar role != detection-column role")
        checks.append(c.to_dict())
    return checks


# ---------------------------------------------------------------- 4. CROP VIZ

def make_crop_sheet(det, img, out_png, n_crops, seed=0):
    """Contact sheet of random tracked crops, labelled single/multi + role/team/JN."""
    info = {"requested": n_crops, "drawn": 0, "path": str(out_png)}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from tracklab.utils.cv2 import cv2_load_image
    except Exception as e:  # noqa: BLE001
        info["error"] = f"matplotlib/cv2 unavailable: {e}"
        return info
    if "track_id" not in det.columns or "bbox_ltwh" not in det.columns:
        info["error"] = "need track_id and bbox_ltwh"
        return info
    tracked = det[det["track_id"].notna()].copy()
    if "crop_single" in tracked.columns:
        singles = tracked[tracked["crop_single"] == True]      # noqa: E712
        multis = tracked[tracked["crop_single"] == False]      # noqa: E712
    else:
        singles, multis = tracked, tracked.iloc[0:0]
    rng = np.random.default_rng(seed)
    half = max(1, n_crops // 2)

    def sample(df, k):
        if len(df) == 0:
            return df
        idx = rng.choice(len(df), size=min(k, len(df)), replace=False)
        return df.iloc[idx]

    picks = pd.concat([sample(singles, half), sample(multis, n_crops - half)])
    if len(picks) == 0:
        info["error"] = "no tracked detections to sample"
        return info
    id2path = {i: str(p) for i, p in img["file_path"].items()} if "file_path" in img.columns else {}
    cols = 6
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.8))
    axes = np.atleast_1d(axes).ravel()
    drawn = 0
    for ax in axes:
        ax.axis("off")
    for ax, (_, r) in zip(axes, picks.iterrows()):
        path = id2path.get(r["image_id"])
        if path is None or not os.path.exists(path):
            continue
        try:
            im = cv2_load_image(path)
        except Exception:  # noqa: BLE001
            continue
        l, t, w, h = [float(x) for x in np.asarray(r["bbox_ltwh"]).ravel()[:4]]
        x1, y1, x2, y2 = max(0, int(l)), max(0, int(t)), int(l + w), int(t + h)
        crop = im[y1:min(im.shape[0], y2), x1:min(im.shape[1], x2)]
        if crop.size == 0:
            continue
        ax.imshow(crop)
        single = r.get("crop_single")
        tag = "single" if single is True else "multi" if single is False else "?"
        rt = r.get("crop_rT"); rb = r.get("crop_rB")
        role = r.get("role"); team = r.get("team"); jn = r.get("jersey_number")
        rt_s = f"{rt:.2f}" if isinstance(rt, (int, float)) and not _is_nan(rt) else "-"
        rb_s = f"{rb:.2f}" if isinstance(rb, (int, float)) and not _is_nan(rb) else "-"
        jn_s = "" if _is_nan(jn) else f" #{jn}"
        color = "green" if single is True else "red" if single is False else "gray"
        ax.set_title(f"{tag} rT{rt_s} rB{rb_s}\n{role or '-'}/{team or '-'}{jn_s}",
                     fontsize=7, color=color)
        drawn += 1
    fig.suptitle("Random tracked crops — single (green) vs multi (red), with role/team/JN",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    info["drawn"] = drawn
    info["singles_available"] = int(len(singles))
    info["multis_available"] = int(len(multis))
    return info


# --------------------------------------------------------------------- report

def print_section(title, checks):
    fails = sum(1 for c in checks if c["verdict"] == FAIL)
    warns = sum(1 for c in checks if c["verdict"] == WARN)
    print(f"\n{BOLD}== {title} =={RESET}  ({fails} FAIL, {warns} WARN)")
    for c in checks:
        col = VERDICT_COLOR[c["verdict"]]
        print(f"  {col}{c['verdict']:4}{RESET} {c['check']}: {c.get('note') or 'ok'}")
    return fails


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", required=True, help="tracker state .pklz")
    ap.add_argument("--dataset-path", default=None, help="(unused; kept for symmetry)")
    ap.add_argument("--eval-set", default="test")
    ap.add_argument("--audit-dir", default="audit")
    ap.add_argument("--role-team-dir", default="audit/role_team")
    ap.add_argument("--team-embed-dir", default="audit/team_embed")
    ap.add_argument("--jn-cache-dir", default="jn_cache")
    ap.add_argument("--out", default="audit_report")
    ap.add_argument("--n-crops", type=int, default=24)
    ap.add_argument("--no-crops", action="store_true", help="skip the crop contact sheet")
    args = ap.parse_args(argv)

    det, img = load_state(args.state)
    print(f"{BOLD}Pipeline column & stage-I/O audit{RESET}")
    print(f"state: {args.state}")
    print(f"detections: {len(det)}  images: {len(img)}  "
          f"sequences: {sorted(set(img['video_id'])) if 'video_id' in img.columns else '?'}")

    report = {"state": str(args.state), "eval_set": args.eval_set}
    total_fail = 0

    report["columns"] = audit_columns(det, img)
    total_fail += print_section("1. COLUMN POPULATION", report["columns"])

    report["jersey"] = audit_jersey(det, img, args.jn_cache_dir)
    total_fail += print_section("2. JERSEY I/O (input -> blob -> detection -> voted)", report["jersey"])

    report["team_role"] = audit_team_role(det, args.role_team_dir, args.team_embed_dir, img)
    total_fail += print_section("3. TEAM / ROLE (determination + invariants)", report["team_role"])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_crops:
        sheet = make_crop_sheet(det, img, out_dir / "crop_check.png", args.n_crops)
        report["crop_sheet"] = sheet
        ok = "error" not in sheet and sheet.get("drawn", 0) > 0
        col = GREEN if ok else YELLOW
        print(f"\n{BOLD}== 4. CROP VISUALIZATION =={RESET}")
        if ok:
            print(f"  {col}PASS{RESET} contact sheet: {sheet['drawn']} crops "
                  f"({sheet.get('singles_available')} single / {sheet.get('multis_available')} multi "
                  f"available) -> {sheet['path']}")
        else:
            print(f"  {YELLOW}WARN{RESET} crop sheet not produced: {sheet.get('error', 'no crops drawn')}")

    (out_dir / "column_audit.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nfull report -> {out_dir / 'column_audit.json'}")
    print(f"{BOLD}RESULT:{RESET} " +
          (f"{GREEN}no FAIL{RESET}" if total_fail == 0 else f"{RED}{total_fail} FAIL{RESET} — do not trust this run"))
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())
