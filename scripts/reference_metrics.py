#!/usr/bin/env python
"""Reference performance metrics — computed and saved SEPARATELY from the pipeline run.

Purpose
-------
One coherent, reviewable performance file per experiment, holding exactly the reference
metrics agreed for this project — nothing mixed into the pipeline's own eval outputs:

  tracking     : HOTA, DetA, AssA, MOTA, IDF1, IDSW           (image space, no attributes)
  gsr          : GS-HOTA, GS-DetA, GS-AssA                    (pitch space, roles+teams+jersey)
  jersey_number: det_acc_all, precision, recall, f1, trk_acc_all, trk_acc_numbered
  calibration  : JaC5, JaC10, MRE, MedRE, CR (+ final_score = CR * JaC5)

Rigor rules
-----------
* Tracking + GSR metrics are produced by the SAME evaluator stack the pipeline uses
  (tracklab TrackEvalEvaluator machinery -> sn-trackeval SoccerNetGS dataset), never
  reimplemented. Per the documented equivalence in tracklab's soccernet_gs.yaml:
  EVAL_SPACE='image' with all attributes disabled == standard HOTA/CLEAR/Identity;
  EVAL_SPACE='pitch' with all attributes enabled == the official GS-HOTA.
* Calibration metrics use the OFFICIAL sn-calibration implementation vendored in
  plugins/calibration (get_polylines / evaluate_camera_prediction / mirror_labels /
  scale_points), at the official 960x540 evaluation resolution (the BroadTrack paper's
  protocol), including the left/right mirror disambiguation. Predicted cameras are
  rescaled exactly (principal point + focal scale linearly; normalized-plane distortion
  coefficients are resolution-invariant).
* Jersey metrics are implemented here (no official standalone implementation exists):
  per-frame Hungarian IoU matching (>= 0.5) between predictions and GT restricted to
  roles {player, goalkeeper}; unnumbered is an explicit class encoded as -1.
* Any metric that cannot be computed is reported as null with a "reason" — never a
  silently wrong number.
* Per-stage attribution: DetA isolates the detector, AssA/IDF1/IDSW isolate association
  (tracking + ReID). The GTA-Link / tracker / calibration contributions are measured by
  running this script on states produced by different pipeline configs (e.g. with and
  without gta_link, NBJW vs BroadTrack) and comparing the saved files; pass several
  --state/--label pairs to get a combined comparison table.

Inputs
------
A tracker-state .pklz saved by a FULL pipeline run (all stages, so that track_id, role,
team, jersey_number, bbox_pitch and image 'parameters' are all present), plus the
SoccerNetGS dataset for ground truth (Labels-GameState.json per sequence).

Usage
-----
    python scripts/reference_metrics.py \
        --state states/sn-gamestate-broadtrack.pklz --label broadtrack \
        --state states/sn-gamestate-botsort.pklz    --label botsort_nbjw \
        --dataset-path data/SoccerNetGS --eval-set test \
        --out reference_metrics

Outputs
-------
    reference_metrics/<label>/reference_metrics.json   (all numbers + provenance)
    reference_metrics/summary.md                       (comparison table, regenerated)
"""
import argparse
import datetime
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

TRACKER_NAME = "tracklab"  # fixed by the tracklab evaluator convention
JN_ROLES = ("player", "goalkeeper")  # jersey/team attributes apply to these roles (GS convention)
UNNUMBERED = -1


# ======================================================================================
# State loading
# ======================================================================================
def load_state(pklz_path: Path):
    """Load a TrackerState .pklz -> (detections_pred, image_pred) concatenated over videos.

    File layout (tracklab TrackerState.save): '<video_id>.pkl' + '<video_id>_image.pkl'
    per video, plus an optional 'summary.json'.
    """
    dets, imgs = [], []
    with zipfile.ZipFile(pklz_path) as zf:
        names = zf.namelist()
        for n in names:
            if not n.endswith(".pkl") or n.endswith("_image.pkl"):
                continue
            with zf.open(n) as fp:
                dets.append(pd.read_pickle(fp))
            img_name = n[:-4] + "_image.pkl"
            if img_name in names:
                with zf.open(img_name) as fp:
                    imgs.append(pd.read_pickle(fp))
    if not dets:
        raise SystemExit(f"No '<video_id>.pkl' members found in {pklz_path}")
    detections = pd.concat(dets)
    images = pd.concat(imgs) if imgs else pd.DataFrame()
    return detections, images


# ======================================================================================
# Tracking + GSR metrics (via the pipeline's own evaluator stack)
# ======================================================================================
def run_trackeval_variant(variant_name, eval_space, use_attributes, metrics_names,
                          tracking_dataset, video_metadatas, detections_pred, image_pred,
                          dataset_path, eval_set, work_dir, num_cores=1):
    """Replicates tracklab.wrappers.TrackEvalEvaluator.run() but RETURNS the summary.

    Same primitives, same file formats, same GT source (Labels-GameState.json read
    directly by the sn-trackeval SoccerNetGS dataset with save_gt=False semantics).
    """
    import trackeval

    work_dir = Path(work_dir)
    trackers_folder = work_dir / variant_name / "pred"
    output_folder = work_dir / variant_name / "results"
    output_folder.mkdir(parents=True, exist_ok=True)

    # 1) predictions in the official GSR JSON format (same encoder as the pipeline)
    pred_save_path = trackers_folder / f"SoccerNetGS-{eval_set}" / TRACKER_NAME
    tracking_dataset.save_for_eval(
        detections_pred.copy(), image_pred.copy(), video_metadatas,
        pred_save_path, "bbox_ltwh", True, is_ground_truth=False,
    )

    # 2) trackeval dataset (GT straight from the dataset labels)
    dataset_class = trackeval.datasets.SoccerNetGS
    dataset_config = dataset_class.get_default_dataset_config()
    dataset_config.update({
        "GT_FOLDER": str(dataset_path),
        "GT_LOC_FORMAT": "{gt_folder}/{seq}/Labels-GameState.json",
        "TRACKERS_FOLDER": str(trackers_folder),
        "TRACKER_SUB_FOLDER": "",
        "OUTPUT_FOLDER": str(output_folder),
        "OUTPUT_SUB_FOLDER": "",
        "SPLIT_TO_EVAL": eval_set,
        "SEQ_INFO": video_metadatas.set_index("name")["nframes"].to_dict(),
        "BENCHMARK": "SoccerNetGS",
        "DO_PREPROC": False,
        "PRINT_CONFIG": False,
        "TRACKER_DISPLAY_NAMES": None,
        "EVAL_SPACE": eval_space,                    # 'image' | 'pitch'
        "USE_ROLES": use_attributes,
        "USE_TEAMS": use_attributes,
        "USE_JERSEY_NUMBERS": use_attributes,
        "EVAL_DIST_TOL": 5,                          # official GS-HOTA tolerance (meters)
    })
    dataset = dataset_class(dataset_config)

    metrics_config = {"METRICS": set(metrics_names), "PRINT_CONFIG": False, "THRESHOLD": 0.5}
    metrics_list = [getattr(trackeval.metrics, m)(metrics_config) for m in metrics_names]

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({
        "USE_PARALLEL": num_cores > 1, "NUM_PARALLEL_CORES": max(1, num_cores),
        "BREAK_ON_ERROR": True, "PRINT_RESULTS": False, "PRINT_ONLY_COMBINED": True,
        "PRINT_CONFIG": False, "TIME_PROGRESS": False, "DISPLAY_LESS_PROGRESS": True,
        "OUTPUT_SUMMARY": True, "OUTPUT_EMPTY_CLASSES": False, "OUTPUT_DETAILED": False,
        "PLOT_CURVES": False,
    })
    evaluator = trackeval.Evaluator(eval_config)
    output_res, _ = evaluator.evaluate([dataset], metrics_list, show_progressbar=False)
    return output_res[dataset.get_name()][TRACKER_NAME]["SUMMARIES"]["cls_comb_det_av"]


def _num(summary, family, key):
    try:
        return float(summary[family][key])
    except (KeyError, TypeError, ValueError):
        return None


def tracking_block(summary):
    return {
        "HOTA": _num(summary, "HOTA", "HOTA"),
        "DetA": _num(summary, "HOTA", "DetA"),
        "AssA": _num(summary, "HOTA", "AssA"),
        "MOTA": _num(summary, "CLEAR", "MOTA"),
        "IDF1": _num(summary, "Identity", "IDF1"),
        "IDSW": _num(summary, "CLEAR", "IDSW"),
        "_context": {
            "DetRe": _num(summary, "HOTA", "DetRe"), "DetPr": _num(summary, "HOTA", "DetPr"),
            "AssRe": _num(summary, "HOTA", "AssRe"), "AssPr": _num(summary, "HOTA", "AssPr"),
            "LocA": _num(summary, "HOTA", "LocA"),
        },
        "_config": "EVAL_SPACE=image, USE_ROLES/TEAMS/JERSEY=False "
                   "(== standard HOTA/CLEAR/Identity per tracklab soccernet_gs.yaml)",
    }


def gsr_block(summary):
    return {
        "GS-HOTA": _num(summary, "HOTA", "HOTA"),
        "GS-DetA": _num(summary, "HOTA", "DetA"),
        "GS-AssA": _num(summary, "HOTA", "AssA"),
        "_config": "EVAL_SPACE=pitch, USE_ROLES/TEAMS/JERSEY=True, EVAL_DIST_TOL=5m "
                   "(official GSR challenge GS-HOTA)",
    }


# ======================================================================================
# Jersey-number metrics (per-frame Hungarian IoU matching, unnumbered == -1)
# ======================================================================================
def norm_jn(v):
    """Jersey value -> int, with -1 for unnumbered (None/NaN/empty/non-digit)."""
    if v is None:
        return UNNUMBERED
    if isinstance(v, float) and np.isnan(v):
        return UNNUMBERED
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return int(s) if s.isdigit() else UNNUMBERED


def _iou_matrix(boxes_a, boxes_b):
    """IoU between two (N,4)/(M,4) ltwh arrays."""
    a = np.asarray(boxes_a, dtype=float)
    b = np.asarray(boxes_b, dtype=float)
    ax1, ay1 = a[:, 0:1], a[:, 1:2]
    ax2, ay2 = ax1 + a[:, 2:3], ay1 + a[:, 3:4]
    bx1, by1 = b[:, 0], b[:, 1]
    bx2, by2 = bx1 + b[:, 2], by1 + b[:, 3]
    ix = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    iy = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = ix * iy
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return np.where(union > 0, inter / union, 0.0)


def jersey_block(detections_pred, detections_gt, iou_thresh=0.5):
    from scipy.optimize import linear_sum_assignment

    needed = {"role", "jersey_number", "track_id", "bbox_ltwh", "image_id"}
    if not needed.issubset(detections_pred.columns):
        return {"reason": f"prediction state lacks columns {sorted(needed - set(detections_pred.columns))} "
                          f"- run the FULL pipeline (incl. tracklet_agg) before evaluating"}

    pred = detections_pred[detections_pred["role"].isin(JN_ROLES)].dropna(subset=["track_id", "bbox_ltwh"])
    gt = detections_gt[detections_gt["role"].isin(JN_ROLES)]

    pairs = []  # (pred_jn, gt_jn, pred_track, gt_track)
    n_pred_unmatched = n_gt_unmatched = 0
    for image_id, gt_grp in gt.groupby("image_id"):
        pred_grp = pred[pred["image_id"] == image_id]
        if len(pred_grp) == 0:
            n_gt_unmatched += len(gt_grp)
            continue
        iou = _iou_matrix(np.stack(pred_grp["bbox_ltwh"].apply(np.asarray).values),
                          np.stack(gt_grp["bbox_ltwh"].apply(np.asarray).values))
        ri, ci = linear_sum_assignment(-iou)
        matched_p, matched_g = set(), set()
        for r, c in zip(ri, ci):
            if iou[r, c] >= iou_thresh:
                pairs.append((
                    norm_jn(pred_grp["jersey_number"].iloc[r]),
                    norm_jn(gt_grp["jersey_number"].iloc[c]),
                    pred_grp["track_id"].iloc[r],
                    gt_grp["track_id"].iloc[c],
                ))
                matched_p.add(r)
                matched_g.add(c)
        n_pred_unmatched += len(pred_grp) - len(matched_p)
        n_gt_unmatched += len(gt_grp) - len(matched_g)

    if not pairs:
        return {"reason": "no matched prediction/GT pairs (IoU >= 0.5, roles player/goalkeeper)"}

    pj = np.array([p[0] for p in pairs])
    gj = np.array([p[1] for p in pairs])

    det_acc_all = float(np.mean(pj == gj))
    tp = int(np.sum((gj != UNNUMBERED) & (pj == gj)))
    fp = int(np.sum((pj != UNNUMBERED) & (pj != gj)))
    fn = int(np.sum((gj != UNNUMBERED) & (pj != gj)))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and (precision + recall) else 0.0
          if precision is not None and recall is not None else None)

    # tracklet-level: majority GT track per pred track; majority jersey values per track
    def _mode(values):
        vals, counts = np.unique(np.asarray(list(values)), return_counts=True)
        return vals[np.argmax(counts)]

    gt_track_jn = {tid: _mode([norm_jn(v) for v in grp["jersey_number"]])
                   for tid, grp in gt.groupby("track_id")}
    per_pred_track = {}
    for p_jn, g_jn, p_tid, g_tid in pairs:
        per_pred_track.setdefault(p_tid, []).append((p_jn, g_tid))
    trk_all, trk_num = [], []
    for p_tid, lst in per_pred_track.items():
        pred_trk_jn = _mode([x[0] for x in lst])
        g_tid = _mode([x[1] for x in lst])
        gt_trk_jn = gt_track_jn.get(g_tid, UNNUMBERED)
        trk_all.append(pred_trk_jn == gt_trk_jn)
        if gt_trk_jn != UNNUMBERED:
            trk_num.append(pred_trk_jn == gt_trk_jn)

    return {
        "det_acc_all": det_acc_all,          # incl. unnumbered (-1) as a class
        "precision": precision, "recall": recall, "f1": f1,   # numbered GT only
        "trk_acc_all": float(np.mean(trk_all)) if trk_all else None,
        "trk_acc_numbered": float(np.mean(trk_num)) if trk_num else None,
        "_counts": {
            "matched_pairs": len(pairs), "tp": tp, "fp": fp, "fn": fn,
            "pred_tracklets": len(per_pred_track),
            "gt_numbered_tracklets_matched": len(trk_num),
            "pred_unmatched": n_pred_unmatched, "gt_unmatched": n_gt_unmatched,
        },
        "_config": f"Hungarian IoU matching >= {iou_thresh}, roles {JN_ROLES}, "
                   f"unnumbered encoded as {UNNUMBERED}",
    }


# ======================================================================================
# Calibration metrics (official sn-calibration protocol, 960x540, mirror-aware)
# ======================================================================================
def _rescale_camera_params(params, eval_w, eval_h):
    """Exact rescale of an sn-calibration parameter dict to the evaluation resolution.

    Principal point and focal lengths scale linearly with resolution; the distortion
    coefficients apply on the normalized image plane and are resolution-invariant.
    """
    src_w = 2.0 * params["principal_point"][0]
    src_h = 2.0 * params["principal_point"][1]
    sx, sy = eval_w / src_w, eval_h / src_h
    out = dict(params)
    out["principal_point"] = [eval_w / 2.0, eval_h / 2.0]
    out["x_focal_length"] = params["x_focal_length"] * sx
    out["y_focal_length"] = params["y_focal_length"] * sy
    return out


def _scale_gt_lines(lines, eval_w, eval_h):
    """GT line annotations -> pixels at evaluation resolution (handles normalized input)."""
    max_c = max((max(p["x"], p["y"]) for pts in lines.values() for p in pts), default=0)
    if max_c <= 1.5:  # normalized [0,1] coordinates (sn-calibration convention)
        fx, fy = eval_w, eval_h
    else:             # already pixels: assume HD annotation resolution
        fx, fy = eval_w / 1920.0, eval_h / 1080.0
    return {name: [{"x": p["x"] * fx, "y": p["y"] * fy} for p in pts]
            for name, pts in lines.items()}


def calibration_block(image_pred, image_gt, eval_w=960, eval_h=540):
    from sn_calibration_baseline.evaluate_camera import (
        get_polylines, evaluate_camera_prediction,
    )
    from sn_calibration_baseline.evaluate_extremities import mirror_labels

    if "lines" not in image_gt.columns:
        return {"reason": "dataset ground truth has no 'lines' annotations"}
    if "parameters" not in image_pred.columns:
        return {"reason": "state has no image 'parameters' column - run a config whose "
                          "calibration stage outputs camera parameters"}

    gt_lines_col = image_gt["lines"]
    labeled_ids = [i for i in image_pred.index
                   if i in gt_lines_col.index and isinstance(gt_lines_col.loc[i], dict)
                   and len(gt_lines_col.loc[i]) > 0]
    if not labeled_ids:
        return {"reason": "no images with GT pitch-line annotations in this split/state"}

    jac5, jac10, errors = [], [], []
    n_pred = 0
    for image_id in labeled_ids:
        params = image_pred.loc[image_id, "parameters"]
        if not isinstance(params, dict) or not params:
            continue  # uncalibrated frame -> counted through CR only (official protocol)
        n_pred += 1
        gt = _scale_gt_lines(gt_lines_col.loc[image_id], eval_w, eval_h)
        proj = get_polylines(_rescale_camera_params(params, eval_w, eval_h),
                             eval_w, eval_h, sampling_factor=0.9)
        frame = {}
        for t in (5, 10):
            best = None
            for gt_variant in (gt, mirror_labels(dict(gt))):
                conf, _, errs = evaluate_camera_prediction(proj, gt_variant, t)
                acc = conf[0, 0] / conf.sum() if conf.sum() > 0 else 0.0
                if best is None or acc > best[0]:
                    best = (acc, errs)
            frame[t] = best
        jac5.append(frame[5][0])
        jac10.append(frame[10][0])
        for errs in frame[5][1].values():  # reprojection distances (threshold-independent)
            errors.extend(errs)

    cr = n_pred / len(labeled_ids)
    if n_pred == 0:
        return {"JaC5": None, "JaC10": None, "MRE": None, "MedRE": None, "CR": 0.0,
                "reason": "no frame has predicted camera parameters"}
    return {
        "JaC5": float(np.mean(jac5)),
        "JaC10": float(np.mean(jac10)),
        "MRE": float(np.mean(errors)) if errors else None,
        "MedRE": float(np.median(errors)) if errors else None,
        "CR": float(cr),
        "final_score": float(cr * np.mean(jac5)),
        "_counts": {"labeled_frames": len(labeled_ids), "predicted_frames": n_pred,
                    "reprojection_points": len(errors)},
        "_config": f"official sn-calibration protocol at {eval_w}x{eval_h}, "
                   f"sampling_factor=0.9, mirror-aware, thresholds 5/10 px "
                   f"(BroadTrack paper protocol)",
    }


# ======================================================================================
# Summary table
# ======================================================================================
ROWS = [
    ("tracking", "HOTA"), ("tracking", "DetA"), ("tracking", "AssA"),
    ("tracking", "MOTA"), ("tracking", "IDF1"), ("tracking", "IDSW"),
    ("gsr", "GS-HOTA"), ("gsr", "GS-DetA"), ("gsr", "GS-AssA"),
    ("jersey_number", "det_acc_all"), ("jersey_number", "f1"),
    ("jersey_number", "trk_acc_all"), ("jersey_number", "trk_acc_numbered"),
    ("calibration", "JaC5"), ("calibration", "JaC10"),
    ("calibration", "MRE"), ("calibration", "MedRE"), ("calibration", "CR"),
]


def _fmt_cell(v) -> str:
    """Summary-table cell: integral -> integer, otherwise 4 decimals, None -> '-'.

    Decided on the VALUE, not on the metric name. TrackEval returns IDSW as a
    float (65.0), so selecting the integer format by `metric == "IDSW"` and then
    applying `:d` raised "Unknown format code 'd' for object of type 'float'" and
    took down the whole summary table.
    """
    if v is None:
        return "-"
    f = float(v)
    return str(int(f)) if f.is_integer() else f"{f:.4f}"


def write_summary(out_dir: Path):
    results = {}
    for f in sorted(out_dir.glob("*/reference_metrics.json")):
        results[f.parent.name] = json.loads(f.read_text())
    if not results:
        return
    labels = list(results)
    lines = ["# Reference metrics summary", "",
             f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}", "",
             "| block | metric | " + " | ".join(labels) + " |",
             "|---|---|" + "---|" * len(labels)]
    for block, metric in ROWS:
        vals = [_fmt_cell(results[lab].get(block, {}).get(metric)) for lab in labels]
        lines.append(f"| {block} | {metric} | " + " | ".join(vals) + " |")
    lines += ["", "Full details, counts and configuration in each "
              "`<label>/reference_metrics.json`."]
    (out_dir / "summary.md").write_text("\n".join(lines))


# ======================================================================================
# Main
# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", action="append", required=True,
                    help="tracker state .pklz (repeatable; pair with --label)")
    ap.add_argument("--label", action="append", default=None,
                    help="label for the matching --state (repeatable)")
    ap.add_argument("--dataset-path", required=True, help="path to SoccerNetGS")
    ap.add_argument("--eval-set", default="test", choices=["train", "valid", "test", "challenge"])
    ap.add_argument("--out", default="reference_metrics")
    ap.add_argument("--num-cores", type=int, default=1)
    ap.add_argument("--eval-width", type=int, default=960)
    ap.add_argument("--eval-height", type=int, default=540)
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["tracking", "gsr", "jersey_number", "calibration"],
                    help="metric blocks to skip")
    args = ap.parse_args()

    labels = args.label or []
    if len(labels) < len(args.state):
        labels += [Path(s).stem for s in args.state[len(labels):]]

    from tracklab.wrappers import SoccerNetGameState
    tracking_dataset = SoccerNetGameState(dataset_path=args.dataset_path, nvid=-1, vids_dict={})
    if args.eval_set not in tracking_dataset.sets:
        raise SystemExit(f"split '{args.eval_set}' not found under {args.dataset_path}")
    tracking_set = tracking_dataset.sets[args.eval_set]

    out_dir = Path(args.out)
    for state_path, label in zip(args.state, labels):
        state_path = Path(state_path)
        print(f"\n=== [{label}] {state_path} ===")
        detections_pred, image_pred = load_state(state_path)
        state_videos = sorted(set(image_pred["video_id"]))
        video_md = tracking_set.video_metadatas[
            tracking_set.video_metadatas.index.isin(state_videos)]
        if len(video_md) == 0:
            print(f"  !! none of the state's videos {state_videos[:5]}... belong to "
                  f"split '{args.eval_set}' - wrong split or dataset path?")
            continue
        det_gt = tracking_set.detections_gt[
            tracking_set.detections_gt["video_id"].isin(video_md.index)]
        img_gt = tracking_set.image_gt[tracking_set.image_gt["video_id"].isin(video_md.index)]

        report = {"meta": {
            "label": label, "state": str(state_path), "eval_set": args.eval_set,
            "videos": video_md["name"].tolist(), "n_videos": len(video_md),
            "n_pred_detections": int(len(detections_pred)),
            "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        }}
        work = out_dir / label
        work.mkdir(parents=True, exist_ok=True)

        if "tracking" not in args.skip:
            print("  [tracking] HOTA/CLEAR/Identity in image space (attributes off)...")
            s = run_trackeval_variant("tracking", "image", False,
                                      ["HOTA", "CLEAR", "Identity"],
                                      tracking_dataset, video_md, detections_pred,
                                      image_pred, args.dataset_path, args.eval_set,
                                      work / "trackeval", args.num_cores)
            report["tracking"] = tracking_block(s)
        if "gsr" not in args.skip:
            print("  [gsr] GS-HOTA in pitch space (attributes on)...")
            s = run_trackeval_variant("gsr", "pitch", True, ["HOTA"],
                                      tracking_dataset, video_md, detections_pred,
                                      image_pred, args.dataset_path, args.eval_set,
                                      work / "trackeval", args.num_cores)
            report["gsr"] = gsr_block(s)
        if "jersey_number" not in args.skip:
            print("  [jersey_number] IoU-matched jersey metrics...")
            report["jersey_number"] = jersey_block(detections_pred, det_gt)
        if "calibration" not in args.skip:
            print("  [calibration] official JaC/MRE/CR protocol...")
            report["calibration"] = calibration_block(
                image_pred, img_gt, args.eval_width, args.eval_height)

        out_file = work / "reference_metrics.json"
        out_file.write_text(json.dumps(report, indent=2, default=float))
        print(f"  -> {out_file}")

    write_summary(out_dir)
    print(f"\nSummary table -> {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
