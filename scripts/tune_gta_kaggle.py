# TEMPORARY TUNING TOOL — remove at productionization (see T7).
"""Two-stage tracking/GTA-Link tuning on Kaggle, on YOLO11L detections only.

Why two stages
--------------
Tracker-level settings (the detector confidence floor, tracker thresholds) change the
detections and the association, so they can only be swept by re-running
``bbox_detector -> reid -> track``. Everything downstream of ``track`` is a pure
function of the saved state, so the whole post-processing grid can be swept by
loading one Stage-1 state and re-running only ``gta_link`` (+ ``interpolation``).

    Stage 1  full runs, one per tracker variant           -> states/tune/<variant>.pklz
    Stage 2  per state, one cheap run per grid point      -> gta_optima.json + CSV

Hard constraint: the detector is ALWAYS the YOLO11L SoccerNet fine-tune
(``yolo_ultralytics_snft``, imgsz 1280, iou 0.7, conf floor 0.1). The previous GTA
optima were tuned on ORACLE detections and never validated on YOLO11L boxes; that
is the defect this script exists to fix, so it refuses to run against any other
detector and stamps the detector identity into every output.

Scoring is delegated to ``scripts/reference_metrics.py`` — metrics are never
reimplemented here. Selection is on HOTA, with AssA and IDSW reported alongside;
near-ties (within ``TIE_HOTA``) are broken in favour of fewer id switches.

Cost note: Stage 2 re-extracts an OSNet feature per detection on every grid point,
which dominates its runtime. Size the grid with ``--max-configs`` to fit the
session; whatever is dropped is logged, never silently truncated.

Usage
-----
    # 1. tracker variants (2-4 full runs)
    python scripts/tune_gta_kaggle.py stage1 --nvid 20 --holdout 8

    # 2. post-processing grid over every Stage-1 state, then the fine pass
    python scripts/tune_gta_kaggle.py stage2 --max-configs 120
    python scripts/tune_gta_kaggle.py stage2 --fine --max-configs 60

    # 3. confirm the winner on clips the sweep never saw
    python scripts/tune_gta_kaggle.py holdout
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import itertools
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "sn_gamestate" / "configs"
WORK = REPO / "states" / "tune"
RESULTS = REPO / "tuning_results"

# --------------------------------------------------------------------------- grids
# EDIT DATA, NOT LOGIC. Axes are swept as a cross product; the `split_*` axes are
# only expanded when use_split is True, so turning Split off does not multiply the
# grid by its irrelevant parameters.
STAGE1_VARIANTS = [
    # name              overrides applied to the full run
    # The tracker is boxmot's BoT-SORT · SOF with OSNet-AIN appearance
    # (modules/track/botsort_ain.yaml); it has no masked-SOF variant, so Stage 1
    # is a single baseline unless a floor A/B below is enabled.
    ("baseline",        {}),
    # T5 option (a): lower BOTH confidence floors together so the BYTE low band is
    # non-empty. Documented as an A/B, not a code change. Enable by uncommenting.
    # ("baseline_c005", {"modules.track.cfg.hyperparams.track_low_thresh": "0.05",
    #                    "modules.bbox_detector.cfg.min_confidence": "0.05"}),
]

COARSE_GRID = {
    "connect_mode": ["agglomerative", "iterative"],
    "appearance_thresh": [0.15, 0.25, 0.35],
    "spatial_thresh": [75, 150, 300],
    "min_tracklet_len": [10, 20, 30],
    "use_split": [False, True],
    "interp_n_dti": [None, 25],          # None = interpolation off
}
COARSE_SPLIT_GRID = {                    # expanded only when use_split is True
    "split_eps": [0.4, 0.6],
    "split_min_samples": [4],
    "split_max_k": [3],
    "split_len_thres": [30],
}
# The fine pass re-centres these on the coarse winner (see build_fine_grid).
FINE_STEPS = {
    "appearance_thresh": [-0.05, 0.0, 0.05],
    "spatial_thresh": [0.67, 1.0, 1.5],          # multiplicative
    "min_tracklet_len": [-5, 0, 5],
    "split_eps": [-0.1, 0.0, 0.1],
}

TIE_HOTA = 0.002        # HOTA gap below which IDSW decides
DETECTOR_STEM = "yolo_ultralytics_snft"
DETECTOR_TARGET = "sn_gamestate.bbox_detector.yolo_snft_api.YOLOUltralyticsSNFT"
DETECTOR_EXPECT = {"imgsz": 1280, "iou": 0.7, "min_confidence": 0.1}
HF_REPO = "Ynniss/sn-gamestate-weights"
HF_DETECTOR_FILE = "yolov11_sn_best.pt"


# ============================================================== provenance / guards
def die(msg: str):
    print(f"\n!! {msg}\n", file=sys.stderr)
    raise SystemExit(2)


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                               capture_output=True, text=True, check=True)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception as e:
        return f"unknown ({e})"


def detector_provenance() -> dict:
    """Resolve and verify the detector. Any deviation from YOLO11L is fatal.

    The configs are read as YAML rather than composed through Hydra so this check
    holds even when the pipeline itself cannot be instantiated (no GPU, no weights).
    """
    entry = yaml.safe_load((CONFIGS / "soccernet.yaml").read_text(encoding="utf-8"))
    chosen = None
    for d in entry.get("defaults", []):
        if isinstance(d, dict) and "modules/bbox_detector" in d:
            chosen = d["modules/bbox_detector"]
    if chosen != DETECTOR_STEM:
        die(f"resolved modules/bbox_detector is {chosen!r}, not {DETECTOR_STEM!r}. "
            f"All tuning must run on the YOLO11L SoccerNet fine-tune.")

    available = sorted(p.stem for p in (CONFIGS / "modules" / "bbox_detector").glob("*.yaml"))
    if available != [DETECTOR_STEM]:
        die(f"configs/modules/bbox_detector holds {available}; exactly one detector "
            f"config ({DETECTOR_STEM}) may exist. Do NOT add an oracle detector.")

    det = yaml.safe_load(
        (CONFIGS / "modules" / "bbox_detector" / f"{DETECTOR_STEM}.yaml").read_text(encoding="utf-8"))
    if det.get("_target_") != DETECTOR_TARGET:
        die(f"detector _target_ is {det.get('_target_')!r}, expected {DETECTOR_TARGET!r}")
    for key, want in DETECTOR_EXPECT.items():
        got = det["cfg"].get(key)
        if got != want:
            die(f"detector {key} is {got!r}, expected {want!r} — the tuning constraint "
                f"fixes imgsz/iou/conf floor. Change the constraint, or the config, "
                f"deliberately.")

    prov = {"detector": DETECTOR_STEM, "target": DETECTOR_TARGET,
            "hf_repo": HF_REPO, "checkpoint": HF_DETECTOR_FILE,
            "checkpoint_sha256": None, **DETECTOR_EXPECT}
    try:
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(HF_REPO, HF_DETECTOR_FILE)
        if isinstance(p, str) and Path(p).is_file():
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            prov["checkpoint_sha256"] = h.hexdigest()
            prov["checkpoint_path"] = p
    except Exception as e:                       # cache miss is not fatal, silence is
        prov["checkpoint_sha256_error"] = str(e)
    if prov["checkpoint_sha256"] is None:
        print("  WARNING: detector checkpoint not in the HF cache; provenance will "
              "carry no checkpoint hash. Fetch it before the runs that matter.")
    return prov


def sidecar(state: Path) -> Path:
    return state.with_suffix(".provenance.json")


def assert_state_is_yolo11l(state: Path) -> dict:
    """Every Stage-2 run must be able to prove which detector made its state."""
    side = sidecar(state)
    if not side.is_file():
        die(f"{state} has no provenance sidecar ({side.name}). Stage-2 results without "
            f"a recorded detector are not usable for T7 — re-run stage1.")
    prov = json.loads(side.read_text(encoding="utf-8"))
    if prov.get("detector", {}).get("detector") != DETECTOR_STEM:
        die(f"{side.name} records detector {prov.get('detector')}, not {DETECTOR_STEM}.")
    # Opportunistic cross-check against whatever tracklab wrote into the state.
    try:
        with zipfile.ZipFile(state) as zf:
            if "summary.json" in zf.namelist():
                text = zf.read("summary.json").decode("utf-8", "replace")
                for other in ("oracle", "ground_truth", "gt_detector"):
                    if other in text.lower():
                        die(f"{state.name}'s summary.json mentions {other!r} — the state "
                            f"may not come from {DETECTOR_STEM}.")
    except zipfile.BadZipFile:
        die(f"{state} is not a readable .pklz")
    return prov


# ============================================================================ clips
def discover_clips(dataset_path: Path, split: str) -> list:
    d = dataset_path / split
    clips = sorted(p.name for p in d.glob("SNGS-*") if (p / "img1").is_dir())
    if not clips:
        die(f"no SNGS-* sequences with img1/ under {d}. Fetch the split first: "
            f"SPLITS=\"valid test\" bash preflight_cpu.sh")
    return clips


def clip_plan(dataset_path: Path, split: str, nvid: int, holdout: int) -> dict:
    clips = discover_clips(dataset_path, split)
    if nvid <= 0 or nvid > len(clips):
        nvid = len(clips) - holdout
    if nvid + holdout > len(clips):
        die(f"asked for {nvid} sweep + {holdout} hold-out clips but split '{split}' "
            f"has {len(clips)}")
    plan = {"split": split, "available": len(clips),
            "sweep": clips[:nvid], "holdout": clips[nvid:nvid + holdout]}
    if holdout == 0:
        print("  WARNING: no hold-out clips reserved — the winner cannot be confirmed "
              "on unseen data, which T7 requires before promotion.")
    return plan


# ======================================================================== execution
def hydra_list(values) -> str:
    """Hydra list literal. Strings are quoted so names like SNGS-100 survive the
    override grammar (subprocess passes argv directly, so no shell eats them)."""
    def fmt(v):
        if isinstance(v, bool):
            return str(v).lower()
        return f"'{v}'" if isinstance(v, str) else str(v)
    return "[" + ",".join(fmt(v) for v in values) + "]"


def run_tracklab(overrides: dict, log_path: Path) -> None:
    cmd = ["tracklab", "-cn", "soccernet"] + [f"{k}={v}" for k, v in overrides.items()]
    print(f"    $ {' '.join(cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        r = subprocess.run(cmd, cwd=str(REPO), stdout=log,
                           stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        die(f"tracklab exited {r.returncode}; see {log_path}")


def score(state: Path, label: str, dataset_path: Path, split: str, out_dir: Path) -> dict:
    """HOTA/AssA/IDSW for one state, via reference_metrics.py (never reimplemented)."""
    cmd = [sys.executable, str(REPO / "scripts" / "reference_metrics.py"),
           "--state", str(state), "--label", label,
           "--dataset-path", str(dataset_path), "--eval-set", split,
           "--out", str(out_dir), "--skip", "gsr", "jersey_number", "calibration"]
    r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    report = out_dir / label / "reference_metrics.json"
    if r.returncode != 0 or not report.is_file():
        print(r.stdout[-4000:])
        print(r.stderr[-4000:], file=sys.stderr)
        die(f"scoring failed for {label} (exit {r.returncode})")
    block = json.loads(report.read_text(encoding="utf-8"))["tracking"]
    return {"HOTA": block.get("HOTA"), "DetA": block.get("DetA"),
            "AssA": block.get("AssA"), "IDF1": block.get("IDF1"),
            "MOTA": block.get("MOTA"), "IDSW": block.get("IDSW")}


# =========================================================================== stage 1
def cmd_stage1(args) -> int:
    dataset = Path(args.dataset_path).resolve()
    det = detector_provenance()
    plan = clip_plan(dataset, args.split, args.nvid, args.holdout)
    WORK.mkdir(parents=True, exist_ok=True)
    print(f"\nStage 1 — {len(STAGE1_VARIANTS)} tracker variant(s) over "
          f"{len(plan['sweep'])} clip(s) of '{args.split}' "
          f"({len(plan['holdout'])} held out, {plan['available']} available)")

    for name, overrides in STAGE1_VARIANTS:
        state = (WORK / f"{name}.pklz").resolve()
        if state.is_file() and not args.force:
            print(f"  [{name}] state exists, skipping (use --force to redo)")
            continue
        print(f"  [{name}]")
        run_tracklab({
            **overrides,
            "pipeline": hydra_list(["bbox_detector", "reid", "track"]),
            "dataset.eval_set": args.split,
            "dataset.nvid": -1,
            f"dataset.vids_dict.{args.split}": hydra_list(plan["sweep"]),
            "state.save_file": str(state),
            "state.load_file": "null",
            "visualization.cfg.save_videos": "false",
            "eval_tracking": "false",
        }, RESULTS / "logs" / f"stage1_{name}.log")
        if not state.is_file():
            die(f"[{name}] tracklab finished but {state} was not written")
        sidecar(state).write_text(json.dumps({
            "stage": 1, "variant": name, "overrides": overrides,
            "detector": det, "clips": plan, "git_commit": git_commit(),
            "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        }, indent=2), encoding="utf-8")
        print(f"    -> {state}")

    (RESULTS / "clip_plan.json").parent.mkdir(parents=True, exist_ok=True)
    (RESULTS / "clip_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"\nClip plan -> {RESULTS / 'clip_plan.json'}")
    return 0


# =========================================================================== stage 2
def expand(grid: dict, split_grid: dict) -> list:
    """Cross product, expanding the split axes only when use_split is True."""
    base_keys = [k for k in grid if k != "use_split"]
    configs = []
    for use_split in grid["use_split"]:
        axes = [grid[k] for k in base_keys]
        extra_keys = list(split_grid) if use_split else []
        axes += [split_grid[k] for k in extra_keys]
        for combo in itertools.product(*axes):
            cfg = dict(zip(base_keys + extra_keys, combo))
            cfg["use_split"] = use_split
            configs.append(cfg)
    return configs


def build_fine_grid(best: dict) -> tuple:
    """Re-centre the coarse grid on the winner (one step either side per axis)."""
    def around(key, steps, mult=False, lo=None, cast=float):
        base = best.get(key)
        if base is None:
            return None
        vals = sorted({cast(base * s) if mult else cast(base + s) for s in steps})
        return [v for v in vals if lo is None or v > lo]

    grid = {
        "connect_mode": [best["connect_mode"]],
        "appearance_thresh": around("appearance_thresh", FINE_STEPS["appearance_thresh"], lo=0.0),
        "spatial_thresh": around("spatial_thresh", FINE_STEPS["spatial_thresh"], mult=True, lo=0.0),
        "min_tracklet_len": around("min_tracklet_len", FINE_STEPS["min_tracklet_len"],
                                   lo=0, cast=int),
        "use_split": [best["use_split"]],
        "interp_n_dti": [best["interp_n_dti"]],
    }
    split_grid = {}
    if best["use_split"]:
        split_grid = {
            "split_eps": around("split_eps", FINE_STEPS["split_eps"], lo=0.0),
            "split_min_samples": [best["split_min_samples"]],
            "split_max_k": [best["split_max_k"]],
            "split_len_thres": [best["split_len_thres"]],
        }
    return grid, split_grid


def config_overrides(cfg: dict) -> dict:
    ov = {
        "modules.gta_link.cfg.connect_mode": cfg["connect_mode"],
        "modules.gta_link.cfg.appearance_thresh": cfg["appearance_thresh"],
        "modules.gta_link.cfg.spatial_thresh": cfg["spatial_thresh"],
        "modules.gta_link.cfg.min_tracklet_len": cfg["min_tracklet_len"],
        "modules.gta_link.cfg.use_split": str(bool(cfg["use_split"])).lower(),
    }
    for key in ("split_eps", "split_min_samples", "split_max_k", "split_len_thres"):
        if key in cfg:
            ov[f"modules.gta_link.cfg.{key}"] = cfg[key]
    if cfg["interp_n_dti"] is None:
        ov["modules.interpolation.cfg.enabled"] = "false"
    else:
        ov["modules.interpolation.cfg.enabled"] = "true"
        ov["modules.interpolation.cfg.n_dti"] = cfg["interp_n_dti"]
    return ov


def cmd_stage2(args) -> int:
    dataset = Path(args.dataset_path).resolve()
    det = detector_provenance()
    plan_file = RESULTS / "clip_plan.json"
    if not plan_file.is_file():
        die(f"{plan_file} missing — run stage1 first (it records the clip split).")
    plan = json.loads(plan_file.read_text(encoding="utf-8"))

    states = sorted(WORK.glob("*.pklz"))
    if not states:
        die(f"no Stage-1 states under {WORK} — run stage1 first.")

    if args.fine:
        prev = RESULTS / "gta_optima.json"
        if not prev.is_file():
            die(f"--fine needs the coarse winner in {prev}; run the coarse pass first.")
        best = {k: _coerce(v) for k, v in
                json.loads(prev.read_text(encoding="utf-8"))["winner"]["config"].items()}
        grid, split_grid = build_fine_grid(best)
        tag = "fine"
        print(f"\nStage 2 (fine) — re-centred on {best}")
    else:
        grid, split_grid = COARSE_GRID, COARSE_SPLIT_GRID
        tag = "coarse"

    configs = expand(grid, split_grid)
    if args.max_configs and len(configs) > args.max_configs:
        print(f"  NOTE: grid has {len(configs)} configs; --max-configs caps it at "
              f"{args.max_configs}. DROPPED {len(configs) - args.max_configs} configs "
              f"— this pass does NOT cover the declared grid.")
        configs = configs[:args.max_configs]

    rows = []
    for state in states:
        assert_state_is_yolo11l(state)
        variant = state.stem
        print(f"\n=== state '{variant}' ({len(configs)} configs) ===")

        # GTA_MUST_BEAT_RAW reference: the same state, same clips, no GTA-Link.
        raw = score(state, f"{variant}__raw", dataset, plan["split"],
                    RESULTS / tag / "scores")
        print(f"  raw (no GTA): HOTA={raw['HOTA']} AssA={raw['AssA']} IDSW={raw['IDSW']}")
        rows.append({"state": variant, "config_id": "raw", "beats_raw": "",
                     **{f"cfg_{k}": "" for k in _ALL_CFG_KEYS}, **raw})

        for i, cfg in enumerate(configs):
            label = f"{variant}__{tag}{i:04d}"
            out_state = (WORK / f"_{label}.pklz").resolve()
            print(f"  [{i + 1}/{len(configs)}] {cfg}")
            run_tracklab({
                **config_overrides(cfg),
                "pipeline": hydra_list(["gta_link", "interpolation"]),
                "dataset.eval_set": plan["split"],
                "dataset.nvid": -1,
                f"dataset.vids_dict.{plan['split']}": hydra_list(plan["sweep"]),
                "state.load_file": str(state),
                "state.save_file": str(out_state),
                "visualization.cfg.save_videos": "false",
                "eval_tracking": "false",
            }, RESULTS / "logs" / f"{label}.log")
            m = score(out_state, label, dataset, plan["split"], RESULTS / tag / "scores")
            beats = (m["HOTA"] is not None and raw["HOTA"] is not None
                     and m["HOTA"] > raw["HOTA"])
            print(f"      HOTA={m['HOTA']} AssA={m['AssA']} IDSW={m['IDSW']} "
                  f"{'BEATS raw' if beats else 'rejected (GTA_MUST_BEAT_RAW)'}")
            rows.append({"state": variant, "config_id": label,
                         "beats_raw": int(beats),
                         **{f"cfg_{k}": cfg.get(k, "") for k in _ALL_CFG_KEYS}, **m})
            if not args.keep_states:
                out_state.unlink(missing_ok=True)
                sidecar(out_state).unlink(missing_ok=True)


    write_csv(RESULTS / f"{tag}_results.csv", rows)
    winner = pick_winner(rows)
    payload = {
        "pass": tag, "winner": winner, "n_configs": len(configs),
        "guards": {"GTA_MUST_BEAT_RAW": "enforced — configs not beating the same "
                                        "state's no-GTA score are rejected",
                   "holdout_confirmation": "run `tune_gta_kaggle.py holdout`"},
        "provenance": {"detector": det, "clips": plan, "git_commit": git_commit(),
                       "generated": datetime.datetime.now().isoformat(timespec="seconds"),
                       "scoring": "scripts/reference_metrics.py --eval-set "
                                  f"{plan['split']} --skip gsr jersey_number calibration"},
    }
    (RESULTS / "gta_optima.json").write_text(json.dumps(payload, indent=2, default=str),
                                             encoding="utf-8")
    print(f"\nWinner: {winner}")
    print(f"-> {RESULTS / 'gta_optima.json'}")
    print(f"-> {RESULTS / f'{tag}_results.csv'}")
    return 0


_ALL_CFG_KEYS = ["connect_mode", "appearance_thresh", "spatial_thresh",
                 "min_tracklet_len", "use_split", "split_eps", "split_min_samples",
                 "split_max_k", "split_len_thres", "interp_n_dti"]


def write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["state", "config_id", "beats_raw"] + [f"cfg_{k}" for k in _ALL_CFG_KEYS] \
        + ["HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def pick_winner(rows: list) -> dict:
    """Best HOTA among configs that beat their own raw baseline; IDSW breaks ties."""
    eligible = [r for r in rows if r["beats_raw"] == 1 and r["HOTA"] is not None]
    if not eligible:
        return {"config": None,
                "reason": "no config beat the no-GTA baseline (GTA_MUST_BEAT_RAW) — "
                          "GTA-Link is not helping at this grid; widen it or leave "
                          "the stage off"}
    top = max(r["HOTA"] for r in eligible)
    near = [r for r in eligible if top - r["HOTA"] <= TIE_HOTA]
    best = min(near, key=lambda r: (r["IDSW"] if r["IDSW"] is not None else 1e9))
    cfg = {k: best[f"cfg_{k}"] for k in _ALL_CFG_KEYS if best[f"cfg_{k}"] != ""}
    return {"config": cfg, "state": best["state"], "config_id": best["config_id"],
            "HOTA": best["HOTA"], "AssA": best["AssA"], "IDSW": best["IDSW"],
            "n_within_tie": len(near), "tie_window": TIE_HOTA}


# ========================================================================== hold-out
def cmd_holdout(args) -> int:
    dataset = Path(args.dataset_path).resolve()
    det = detector_provenance()
    optima = RESULTS / "gta_optima.json"
    if not optima.is_file():
        die(f"{optima} missing — run stage2 first.")
    payload = json.loads(optima.read_text(encoding="utf-8"))
    winner = payload["winner"]
    if not winner.get("config"):
        die(f"no winner to confirm: {winner.get('reason')}")
    plan = payload["provenance"]["clips"]
    if not plan["holdout"]:
        die("no hold-out clips were reserved; re-run stage1 with --holdout N. "
            "T7 promotion requires confirmation on clips the sweep never saw.")

    cfg = {k: _coerce(v) for k, v in winner["config"].items()}
    state = WORK / f"{winner['state']}.pklz"
    if not state.is_file():
        die(f"Stage-1 state {state} missing; re-run stage1 for variant "
            f"{winner['state']!r}.")
    variant_overrides = dict(json.loads(sidecar(state).read_text(encoding="utf-8"))["overrides"])

    print(f"\nHold-out confirmation on {len(plan['holdout'])} unseen clip(s): "
          f"{plan['holdout']}")
    hold_state = (WORK / "holdout_track.pklz").resolve()
    if not hold_state.is_file() or args.force:
        run_tracklab({
            **variant_overrides,
            "pipeline": hydra_list(["bbox_detector", "reid", "track"]),
            "dataset.eval_set": plan["split"], "dataset.nvid": -1,
            f"dataset.vids_dict.{plan['split']}": hydra_list(plan["holdout"]),
            "state.save_file": str(hold_state), "state.load_file": "null",
            "visualization.cfg.save_videos": "false", "eval_tracking": "false",
        }, RESULTS / "logs" / "holdout_track.log")
        sidecar(hold_state).write_text(json.dumps(
            {"stage": "holdout", "variant": winner["state"],
             "overrides": variant_overrides, "detector": det, "clips": plan,
             "git_commit": git_commit()}, indent=2), encoding="utf-8")
    assert_state_is_yolo11l(hold_state)

    out = RESULTS / "holdout" / "scores"
    raw = score(hold_state, "holdout__raw", dataset, plan["split"], out)
    tuned_state = (WORK / "_holdout_tuned.pklz").resolve()
    run_tracklab({
        **config_overrides(cfg),
        "pipeline": hydra_list(["gta_link", "interpolation"]),
        "dataset.eval_set": plan["split"], "dataset.nvid": -1,
        f"dataset.vids_dict.{plan['split']}": hydra_list(plan["holdout"]),
        "state.load_file": str(hold_state), "state.save_file": str(tuned_state),
        "visualization.cfg.save_videos": "false", "eval_tracking": "false",
    }, RESULTS / "logs" / "holdout_tuned.log")
    tuned = score(tuned_state, "holdout__tuned", dataset, plan["split"], out)

    held = (tuned["HOTA"] is not None and raw["HOTA"] is not None
            and tuned["HOTA"] > raw["HOTA"])
    result = {
        "confirmed": bool(held), "clips": plan["holdout"],
        "raw": raw, "tuned": tuned, "config": winner["config"],
        "sweep_HOTA": winner["HOTA"],
        "verdict": ("winner holds on unseen clips — promotable to T7" if held else
                    "winner does NOT beat raw on unseen clips — widen the clip set "
                    "and re-tune; do NOT promote"),
        "provenance": {"detector": det, "git_commit": git_commit(),
                       "generated": datetime.datetime.now().isoformat(timespec="seconds")},
    }
    (RESULTS / "holdout_confirmation.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\n  raw   HOTA={raw['HOTA']} AssA={raw['AssA']} IDSW={raw['IDSW']}")
    print(f"  tuned HOTA={tuned['HOTA']} AssA={tuned['AssA']} IDSW={tuned['IDSW']}")
    print(f"\n{result['verdict']}")
    print(f"-> {RESULTS / 'holdout_confirmation.json'}")
    if not args.keep_states:
        tuned_state.unlink(missing_ok=True)
    return 0 if held else 1


def _coerce(v):
    if isinstance(v, str):
        low = v.strip().lower()
        if low in ("true", "false"):
            return low == "true"
        if low in ("none", ""):
            return None
        for cast in (int, float):
            try:
                return cast(v)
            except ValueError:
                pass
    return v


# ============================================================================ check
def cmd_check(args) -> int:
    """Dry run of every guard: no GPU, no weights, no tracklab needed."""
    det = detector_provenance()
    print("\ndetector constraint: OK")
    for k, v in det.items():
        print(f"  {k}: {v}")

    coarse = expand(COARSE_GRID, COARSE_SPLIT_GRID)
    n_split = sum(1 for c in coarse if c["use_split"])
    print(f"\ncoarse grid: {len(coarse)} configs "
          f"({len(coarse) - n_split} without Split, {n_split} with)")
    print(f"  example: {coarse[0]}")
    print(f"  overrides: {config_overrides(coarse[0])}")
    fake = dict(coarse[0])
    fake.update(use_split=True, split_eps=0.6, split_min_samples=4,
                split_max_k=3, split_len_thres=30)
    g, sg = build_fine_grid(fake)
    print(f"  fine grid around it: {len(expand(g, sg))} configs")

    data = Path(args.dataset_path)
    if (data / args.split).is_dir():
        plan = clip_plan(data, args.split, 0, 8)
        print(f"\nclips: {plan['available']} available, {len(plan['sweep'])} sweep, "
              f"{len(plan['holdout'])} hold-out")
    else:
        print(f"\nclips: {data / args.split} absent — fetch with "
              f"SPLITS=\"valid test\" bash preflight_cpu.sh")
    print(f"\ngit commit: {git_commit()}")
    print("\nAll static guards pass. Runs still require tracklab + GPU.")
    return 0


# ============================================================================== cli
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-path", default=str(REPO / "data" / "SoccerNetGS"))
    ap.add_argument("--split", default="valid", choices=["valid", "test"],
                    help="tuning runs on valid; test is reserved for the frozen run")
    ap.add_argument("--keep-states", action="store_true",
                    help="keep every Stage-2 state (disk-hungry; off by default)")
    ap.add_argument("--force", action="store_true", help="redo existing states")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("stage1", help="tracker variants (full runs)")
    p1.add_argument("--nvid", type=int, default=0,
                    help="sweep clips (0 = every clip except the hold-out)")
    p1.add_argument("--holdout", type=int, default=8,
                    help="clips reserved for the confirmation pass")
    p1.set_defaults(func=cmd_stage1)

    p2 = sub.add_parser("stage2", help="post-processing grid over the Stage-1 states")
    p2.add_argument("--max-configs", type=int, default=0,
                    help="cap the grid (0 = no cap); drops are logged, never silent")
    p2.add_argument("--fine", action="store_true",
                    help="fine pass re-centred on the coarse winner")
    p2.set_defaults(func=cmd_stage2)

    p3 = sub.add_parser("holdout", help="confirm the winner on unseen clips")
    p3.set_defaults(func=cmd_holdout)

    p4 = sub.add_parser("check", help="verify every guard without running anything")
    p4.set_defaults(func=cmd_check)

    args = ap.parse_args(argv)
    if args.split != "valid":
        print("  WARNING: tuning on a split other than 'valid'. The test split is "
              "reserved for the frozen production run.")
    if shutil.which("tracklab") is None and args.cmd != "check":
        die("`tracklab` is not on PATH — activate the project venv first.")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
