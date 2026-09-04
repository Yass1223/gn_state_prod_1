"""jn_pipeline_gsr as the jersey-number stage: tracklet-level, multi-GPU, subprocess-run.

Replaces the per-detection MMOCR stage with the jersey-number pipeline from
``plugins/jn_gsr`` (legibility > ``cfg.legibility_thr`` (0.72) -> DBNet++ ROI >
0.52 -> PARSeq ``parseq_gsr_ft_s1.ckpt`` AND SATRN
``recog2/best_recog_word_acc_epoch_10.pth`` on the same surviving crops ->
vote_pool consolidation: the pooled per-frame decodes of both models are
majority-voted; see ``plugins/jn_gsr/jn_recognizer.py``). vote_pool is the
only rule of this build; there is no ``rule`` setting.

Crop selection: with ``cfg.single_crops_only`` (default true) only the detections
the crop filter labelled ``crop_single`` are handed to the recognisers; overlapping
crops never enter the manifest, and a tracklet with no single crop is left
unnumbered. The worker's ``stride`` therefore steps over the tracklet's single
crops, not over all its frames.

Measured on GT-box tracklets (fusion run of 2026-08-17, both audits 0 fail):
GSR-2024 test trk_acc 87.86 / numbered 91.34 / -1 F1 0.82 (PARSeq maxconf
alone: 87.48 / 90.83 / 0.82); SN jersey-2023 test 85.30 / 84.58 / 0.81
(84.72 / 83.76 / 0.81). No number exists yet for predicted tracklets through
this module; ``scripts/reference_metrics.py`` on a saved state produces it.

Why a subprocess (not an import)
--------------------------------
The JN package requires python 3.10 / torch 2.0.1+cu118 / mmcv 2.0.1 / numpy 1.25.2
while this repo pins python 3.9 / torch 1.13.1 -- incompatible interpreters by
design. The package ships its own venv builder (``setup_env.py`` -> ``.venv_jn``,
"used only as a subprocess interpreter"), so this module writes a tracklet manifest,
launches ``predict_tracklets.py`` inside that venv (one worker per GPU), and merges
the shard outputs. Build everything once with ``scripts/setup_jn_gsr.sh``.

GPU sharding (jnintegration.txt requirement)
--------------------------------------------
``gpus: auto`` detects the available GPUs via ``nvidia-smi -L`` (environment-
independent -- the host torch build is irrelevant) and launches one worker per GPU
with ``CUDA_VISIBLE_DEVICES=<i>`` -- 2 workers on Kaggle 2xT4, 4 on Lightning 4xT4,
1 worker when one GPU (or none detected). Tracklets are partitioned
``sorted(tids)[i::n]`` (deterministic, disjoint, complete -- asserted at merge).

Zero collateral changes
-----------------------
Output contract is byte-compatible with the MMOCR stage it replaces: per-detection
``jersey_number_detection`` (digit string, or None for "-1"/unrecognized) and
``jersey_number_confidence`` (the consolidation share; 0.0 with None). The stage
keeps the same pipeline slot; ``MajorityVoteTracklet`` then votes over a per-track
constant, yielding the identical final ``jersey_number`` -- the GSR encoder and
the eval are untouched.

Two ADDITIVE columns feed the ``traj_refine`` stage (blob schema 2):
``jersey_number_candidates`` -- every pooled label of the two recognisers as
``[label, mx, conf_sum, votes]``, ranked by the maxconf score exp(mx) *
conf_sum (fuse_jn.ranked_candidates; stats, not scores, so merged tracklets
recombine exactly: mx = max, conf_sum/votes add) -- and
``jersey_number_maxconf``, the assigned number's maxconf score (0.0 when
unnumbered). The assigned number itself is still vote_pool, unchanged.

Scope: every fragment holding at least one single crop is recognized -- there
is no role filter, because roles do not exist yet (``role_team`` runs AFTER
``traj_refine`` in this architecture); the legibility filter inside the workers
is what naturally discards referee and backs-turned crops, exactly like MMOCR's
no-digit outcome. Team plays no part here.

Caching, rigorously
-------------------
Per-sequence cache keyed on a sha256 of the manifest CONTENT (track ids + frame
paths + boxes + rule + stride + legibility_thr + single_crops_only + the sha256 of
BOTH recogniser checkpoints). Track ids change whenever tracking or split_merge is
retuned, so a
name-only cache would silently serve stale numbers; exact-hash match only.

The last components exist because they are the ways this stage's output can
change WITHOUT the manifest moving: retuning the legibility gate, and staging
a different checkpoint under the same filename (either recogniser). Keying on
the files' CONTENT rather than their names is what makes a checkpoint swap
take effect instead of silently reusing the previous run's numbers.

Any worker failure or timeout -> (None, 0.0) columns and a loud log, never a
crash and never a fabricated value.
"""
import hashlib
import json
import logging
import math
import os
import subprocess
import time
from pathlib import Path

import pandas as pd

from tracklab.pipeline.videolevel_module import VideoLevelModule

log = logging.getLogger(__name__)

UNNUMBERED = "-1"

# The build's one consolidation rule (jn_recognizer.RULE / predict_tracklets.
# RULE); recorded in the cache key and checked against every shard's output.
RULE = "vote_pool"

# Worker blob schema this stage requires (predict_tracklets.SCHEMA). 2 adds
# per-tracklet "candidates". Folded into the cache key (an old cache misses
# and recomputes once) and checked on every shard and cached blob.
SCHEMA = 2

# Must match jn_recognizer's `parseq_ckpt` / `satrn_ckpt` defaults. They cannot
# be imported: that module lives in the OTHER interpreter (python 3.10 /
# torch 2.0.1). Override with cfg.parseq_ckpt / cfg.satrn_ckpt; both are
# passed to the worker AND hashed into the cache key.
PARSEQ_CKPT = "parseq_gsr_ft_s1.ckpt"
SATRN_CKPT = "recog2/best_recog_word_acc_epoch_10.pth"


def detect_gpus():
    """GPU indices via nvidia-smi (host-env independent). [] when none/absent."""
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                             text=True, timeout=30)
        if out.returncode != 0:
            return []
        return [str(i) for i, line in enumerate(
            l for l in out.stdout.splitlines() if l.strip().startswith("GPU "))]
    except (OSError, subprocess.TimeoutExpired):
        return []


class JNGsrTrackletRecognizer(VideoLevelModule):
    """Tracklet-level jersey recognition via the vendored jn_pipeline_gsr package."""

    input_columns = ["track_id", "bbox_ltwh", "image_id", "crop_single"]
    output_columns = ["jersey_number_detection", "jersey_number_confidence",
                      "jersey_number_candidates", "jersey_number_maxconf"]

    def __init__(self, cfg, device, tracking_dataset=None):
        self.cfg = cfg
        self.device = device
        self.pipeline_dir = Path(str(cfg.pipeline_dir))
        self.venv_python = Path(str(cfg.venv_python))
        self.worker = Path(str(getattr(cfg, "worker", "") or
                               self.pipeline_dir / "predict_tracklets.py"))
        self.models_dir = Path(str(cfg.models_dir))
        self.cache_dir = Path(str(cfg.cache_dir))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = bool(getattr(cfg, "use_cache", True))
        if getattr(cfg, "rule", None) not in (None, RULE):
            raise ValueError(
                f"[jn_gsr] cfg.rule={cfg.rule!r} is not configurable: this "
                f"build consolidates with {RULE!r} only (PARSeq + SATRN). "
                f"Remove `rule` from jn_gsr.yaml.")
        self.rule = RULE
        self.stride = int(getattr(cfg, "stride", 5))           # frozen default
        self.legibility_thr = float(getattr(cfg, "legibility_thr", 0.72))
        self.parseq_ckpt = str(getattr(cfg, "parseq_ckpt", "") or PARSEQ_CKPT)
        self.satrn_ckpt = str(getattr(cfg, "satrn_ckpt", "") or SATRN_CKPT)
        self._ckpt_ids = {}                       # memoised; see _ckpt_id()
        self.fp16 = bool(getattr(cfg, "fp16", True))
        # Single crops only: hand the recognisers the detections the crop filter
        # labelled crop_single (one person in the box); overlapping crops are
        # excluded from the manifest. A tracklet with no single crop is left
        # unnumbered. Recorded in the cache key and in the blob; audit-checked.
        self.single_crops_only = bool(getattr(cfg, "single_crops_only", True))
        self.gpus = getattr(cfg, "gpus", "auto")
        self.timeout = int(getattr(cfg, "timeout", 10800))     # per video
        self.worker_extra_args = [str(x) for x in
                                  getattr(cfg, "worker_extra_args", []) or []]

    # ------------------------------------------------------------- manifest --
    def _build_manifest(self, detections, metadatas):
        """{tid: [[frame_path, [l,t,w,h]], ...]} chronological; with
        single_crops_only, restricted to crop_single detections. Returns
        (manifest, stats) with stats = dict(tracklets_skipped_no_single,
        frames_excluded_multi, manifest_tracklets, manifest_frames)."""
        if self.single_crops_only and "crop_single" not in detections.columns:
            raise RuntimeError("[jn_gsr] crop_single column missing - the crop_filter "
                               "stage must run before jersey_number_detect")
        id2path = {idx: str(p) for idx, p in metadatas["file_path"].items()}
        order = {idx: Path(p).name for idx, p in id2path.items()}
        tracked = detections.dropna(subset=["track_id"])
        manifest = {}
        stats = dict(tracklets_skipped_no_single=0,
                     frames_excluded_multi=0, manifest_tracklets=0, manifest_frames=0)
        for tid, grp in tracked.groupby("track_id"):
            # Eligibility is single-crop presence only: roles do not exist yet
            # (role_team runs AFTER traj_refine in this architecture); the
            # legibility filter inside the workers is what naturally discards
            # referee crops.
            if self.single_crops_only:
                keep = grp["crop_single"].astype(bool)
                stats["frames_excluded_multi"] += int((~keep).sum())
                grp = grp[keep]
                if len(grp) == 0:
                    stats["tracklets_skipped_no_single"] += 1
                    continue
            grp = grp.assign(_fn=grp["image_id"].map(order)).sort_values("_fn")
            frames = []
            for _, row in grp.iterrows():
                fp = id2path.get(row["image_id"])
                if fp is None:
                    continue
                l, t, w, h = [float(v) for v in row["bbox_ltwh"]]
                frames.append([fp, [l, t, w, h]])
            if frames:
                manifest[str(tid)] = frames
        stats["manifest_tracklets"] = len(manifest)
        stats["manifest_frames"] = sum(len(v) for v in manifest.values())
        return manifest, stats

    # ------------------------------------------------------- cache identity --
    def _ckpt_id(self, rel):
        """sha256 of a staged checkpoint (path relative to models_dir),
        computed ONCE per process.

        286 MB (PARSeq) / 48 MB (SATRN), ~1 s -- acceptable once, not once
        per sequence and certainly not once per tracklet. A missing file is
        not an error here (the worker launch reports that far more usefully);
        it just keys the cache under a distinct sentinel so a run made before
        the checkpoint was staged can never be confused with one made after."""
        if rel not in self._ckpt_ids:
            p = self.models_dir / rel
            try:
                h = hashlib.sha256()
                with open(p, "rb") as f:
                    for b in iter(lambda: f.read(1 << 22), b""):
                        h.update(b)
                self._ckpt_ids[rel] = h.hexdigest()
            except OSError as e:
                log.warning(f"[jn_gsr] could not hash {p} ({e}); the cache key "
                            f"will use a sentinel instead of the checkpoint's "
                            f"content")
                self._ckpt_ids[rel] = f"unreadable:{rel}"
        return self._ckpt_ids[rel]

    def _manifest_hash(self, manifest, rule, stride):
        """Everything that changes this stage's OUTPUT, hashed.

        legibility_thr and both checkpoints' CONTENT are in here deliberately:
        they change the numbers without touching the manifest, so a key
        without them serves stale results after exactly the edits this stage
        is most likely to receive."""
        payload = json.dumps({"tracklets": manifest, "rule": rule,
                              "schema": SCHEMA,
                              "stride": stride,
                              "legibility_thr": self.legibility_thr,
                              "single_crops_only": self.single_crops_only,
                              "parseq_sha256": self._ckpt_id(self.parseq_ckpt),
                              "satrn_sha256": self._ckpt_id(self.satrn_ckpt)},
                             sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    # -------------------------------------------------------------- workers --
    def _launch_workers(self, manifest_path, out_dir):
        if self.gpus == "auto":
            gpu_ids = detect_gpus() or ["0"]  # no GPU visible -> 1 worker (CPU/default)
        elif isinstance(self.gpus, int):
            gpu_ids = [str(i) for i in range(max(1, self.gpus))]
        else:
            gpu_ids = [str(g) for g in self.gpus] or ["0"]
        n = len(gpu_ids)
        log.info(f"[jn_gsr] launching {n} worker(s) on GPU(s) {gpu_ids}")
        procs = []
        for i, g in enumerate(gpu_ids):
            cmd = [str(self.venv_python), str(self.worker),
                   "--manifest", str(manifest_path),
                   "--out", str(out_dir / f"shard_{i}.json"),
                   "--models-dir", str(self.models_dir),
                   "--shard", str(i), "--num-shards", str(n),
                   "--stride", str(self.stride),
                   "--parseq-ckpt", self.parseq_ckpt,
                   "--satrn-ckpt", self.satrn_ckpt,
                   "--legibility-thr", str(self.legibility_thr),
                   ] + ([] if self.fp16 else ["--no-fp16"]) \
                     + self.worker_extra_args
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=g,
                       PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
            procs.append(subprocess.Popen(
                cmd, env=env, cwd=str(self.pipeline_dir),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
        deadline = time.time() + self.timeout
        outputs = [[] for _ in procs]
        for i, p in enumerate(procs):
            try:
                out, _ = p.communicate(timeout=max(1, deadline - time.time()))
                outputs[i] = out.splitlines()
            except subprocess.TimeoutExpired:
                for q in procs:
                    if q.poll() is None:
                        q.kill()
                log.error(f"[jn_gsr] timeout after {self.timeout}s")
                return None
        failed = [i for i, p in enumerate(procs) if p.returncode != 0]
        if failed:
            for i in failed:
                tail = "\n".join(outputs[i][-25:])
                log.error(f"[jn_gsr] worker {i} FAILED (rc="
                          f"{procs[i].returncode}). Output tail:\n{tail}")
            return None
        merged = {}
        for i in range(len(procs)):
            shard_file = out_dir / f"shard_{i}.json"
            try:
                blob = json.loads(shard_file.read_text())
                if blob.get("rule") != self.rule:
                    log.error(f"[jn_gsr] {shard_file} was produced under rule "
                              f"{blob.get('rule')!r}, expected {self.rule!r}")
                    return None
                if blob.get("schema") != SCHEMA:
                    log.error(f"[jn_gsr] {shard_file} was produced under schema "
                              f"{blob.get('schema')!r}, expected {SCHEMA} -- the "
                              f"vendored worker predates the candidates output")
                    return None
                merged.update(blob["results"])
            except (OSError, json.JSONDecodeError, KeyError) as e:
                log.error(f"[jn_gsr] could not read {shard_file}: {e}")
                return None
        return merged

    # ---------------------------------------------------------------- main ---
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        detections = detections.copy()
        detections["jersey_number_detection"] = None
        detections["jersey_number_confidence"] = 0.0
        detections["jersey_number_candidates"] = None
        detections["jersey_number_maxconf"] = 0.0
        if len(detections) == 0 or "track_id" not in detections.columns \
                or detections["track_id"].isna().all():
            return detections

        seq = Path(str(metadatas["file_path"].iloc[0])).parent.parent.name \
            if len(metadatas) else "unknown"
        manifest, stats = self._build_manifest(detections, metadatas)
        if not manifest:
            log.warning(f"[jn_gsr] {seq}: no eligible tracklets "
                        f"({stats['tracklets_skipped_no_single']} with no single crop)")
            return detections
        mhash = self._manifest_hash(manifest, self.rule, self.stride)
        cache_file = self.cache_dir / f"{seq}.{mhash[:12]}.json"

        results = None
        if self.use_cache and cache_file.is_file():
            try:
                blob = json.loads(cache_file.read_text())
                # exact-content match only, and only a blob of this schema
                if blob.get("manifest_sha256") == mhash \
                        and blob.get("schema") == SCHEMA:
                    results = blob["results"]
                    log.info(f"[jn_gsr] {seq}: cache hit ({cache_file.name})")
            except (OSError, json.JSONDecodeError, KeyError):
                results = None
        if results is None:
            if not self.venv_python.is_file() or not self.worker.is_file():
                log.error(
                    f"[jn_gsr] missing venv python ('{self.venv_python}') or worker "
                    f"('{self.worker}'). Run scripts/setup_jn_gsr.sh first. "
                    f"Jersey numbers left empty for {seq}.")
                return detections
            run_dir = self.cache_dir / f"_run_{seq}_{mhash[:8]}"
            run_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = run_dir / "manifest.json"
            manifest_path.write_text(json.dumps({"tracklets": manifest}))
            results = self._launch_workers(manifest_path, run_dir)
            if results is None:
                return detections  # loud logs above; empty columns, no fabrication
            missing = set(manifest) - set(results)
            if missing:
                log.error(f"[jn_gsr] {seq}: {len(missing)} tracklet(s) missing "
                          f"from worker output -- left unnumbered")
            cache_file.write_text(json.dumps({
                "manifest_sha256": mhash, "rule": self.rule,
                "schema": SCHEMA,
                "stride": self.stride,
                "legibility_thr": self.legibility_thr,
                "single_crops_only": self.single_crops_only,
                "manifest_stats": stats,
                "parseq_ckpt": self.parseq_ckpt,
                "parseq_sha256": self._ckpt_id(self.parseq_ckpt),
                "satrn_ckpt": self.satrn_ckpt,
                "satrn_sha256": self._ckpt_id(self.satrn_ckpt),
                "results": results}))

        n_numbered = 0
        tid_str = detections["track_id"].map(
            lambda v: None if pd.isna(v) else str(v))
        for tid, res in results.items():
            number = res.get("number", UNNUMBERED)
            mask = tid_str == tid
            cand = res.get("candidates") or []
            if cand:
                # one shared list object per tracklet; consumers treat it
                # read-only (traj_refine copies the stats before combining)
                detections.loc[mask, "jersey_number_candidates"] = \
                    pd.Series([cand] * int(mask.sum()),
                              index=detections.index[mask])
            if number != UNNUMBERED and str(number).isdigit():
                detections.loc[mask, "jersey_number_detection"] = str(int(number))
                detections.loc[mask, "jersey_number_confidence"] = \
                    float(res.get("confidence", 0.0))
                entry = next((c for c in cand if str(c[0]) == str(number)), None)
                if entry is not None:
                    detections.loc[mask, "jersey_number_maxconf"] = \
                        float(math.exp(float(entry[1])) * float(entry[2]))
                n_numbered += 1
        log.info(f"[jn_gsr] {seq}: {n_numbered}/{len(manifest)} tracklets "
                 f"numbered (no role filter - roles are assigned after refine; "
                 f"{stats['tracklets_skipped_no_single']} with no single crop, "
                 f"{stats['frames_excluded_multi']} overlapping crops excluded, "
                 f"single_crops_only={self.single_crops_only}, "
                 f"legibility > {self.legibility_thr})")
        return detections
