# SoccerNet Game State Reconstruction — pipeline reference (repo_prod_v1)

Reference date: 2026-09-04. Derived from direct inspection of the repository
(configs, source, scripts), not from the README alone, and validated by an end-to-end
Kaggle run on one test sequence (see §7 findings and `docs/KAGGLE_GUIDE.md`). Extended
2026-09-03 with the `traj_refine` stage, the jersey candidate output (blob schema 2),
and the second detector (YOLOv11L_HM, now the default) — first exercised on Kaggle the
same day (run 5, §7): full pipeline exit 0, RUN INTEGRITY OK, all 13 audit checks PASS;
the detector-default decision is pending an A/B (§7 item 9, §9). Restructured
2026-09-04 to the split-only architecture: `split_merge` retired, replaced by
`tracklet_split` (Stage 1, split only) with the pipeline's ONE merge in `traj_refine`
(Stage 2 + stage-3 duplicate-frame resolution); not yet exercised on Kaggle (§8 batch 5,
§9).
Intended to
be pasted into any working context as the ground-truth description of the pipeline, so the
repository does not need to be re-analyzed each time. Statements are marked **[verified]**
(read directly from code/config or observed at runtime), **[assumption]** (design intent,
not independently confirmed), or **[unverified]** (not yet exercised). The GitHub remote
is `https://github.com/Yass1223/gn_state_prod_1` (main). Canonical copy:
`docs/PIPELINE_REFERENCE.md` in the repository.

---

## 1. What the repository is

A single-path SoccerNet Game State Reconstruction (GSR) pipeline built on TrackLab, with
one entry config (`sn_gamestate/configs/soccernet.yaml`) and no alternative backends.
The pipeline order **[verified]**:

```
bbox_detector -> track -> crop_filter -> tracklet_split -> calibration -> pitch_gate
             -> team_embed -> role_team -> jersey_number_detect -> traj_refine
             -> tracklet_agg -> audit
```

The bbox detector is a Hydra defaults-group choice between two YOLO11-L fine-tunes with
an identical operating point (default `yolo_ultralytics_snft_hm`; switch with
`modules/bbox_detector=yolo_ultralytics_snft`) — one path through the pipeline, one
stage module, two weight variants **[verified]**.

Package name `sn-gamestate` 1.0.0, licence GPL-3.0 (repository code). BroadTrack (EVS)
sources/weights are licensed noncommercial-research with no redistribution and are never
committed here; the binary, its weights, and generated calibration JSONs must not be
published **[verified: setup script and config comments; the licence terms themselves
are as stated by the repository, not independently reviewed]**.

## 2. Execution environments (three, by design) [verified]

| Environment | Interpreter | Used by | Provisioning |
|---|---|---|---|
| Main venv `.venv` | Python 3.9 (pinned `>=3.9,<3.10`), torch 1.13.1, numpy 1.26.4 | all pipeline stages except the two below | `uv venv --python 3.9 .venv && uv pip install --python .venv -e .` then `uv pip install --python .venv --no-deps boxmot==19.0.0` |
| Jersey venv `plugins/jn_gsr/.venv_jn` | Python 3.10, torch 2.0.1+cu118, mmcv 2.0.1 | `jersey_number_detect` (subprocess workers) | `bash scripts/setup_jn_gsr.sh` (needs the package vendored at `plugins/jn_gsr`, which it is) |
| BroadTrack native binary | C++ + libtorch 2.5.1+cu124 | `calibration` (subprocess) | `bash scripts/setup_broadtrack.sh` (clone EVS repo, apt deps, libtorch, cmake; no Docker) |

Critical environment facts **[verified from pyproject.toml / scripts]**:

* Python is pinned to 3.9 because torch 1.13.1 publishes no wheels for newer
  interpreters. Hosted images ship newer Pythons, so the venv must be built against a
  provisioned 3.9 (`uv python install 3.9`), never the image's own interpreter. Gate:
  `.venv/bin/python -c "import sys, torch; print(sys.version, torch.__version__, torch.cuda.is_available())"`
  must print `3.9.x`, `1.13.1`, `True` on GPU.
* `boxmot==19.0.0` is installed with `--no-deps` and is deliberately absent from
  `pyproject.toml` dependencies: its metadata requires torch>=2.2.1 and
  huggingface-hub>=1.7.1, which contradict the project pins. Its import chain only needs
  numpy, cv2, lap (`lapx`), scipy, rich, all already installed. Version matters: 19.x
  exposes `boxmot.trackers.botsort.botsort.BotSort`; 20+ moved it, and the 19.0.0 wheel
  misreports `__version__` as "18.0.0".
* Pins that exist to stop dependency drift: `setuptools<81` (pkg_resources removal vs
  torchmetrics 0.10.3), `albumentations<2` (top-level `functional` export vs the
  torchreid fork), `huggingface_hub>=0.23,<1.0` (HfFolder removal), `scikit-learn<1.7`.
* torchreid is the VlSomers/bpbreid fork (git dependency); it provides `osnet_ain_x1_0`
  (tracker/tracklet_split/traj_refine embedder) and `osnet_x1_0` (team model backbone).
* An installed-TrackLab patch is required and applied by the scripts, not by the package:
  `sed -i 's/gamestate-2025/gamestate-2024/g'` on
  `tracklab/wrappers/dataset/soccernet/soccernet_game_state.py`. A fresh environment
  without this patch requests the wrong dataset task name.
* Do not use `uv run` after installation (no lock covers the boxmot side-install; `uv run`
  re-resolves and was observed to swap 12 packages). Invoke `.venv/bin/python` /
  `.venv/bin/tracklab` directly.
* `scripts/preflight_imports.py` imports every `_target_` plus the runtime-only imports
  (boxmot, shared embedder, rules) in seconds and should be run before any long job.
  `preflight_cpu.sh` audits all artifacts (paths, sizes, checksums) and fetches only what
  is missing; `CHECK_ONLY=1` audits without downloading.

## 3. Stages, configs, parameters [verified from configs and module sources]

Config root: `sn_gamestate/configs/`; per-module files under `configs/modules/<stage>/`.
`project_dir` resolves to the launch directory (`${hydra:runtime.cwd}`), so `tracklab`
must be launched from the repository root; outputs go to `outputs/sn-gamestate/<date>/<time>/`.

| Stage | `_target_` (module) | Key configuration |
|---|---|---|
| `bbox_detector` | `sn_gamestate.bbox_detector.yolo_snft_api.YOLOUltralyticsSNFT` | Defaults-group switch between two YOLO11-L fine-tunes, same module and operating point (imgsz 1280, conf floor 0.1 kept with `>=`, iou 0.7, max_det 300, RGB→BGR fix; optional TensorRT, off by default). **Default (since 2026-09-03): `yolo_ultralytics_snft_hm`** — HF `${hf:Ynniss/YOLOv11L_HM,best.zip,yolov11l_hm_best.pt}` (the 3-arg resolver form copies the download under a .pt name). Alternative: `yolo_ultralytics_snft` — `${hf:Ynniss/sn-gamestate-weights,yolov11_sn_best.pt}` (the run-4 baseline detector). Distinct `engine_path` per variant; `build_trt_engines.py` builds only the snft engine (missing engine ⇒ warn + PyTorch fallback) |
| `track` | `sn_gamestate.track.bot_sort.BotSortSOF` | boxmot BotSort called directly; embeddings injected from the shared OSNet-AIN module; SOF camera motion (scale 0.15) computed outside; thresholds: high 0.3, low 0.05, new 0.4, match 0.85, proximity 0.5, appearance 0.35, buffer 60, frame_rate 25; fp16 autocast, fp32 outputs; per-frame audit sidecar `audit/track/` |
| `crop_filter` | `sn_gamestate.crop_filter.CropFilter` | single iff rT ≤ 0.25 and rB < 0.40, contaminators must carry a track_id (`contam_mode: tracked`); writes `crop_single/crop_rT/crop_rB/crop_trigger`; removes nothing |
| `tracklet_split` | `sn_gamestate.track.tracklet_split_api.TrackletSplit` | NEW 2026-09-04, replaces `split_merge`. Stage 1 of the refinement method, SPLIT ONLY — the pipeline's one merge is `traj_refine`, and the audit FAILS on any merge threshold or merging evidence here. Per tracklet, DBSCAN (eps 0.2, min_samples 5, precomputed cosine) over ALL detections (clean and multi); noise — single or multi crop — attaches to the nearest clean-only centroid; all-multi fragments dissolve per-detection into the nearest remaining fragment; degenerate cases deterministic (small/all-noise tracklet → one fragment; all-multi tracklet keeps its DBSCAN fragments). Fragments → trajectories 1..T; every tracked row stays assigned; incoming id snapshotted per row in `track_id_presplit` (single-source-origin audit check); validates the tracker invariant (one detection per tracklet per frame, raises on violation); same OSNet-AIN pin as `track` (audit-enforced); sidecar `audit/tracklet_split/`; algorithm `track/tracklet_split.py` (numpy+sklearn, 12 unit tests in `tests/test_tracklet_split.py`) |
| `calibration` | `sn_gamestate.calibration.broadtrack_api.BroadTrackCalibration` | BroadTrack binary at `pretrained_models/broadtrack/`; camera prior (0, 55, −12); `min_score 0.3` rejects lost frames and reuses the last accepted camera (`use_prev_parameters: true`, `max_carry_frames 0`); per-sequence JSON cache `broadtrack_calib/` (`use_cached_json: true`); writes human-bbox masks from own detections; `staging_dir` for read-only datasets; emits camera `parameters` and `bbox_pitch` |
| `pitch_gate` | `sn_gamestate.pitch_gate.PitchGate` | enabled, margin_m 3.5 (untuned); off-pitch iff |mean_x| > 52.5+m or |mean_y| > 34+m on the tracklet mean of finite `bbox_pitch`; gated tracklets: `track_id` → NaN, original kept in `track_id_pregate`; no row deleted; sidecar `audit/pitch_gate/` |
| `team_embed` | `sn_gamestate.team.TeamEmbedding` | osnet_team (OSNet x1.0, 128×64, 256-d) from HF `Ynniss/osnet_team/osnet_team_best.pt`; ≤ 16 crops/tracklet on the stride-5 grid; fp32 + flip TTA; `team_sha256` currently **null** (recorded, not enforced — pin after first verified run); sidecar `audit/team_embed/` |
| `role_team` | `sn_gamestate.team.RoleTeamAssignment` | notebook rule chain (`team/rules.py`): role player/goalkeeper/referee, team k-means on single-crop descriptors, side; frozen params (k 3.25, tau_n 5, max_ref 3, side_rule keeper, …), untuned on this pipeline's tracklets/BroadTrack projections; sidecar `audit/role_team/` |
| `jersey_number_detect` | `sn_gamestate.jersey.jn_gsr_api.JNGsrTrackletRecognizer` | subprocess workers in the 3.10 venv; roles [player, goalkeeper]; `single_crops_only: true`; legibility > 0.72 → DBNet++ ROI → PARSeq + SATRN → `vote_pool` (the only rule); stride 5; fp16; GPU sharding auto via nvidia-smi (2 workers on Kaggle 2×T4); content-hash cache `jn_cache/`. Since 2026-09-03 (blob **schema 2**): two ADDITIVE columns for `traj_refine` — `jersey_number_candidates` (every pooled label of the two recognisers as `[label, mx, conf_sum, votes]`, ranked by the maxconf score exp(mx)·conf_sum; stats, not scores, so merged tracklets recombine exactly: mx=max, conf_sum/votes add) and `jersey_number_maxconf` (assigned number's score). The schema is folded into the cache key (old caches miss and recompute once) and checked on every shard and cached blob; the assigned number stays `vote_pool`, byte-identical |
| `traj_refine` | `sn_gamestate.refine.traj_refine_api.TrajRefine` | NEW 2026-09-03, extended 2026-09-04. The pipeline's ONE merge (Stage 2) plus stage-3 duplicate-frame resolution, between jersey and the vote: phase 2a merges in-scope (player/goalkeeper) cluster pairs with equal team AND equal number, in descending JOINT pooled maxconf (exp(max mx)·Σconf_sum over the pair's clean-detection stats — the maxconf the merged cluster would carry; changed 2026-09-03 from the sum of the two separate scores), when CLEAN frame sets are disjoint ∧ re-enter consistent ∧ distance ≤ tau; a pair claiming one number at the same time (clean-frame overlap) is a conflict — the lower-maxconf side walks to its best candidate not previously lost (banned set; cascades walk down the list; "-1"/exhaustion → unnumbered); phase 2b merges agglomeratively (average linkage, the splitter's distance convention) under clean-frames-disjoint ∧ re-enter ∧ vacuous-when-unknown team/number agreement. Multi-player detections are ignored until stage 3 (no part in centroids, distances or disjointness; re-enter endpoints are clean-anchored). Stage 3, in-scope clusters only, after the merger: 3a keeps one detection per (frame, trajectory) — clean wins (a second clean is a counted anomaly), else the multi nearest the clean-first centroid — and holds the rest; centroids are then recomputed over ALL remaining detections (3b's metric, fixed); 3b places held detections in ascending distance to the nearest in-scope trajectory with that frame free (occupancy grows; no distance cap by specification) and unassigns the rest (`track_id` NaN — the ONLY way any refinement stage drops a row). Same OSNet-AIN pin as track/tracklet_split (audit-enforced; tau 0.60 inherited from the retired split_merge operating point, untuned for this stage; edge_margin 0.02 untuned). Role is NOT a merge condition (merged rows → row-majority role); referees untouched; no-centroid trajectories never merge; rows adopted in 3b take the target cluster's labels. Snapshots written unconditionally: `track_id_prerefine`, `jersey_number_detection_prerefine`, `jersey_number_confidence_prerefine`; `enabled: false` = snapshots + sidecar only (the A/B switch). Output invariant: one detection per (image_id, track_id) over ALL detections; tracked rows out = in − unassigned. Sidecar `audit/traj_refine/` (incl. `stage3` block and `rows_unassigned`); algorithm in `refine/traj_refine.py` (pure numpy, 22 unit tests in `tests/test_traj_refine.py`) |
| `tracklet_agg` | `tracklab.wrappers.MajorityVoteTracklet` | majority vote over `[jersey_number]` only (role is per-tracklet from role_team) |
| `audit` | `sn_gamestate.audit.RunAudit` | read-only last stage; per-sequence, per-component PASS/WARN/FAIL to `audit/<seq>.json`; cross-checks every sidecar against the composed config (including track vs tracklet_split checkpoint-pin equality); `scripts/verify_run_integrity.py` exits non-zero on any FAIL. Since 2026-09-04: `_check_tracklet_split` replaces `_check_split_merge` (sidecar present; eps/min_samples/digest ran == configured; FAIL on any merge threshold or merge/pass evidence anywhere; per-tracklet fragment counts sum to the total; every tracked row assigned — the splitter never drops or merges; fragments == trajectories in the state; no (image_id, track_id) collision; every fragment from exactly one source tracklet via `track_id_presplit`; fragments-without-clean recomputed == sidecar, a consistency check, not a standalone FAIL). The `traj_refine` check now does row accounting (rows losing an id == sidecar `rows_unassigned`, no row may gain one, tracked_after == tracked_before − unassigned) with the pin-equality key `ain_sha256_tracklet_split`; the pitch_gate / team_embed / role_team / jersey checks audit against the pre-refine snapshots (`track_id_prerefine`, jersey snapshot columns) so every earlier stage is held to the state it actually produced |

Not in the pipeline **[verified]**: no `pitch` stage (BroadTrack runs NBJW keypoints and
TVCalib lines internally); no `reid`/prtreid stage; `interpolation` (dti.yaml) exists as a
module with `enabled: false` and is excluded from `pipeline:` because synthesized rows
carry no crop labels or team embeddings.

Precision policy **[verified]**: fp16 baked in for the detector (ultralytics half) and the
OSNet-AIN embedder (autocast, fp32 outputs); team_embed is fp32 + flip TTA; BroadTrack and
the jersey venv are out of scope by construction. Reported fp16-vs-fp32 validation numbers
(HOTA 71.61 vs 71.08, GS-HOTA 64.98 vs 64.70) are repository claims from a prior run
**[unverified here]**.

## 4. External artifacts: exactly where every download comes from

All verified against the fetching code, not the README.

| Artifact | Primary source | Fallback | Integrity |
|---|---|---|---|
| Dataset (SoccerNetGS splits) | SoccerNet server (KAUST) via `SoccerNet` pip package, task `gamestate-2024` | **Hugging Face dataset `SoccerNet/SN-GSR-2024`** (`<split>.zip` at repo root; train 9.76 GB, valid 11.2 GB, test 8.85 GB, challenge 5.31 GB) — automatic on server error/no response/truncated zip (added 2026-09-02) | zip central-directory check triggers the fallback; sequence-count and `img1`-depth checks in `preflight_cpu.sh` |
| Detector `yolov11_sn_best.pt` | HF `Ynniss/sn-gamestate-weights` via the `${hf:...}` OmegaConf resolver at config-resolution time | none | none (no digest pin) |
| Detector `best.zip` (YOLOv11L_HM, default since 2026-09-03) | HF `Ynniss/YOLOv11L_HM` via the 3-arg `${hf:...}` form (download copied to `yolov11l_hm_best.pt` beside the cache entry) | none | none (no digest pin, matching the other detector). The .zip IS the torch checkpoint **[verified on Kaggle run 5, 2026-09-03: resolver downloaded it, ultralytics loaded it under torch 1.13.1, bbox_detector audit PASS]**; ~48.5 MB fp16 ultralytics YOLO11 DetectionModel |
| OSNet-AIN `best_ain_full.zip` (tracker + tracklet_split + traj_refine) | HF `Ynniss/osnet_ain`, revision `d78f65de…`, the packed file is itself a torch.save archive | exploded snapshot `Ynniss/osnet_ain_ckp` path exists in code (used when `ain_file` is cleared); `ain_local_path` reads from disk | **sha256 enforced** (`a0a7e426…`); audit fails on any pin mismatch among the three stages |
| Team model `osnet_team_best.pt` | HF `Ynniss/osnet_team` | `team_local_path` | sha256 **recorded only** (`team_sha256: null`) — pin after first verified run |
| BroadTrack source code | `github.com/evs-broadcast/BroadTrack` git clone, 3 attempts, no credential prompts (LFS smudge off) | GitHub codeload tarball (`main` verified live, then `master`); optional third path `BT_SRC_FALLBACK_REPO` = private HF repo with `broadtrack_src.tar.gz` (added 2026-09-02 after Kaggle observed GitHub refusing anonymous git-over-HTTPS transiently) | `CMakeLists.txt` presence; tarball trees carry `SOURCE_SNAPSHOT.txt` |
| BroadTrack weights `nbjw_keypoint_model`, `tvcalib_model` (~230–266 MB TorchScript each) | EVS git-lfs (`git lfs pull`) | **automatic HF fallback** to `BT_WEIGHTS_FALLBACK_REPO` (default `Ynniss/calibiration_weights`, .zip torch.jit containers staged by copy) when the pull fails or leaves pointer stubs (added 2026-09-02); manual overrides: `BT_WEIGHTS_REPO` (skip EVS), `BT_WEIGHTS_DIR` (local/Kaggle dataset) | size floor 1 MB; a zip that is a plain archive-of-files is rejected |
| libtorch 2.5.1+cu124 | download.pytorch.org | `LIBTORCH_URL` override | none |
| DBNet++ `best_icdar_hmean_epoch_10.pth` (111 MB) | HF `Ynniss/dbnetppp_jn` | `--dbnet-attached` (Kaggle dataset) | sha256 checked, mismatch reported not fatal |
| Legibility ResNet-34 → `sn_legibility.pth` (85 MB) | HF `Ynniss/Legibility_classifier` | `--legibility-attached` | sha256 checked, mismatch reported not fatal |
| SATRN `best_recog_word_acc_epoch_10.zip` → `recog2/…​.pth` (48 MB; the .zip IS the torch checkpoint, staged by copy) | HF `Ynniss/satrn_small` | `--satrn-attached` | only a 16-hex sha256 **prefix** on record (`9e8f73b300754c35`) |
| PARSeq `parseq_gsr_ft_s1.zip` → `.ckpt` (286 MB; .zip IS the checkpoint, rename not unzip) | HF `Ynniss/final_parseq_jn` (after `--attached`) | none by design (old Drive mirror removed: it served superseded weights) | sha256 reported (`22d93644…`); `audit_parseq.py --stages d1,d2` is the asserting arm + strhub load gate |

Points that contradict a casual reading of the README **[verified]**:

* **Calibration weights are not downloaded from HF by default.** The default path is
  EVS's git-lfs; HF is the fallback/override (`Ynniss/calibiration_weights`). For a Kaggle
  run that must take all weights from HF, either rely on the automatic fallback or set
  `BT_WEIGHTS_REPO=Ynniss/calibiration_weights` explicitly (deterministic, recommended for
  the test run).
* The staged PARSeq path must contain `parseq` and none of `abinet/crnn/trba/trbc/vitstr`
  anywhere in the absolute path (strhub routes the model class on the path string).
* `sports_model.pth.tar-60` was checked/fetched by `preflight_cpu.sh` although nothing
  references it; removed 2026-09-02 after the end-to-end Kaggle run confirmed it unused.
  **[verified]**
* `HF_TOKEN` must be exported if any `Ynniss/*` repo is private; internet must be ON in
  Kaggle notebook settings.

## 5. Dataset expectations [verified]

Layout: `data/SoccerNetGS/<split>/SNGS-*/img1/*.jpg` plus `SNGS-*/Labels-GameState.json`
per sequence (the zips extract with `unzip -o <zip> -d data/SoccerNetGS/<split>`).
Expected sequence counts (from `preflight_cpu.sh`): train 57, valid 59, test 49. Default
run scope: `dataset.nvid: 1`, `eval_set: test`; `dataset.nvid=-1` for the full split;
`dataset.vids_dict.test: ['SNGS-116']`-style pinning selects specific clips — this is the
mechanism for the one-sequence Kaggle test. Server zip layout **[verified on Kaggle]**:
members are `SNGS-XXX/...` at the zip root (36898 members, 49 sequences in `test.zip`).
The HF mirror's internal layout is **[unverified]** (the fallback never triggered); it is
published by the same team and expected identical, and the preflight's `img1`-depth check
would catch a mismatch.

Policy encoded in the repo: `valid` is for tuning sweeps; `test` is reserved for the
frozen production run.

## 6. Running and verifying [verified]

```
tracklab -cn soccernet                       # 1 clip (dataset.nvid=1), test split, HM detector
tracklab -cn soccernet modules/bbox_detector=yolo_ultralytics_snft   # the run-4 baseline detector
tracklab -cn soccernet modules.traj_refine.cfg.enabled=false         # refine stage passthrough (A/B)
tracklab -cn soccernet dataset.nvid=-1       # full split
bash scripts/lightning_eval.sh               # end-to-end: venv + patch + data + setup + eval
python scripts/preflight_imports.py          # seconds; every stage importable (incl. traj_refine)
CHECK_ONLY=1 bash preflight_cpu.sh           # artifact audit only (audits BOTH detector weights)
python scripts/verify_run_integrity.py --expect-sequences <N>   # non-zero on any audit FAIL
python scripts/verify_broadtrack_conversion.py -c broadtrack_calib/<seq>.json -f data/.../img1 -o broadtrack_check
python scripts/reference_metrics.py --state outputs/sn-gamestate/<date>/<time>/states/sn-gamestate.pklz --dataset-path data/SoccerNetGS --eval-set test --out reference_metrics
```

A ready-to-run Kaggle notebook for the one-sequence test (full pipeline incl.
`traj_refine`, the verification chain, and optional detector / refine A/B runs) is at
`docs/kaggle_one_sequence_test.ipynb` (linked from `docs/KAGGLE_GUIDE.md`) **[verified:
nbformat-valid, every bash cell syntax-checked; not yet executed on Kaggle]**.

Reference metrics blocks: tracking (HOTA/DetA/AssA/MOTA/IDF1/IDSW), gsr (GS-HOTA family,
5 m tolerance), jersey_number (Hungarian IoU ≥ 0.5, unnumbered = −1), calibration (JaC5,
JaC10, MRE, MedRE, CR at 960×540). Multiple `--state/--label` pairs produce a comparison
table — the mechanism for per-stage attribution (e.g. with/without `pitch_gate` via
`modules.pitch_gate.cfg.enabled=false`).

Caches that make re-runs cheap: `broadtrack_calib/<seq>.json` (calibration),
`jn_cache/` (jersey, keyed on manifest content + thresholds + both checkpoint digests),
the saved state `outputs/sn-gamestate/<date>/<time>/states/sn-gamestate.pklz`
(`state.load_file` + a truncated `pipeline:` re-runs late stages only), and the
huggingface_hub disk cache.

Kaggle specifics **[verified from scripts/comments]**: root shell with CUDA is available
(no Docker needed); `JN_VENV=/kaggle/tmp/.venv_jn` puts the ~7 GB jersey venv on scratch;
`modules.calibration.cfg.staging_dir` handles the read-only mounted dataset;
`num_cores=0` avoids the cuDNN/torch_shm_manager multiprocessing issue; jersey workers
auto-shard to 2 on 2×T4; `MPLBACKEND` is forced off the inline backend where needed;
`echo "n" |` is piped into `tracklab` by `lightning_eval.sh` (answers an interactive
prompt). `ain_local_path` / `team_local_path` / `BT_WEIGHTS_DIR` / `--*-attached` all
exist so weights can be mounted as Kaggle datasets instead of downloaded per session.

## 7. Known caveats and open items

Stated by the repository itself and confirmed as real caveats **[verified as claims in
code/config comments]**:

1. The DBSCAN operating point (`tracklet_split` eps/min_samples) and `traj_refine`'s
   tau were tuned on embeddings from a notebook export whose producing
   checkpoint cannot be verified from this repository; if it was not the pinned
   OSNet-AIN, they do not transfer and need re-tuning on `valid`.
2. `pitch_gate.margin_m` (3.5), `crop_filter` thresholds, `role_team` params, and
   `interpolation` n_dti/n_min are untuned on this exact pipeline.
3. `team_sha256` and `team_revision` are unset; pin them after the first verified run.
4. SATRN has only a sha256 prefix on record; replace with a full digest once read off a
   verified download.
5. Weight-hash mismatches in the jersey fetchers are reported, not fatal (private
   fine-tunes are legitimate); `audit_parseq.py` and the run audit are the asserting arms.
6. Jersey-stage quality numbers in configs are for ground-truth-box tracklets; no measured
   row exists yet for predicted tracklets through this pipeline.
7. UPDATED 2026-09-04: `traj_refine` `tau` (0.60) is now the pipeline's ONLY merge
   threshold (the splitter never merges). It is inherited from the retired split_merge
   stage's notebook-validated operating point, carried over because the stage uses the
   same embedder pin and distance convention; it is NOT tuned for this stage, and
   `edge_margin` (0.02) is untuned. Tune on `valid` if the Kaggle test shows
   over/under-merging.
8. NEW 2026-09-03: the calibration cache `broadtrack_calib/<seq>.json` is keyed by
   sequence name only, and BroadTrack's player masks come from our own detections — a
   cached JSON silently carries the detector that produced it. Reusing it across
   detector variants is deliberate for A/B comparisons (same camera removes a
   confound); a from-scratch production run under a new detector should clear
   `broadtrack_calib/` or override `modules.calibration.cfg.calib_dir`. No audit check
   catches this staleness.
9. NEW 2026-09-03, updated after run 5: run-4 baseline numbers (in-run GS-HOTA 58.4–59.4
   across two sessions on the single sequence test/SNGS-116) were produced with the snft
   detector and WITHOUT `traj_refine`. Run 5 (HM default + traj_refine) scored in-run
   GS-HOTA 55.07 on the same sequence — and since `traj_refine` was a no-op there (see
   run-5 findings below), the drop is attributable to the detector change plus its fresh
   calibration. Evidence is one sequence; **the HM-vs-snft default decision is pending
   the confound-free A/B** (notebook flag `RUN_DET_AB=1`, which reruns snft against the
   session's HM baseline on a shared calibration cache).
10. NEW 2026-09-03: 2a conflicts among clusters that become same-numbered only through
    2b inheritance are not re-resolved (2a runs once before 2b, faithful to the
    method's specification; documented in `refine/traj_refine.py`).

Discrepancies found in this analysis:

11. RESOLVED 2026-09-02: `preflight_cpu.sh` no longer checks/fetches
    `sports_model.pth.tar-60` (was referenced nowhere; confirmed unused by the successful
    end-to-end Kaggle run).
12. RESOLVED 2026-09-02: the README reference-metrics command now locates the newest
    state under `outputs/sn-gamestate/*/*/states/` (TrackLab saves it inside the Hydra run
    directory, confirmed on Kaggle; the old `states/...` root-relative path never exists).
13. NEW 2026-09-04: per-fragment jersey voting. `tracklet_split` emits fragments as the
    tracklets every later stage sees (run-6 log on SNGS-116: 52 tracklets → 67
    fragments), so jersey recognition votes over fewer frames per tracklet (noisier
    votes, potentially more unnumbered tracklets) and processes more tracklets (longer
    jersey worker time). The jersey cache keys on tracklet content, so the first
    post-conformance run recomputes the entire jersey cache. Fragment-level label
    quality on real data is unmeasured.
14. NEW 2026-09-04: stage 3b has no distance cap, by specification — a held
    multi-player detection is placed into the nearest in-scope trajectory whose frame
    is free even at cosine distance ≈ 1; only the absence of any admissible slot
    unassigns it. Verified as spec-faithful in the offline harness; its effect on
    association metrics is unmeasured.
15. NEW 2026-09-04: all pre-conformance metrics (runs 4–6, §7) are NOT comparable to
    post-conformance runs. The retired split_merge stage merged label-blind at the same
    tau before team/jersey labels existed (run 6: 39 merges before `traj_refine` saw the
    sequence, leaving it a no-op); the conformant pipeline defers ALL merging to
    `traj_refine`. Baselines must be re-established.
16. RESOLVED 2026-09-04: the split-vs-merge architecture deviation. The retired
    `split_merge` stage merged fragments on appearance alone, before team and jersey
    evidence existed — contrary to the method, in which Stage 1 only splits and the one
    merge is the label-aware Stage 2. Resolved by the split-only restructuring (§8
    batch 5): `tracklet_split` splits and never merges (audit-enforced), `traj_refine`
    holds the pipeline's one merge plus stage-3 duplicate-frame resolution.

Kaggle findings so far (runs 1-3, 2026-09-02): environment recipe, TrackLab patch, and
all 22 stage imports work on the current image (host Python 3.12.13, provisioned 3.9.25);
KAUST served `test.zip` in all three runs (2.8-15.1 MiB/s, so the HF dataset fallback is
justified but untriggered); zip layout is `SNGS-XXX/...` at the root (no split prefix; §5); BroadTrack builds and smoke-tests natively on the
image with weights staged from HF; the jersey stage provisions fully (all four checkpoint
digests verified, `audit_parseq` d1+d2 PASS). Kaggle constraints learned: `/kaggle/working`
is a 20 GB device (dataset zip and `BT_ROOT` belong on `/kaggle/tmp`, with the five
calibration path overrides at run time); Jupyter's `MPLBACKEND` inline backend leaks into
`%%bash` and kills the 3.9 venv's matplotlib import -- `export MPLBACKEND=Agg` for every
venv invocation; GitHub transiently refuses anonymous git-over-HTTPS from shared Kaggle
IPs (hence the source-acquisition hardening above).

Run 4 (2026-09-02, T4 x2): the FULL pipeline completed on test/SNGS-116, exit 0, and
`verify_run_integrity.py` reported RUN INTEGRITY: OK (1 sequence, 0 FAIL, 0 WARN;
1 calibration JSON; 2 jersey cache entries; PARSeq matches upstream). In-run evaluation on
that single sequence: GS-HOTA 58.375 (pitch space, jerseys+teams+roles, 5 m tolerance),
HOTA 58.375, DetA 50.014, AssA 68.136, MOTA 38.393, IDF1 67.818, IDSW 0, MT 13 / PT 8 /
ML 5 of 26 GT ids. A second full session reproduced the run
(in-run GS-HOTA 59.447) and completed the whole verification chain: reference metrics
(tracking HOTA 64.85 / MOTA 86.90 / IDF1 80.57 image-space; GS-HOTA 59.45 pitch; jersey
F1 0.806, trk_acc 0.857; calibration JaC5 0.481, MRE 4.74 px, CR 1.0 over 749/750 frames)
and the BroadTrack conversion check (model equivalence 3.17e-05 px, roundtrip 6.37e-06 m,
both PASS). Single-sequence numbers are not comparable to full-split figures. Full-split
behavior and the HF dataset fallback network path remain unexercised.

Run 5 (2026-09-03, T4 x2, via `docs/kaggle_one_sequence_test.ipynb`, first exercise of
the 2026-09-03 batch): full pipeline on test/SNGS-116 with the HM detector default and
`traj_refine` enabled, exit 0; RUN INTEGRITY: OK — 13 checks PASS, 0 FAIL, 0 WARN,
including the new `traj_refine` check. HM weights loaded through the 3-arg resolver
(`best.zip` → `yolov11l_hm_best.pt`); jersey stage computed fresh under blob schema 2
(2 workers, 15/21 tracklets numbered); fresh calibration from HM detections (CR 749/750,
mean score 0.577 vs 0.647 under snft; conversion checks PASS at 3.70e-05 px /
3.27e-06 m). In-run eval (pitch, attributes on): GS-HOTA 55.07, DetA 46.74, AssA 64.06,
MOTA 30.21, IDF1 64.49, IDSW 0, MT 8 / PT 11 / ML 7 of 26 GT ids, 10200 predicted
detections vs 10567 GT — below the snft baseline across the board (§7 item 9).
`traj_refine` was a NO-OP on this sequence: 23 tracklets in, 23 out, 0 merges,
0 conflicts, 0 rejections, 0 rows relabelled (21 in scope, 15 numbered, all embeddings
valid, img_w read) — split_merge left nothing for it to act on here, so the stage's
merge/conflict behavior on real data remains unmeasured. Both optional A/B cells were
skipped (flags 0). The reference-metrics tables went to the session's
`reference_metrics/summary.md`, not into the notebook output.

Run 6 (2026-09-04 03-33-37, T4 x2, same configuration as run 5 — HM detector,
`traj_refine` enabled, pre-conformance pipeline with `split_merge` still in place): full
pipeline on test/SNGS-116, exit 0, RUN INTEGRITY OK; calibration CR 749/750, mean score
0.583; `traj_refine` again a NO-OP (23 tracklets in, 23 out; inputs 12634 detections /
10361 tracked / 21 in scope / 15 numbered / 21 with team / img_w 1920 / 0 zero
embeddings). The decisive log line: `[split_merge] SNGS-116: 52 tracklets -> 67
fragments (11 split) -> 23 trajectories (39 merges)` — split_merge merged 67 fragments
down to 23 label-blind at the same tau BEFORE team/jersey evidence existed, leaving
`traj_refine` nothing to merge. This run is the direct motivation for the 2026-09-04
split-only restructuring (§7 item 16, §8 batch 5). The in-run evaluation tables were
not in the captured stdout (they went to the session's `reference_metrics/summary.md`,
not retrieved).

Still **[unverified]** after run 6 and the 2026-09-04 restructuring: full-split behavior and
runtime; the HF dataset fallback's actual network path (code path tested synthetically,
never against a live server failure; KAUST served `test.zip` again in run 5 at
~15 MiB/s); the EVS git-lfs weights path and its automatic
fallback (runs pinned `BT_WEIGHTS_REPO`); the HF mirror zip's internal layout (§5); the
line-by-line behavior of files verified only at the config/contract level
(`broadtrack_api.py`, the jersey worker); the ENTIRE conformant pipeline live —
`tracklet_split` (12 unit tests + a stubbed `process()` harness, never run in the
pipeline), the `traj_refine` merge/conflict paths and stage 3 on real sequences (22 unit
tests + a stubbed `process()` harness; the pre-conformance stage ran live in runs 5–6
but had nothing to merge), per-fragment jersey quality and runtime (§7 item 13); and the
snft-vs-HM A/B on a shared calibration (the pending detector-default decision, to be
re-based on post-conformance baselines per §7 item 15).

## 8. Changes made on 2026-09-02, 2026-09-03 and 2026-09-04

Batch 1 (pushed before the Kaggle test; presence in the clone verified by the notebook):

1. `scripts/lightning_eval.sh` — dataset download: SoccerNet server primary, automatic
   fallback to HF `SoccerNet/SN-GSR-2024` on failure/no response/truncated zip; the HF
   zip is unzipped directly from the huggingface_hub cache (no duplicate copy on disk).
2. `preflight_cpu.sh` — same dataset fallback in its fetch phase (non-fatal style,
   matching the script's audit/repair design).
3. `scripts/setup_broadtrack.sh` — BroadTrack weights: EVS git-lfs primary; automatic
   per-file fallback to `BT_WEIGHTS_FALLBACK_REPO` (default `Ynniss/calibiration_weights`)
   when the pull fails or leaves pointer stubs; `BT_WEIGHTS_REPO` / `BT_WEIGHTS_DIR`
   overrides unchanged. Then, after Kaggle run 3 hit GitHub refusing anonymous
   git-over-HTTPS: source-code acquisition hardened (3 clone attempts with
   `GIT_TERMINAL_PROMPT=0`, codeload-tarball fallback `main`→`master`, optional
   `BT_SRC_FALLBACK_REPO` private HF snapshot).
4. `README.md` — "Download fallbacks" section covering all three behaviors.

Batch 2 (after the test passed; in the local tree, pending the final push):

5. `docs/KAGGLE_GUIDE.md` (new) — verified Kaggle procedure, error guide, timings, and
   the one-sequence reference results; linked from the README.
6. `README.md` — reference-metrics `--state` corrected to locate the newest state under
   `outputs/sn-gamestate/*/*/states/`; link to the Kaggle guide.
7. `preflight_cpu.sh` — unreferenced `sports_model.pth.tar-60` removed from the audit
   and fetch phases.
8. `docs/PIPELINE_REFERENCE.md` (new) — this document, as the repository's canonical
   ground-truth reference.

Batch 3 (2026-09-03, pushed the same day — presence in the clone verified by the run-5
notebook check; all verified offline before the push: unit tests,
plugin self-tests, py_compile, YAML parses, byte-compares of installed copies):

9. `traj_refine` stage: `sn_gamestate/refine/` (new package: `traj_refine.py` algorithm,
   `traj_refine_api.py` stage, `__init__.py`), `configs/modules/traj_refine/
   traj_refine.yaml` (same OSNet-AIN pin as split_merge; tau 0.60, use_reenter true,
   edge_margin 0.02, roles [player, goalkeeper], enabled true), `tests/
   test_traj_refine.py` (18 tests, all passing), wired into `soccernet.yaml`
   (defaults + pipeline between `jersey_number_detect` and `tracklet_agg`) and
   `scripts/preflight_imports.py`.
10. Jersey candidate output, blob **schema 2**: `plugins/jn_gsr/fuse_jn.py`
    (+`pooled_label_stats`, `ranked_candidates`), `jn_recognizer.py` (+`consolidate_full`,
    `predict_full`; `predict` unchanged, delegates), `predict_tracklets.py` (candidates in
    shard results, schema in blob, stub updated); `sn_gamestate/jersey/jn_gsr_api.py`
    (+`jersey_number_candidates`, `jersey_number_maxconf` columns; schema in the cache
    key and enforced on shards and cached blobs). All plugin self-tests pass;
    `plugins/jn_gsr/MANIFEST.sha256` entries for the three edited files regenerated
    (provenance record; consumed by no script — verified).
11. Audit extension: `sn_gamestate/audit/run_audit_api.py` (+`_check_traj_refine`;
    pitch_gate/team_embed/role_team/jersey checks now audit against the pre-refine
    snapshots), `configs/modules/audit/run_audit.yaml` (+`traj_refine_sidecar_dir`,
    `expected_traj_refine` incl. pin equality vs split_merge).
12. Second detector + switch: `configs/modules/bbox_detector/yolo_ultralytics_snft_hm.yaml`
    (new; HF `Ynniss/YOLOv11L_HM` `best.zip`, identical operating point);
    `preflight_cpu.sh` audits/fetches both detector weights; `soccernet.yaml` default
    flipped to `yolo_ultralytics_snft_hm` (the earlier detector stays selectable:
    `modules/bbox_detector=yolo_ultralytics_snft`).
13. `docs/kaggle_one_sequence_test.ipynb` (new) — ready-to-run notebook implementing the
    guide's one-sequence recipe end to end (fail-fast clone check for the pushed files,
    environment gate, single-sequence extraction, full run with the guide's overrides,
    verification chain, traj_refine sidecar summary, optional detector / refine A/B
    cells); linked from `docs/KAGGLE_GUIDE.md`.
14. This document updated to the 2026-09-03 state (§§1, 3, 4, 6, 7, 9).

Batch 4 (2026-09-03, after run 5; in the local tree, pending push):

15. `docs/kaggle_one_sequence_test.ipynb` — detector A/B cell repaired: after the
    default flip to HM it selected `yolo_ultralytics_snft_hm` (the default — comparing
    HM to itself). Flag renamed `RUN_HM_AB` → `RUN_DET_AB`; the cell now selects
    `modules/bbox_detector=yolo_ultralytics_snft`, labels `hm` (session baseline) vs
    `snft`, log/output names updated. Revalidated (nbformat schema, `bash -n` per cell).
16. This document updated with the run-5 findings (§§ header, 4, 7, 8, 9).
17. 2026-09-03, algorithm amendment (approved): phase-2a pair ordering changed from the
    sum of the two fragments' maxconf scores to the JOINT pooled maxconf
    `exp(max(mx_F, mx_G))·(conf_sum_F + conf_sum_G)` — the value the merged cluster
    carries, recomputed from the pooled clean-detection statistics (`pair_maxconf`
    helper in `refine/traj_refine.py`; sidecar field `pair_score` → `pair_maxconf`).
    Conflict resolution, merge conditions and 2b unchanged. Tests: 19 (new
    joint-vs-sum ordering case where the two orders differ + helper identity vs
    `combine_cand`); audit unaffected (it never read the ordering field).

Batch 5 (2026-09-04, split-only conformance — approved restructuring to the locked
method specification; in the local tree, pending push; motivated by the run-6 finding,
§7 items 15–16):

18. `tracklet_split` stage (new), replacing `split_merge`:
    `sn_gamestate/track/tracklet_split.py` (split-only algorithm: DBSCAN over ALL
    detections per tracklet, noise → nearest clean-only centroid, per-detection
    dissolution of all-multi fragments, deterministic degenerate cases, tracker-invariant
    validation with fail-fast raise), `tracklet_split_api.py` (stage: same OSNet-AIN
    extraction as the tracker, fragments → trajectories 1..T, `track_id_presplit`
    snapshot, split-only sidecar), `configs/modules/tracklet_split/tracklet_split.yaml`
    (eps 0.2, min_samples 5, same AIN pin, deliberately NO tau),
    `tests/test_tracklet_split.py` (12 tests). `split_merge` retired: its five files
    (`split_merge.py`, `split_merge_api.py`, `split_merge.yaml`, `test_split_merge.py`,
    `notebook_split_merge_reference.py`) moved to `_deleted_pending_git_rm/` with a
    README — run `git rm -r _deleted_pending_git_rm` and remove the empty
    `configs/modules/split_merge/` directory before committing (the editing tools
    cannot delete files).
19. `traj_refine` extended to the locked specification: 2a/2b temporal disjointness now
    over CLEAN frames only (multi-player detections ignored until stage 3; re-enter
    endpoints clean-anchored, fallback to all rows for clean-less fragments); stage 3
    added inside the stage after the merger (3a keep-clean / nearest-multi with a
    counted clean anomaly, all-remaining-detections centroid recompute, 3b greedy
    ascending-distance placement into the nearest in-scope free slot with deterministic
    ties, unassignment when no slot is admissible); driver `-2` bookkeeping sentinel and
    a final all-frames (frame, cluster) collision assertion. `traj_refine_api.py`:
    unassigned rows → `track_id` NaN, rows adopted in 3b take the target cluster's
    labels, self-check `tracked_out == tracked_in − unassigned`, sidecar `stage3` block
    and `rows_unassigned`, log line extended. `tests/test_traj_refine.py`: 22 tests
    (invariant helper skips unassigned rows; three new stage-3 cases).
20. Audit rework: `_check_tracklet_split` replaces `_check_split_merge` (split-only
    assertions — FAIL on any merge threshold or merge/pass evidence, on dropped rows,
    on multi-origin fragments via `track_id_presplit`; fragments-without-clean is a
    recomputed-vs-sidecar consistency check); `_check_traj_refine` row accounting
    (`rows_losing_id == rows_unassigned`, no gained ids, tracked_after == before −
    unassigned) and pin key `ain_sha256_tracklet_split`; tracker-internals pin trio
    renamed; thresholds: `tracklet_split_zero_emb_warn` replaces the two split_merge
    keys. `run_audit.yaml`: `tracklet_split_sidecar_dir`, `expected_tracklet_split`
    (eps, min_samples, ain_sha256 — NO tau), interpolations repointed, header comments
    rewritten.
21. Wiring and reference sweep: `soccernet.yaml` (defaults + pipeline →
    `tracklet_split`, comments), `traj_refine.yaml` comments (tau = the pipeline's only
    merge threshold, inherited from the retired split_merge operating point),
    `scripts/preflight_imports.py` (target → `tracklet_split.split_video`),
    `track/__init__.py`, `botsort_ain.yaml` comment, `tests/README.md`, root `README.md`
    (diagram, stage table incl. a previously missing `traj_refine` row, stage section,
    precision/audit/tuning/layout paragraphs), comment-level references in
    `audit_pipeline_columns.py`, `lightning_eval.sh`, `reference_metrics.py`,
    `build_trt_engines.py`, `inspect_ain_checkpoint.py`;
    `docs/kaggle_one_sequence_test.ipynb` cell-3 fail-fast list extended with the three
    tracklet_split files and a `tracklet_split` grep on `soccernet.yaml`.
22. Verification performed (batch 5): every algorithm/stage/audit file developed and
    tested in a sandbox first; the seven installed Python files byte-identical (sha256)
    to the sandbox-verified copies; both suites re-run from repo copies (22 + 12,
    all passing); every touched Python file compiles; the four YAMLs parsed with
    structural assertions (pipeline order, defaults, expected blocks, no tau on the
    splitter); notebook JSON validated (fail-fast list + zero split_merge strings);
    stubbed `process()` harnesses executed both stage wrappers end to end offline
    (splitter: two-identity split, single-origin fragments, sidecar consistency;
    refiner: 2b merge on clean disjointness, 3a keep-clean, 3b place-when-free and
    unassign-when-blocked, label propagation onto the adopted row, snapshots, row
    accounting); repo-wide reference sweep — the only remaining `split_merge` strings
    in live files are historical-provenance comments marked "retired".

Verification performed per edit (2026-09-02 batches): `bash -n` on every touched script; the zip-validity
fallback trigger exercised against missing/empty/truncated/valid zips (all four correct);
the clone retry→tarball chain exercised with stubs; the codeload endpoint for
`evs-broadcast/BroadTrack` probed live (`main` = HTTP 200); every edited region re-read
in its final on-disk state.

## 9. Plan of record — 2026-09-04, after the split-only restructuring

The conformant pipeline (batch 5) is installed and fully verified offline but has never
run live; every pre-conformance metric (runs 4–6) is void as a baseline (§7 item 15).
Current steps: **(1) push batch 5** — before committing, run
`git rm -r _deleted_pending_git_rm` and remove the empty
`configs/modules/split_merge/` directory (§8 item 18); **(2) one Kaggle session of the
conformant pipeline** on test/SNGS-116 via `docs/kaggle_one_sequence_test.ipynb`
(fail-fast cell already checks the new files). Expectations for that run: the
`tracklet_split` log should show ~52 tracklets → ~67 fragments with 0 merges;
`traj_refine` receives the fragments and should finally show real 2a/2b/stage-3
activity; the jersey stage recomputes its ENTIRE cache at fragment level (more, shorter
tracklets — expect a longer jersey phase and possibly more unnumbered tracklets, §7
item 13); the audit runs `_check_tracklet_split` and the row-accounting `traj_refine`
check for the first time live. Read the sidecars (`audit/tracklet_split/`,
`audit/traj_refine/` incl. the `stage3` block) and the in-run metrics as the FIRST
post-conformance baseline. **(3) Then the detector A/B** (`RUN_DET_AB=1`, snft vs HM on
a shared calibration cache) re-based on post-conformance numbers, and the detector
default decided (one line in `soccernet.yaml`; the pre-conformance evidence — snft
58.4–59.4 vs HM 55.07 — favored snft but is no longer directly applicable). After the
first accepted configuration: pin `team_sha256` (§7 item 3), replace the SATRN digest
prefix (§7 item 4), and tune `traj_refine` tau/edge_margin on `valid` if merging looks
off (§7 item 7) — tau is now the pipeline's only merge threshold.
