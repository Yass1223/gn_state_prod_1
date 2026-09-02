# SoccerNet Game State Reconstruction — pipeline reference (repo_prod_v1)

Reference date: 2026-09-02, final. Derived from direct inspection of the repository
(configs, source, scripts), not from the README alone, and validated by an end-to-end
Kaggle run on one test sequence (see §7 findings and `docs/KAGGLE_GUIDE.md`). Intended to
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
bbox_detector -> track -> crop_filter -> split_merge -> calibration -> pitch_gate
             -> team_embed -> role_team -> jersey_number_detect -> tracklet_agg -> audit
```

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
  (tracker/split_merge embedder) and `osnet_x1_0` (team model backbone).
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
| `bbox_detector` | `sn_gamestate.bbox_detector.yolo_snft_api.YOLOUltralyticsSNFT` | YOLO11-L fine-tune from HF (`${hf:Ynniss/sn-gamestate-weights,yolov11_sn_best.pt}`); imgsz 1280, conf floor 0.1 (kept with `>=`), iou 0.7, max_det 300, RGB→BGR fix; optional TensorRT (off by default) |
| `track` | `sn_gamestate.track.bot_sort.BotSortSOF` | boxmot BotSort called directly; embeddings injected from the shared OSNet-AIN module; SOF camera motion (scale 0.15) computed outside; thresholds: high 0.3, low 0.05, new 0.4, match 0.85, proximity 0.5, appearance 0.35, buffer 60, frame_rate 25; fp16 autocast, fp32 outputs; per-frame audit sidecar `audit/track/` |
| `crop_filter` | `sn_gamestate.crop_filter.CropFilter` | single iff rT ≤ 0.25 and rB < 0.40, contaminators must carry a track_id (`contam_mode: tracked`); writes `crop_single/crop_rT/crop_rB/crop_trigger`; removes nothing |
| `split_merge` | `sn_gamestate.track.split_merge_api.SplitMerge` | DBSCAN per tracklet (eps 0.2, min_samples 5, precomputed cosine) on `crop_single` rows; agglomerative fragment merge (tau 0.60, disjoint clean frame sets); one detection per trajectory+frame; unplaceable rows get `track_id` NaN; same OSNet-AIN pin as `track` (audit-enforced); sidecar `audit/split_merge/` |
| `calibration` | `sn_gamestate.calibration.broadtrack_api.BroadTrackCalibration` | BroadTrack binary at `pretrained_models/broadtrack/`; camera prior (0, 55, −12); `min_score 0.3` rejects lost frames and reuses the last accepted camera (`use_prev_parameters: true`, `max_carry_frames 0`); per-sequence JSON cache `broadtrack_calib/` (`use_cached_json: true`); writes human-bbox masks from own detections; `staging_dir` for read-only datasets; emits camera `parameters` and `bbox_pitch` |
| `pitch_gate` | `sn_gamestate.pitch_gate.PitchGate` | enabled, margin_m 3.5 (untuned); off-pitch iff |mean_x| > 52.5+m or |mean_y| > 34+m on the tracklet mean of finite `bbox_pitch`; gated tracklets: `track_id` → NaN, original kept in `track_id_pregate`; no row deleted; sidecar `audit/pitch_gate/` |
| `team_embed` | `sn_gamestate.team.TeamEmbedding` | osnet_team (OSNet x1.0, 128×64, 256-d) from HF `Ynniss/osnet_team/osnet_team_best.pt`; ≤ 16 crops/tracklet on the stride-5 grid; fp32 + flip TTA; `team_sha256` currently **null** (recorded, not enforced — pin after first verified run); sidecar `audit/team_embed/` |
| `role_team` | `sn_gamestate.team.RoleTeamAssignment` | notebook rule chain (`team/rules.py`): role player/goalkeeper/referee, team k-means on single-crop descriptors, side; frozen params (k 3.25, tau_n 5, max_ref 3, side_rule keeper, …), untuned on this pipeline's tracklets/BroadTrack projections; sidecar `audit/role_team/` |
| `jersey_number_detect` | `sn_gamestate.jersey.jn_gsr_api.JNGsrTrackletRecognizer` | subprocess workers in the 3.10 venv; roles [player, goalkeeper]; `single_crops_only: true`; legibility > 0.72 → DBNet++ ROI → PARSeq + SATRN → `vote_pool` (the only rule); stride 5; fp16; GPU sharding auto via nvidia-smi (2 workers on Kaggle 2×T4); content-hash cache `jn_cache/` |
| `tracklet_agg` | `tracklab.wrappers.MajorityVoteTracklet` | majority vote over `[jersey_number]` only (role is per-tracklet from role_team) |
| `audit` | `sn_gamestate.audit.RunAudit` | read-only last stage; per-sequence, per-component PASS/WARN/FAIL to `audit/<seq>.json`; cross-checks every sidecar against the composed config (including track vs split_merge checkpoint-pin equality); `scripts/verify_run_integrity.py` exits non-zero on any FAIL |

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
| OSNet-AIN `best_ain_full.zip` (tracker + split_merge) | HF `Ynniss/osnet_ain`, revision `d78f65de…`, the packed file is itself a torch.save archive | exploded snapshot `Ynniss/osnet_ain_ckp` path exists in code (used when `ain_file` is cleared); `ain_local_path` reads from disk | **sha256 enforced** (`a0a7e426…`); audit fails on any pin mismatch between the two stages |
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
tracklab -cn soccernet                       # 1 clip (dataset.nvid=1), test split
tracklab -cn soccernet dataset.nvid=-1       # full split
bash scripts/lightning_eval.sh               # end-to-end: venv + patch + data + setup + eval
python scripts/preflight_imports.py          # seconds; every stage importable
CHECK_ONLY=1 bash preflight_cpu.sh           # artifact audit only
python scripts/verify_run_integrity.py --expect-sequences <N>   # non-zero on any audit FAIL
python scripts/verify_broadtrack_conversion.py -c broadtrack_calib/<seq>.json -f data/.../img1 -o broadtrack_check
python scripts/reference_metrics.py --state outputs/sn-gamestate/<date>/<time>/states/sn-gamestate.pklz --dataset-path data/SoccerNetGS --eval-set test --out reference_metrics
```

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

1. `split_merge` eps/tau were tuned on embeddings from a notebook export whose producing
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

Discrepancies found in this analysis:

7. RESOLVED 2026-09-02: `preflight_cpu.sh` no longer checks/fetches
   `sports_model.pth.tar-60` (was referenced nowhere; confirmed unused by the successful
   end-to-end Kaggle run).
8. RESOLVED 2026-09-02: the README reference-metrics command now locates the newest
   state under `outputs/sn-gamestate/*/*/states/` (TrackLab saves it inside the Hydra run
   directory, confirmed on Kaggle; the old `states/...` root-relative path never exists).

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

Still **[unverified]** after the completed one-sequence test: full-split behavior and
runtime; the HF dataset fallback's actual network path (code path tested synthetically,
never against a live server failure); the EVS git-lfs weights path and its automatic
fallback (runs pinned `BT_WEIGHTS_REPO`); the HF mirror zip's internal layout (§5); and
the line-by-line behavior of files verified only at the config/contract level
(`broadtrack_api.py`, `run_audit_api.py`, the jersey worker).

## 8. Changes made on 2026-09-02

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

Verification performed per edit: `bash -n` on every touched script; the zip-validity
fallback trigger exercised against missing/empty/truncated/valid zips (all four correct);
the clone retry→tarball chain exercised with stubs; the codeload endpoint for
`evs-broadcast/BroadTrack` probed live (`main` = HTTP 200); every edited region re-read
in its final on-disk state.

## 9. Plan of record — all three steps complete (2026-09-02)

Step 1: this document + dataset/BroadTrack download fallbacks (later extended with the
BroadTrack source-acquisition hardening). Step 2: Kaggle test passed on test/SNGS-116 —
full pipeline exit 0, RUN INTEGRITY OK (0 FAIL / 0 WARN), reference metrics + calibration
conversion checks all green (results in `docs/KAGGLE_GUIDE.md` and §7 above). Step 3
(final edits, in the local tree, pending the push): `docs/KAGGLE_GUIDE.md`, the README
`--state` correction and guide link, the `sports_model.pth.tar-60` removal from
`preflight_cpu.sh`, and this document at `docs/PIPELINE_REFERENCE.md` (full list in §8).
