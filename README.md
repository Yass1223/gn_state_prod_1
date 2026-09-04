# SoccerNet Game State Reconstruction — clean pipeline

A single, rigorously-scoped GSR pipeline. One entry config, one path, no alternative
backends: every file in this repository belongs to the active pipeline.

```
bbox_detector -> track -> crop_filter -> tracklet_split -> calibration -> pitch_gate -> team_embed
              -> role_team -> jersey_number_detect -> traj_refine -> tracklet_agg -> audit
```

| Stage | Implementation | Notes |
|---|---|---|
| `bbox_detector` | YOLO11-L SoccerNet fine-tune | imgsz 1280, conf floor 0.1 (`>=`), RGB→BGR fix |
| `track` | BoT-SORT · SOF + OSNet-AIN (boxmot) | calls boxmot's `BotSort` directly with injected embeddings and external SOF camera motion; per-frame diagnostics sidecar for the audit |
| `crop_filter` | single / multi label per detection, tracked-only rule | rT ≤ 0.25, rB < 0.40; only boxes carrying a `track_id` may make another box multi; every detection gets `crop_single`/`crop_rT`/`crop_rB`/`crop_trigger`; no detection is removed |
| `tracklet_split` | DBSCAN tracklet splitting (SPLIT ONLY) | same OSNet-AIN as the tracker (shared module, same checkpoint pin and precision, audit-enforced); per tracklet, DBSCAN over ALL detections; noise attaches to the nearest clean-only centroid; all-multi fragments dissolve into the nearest remaining fragment; `eps 0.2, min_samples 5`, deliberately NO merge threshold (the pipeline's one merge is `traj_refine`); every tracked detection stays assigned; the incoming id is kept in `track_id_presplit`; per-sequence sidecar for the audit |
| `calibration` | **BroadTrack** (EVS, WACV'25) | temporal camera tracking + image-to-pitch; replaces pitch localization, camera calibration and projection in one stage |
| `pitch_gate` | off-pitch tracklet gate on the mean projected position | per final `track_id`, the mean of `bbox_pitch` (bottom-middle) over the rows with a finite projection; off-pitch iff `abs(mean_x) > 52.5 + margin_m` or `abs(mean_y) > 34 + margin_m`; the rows of an off-pitch tracklet lose their `track_id` (NaN, so no later stage and no export sees them), the original id stays in `track_id_pregate`; `enabled` switch (off = columns and sidecar still written, `track_id` untouched); `margin_m 3.5` untuned; a tracklet without a projection is kept; no detection is removed; per-sequence sidecar for the audit |
| `team_embed` | `osnet_team` (OSNet x1.0, 128×64, 256-d) on ≤ 16 crops per tracklet | the appearance model of the role/team notebook; fp32 + flip TTA; sampled on the stride-5 frame grid; single and multi crops embedded, the filter is applied afterwards |
| `role_team` | notebook rule chain (`sn_gamestate/team/rules.py`) | per tracklet: role (player / goalkeeper / referee), team cluster, left/right side; k-means on the single-crop descriptors, position rules on `bbox_pitch`, appearance-outlier channels, keeper-cue naming; frozen parameters from the notebook's tuning split; per-sequence sidecar for the audit |
| `jersey_number_detect` | **jn_pipeline_gsr** | tracklet-level, single crops only (`crop_single`): legibility → DBNet++ ROI → PARSeq + SATRN on the same crops → vote_pool (pooled per-frame majority vote); multi-GPU workers |
| `traj_refine` | label-aware trajectory refinement (the pipeline's ONE merge) | same OSNet-AIN pin as `tracklet_split` (audit-enforced); phase 2a merges same-team same-number fragments (clean frame sets disjoint, re-enter consistent, distance ≤ `tau 0.60`) and resolves same-time number conflicts; phase 2b merges agglomeratively under clean-frame disjointness, re-enter consistency and vacuous-when-unknown label agreement; stage 3 then enforces one detection per (frame, trajectory): clean wins, held multi crops are placed into the nearest in-scope trajectory with a free slot or unassigned (`track_id` NaN); snapshots in `track_id_prerefine` + jersey columns; per-sequence sidecar for the audit |
| `tracklet_agg` | majority vote | `[jersey_number]` (role is per-tracklet from `role_team`) |
| `audit` | per-component verdicts | read-only last stage: PASS/WARN/FAIL per component per sequence → `audit/<seq>.json`; `scripts/verify_run_integrity.py` refuses the run on any FAIL |

The jersey stage recognises tracklets whose `role` is player or goalkeeper (referees carry
no number) and, with `single_crops_only: true` (the default), hands the recognisers only the
single crops of a tracklet (`crop_single`); overlapping crops are excluded and a tracklet with
no single crop stays unnumbered. Team plays no part in it.
The role/team stage ignores multi crops, as the notebook did. There is no `reid` (prtreid)
stage: role and team come from `team_embed` + `role_team`, and the tracker, `tracklet_split`
and `traj_refine` use OSNet-AIN. `interpolation` is kept as a module for tracking-only experiments but is not
in the pipeline (synthesized rows would carry neither crop labels nor team embeddings).

### The `tracklet_split` stage

Stage 1 of the refinement method and split only
(`sn_gamestate/track/tracklet_split.py`, algorithm; `tracklet_split_api.py`, stage). Per
tracklet, DBSCAN (`eps 0.2`, `min_samples 5`, precomputed cosine distance) over ALL its
detections breaks a tracklet holding more than one identity into fragments; noise points
-- single or multi crop -- attach to the fragment with the nearest clean-only centroid;
fragments holding only multi-player detections are dissolved, detection by detection,
into the nearest remaining fragment. Fragments become the new trajectories; every tracked
detection stays assigned and the incoming id is kept per row in `track_id_presplit`.
There is deliberately no merging and no merge threshold in this stage: the pipeline's one
merge is `traj_refine`, which runs after `role_team` and `jersey_number_detect` so every
merge decision can use team and jersey-number evidence, and the run audit FAILS the run
if merging evidence appears in the splitter's config or sidecar.

`eps` and `min_samples` are the notebook splitter's operating point (the retired
split_merge stage ran the same split with these values), tuned on the embeddings held in
the notebook's Kaggle export. Whether that export was produced by the checkpoint pinned in
`modules/tracklet_split/tracklet_split.yaml` (`Ynniss/osnet_ain/best_ain_full.zip`) cannot
be verified from this repository; if it was not, `eps` (and `traj_refine`'s `tau`, which
lives on the same cosine-distance scale) do not transfer and must be re-tuned. The stage
writes `audit/tracklet_split/<seq>.json` (settings, embedder digest, per-tracklet split
report, output counts); the audit stage checks it against the config and against the
detections (every tracked row assigned, one detection per frame per fragment, every
fragment from exactly one source tracklet via `track_id_presplit`).

There is **no `pitch` stage**: BroadTrack runs its own keypoint (NBJW) and line (TVCalib)
detectors internally and emits camera parameters directly.

### The `pitch_gate` stage

The detector is a single-class person model and no stage before `calibration` can tell
a bench player, a steward or a photographer from an athlete, so without this stage every
tracklet is exported and receives a role. `pitch_gate`
(`sn_gamestate/pitch_gate/pitch_gate_api.py`) runs directly after `calibration` and
before `team_embed`. Per final `track_id` it takes the mean of the projected bottom-middle
points over the rows that carry a finite `bbox_pitch` and calls the tracklet off-pitch when
`|mean_x| > 52.5 + margin_m` or `|mean_y| > 34 + margin_m` (105 x 68 m pitch, metres,
centre at the origin). The tracklet mean is used rather than a per-frame test so that
projection error at the far touchline cannot flicker single frames of an on-pitch player.
With `enabled: true` the rows of an off-pitch tracklet get `track_id` NaN: the evaluation
export drops them and `team_embed`, `role_team`, the jersey stage, `tracklet_agg` and the
radar never see them, so the role rules run their k-means and outlier statistics on
on-pitch tracklets only. With `enabled: false` `track_id` is untouched. Either way the stage
writes `track_id_pregate` (the id `tracklet_split` left), `pitch_gate_offpitch` (the rule's
outcome), `pitch_mean_x` / `pitch_mean_y` and a sidecar `audit/pitch_gate/<seq>.json`; no
detection row is deleted and a tracklet without any projection is kept.

`margin_m` (`configs/modules/pitch_gate/pitch_gate.yaml`, default 3.5 m) is untuned: it is
an assumption meant to absorb projection error at the touchline and players a step outside
it (throw-ins, retrieving a ball) while still gating the benches and the area behind the
goals. Sweep it on the **valid** split; the audit records how many tracklets were gated
per sequence and warns when the share is implausibly high (a calibration problem rather
than off-pitch people). `tracklab -cn soccernet modules.pitch_gate.cfg.enabled=false` runs
the pipeline without the gate (the stage still writes its columns and sidecar), which is
how its contribution is attributed with `scripts/reference_metrics.py`.

## Why BroadTrack

On the sn-gamestate test split (WACV'25 paper, Table 1): JaC₅ **56.88** vs 37.14 for NBJW,
MRE **5.02 px** vs 10.28, completeness **100 %** vs 93.67. Radial-distortion modelling is
the dominant term — and the projection path here honours it (`unproject_point_on_planeZ0`
undistorts by default).

The binary writes a camera for every frame, including frames on which it lost track
(`score < 0.3` in its `main.cpp`; after more than 5 such frames it falls back to the position
prior). Those cameras are recorded with their low `score`, so the calibration stage rejects
frames below `min_score` (0.3, the binary's own threshold) and reuses the last accepted camera
for them; otherwise every player of such a frame is projected with a meaningless camera, which
propagates through `bbox_pitch` into the pitch gate, the role/team rules and, via `role`, the
jersey stage. The audit reports the share of rejected frames per sequence
(`calib_lost_frames_warn`).

## Install

```bash
uv venv --python 3.9 .venv
uv pip install --python .venv -e .
```

Two stages run as subprocesses in their own environments and need one-off provisioning
(both Docker-free — they work on Kaggle and Lightning.ai, which give a root shell with CUDA):

```bash
bash scripts/setup_broadtrack.sh   # C++ binary + LFS weights, built natively
JN_SRC=/path/to/jn_pipeline_gsr bash scripts/setup_jn_gsr.sh   # python 3.10 venv + weights
```

Everything else downloads at runtime: the detector from Hugging Face (`${hf:...}`
resolver), the tracker/tracklet_split/traj_refine OSNet-AIN checkpoint from Hugging Face with an enforced
sha256 pin (`sn_gamestate/reid/osnet_ain.py`; export `HF_TOKEN` if a repo is private),
and the team-appearance checkpoint `Ynniss/osnet_team/osnet_team_best.pt`
(`sn_gamestate/reid/osnet_team.py`; `team_sha256` in `modules/team_embed/osnet_team.yaml`
is unset until the first verified run records the digest — pin it then).

For the verified Kaggle procedure (disk layout, environment rules, timings, one-sequence
recipe) and every error observed on the way there, see
[`docs/KAGGLE_GUIDE.md`](docs/KAGGLE_GUIDE.md).

Download fallbacks (automatic, no flags needed):

* **Dataset.** `scripts/lightning_eval.sh` and `preflight_cpu.sh` download each split
  from the SoccerNet server (KAUST) first; if the server errors, does not respond, or
  leaves a truncated zip, they fall back to the official Hugging Face mirror
  [`SoccerNet/SN-GSR-2024`](https://huggingface.co/datasets/SoccerNet/SN-GSR-2024)
  (same `train/valid/test/challenge.zip` files) and unzip straight from the
  huggingface_hub cache.
* **BroadTrack sources.** The EVS clone now retries 3x (no credential prompts) and falls
  back to the GitHub tarball endpoint (`main`, then `master`); optionally set
  `BT_SRC_FALLBACK_REPO` to a **private** Hugging Face repo holding
  `broadtrack_src.tar.gz` as a third path. (Observed on Kaggle: GitHub transiently
  refusing anonymous git-over-HTTPS from shared IPs.)
* **BroadTrack weights.** `scripts/setup_broadtrack.sh` pulls the two TorchScript models
  from EVS's git-lfs first; if the pull fails or leaves pointer stubs, it fetches them
  from `BT_WEIGHTS_FALLBACK_REPO` (default `Ynniss/calibiration_weights`) on Hugging
  Face. `BT_WEIGHTS_REPO` still skips EVS entirely; `BT_WEIGHTS_DIR` reads them from
  disk (e.g. a Kaggle dataset).

### Python version on Kaggle / Lightning

`pyproject.toml` pins `requires-python = ">=3.9,<3.10"` because of `torch==1.13.1`, which
publishes no wheels for interpreters newer than 3.10. Hosted notebook images generally ship
a **newer** interpreter than that, so the venv must be built against a *provisioned* 3.9 —
never against the image's own `python`. Check the image first, then provision:

```bash
python -V                                  # the image's interpreter, for the record
uv python install 3.9                      # standalone CPython; no root, no apt
uv venv --python 3.9 .venv && uv pip install --python .venv -e .
.venv/bin/python -c "import sys, torch; print(sys.version, torch.__version__, torch.cuda.is_available())"
```

The last line is the gate: it must print `3.9.x`, `1.13.1`, and `True` on a GPU box. If a
future image makes a provisioned 3.9 impossible, the alternative is widening the pin set
(`requires-python` **and** the `torch` pin together) and re-validating — not silently
running on whatever the image provides, which would change the numbers.

> **Licence note.** BroadTrack is EVS noncommercial-research with no redistribution. Its
> sources and weights are fetched from the official repo at setup time and are **not**
> vendored here; do not publish the binary, weights, or generated calibration JSONs.

## Run

```bash
tracklab -cn soccernet                     # 1 clip by default (dataset.nvid)
tracklab -cn soccernet dataset.nvid=-1     # full split
bash scripts/lightning_eval.sh             # end-to-end runner (download + setup + eval)
```

Numerical precision: fp16, baked in (no switch). The detector runs ultralytics `half`,
and the OSNet-AIN embedder in `track` + `tracklet_split` + `traj_refine` runs under torch autocast on CUDA with
fp32 outputs; the `team_embed` stage runs fp32 with flip TTA, as the notebook it reproduces. Validated against fp32 on the Kaggle
5-sequence test run: tracking HOTA 71.61 (fp16) vs 71.08 (fp32), GS-HOTA 64.98 vs
64.70, calibration bit-identical - deltas inside the observed run-to-run noise (measured
with the former GTA-Link stage in the tracklet-refinement slot). The audit fails any run in
which `track`, `tracklet_split` and `traj_refine` pin different checkpoints.

GPU use: the jersey stage auto-detects GPUs via `nvidia-smi` and shards tracklets across
all of them (2 workers on Kaggle 2×T4, 4 on Lightning 4×T4, 1 otherwise). BroadTrack and
the jersey stage both cache per sequence, so re-runs skip recomputation — the jersey cache
is keyed on a content hash covering the tracklet manifest, `legibility_thr` and the sha256
of both recogniser checkpoints (PARSeq and SATRN), so retuning tracking, moving the gate, or
staging a different checkpoint under the same filename each invalidate it automatically.
The jersey consolidation rule is fixed to `vote_pool` (see
`plugins/jn_gsr/README.md` for the measured GT-box numbers and their caveats).

## Experimental switches (not in the production path)

| Flag | Default | On means | Where |
|---|---|---|---|
| `modules.interpolation.cfg.enabled` | `false` | fill tracklet gaps of `1 < dt < n_dti` frames by linear interpolation (`n_dti`, `n_min` — untuned) | `configs/modules/interpolation/dti.yaml` |
| detector + tracker conf floor | `0.1` / `0.1` | lowering **both** to `0.05` is the only way to make BoT-SORT's BYTE low band non-empty; an A/B, not a code change | `bbox_detector` `min_confidence` + `track_low_thresh` |

The GTA-Link tuning tool (`scripts/tune_gta_kaggle.py`) was removed with the GTA-Link
stage. `tracklet_split` has two parameters (`eps`, `min_samples` in
`configs/modules/tracklet_split/tracklet_split.yaml`) and `traj_refine` carries the
pipeline's only merge threshold (`tau` in `configs/modules/traj_refine/traj_refine.yaml`);
re-tuning them means re-running the
notebook's sweep on embeddings produced by the pinned checkpoint, on the **valid** split.
`test` is reserved for the frozen production run.

> `interpolation` is not in the production pipeline. `interpolation.enabled=true`
> synthesizes detection rows that carry neither crop labels nor team embeddings; that is
> fine for tracking metrics but not for the role/team stage.

## Verify before you measure

Every run ends with the `audit` stage (`sn_gamestate/audit/run_audit_api.py`): for each
sequence and each component it records what the component was supposed to produce, what
was observed, and a verdict. Neither BroadTrack nor the jersey stage aborts on failure,
so without this a run can finish with empty pitch coordinates or empty jersey numbers and
still print plausible metrics. The crop filter, team-embedding and role/team stages are
checked against what their configs declare: labels must agree with the stored overlap
ratios at the configured thresholds, the `tracklet_split` sidecar must record the configured
`eps`/`min_samples` and checkpoint digest with NO merge threshold anywhere, its per-tracklet
split counts must be
consistent with each other and with the final `track_id`s (every tracked row assigned, one
detection per frame per fragment, every fragment from one source tracklet), the `pitch_gate` sidecar must record
the configured switch and margin, its mean positions and off-pitch flags must equal the
ones recomputed from `bbox_pitch`, and `track_id` must have been cleared on off-pitch
tracklets if and only if the gate is enabled (the tracklet_split, crop_filter and calibration
checks compare against the pre-gate ids in `track_id_pregate`), every tracklet must carry a team embedding, the
checkpoint digest and the rule parameters that ran (per-sequence sidecars under
`audit/team_embed/`, `audit/role_team/`) must equal the configured ones, every tracked
row must have a valid role, players/keepers a side and referees none. The jersey check is the strictest: the per-sequence cache
blob must carry `rule = vote_pool`, the sha256 of both staged checkpoints and the
`single_crops_only` value that ran (equal to the config), its manifest counts must equal
the eligible tracklets and their single crops in the state, it must answer every
player/goalkeeper tracklet that holds a single crop, no number may sit on a tracklet without
one, the columns must agree with the blob, and the provisioning
provenance of SATRN and PARSeq must be on record. Thresholds are in
`configs/modules/audit/run_audit.yaml`.

```bash
python scripts/verify_run_integrity.py --expect-sequences 49   # exits non-zero on any FAIL
```

The radar (minimap) draws only tracked rows with a team or referee colour. Untracked
detections and tracklets without a team are not painted in a neutral colour; the audit
reports how many rows were skipped for that reason (`visualization (radar)` check).

The BroadTrack → sn-calibration parameter conversion includes a distortion rescale derived
from the C++ source. Confirm it on real output before trusting any metric:

```bash
python scripts/verify_broadtrack_conversion.py -c broadtrack_calib/SNGS-116.json \
    -f data/SoccerNetGS/test/SNGS-116/img1 -o broadtrack_check
```

Checks: model equivalence against an independent reimplementation (< 0.01 px), Z=0
roundtrip (< 1 cm), pitch-wireframe overlay (must sit on the painted lines), frame coverage.

## Reference metrics

Computed and stored **separately** from the pipeline's own eval output:

```bash
# TrackLab saves the state INSIDE the Hydra run directory (verified on the Kaggle run
# of 2026-09-02), so point --state at the newest one:
python scripts/reference_metrics.py \
    --state "$(ls -t outputs/sn-gamestate/*/*/states/sn-gamestate.pklz | head -1)" \
    --dataset-path data/SoccerNetGS --eval-set test --out reference_metrics
```

| Block | Metrics | Source |
|---|---|---|
| `tracking` | HOTA, DetA, AssA, MOTA, IDF1, IDSW | pipeline evaluator, image space, attributes off |
| `gsr` | GS-HOTA, GS-DetA, GS-AssA, GS-IDF1 | same evaluator, pitch space, attributes on, 5 m tolerance |
| `jersey_number` | det_acc_all, precision, recall, F1, trk_acc_all, trk_acc_numbered | Hungarian IoU ≥ 0.5, unnumbered = −1 |
| `calibration` | JaC5, JaC10, MRE, MedRE, CR | official sn-calibration protocol at 960×540, mirror-aware |

Pass several `--state/--label` pairs to get a comparison table (`summary.md`) — that is how
per-stage contributions (e.g. `traj_refine` enabled vs disabled) are attributed.

## Layout

```
sn_gamestate/
  bbox_detector/yolo_snft_api.py
  crop_filter/crop_filter_api.py         # single/multi label, tracked-only rule
  reid/osnet_ain.py, osnet_team.py       # tracker/tracklet_split/traj_refine embedder; team-appearance model
  track/bot_sort.py, tracklet_split.py, tracklet_split_api.py, hf_resolver.py
  refine/traj_refine.py, traj_refine_api.py  # the pipeline's one merge + stage-3 resolution
  calibration/broadtrack_api.py          # calibration + image-to-pitch
  pitch_gate/pitch_gate_api.py           # off-pitch tracklet gate (mean bbox_pitch, on/off switch)
  team/rules.py, team_embed_api.py, role_team_api.py   # notebook rules; the two stages
  jersey/jn_gsr_api.py                   # multi-GPU tracklet recognition
  audit/run_audit_api.py                 # per-component verdicts (last stage)
  visualization/, configs/
tests/                                         # rules == notebook, stage contracts, audit
plugins/calibration/sn_calibration_baseline/   # Camera + SoccerPitch geometry
plugins/jn_gsr/                                # vendored jersey package (+ its own venv)
scripts/                                       # setup, verification, metrics, TRT
```
