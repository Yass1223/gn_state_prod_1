# SoccerNet Game State Reconstruction — clean pipeline

A single, rigorously-scoped GSR pipeline. One entry config, one path, no alternative
backends: every file in this repository belongs to the active pipeline.

```
bbox_detector -> track -> crop_filter -> gta_link -> calibration -> team_embed
              -> role_team -> jersey_number_detect -> tracklet_agg -> audit
```

| Stage | Implementation | Notes |
|---|---|---|
| `bbox_detector` | YOLO11-L SoccerNet fine-tune | imgsz 1280, conf floor 0.1 (`>=`), RGB→BGR fix |
| `track` | BoT-SORT · SOF + OSNet-AIN (boxmot) | calls boxmot's `BotSort` directly with injected embeddings and external SOF camera motion; per-frame diagnostics sidecar for the audit |
| `crop_filter` | single / multi label per detection, tracked-only rule | rT ≤ 0.25, rB < 0.40; only boxes carrying a `track_id` may make another box multi; every detection gets `crop_single`/`crop_rT`/`crop_rB`/`crop_trigger`; no detection is removed |
| `gta_link` | offline tracklet stitching (Split + Connect) | same OSNet-AIN as the tracker (shared module, same checkpoint pin and precision, audit-enforced); symmetric merge gate; per-frame collision guard |
| `calibration` | **BroadTrack** (EVS, WACV'25) | temporal camera tracking + image-to-pitch; replaces pitch localization, camera calibration and projection in one stage |
| `team_embed` | `osnet_team` (OSNet x1.0, 128×64, 256-d) on ≤ 16 crops per tracklet | the appearance model of the role/team notebook; fp32 + flip TTA; sampled on the stride-5 frame grid; single and multi crops embedded, the filter is applied afterwards |
| `role_team` | notebook rule chain (`sn_gamestate/team/rules.py`) | per tracklet: role (player / goalkeeper / referee), team cluster, left/right side; k-means on the single-crop descriptors, position rules on `bbox_pitch`, appearance-outlier channels, keeper-cue naming; frozen parameters from the notebook's tuning split; per-sequence sidecar for the audit |
| `jersey_number_detect` | **jn_pipeline_gsr** | tracklet-level: legibility → DBNet++ ROI → PARSeq + SATRN on the same crops → vote_pool (pooled per-frame majority vote); multi-GPU workers |
| `tracklet_agg` | majority vote | `[jersey_number]` (role is per-tracklet from `role_team`) |
| `audit` | per-component verdicts | read-only last stage: PASS/WARN/FAIL per component per sequence → `audit/<seq>.json`; `scripts/verify_run_integrity.py` refuses the run on any FAIL |

The jersey stage recognises tracklets whose `role` is player or goalkeeper (referees carry
no number) and uses every crop of a tracklet, single or multi; team plays no part in it.
The role/team stage ignores multi crops, as the notebook did. There is no `reid` (prtreid)
stage: role and team come from `team_embed` + `role_team`, and the tracker and GTA-Link use
OSNet-AIN. `interpolation` is kept as a module for `scripts/tune_gta_kaggle.py` but is not in
the pipeline (synthesized rows would carry neither crop labels nor team embeddings).

There is **no `pitch` stage**: BroadTrack runs its own keypoint (NBJW) and line (TVCalib)
detectors internally and emits camera parameters directly.

## Why BroadTrack

On the sn-gamestate test split (WACV'25 paper, Table 1): JaC₅ **56.88** vs 37.14 for NBJW,
MRE **5.02 px** vs 10.28, completeness **100 %** vs 93.67. Radial-distortion modelling is
the dominant term — and the projection path here honours it (`unproject_point_on_planeZ0`
undistorts by default).

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
resolver), the tracker/GTA-Link OSNet-AIN checkpoint from Hugging Face with an enforced
sha256 pin (`sn_gamestate/reid/osnet_ain.py`; export `HF_TOKEN` if a repo is private),
and the team-appearance checkpoint `Ynniss/osnet_team/osnet_team_best.pt`
(`sn_gamestate/reid/osnet_team.py`; `team_sha256` in `modules/team_embed/osnet_team.yaml`
is unset until the first verified run records the digest — pin it then).

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
and the OSNet-AIN embedder in `track` + `gta_link` runs under torch autocast on CUDA with
fp32 outputs; the `team_embed` stage runs fp32 with flip TTA, as the notebook it reproduces. Validated against fp32 on the Kaggle
5-sequence test run: tracking HOTA 71.61 (fp16) vs 71.08 (fp32), GS-HOTA 64.98 vs
64.70, calibration bit-identical - deltas inside the observed run-to-run noise. The
audit fails any run in which `track` and `gta_link` pin different checkpoints.

GPU use: the jersey stage auto-detects GPUs via `nvidia-smi` and shards tracklets across
all of them (2 workers on Kaggle 2×T4, 4 on Lightning 4×T4, 1 otherwise). BroadTrack and
the jersey stage both cache per sequence, so re-runs skip recomputation — the jersey cache
is keyed on a content hash covering the tracklet manifest, `legibility_thr` and the sha256
of both recogniser checkpoints (PARSeq and SATRN), so retuning tracking, moving the gate, or
staging a different checkpoint under the same filename each invalidate it automatically.
The jersey consolidation rule is fixed to `vote_pool` (see
`plugins/jn_gsr/README.md` for the measured GT-box numbers and their caveats).

## Tuning flags (TEMPORARY — removed at productionization)

Four switches exist only so the Kaggle sweeps can A/B them without code edits. **Every
default below reproduces today's baseline**, so a plain `tracklab -cn soccernet` is
unaffected by their presence. They are deleted once the winners are baked in, together
with `scripts/tune_gta_kaggle.py`; where a winner is "off", the feature's code goes too.

| Flag | Default | On means | Where |
|---|---|---|---|
| `modules.gta_link.cfg.connect_mode` | `agglomerative` | `iterative` = the sjc042/gta-link greedy merge with per-pair gate and row/col recompute | `configs/modules/gta_link/gta_link.yaml` |
| `modules.gta_link.cfg.use_split` | `false` | DBSCAN Split before Connect, breaking tracklets that hold an id switch (`split_eps`, `split_min_samples`, `split_max_k`, `split_len_thres` — all untuned) | `configs/modules/gta_link/gta_link.yaml` |
| `modules.interpolation.cfg.enabled` | `false` | fill tracklet gaps of `1 < dt < n_dti` frames by linear interpolation (`n_dti`, `n_min` — untuned) | `configs/modules/interpolation/dti.yaml` |
| detector + tracker conf floor | `0.1` / `0.1` | lowering **both** to `0.05` is the only way to make BoT-SORT's BYTE low band non-empty; an A/B, not a code change | `bbox_detector` `min_confidence` + `track_low_thresh` |

Sweeping them:

```bash
python scripts/tune_gta_kaggle.py check                    # guards only, no GPU
python scripts/tune_gta_kaggle.py stage1 --nvid 20 --holdout 8
python scripts/tune_gta_kaggle.py stage2 --max-configs 120
python scripts/tune_gta_kaggle.py stage2 --fine --max-configs 60
python scripts/tune_gta_kaggle.py holdout
```

All tuning runs on the **valid** split and on YOLO11L detections — the tool exits
non-zero against any other detector. `test` is reserved for the frozen production run.

> `interpolation` is not in the production pipeline. `interpolation.enabled=true`
> synthesizes detection rows that carry neither crop labels nor team embeddings; that is
> fine for tracking metrics (the tuning path) but not for the role/team stage.

## Verify before you measure

Every run ends with the `audit` stage (`sn_gamestate/audit/run_audit_api.py`): for each
sequence and each component it records what the component was supposed to produce, what
was observed, and a verdict. Neither BroadTrack nor the jersey stage aborts on failure,
so without this a run can finish with empty pitch coordinates or empty jersey numbers and
still print plausible metrics. The crop filter, team-embedding and role/team stages are
checked against what their configs declare: labels must agree with the stored overlap
ratios at the configured thresholds, every tracklet must carry a team embedding, the
checkpoint digest and the rule parameters that ran (per-sequence sidecars under
`audit/team_embed/`, `audit/role_team/`) must equal the configured ones, every tracked
row must have a valid role, players/keepers a side and referees none. The jersey check is the strictest: the per-sequence cache
blob must carry `rule = vote_pool` and the sha256 of both staged checkpoints, answer every
player/goalkeeper tracklet, agree with the detection columns, and the provisioning
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
python scripts/reference_metrics.py --state states/sn-gamestate.pklz \
    --dataset-path data/SoccerNetGS --eval-set test --out reference_metrics
```

| Block | Metrics | Source |
|---|---|---|
| `tracking` | HOTA, DetA, AssA, MOTA, IDF1, IDSW | pipeline evaluator, image space, attributes off |
| `gsr` | GS-HOTA, GS-DetA, GS-AssA | same evaluator, pitch space, attributes on, 5 m tolerance |
| `jersey_number` | det_acc_all, precision, recall, F1, trk_acc_all, trk_acc_numbered | Hungarian IoU ≥ 0.5, unnumbered = −1 |
| `calibration` | JaC5, JaC10, MRE, MedRE, CR | official sn-calibration protocol at 960×540, mirror-aware |

Pass several `--state/--label` pairs to get a comparison table (`summary.md`) — that is how
per-stage contributions (e.g. with/without `gta_link`) are attributed.

## Layout

```
sn_gamestate/
  bbox_detector/yolo_snft_api.py
  crop_filter/crop_filter_api.py         # single/multi label, tracked-only rule
  reid/osnet_ain.py, osnet_team.py       # tracker/GTA-Link embedder; team-appearance model
  track/bot_sort.py, gta_link_api.py, hf_resolver.py
  calibration/broadtrack_api.py          # calibration + image-to-pitch
  team/rules.py, team_embed_api.py, role_team_api.py   # notebook rules; the two stages
  jersey/jn_gsr_api.py                   # multi-GPU tracklet recognition
  audit/run_audit_api.py                 # per-component verdicts (last stage)
  visualization/, configs/
tests/                                         # rules == notebook, stage contracts, audit
plugins/calibration/sn_calibration_baseline/   # Camera + SoccerPitch geometry
plugins/jn_gsr/                                # vendored jersey package (+ its own venv)
scripts/                                       # setup, verification, metrics, TRT
```
