# Running on Kaggle — verified procedure and error guide

A ready-to-run notebook implementing this guide's one-sequence recipe (including the
`traj_refine` stage, the verification chain, and optional detector / refine A/B runs)
is at `docs/kaggle_one_sequence_test.ipynb` — upload it to Kaggle as-is.

Everything below was established empirically on Kaggle GPU T4 x2 sessions on 2026-09-02
(image host Python 3.12.13, Ubuntu 22.04, CUDA driver for 2x Tesla T4). Five sessions were
run; the final one completed the full pipeline on one `test` sequence (SNGS-116) end to
end with `verify_run_integrity.py` reporting 0 FAIL / 0 WARN. Each rule here traces to a
failure actually observed on the way there.

## Session requirements

* Accelerator **GPU T4 x2**, Internet **ON** (Settings ▸ Internet).
* If any `Ynniss/*` Hugging Face repo is private: a Kaggle secret `HF_TOKEN`, exported
  into the environment before the setup scripts run.
* Licence: BroadTrack (EVS) is noncommercial-research with no redistribution. Keep the
  notebook and any Kaggle dataset made from outputs **private**; never publish the built
  binary, the two TorchScript weights, or generated `broadtrack_calib/*.json`.

## Disk layout (the #1 failure cause)

`/kaggle/working` is a **20 GB** device; `/kaggle/tmp` sits on the ~TB root overlay.
Observed failure: building BroadTrack under `/kaggle/working` after the dataset download
filled the disk mid-libtorch-extraction (`unzip` exit 50, "write error (disk full?)").

Put on `/kaggle/working` (small, persisted): the repo checkout, `.venv` (~5 GB), the
extracted sequence(s), run outputs, logs.
Put on `/kaggle/tmp` (big, ephemeral): the dataset zip (8.85 GB for `test`), the
BroadTrack tree (`BT_ROOT=/kaggle/tmp/broadtrack`, ~7 GB with libtorch), the jersey venv
(`JN_VENV=/kaggle/tmp/.venv_jn`, ~7 GB). The huggingface_hub cache (`~/.cache`) is on the
big overlay already.

With `BT_ROOT` moved, the run needs the matching path overrides (the setup script prints
them): `modules.calibration.cfg.binary`, `.keypoint_model`, `.line_model`,
`.libtorch_lib`, `.compute_tripod_script` — all under `/kaggle/tmp/broadtrack`.

## Environment rules

* Build the venv with uv-provisioned **Python 3.9** (`uv python install 3.9`), never the
  image interpreter; gate on `3.9.x / torch 1.13.1 / cuda: True`. Then
  `uv pip install --python .venv --no-deps boxmot==19.0.0` and apply the installed-
  TrackLab patch (`sed -i 's/gamestate-2025/gamestate-2024/g'` on
  `tracklab/wrappers/dataset/soccernet/soccernet_game_state.py`).
* **`export MPLBACKEND=Agg` for every venv invocation from a notebook.** Jupyter exports
  `MPLBACKEND=module://matplotlib_inline.backend_inline`; `%%bash` children inherit it,
  and the 3.9 venv has no `matplotlib_inline`, so `tracklab` (and any script importing
  matplotlib) dies at import with `ValueError: Key backend: ... is not a valid value`.
  The import preflight is not a canary for this — it guards itself.
* `num_cores=0` on the run (cuDNN / torch_shm_manager multiprocessing issue).
* Pipe `echo "n" |` into `tracklab` (answers its interactive prompt).
* Jersey workers auto-shard to 2 on the two T4s; nothing to configure.

## Network flakiness (all three now have automatic fallbacks in the scripts)

* **SoccerNet (KAUST) server**: worked in every session but at 2.8–15.1 MiB/s (10–53 min
  for `test.zip`). On error / no response / truncated zip, `scripts/lightning_eval.sh`
  and `preflight_cpu.sh` fall back to the Hugging Face mirror
  `SoccerNet/SN-GSR-2024` (same `<split>.zip` files). The server zip's members are
  `SNGS-XXX/...` at the root (verified); the mirror is published by the same team and
  expected identical, but the fallback never triggered in these sessions, so the mirror
  zip's internal layout is unverified — the preflight's `img1`-depth check would catch a
  mismatch.
* **GitHub anonymous git-over-HTTPS**: transiently refused from shared Kaggle IPs
  (`fatal: could not read Username for 'https://github.com'`) — the same clone succeeded
  hours earlier. `scripts/setup_broadtrack.sh` now retries 3x with
  `GIT_TERMINAL_PROMPT=0`, then falls back to the codeload tarball (`main`, then
  `master`), then optionally `BT_SRC_FALLBACK_REPO` (a private HF snapshot holding
  `broadtrack_src.tar.gz`). Harden your own repo clone the same way.
* **EVS git-lfs**: untested here because runs pinned `BT_WEIGHTS_REPO=Ynniss/calibiration_weights`
  (recommended on Kaggle for determinism); if EVS LFS is used and fails, the automatic
  per-file HF fallback (`BT_WEIGHTS_FALLBACK_REPO`) takes over.

## Paths that differ from a casual reading

* TrackLab saves the state **inside the Hydra run directory**:
  `outputs/sn-gamestate/<date>/<time>/states/sn-gamestate.pklz`. Locate it with
  `ls -t outputs/sn-gamestate/*/*/states/sn-gamestate.pklz | head -1` before calling
  `scripts/reference_metrics.py`.
* `broadtrack_calib/<seq>.json`, `audit/<seq>.json`, `jn_cache/` are repo-root relative
  (project_dir = launch directory — always launch from the repo root).

## One-sequence test recipe (verified end to end)

1. Clone the repo (with retry/tarball hardening), build `.venv`, patch TrackLab, run
   `scripts/preflight_imports.py`.
2. Download `test.zip` to `/kaggle/tmp` (server → HF fallback), extract **only** the
   target sequence to `data/SoccerNetGS/test/SNGS-116/` (saves ~9 GB and ~30 min over a
   full extraction).
3. `BT_ROOT=/kaggle/tmp/broadtrack BT_WEIGHTS_REPO=Ynniss/calibiration_weights
   bash scripts/setup_broadtrack.sh`
4. `JN_VENV=/kaggle/tmp/.venv_jn bash scripts/setup_jn_gsr.sh`
5. Run with `dataset.eval_set=test dataset.nvid=1 "dataset.vids_dict.test=['SNGS-116']"`,
   the five calibration overrides, the jersey `venv_python` override, `num_cores=0`,
   `MPLBACKEND=Agg`.
6. `scripts/verify_run_integrity.py --expect-sequences 1`, then `reference_metrics.py`
   on the located state, then `verify_broadtrack_conversion.py` on the sequence.

Approximate wall time: dataset 10–55 min (server speed), BroadTrack build ~20 min,
jersey venv ~15 min, pipeline on one 750-frame sequence ~2.5 h, metrics ~25 min.

## Reference results, one sequence (test/SNGS-116, 2026-09-02)

Single-sequence numbers; not comparable to full-split figures.

| block | metrics |
|---|---|
| in-run eval (pitch, attrs on) | GS-HOTA 58.4–59.4 across two independent sessions |
| reference `tracking` (image, attrs off) | HOTA 64.85, DetA 70.86, AssA 59.42, MOTA 86.90, IDF1 80.57, IDSW 208 |
| reference `gsr` | GS-HOTA 59.45, GS-DetA 51.08, GS-AssA 69.19, GS-IDF1 69.68 |
| reference `jersey_number` | det_acc_all 0.746, F1 0.806, trk_acc_all 0.857, trk_acc_numbered 0.875 |
| reference `calibration` | JaC5 0.481, JaC10 0.677, MRE 4.74 px, MedRE 2.86 px, CR 1.0 (749/750 frames, mean score 0.647) |

Conversion spot check: model equivalence max 3.17e-05 px, plane roundtrip max 6.37e-06 m
— both far inside thresholds.
